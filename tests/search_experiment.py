"""Run and record one search-improvement experiment (build, oracle, or timed evaluation).

Usage (from the repository root):
  .venv/Scripts/python.exe tests/search_experiment.py --record gap <run_id> -m tests.test_gemma_output --config decode --optimal --timeout 600
  .venv/Scripts/python.exe tests/search_experiment.py --record inference <run_id> -m tests.inference_benchmark --run-dir logs/inference_runs/<run_id>
  .venv/Scripts/python.exe tests/search_experiment.py --record prefill_decode_seq_len <run_id> -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 4 --run-dir logs/gemma_output_runs/<run_id>
  .venv/Scripts/python.exe tests/search_experiment.py --record gap <run_id> build.py --targets bindings test --log-level INFO
  .venv/Scripts/python.exe tests/search_experiment.py --record gap <run_id> --native-command tensor_graphs_cpp/test.exe --start-rank-window

Every run writes logs/<record>_improvements/<run_id>.{log,json,patch} (refusing to
overwrite) and appends an in-progress entry, then a completion entry, to the
record document (docs/agents/targets/gap_integral/progress.md for --record gap,
docs/agents/targets/<name>/progress.md for --record <name>, progress_<machine>.md for
--record <name>:<machine>). The patch
captures the uncommitted state of tensor_graphs_cpp, tests and docs/core,
including untracked files, relative to HEAD.
"""
import datetime
import difflib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
RECORDS = {
    "gap": ("logs/gap_improvements", "docs/agents/targets/gap_integral/progress.md"),
    "search": ("logs/search_improvements", "docs/agents/targets/search_volume/progress.md"),
    "inference": ("logs/inference_improvements", "docs/agents/targets/inference_speed/progress.md"),
}
# Target folders whose runs predate logs/targets/ keep their log folder under either name.
RECORDS.update({"gap_integral": RECORDS["gap"], "search_volume": RECORDS["search"],
                "inference_speed": RECORDS["inference"]})
TARGETS_DIR = ROOT / "docs" / "agents" / "targets"
PATCH_PATHS = ["tensor_graphs_cpp", "tests", "docs/core"]


def recordPaths(record_name):
    """Legacy records above, or any docs/agents/targets/<name>/ (logs in logs/targets/<name>).

    `<name>:<machine>` records into docs/agents/targets/<name>/progress_<machine>.md, for targets
    that keep one progress document per machine."""
    if record_name in RECORDS:
        return RECORDS[record_name]
    name, _, machine = record_name.partition(":")
    document = f"progress_{machine}.md" if machine else "progress.md"
    if name.startswith("_") or not (TARGETS_DIR / name / document).is_file():
        return None
    return f"logs/targets/{name}", f"docs/agents/targets/{name}/{document}"


def capturePatch():
    patch = subprocess.check_output(["git", "diff", "HEAD", "--", *PATCH_PATHS], cwd=ROOT)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *PATCH_PATHS], cwd=ROOT).decode().splitlines()
    for filename in untracked:
        contents = (ROOT / filename).read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        patch += "".join(difflib.unified_diff([], contents, fromfile="/dev/null", tofile="b/" + filename)).encode()
    return patch


def parseMetrics(output, timeout):
    metrics = {
        "volume_reports": re.findall(r"\[SearchEngine volume\].*", output),
        "program_evaluation": re.findall(r"\[Program evaluation\].*", output),
        "overshoot": re.findall(r"\[Program deadline\] overshoot=.*", output),
        "gap_report": re.findall(r"\[Program gap\].*", output),
        "incumbent_events": re.findall(r"\[SearchEngine\] New best cost:.*", output),
        "oracle_verdict": re.findall(r"\d+/\d+ cases passed", output),
        "inference_lines": re.findall(r"\[Inference (?:compile\] supervisor|case|failure|summary)\].*", output),
        "gemma_lines": re.findall(r"\[Gemma (?:compile\] supervisor|logits\] ok|failure|prefill\+decode)\].*", output),
    }
    if metrics["gap_report"]:
        metrics["gap"] = {key: (None if value == "none" else float(value))
                          for key, value in re.findall(r"([\w-]+)=([^ ]+)", metrics["gap_report"][-1])}
    first = None
    for line in metrics["incumbent_events"]:
        event = re.search(r"cost: ([0-9.eE+\-]+).*event-epoch=([0-9.eE+\-]+) program-deadline=([0-9.eE+\-]+)", line)
        if event and timeout:
            if float(event[3]) > 0:
                elapsed = float(event[2]) - float(event[3]) + timeout
            else:
                # Without TG_PROGRAM_DEADLINE (search-deadline compiles) the
                # supervisor's gap report has the time since its start.
                elapsed = metrics.get("gap", {}).get("first-incumbent-elapsed")
                if elapsed is None:
                    raise ValueError("Incumbent event without a program deadline and no [Program gap] report")
            first = {"cost": float(event[1]), "program_elapsed": elapsed}
            break
    metrics["first_incumbent"] = first
    costs = re.findall(r"best=([0-9.eE+\-]+)", output)
    metrics["best_incumbent_cost"] = float(costs[-1]) if costs and first else (first["cost"] if first else None)
    if metrics["volume_reports"]:
        fields = dict(re.findall(r"([\w-]+)=([^ ]+)", metrics["volume_reports"][-1]))
        total, remaining = float(fields["total-domain-bits"]), float(fields["remaining-domain-bits"])
        metrics["eliminated_bits"] = float(fields["eliminated-bits"])
        metrics["volume_target_met"] = remaining <= total / 2
    return metrics


def main():
    arguments = sys.argv[1:]
    if len(arguments) < 3 or arguments[0] != "--record" or recordPaths(arguments[1]) is None:
        raise SystemExit(__doc__)
    record_name, run_id, *arguments = arguments[1:]
    native = bool(arguments) and arguments[0] == "--native-command"
    if native:
        arguments = arguments[1:]
    log_dir, document_name = recordPaths(record_name)
    directory = ROOT / log_dir
    directory.mkdir(parents=True, exist_ok=True)
    document = ROOT / document_name
    log_path = directory / (run_id + ".log")
    if log_path.exists():
        raise SystemExit("Run already exists: " + run_id)
    command = arguments if native else [sys.executable, *arguments]
    if native:
        command[0] = str((ROOT / command[0]).resolve()) if (ROOT / command[0]).exists() else command[0]
    (directory / (run_id + ".patch")).write_bytes(capturePatch())
    timeout = float(arguments[arguments.index("--timeout") + 1]) if "--timeout" in arguments else None
    record = {
        "id": run_id,
        "command": arguments if native else [".venv/Scripts/python.exe", *arguments],
        "cwd": str(ROOT),
        "start": datetime.datetime.now().astimezone().isoformat(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "patch": run_id + ".patch",
        "log": run_id + ".log",
        "timeout": timeout,
        "environment": {key: os.environ[key] for key in os.environ if key.startswith("TG_")},
        "status": "in progress",
    }
    record_path = directory / (run_id + ".json")
    record_path.write_text(json.dumps(record, indent=2))
    with document.open("a", encoding="utf-8", newline="\n") as file:
        file.write(f"\n### {run_id}\n- In progress: {record['start']}\n"
                   f"- Metadata `{log_dir}/{record_path.name}`; log `{log_dir}/{log_path.name}`; patch `{log_dir}/{run_id}.patch`\n"
                   f"- Command: `{' '.join(record['command'])}`\n")
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        try:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            exit_status = result.returncode
        except OSError:
            log.write(traceback.format_exc())
            exit_status = 127
    output = log_path.read_text(encoding="utf-8", errors="replace")
    record.update(status="complete", exit_status=exit_status, end=datetime.datetime.now().astimezone().isoformat(),
                  runner_elapsed=time.monotonic() - start, **parseMetrics(output, timeout))
    record_path.write_text(json.dumps(record, indent=2))
    with document.open("a", encoding="utf-8", newline="\n") as file:
        file.write(f"- Completed: {record['end']}; exit {exit_status}; runner elapsed {record['runner_elapsed']:.3f} s.\n")
        for line in record["volume_reports"] + record["program_evaluation"] + record["overshoot"] + record["gap_report"]:
            file.write(f"  `{line}`\n")
        if record["incumbent_events"]:
            file.write(f"  `{record['incumbent_events'][0]}`\n")
        if timeout:
            if not record["volume_reports"] or not record["gap_report"]:
                file.write("- Missing volume or gap report: FAILED evaluation.\n")
            first = record["first_incumbent"]
            file.write("- First incumbent: " + (f"{first['program_elapsed']:.9f} s, cost {first['cost']}" if first
                                               else "none within budget (+infinity)")
                       + f"; best cost {record['best_incumbent_cost']}.\n")
        if record["oracle_verdict"]:
            file.write(f"- Oracle verdict: {record['oracle_verdict'][-1]}.\n")
        for line in record["inference_lines"]:
            file.write(f"  `{line}`\n")
    print(json.dumps({k: record[k] for k in ("id", "exit_status", "first_incumbent", "best_incumbent_cost")
                      if k in record} | {"gap": record.get("gap"), "eliminated_bits": record.get("eliminated_bits")},
                     indent=2))
    sys.exit(exit_status)


if __name__ == "__main__":
    main()
