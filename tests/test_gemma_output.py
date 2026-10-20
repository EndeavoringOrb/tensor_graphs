import argparse
import datetime
import json
import os
import math
import re
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "google" / "gemma-3-270m"
# prefill+decode with --timeout: seconds between the search deadline and the
# compile budget, kept for extraction, compiling and writing the plan
# (agent-tunable; docs/agents/targets/prefill_decode_seq_len/target.md).
# Extraction and plan writing takes ~1.2s; 3.0s margin leaves an extra ~12s for search.
SEARCH_MARGIN_SECONDS = 3.0
PLAN_NAME = "plan.bin"
RUNS_DIR = PROJECT_ROOT / "logs" / "gemma_output_runs"
BENCH_EXE = PROJECT_ROOT / "tensor_graphs_cpp" / ("bench.exe" if sys.platform == "win32" else "bench")
# Saturate-only + bench rounds before giving up on reaching exact cost records.
MAX_CALIBRATION_ROUNDS = 5
# Whether the planner may use OpenCL, per machine: the Windows arm64 laptop
# plans with its GPU through OpenCL; the Linux CUDA box does not use OpenCL.
OPENCL_BY_PLATFORM = {"win32": True, "linux": False}


def pinPlatformEnvironment(env):
    """Set TG_DISABLE_OPENCL for this machine (fail loud on an unknown platform)."""
    if sys.platform not in OPENCL_BY_PLATFORM:
        raise SystemExit(f"No OpenCL policy for platform {sys.platform}; add it to OPENCL_BY_PLATFORM")
    if OPENCL_BY_PLATFORM[sys.platform]:
        env.pop("TG_DISABLE_OPENCL", None)
    else:
        env["TG_DISABLE_OPENCL"] = "1"
    return env

CONFIGS = {
    "full": {
        "compile_no_weights_bucket": False,
    },
    "no-weights": {
        "compile_no_weights_bucket": True,
    },
    "decode": {
        "token_buckets": [(i, i + 1, i, 1.0) for i in range(16)],
        "compile_no_weights_bucket": False,
    },
    # Buckets come from --seq-len (prefillDecodeBuckets); every compiled graph's
    # logits are checked against Hugging Face transformers.
    "prefill+decode": {},
}
LOGITS_CONFIGS = {"prefill+decode"}


def prefillDecodeBuckets(seq_len):
    """Prefill [0, t) and decode [t-1, t), both needing row t-1, for t = 1..seq_len.

    With the full bucket the session adds, that is 2 * seq_len + 1 buckets
    (prefill 1 and decode 1 coincide, as in tests/verify_saturation.py).
    seq_len 0 is the special case with only the full bucket (of one token).
    """
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    return [bucket for t in range(1, seq_len + 1) for bucket in ((0, t, t - 1, 1.0), (t - 1, t, t - 1, 1.0))]


def openLogitsSession(seq_len, cache_file="", min_compile_time=0.0, repo_path="", log_cost_calls=False,
                      saturate_only=False):
    import tensor_graphs

    return tensor_graphs.LLMSession(
        "gemma-3-270m",
        str(MODEL_PATH),
        tensor_graphs.HeuristicSearchDelegate(),
        min_compile_time=min_compile_time,
        max_sequence_length=max(1, seq_len),
        token_buckets=prefillDecodeBuckets(seq_len),
        cache_file=cache_file,
        disable_node_caching=False,
        disable_compilation_caching=not bool(cache_file),
        log_cost_calls=log_cost_calls,
        saturate_only=saturate_only,
        repo_path=repo_path,
    )


def referenceTokens(seq_len):
    """First seq_len tokens (with BOS) of the model card text."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    ids = tokenizer((MODEL_PATH / "README.md").read_text(encoding="utf-8"), add_special_tokens=True)["input_ids"]
    if len(ids) < seq_len:
        raise SystemExit(f"Reference text has only {len(ids)} tokens; need {seq_len}")
    if ids[0] != tokenizer.bos_token_id:
        raise SystemExit("Tokenizer did not prepend BOS")
    return [int(t) for t in ids[:seq_len]]


def referenceLogits(tokens):
    """float32 eager-attention logits of every position; row p only depends on tokens[:p + 1]."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), dtype=torch.float32,
                                                 attn_implementation="eager")
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor([tokens], dtype=torch.long), use_cache=False).logits[0]
    if logits.dtype != torch.float32 or not torch.isfinite(logits).all():
        raise SystemExit("Reference logits are not finite float32")
    return logits.numpy()


def expectedGraphRanges(seq_len):
    """(dirty_start, dirty_end, row_start, row_end) of every compiled graph: the
    requested buckets plus the full bucket the session adds (2N+1; 1 for N = 0).
    """
    length = max(1, seq_len)
    return sorted([(s, e, r, r + 1) for s, e, r, _ in prefillDecodeBuckets(seq_len)] + [(0, length, 0, length)])


def runEveryGraph(session, tokens, seq_len):
    """Run every compiled graph directly; returns [(graph, (s, e, r0, r1), rows)].

    Graph g has dirty tokens [s, e) and needed rows [r0, r1). Its persistent
    state must describe tokens[:s], so the full graph first runs on tokens[:s]
    (zero padded), then graph g on tokens[:e].
    """
    import numpy as np

    length = max(1, seq_len)
    ranges = [tuple(r) for r in session.graph_token_ranges()]
    expected = expectedGraphRanges(seq_len)
    if sorted(ranges) != expected:
        raise SystemExit(f"Compiled graphs {sorted(ranges)} do not match the requested buckets {expected}")
    full = ranges.index((0, length, 0, length))
    results = []
    for graph, (start, end, row_start, row_end) in enumerate(ranges):
        if row_end > end:
            raise SystemExit(f"Graph {graph} needs rows past its tokens: {ranges[graph]}")
        session.reset()
        if graph != full:
            session.run_graph_logits(full, tokens[:start])
        rows = session.run_graph_logits(graph, tokens[:end])
        results.append((graph, (start, end, row_start, row_end), np.array(rows[row_start:row_end], dtype=np.float32)))
    return results


def checkLogits(results, reference):
    """Compare every graph's needed rows with the reference; returns (ok, report lines)."""
    from tests.inference_benchmark import compare_row

    ok, lines = True, []
    for graph, (start, end, row_start, row_end), rows in results:
        worst = (0.0, 0.0, 0.0)
        failures = []
        for offset, row in enumerate(rows):
            position = row_start + offset
            row_ok, max_abs, lse_err, argmax_gap, reason = compare_row(row, reference[position])
            worst = tuple(max(a, b) for a, b in zip(worst, (max_abs, lse_err, argmax_gap)))
            if not row_ok:
                failures.append(f"row {position}: {reason}")
        ok = ok and not failures
        lines.append(f"[Gemma logits] graph={graph} dirty=[{start},{end}) rows=[{row_start},{row_end}) "
                     f"ok={str(not failures).lower()} max_abs_err={worst[0]:.4g} max_lse_err={worst[1]:.4g} "
                     f"max_argmax_gap={worst[2]:.4g}" + (f" failures={'; '.join(failures)}" if failures else ""))
    lines.append(f"[Gemma logits] ok={str(ok).lower()} graphs={len(results)}")
    return ok, lines


def runLogitsCheck(session, seq_len):
    tokens = referenceTokens(max(1, seq_len))
    ok, lines = checkLogits(runEveryGraph(session, tokens, seq_len), referenceLogits(tokens))
    for line in lines:
        print(line, flush=True)
    if not ok:
        raise SystemExit("Logits differ from the transformers reference")


def run_gemma_output(config_name, cache_file="", min_compile_time=0.0, saturate_only=False, seq_len=4):
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown Gemma test config: {config_name}")
    if config_name in LOGITS_CONFIGS:
        if saturate_only:
            raise ValueError(f"{config_name} does not support --saturate-only")
        runLogitsCheck(openLogitsSession(seq_len, cache_file, min_compile_time), seq_len)
        return

    import tensor_graphs
    from main import decode_tokens, encode_text
    from utils.decode import load_tokenizer

    model_path = MODEL_PATH

    disable_compilation_caching = saturate_only or not bool(cache_file)
    session = tensor_graphs.LLMSession(
        "gemma-3-270m",
        str(model_path),
        tensor_graphs.HeuristicSearchDelegate(),
        min_compile_time=min_compile_time,
        max_sequence_length=16,
        **CONFIGS[config_name],
        cache_file=cache_file,
        disable_node_caching=False,
        disable_compilation_caching=disable_compilation_caching,
        saturate_only=saturate_only,
    )

    if saturate_only:
        return

    tokenizer = load_tokenizer([str(model_path), "gemma-3-270m"])
    conversation_tokens = encode_text(tokenizer, "Hi, my name", is_first=True)
    expected_string = " is <strong>Jasmine"

    generated_string = ""
    while len(generated_string) <= len(expected_string):
        next_token = session.generate_step(conversation_tokens)
        decoded = decode_tokens(tokenizer, [next_token]) if next_token != -1 else ""
        print(f"[DEBUG_TOKEN] next_token={next_token} decoded={decoded!r}", flush=True)
        assert next_token != -1, (
            f"{config_name}: generation stopped at sequence length "
            f"{len(conversation_tokens)} after producing {generated_string!r}"
        )
        conversation_tokens.append(next_token)
        generated_string += decoded
        assert generated_string.startswith(expected_string[: len(generated_string)]), (
            f"{config_name}: generated {generated_string!r}, expected prefix of "
            f"{expected_string!r}"
        )


def test_gemma_output(cache_file="", min_compile_time=0.0, saturate_only=False):
    for config_name in CONFIGS:
        if saturate_only and config_name in LOGITS_CONFIGS:
            continue
        run_gemma_output(
            config_name,
            cache_file=cache_file,
            min_compile_time=min_compile_time,
            saturate_only=saturate_only,
        )


# ----------------------------------------------------------------------------
# Timed logits evaluation: calibration (unbudgeted), compile (budgeted) and
# check (unbudgeted) run in separate processes; the reference and comparison
# run in the supervisor.
# ----------------------------------------------------------------------------
def phaseCalibrate(run_dir, seq_len):
    """Saturate only, logging every cost lookup without an exact record to TG_COST_CALLS_PATH."""
    if "TG_COST_CALLS_PATH" not in os.environ:
        raise SystemExit("[Gemma calibrate] TG_COST_CALLS_PATH must name this round's calls log")
    openLogitsSession(seq_len, "", -1.0, str(run_dir / "calibrate_repo"), log_cost_calls=True, saturate_only=True)


def calibrate(common, run_dir):
    """Saturate-only + bench rounds until every cost lookup of this N has an exact record.

    New records can change which fused candidates saturation keeps, so a round
    can log calls the previous one did not; stop when a round logs none that
    still lack a record. Returns (ok, rounds, total benchmarked).
    """
    if not BENCH_EXE.exists():
        raise SystemExit(f"Missing {BENCH_EXE}; build it: .venv/Scripts/python.exe build.py --targets bench")
    total = 0
    for round_index in range(1, MAX_CALIBRATION_ROUNDS + 1):
        calls = run_dir / f"calibrate{round_index}_calls.bin"
        env = pinPlatformEnvironment(os.environ.copy())
        for key in ("TG_PROGRAM_DEADLINE", "TG_SEARCH_DEADLINE", "TG_SEARCH_VOLUME", "TG_SEARCH_GAP"):
            env.pop(key, None)
        env["TG_COST_CALLS_PATH"] = str(calls)
        code, output, _ = runChild(["--phase", "calibrate", *common], env, None, echo=False)
        (run_dir / f"calibrate{round_index}.log").write_text(output, encoding="utf-8")
        if code != 0:
            print(f"[Gemma failure] calibration round {round_index} saturate-only run exited {code}", flush=True)
            return False, round_index, total
        bench_log = run_dir / f"calibrate{round_index}_bench.log"
        with open(bench_log, "w", encoding="utf-8", errors="replace") as log:
            bench = subprocess.run([str(BENCH_EXE), "--calls", str(calls)], cwd=PROJECT_ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        summaries = re.findall(r"\[Bench summary\] pending=(\d+) benchmarked=(\d+) failed=(\d+)",
                               bench_log.read_text(encoding="utf-8", errors="replace"))
        if bench.returncode != 0 or not summaries:
            print(f"[Gemma failure] bench exited {bench.returncode} without a summary; see {bench_log}", flush=True)
            return False, round_index, total
        pending, benchmarked, failed = (int(v) for v in summaries[-1])
        total += benchmarked
        print(f"[Gemma calibrate] round={round_index} pending={pending} benchmarked={benchmarked} failed={failed} "
              f"log={bench_log}", flush=True)
        if pending == 0:
            return True, round_index, total
        if benchmarked == 0:
            print(f"[Gemma failure] {pending} cost configurations cannot be benchmarked; see {bench_log}", flush=True)
            return False, round_index, total
    print(f"[Gemma failure] cost records still incomplete after {MAX_CALIBRATION_ROUNDS} rounds", flush=True)
    return False, MAX_CALIBRATION_ROUNDS, total


def phaseCompile(run_dir, seq_len):
    started = time.time()
    openLogitsSession(seq_len, str(run_dir / PLAN_NAME), -1.0, str(run_dir / "repo"))
    plans = sorted(p.name for p in run_dir.iterdir() if p.name.startswith(PLAN_NAME))
    if not plans:
        raise SystemExit("[Gemma compile] no plan was written to the run directory")
    print(f"[Gemma compile] elapsed={time.time() - started:.3f} plans={','.join(plans)}", flush=True)


def phaseCheck(run_dir, seq_len):
    import numpy as np

    tokens = referenceTokens(max(1, seq_len))
    session = openLogitsSession(seq_len, str(run_dir / PLAN_NAME), -1.0, str(run_dir / "repo"))
    results = runEveryGraph(session, tokens, seq_len)
    np.savez(run_dir / "logits.npz", tokens=np.array(tokens, dtype=np.int64),
             ranges=np.array([r for _, r, _ in results], dtype=np.int64),
             **{f"graph{graph}": rows for graph, _, rows in results})
    print(f"[Gemma check] graphs={len(results)} saved={run_dir / 'logits.npz'}", flush=True)


def runChild(arguments, env, timeout, echo=True):
    """Run one phase; timeout None means no limit."""
    command = [sys.executable, "-u", "-m", "tests.test_gemma_output", *arguments]
    process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace")
    try:
        output, _ = process.communicate(timeout=None if timeout is None else max(0.001, timeout))
        timed_out = False
    except subprocess.TimeoutExpired:
        process.kill()
        output, _ = process.communicate()
        timed_out = True
    if echo or process.returncode != 0:
        print(output, end="", flush=True)
    return process.returncode, output, timed_out


def superviseLogits(args):
    """Calibrate cost records (untimed), compile within --timeout seconds, then check every graph's logits.

    Prints `[Gemma prefill+decode] ok=... final-gap=... gap-target-met=...`
    and writes <run-dir>/results.json. Exit 0 iff the compile finished within
    the budget, loaded without replanning and every logits row is correct.
    """
    import numpy as np
    from tests.search_gap import formatGapReport, gapIntegral, parseGapSamples

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else RUNS_DIR / f"{args.config}-seq{args.seq_len}-{stamp}"
    run_dir = run_dir.resolve()
    if run_dir.exists():
        raise SystemExit(f"Run directory {run_dir} already exists; every run starts from an empty directory")
    run_dir.mkdir(parents=True)
    common = ["--config", args.config, "--seq-len", str(args.seq_len), "--run-dir", str(run_dir)]
    if args.ignore_checkpoints:
        common.append("--ignore-checkpoints")
    buckets = len(expectedGraphRanges(args.seq_len))
    summary = {"config": args.config, "seq_len": args.seq_len, "buckets": buckets,
               "budget": args.timeout, "search_margin": SEARCH_MARGIN_SECONDS}

    # Untimed: benchmark every cost configuration this N looks up, so the timed
    # compile (which does not log cost calls) uses exact records only.
    calibrated, rounds, benchmarked = calibrate(common, run_dir)
    summary.update(calibrated=calibrated, calibration_rounds=rounds, calibration_benchmarked=benchmarked)
    if not calibrated:
        (run_dir / "results.json").write_text(json.dumps(dict(summary, ok=False), indent=2), encoding="utf-8")
        print(f"[Gemma {args.config}] ok=false seq_len={args.seq_len} buckets={buckets} calibrated=false "
              f"run_dir={run_dir}", flush=True)
        raise SystemExit(1)

    started = time.time()
    env = pinPlatformEnvironment(os.environ.copy())
    env["TG_SEARCH_DEADLINE"] = repr(started + args.timeout - SEARCH_MARGIN_SECONDS)
    env["TG_SEARCH_VOLUME"] = "1"
    env["TG_SEARCH_GAP"] = "1"
    env.pop("TG_PROGRAM_DEADLINE", None)
    code, output, timed_out = runChild(["--phase", "compile", *common], env, args.timeout - (time.time() - started))
    compile_elapsed = time.time() - started
    ok = code == 0 and not timed_out
    print(f"[Gemma compile] supervisor elapsed={compile_elapsed:.3f} budget={args.timeout} exit={code} "
          f"timed_out={str(timed_out).lower()}", flush=True)
    summary.update(compile_elapsed=compile_elapsed, compile_exit=code, compile_timed_out=timed_out)
    gap = None
    if ok:
        try:
            gap = gapIntegral(parseGapSamples(output), started, args.timeout)
            print(formatGapReport(gap), flush=True)
        except ValueError as error:
            print(f"[Gemma failure] invalid search gap samples: {error}", flush=True)
            ok = False
    summary["gap"] = gap

    logits_ok = False
    if ok:
        env = pinPlatformEnvironment(os.environ.copy())
        for key in ("TG_PROGRAM_DEADLINE", "TG_SEARCH_DEADLINE", "TG_SEARCH_VOLUME", "TG_SEARCH_GAP"):
            env.pop(key, None)
        code, output, timed_out = runChild(["--phase", "check", *common], env, args.check_timeout)
        replanned = "Planning new execution graph" in output or "Using cached compilation" not in output
        if replanned:
            print("[Gemma failure] check phase did not load the compiled plan (it planned again)", flush=True)
        summary.update(check_exit=code, check_timed_out=timed_out, check_replanned=replanned)
        ok = code == 0 and not timed_out and not replanned
    if ok:
        saved = np.load(run_dir / "logits.npz")
        ranges = [tuple(int(v) for v in r) for r in saved["ranges"]]
        results = [(graph, ranges[graph], saved[f"graph{graph}"]) for graph in range(len(ranges))]
        logits_ok, lines = checkLogits(results, referenceLogits([int(t) for t in saved["tokens"]]))
        for line in lines:
            print(line, flush=True)
        summary["graphs"] = len(results)
        ok = logits_ok

    final_gap = gap["final_gap"] if gap else 1.0
    target_met = ok and final_gap < 0.05
    summary.update(logits_ok=logits_ok, ok=ok, final_gap=final_gap, gap_target_met=target_met)
    (run_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Gemma {args.config}] ok={str(ok).lower()} seq_len={args.seq_len} buckets={buckets} "
          f"compile_elapsed={compile_elapsed:.3f} final-gap={final_gap:.6g} "
          f"gap-target-met={str(target_met).lower()} run_dir={run_dir}", flush=True)
    raise SystemExit(0 if ok else 1)


def main():
    parser = argparse.ArgumentParser(description="Check Gemma output for bucket configurations")
    parser.add_argument(
        "--config",
        choices=["all", *CONFIGS],
        default="all",
        help="Configuration to test (default: all)",
    )
    parser.add_argument(
        "--cache-file",
        "--cache",
        dest="cache_file",
        type=str,
        default="",
        help="Path to compiled cache file. If specified, enables compilation caching to/from this file.",
    )
    parser.add_argument(
        "--min-compile-time",
        type=float,
        default=0.0,
        help="Search time budget in seconds (0 stops at the first feasible plan)",
    )
    parser.add_argument(
        "--optimal",
        action="store_true",
        help="Search until the planner exhausts its search space",
    )
    parser.add_argument(
        "--saturate-only",
        action="store_true",
        help="Run saturation/planning only to log cost model calls without running the model",
    )
    parser.add_argument(
        "--search-volume",
        action="store_true",
        help="Report remaining frontier hyperbox volume relative to initial domains, in bits",
    )
    parser.add_argument("--timeout", type=float,
                        help="Whole worker process budget in seconds; report frontier and stop. "
                             "For prefill+decode: compile budget, followed by an unbudgeted logits check")
    parser.add_argument("--seq-len", type=int, default=4,
                        help="prefill+decode: sequence length N; compiles 2N+1 buckets, "
                             "only the one-token full bucket for N = 0 (default: 4)")
    parser.add_argument("--run-dir", help="prefill+decode with --timeout: fresh directory for the plan and results")
    parser.add_argument("--check-timeout", type=float, default=3600.0,
                        help="prefill+decode with --timeout: limit for the logits check process")
    parser.add_argument("--ignore-checkpoints", action="store_true",
                        help="Ignore dumped checkpoints and recompute from scratch (used for final benchmarking)")
    parser.add_argument("--phase", choices=["calibrate", "compile", "check"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.ignore_checkpoints:
        os.environ["TG_IGNORE_CHECKPOINTS"] = "1"
    if args.seq_len < 0:
        parser.error("--seq-len must be non-negative")
    if args.phase:
        run_dir = Path(args.run_dir).resolve()
        if args.phase == "calibrate":
            phaseCalibrate(run_dir, args.seq_len)
        elif args.phase == "compile":
            phaseCompile(run_dir, args.seq_len)
        else:
            phaseCheck(run_dir, args.seq_len)
        return
    if args.timeout is not None:
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            parser.error("--timeout must be finite and positive")
        if args.config == "all" or not args.optimal or args.cache_file or args.saturate_only:
            parser.error("--timeout requires one --config, --optimal, and no compiled cache")
        if args.config in LOGITS_CONFIGS:
            if args.timeout <= SEARCH_MARGIN_SECONDS:
                parser.error(f"--timeout must exceed the {SEARCH_MARGIN_SECONDS} s search margin")
            superviseLogits(args)
        if "TG_PROGRAM_DEADLINE" not in os.environ:
            start_time = time.monotonic()
            environment = os.environ.copy()
            program_deadline = time.time() + args.timeout
            environment["TG_PROGRAM_DEADLINE"] = repr(program_deadline)
            environment["TG_SEARCH_VOLUME"] = "1"
            environment["TG_SEARCH_GAP"] = "1"
            command = [sys.executable, "-u", "-m", "tests.test_gemma_output", *sys.argv[1:]]
            worker = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True)
            try:
                output, _ = worker.communicate(timeout=max(0.001, args.timeout - (time.monotonic() - start_time)) + 2.0)
            except subprocess.TimeoutExpired:
                worker.kill()
                output, _ = worker.communicate()
                print(output, end="", flush=True)
                raise SystemExit("Evaluation failed: program exceeded deadline and reporting grace")
            elapsed = time.monotonic() - start_time
            print(output, end="", flush=True)
            print(f"[Program evaluation] elapsed={elapsed:.9f} budget={args.timeout}", flush=True)
            if worker.returncode != 0 or "[SearchEngine volume]" not in output:
                raise SystemExit("Evaluation failed: worker failed or did not report frontier volume")
            overshoots = re.findall(r"\[Program deadline\] overshoot=([0-9.eE+\-]+)", output)
            overshoot = float(overshoots[-1]) if overshoots else max(0.0, elapsed - args.timeout)
            if not math.isfinite(overshoot) or overshoot > 1.0:
                raise SystemExit("Evaluation failed: deadline overshoot exceeds 1 second")
            from tests.search_gap import formatGapReport, gapIntegral, parseGapSamples
            try:
                gap = gapIntegral(parseGapSamples(output), program_deadline - args.timeout, args.timeout)
            except ValueError as error:
                raise SystemExit(f"Evaluation failed: invalid search gap samples: {error}")
            print(formatGapReport(gap), flush=True)
            return
    if args.search_volume:
        os.environ["TG_SEARCH_VOLUME"] = "1"

    config_names = CONFIGS if args.config == "all" else [args.config]
    for config_name in config_names:
        print(f"\n=== Gemma output config: {config_name} ===", flush=True)
        run_gemma_output(
            config_name,
            cache_file=args.cache_file,
            min_compile_time=-1.0 if args.optimal else args.min_compile_time,
            saturate_only=args.saturate_only,
            seq_len=args.seq_len,
        )

    if args.saturate_only:
        print("[Main] Saturate-only complete. Cost model calls logged.")


if __name__ == "__main__":
    main()
