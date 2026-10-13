"""Inference speed benchmark (fixed contract; docs/agents/improve_inference_speed.md).

Usage (from the repository root):
  .venv/Scripts/python.exe -m tests.inference_benchmark --run-dir logs/inference_runs/<run_id>

Two phases, each in its own process:
1. compile: tests.inference_session.open_session(cache_file, deadline) must
   build and write the compiled plan within --compile-budget seconds (600),
   counted from launching the process. The process is killed at the budget.
2. benchmark: a fresh process opens the session from the plan written in
   phase 1 (it must not plan again), then runs every case of
   tests/inference_cases.py: warm-up repetitions, then timed repetitions.
   Only the step_logits calls are timed. Every returned logits row is
   compared with the HF transformers reference (all vocabulary entries).

Prints `[Inference case] ...` per case and `[Inference summary] ...`; writes
<run-dir>/results.json. Exit status 0 only if both phases succeed and every
checked row is correct.
"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
import time

import numpy as np

from tests.inference_cases import CASES, MODEL_NAME, REFERENCE_PATH, SEQUENCE_LENGTH, checked_positions

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Correctness tolerances against the float32 HF reference (fixed; see the doc).
ABS_TOL = 2e-2  # per logit: |ours - ref| <= ABS_TOL + REL_TOL * |ref|
REL_TOL = 1e-3
LSE_TOL = 2e-2  # |logsumexp(ours) - logsumexp(ref)|
ARGMAX_TOL = 2e-2  # ref[argmax(ours)] >= max(ref) - ARGMAX_TOL

PLAN_NAME = "plan.bin"


# ----------------------------------------------------------------------------
# Reference
# ----------------------------------------------------------------------------
def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference(verify_hash=True):
    manifest_path = ROOT / REFERENCE_PATH
    if not manifest_path.exists():
        raise SystemExit(f"Missing reference manifest {manifest_path}; run -m tests.inference_reference")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    logits_path = manifest_path.parent / manifest["logits_file"]
    if not logits_path.exists():
        raise SystemExit(f"Missing reference logits {logits_path}; run -m tests.inference_reference")
    if verify_hash and sha256_file(logits_path) != manifest["logits_sha256"]:
        raise SystemExit(f"Reference logits {logits_path} do not match the manifest SHA-256")
    if manifest["model"] != MODEL_NAME or manifest["sequence_length"] != SEQUENCE_LENGTH:
        raise SystemExit("Reference manifest does not match tests/inference_cases.py")
    if manifest["cases"] != CASES:
        raise SystemExit("Reference was generated for different cases; regenerate it")
    if len(manifest["tokens"]) != SEQUENCE_LENGTH:
        raise SystemExit("Reference token sequence has the wrong length")
    logits = np.load(logits_path, mmap_mode="r")
    if list(logits.shape) != manifest["logits_shape"] or logits.dtype != np.float32:
        raise SystemExit("Reference logits have an unexpected shape or dtype")
    row_of = {position: row for row, position in enumerate(manifest["positions"])}
    return manifest, logits, row_of


def logsumexp(row):
    row = row.astype(np.float64)
    peak = row.max()
    return float(peak + math.log(np.exp(row - peak).sum()))


def compare_row(ours, ref):
    """Returns (ok, max_abs_err, lse_err, argmax_gap, reason)."""
    ours = np.asarray(ours)
    if ours.dtype != np.float32 or ours.shape != ref.shape:
        return False, math.inf, math.inf, math.inf, f"logits row has dtype {ours.dtype} shape {ours.shape}"
    if not np.isfinite(ours).all():
        return False, math.inf, math.inf, math.inf, "non-finite logits"
    ref64 = ref.astype(np.float64)
    diff = np.abs(ours.astype(np.float64) - ref64)
    max_abs = float(diff.max())
    within = bool((diff <= ABS_TOL + REL_TOL * np.abs(ref64)).all())
    lse_err = abs(logsumexp(ours) - logsumexp(ref))
    argmax_gap = float(ref64.max() - ref64[int(np.argmax(ours))])
    reasons = []
    if not within:
        reasons.append(f"logit error {max_abs:.4g} beyond tolerance")
    if lse_err > LSE_TOL:
        reasons.append(f"logsumexp error {lse_err:.4g}")
    if argmax_gap > ARGMAX_TOL:
        reasons.append(f"argmax differs (reference gap {argmax_gap:.4g})")
    return not reasons, max_abs, lse_err, argmax_gap, "; ".join(reasons)


# ----------------------------------------------------------------------------
# Phases (child processes)
# ----------------------------------------------------------------------------
def phase_compile(run_dir):
    from tests.inference_session import open_session

    deadline = float(os.environ["TG_COMPILE_DEADLINE"])
    started = time.time()
    open_session(str(run_dir / PLAN_NAME), deadline)
    plans = sorted(p.name for p in run_dir.iterdir() if p.name.startswith(PLAN_NAME))
    if not plans:
        raise SystemExit("[Inference compile] no plan was written to the run directory")
    print(f"[Inference compile] elapsed={time.time() - started:.3f} plans={','.join(plans)}", flush=True)


def phase_bench(run_dir, warmup, reps, case_names):
    from tests.inference_session import open_session

    manifest, ref_logits, row_of = load_reference(verify_hash=False)  # verified by the supervisor
    tokens = [int(t) for t in manifest["tokens"]]
    loaded = time.time()
    session = open_session(str(run_dir / PLAN_NAME), None)
    print(f"[Inference load] elapsed={time.time() - loaded:.3f}", flush=True)

    results = []
    all_correct = True
    for case in CASES:
        if case_names and case["name"] not in case_names:
            continue
        context, count, kind = case["context"], case["tokens"], case["kind"]
        positions = checked_positions(case)
        rates, failures = [], []
        worst_abs = worst_lse = worst_gap = 0.0
        for rep in range(warmup + reps):
            session.reset()
            if context > 0:
                session.step_logits(tokens[:context])  # untimed context prefill
            elapsed = 0.0
            rows = []
            if kind == "pp":
                start = time.perf_counter()
                row = session.step_logits(tokens[:context + count])
                elapsed = time.perf_counter() - start
                rows.append((positions[0], row))
            else:
                for position in positions:
                    start = time.perf_counter()
                    row = session.step_logits(tokens[:position + 1])
                    elapsed += time.perf_counter() - start
                    rows.append((position, row))
            for position, row in rows:
                ok, max_abs, lse_err, gap, reason = compare_row(row, ref_logits[row_of[position]])
                worst_abs, worst_lse, worst_gap = max(worst_abs, max_abs), max(worst_lse, lse_err), max(worst_gap, gap)
                if not ok and len(failures) < 5:
                    failures.append(f"rep {rep} position {position}: {reason}")
                all_correct &= ok
            if rep >= warmup:
                rates.append(count / elapsed)
        correct = not failures
        result = {
            "name": case["name"], "kind": kind, "context": context, "tokens": count,
            "tok_per_s": statistics.median(rates), "min_tok_per_s": min(rates), "max_tok_per_s": max(rates),
            "reps": reps, "warmup": warmup, "correct": correct, "max_abs_err": worst_abs,
            "max_lse_err": worst_lse, "max_argmax_gap": worst_gap, "failures": failures,
        }
        results.append(result)
        print(f"[Inference case] name={case['name']} kind={kind} context={context} tokens={count} "
              f"tok_per_s={result['tok_per_s']:.6g} min_tok_per_s={result['min_tok_per_s']:.6g} "
              f"max_tok_per_s={result['max_tok_per_s']:.6g} reps={reps} correct={str(correct).lower()} "
              f"max_abs_err={worst_abs:.4g} max_lse_err={worst_lse:.4g} max_argmax_gap={worst_gap:.4g}", flush=True)
        for failure in failures:
            print(f"[Inference failure] case={case['name']} {failure}", flush=True)
    (run_dir / "bench_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if not all_correct:
        raise SystemExit(2)


# ----------------------------------------------------------------------------
# Supervisor
# ----------------------------------------------------------------------------
def run_child(arguments, env, timeout):
    command = [sys.executable, "-u", "-m", "tests.inference_benchmark", *arguments]
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace")
    try:
        output, _ = process.communicate(timeout=max(0.001, timeout))
        timed_out = False
    except subprocess.TimeoutExpired:
        process.kill()
        output, _ = process.communicate()
        timed_out = True
    print(output, end="", flush=True)
    return process.returncode, output, timed_out


def supervise(args):
    run_dir = (ROOT / args.run_dir).resolve()
    if run_dir.exists():
        raise SystemExit(f"Run directory {run_dir} already exists; every run starts from an empty directory")
    manifest, _, _ = load_reference(verify_hash=True)
    stale = ROOT / "benchmarks" / f"repo_{MODEL_NAME}-seq{SEQUENCE_LENGTH}.reftensors"
    if stale.exists() and stale.stat().st_size > 0:
        raise SystemExit(f"{stale} holds tensors from an earlier run; remove it so the compile starts from scratch")
    run_dir.mkdir(parents=True)

    summary = {"compile_budget": args.compile_budget, "reference_sha256": manifest["logits_sha256"]}
    env = os.environ.copy()
    started = time.time()
    env["TG_COMPILE_DEADLINE"] = repr(started + args.compile_budget)
    code, output, timed_out = run_child(["--phase", "compile", "--run-dir", str(run_dir)], env,
                                        args.compile_budget - (time.time() - started))
    compile_elapsed = time.time() - started
    summary.update(compile_elapsed=compile_elapsed, compile_exit=code, compile_timed_out=timed_out)
    print(f"[Inference compile] supervisor elapsed={compile_elapsed:.3f} budget={args.compile_budget} "
          f"exit={code} timed_out={str(timed_out).lower()}", flush=True)
    ok = code == 0 and not timed_out

    if ok:
        env = os.environ.copy()
        for key in ("TG_PROGRAM_DEADLINE", "TG_COMPILE_DEADLINE", "TG_SEARCH_VOLUME", "TG_SEARCH_GAP"):
            env.pop(key, None)
        arguments = ["--phase", "bench", "--run-dir", str(run_dir), "--warmup", str(args.warmup),
                     "--reps", str(args.reps)]
        if args.cases:
            arguments += ["--cases", *args.cases]
        code, output, timed_out = run_child(arguments, env, args.bench_timeout)
        replanned = "Planning new execution graph" in output or "Using cached compilation" not in output
        summary.update(bench_exit=code, bench_timed_out=timed_out, bench_replanned=replanned)
        if replanned:
            print("[Inference failure] benchmark phase did not load the compiled plan (it planned again)", flush=True)
        ok = code == 0 and not timed_out and not replanned

    results_path = run_dir / "bench_results.json"
    cases = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else []
    correct = bool(cases) and all(case["correct"] for case in cases)
    complete = len(cases) == (len(args.cases) if args.cases else len(CASES))
    score = math.exp(statistics.mean(math.log(case["tok_per_s"]) for case in cases)) if cases else 0.0
    ok = ok and correct and complete
    summary.update(cases=cases, correct=correct, complete=complete, score=score, ok=ok)
    (run_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Inference summary] ok={str(ok).lower()} correct={str(correct).lower()} "
          f"complete={str(complete).lower()} score={score:.6g} compile_elapsed={compile_elapsed:.3f} "
          f"cases={len(cases)}", flush=True)
    sys.exit(0 if ok else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, help="fresh directory for the plan and results")
    parser.add_argument("--phase", choices=["compile", "bench"], help=argparse.SUPPRESS)
    parser.add_argument("--compile-budget", type=float, default=600.0)
    parser.add_argument("--bench-timeout", type=float, default=7200.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--cases", nargs="*", help="subset of case names (diagnostics only)")
    args = parser.parse_args()
    if args.warmup < 0 or args.reps < 1:
        parser.error("--warmup must be >= 0 and --reps >= 1")
    run_dir = (ROOT / args.run_dir).resolve()
    if args.phase == "compile":
        phase_compile(run_dir)
    elif args.phase == "bench":
        phase_bench(run_dir, args.warmup, args.reps, args.cases)
    else:
        supervise(args)


if __name__ == "__main__":
    main()
