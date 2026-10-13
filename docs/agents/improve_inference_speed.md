# Improve inference speed (prefill and decode tokens per second)

Make Gemma-3-270m run faster end to end: higher prefill (pp) and decode (tg) throughput at several context lengths, with output that matches a fixed Hugging Face transformers reference. Any part of the system may change: buckets (prefill and decode, partially symbolic or shared), planner, search, kernels, memory layout, runtime dispatch. The constraints are correctness, precision, the compile budget and generality. This is the first rung towards much longer contexts (131072 and beyond) and larger models (DeepSeek V4 Flash, GLM 5.3), so prefer changes that scale with sequence length.

## Execution mandate

When invoked, treat the evaluation contract below as fixed and run an autonomous optimization loop. Establish (or reuse) the baseline, then implement and evaluate improvements. A failed or rejected candidate ends one evaluation, not the task. Continue without asking permission for routine edits, rebuilds, evaluations and diagnostics. If a concrete blocker prevents further work, report it with the work completed rather than asking whether to start the next routine step. Record every retained change and every rejected trial.

## Fixed evaluation contract

```powershell
# Build (after native changes)
.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO
# One-time setup: the HF reference (already generated; regenerate only if tests/inference_cases.py changes)
.venv/Scripts/python.exe -m tests.inference_reference
# Evaluation (one run; always through the runner)
.venv/Scripts/python.exe tests/search_experiment.py --record inference <run_id> -m tests.inference_benchmark --run-dir logs/inference_runs/<run_id>
```

`tests/inference_benchmark.py` runs two processes:
1. **Compile.** `tests.inference_session.open_session(cache_file, deadline)` builds the session and writes the compiled plan into the fresh run directory. Everything (process start, imports, model loading, saturation, search, extraction, writing the plan) must finish within **600 s**, counted from launching the process. The process is killed at 600 s, and a killed or failed compile fails the run.
2. **Benchmark.** A new process opens the session from that plan without planning again (a run that plans in this phase fails). It then runs every case: one warm-up and three timed repetitions by default. Loading is not timed; only the `step_logits` calls are.

Run evaluations one at a time with nothing else running (throughput is sensitive to load and thermals). Diagnostic runs may use `--cases`, `--reps`, `--warmup` or a smaller compile budget, but are not evaluations.

The correctness suites still apply to the parts they cover: after planner or propagator changes run `tests.test_planner_oracle` (including the certified-bound check) and `tests.test_search_gap`. After native changes run the native suites (`tensor_graphs_cpp/test.exe` with `--private-layout`, `--write-after-read-cache`, `--incremental-propagators`, `--start-rank-window`, `--relaxation-bound`). Add tests for new kernels.

## Cases

All cases are slices of one fixed 1024-token sequence S (BOS + the model card text; token ids stored in `tests/inference_reference/gemma-3-270m.json`). Definitions are in `tests/inference_cases.py`.

| Case | Untimed setup | Timed work | Logits checked |
|---|---|---|---|
| `pp8@0` | none | prefill S[0:8] in one step | position 7 |
| `tg8@8` | prefill S[0:8] | 8 decode steps, positions 8..15 | each step |
| `pp64@64` | prefill S[0:64] | prefill S[64:128] in one step | position 127 |
| `tg64@64` | prefill S[0:64] | 64 decode steps, positions 64..127 | each step |
| `pp896@0` | none | prefill S[0:896] in one step | position 895 |
| `tg128@896` | prefill S[0:896] | 128 decode steps, positions 896..1023 | each step |

Decode is teacher-forced (the next input is always S, never our own sample), so a small numeric difference cannot change later inputs. Each repetition starts with `reset()`, so the context prefill recomputes from scratch, and the timed step's dirty region is exactly the new tokens.

## Metrics

| Metric | Source line | Direction |
|---|---|---|
| **Primary: score** = geometric mean over cases of median tok/s | `[Inference summary] score=` | higher is better |
| Per-case median tok/s (min/max over repetitions) | `[Inference case] ... tok_per_s=` | higher is better |
| Correctness (every checked row) | `correct=` | must be true |
| Compile time | `[Inference compile] supervisor elapsed=` | must be ≤ 600 s |

`pp` tok/s = prefill tokens / step time; `tg` tok/s = decode tokens / summed step times.

## Correctness

The HF reference stores the **full** float32 logits row (all 262144 entries) of every checked position (`tests/inference_reference/gemma-3-270m.logits.npy`, 202 positions × 262144, ~202 MiB, git-ignored; its SHA-256 is in the manifest and is verified before every run). Every returned row is compared:
- every logit: `|ours − ref| ≤ 0.02 + 0.001·|ref|`;
- `|logsumexp(ours) − logsumexp(ref)| ≤ 0.02`;
- the reference logit at our argmax is within 0.02 of the reference maximum;
- no non-finite values.

These tolerances, the reference files and the comparison code are fixed. A run with any failing row fails, whatever its speed. If the tolerances turn out to be wrong for the current implementation for a reason other than a bug, stop and report it instead of changing them.

**Known issues to fix first** (found by reading the code, not by running it). These will probably make the reference comparison fail at first, and fixing them is part of this task (correctness before speed):
- `tensor_graphs_cpp/models/gemma-3-270m.hpp` uses one RoPE base (10,000) for every layer. HF Gemma 3 uses `rope_theta = 1,000,000` on the full-attention layers (every 6th layer) and `rope_local_base_freq = 10,000` on the sliding layers.
- The graph has no sliding-window mask; HF restricts the 15 sliding layers to the last 512 positions. This matters for every position ≥ 512 (`pp896@0`, `tg128@896`).
- `Session::run` and `LLMSession::generate_step` print debug lines on every call. These cost time inside the measured steps, so remove or gate them.

## Rules

**Precision may not change.** Weights are bfloat16 in the checkpoint and are converted exactly to float32. All arithmetic, activations, the KV cache and the logits stay float32. Changes that keep every value identical are allowed. For example, a weight may stay bfloat16 in memory and be converted inside a kernel (the conversion is exact), or operations may be fused or reordered. Not allowed: quantization, lower-precision storage of anything computed (activations, KV cache), lower-precision accumulation, and approximate math (e.g. fast exp or tanh) less accurate than the float32 library functions in use. The correctness check is a backstop, not the definition: a precision reduction that happens to pass is still forbidden.

**Everything happens inside the timed compile.** No reuse of plans, saturated e-graphs, search results or other artifacts from earlier runs: the run directory starts empty, and the supervisor refuses to run if `benchmarks/repo_gemma-3-270m-seq1024.reftensors` holds tensors. No offline solver runs or precomputed answers. The one exception is the cost model's kernel timing records (`benchmarks/records.bin`). They are hardware calibration and may be extended by benchmarking kernels with the existing calibration tooling. They hold kernel timings only, never plans or schedules, and every addition must be recorded with the command that produced it.

**The benchmark phase runs the compiled plan.** It may load data, allocate memory and prepare device programs, but it may not plan, search or choose a plan using information that was not available during the compile phase.

**Generality.** Buckets, kernels and dispatch must follow general rules, e.g. prefill and decode buckets for ranges of positions, or power-of-two lengths. They may not special-case the benchmark's exact (context, tokens) pairs or token ids. Any request (any context and any number of new tokens up to the maximum sequence length) must still produce correct logits, using a slower path when no specialized plan fits. Exploiting structure common to transformers is welcome (repeated layers, KV caches, causal masks, sliding windows, GQA). Hard-coding facts about this one checkpoint is not.

**Kernels** follow `AGENTS.md`: no branching inside a kernel to choose between variants (that is the planner's job); capability guards wrap whole kernel files; fail loudly.

**Fixed files** (changing them needs separate review and a new baseline): `tests/inference_benchmark.py`, `tests/inference_cases.py`, `tests/inference_reference.py`, the reference files under `tests/inference_reference/`, and the timing and correctness rules above. `tests/inference_session.py` is the agent's entry point: change it freely within its documented contract.

## Acceptance gate

Run each candidate once. If promising, run two confirmations (three runs total) with identical source and build. Retain the change, labelled an **optimization**, only if:
1. **Correctness:** all three runs pass the reference comparison; the compile finishes within 600 s in every run; the suites that apply to the changed code pass.
2. **Primary improvement:** the median score of the three runs is higher than the accepted implementation's median by more than the spread between the accepted implementation's own runs. Repeat uncertain comparisons rather than accepting noise.
3. **No case regression:** no case's median tok/s (over the three runs) is more than 5 % below the accepted implementation's.

A change that only fixes correctness (for example the known RoPE and sliding-window issues) is retained when it passes the reference comparison and does not reduce any case's throughput by more than 5 %. The first implementation that passes the comparison becomes the baseline for speed.

## Iteration

1. State a specific hypothesis and the cases it should speed up.
2. Make one focused change, with tests for new kernels or planner rules.
3. Rebuild, run the applicable suites, then the benchmark through `tests/search_experiment.py --record inference`.
4. Compare with the accepted implementation. Run confirmations when promising.
5. Retain or reject. Keep rejected trials' patches and logs (the runner saves a patch per run), and restore the accepted source before the next trial. Touch restored files so the build recompiles them.

Profile freely (`build.py --profile`, temporary timing logs, `--cases` subsets), but label such runs diagnostics and remove instrumentation before an evaluation.

## Logs and resumable progress

Keep `docs/agents/INFERENCE_IMPROVEMENTS.md` and `logs/inference_improvements/` sufficient to resume without conversation history. The runner appends an in-progress and a completion entry per run with the result lines. After each run add a short note: the hypothesis; per-case tok/s and the score compared with the accepted implementation; the gate verdict and its reason. Maintain:
- a **comparison table** (run, implementation, score, per-case median tok/s, compile seconds, correct);
- a **resume checkpoint**: the accepted implementation and its patch, the working-tree and build state, any running process and its log, and the next concrete step.

Never append to the record with Windows PowerShell's `Add-Content`/`Set-Content`, which write ANSI. Use Python with UTF-8 or the Edit tool.

## Starting points (verify before relying on them)

- The default `tests/inference_session.py` compiles only the full bucket (one plan for the whole 1024-token graph), so every step recomputes the whole sequence. Decode steps are therefore as slow as a full prefill. Adding decode and prefill buckets, or symbolic and shared ones, is the obvious first speed lever once outputs are correct. One bucket per length and phase would mean 2 × 1024 buckets; the current per-bucket planning cost (seconds per bucket) makes that infeasible within 600 s.
- `LLMSession.step_logits(tokens)` diffs `tokens` against the previous call to form the dirty region, writes the input, runs the best compiled bucket for the request (`Session::getBestGraphIdx`) and copies one logits row back.
- The planner work in `docs/agents/GAP_IMPROVEMENTS.md` (relaxation bounds, root primal heuristic, faster preprocessing) and its notes on startup costs apply directly to the compile budget.
