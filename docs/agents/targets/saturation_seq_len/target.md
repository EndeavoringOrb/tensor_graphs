# Saturate longer prefill+decode bucket sets in under 30 seconds

Increase the sequence length N for which `tests/verify_saturation.py` saturates all 2N+1 buckets (1 full + N prefill + N decode) in under 30 seconds, while every bucket's pruned e-graph stays exactly equivalent to the reference build's. The long-term goal is N = 131,072. The rules in `docs/agents/targets/README.md` apply.

This target replaces the goal stated at the top of `progress.md` ("< 100 ms at seq_len 4"), which was met at Step 2. Since Step 10 the record has been the largest N under 30 s. The current record is N = 805 at 29,741 ms (Step 24); a later gold regeneration changed the reference graph, so re-measure before comparing.

## Objective

- **Metric:** N\*, the largest `--seq-len N` whose run prints `VERDICT: PASS (EGraph Equivalent)` with `CANDIDATE SCORE` below 30,000 ms.
- **Goal:** N = 131,072. Keep raising N\* while there are plausible, sound ideas.
- **Secondary:** score at fixed N (`--seq-len 4`, `--seq-len 16`, and the accepted N\*), lower is better.

## Fixed evaluation contract

```powershell
.venv/Scripts/python.exe build.py --targets bindings --log-level INFO
.venv/Scripts/python.exe tests/verify_saturation.py --seq-len <N>
```

- The score is `CANDIDATE SCORE` = full + prefill + decode saturation time measured inside `run_saturation_benchmark` (`tensor_graphs_cpp/tests/saturation.hpp`). Dumping and the equivalence check are not timed.
- The buckets come from `runSaturationBenchmark`: full `[0, N)`; then for `t = 1..N`, prefill dirty rows `[0, t)` and decode dirty row `t-1`, each needing the same rows.
- **Where the gold comes from:** the accepted implementation's build at the N being evaluated. `verify_saturation.py` refuses to verify when the gold for that N is missing or incomplete, and it never regenerates gold itself. Gold metadata records the source tree (`GOLD SOURCE`) and every verdict prints it next to `CANDIDATE SOURCE`. A verdict whose two fingerprints match only counts as a timing run of the accepted implementation, never as an equivalence check of a candidate.
- **Gold workflow:**
  1. When the accepted implementation passes at the current N (under 30 s), and *before* starting the next optimization, build it and run `--generate-gold --seq-len <next N>`. Gold for a new N is never generated after a candidate change.
  2. Record the gold's `GOLD SOURCE` fingerprint and N in `progress.md`.
  3. Only then implement the candidate, rebuild, and run `--seq-len <next N>` plus the fixed sizes in the gate. Each N keeps its own gold in `tests/data/gold_saturation/seq<N>/`, so the fixed sizes only need gold generated once per accepted implementation.
- **Intended saturation changes:** a change that alters saturation output on purpose (a new sound rule, a bug fix) needs the user's approval, gold regenerated from the changed build, and a note in `progress.md`, as in the 2026-10-17 regeneration.
- **Fixed:** `tests/verify_saturation.py` (equivalence check and scoring), the benchmark graph and buckets in `tensor_graphs_cpp/tests/saturation.hpp`, and the gold files once generated for a run.
- **Agent-editable:** saturation (`Planner::saturate`, `saturateBucket`), rewrite rules, the e-graph data structures, parallelism, memory management.

## Result lines

| Line | Meaning |
|---|---|
| `Family vertices: reference=..., candidate=...` | Size of the compared bucket family |
| `VERDICT: PASS (EGraph Equivalent)` / `FAIL` | Exact equivalence of every bucket's pruned e-graph |
| `CANDIDATE SCORE: <ms> (Full: ..., Prefill + Decode: ...)` | Timed saturation |
| `GOLD SOURCE:` / `CANDIDATE SOURCE:` | Revision and uncommitted native diff hash of the build that made the gold, and of the candidate |

A run is **failed** if it crashes, prints `FAIL`, is refused for missing or mismatched gold, or its `GOLD SOURCE` is not the accepted implementation.

## Acceptance gate

1. **Equivalence:** `PASS` at `--seq-len 4`, `--seq-len 16` and the evaluated N, each against gold generated from the accepted build (matching `GOLD SOURCE`).
2. **Primary improvement:** the new N\* passes under 30 s in three runs with identical source and build.
3. **No regression:** the median scores at `--seq-len 4` and `--seq-len 16` are no worse than the accepted implementation's by more than its own run-to-run spread.

## Target-specific rules

- Never prune sound saturation alternatives for speed. Speed up or parallelize instead. Equivalence to the gold is the check that enforces this.
- Changes that only alter the dump format or the equivalence check are not improvements.

## Starting points

- "Potential paths to try next" at the end of `progress.md`: parallel buckets, a base-graph delta sweep filter, an undo log instead of copying the e-graph per bucket, and in-memory fingerprints instead of text dumps.
