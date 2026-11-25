# Primal-dual gap evaluation

The timed Gemma evaluation scores how quickly the search finds good plans **and** proves bounds, using the primal-dual integral (Berthold, "Measuring the impact of primal heuristics", Oper. Res. Letters 2013).

```powershell
.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600
```

`--timeout` sets `TG_SEARCH_GAP=1` (alongside `TG_SEARCH_VOLUME=1`) for the worker. The supervisor prints, after the volume and deadline lines:

```
[Program gap] primal-dual-integral=<s> normalized=<0..1> final-gap=<0..1> final-incumbent=<cost> final-bound=<cost> first-incumbent-elapsed=<s> samples=<n>
```

Lower `normalized` is better. 1.0 means no incumbent within the budget.

## Definitions

- **Incumbent (primal bound) `UB(t)`**: the best feasible plan cost found so far; +inf before the first plan.
- **Certified bound (dual bound) `LB(t)`**: the smallest stored node lower bound over open search nodes whose bound is below the incumbent, or the incumbent once no such node remains. Each node's bound is the engine's weighted sum of per-bucket lower bounds after propagation. Every unexplored assignment that could still improve lies in an open node, so `LB(t)` is a lower bound on the optimum if node bounds are valid.
- **Gap** `γ(UB, LB)`:
  - 1 if there is no incumbent, or `UB` and `LB` have opposite signs;
  - 0 if `LB ≥ UB` (or both are 0);
  - otherwise `|UB − LB| / max(|UB|, |LB|)`.
- **Primal-dual integral** `P(T) = ∫₀ᵀ γ(t) dt`, with `t` measured from program evaluation start (`TG_PROGRAM_DEADLINE − timeout`), so the 1.0 gap during process startup, model loading and preprocessing is included. `normalized = P(T)/T`.

## Sampling

The engine logs `[SearchEngine gap] epoch=<unix seconds> event=<root|sample|incumbent|deadline|final> iterations=<n> incumbent=<cost|inf> bound=<cost|inf>`:
- at the root after propagation;
- at iteration boundaries at most every `gap_sample_interval_seconds` (0.25 s; scanning the open queue is O(queue));
- immediately at every new incumbent;
- at the deadline and when the search ends.

`TG_SEARCH_GAP_INTERVAL=<seconds>` overrides the periodic interval for diagnostics only (a huge value keeps just the root, incumbent, deadline and final samples). Evaluations use the default.

`tests/search_gap.py` holds each sample until the next one. Incumbents only decrease and certified bounds only increase (children's bounds are at least their parent's), so this step function never under-estimates the gap. The supervisor rejects samples that violate these monotonicity rules (beyond print rounding), a bound above the incumbent, or a run with no samples. Samples after the budget are ignored.

## Validity of the bound

A wrong lower bound would make the score look better while being unsound. The planner oracle (`tests/test_planner_oracle.py`) samples the certified bound at every iteration of exhaustive native search on each small case. It requires both `root_lower_bound` and `max_certified_bound` to be at most the independently enumerated optimum. Any change to lower-bound propagators must keep this passing, and should add an adversarial case for its new inequality.

## Tests

```powershell
.venv/Scripts/python.exe -m tests.test_search_gap
```

checks the gap function, integration, budget clipping, closing at exhaustion, the no-incumbent case and the validation failures.
