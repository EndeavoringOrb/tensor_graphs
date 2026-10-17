# Search improvements record

Task: follow `docs/agents/targets/search_volume/target.md`; goal `remaining-domain-bits <= total-domain-bits / 2`
(i.e. `eliminated-bits >= total-domain-bits / 2`) within the fixed budget.

## Fixed evaluation contract
- Command (cwd `C:\Users\aaron\CODING\tensor_graphs`): `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Build: `.venv/Scripts/python.exe build.py --targets bindings --log-level INFO`
- Hardware: Snapdragon X Elite X1E78100 (Windows arm64), 31.5 GB RAM. Repeats: 1 baseline + 2 confirmations for promising candidates.
- Base revision: `d6397868` (branch `faster-search`).
- Correctness gates: `tests.test_planner_oracle` (full suite) after every native change.
- Overshoot limit: 50 ms originally; **changed by the user to 1 second** (mid-session, after R013). Edited `tests/test_gemma_output.py`, `docs/core/search_volume.md`, `docs/agents/targets/search_volume/target.md` accordingly. R002/R005/R010 also satisfy the old limit; R013 (0.1167 s) is valid under the new one.

## Runs
Target for all decode runs = total-domain-bits / 2 = 2425756.58187016 (total 4851513.1637403192).

### R001 build (baseline source) - `logs/search_improvements/001_build_baseline.log`
Command: build command above. Exit 0, "Target up-to-date". Source: HEAD d6397868, no candidate changes.

### R002 BASELINE decode - `logs/search_improvements/002_baseline_decode.log`
- Start/end: see `002_baseline_decode.start` / `.end` (end 2026-10-11T17:23:31-04:00). Exit 0. Source: HEAD d6397868, clean (only docs staged).
- Program elapsed 600.266 s, budget 600.0, deadline overshoot 0.0407553 s (<50 ms, accepted). Final best incumbent seen in progress: 75.009705.
- Oracle: not applicable (no source change).
- Volume report:
  `[SearchEngine volume] iterations=121320 elapsed=585.83109569999999 total-domain-bits=4851513.1637403192 remaining-domain-bits=4848823.0628079604 log2-remaining-fraction=-2690.1009323587641 eliminated-bits=2690.1009323587641 remaining-nodes=111868 exhausted=false incomplete-branches=35672`
- Half-total target: 2425756.58187016. Remaining/total ratio: 0.99944550 . Target met: **NO** (eliminated 0.055% of the needed bits).

### R003 build H1 - `logs/search_improvements/003_build_H1.log` (exit 0, 52 s). Patch: `logs/search_improvements/H1_conditional_critical_path.patch`
### R004 oracle H1 - `logs/search_improvements/004_oracle_H1.log`: exit 0, 14/14 cases passed.
### R005 H1 decode - `logs/search_improvements/005_H1_decode.log` (start/end files alongside; end 2026-10-11T17:35:09-04:00), exit 0
- Program elapsed 600.235 s, overshoot 0.0259662 s (accepted). Last progress line: iterations=122000 queue=112359 lower-bound-prunes=0 best=152.511688.
- `[SearchEngine volume] iterations=122678 elapsed=585.94394929999999 total-domain-bits=4851513.1637403192 remaining-domain-bits=4848823.0628079604 log2-remaining-fraction=-2690.1009323587641 eliminated-bits=2690.1009323587641 remaining-nodes=113039 exhausted=false incomplete-branches=36202`
- Target 2425756.58187016; ratio 0.99944550; met: **NO**.

### R006 diagnostic build + R007 diagnostic decode (90 s, NOT an evaluation) - `006_build_diag.log`, `007_diag_decode_90s.log`, patch `D1_rootdiag.patch`
Root-state bits per var type (after root propagation): CACHED 1842 bits (1842 vars, 0 fixed); SELECTED 177282 (128501 vars, 32 fixed); START 1.65519e6 (128501 vars, 32 fixed); OFFSET 3.01584e6 (124489 vars, 984 fixed). Sum unfixed OFFSET size_pages=3.49721e9, max 10485760. => OFFSET 62%, START 34% of the bits; a first-fit size bound on offsets is useless (page sum >> 2^26 domain). Diagnostic code reverted.
### R008 build H3 `008_build_H3.log`, R009 oracle H3 `009_oracle_H3.log` (14/14 passed), patch `H3_eager_right_propagation.patch`
### R010 H3 decode - `logs/search_improvements/010_H3_decode.log` (+ .start/.end), exit 0
- Program elapsed 600.281 s, overshoot 0.0253444 s (accepted). Last progress: iterations=118000 queue=107778 right-eager-prunes=768 lower-bound-prunes=0 best=75.009705.
- `[SearchEngine volume] iterations=118109 elapsed=585.41836339999998 total-domain-bits=4851513.1637403192 remaining-domain-bits=4848704.2230967265 log2-remaining-fraction=-2808.9406435927376 eliminated-bits=2808.9406435927376 remaining-nodes=107889 exhausted=false incomplete-branches=35672`
- Target 2425756.58187016; ratio 0.99942108; met: **NO**.

## Hypotheses
### H1 (REJECTED): conditional critical path through optional children
- Before: `CriticalPathPropagator` returns 0 for a class whose selection domain contains 0, so optional children vanish from the bound (baseline R002).
- Change: ignore the zero value when computing the exec bound of non-root classes (cost(alt)+max child execLB over nonzero alternatives). Sound: choosing an alternative forces all its children to run. Patch: `H1_conditional_critical_path.patch`.
- After: R005. Eliminated bits identical to baseline to all digits (2690.1009323587641): all elimination comes from root propagation; the search never closes a box (lower-bound-prunes=0). Oracle passes. No improvement -> reverted.
- Insight: START (~13 bits x 7559/bucket) and OFFSET (~25 bits x 7323/bucket) hold ~98% of the 4.85M bits. Lazy right siblings carry almost the whole parent volume; closing or reducing these large boxes and root pruning are candidate ways to improve the metric. Dive phase (deepest first until an incumbent) leaves a ~110k-node queue untouched.

### H2 (UNTESTED, patch saved): mandatory minimum engine workload
`H2_mandatory_min_workload.patch`: EngineWorkloadPropagator charges each engine the min over remaining alternatives of a mandatory class (0 excluded). Sound: a mandatory class runs exactly one alternative; makespan >= total unary-engine occupancy. Not yet built/evaluated (lower-bound-prunes were 0 in every run so far, so pruning potential is unproven).

### H3 (small gain, under evaluation): eager propagation of right child at creation
- Before: lazy right children carry the parent's propagated box and bound (R002/R005); popped boxes only get their real volume at restore.
- Change: after choosing a branch, set right delta, run propagators, record bound+volume, backtrack, then dive left. Prune right child if contradiction or LB >= incumbent. Sound: standard child propagation; pruned only when infeasible or cannot improve. Patch `H3_eager_right_propagation.patch`.
- After: R010: eliminated 2808.94 bits vs baseline 2690.10 (+118.8, +4.4%). Oracle passes. Still 0.058% of target. Needs confirmations before retaining.
- Analysis: remaining bits ~ log2(sum of frontier box volumes) ~ the largest box. The measured initial DFS dive leaves shallow right siblings unresolved. Meeting the target requires the combined unresolved volume to fall below the half-total-bit threshold; reductions need not be distributed equally across all 17 buckets.

### R011 build H4 `011_build_H4.log`, R012 oracle H4 `012_oracle_H4.log` (14/14 passed), patches `H4_largest_box_first.patch` (search_node.hpp only) and `H3H4_combined.patch`
### R013 H3+H4 decode - `logs/search_improvements/013_H4_decode.log` (+ .start/.end), exit 0 - valid under the 1 s limit (failed the original 50 ms limit)
- Program elapsed 600.375 s, overshoot **0.116713 s** (> old 50 ms limit, < new 1 s limit set by the user).
- `[SearchEngine volume] iterations=115802 elapsed=585.45538729999998 total-domain-bits=4851513.1637403192 remaining-domain-bits=4848292.1309524812 log2-remaining-fraction=-3221.0327878380194 eliminated-bits=3221.0327878380194 remaining-nodes=105582 exhausted=false incomplete-branches=35672`
- Target 2425756.58187016; ratio 0.99933599; met: **NO**.
- Key observation: progress line at iteration 114000 (t=357.6 s) still shows `depth=113999 best=inf`; the first incumbent (75.0097) appears at iteration 114822 (~380 s). The FIRST DIVE (one START/OFFSET/SELECTED variable per iteration, ~115k variables) consumes ~63% of the budget; before it ends no box is ever closed.

### H4 (REJECTED): largest-volume-first pop order after an incumbent exists (stacked on H3)
- Before: optimization-phase order = lowest priority (LB+1 for right, LB-0.01*depth for left), then deeper.
- Change: compare log2_volume first (larger first). Rationale: remaining bits are dominated by the largest frontier boxes, so prioritizing their resolution may improve the metric. Order does not affect completeness.
- After: R013: eliminated 3221.03 bits (vs H3 2808.94, baseline 2690.10) but only ~220 s remained after the dive, and restores replaying deep paths caused 0.1167 s overshoot (acceptable under the 1 s limit). Gain +20% on a negligible base; source reverted pending stronger ideas (patch saved).

## Interim conclusion (2026-10-11)
All variants eliminate 0.055-0.066% of the 4,851,513 bits (target 2,425,757). Bound improvements (H1) are irrelevant because no box is ever closed during the first dive. Gap is ~750x in eliminated bits.

### H5 (REJECTED): H2 mandatory-min-workload bound stacked on H3+H4
- Builds `015_build_H5.log`, oracle `016_oracle_H5.log` (14/14 passed), patch `H5_H2_on_H3H4.patch` (H3+H4+H2).
- R017 decode `017_H5_decode.log` (+ .start/.end), exit 0, elapsed 600.437 s, overshoot 0.131208 s (valid under 1 s limit).
- `[SearchEngine volume] iterations=115711 elapsed=585.64563350000003 total-domain-bits=4851513.1637403192 remaining-domain-bits=4848311.609205462 log2-remaining-fraction=-3201.5545348571613 eliminated-bits=3201.5545348571613 remaining-nodes=105491 exhausted=false incomplete-branches=35672`
- Target 2425756.58187016; ratio 0.99933996; met: **NO**. Same incumbent (75.0097 at iteration 114822) and lower-bound-prunes=0: the bound never matters because the first dive ends at ~380-410 s. Rejected; source reverted.

### R019 DIAGNOSTIC (not an evaluation): clean baseline at --timeout 1800 - `019_diag_baseline_1800s.log` (+ .start/.end)
COMPLETED (exit 0, elapsed 1800.204 s, overshoot 0.00687003 s). Purpose: see how eliminated-bits evolves after the first dive with 3x budget. Build `018_build_clean.log` (clean source, only tests/docs changed for the 1 s overshoot limit).

- `[SearchEngine volume] iterations=139304 elapsed=1785.2760375 total-domain-bits=4851513.1637403192 remaining-domain-bits=4848823.0628079604 log2-remaining-fraction=-2690.1009323587641 eliminated-bits=2690.1009323587641 remaining-nodes=129852 exhausted=false incomplete-branches=35672`
- Target 2425756.58187016 (for this diagnostic's own total); ratio 0.99944550; met: **NO**. Eliminated bits are BIT-IDENTICAL to the 600 s baseline (R002) despite 3x budget and +18k iterations: baseline DFS never shrinks the frontier volume after root propagation; late rate fell to ~79 it/s.

## Session 2 (2026-10-11, resumed from the session 1 checkpoint)
New analysis: the metric is dominated by the largest frontier box (log2 of a sum). Right siblings in the selection phase are near-root boxes. They vanish from the metric only when their stored LB >= incumbent (evaluateSearchVolume filters them, even without popping). Unselected classes already get canonical START/OFFSET (`UnselectedStartOffsetPropagator`), and ~62% of classes per bucket end up unselected. So once all right siblings above some depth close, the largest box is the left-path box at that depth, which falls below half the bits after roughly 9-10 buckets of selection decisions. The critical quantity is therefore the root-LB vs incumbent gap.

### R020 build diagnostic D2 (LB gap) - `020_build_diag_lbgap.log` (exit 0). Patch `D2_lbgap_diag.patch` (logs per-bucket cp/ew/sr bounds at root and at each new incumbent, plus engine work). NOT an evaluation candidate.
### R021 DIAGNOSTIC decode 600 s with D2 - `021_diag_lbgap_decode.log` (+ .start/.end). COMPLETED, exit 0, end 2026-10-11T19:15:00-04:00, overshoot 0.042464 s.
- `[SearchEngine volume] iterations=121235 elapsed=585.911024 total-domain-bits=4851513.1637403192 remaining-domain-bits=4848823.0628079604 log2-remaining-fraction=-2690.1009323587641 eliminated-bits=2690.1009323587641 remaining-nodes=111783 exhausted=false incomplete-branches=35672`
- Target 2425756.58187016; ratio 0.99944550; met: **NO** (diagnostic, behaviorally identical to baseline).
- **Root LB per bucket = 0.52775 (critical path only; ew=0, sr=0); total root LB 0.52775 vs incumbent 75.0097 (a 142x gap).**
- Weights: buckets 0-15 = 0.0625 each, bucket 16 = 0 (feasibility only). Buckets 0-15 differ only by decode token position (input slice i..i+1, a different KV-cache write region), so they are not exact copies.
- At the first leaf (all variables fixed), buckets 0-15 each: makespan 75.0097, cp 62.0204, ew 41.2405 (IGPU 41.2405, CPU 34.0542), sr 75.0097. Bucket 16: makespan 270.829, cp 83.52, ew 203.83. For this measured DAG, cp/ew remain below the incumbent and only the start-run bound reaches its makespan. This does not establish bounds for other selections or exclude stronger bound formulations.
- Observation: no selection-phase right siblings were closed by bound in the measured trace; see R023. This is a limitation observed in these runs, not proof that such boxes can never close.

### R022 build diagnostic D3 - `022_build_diag_divetrace.log` (exit 0). Patch `D3_dive_trace_H3_D2.patch` (H3 eager right propagation + D2 + a dive trace recording, for each dive branch before the first incumbent, the node depth, box bits, the eagerly propagated right child's LB/bits, and contradiction status; summarized at the deadline). NOT an evaluation candidate.
### R023 DIAGNOSTIC decode 600 s with D3 - `023_diag_divetrace_decode.log` (+ .start/.end). COMPLETED, exit 0, end 2026-10-11T19:29:06-04:00, overshoot 0.0343776 s.
- `[SearchEngine volume] iterations=118446 elapsed=585.74621279999997 total-domain-bits=4851513.1637403192 remaining-domain-bits=4848704.2230967265 log2-remaining-fraction=-2808.9406435927376 eliminated-bits=2808.9406435927376 remaining-nodes=108226 exhausted=false incomplete-branches=35672`
- Target 2425756.58187016; ratio 0.99942108; met: **NO** (diagnostic; same search as H3/R010, reproduces R010's 2808.94 bits exactly).
- Dive box bits by depth (full table in log, `[DIAG dive]`): depth 0: 4850156.66; 10000: 3659426.74; 20000: 2634128.42; **22153: first depth <= half (2425756.58)**; 28000 (end of selection phase): 1861505.94; 60000: 1197008.34; 114000: 25796.34. Selection decisions remove ~109 bits each on average (deselected classes get canonical START/OFFSET). Schedule/offset decisions remove ~30 bits each. Largest single-decision drop on the path: 8384.02 bits.
- `[DIAG dive summary]`: 22153 right siblings sit above D. 580 were contradiction-pruned, **0 closed by bound**, and 21572 are open with bits > half. Eagerly propagated right-sibling LBs rise roughly linearly from 0.53 (depth 0) to 52.5 (depth 22000) and 62.02 (end of selection), always below the incumbent 75.0097. Trace-based extrapolations for extra iterations: sum over siblings of ceil((bits-half)/max-drop) = **2,947,854**, and sum(D-depth) = 239,493,248. These use observed drops/depths, ignore possible new closures and propagation effects, and are not proven lower bounds. Budget: ~118k-122k iterations in 600 s.
### R024 build clean source - `024_build_clean_session2.log` (exit 0). Source restored to HEAD; extension matches clean source.

### R025 build D4 / R026 DIAGNOSTIC (150 s, not an evaluation) - `025_build_diag_memfit.log`, `026_diag_memfit_decode150.log`, patch `D4_memfit_diag.patch`
Per bucket (0-15 identical, 16 nearly so): CPP n=4045 (unfixed 3987, viewonly 1047, cacheable 893), sum of all CPP offset-class sizes 204,103,365 pages; OPENCL n=3278 (cacheable 677), sum 1,631,336 pages. Caps: CPP 31,514,855,301 B = 492,419,614 pages (64 B pages, 56 preallocated pages); OPENCL 2 GiB = 524,288 pages (4 KiB pages). So **every CPP allocation fits a private range at once**, while OPENCL does not (cacheable OPENCL alone is 1.58M pages).
### R027 build D5 / R028 DIAGNOSTIC (150 s) - `027_build_diag_offcat.log`, `028_diag_offcat_decode150.log`, patch `D5_offset_category_diag.patch`
Root OFFSET bits by category (after root propagation, all buckets): CPP plain 1,005,322.77 (34,816 vars), CPP view-only 513,454.93 (17,782), CPP cacheable 438,351.93 (15,181; 893 distinct bases, K = 83,992,851 pages, 1 class per base per bucket), CPP mixed view/non-view 0; OPENCL plain 840,120.60 (44,217), OPENCL cacheable 218,593.28 (11,509; K = 1,577,470 pages).

### H6 (candidate): static private memory layout (dominance on cost-irrelevant offsets)
- Before: every CPP offset spans ~28.9 bits (`[0, cap-size]`), and they hold 1.96M of the 4.85M bits. Baseline R002.
- Change: new `StaticPrivateLayoutPropagator` (`propagators/static_private_layout.hpp`), registered before `UnselectedStartOffsetPropagator`. It runs once during root propagation. Per memory space it plans: preallocated region unchanged, then one slot per cacheable base (shared by every bucket, base-id order), then one slot per other non-view e-class per bucket (buckets reuse the same range). It fixes those offsets plus view-only classes whose alternatives all resolve to one address. It applies to a space only if the whole layout fits the capacity, every slot is inside its variable's current domain, every candidate base address is inside each view's domain, and no cacheable class has a view alternative and no class mixes view/non-view alternatives; otherwise the space is skipped (logged). `UnselectedStartOffsetPropagator` no longer overwrites an already fixed offset of an unselected class (that coordinate is irrelevant).
- Soundness (dominance): offsets never enter the objective (`evaluateMakespan` uses only selections and STARTs). Take any feasible plan x. Move every allocation to its private slot; a selected view takes its base's slot; cached copies all take their base's slot. (1) No two distinct allocations in a bucket overlap, so no-overlap, persistence, reuse-after-reader and write-after-read constraints hold vacuously, and start constraints derived from overlaps disappear; (2) allocation pairs are only compared within one bucket (`propagators.md`, MemoryNoOverlap), so reusing the per-bucket range across buckets is fine; (3) cached copies share one address in every bucket, and cache slots are disjoint from each other and from all temporaries; (4) every slot fits the capacity and lies in the variable's domain (checked); (5) selections and STARTs are unchanged, so the cost is identical. Hence an optimal plan survives in the restricted box. OPENCL is skipped by the capacity check (does not fit).
- Oracle: added cases `15_static_layout_views` (fires), `16_static_layout_shared_cache` (fires), `17_static_layout_tight_memory` (must skip: only reuse fits), `18_static_layout_multi_base_view` (fires; multi-base views stay free). Verified via worker logs that the layout fires/skips as intended.
- Runs: R029 build (first version had `interestedVarTypes()=0` and was never invoked, because root propagation visits dirty variables by type, not with `kInvalidVarId`); R030 oracle 14/14 (rule inactive, uninformative). R031 build v2 (subscribes to OFFSET, applies once on first call). R032 oracle: 17/18, the failure was my first case-18 draft exceeding the 250,000-step enumeration limit (shrunk; not a candidate failure). **R033 oracle 18/18 passed** (`033_oracle_H6v2.log`). Patches: `H6_static_private_layout_v2.patch` (native), `H6_static_private_layout_v3_with_cases.patch` (native + cases).
### R034 H6 decode - `034_H6_decode.log` (+ .start/.end). Source: HEAD d6397868 + `H6_static_private_layout_v3_with_cases.patch`. Exit 0, end 2026-10-11T20:00:01-04:00, overshoot 0.0118251 s.
- Log: `[StaticPrivateLayout] space=MemSpace(idx=1, type=CPP) fixed 67302 of 67302 planned offsets`; `space=MemSpace(idx=1, type=OPENCL) skipped: private layout exceeds memory capacity`.
- `[SearchEngine volume] iterations=80321 elapsed=585.78931090000003 total-domain-bits=4851513.1637403192 remaining-domain-bits=2907262.2387936418 log2-remaining-fraction=-1944250.9249466774 eliminated-bits=1944250.9249466774 remaining-nodes=80275 exhausted=false incomplete-branches=5312`
- Target 2425756.58187016; ratio 0.59924471; met: **NO** (short by 481,505.66 bits). 723x the baseline's eliminated bits.
- Side effect: the dive is slower (~138 it/s vs ~320 it/s) and found no incumbent within the budget (the evaluation does not need one). Likely cause: memory propagators now see ~67k fixed CPP allocations. Not investigated.
### R035 H6 confirmation 1 - `035_H6_confirm1_decode.log` (+ .start/.end). Exit 0, end 2026-10-11T20:11:54-04:00, overshoot 0.0210478 s.
- `[SearchEngine volume] iterations=80813 elapsed=585.79295360000003 total-domain-bits=4851513.1637403192 remaining-domain-bits=2907262.2387936418 log2-remaining-fraction=-1944250.9249466774 eliminated-bits=1944250.9249466774 remaining-nodes=80767 exhausted=false incomplete-branches=5312`
- Target 2425756.58187016; ratio 0.59924471; met: **NO**.
### R036 build D6 / R037 DIAGNOSTIC (150 s) - `036_build_diag_alts.log`, `037_diag_alts_decode150.log`, patch `D6_alternatives_diag_on_H6.patch`
Bucket 0 selection domains: 3336 classes with 1 alternative, 4221 with >= 2. Patterns: same engines + different children 2597, compute-vs-CACHE (children subset) 1570, different engines (all with cache) 274, same-engine subset children without cache 54. So subset-children dominance (`p3` A4) has almost no scope; most alternatives differ in children.
### R038 build D7 / R039 DIAGNOSTIC (150 s) - `038_build_diag_startwin.log`, `039_diag_startwin_decode150.log`, patch `D7_start_window_diag_on_H6.patch`
Precedence START window `[D_min(c), max_start - dist(root,c)]` (min-over-alternatives chain below c, shortest path from root): saves only 810.5 bits per bucket (~14k total). Root min-chain is 1434, but every class is within 66 steps of the root (fused shortcuts). Not pursued.
### R040 build H6 (diagnostics reverted; source == `H6_static_private_layout_v3_with_cases.patch`) - `040_build_H6_again.log`.
### R041 H6 confirmation 2 - `041_H6_confirm2_decode.log` (+ .start/.end). Exit 0, end 2026-10-11T20:31:44-04:00, overshoot 0.00422716 s.
- `[SearchEngine volume] iterations=80131 elapsed=585.28475800000001 total-domain-bits=4851513.1637403192 remaining-domain-bits=2907262.2387936418 log2-remaining-fraction=-1944250.9249466774 eliminated-bits=1944250.9249466774 remaining-nodes=80085 exhausted=false incomplete-branches=5312`
- Target 2425756.58187016; ratio 0.59924471; met: **NO**.
- **Decision: H6 RETAINED** (oracle 18/18 in R033, three identical 600 s measurements R034/R035/R041). Left uncommitted in the working tree, as with the other session changes.

### Remaining gap after H6 (481,505.66 bits): measured reservoirs and candidate limitations
Root box after H6 propagation: 2,907,262 bits = START 1.655M + OPENCL offsets 1.059M (plain 0.840M, cacheable 0.219M) + SELECTED 0.177M + CACHED 0.002M.
- OPENCL offsets: a static layout needs cacheable bases (K = 1,577,470 pages) + plain (53,866 pages per bucket) <= 524,288 pages, which fails. A partial layout (plain only, at the top) is unsound in general. The cache budget equals the full capacity (`CacheBudgetPropagator`), so a feasible or optimal plan may fill nearly all of OPENCL with cached buffers, leaving no room for private plain slots. Offsets cannot be represented as strided sets (Domain = range or 32-value mask), so gcd/normal-pattern reductions do not shrink the counted volume either.
- START: dense-rank compression is sound but bounded by log2(7559/2900) = 1.38 bits/var (<= ~177k total, since every plan selects >= 2900 classes in buckets 0-15). The precedence window adds only ~14k (R039). Combined well below 481k.
- SELECTED: 177k in total; root dominance among alternatives has little scope (R037).
- Search-time closure: shallow right siblings retained large boxes in the tested traces, with root LB 0.53 vs incumbent 75.0 before H6. No bound closures were observed there. The cost of resolving these boxes under other bounds or branching strategies remains unmeasured.
- R042 build D8 / R043 DIAGNOSTIC (150 s) - `042_build_diag_kmax.log`, `043_diag_kmax_decode150.log`, patch `D8_kmax_greedy_diag_on_H6.patch`: a greedy plan (fewest-children alternative per class) selects 6958 of 7559 classes. The proposed uniform dense-rank cap therefore appears to offer only ~0.12 bits/var (~15k total), assuming this plan must remain representable. This is not a limit on all START reductions or on restrictions justified by optimal-plan dominance. R044 rebuilt H6-only source (`044_build_H6_restore.log`).
- Considered but not evaluated: (a) an OPENCL "union" layout where mutually exclusive classes share slots. R043's 6958 selected classes suggest limited exclusivity, but class count alone does not establish selected OPENCL allocation sizes or rule out useful sharing. (b) Early balanced splits: the pre-H6 trace's largest observed single-decision drop was ~8.4k bits. Extrapolations to ~2^68 frontier boxes assume independent balanced decisions across buckets and are not established complexity bounds; alternative branches and propagation may behave differently.
With H6 the eliminated bits are 80.1% of the target. OPENCL layout/dominance and stronger propagation are possible directions for the remaining gap; the experiments do not establish an exhaustive list of ways to reach the target.

## Observed limitations (session 2)
1. **Frontier volume.** Remaining bits are log2(sum of frontier box volumes), dominated by large boxes. The measured pre-H6 dive first crossed half-total bits at depth 22153 (R023). That depth is specific to this path and brancher.
2. **Measured closures.** The trace recorded 580 contradiction-pruned shallow siblings, 0 closed by bound, and 21572 open siblings above half-total bits. Root LB was 0.528 against incumbent 75.01 (R021). These bounds were weak on the measured path; other selections, bounds, or dominance rules remain possible.
3. **Trace-based extrapolations.** R023 estimated 2.9M extra iterations using the largest observed drop and 2.4e8 using depth gaps. These assume observed reductions characterize future work; they are not proven lower bounds for other strategies. The earlier claim of at least 2^289 nodes even with a perfect brancher was unsupported and is withdrawn.
4. **Root reductions.** Simple size bounds and uniform rank caps appeared limited in diagnostics, but H6 subsequently eliminated 1.94M bits through layout dominance. Thus those estimates cannot serve as ceilings on all sound root reductions. Buckets write different KV-cache positions; any symmetry reduction would require a separate justification.
5. **Objective and encoding.** Offsets represented ~62% of initial bits while not directly entering cost. H6 demonstrates that some such coordinates can be fixed by dominance without changing the evaluation contract. Other reductions and stronger lower bounds have not been ruled out.
Conclusion: no tested candidate reached `remaining <= total/2` within 600 s. H6 made substantial bit progress but regressed incumbent discovery. Target feasibility remains unresolved; these runs do not prove impossibility or establish that the evaluation contract must change.

## Resume checkpoint (current)
- State (2026-10-11, after R041): **H6 retained** in the working tree (uncommitted): `tensor_graphs_cpp/core/plan/propagators/static_private_layout.hpp` (new), registration in `propagator.hpp`, the fixed-offset guard in `unselected_start_offset.hpp`, oracle cases `tests/planner_oracle_cases/15..18_static_layout_*.json`, and docs in `docs/core/propagators.md`. Exact patch: `logs/search_improvements/H6_static_private_layout_v3_with_cases.patch`. Extension built from this source (R044). No running processes.
- Current best = H6: eliminated 1,944,250.9249466774 bits (ratio 0.59924471, target 2,425,756.58 not met; short by 481,505.66). Baseline R002: 2,690.10. Oracle 18/18 (R033).
- Older candidate patches (oracle-clean, not retained, all built on pre-H6 source): H1, H2, H3, H3+H4, H5. H3+H4 gained fewer than 600 bits before H6; its effect on top of H6 is untested.
- Next concrete steps if work resumes: (1) profile and repair H6's incumbent-discovery regression; reassess retention under the new multi-metric gate. (2) Investigate OPENCL layout/dominance, including whether some cached bases are cost-dominated or an optimal plan leaves room for private temporaries. (3) Explore stronger START/selection propagation or bounds; the uniform dense-rank estimate alone does not exclude these directions.
- Earlier limitations: see "Observed limitations (session 2)" and "Remaining gap after H6" above.

## Earlier analysis (pre-H6; revised to separate observations from extrapolation)
1. Root diagnostics: OFFSET 3.016e6 bits (62%), START 1.655e6 (34%), SELECTED 1.77e5 (4%). Simple size/rank proposals appeared to offer modest savings. These were proposal-specific estimates, not ceilings on sound reductions; H6 later achieved much larger savings.
2. Search observations: baseline R019 at 1800 s and R002 at 600 s both reported 2690.1009323587641 eliminated bits. H3+H4 gained about 531 bits over baseline. A linear extrapolation from that gain cannot predict performance of other algorithms or establish a minimum budget for the target.
3. Bound observations: H1/H2 produced no lower-bound prunes in tested runs. This does not make stronger lower bounds irrelevant or establish their maximum benefit.
Conclusion: the pre-H6 trials did not meet the target. Subsequent H6 results supersede their pessimistic root-reduction estimates. Faster incumbent discovery, bucket-decomposed search, root dominance, and stronger propagation remain research directions whose effectiveness must be evaluated rather than dismissed in advance.

## Session 3 (2026-10-12)
H6 historical retention is superseded: R034/R035/R041 fail the incumbent non-regression gate (no incumbent against R002 finite incumbent). H6 requires repair. Reuse R002 as permanent baseline; instrument native incumbent event epoch and deadline (observational only) and rerun baseline to resolve timing. Same hardware, default build flags, decode optimal timeout 600, no compiled cache, two confirmations. Current staged changes preserved in run patches. H7 hypothesis: replace quadratic global fixed-allocation scans and max-size incremental scans with exact interval queries. Disjoint private ranges should be cheap; pruning semantics unchanged. All overlapping pairs must still be checked, including views, optional nodes, persistent caches, sparse domains and restores. No new pruning rule.

### 045_build_timed_baseline
- In progress: 2026-10-12T11:50:20.588191-04:00
- Complete metadata: `logs/search_improvements/045_build_timed_baseline.json`; log: `logs/search_improvements/045_build_timed_baseline.log`; patch: `logs/search_improvements/045_build_timed_baseline.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings --no-lint --log-level INFO`

### 046_timed_baseline_decode
- In progress: 2026-10-12T11:51:19.102902-04:00
- Complete metadata: `logs/search_improvements/046_timed_baseline_decode.json`; log: `logs/search_improvements/046_timed_baseline_decode.log`; patch: `logs/search_improvements/046_timed_baseline_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- R045 reconciliation: build completed successfully (complete log ends BUILD SUCCESS/link). The experiment wrapper failed decoding Windows build output as UTF-8 after the build. Source/build succeeded; wrapper now uses replacement decoding, retaining original log bytes. R045 build verdict PASS; no benchmark metrics applicable.
- Checkpoint: R046 timed original baseline running (exec session 6923, log 046_timed_baseline_decode.log; supervisor buffers worker output). Working source has pending H7 atop H6, extension still timed baseline without H6; do not rebuild until R046 ends. H7 uses interval hull queries for global write-after-read, address sweeps for global fixed-start checks, and the existing interval treap for incremental start checks. No candidate evaluations yet.
- Soundness review found a separate H6 risk: a nested view whose child can alias multiple private slots is not checked against all those slots when resolveViewAddress returns false. Added case 19 with outer view addresses {0,1}, inner view alternatives p/q, q cheaper, and planned q slot 2. Original feasible optimum can relocate q to 1; H6 must skip the layout rather than discard it. This case will run before candidate retention. H7 memory scan does not change this rule.
- Completed: 2026-10-12T12:01:19.488680-04:00; exit 0; runner elapsed 600.375 s.
  `[SearchEngine volume] iterations=120937 elapsed=586.00172050000003 total-domain-bits=4851513.1637403192 remaining-domain-bits=4848823.0628079604 log2-remaining-fraction=-2690.1009323587641 eliminated-bits=2690.1009323587641 remaining-nodes=111485 exhausted=false incomplete-branches=35672`
  `[Program evaluation] elapsed=600.312000000 budget=600.0`
  `[Program deadline] overshoot=0.0730863`
  `[SearchEngine] New best cost: 75.0097 at iteration 114822 event-epoch=1791820509.3317065 program-deadline=1791820879.1528299`
- Target 2425756.5818701596; remaining/total 0.99944551300974216; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.

### 047_build_H7
- In progress: 2026-10-12T12:01:33.752810-04:00
- Complete metadata: `logs/search_improvements/047_build_H7.json`; log: `logs/search_improvements/047_build_H7.log`; patch: `logs/search_improvements/047_build_H7.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings --no-lint --log-level INFO`
- R046 baseline reproduced R002 volume exactly; first incumbent event elapsed = 1791820509.3317065 - 1791820879.1528299 + 600 = 230.17887664 s (evaluation timing origin). First cost logged 75.0097 (rounded); best deadline cost 75.009705. R002 event bracket search-only [209.25,222.85] s, setup approximately 14.435 s from deadline/volume elapsed; program bracket approximately [223.69,237.29] s. R046 supplies precise contemporary timing, but acceptance should conservatively beat the original bracket lower endpoint + 1 second where feasible. Original R002 remains permanent bit/cost anchor. H6 historical no-incumbent trials fail gate.
- Completed: 2026-10-12T12:02:24.980515-04:00; exit 0; runner elapsed 51.21899999980815 s.

### 048_oracle_H7
- In progress: 2026-10-12T12:03:59.274796-04:00
- Complete metadata: `logs/search_improvements/048_oracle_H7.json`; log: `logs/search_improvements/048_oracle_H7.log`; patch: `logs/search_improvements/048_oracle_H7.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T12:04:01.304604-04:00; exit 1; runner elapsed 2.030999999959022 s.
- R048 oracle: 18/19 passed. Case 19 demonstrates H6 loss of an optimal solution: native lower bound 5 exceeds reference optimum 3, witness q=1 and outer/inner view address 1. H7 scan changes pass the previous 18 cases but combined candidate is invalid until H6 repaired. H8 guard repair collects the union of ultimate private/preallocated addresses recursively through every view alternative, checks each against each view domain, and skips the entire space for an unknown/cyclic alias or any excluded address. This strengthens certificate prerequisites only; it cannot introduce pruning relative to H6. Rebuild and full oracle pending.

### 049_build_H7_H8
- In progress: 2026-10-12T12:04:45.340786-04:00
- Complete metadata: `logs/search_improvements/049_build_H7_H8.json`; log: `logs/search_improvements/049_build_H7_H8.log`; patch: `logs/search_improvements/049_build_H7_H8.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings --no-lint --log-level INFO`
- Completed: 2026-10-12T12:05:36.476652-04:00; exit 0; runner elapsed 51.14099999982864 s.

### 050_oracle_H7_H8
- In progress: 2026-10-12T12:05:45.694366-04:00
- Complete metadata: `logs/search_improvements/050_oracle_H7_H8.json`; log: `logs/search_improvements/050_oracle_H7_H8.log`; patch: `logs/search_improvements/050_oracle_H7_H8.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T12:05:47.533632-04:00; exit 0; runner elapsed 1.827999999979511 s.

### 051_H7_H8_decode
- In progress: 2026-10-12T12:05:51.945718-04:00
- Complete metadata: `logs/search_improvements/051_H7_H8_decode.json`; log: `logs/search_improvements/051_H7_H8_decode.log`; patch: `logs/search_improvements/051_H7_H8_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- R049 build PASS; R050 full oracle PASS 19/19. R051 fixed-budget H7+H8 evaluation now running (session 89788, log 051_H7_H8_decode.log); extension and native headers match saved R051 patch. H8 rationale: every possible ultimate base address must remain representable for every selected view along its alias chain; preserving only a single resolved address test is insufficient. The guard is conservative and leaves the original domains untouched if certificate unavailable. H9 prepared offline in logs/search_improvements/edit_h9.py only, not applied: cache physical overlap pairs by allocation revision (SELECTED/OFFSET/CACHED and undo), evaluate all temporal rules against current START domains. Do not build during R051.
- Completed: 2026-10-12T12:15:52.286172-04:00; exit 0; runner elapsed 600.3279999999795 s.
  `[SearchEngine volume] iterations=120940 elapsed=585.919668 total-domain-bits=4851513.1637403192 remaining-domain-bits=4848823.0628079604 log2-remaining-fraction=-2690.1009323587641 eliminated-bits=2690.1009323587641 remaining-nodes=111488 exhausted=false incomplete-branches=35672`
  `[Program evaluation] elapsed=600.250000000 budget=600.0`
  `[Program deadline] overshoot=0.0482612`
  `[SearchEngine] New best cost: 75.0097 at iteration 114822 event-epoch=1791821382.3369367 program-deadline=1791821752.0106587`
- Target 2425756.5818701596; remaining/total 0.99944551300974216; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- R051: H8 conservatively skips CPP (offset_0_7 uncertified alias), so bits identical to R002/R046, first incumbent 230.326278 s, first/best costs unchanged, overshoot 0.0482612. No meaningful improvement; H7/H8 retention pending certificate repair. H8 correctness repair required regardless of performance. Next diagnostic identifies actual preallocated/leaf alias behind cid 7 before relaxing any guard. H9/H9a performance scripts remain unapplied.

### 052_build_alias_diag
- In progress: 2026-10-12T12:17:26.580268-04:00
- Complete metadata: `logs/search_improvements/052_build_alias_diag.json`; log: `logs/search_improvements/052_build_alias_diag.log`; patch: `logs/search_improvements/052_build_alias_diag.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings --no-lint --log-level INFO`
- Completed: 2026-10-12T12:18:18.480966-04:00; exit 0; runner elapsed 51.90599999995902 s.

### 053_diag_alias_60s
- In progress: 2026-10-12T12:19:27.985509-04:00
- Complete metadata: `logs/search_improvements/053_diag_alias_60s.json`; log: `logs/search_improvements/053_diag_alias_60s.log`; patch: `logs/search_improvements/053_diag_alias_60s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 60`
- Completed: 2026-10-12T12:20:28.139443-04:00; exit 0; runner elapsed 60.15600000019185 s.
  `[SearchEngine volume] iterations=5649 elapsed=45.612599500000002 total-domain-bits=4851513.1637403192 remaining-domain-bits=4850156.3334197225 log2-remaining-fraction=-1356.8303205966949 eliminated-bits=1356.8303205966949 remaining-nodes=5650 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=60.094000000 budget=60.0`
  `[Program deadline] overshoot=0.0040772`
- Target 2425756.5818701596; remaining/total 0.99972032842645109; target met: False.
- Incumbent: no incumbent within budget; time and both costs unavailable (+infinity).
- R052 build and R053 diagnostic 60 s PASS (not comparable to 600 s evaluations). Uncertified cid 7 is FILL view with TWO children, not an unknown address: production ViewOffsetHelpers/getConfirmedViewBase aliases child zero and allows additional dependency children. H8 had unnecessarily required exactly one child. Corrected to nonempty children and recurse into child zero, matching native view semantics. This preserves the recursive union guard/case 19. No H9 performance changes yet.

### 054_build_H8_multi_input_view
- In progress: 2026-10-12T12:21:25.353649-04:00
- Complete metadata: `logs/search_improvements/054_build_H8_multi_input_view.json`; log: `logs/search_improvements/054_build_H8_multi_input_view.log`; patch: `logs/search_improvements/054_build_H8_multi_input_view.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings --no-lint --log-level INFO`
- Completed: 2026-10-12T12:22:17.150770-04:00; exit 0; runner elapsed 51.79700000002049 s.

### 055_oracle_H8_multi_input_view
- In progress: 2026-10-12T12:22:18.403269-04:00
- Complete metadata: `logs/search_improvements/055_oracle_H8_multi_input_view.json`; log: `logs/search_improvements/055_oracle_H8_multi_input_view.log`; patch: `logs/search_improvements/055_oracle_H8_multi_input_view.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T12:22:20.189859-04:00; exit 0; runner elapsed 1.7809999999590218 s.
- Oracle verdict: 19/19 cases passed.

### 056_H7_H8_decode
- In progress: 2026-10-12T12:22:25.676948-04:00
- Complete metadata: `logs/search_improvements/056_H7_H8_decode.json`; log: `logs/search_improvements/056_H7_H8_decode.log`; patch: `logs/search_improvements/056_H7_H8_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- H10 prepared but unapplied: necessary selection certificates are unions over incoming support paths (intersection of available literals, union of allowed values), conditional on a class being reachable through selected alternatives; add own positive selection literal. Truncating cyclic recursion or dropping excess literals only weakens certificates. A common variable with disjoint allowed masks proves two classes cannot both be active. Every same-bucket pair of copies must pass; at least one common bucket is required so two shared caches cannot coexist across calls. Cacheable bases may then share a private slot, whose size is the maximum member footprint. All simultaneously active distinct allocation groups remain disjoint, and all original view/domain/capacity checks still apply; selections, cache choices, STARTs and costs are unchanged. Cases 20 (exclusive caches) and 21 (uncached coexistence in another bucket despite global cache exclusion) prepared. Current native candidate R056 still H7/H8 only. Colored layout should be evaluated only after incumbent regression is repaired, unless used as a short diagnostic.
- Completed: 2026-10-12T12:32:26.138664-04:00; exit 0; runner elapsed 600.4529999999795 s.
  `[SearchEngine volume] iterations=90068 elapsed=585.91825570000003 total-domain-bits=4851513.1637403192 remaining-domain-bits=2906102.208436559 log2-remaining-fraction=-1945410.9553037602 eliminated-bits=1945410.9553037602 remaining-nodes=84605 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.407000000 budget=600.0`
  `[Program deadline] overshoot=0.0364728`
  `[SearchEngine] New best cost: 75.0097 at iteration 85093 event-epoch=1791822448.3538871 program-deadline=1791822745.7322259`
- Target 2425756.5818701596; remaining/total 0.5990094451678396; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 302.621661186 s, logged first cost 75.0097, logged best cost 75.0097.
- Oracle verdict: not applicable.
- R056 H7/H8: CPP layout active again, eliminated 1,945,410.9553037602 (+1,160.03 vs historical H6, +1,942,720.85 vs R002), first incumbent 302.621661186 s (+72.442785 vs R046), first/best costs 75.0097/75.009705 unchanged, oracle R055 PASS. Gate FAIL on first time; target not met. H7 scan improvements improved first dive substantially vs H6 no-incumbent runs, but not enough. H9 now applied: allocation revision increments on SELECTED/OFFSET/CACHED and undo; WriteAfterReadStart stores allocation descriptors and overlapping index pairs by revision, reads live START domains on every call; defers FixedOffsetStart structural rebuild on irrelevant START events. All pair predicates and reader rule unchanged, no initial domain/objective changes. Compared against original R002/R046 and pending H7/H8 R056; build/oracle/evaluation pending.

### 057_build_H9
- In progress: 2026-10-12T12:33:28.800251-04:00
- Complete metadata: `logs/search_improvements/057_build_H9.json`; log: `logs/search_improvements/057_build_H9.log`; patch: `logs/search_improvements/057_build_H9.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings --no-lint --log-level INFO`
- Completed: 2026-10-12T12:33:36.154462-04:00; exit 1; runner elapsed 7.343000000109896 s.
- Oracle verdict: not applicable.
- R057 build FAILED: allocation_revision is private in SearchState; added getAllocationRevision accessor and corrected caller. No evaluation ran with the failed build. Rebuild pending.

### 058_build_H9_accessor
- In progress: 2026-10-12T12:33:47.512004-04:00
- Complete metadata: `logs/search_improvements/058_build_H9_accessor.json`; log: `logs/search_improvements/058_build_H9_accessor.log`; patch: `logs/search_improvements/058_build_H9_accessor.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings --no-lint --log-level INFO`
- Completed: 2026-10-12T12:34:38.887927-04:00; exit 0; runner elapsed 51.375 s.
- Oracle verdict: not applicable.

### 059_oracle_H9
- In progress: 2026-10-12T12:35:08.800104-04:00
- Complete metadata: `logs/search_improvements/059_oracle_H9.json`; log: `logs/search_improvements/059_oracle_H9.log`; patch: `logs/search_improvements/059_oracle_H9.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T12:35:10.819440-04:00; exit 0; runner elapsed 2.0149999998975545 s.
- Oracle verdict: 21/21 cases passed.

### 060_H9_decode
- In progress: 2026-10-12T12:35:14.842869-04:00
- Complete metadata: `logs/search_improvements/060_H9_decode.json`; log: `logs/search_improvements/060_H9_decode.log`; patch: `logs/search_improvements/060_H9_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T12:45:15.313041-04:00; exit 0; runner elapsed 600.469000000041 s.
  `[SearchEngine volume] iterations=90527 elapsed=586.06826960000001 total-domain-bits=4851513.1637403192 remaining-domain-bits=2906102.208436559 log2-remaining-fraction=-1945410.9553037602 eliminated-bits=1945410.9553037602 remaining-nodes=85064 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.407000000 budget=600.0`
  `[Program deadline] overshoot=0.127546`
  `[SearchEngine] New best cost: 75.0097 at iteration 85093 event-epoch=1791823186.2435436 program-deadline=1791823514.8990054`
- Target 2425756.5818701596; remaining/total 0.5990094451678396; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 271.344538212 s, logged first cost 75.0097, logged best cost 75.0097.
- Oracle verdict: not applicable.
- R060 H9 oracle R059 PASS 21/21; same bits as R056 (1,945,410.9553037602), first incumbent 271.344538212 s (31.277123 s faster vs R056, 41.165662 s slower vs R046); costs unchanged; overshoot0.127546. Gate FAIL timing, target not met. H9 pending as part of regression repair, not accepted optimization. H11 now applied: FixedOffsetStart builds reader metadata only for an actual nontrivial overlap. Both-view and proven selected view/base alias pairs use the identical unconditional acceptance already present in resolvePair. Alias proof follows live fixed positive selections through child zero, up to the existing 32-step limit; optional/unknown/long chains fall through to full checks. Spatial-index invalidation unchanged. Global scans and write-after-read checks still build their normal metadata. This removes work only, no new pruning.

### 061_build_H11
- In progress: 2026-10-12T12:48:12.719979-04:00
- Complete metadata: `logs/search_improvements/061_build_H11.json`; log: `logs/search_improvements/061_build_H11.log`; patch: `logs/search_improvements/061_build_H11.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings --no-lint --log-level INFO`
- Completed: 2026-10-12T12:49:04.663049-04:00; exit 0; runner elapsed 51.95299999997951 s.
- Oracle verdict: not applicable.

### 062_oracle_H11
- In progress: 2026-10-12T12:49:27.137224-04:00
- Complete metadata: `logs/search_improvements/062_oracle_H11.json`; log: `logs/search_improvements/062_oracle_H11.log`; patch: `logs/search_improvements/062_oracle_H11.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T12:49:29.545929-04:00; exit 1; runner elapsed 2.405999999959022 s.
- Oracle verdict: 21/22 cases passed.
- R061 build PASS. R062 oracle 21/22: new case 22 exceeded reference enumeration budget, not a native failure. Reduced only the new case's schedule domains to its intended fixed order s,p,q,r,out (all alias-support alternatives and address choices retained), avoiding unnecessary rank enumeration. Existing evaluator and cases unchanged. Re-run full oracle.

### 063_oracle_H11_small_case
- In progress: 2026-10-12T12:50:09.463230-04:00
- Complete metadata: `logs/search_improvements/063_oracle_H11_small_case.json`; log: `logs/search_improvements/063_oracle_H11_small_case.log`; patch: `logs/search_improvements/063_oracle_H11_small_case.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T12:50:11.586012-04:00; exit 0; runner elapsed 2.125 s.
- Oracle verdict: 22/22 cases passed.

### 064_H11_decode
- In progress: 2026-10-12T12:50:16.080944-04:00
- Complete metadata: `logs/search_improvements/064_H11_decode.json`; log: `logs/search_improvements/064_H11_decode.log`; patch: `logs/search_improvements/064_H11_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`

- Completed: 2026-10-12T13:00:16.322919-04:00; exit 0; runner elapsed 600.25 s.
  `[SearchEngine volume] iterations=92067 elapsed=585.90102809999996 total-domain-bits=4851513.1637403192 remaining-domain-bits=2906102.208436559 log2-remaining-fraction=-1945410.9553037602 eliminated-bits=1945410.9553037602 remaining-nodes=86604 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.172000000 budget=600.0`
  `[Program deadline] overshoot=0.002316`
  `[SearchEngine] New best cost: 75.0097 at iteration 85093 event-epoch=1791824017.9769309 program-deadline=1791824416.1400912`
- Target 2425756.5818701596; remaining/total 0.5990094451678396; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 201.836839676 s, logged first cost 75.0097, logged best cost 75.0097.
- Oracle verdict: not applicable.

- R064 H11 PROMISING: eliminated 1,945,410.9553037602, remaining 2,906,102.208436559; first incumbent 201.836839676 s (28.342037 s faster than R046, 69.507699 s faster than R060). First/best costs unchanged75.0097/75.009705. Oracle R063 22/22 PASS. Program elapsed600.172, overshoot0.002316. Timing also beats R002 conservative lower bound: its search event alone was after209.25 s, while R064 program event <209.25 s, so original +1 s gate holds without inventing setup precision. Target still short480,345.63 bits. Begin two confirmations with identical native source/build. H13 linear-storage overlap ranges prepared offline only; H10 exclusive-slot layout likewise unapplied.

### 065_H11_confirm1_decode
- In progress: 2026-10-12T13:01:07.634283-04:00
- Complete metadata: `logs/search_improvements/065_H11_confirm1_decode.json`; log: `logs/search_improvements/065_H11_confirm1_decode.log`; patch: `logs/search_improvements/065_H11_confirm1_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T13:11:07.907056-04:00; exit 0; runner elapsed 600.2650000001304 s.
  `[SearchEngine volume] iterations=92098 elapsed=585.97919460000003 total-domain-bits=4851513.1637403192 remaining-domain-bits=2906102.208436559 log2-remaining-fraction=-1945410.9553037602 eliminated-bits=1945410.9553037602 remaining-nodes=86635 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.203000000 budget=600.0`
  `[Program deadline] overshoot=0.0412462`
  `[SearchEngine] New best cost: 75.0097 at iteration 85093 event-epoch=1791824668.8645909 program-deadline=1791825067.6821246`
- Target 2425756.5818701596; remaining/total 0.5990094451678396; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 201.182466269 s, logged first cost 75.0097, logged best cost 75.0097.
- Oracle verdict: not applicable.

### 066_H11_confirm2_decode
- In progress: 2026-10-12T13:11:08.096019-04:00
- Complete metadata: `logs/search_improvements/066_H11_confirm2_decode.json`; log: `logs/search_improvements/066_H11_confirm2_decode.log`; patch: `logs/search_improvements/066_H11_confirm2_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`


- Resume update: R065 confirmation1 passed (first201.182466269 s, same bits/costs). R066 confirmation2 is running, session61066, native source frozen at R061. Existing native warm/cold cache and incremental equivalence tests will run before H10. Prepared H12 one-pass CacheFreshness index and H13 linear-storage overlap ranges remain unapplied. First incumbent event cost is rounded75.0097; deadline progress best75.009705, correcting runner rounded best summaries for R064/R065 (full logs unchanged).
- Completed: 2026-10-12T13:21:08.429451-04:00; exit 0; runner elapsed 600.3279999999795 s.
  `[SearchEngine volume] iterations=91927 elapsed=585.3213978 total-domain-bits=4851513.1637403192 remaining-domain-bits=2906102.208436559 log2-remaining-fraction=-1945410.9553037602 eliminated-bits=1945410.9553037602 remaining-nodes=86464 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.265000000 budget=600.0`
  `[Program deadline] overshoot=0.0483446`
  `[SearchEngine] New best cost: 75.0097 at iteration 85093 event-epoch=1791825273.4843404 program-deadline=1791825668.1449361`
- Target 2425756.5818701596; remaining/total 0.5990094451678396; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 205.339404345 s, logged first cost 75.0097, logged best cost 75.0097.
- Oracle verdict: not applicable.

### 067_build_native_tests_H11
- In progress: 2026-10-12T13:21:33.540253-04:00
- Complete metadata: `logs/search_improvements/067_build_native_tests_H11.json`; log: `logs/search_improvements/067_build_native_tests_H11.log`; patch: `logs/search_improvements/067_build_native_tests_H11.patch`
- Command: `.venv/Scripts/python.exe build.py --targets test --no-lint --log-level INFO`

- O1 OPTIMIZATION retained: R064/R065/R066 identical eliminated1945410.9553037602 / remaining2906102.208436559, first201.836839676 /201.182466269 /205.339404345 s, costs75.0097 /75.009705 unchanged. All program times600.172/600.203/600.265 and overshoots0.002316/0.0412462/0.0483446 pass. All three first events beat R002 conservative search lower bound209.25 s even including program setup, and R046 precise230.178876638 s. Full oracle R06322/22 PASS. Total bits unchanged4851513.1637403192; half target2425756.5818701596 still unmet. H6 alone historical retention stays invalid; O1 includes H8 correctness repair and H7/H9/H11 speedups.

- Completed: 2026-10-12T13:22:21.689972-04:00; exit 0; runner elapsed 48.156999999890104 s.
- Oracle verdict: not applicable.

### 068_native_reader_cache_H11
- In progress: 2026-10-12T13:23:03.154556-04:00
- Complete metadata: `logs/search_improvements/068_native_reader_cache_H11.json`; log: `logs/search_improvements/068_native_reader_cache_H11.log`; patch: `logs/search_improvements/068_native_reader_cache_H11.patch`
- Command: `./test.exe --write-after-read-cache`

### 069_native_incremental_H11
- In progress: 2026-10-12T13:23:03.349612-04:00
- Complete metadata: `logs/search_improvements/069_native_incremental_H11.json`; log: `logs/search_improvements/069_native_incremental_H11.log`; patch: `logs/search_improvements/069_native_incremental_H11.patch`
- Command: `./test.exe --incremental-propagators`

### 070_native_reader_cache_H11
- In progress: 2026-10-12T13:23:18.806967-04:00
- Complete metadata: `logs/search_improvements/070_native_reader_cache_H11.json`; log: `logs/search_improvements/070_native_reader_cache_H11.log`; patch: `logs/search_improvements/070_native_reader_cache_H11.patch`
- Command: `tensor_graphs_cpp/test.exe --write-after-read-cache`
- Completed: 2026-10-12T13:23:19.128246-04:00; exit 0; runner elapsed 0.3279999999795109 s.
- Oracle verdict: not applicable.

### 071_native_incremental_H11
- In progress: 2026-10-12T13:23:19.313455-04:00
- Complete metadata: `logs/search_improvements/071_native_incremental_H11.json`; log: `logs/search_improvements/071_native_incremental_H11.log`; patch: `logs/search_improvements/071_native_incremental_H11.patch`
- Command: `tensor_graphs_cpp/test.exe --incremental-propagators`
- Completed: 2026-10-12T13:23:19.362293-04:00; exit 0; runner elapsed 0.04600000008940697 s.
- Oracle verdict: not applicable.

- 068_native_reader_cache_H11 failed launch: WinError2 incorrect executable path ./test.exe; no tests ran/no volume; corrected invocations R070/R071 passed.

- 069_native_incremental_H11 failed launch: WinError2 incorrect executable path ./test.exe; no tests ran/no volume; corrected invocations R070/R071 passed.

- R070 warm/cold cache PASS300 randomized restoration histories, R071 incremental PASS15 tests. H10 now applied as a trial: cache-slot sharing via necessary root-path selection conditions, requiring pairwise exclusivity in every common bucket and at least one common bucket. Maximum footprint group slots stay disjoint. Raw-fitting CPP preserves O1 placement/order. Duplicate base allocations within a bucket defer. Capacity and sparse/nested-view domains still certify entire layout before fixing anything. Tests20/21/22 cover exclusivity, local coexistence and alternate support paths. Rebuild/full oracle then fixed600-second benchmark; O1 source checkpoint preserved for rollback.

### 072_build_H10
- In progress: 2026-10-12T13:23:35.779913-04:00
- Complete metadata: `logs/search_improvements/072_build_H10.json`; log: `logs/search_improvements/072_build_H10.log`; patch: `logs/search_improvements/072_build_H10.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings --no-lint --log-level INFO`
- Completed: 2026-10-12T13:24:29.065275-04:00; exit 0; runner elapsed 53.281999999890104 s.
- Oracle verdict: not applicable.

### 073_oracle_H10
- In progress: 2026-10-12T13:24:37.783891-04:00
- Complete metadata: `logs/search_improvements/073_oracle_H10.json`; log: `logs/search_improvements/073_oracle_H10.log`; patch: `logs/search_improvements/073_oracle_H10.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T13:24:40.076338-04:00; exit 0; runner elapsed 2.281999999890104 s.
- Oracle verdict: 22/22 cases passed.

### 074_H10_decode
- In progress: 2026-10-12T13:24:48.109486-04:00
- Complete metadata: `logs/search_improvements/074_H10_decode.json`; log: `logs/search_improvements/074_H10_decode.log`; patch: `logs/search_improvements/074_H10_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`


- Completed: 2026-10-12T13:34:48.526941-04:00; exit 0; runner elapsed 600.4069999998901 s.
  `[SearchEngine volume] iterations=92201 elapsed=585.58213430000001 total-domain-bits=4851513.1637403192 remaining-domain-bits=2906102.208436559 log2-remaining-fraction=-1945410.9553037602 eliminated-bits=1945410.9553037602 remaining-nodes=86738 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.344000000 budget=600.0`
  `[Program deadline] overshoot=0.0491199`
  `[SearchEngine] New best cost: 75.0097 at iteration 85093 event-epoch=1791826088.4044015 program-deadline=1791826488.1635053`
- Target 2425756.5818701596; remaining/total 0.5990094451678396; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 200.240896225 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.

- H10 R074 trial rejected/reverted: groupedGPU cache1004030 pages from1577470 (648 groups/677 bases), still exceeds524288 cap even beforeplain53866 pages. Bits unchangedO1, first200.240896225 s vsO1 range201.18..205.34 (timing gain uncertain/no pruning). Costs unchanged, oracle22/22, overshoot0.0491199. Full patch/log retained; O1 staticlayout restored; untracked helper removed.
- H14 PersistentSourceOrder trial: canonicalize a closed DAG of childless INPUT/CACHE and source-only storage views (exclude write views FILL/SCATTER). These have zero runtime in the trusted cost model and persistent ultimate storage, so moving them before all other operations changes neither runtime nor any nonpersistent lifetime ordering. Reserve fixed topological ranks for all possibly selected sources; keep every other active operation in its original relative order after the prefix and compact those ranks. At mostN possible classes ensures ranks fit0..N-1. Check every canonical rank and the entire possible remapped range against every current start domain; sparse/too-tight buckets defer. Selections/offsets/cachechoices unchanged, other execution order/cost preserved. Unselected canonical starts still clean up to0, their irrelevant existing semantics; no cleanup change. Cases23 parallel engines/source view and24 sparse fixed starts added. Root-only rule; initial domains unchanged. Build/fulloracle/evaluation pending.

### 075_build_H14
- In progress: 2026-10-12T13:40:01.088895-04:00
- Complete metadata: `logs/search_improvements/075_build_H14.json`; log: `logs/search_improvements/075_build_H14.log`; patch: `logs/search_improvements/075_build_H14.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings --no-lint --log-level INFO`
- Completed: 2026-10-12T13:40:52.021186-04:00; exit 0; runner elapsed 50.92200000002049 s.
- Oracle verdict: not applicable.

### 076_oracle_H14
- In progress: 2026-10-12T13:40:52.673764-04:00
- Complete metadata: `logs/search_improvements/076_oracle_H14.json`; log: `logs/search_improvements/076_oracle_H14.log`; patch: `logs/search_improvements/076_oracle_H14.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T13:40:54.778325-04:00; exit 1; runner elapsed 2.109999999869615 s.
- Oracle verdict: 23/24 cases passed.

### 077_oracle_H14_valid_view
- In progress: 2026-10-12T13:41:12.982602-04:00
- Complete metadata: `logs/search_improvements/077_oracle_H14_valid_view.json`; log: `logs/search_improvements/077_oracle_H14_valid_view.log`; patch: `logs/search_improvements/077_oracle_H14_valid_view.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T13:41:15.117942-04:00; exit 0; runner elapsed 2.140000000130385 s.
- Oracle verdict: 24/24 cases passed.

### 078_H14_decode
- In progress: 2026-10-12T13:41:20.043365-04:00
- Complete metadata: `logs/search_improvements/078_H14_decode.json`; log: `logs/search_improvements/078_H14_decode.log`; patch: `logs/search_improvements/078_H14_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`

- Completed: 2026-10-12T13:51:20.626105-04:00; exit 0; runner elapsed 600.5779999999795 s.
  `[SearchEngine volume] iterations=90294 elapsed=585.835106 total-domain-bits=4851513.1637403192 remaining-domain-bits=2831709.7065227679 log2-remaining-fraction=-2019803.4572175513 eliminated-bits=2019803.4572175513 remaining-nodes=85123 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.516000000 budget=600.0`
  `[Program deadline] overshoot=0.0163472`
  `[SearchEngine] New best cost: 75.0097 at iteration 84209 event-epoch=1791827129.6180396 program-deadline=1791827480.0996385`
- Target 2425756.5818701596; remaining/total 0.5836755690341433; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 249.518401146 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.

- H14 R078 trial rejected/reverted: sources307 perdecode bucket and306 zero-weightfull, eliminated2019803.4572175513 / remaining2831709.7065227679 (+74392.501913791 vsO1), first249.518401146 s (timing gateFAIL vsR046230.178876638 andO1~201..205), costsunchanged, oracleR07724/24PASS, overshoot0.0163472. Propagatorregistration restoredO1 and untrackedsource removed, patch/log retained. Newcases23/24 retained. Extension stillH14 until nextbuild.
- H13 trial now applied toO1: replace explicit O(overlap-pairs) cached pairvector with O(allocations) sorted spatialranges. Exclude views/persistent allocations before sorting (exactoldpairpredicate); within eachspace, later sortedstarts before firstend are exactly its physicaloverlaps for positivefootprints. Skip samebase live, retainliveSTART reads/readercache/invalidation/restoration rules. No pruningpolicy change; sourcepatch captures exactwork. Nativebuild/fulloracle/cacheequivalence tests then600s evaluation.

### 079_build_H13
- In progress: 2026-10-12T13:55:30.430266-04:00
- Complete metadata: `logs/search_improvements/079_build_H13.json`; log: `logs/search_improvements/079_build_H13.log`; patch: `logs/search_improvements/079_build_H13.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-12T13:56:27.318618-04:00; exit 0; runner elapsed 56.89099999982864 s.
- Oracle verdict: not applicable.

### 080_oracle_H13
- In progress: 2026-10-12T13:57:52.068756-04:00
- Complete metadata: `logs/search_improvements/080_oracle_H13.json`; log: `logs/search_improvements/080_oracle_H13.log`; patch: `logs/search_improvements/080_oracle_H13.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T13:57:54.164445-04:00; exit 0; runner elapsed 2.094000000040978 s.
- Oracle verdict: 24/24 cases passed.

### 081_native_reader_cache_H13
- In progress: 2026-10-12T13:57:54.355003-04:00
- Complete metadata: `logs/search_improvements/081_native_reader_cache_H13.json`; log: `logs/search_improvements/081_native_reader_cache_H13.log`; patch: `logs/search_improvements/081_native_reader_cache_H13.patch`
- Command: `tensor_graphs_cpp/test.exe --write-after-read-cache`
- Completed: 2026-10-12T13:57:54.455690-04:00; exit 0; runner elapsed 0.0940000000409782 s.
- Oracle verdict: not applicable.

### 082_native_incremental_H13
- In progress: 2026-10-12T13:57:54.639364-04:00
- Complete metadata: `logs/search_improvements/082_native_incremental_H13.json`; log: `logs/search_improvements/082_native_incremental_H13.log`; patch: `logs/search_improvements/082_native_incremental_H13.patch`
- Command: `tensor_graphs_cpp/test.exe --incremental-propagators`
- Completed: 2026-10-12T13:57:54.687483-04:00; exit 0; runner elapsed 0.0470000000204891 s.
- Oracle verdict: not applicable.

### 083_H13_decode
- In progress: 2026-10-12T13:58:01.503427-04:00
- Complete metadata: `logs/search_improvements/083_H13_decode.json`; log: `logs/search_improvements/083_H13_decode.log`; patch: `logs/search_improvements/083_H13_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`

- Completed: 2026-10-12T14:08:01.897709-04:00; exit 0; runner elapsed 600.375 s.
  `[SearchEngine volume] iterations=92097 elapsed=586.04395869999996 total-domain-bits=4851513.1637403192 remaining-domain-bits=2906102.208436559 log2-remaining-fraction=-1945410.9553037602 eliminated-bits=1945410.9553037602 remaining-nodes=86634 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.328000000 budget=600.0`
  `[Program deadline] overshoot=0.152505`
  `[SearchEngine] New best cost: 75.0097 at iteration 85093 event-epoch=1791828079.9355271 program-deadline=1791828481.5506754`
- Target 2425756.5818701596; remaining/total 0.5990094451678396; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 198.384851694 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.

### 084_H13_confirm1_decode
- In progress: 2026-10-12T14:08:22.784938-04:00
- Complete metadata: `logs/search_improvements/084_H13_confirm1_decode.json`; log: `logs/search_improvements/084_H13_confirm1_decode.log`; patch: `logs/search_improvements/084_H13_confirm1_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`

- H13R083 promising: eliminated1945410.9553037602 / remaining2906102.208436559 unchangedO1; first198.384851694 s (2.797614575 s faster thanfastestO1R065201.182466269;31.794024944 s vsR046), costs75.0097/75.009705 unchanged, oracle24/24 +300restorationhistories +15incrementaltests PASS. Overshoot0.152505 within1s limit. R084/R085 confirmations running session64109 samebuild/source. H18 cost-first branchordering afterfirstincumbent prepared next, notapplied; preserveO1 beforecandidate acceptance.

- Completed: 2026-10-12T14:18:23.075812-04:00; exit 0; runner elapsed 600.280999999959 s.
  `[SearchEngine volume] iterations=92017 elapsed=585.81306189999998 total-domain-bits=4851513.1637403192 remaining-domain-bits=2906102.208436559 log2-remaining-fraction=-1945410.9553037602 eliminated-bits=1945410.9553037602 remaining-nodes=86554 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.218000000 budget=600.0`
  `[Program deadline] overshoot=0.0398231`
  `[SearchEngine] New best cost: 75.0097 at iteration 85093 event-epoch=1791828706.52753 program-deadline=1791829102.8356097`
- Target 2425756.5818701596; remaining/total 0.5990094451678396; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 203.691920280 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.

### 085_H13_confirm2_decode
- In progress: 2026-10-12T14:18:23.291612-04:00
- Complete metadata: `logs/search_improvements/085_H13_confirm2_decode.json`; log: `logs/search_improvements/085_H13_confirm2_decode.log`; patch: `logs/search_improvements/085_H13_confirm2_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`

- R084 H13 confirmation1: bits/costs identicalO1, first203.691920280 s withinO1 measured201.18..205.34 range (2.509454012 slower thanfastestO1,1.647484064 faster thanworst); overshoot0.0398231. No timinggain substantiated bythis repeat alone; H13 remainsuncertain pendingR085, notretained. R085 running session64109. H18 prepared inedit_h18.py butnativebrancher untouched.

- Completed: 2026-10-12T14:28:23.630332-04:00; exit 0; runner elapsed 600.3279999999795 s.
  `[SearchEngine volume] iterations=92037 elapsed=585.72140950000005 total-domain-bits=4851513.1637403192 remaining-domain-bits=2906102.208436559 log2-remaining-fraction=-1945410.9553037602 eliminated-bits=1945410.9553037602 remaining-nodes=86574 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.265000000 budget=600.0`
  `[Program deadline] overshoot=0.0310791`
  `[SearchEngine] New best cost: 75.0097 at iteration 85093 event-epoch=1791829304.9909821 program-deadline=1791829703.348737`
- Target 2425756.5818701596; remaining/total 0.5990094451678396; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 201.642245054 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.

- H13R083/R084/R085 correctnessclean but REJECTED/reverted pending measurable benefit: first198.384851694 /203.691920280 /201.642245054 s; median201.642 vsO1 median201.837 is only0.195 s within4+seconds observedvariation. Identical bits/costs and nearidentical iterations toO1. All overshoots0.152505/0.0398231/0.0310791 pass. Do not claim repeatable performancegain. Linearstorage source/patch/log retained for future dense-overlap workload, O1WARStart restored.
- H18 now applied asfocused branch-order trial onO1: getENodeScore retains exact feasibility-first score while bestcost isINF; afteranincumbent, useoptimistic DAGruntime+smalllocalcost tie-break insteadof memorypressure-dominatedscore. Bothbinary branches preserved; no cuts, domainchanges, objective orincumbentpolicy changes. Firstdive unchanged; promisingdevice alternatives mayfindlowercostwitnesses andclosemorefrontier afterfirst incumbent. Fulloracle thenfixed600s evaluation. Originalbrancher backupO1_brancher.hpp.

### 086_build_H18
- In progress: 2026-10-12T14:29:44.500899-04:00
- Complete metadata: `logs/search_improvements/086_build_H18.json`; log: `logs/search_improvements/086_build_H18.log`; patch: `logs/search_improvements/086_build_H18.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings --no-lint --log-level INFO`
- Completed: 2026-10-12T14:30:36.328566-04:00; exit 0; runner elapsed 51.82799999997951 s.
- Oracle verdict: not applicable.

### 087_oracle_H18
- In progress: 2026-10-12T14:31:13.906870-04:00
- Complete metadata: `logs/search_improvements/087_oracle_H18.json`; log: `logs/search_improvements/087_oracle_H18.log`; patch: `logs/search_improvements/087_oracle_H18.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T14:31:16.123052-04:00; exit 0; runner elapsed 2.219000000040978 s.
- Oracle verdict: 24/24 cases passed.

### 088_H18_decode
- In progress: 2026-10-12T14:31:35.091364-04:00
- Complete metadata: `logs/search_improvements/088_H18_decode.json`; log: `logs/search_improvements/088_H18_decode.log`; patch: `logs/search_improvements/088_H18_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`


- Completed: 2026-10-12T14:41:35.618211-04:00; exit 0; runner elapsed 600.5149999998976 s.
  `[SearchEngine volume] iterations=92073 elapsed=585.90852059999997 total-domain-bits=4851513.1637403192 remaining-domain-bits=2906102.208436559 log2-remaining-fraction=-1945410.9553037602 eliminated-bits=1945410.9553037602 remaining-nodes=86610 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.454000000 budget=600.0`
  `[Program deadline] overshoot=0.0562119`
  `[SearchEngine] New best cost: 75.0097 at iteration 85093 event-epoch=1791830095.9393449 program-deadline=1791830495.1389928`
- Target 2425756.5818701596; remaining/total 0.5990094451678396; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 200.800352097 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.

- H18R088 rejected/reverted: eliminated1945410.9553037602 / remaining2906102.208436559 unchanged, first200.800352097 s (samefirstdive), costs75.0097/75.009705 unchanged; fulloracle24/24 andovershoot0.0562119 pass, no demonstrated benefit. Originalbrancher restored. H17 audit: EGraph permits differing alternative strides; StaticPrivateLayout reservesclass VarInfo span whilememorypropagators use selectedENode actualspan. A newnative regression forces a three-page paddedchild whosemetadata says two pages plusa two-page persistentroot in fivepages. Witnesschild0/root3 isfeasible; oldstatic placementchild0/root2 overlaps. Addtest harness, runbeforefix, thenreserve maxactualspan perallocation andguardviewspans beforeevaluatingfixedcontract.

### 089_build_padded_regression_beforefix
- In progress: 2026-10-12T14:42:38.064061-04:00
- Complete metadata: `logs/search_improvements/089_build_padded_regression_beforefix.json`; log: `logs/search_improvements/089_build_padded_regression_beforefix.log`; patch: `logs/search_improvements/089_build_padded_regression_beforefix.patch`
- Command: `.venv/Scripts/python.exe build.py --targets test --no-lint --log-level INFO`
- Completed: 2026-10-12T14:43:22.749805-04:00; exit 0; runner elapsed 44.67199999978766 s.
- Oracle verdict: not applicable.

### 090_padded_regression_beforefix
- In progress: 2026-10-12T14:44:09.900436-04:00
- Complete metadata: `logs/search_improvements/090_padded_regression_beforefix.json`; log: `logs/search_improvements/090_padded_regression_beforefix.log`; patch: `logs/search_improvements/090_padded_regression_beforefix.patch`
- Command: `tensor_graphs_cpp/test.exe --private-layout`
- Completed: 2026-10-12T14:44:09.995842-04:00; exit 1; runner elapsed 0.09399999980814755 s.
- Oracle verdict: not applicable.

- R090 reproduced correctnessfailure: originalmanual paddedchild0/root3 witness accepted byFixedOffsetStart, oldStaticPrivateLayout child0/root2 rejected. O1 retention now requires H17 repair; preserve its historicalthreevalidtimingresults but do notretain knownincorrect sizing. H17 nowreserves max actualENodepagefootprint plusmetadata minimum for eachplain/privatecache allocation. View actualspans mustfit each candidate privatebase range or existingpreallocationprefix; otherwise defer entire space. Root-onlycertificate/noinitialdomaincostchanges. Testcachedlayoutrequires fivepages twoallocs insteadoffour metadata-onlypages.
- Temporary LP diagnostic export added toSearchEngine, guarded byTG_SEARCH_LP_DUMP, before rootpropagation. Helper logs/search_improvements/search_lp_export.hpp and exactnativepatch retained. This instrumentation will be removed before H17 comparablebenchmark; diagnostic will export initialstate, cachebudget/engines/altcosts/spans for offlineLP analysis. NoLPpruningrule installed. Nativebuild/fulloracle/newregression first.

### 091_build_H17_diagnostic
- In progress: 2026-10-12T14:45:00.117959-04:00
- Complete metadata: `logs/search_improvements/091_build_H17_diagnostic.json`; log: `logs/search_improvements/091_build_H17_diagnostic.log`; patch: `logs/search_improvements/091_build_H17_diagnostic.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-12T14:45:58.007800-04:00; exit 0; runner elapsed 57.89100000006147 s.
- Oracle verdict: not applicable.

### 092_padded_regression_H17
- In progress: 2026-10-12T14:46:39.638101-04:00
- Complete metadata: `logs/search_improvements/092_padded_regression_H17.json`; log: `logs/search_improvements/092_padded_regression_H17.log`; patch: `logs/search_improvements/092_padded_regression_H17.patch`
- Command: `tensor_graphs_cpp/test.exe --private-layout`
- Completed: 2026-10-12T14:46:39.709445-04:00; exit 0; runner elapsed 0.0779999999795109 s.
- Oracle verdict: not applicable.

### 093_oracle_H17_diagnostic
- In progress: 2026-10-12T14:46:39.900823-04:00
- Complete metadata: `logs/search_improvements/093_oracle_H17_diagnostic.json`; log: `logs/search_improvements/093_oracle_H17_diagnostic.log`; patch: `logs/search_improvements/093_oracle_H17_diagnostic.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T14:46:42.042345-04:00; exit 0; runner elapsed 2.1409999998286366 s.
- Oracle verdict: 24/24 cases passed.

### 094_native_reader_cache_H17
- In progress: 2026-10-12T14:46:42.249102-04:00
- Complete metadata: `logs/search_improvements/094_native_reader_cache_H17.json`; log: `logs/search_improvements/094_native_reader_cache_H17.log`; patch: `logs/search_improvements/094_native_reader_cache_H17.patch`
- Command: `tensor_graphs_cpp/test.exe --write-after-read-cache`
- Completed: 2026-10-12T14:46:42.298978-04:00; exit 0; runner elapsed 0.0470000000204891 s.
- Oracle verdict: not applicable.

### 095_native_incremental_H17
- In progress: 2026-10-12T14:46:42.498161-04:00
- Complete metadata: `logs/search_improvements/095_native_incremental_H17.json`; log: `logs/search_improvements/095_native_incremental_H17.log`; patch: `logs/search_improvements/095_native_incremental_H17.patch`
- Command: `tensor_graphs_cpp/test.exe --incremental-propagators`
- Completed: 2026-10-12T14:46:42.547373-04:00; exit 0; runner elapsed 0.0470000000204891 s.
- Oracle verdict: not applicable.

- H17R092 paddedregression PASS (oldR090 FAIL), R093oracle24/24 PASS, R094300warm/cold histories PASS, R09515incremental PASS. Beforecomparable600s evaluation, shortdiagnostic96 includesTEMPORARY guardedinitial-stateexport (sourceinR091patch, helperunderlogs). Exportenvironmentcaptured inmetadata; 60s diagnostic isnot a600s comparison/retentionrun. Instrumentation mustbe removed andbindingsrebuilt beforeH17benchmark.

### 096_lp_export_diagnostic_decode60
- In progress: 2026-10-12T14:47:21.850169-04:00
- Complete metadata: `logs/search_improvements/096_lp_export_diagnostic_decode60.json`; log: `logs/search_improvements/096_lp_export_diagnostic_decode60.log`; patch: `logs/search_improvements/096_lp_export_diagnostic_decode60.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 60`

## Session 3 comparison and resume checkpoint

Original permanent baseline is R002. R046 repeats the same original implementation with native incumbent event timestamps, supplying a precise contemporary timing anchor. R002 only brackets its first event at search elapsed [209.25,222.85] seconds; program setup was not timestamped, so its exact program event time is unavailable. First costs in these logs are rounded to 75.0097; deadline progress logs report 75.009705. Do not infer extra cost precision.

| Run / implementation | Eliminated / remaining bits | First incumbent program seconds | First / deadline cost | Delta bits / first seconds vs R046 | Program elapsed / overshoot seconds | Gate |
|---|---:|---:|---:|---:|---:|---|
| [R002 original baseline](../../logs/search_improvements/002_baseline_decode.log) | 2690.1009323587641 / 4848823.0628079604 | unavailable; search bracket above | 75.0097 / 75.009705 | 0 / unavailable | 600.266 / 0.0407553 | Permanent bit/cost anchor |
| Currently accepted implementation | O1 repaired H6 + H7/H8/H9/H11 (1,945,410.9553037602 / 2,906,102.208436559) | 201.182466269 to205.339404345 (R064/R065/R066) | 75.0097 /75.009705 | +1942720.854371401 / -28.996410370 to-24.839472294 | 600.172 to600.265 /0.002316 to0.0483446 | Historical O1 passes; R090 padded-storage failure requires H17 repair |
| [046_timed_baseline_decode](../../logs/search_improvements/046_timed_baseline_decode.log) | 2690.1009323587641 / 4848823.0628079604 | 230.178876638 | 75.0097 / 75.009705 | +0.000000000 / +0.000000000 | 600.312000000 / 0.0730863 | Precise timing baseline |
| [051_H7_H8_decode](../../logs/search_improvements/051_H7_H8_decode.log) | 2690.1009323587641 / 4848823.0628079604 | 230.326277971 | 75.0097 / 75.009705 | +0.000000000 / +0.147401333 | 600.250000000 / 0.0482612 | No gain; not retained |
| [056_H7_H8_decode](../../logs/search_improvements/056_H7_H8_decode.log) | 1945410.9553037602 / 2906102.208436559 | 302.621661186 | 75.0097 / 75.009705 | +1942720.854371401 / +72.442784548 | 600.407000000 / 0.0364728 | FAIL: first incumbent regresses |
| [060_H9_decode](../../logs/search_improvements/060_H9_decode.log) | 1945410.9553037602 / 2906102.208436559 | 271.344538212 | 75.0097 / 75.009705 | +1942720.854371401 / +41.165661573 | 600.407000000 / 0.127546 | FAIL: first incumbent regresses |
| [064_H11_decode](../../logs/search_improvements/064_H11_decode.log) | 1945410.9553037602 / 2906102.208436559 | 201.836839676 | 75.0097 / 75.009705 | +1942720.854371401 / -28.342036963 | 600.172000000 / 0.002316 | Historical O1 metric pass; R090 requires H17 correctness repair |
| [065_H11_confirm1_decode](../../logs/search_improvements/065_H11_confirm1_decode.log) | 1945410.9553037602 / 2906102.208436559 | 201.182466269 | 75.0097 / 75.009705 | +1942720.854371401 / -28.996410370 | 600.203000000 / 0.0412462 | Historical O1 metric pass; R090 requires H17 correctness repair |
| [066_H11_confirm2_decode](../../logs/search_improvements/066_H11_confirm2_decode.log) | 1945410.9553037602 / 2906102.208436559 | 205.339404345 | 75.0097 / 75.009705 | +1942720.854371401 / -24.839472294 | 600.265000000 / 0.0483446 | Historical O1 metric pass; R090 requires H17 correctness repair |
| [074_H10_decode](../../logs/search_improvements/074_H10_decode.log) | 1945410.9553037602 / 2906102.208436559 | 200.240896225 | 75.0097 / 75.009705 | +1942720.854371401 / -29.937980413 | 600.344000000 / 0.0491199 | Rejected: no bit gain / timing noise |
| [078_H14_decode](../../logs/search_improvements/078_H14_decode.log) | 2019803.4572175513 / 2831709.7065227679 | 249.518401146 | 75.0097 / 75.009705 | +2017113.356285193 / +19.339524508 | 600.516000000 / 0.0163472 | FAIL: first incumbent regresses |
| [083_H13_decode](../../logs/search_improvements/083_H13_decode.log) | 1945410.9553037602 / 2906102.208436559 | 198.384851694 | 75.0097 / 75.009705 | +1942720.854371401 / -31.794024944 | 600.328000000 / 0.152505 | Rejected: speed gain not distinguished from noise |
| [084_H13_confirm1_decode](../../logs/search_improvements/084_H13_confirm1_decode.log) | 1945410.9553037602 / 2906102.208436559 | 203.691920280 | 75.0097 / 75.009705 | +1942720.854371401 / -26.486956358 | 600.218000000 / 0.0398231 | Rejected: speed gain not distinguished from noise |
| [085_H13_confirm2_decode](../../logs/search_improvements/085_H13_confirm2_decode.log) | 1945410.9553037602 / 2906102.208436559 | 201.642245054 | 75.0097 / 75.009705 | +1942720.854371401 / -28.536631584 | 600.265000000 / 0.0310791 | Rejected: speed gain not distinguished from noise |
| [088_H18_decode](../../logs/search_improvements/088_H18_decode.log) | 1945410.9553037602 / 2906102.208436559 | 200.800352097 | 75.0097 / 75.009705 | +1942720.854371401 / -29.378524542 | 600.454000000 / 0.0562119 | Historical O1 metric pass; R090 requires H17 correctness repair |

Costs have zero logged delta in every comparable run. H6 historical R034/R035/R041 had no incumbent within budget, so time and both costs are unavailable (+infinity) and its retention fails the gate. R048 also demonstrated H6 unsoundness on nested sparse views; H8 repairs this with union-of-addresses validation (R050/R055). R059 passed 21 cases; R063 passed 22, including alternate support paths. All reported deadlines are within the 1-second contract limit; they are not exact zero-overshoot measurements.

Working native state: retained O1 H6 plus H7 interval scans, H8 alias certificate repair, H9 allocation-overlap caching and deferred index rebuild, H11 lazy reader metadata. O1 buildR061/oracleR06322/22 passed. H10 rejected/reverted afterR074 because GPU grouping still exceeds cap and bits unchanged. H14 rejected/reverted afterR078 timing249.518401146 s. H13 rejected/reverted afterthree measurements with indistinguishable timing and unchangedbits/costs; logs/source retained. H18 rejected/reverted afterR088 no bit/cost gain. O1 failed new padded-storage regressionR090. Current candidate adds H17 actual alternative footprint sizing plus view-range guards; R092 regression/R093oracle24/24/R094300histories/R09515tests PASS. R091 build has temporary export instrumentation, which will be removed before benchmark. Initial domains, costs, incumbent policy, evaluator and reference semantics remain unchanged.

Running/pending: R096 60-second LP export DIAGNOSTIC running session12838, output096_native_initial_state.json; not a comparable600s score. After completion remove temporary SearchEngine export include/envcall (restore H11 source checkpoint search_engine.hpp), rebuild bindings, fulloracle, then H17 fixed600s evaluation and two confirmations if promising. Offline LP solver underlogs uses .venvx64 Python, no LP bound in native code. Original baseline remains permanent; O1 requires H17 repair for retention. H12/H13/H18 trial sources retained offline. Target still unmet.

No half-bit target achieved yet. All full command/source/patch/start/end records are linked in individual run entries and companion JSON files. All rejected and failed trials remain in the record.
<!-- SESSION3_COMPARISON_END -->
- Completed: 2026-10-12T14:48:22.019987-04:00; exit 0; runner elapsed 60.17100000008941 s.
  `[SearchEngine volume] iterations=8188 elapsed=45.802532200000002 total-domain-bits=4851513.1637403192 remaining-domain-bits=2907262.2387936418 log2-remaining-fraction=-1944250.9249466774 eliminated-bits=1944250.9249466774 remaining-nodes=8189 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=60.110000000 budget=60.0`
  `[Program deadline] overshoot=0.000378847`
- Target 2425756.5818701596; remaining/total 0.59924855208519334; target met: False.
- Incumbent: no incumbent within budget; time and both costs unavailable (+infinity).
- Oracle verdict: not applicable.

### 097_offline_lp_diagnostic
- In progress: 2026-10-12T14:49:02.251615-04:00
- Complete metadata: `logs/search_improvements/097_offline_lp_diagnostic.json`; log: `logs/search_improvements/097_offline_lp_diagnostic.log`; patch: `logs/search_improvements/097_offline_lp_diagnostic.patch`
- Command: `.venvx64/Scripts/python.exe logs/search_improvements/solve_search_lp.py logs/search_improvements/096_native_initial_state.json`
- Completed: 2026-10-12T14:51:13.700700-04:00; exit 1; runner elapsed 131.45300000021234 s.
- Oracle verdict: not applicable.

### 098_offline_lp_bucket0_diagnostic
- In progress: 2026-10-12T14:53:53.169701-04:00
- Complete metadata: `logs/search_improvements/098_offline_lp_bucket0_diagnostic.json`; log: `logs/search_improvements/098_offline_lp_bucket0_diagnostic.log`; patch: `logs/search_improvements/098_offline_lp_bucket0_diagnostic.patch`
- Command: `.venvx64/Scripts/python.exe logs/search_improvements/solve_search_lp.py logs/search_improvements/096_native_initial_state.json 0`
- Completed: 2026-10-12T14:53:58.577928-04:00; exit 0; runner elapsed 5.4060000001918525 s.
- Oracle verdict: not applicable.

### 099_h17_clean_build
- In progress: 2026-10-12T14:57:21.010945-04:00
- Complete metadata: `logs/search_improvements/099_h17_clean_build.json`; log: `logs/search_improvements/099_h17_clean_build.log`; patch: `logs/search_improvements/099_h17_clean_build.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-12T14:57:21.610487-04:00; exit 0; runner elapsed 0.5940000000409782 s.
- Oracle verdict: not applicable.

### 100_h17_clean_rebuild
- In progress: 2026-10-12T14:57:33.640298-04:00
- Complete metadata: `logs/search_improvements/100_h17_clean_rebuild.json`; log: `logs/search_improvements/100_h17_clean_rebuild.log`; patch: `logs/search_improvements/100_h17_clean_rebuild.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-12T14:58:33.075290-04:00; exit 0; runner elapsed 59.437000000150874 s.
- Oracle verdict: not applicable.

### 101_oracle_H17_clean
- In progress: 2026-10-12T14:59:48.345109-04:00
- Complete metadata: `logs/search_improvements/101_oracle_H17_clean.json`; log: `logs/search_improvements/101_oracle_H17_clean.log`; patch: `logs/search_improvements/101_oracle_H17_clean.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T14:59:50.570833-04:00; exit 0; runner elapsed 2.2340000001713634 s.
- Oracle verdict: 24/24 cases passed.

### 102_padded_regression_H17_clean
- In progress: 2026-10-12T14:59:50.794241-04:00
- Complete metadata: `logs/search_improvements/102_padded_regression_H17_clean.json`; log: `logs/search_improvements/102_padded_regression_H17_clean.log`; patch: `logs/search_improvements/102_padded_regression_H17_clean.patch`
- Command: `tensor_graphs_cpp/test.exe --private-layout`
- Completed: 2026-10-12T14:59:50.801537-04:00; exit 127; runner elapsed 0.0 s.
- Oracle verdict: not applicable.

### 103_padded_regression_H17_clean
- In progress: 2026-10-12T14:59:55.193526-04:00
- Complete metadata: `logs/search_improvements/103_padded_regression_H17_clean.json`; log: `logs/search_improvements/103_padded_regression_H17_clean.log`; patch: `logs/search_improvements/103_padded_regression_H17_clean.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --private-layout`
- Completed: 2026-10-12T14:59:55.271198-04:00; exit 0; runner elapsed 0.0779999999795109 s.
- Oracle verdict: not applicable.

- 102_padded_regression_H17_clean failed launch (WinError2, relative exe path); no test ran. Rerun as R103.

### 104_H17_decode
- In progress: 2026-10-12T14:59:58.093988-04:00
- Complete metadata: `logs/search_improvements/104_H17_decode.json`; log: `logs/search_improvements/104_H17_decode.log`; patch: `logs/search_improvements/104_H17_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`

## Session 4 (2026-10-12, resumed)
- Reconciled: R096 diagnostic finished; R097 offline LP failed (exit 1), R098 bucket-0 LP objective 10.90 (primal not verified; diagnostic only, no native LP rule). R099/R100 rebuilt H17 with the temporary LP export removed (no `TG_SEARCH_LP_DUMP` in native source). R101 oracle 24/24 PASS; R102 launch failure (relative exe path, no test ran); R103 padded regression PASS. R104 H17 fixed 600 s evaluation started (background).
- H19 prepared while R104 runs (native source edited but NOT built until R104 finishes; R104 patch was captured at its start): `StartRankWindowPropagator` (root-only). START holds 1.655M of the 2.906M remaining root bits. Lower bound start(c) >= min_start + |ND(c)|, where ND(c) = intersection over remaining alternatives of union over children of {child} + ND(child): every such class is active and strictly earlier whenever c is active, and active starts are distinct (StartUnique). Valid for every feasible plan. Upper bound start(c) <= N-1-|NA(c)| with NA(c) = intersection over possible parents p of {p}+NA(p) (a selected non-root class needs a selected parent, reachability): dominance by rank compression. evaluateMakespan/extractSolution only sort active starts, and every start constraint compares active starts, so compressing active ranks to 0..k-1 (k <= N reachable) keeps feasibility and cost; compressed ranks never exceed the originals, so they stay in the initial domains only if all start domains are downward closed from 0 (checked per bucket; otherwise only the lower bound applies; e.g. oracle case 24's sparse ranks). Offsets/selections are untouched, so it composes with the H6 layout certificate (layout argument holds for any feasible plan, including compressed ones). Unselected starts are irrelevant: `UnselectedStartOffsetPropagator` now fixes them to 0 if in domain, else the domain minimum (subset of the box, keeps branch-volume invariants). Classes without parent support or with an empty window are left untouched (optional) or reported as conflict (mandatory). New oracle case 25 (optional class with a positive rank lower bound left unselected by the optimum; multi-parent NA intersection).
- Completed: 2026-10-12T15:09:58.708241-04:00; exit 0; runner elapsed 600.6090000001714 s.
  `[SearchEngine volume] iterations=91909 elapsed=585.87942869999995 total-domain-bits=4851513.1637403192 remaining-domain-bits=2906102.208436559 log2-remaining-fraction=-1945410.9553037602 eliminated-bits=1945410.9553037602 remaining-nodes=86446 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.547000000 budget=600.0`
  `[Program deadline] overshoot=0.269035`
  `[SearchEngine] New best cost: 75.0097 at iteration 85093 event-epoch=1791831817.202038 program-deadline=1791832198.1574659`
- Target 2425756.5818701596; remaining/total 0.5990094451678396; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 219.044572115 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.
- R104 H17 verdict: bits identical to O1 (1945410.9553037602), first incumbent 219.044572115 s (within original gate 230.178876638+1; ~14-18 s slower than O1's 201-205 s, H17 reserves max alternative footprints so private CPP slots change), costs 75.0097/75.009705 unchanged, overshoot 0.269035. H17 is a required correctness repair (R090), retained as current base O2 = O1 + H17. Next: build H19 on O2 + oracle case 26 (fused bypass, ND intersection).

### 105_build_H19
- In progress: 2026-10-12T15:10:09.756033-04:00
- Complete metadata: `logs/search_improvements/105_build_H19.json`; log: `logs/search_improvements/105_build_H19.log`; patch: `logs/search_improvements/105_build_H19.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-12T15:11:06.354702-04:00; exit 0; runner elapsed 56.59400000004098 s.
- Oracle verdict: not applicable.

### 106_oracle_H19
- In progress: 2026-10-12T15:11:13.329270-04:00
- Complete metadata: `logs/search_improvements/106_oracle_H19.json`; log: `logs/search_improvements/106_oracle_H19.log`; patch: `logs/search_improvements/106_oracle_H19.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T15:11:15.759787-04:00; exit 0; runner elapsed 2.4219999997876585 s.
- Oracle verdict: 26/26 cases passed.

### 107_native__private_layout
- In progress: 2026-10-12T15:11:15.992547-04:00
- Complete metadata: `logs/search_improvements/107_native__private_layout.json`; log: `logs/search_improvements/107_native__private_layout.log`; patch: `logs/search_improvements/107_native__private_layout.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --private-layout`
- Completed: 2026-10-12T15:11:16.069589-04:00; exit 0; runner elapsed 0.0779999999795109 s.
- Oracle verdict: not applicable.

### 107_native__write_after_read_cache
- In progress: 2026-10-12T15:11:16.296927-04:00
- Complete metadata: `logs/search_improvements/107_native__write_after_read_cache.json`; log: `logs/search_improvements/107_native__write_after_read_cache.log`; patch: `logs/search_improvements/107_native__write_after_read_cache.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --write-after-read-cache`
- Completed: 2026-10-12T15:11:16.345645-04:00; exit 0; runner elapsed 0.0470000000204891 s.
- Oracle verdict: not applicable.

### 107_native__incremental_propagators
- In progress: 2026-10-12T15:11:16.560142-04:00
- Complete metadata: `logs/search_improvements/107_native__incremental_propagators.json`; log: `logs/search_improvements/107_native__incremental_propagators.log`; patch: `logs/search_improvements/107_native__incremental_propagators.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --incremental-propagators`
- Completed: 2026-10-12T15:11:16.614009-04:00; exit 0; runner elapsed 0.0470000000204891 s.
- Oracle verdict: not applicable.

### 108_H19_decode
- In progress: 2026-10-12T15:11:21.042450-04:00
- Complete metadata: `logs/search_improvements/108_H19_decode.json`; log: `logs/search_improvements/108_H19_decode.log`; patch: `logs/search_improvements/108_H19_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T15:21:21.402746-04:00; exit 0; runner elapsed 600.3599999998696 s.
  `[SearchEngine volume] iterations=92315 elapsed=585.73780069999998 total-domain-bits=4851513.1637403192 remaining-domain-bits=2891972.9489357956 log2-remaining-fraction=-1959540.2148045236 eliminated-bits=1959540.2148045236 remaining-nodes=86852 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.281000000 budget=600.0`
  `[Program deadline] overshoot=0.0357807`
  `[SearchEngine] New best cost: 75.0097 at iteration 85093 event-epoch=1791832477.9992139 program-deadline=1791832881.1014454`
- Target 2425756.5818701596; remaining/total 0.59609710441478059; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 196.897768497 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.
- R105 build H19 PASS; R106 oracle 26/26 PASS (new cases 25, 26); R107 native private-layout/300 WAR histories/15 incremental PASS. R108 H19: +14129.26 bits vs O2 (1959540.2148045236), first incumbent 196.897768497 s, costs unchanged, overshoot 0.0357807. Per-bucket START bits 97390 -> 96587 (buckets 1-15), bucket 0 lower-bound only (start domains not downward closed from 0), bucket 16 97361 -> 96070. ND/NA sets are small because fused alternatives bypass intermediates (consistent with D7). Gate: PASS on candidate run; confirmations R109/R110 next.

### 109_H19_confirm1_decode
- In progress: 2026-10-12T15:25:56.242507-04:00
- Complete metadata: `logs/search_improvements/109_H19_confirm1_decode.json`; log: `logs/search_improvements/109_H19_confirm1_decode.log`; patch: `logs/search_improvements/109_H19_confirm1_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T15:35:56.602077-04:00; exit 0; runner elapsed 600.3589999999385 s.
  `[SearchEngine volume] iterations=92424 elapsed=585.65280050000001 total-domain-bits=4851513.1637403192 remaining-domain-bits=2891972.9489357956 log2-remaining-fraction=-1959540.2148045236 eliminated-bits=1959540.2148045236 remaining-nodes=86961 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.282000000 budget=600.0`
  `[Program deadline] overshoot=0.0830841`
  `[SearchEngine] New best cost: 75.0097 at iteration 85093 event-epoch=1791833353.3425219 program-deadline=1791833756.2955127`
- Target 2425756.5818701596; remaining/total 0.59609710441478059; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 197.047009230 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.

### 110_H19_confirm2_decode
- In progress: 2026-10-12T15:36:01.929433-04:00
- Complete metadata: `logs/search_improvements/110_H19_confirm2_decode.json`; log: `logs/search_improvements/110_H19_confirm2_decode.log`; patch: `logs/search_improvements/110_H19_confirm2_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T15:46:02.230259-04:00; exit 0; runner elapsed 600.2970000000205 s.
  `[SearchEngine volume] iterations=92345 elapsed=585.36746900000003 total-domain-bits=4851513.1637403192 remaining-domain-bits=2891972.9489357956 log2-remaining-fraction=-1959540.2148045236 eliminated-bits=1959540.2148045236 remaining-nodes=86882 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.234000000 budget=600.0`
  `[Program deadline] overshoot=0.0419533`
  `[SearchEngine] New best cost: 75.0097 at iteration 85093 event-epoch=1791833964.9239676 program-deadline=1791834361.9847503`
- Target 2425756.5818701596; remaining/total 0.59609710441478059; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 202.939217329 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.
- O3 OPTIMIZATION retained: H19 on O2. R108/R109/R110 identical eliminated 1959540.2148045236 (+14129.2594 vs O2 R104), first incumbent 196.897768497/197.047009230/202.939217329 s (all < original gate 231.18 and <= O2 219.04), costs 75.0097/75.009705 unchanged, overshoots 0.0357807/0.0830841/0.0419533, oracle R106 26/26 + native R107 PASS. H19 source backup logs/search_improvements/H19_start_rank_window.hpp. H20 now applied: persistent-source prefix folded into the StartRankWindow canonical form (sources fixed ranks 0..S-1 children-first, others start >= S+|ND\Src|; only for downward-closed buckets). Composition argument recorded in docs/core/propagators.md.

### 111_build_H20
- In progress: 2026-10-12T15:46:11.754536-04:00
- Complete metadata: `logs/search_improvements/111_build_H20.json`; log: `logs/search_improvements/111_build_H20.log`; patch: `logs/search_improvements/111_build_H20.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-12T15:47:09.662706-04:00; exit 0; runner elapsed 57.90599999995902 s.
- Oracle verdict: not applicable.

### 112_oracle_H20
- In progress: 2026-10-12T15:47:09.925479-04:00
- Complete metadata: `logs/search_improvements/112_oracle_H20.json`; log: `logs/search_improvements/112_oracle_H20.log`; patch: `logs/search_improvements/112_oracle_H20.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T15:47:12.331289-04:00; exit 0; runner elapsed 2.406999999890104 s.
- Oracle verdict: 26/26 cases passed.

### 113_native__private_layout
- In progress: 2026-10-12T15:47:15.627816-04:00
- Complete metadata: `logs/search_improvements/113_native__private_layout.json`; log: `logs/search_improvements/113_native__private_layout.log`; patch: `logs/search_improvements/113_native__private_layout.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --private-layout`
- Completed: 2026-10-12T15:47:15.806119-04:00; exit 0; runner elapsed 0.17100000008940697 s.
- Oracle verdict: not applicable.

### 113_native__write_after_read_cache
- In progress: 2026-10-12T15:47:16.018281-04:00
- Complete metadata: `logs/search_improvements/113_native__write_after_read_cache.json`; log: `logs/search_improvements/113_native__write_after_read_cache.log`; patch: `logs/search_improvements/113_native__write_after_read_cache.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --write-after-read-cache`
- Completed: 2026-10-12T15:47:16.066076-04:00; exit 0; runner elapsed 0.0470000000204891 s.
- Oracle verdict: not applicable.

### 113_native__incremental_propagators
- In progress: 2026-10-12T15:47:16.278514-04:00
- Complete metadata: `logs/search_improvements/113_native__incremental_propagators.json`; log: `logs/search_improvements/113_native__incremental_propagators.log`; patch: `logs/search_improvements/113_native__incremental_propagators.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --incremental-propagators`
- Completed: 2026-10-12T15:47:16.331171-04:00; exit 0; runner elapsed 0.0470000000204891 s.
- Oracle verdict: not applicable.

### 114_H20_decode
- In progress: 2026-10-12T15:47:19.596052-04:00
- Complete metadata: `logs/search_improvements/114_H20_decode.json`; log: `logs/search_improvements/114_H20_decode.log`; patch: `logs/search_improvements/114_H20_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T15:57:19.947377-04:00; exit 0; runner elapsed 600.3439999998081 s.
  `[SearchEngine volume] iterations=90541 elapsed=585.71482760000004 total-domain-bits=4851513.1637403192 remaining-domain-bits=2821900.1344686584 log2-remaining-fraction=-2029613.0292716608 eliminated-bits=2029613.0292716608 remaining-nodes=85370 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.265000000 budget=600.0`
  `[Program deadline] overshoot=0.0603485`
  `[SearchEngine] New best cost: 75.0097 at iteration 84246 event-epoch=1791834685.321758 program-deadline=1791835039.6545892`
- Target 2425756.5818701596; remaining/total 0.58165360769485952; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 245.667168856 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.
- R111 build H20 PASS, R112 oracle 26/26, R113 native tests PASS. R114 H20: eliminated 2029613.0292716608 (+70072.8145 vs O3), first incumbent 245.667168856 s at iteration 84246 => gate FAIL (> 231.18; O3 197-203). Costs unchanged, overshoot 0.0603485. Same failure mode as H14 (R078 249.5 s): fewer iterations to incumbent but slower per iteration. Not retained yet; next: TG_PROFILE diagnostics (H20 vs O3, 300 s, not evaluations) to locate the propagator cost of fixed source starts. Source also gained a diagnostic log naming the first non-downward-closed start (bucket 0 skips windows).

### 115_build_profile_H20
- In progress: 2026-10-12T15:57:39.146393-04:00
- Complete metadata: `logs/search_improvements/115_build_profile_H20.json`; log: `logs/search_improvements/115_build_profile_H20.log`; patch: `logs/search_improvements/115_build_profile_H20.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings --no-lint --log-level INFO --profile`
- Completed: 2026-10-12T15:58:57.443442-04:00; exit 0; runner elapsed 78.29700000002049 s.
- Oracle verdict: not applicable.

### 116_diag_profile_H20_300s
- In progress: 2026-10-12T15:59:01.313134-04:00
- Complete metadata: `logs/search_improvements/116_diag_profile_H20_300s.json`; log: `logs/search_improvements/116_diag_profile_H20_300s.log`; patch: `logs/search_improvements/116_diag_profile_H20_300s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 300`
- Completed: 2026-10-12T16:04:02.246856-04:00; exit 0; runner elapsed 300.9379999998491 s.
  `[SearchEngine volume] iterations=84826 elapsed=285.63525879999997 total-domain-bits=4851513.1637403192 remaining-domain-bits=2821900.1344686584 log2-remaining-fraction=-2029613.0292716608 eliminated-bits=2029613.0292716608 remaining-nodes=79655 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=300.859000000 budget=300.0`
  `[Program deadline] overshoot=0.435079`
  `[SearchEngine] New best cost: 75.0097 at iteration 84246 event-epoch=1791835410.7255616 program-deadline=1791835441.3607161`
- Target 2425756.5818701596; remaining/total 0.58165360769485952; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 269.364845514 s, logged first cost 75.0097, logged best cost 75.0097.
- Oracle verdict: not applicable.
- R115 profile build / R116 300 s TG_PROFILE DIAGNOSTIC (H20, not an evaluation): MemoryNoOverlap 81.4 s (33.3M calls, 0 domain changes), StartUnique 45.0 s, branch 31.6 s, StartBounds 18.7 s, InputPrune 14.3 s, propagation overhead 10.0 s, StartPrecedence 9.6 s, ConsumerStart 8.6 s. Bucket 0 first non-downward-closed start: start_0_3569=[1..7558] (sound root narrowing, not an entry restriction). H21 (pure speed, no pruning change): StaticPrivateLayout records certified spaces in SearchState::private_layout_spaces; MemoryNoOverlap returns immediately for START/OFFSET changes of allocations in those spaces (distinct non-alias allocations there never overlap, so every reuse check is vacuous; views only alias). Evaluated alone on O3 (H19 file restored). H20b (= H20 + entry-domain record SearchState::recordStartDomainShapeAtEntry called in solve before root propagation) saved at logs/search_improvements/H20b_start_rank_window.hpp for the next trial.

### 117_build_H21
- In progress: 2026-10-12T16:05:53.365221-04:00
- Complete metadata: `logs/search_improvements/117_build_H21.json`; log: `logs/search_improvements/117_build_H21.log`; patch: `logs/search_improvements/117_build_H21.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-12T16:06:50.778364-04:00; exit 0; runner elapsed 57.40599999995902 s.
- Oracle verdict: not applicable.

### 118_oracle_H21
- In progress: 2026-10-12T16:06:55.319326-04:00
- Complete metadata: `logs/search_improvements/118_oracle_H21.json`; log: `logs/search_improvements/118_oracle_H21.log`; patch: `logs/search_improvements/118_oracle_H21.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T16:06:57.672011-04:00; exit 0; runner elapsed 2.344000000040978 s.
- Oracle verdict: 26/26 cases passed.

### 119_native__private_layout
- In progress: 2026-10-12T16:06:57.930300-04:00
- Complete metadata: `logs/search_improvements/119_native__private_layout.json`; log: `logs/search_improvements/119_native__private_layout.log`; patch: `logs/search_improvements/119_native__private_layout.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --private-layout`
- Completed: 2026-10-12T16:06:58.017898-04:00; exit 0; runner elapsed 0.07899999991059303 s.
- Oracle verdict: not applicable.

### 119_native__write_after_read_cache
- In progress: 2026-10-12T16:06:58.242754-04:00
- Complete metadata: `logs/search_improvements/119_native__write_after_read_cache.json`; log: `logs/search_improvements/119_native__write_after_read_cache.log`; patch: `logs/search_improvements/119_native__write_after_read_cache.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --write-after-read-cache`
- Completed: 2026-10-12T16:06:58.290710-04:00; exit 0; runner elapsed 0.0470000000204891 s.
- Oracle verdict: not applicable.

### 119_native__incremental_propagators
- In progress: 2026-10-12T16:06:58.510441-04:00
- Complete metadata: `logs/search_improvements/119_native__incremental_propagators.json`; log: `logs/search_improvements/119_native__incremental_propagators.log`; patch: `logs/search_improvements/119_native__incremental_propagators.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --incremental-propagators`
- Completed: 2026-10-12T16:06:58.561570-04:00; exit 0; runner elapsed 0.04600000008940697 s.
- Oracle verdict: not applicable.

### 120_H21_decode
- In progress: 2026-10-12T16:07:01.468988-04:00
- Complete metadata: `logs/search_improvements/120_H21_decode.json`; log: `logs/search_improvements/120_H21_decode.log`; patch: `logs/search_improvements/120_H21_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T16:17:02.235474-04:00; exit 0; runner elapsed 600.7650000001304 s.
  `[SearchEngine volume] iterations=92803 elapsed=585.6156843 total-domain-bits=4851513.1637403192 remaining-domain-bits=2891972.9489357956 log2-remaining-fraction=-1959540.2148045236 eliminated-bits=1959540.2148045236 remaining-nodes=87340 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.688000000 budget=600.0`
  `[Program deadline] overshoot=0.0453484`
  `[SearchEngine] New best cost: 75.0097 at iteration 85093 event-epoch=1791835794.1848946 program-deadline=1791836221.5268843`
- Target 2425756.5818701596; remaining/total 0.59609710441478059; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 172.658010244 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.
- R117 build H21 PASS, R118 oracle 26/26, R119 native PASS. R120 H21: bits identical to O3 (1959540.2148045236), first incumbent 172.658010244 s (-24.2 s vs fastest O3), costs unchanged, overshoot 0.0453484. Promising; confirmations R121/R122.

### 121_H21_confirm1_decode
- In progress: 2026-10-12T16:17:16.554139-04:00
- Complete metadata: `logs/search_improvements/121_H21_confirm1_decode.json`; log: `logs/search_improvements/121_H21_confirm1_decode.log`; patch: `logs/search_improvements/121_H21_confirm1_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T16:27:16.850180-04:00; exit 0; runner elapsed 600.2970000000205 s.
  `[SearchEngine volume] iterations=92889 elapsed=585.56904640000005 total-domain-bits=4851513.1637403192 remaining-domain-bits=2891972.9489357956 log2-remaining-fraction=-1959540.2148045236 eliminated-bits=1959540.2148045236 remaining-nodes=87426 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.219000000 budget=600.0`
  `[Program deadline] overshoot=0.0321064`
  `[SearchEngine] New best cost: 75.0097 at iteration 85093 event-epoch=1791836405.7892637 program-deadline=1791836836.6045504`
- Target 2425756.5818701596; remaining/total 0.59609710441478059; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 169.184713364 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.

### 122_H21_confirm2_decode
- In progress: 2026-10-12T16:27:17.078455-04:00
- Complete metadata: `logs/search_improvements/122_H21_confirm2_decode.json`; log: `logs/search_improvements/122_H21_confirm2_decode.log`; patch: `logs/search_improvements/122_H21_confirm2_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T16:37:17.530747-04:00; exit 0; runner elapsed 600.438000000082 s.
  `[SearchEngine volume] iterations=92790 elapsed=585.61994130000005 total-domain-bits=4851513.1637403192 remaining-domain-bits=2891972.9489357956 log2-remaining-fraction=-1959540.2148045236 eliminated-bits=1959540.2148045236 remaining-nodes=87327 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.391000000 budget=600.0`
  `[Program deadline] overshoot=0.0415218`
  `[SearchEngine] New best cost: 75.0097 at iteration 85093 event-epoch=1791837009.4125185 program-deadline=1791837437.1316745`
- Target 2425756.5818701596; remaining/total 0.59609710441478059; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 172.280843973 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.
- O4 OPTIMIZATION retained: H21 on O3. R120/R121/R122 identical bits 1959540.2148045236, first incumbent 172.658010244/169.184713364/172.280843973 s (O3 196.9-202.9), costs unchanged, overshoots 0.0453484/0.0321064/0.0415218; oracle R118 26/26, native R119 PASS. Next: TG_PROFILE 150 s DIAGNOSTICS of O4 (R123/R124) and H20b+O4 (R125/R126) to find the selection-phase slowdown of the source prefix (R114 vs R108: +52 s between iterations 1k and 30k).

### 123_build_profile_O4
- In progress: 2026-10-12T16:37:28.310628-04:00
- Complete metadata: `logs/search_improvements/123_build_profile_O4.json`; log: `logs/search_improvements/123_build_profile_O4.log`; patch: `logs/search_improvements/123_build_profile_O4.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings --no-lint --log-level INFO --profile`
- Completed: 2026-10-12T16:38:45.956354-04:00; exit 0; runner elapsed 77.65599999995902 s.
- Oracle verdict: not applicable.

### 124_diag_profile_O4_150s
- In progress: 2026-10-12T16:38:46.186893-04:00
- Complete metadata: `logs/search_improvements/124_diag_profile_O4_150s.json`; log: `logs/search_improvements/124_diag_profile_O4_150s.log`; patch: `logs/search_improvements/124_diag_profile_O4_150s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 150`
- Completed: 2026-10-12T16:41:16.816391-04:00; exit 0; runner elapsed 150.64000000013039 s.
  `[SearchEngine volume] iterations=67752 elapsed=135.6339595 total-domain-bits=4851513.1637403192 remaining-domain-bits=2893129.5565821803 log2-remaining-fraction=-1958383.6071581389 eliminated-bits=1958383.6071581389 remaining-nodes=67706 exhausted=false incomplete-branches=3984`
  `[Program evaluation] elapsed=150.563000000 budget=150.0`
  `[Program deadline] overshoot=0.00196981`
- Target 2425756.5818701596; remaining/total 0.59633550583869699; target met: False.
- Incumbent: no incumbent within budget; time and both costs unavailable (+infinity).
- Oracle verdict: not applicable.

### 125_build_profile_H20b
- In progress: 2026-10-12T16:41:17.056474-04:00
- Complete metadata: `logs/search_improvements/125_build_profile_H20b.json`; log: `logs/search_improvements/125_build_profile_H20b.log`; patch: `logs/search_improvements/125_build_profile_H20b.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings --no-lint --log-level INFO --profile`
- Completed: 2026-10-12T16:41:17.605981-04:00; exit 0; runner elapsed 0.5470000000204891 s.
- Oracle verdict: not applicable.

### 126_diag_profile_H20b_150s
- In progress: 2026-10-12T16:41:17.825415-04:00
- Complete metadata: `logs/search_improvements/126_diag_profile_H20b_150s.json`; log: `logs/search_improvements/126_diag_profile_H20b_150s.log`; patch: `logs/search_improvements/126_diag_profile_H20b_150s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 150`
- Completed: 2026-10-12T16:43:48.056960-04:00; exit 0; runner elapsed 150.23499999986961 s.
  `[SearchEngine volume] iterations=66639 elapsed=135.67574250000001 total-domain-bits=4851513.1637403192 remaining-domain-bits=2893129.5565821803 log2-remaining-fraction=-1958383.6071581389 eliminated-bits=1958383.6071581389 remaining-nodes=66593 exhausted=false incomplete-branches=3652`
  `[Program evaluation] elapsed=150.156000000 budget=150.0`
  `[Program deadline] overshoot=0.000844479`
- Target 2425756.5818701596; remaining/total 0.59633550583869699; target met: False.
- Incumbent: no incumbent within budget; time and both costs unavailable (+infinity).
- Oracle verdict: not applicable.
- R125/R126 INVALID as H20b diagnostics: Copy-Item preserved the backup's old timestamp, so R125 did not recompile (R126 logs show H19-level windows); R126 is effectively a second O4 profile. Touched the file; redo as R127/R128. O4 profile (R124, 150 s): MemoryNoOverlap 55.4 s (still dominant: OPENCL work), branch 15.0, StartBounds 11.4, InputPrune 9.6, StartPrecedence 6.8, overhead 6.6, CacheFreshness 5.6 (one-time init), ConsumerStart 5.3.

### 127_build_profile_H20b
- In progress: 2026-10-12T16:44:17.585703-04:00
- Complete metadata: `logs/search_improvements/127_build_profile_H20b.json`; log: `logs/search_improvements/127_build_profile_H20b.log`; patch: `logs/search_improvements/127_build_profile_H20b.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings --no-lint --log-level INFO --profile`
- Completed: 2026-10-12T16:45:35.399109-04:00; exit 0; runner elapsed 77.81300000008196 s.
- Oracle verdict: not applicable.

### 128_diag_profile_H20b_150s
- In progress: 2026-10-12T16:45:41.368846-04:00
- Complete metadata: `logs/search_improvements/128_diag_profile_H20b_150s.json`; log: `logs/search_improvements/128_diag_profile_H20b_150s.log`; patch: `logs/search_improvements/128_diag_profile_H20b_150s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 150`
- Completed: 2026-10-12T16:48:11.889090-04:00; exit 0; runner elapsed 150.51599999982864 s.
  `[SearchEngine volume] iterations=32026 elapsed=135.41258389999999 total-domain-bits=4851513.1637403192 remaining-domain-bits=2818657.0700338297 log2-remaining-fraction=-2032856.0937064895 eliminated-bits=2032856.0937064895 remaining-nodes=31980 exhausted=false incomplete-branches=332`
  `[Program evaluation] elapsed=150.453000000 budget=150.0`
  `[Program deadline] overshoot=0.00324798`
- Target 2425756.5818701596; remaining/total 0.58098514317144712; target met: False.
- Incumbent: no incumbent within budget; time and both costs unavailable (+infinity).
- Oracle verdict: not applicable.
- R127 profile build H20b (real rebuild), R128 150 s DIAGNOSTIC: source prefix active in all 17 buckets (bucket 0 too, thanks to the entry-domain record: start bits 97390 -> 92184.4); StartUnique 44.7 s (O4: <1 s): fixed source starts become FIXED_START_POSITIVE during the selection phase and every call rebuilt the active-start lists of all 17 buckets (global selection revision changes each iteration). MemoryNoOverlap global/initial path: 638 calls 40.8 s in O4 (CACHED-change GLOBAL rescans include CPP). H22 (speed): StartUnique keeps per-bucket active lists/revisions, rebuilding only the bucket it scans; identical removals. H23 (speed): allocations in certified private spaces are excluded from the fixed-offset index (rebuild and incremental insert), the WAR global scan and the FixedOffsetStart global scan (queries for such allocations already return early per H21; pairs are only compared within one space). Evaluating H22+H23 on O4 first (H19 window file restored and touched).

### 129_build_H22_H23
- In progress: 2026-10-12T16:49:30.816014-04:00
- Complete metadata: `logs/search_improvements/129_build_H22_H23.json`; log: `logs/search_improvements/129_build_H22_H23.log`; patch: `logs/search_improvements/129_build_H22_H23.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-12T16:50:27.558178-04:00; exit 0; runner elapsed 56.75 s.
- Oracle verdict: not applicable.

### 130_oracle_H22_H23
- In progress: 2026-10-12T16:50:27.834796-04:00
- Complete metadata: `logs/search_improvements/130_oracle_H22_H23.json`; log: `logs/search_improvements/130_oracle_H22_H23.log`; patch: `logs/search_improvements/130_oracle_H22_H23.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T16:50:30.192038-04:00; exit 0; runner elapsed 2.344000000040978 s.
- Oracle verdict: 26/26 cases passed.

### 131_native__private_layout
- In progress: 2026-10-12T16:50:30.429134-04:00
- Complete metadata: `logs/search_improvements/131_native__private_layout.json`; log: `logs/search_improvements/131_native__private_layout.log`; patch: `logs/search_improvements/131_native__private_layout.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --private-layout`
- Completed: 2026-10-12T16:50:30.515080-04:00; exit 0; runner elapsed 0.0779999999795109 s.
- Oracle verdict: not applicable.

### 131_native__write_after_read_cache
- In progress: 2026-10-12T16:50:30.746950-04:00
- Complete metadata: `logs/search_improvements/131_native__write_after_read_cache.json`; log: `logs/search_improvements/131_native__write_after_read_cache.log`; patch: `logs/search_improvements/131_native__write_after_read_cache.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --write-after-read-cache`
- Completed: 2026-10-12T16:50:30.795297-04:00; exit 0; runner elapsed 0.04699999978765845 s.
- Oracle verdict: not applicable.

### 131_native__incremental_propagators
- In progress: 2026-10-12T16:50:31.011286-04:00
- Complete metadata: `logs/search_improvements/131_native__incremental_propagators.json`; log: `logs/search_improvements/131_native__incremental_propagators.log`; patch: `logs/search_improvements/131_native__incremental_propagators.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --incremental-propagators`
- Completed: 2026-10-12T16:50:31.062776-04:00; exit 0; runner elapsed 0.0470000000204891 s.
- Oracle verdict: not applicable.

### 132_H22_H23_decode
- In progress: 2026-10-12T16:50:34.492953-04:00
- Complete metadata: `logs/search_improvements/132_H22_H23_decode.json`; log: `logs/search_improvements/132_H22_H23_decode.log`; patch: `logs/search_improvements/132_H22_H23_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T17:00:34.797677-04:00; exit 0; runner elapsed 600.2969999997877 s.
  `[SearchEngine volume] iterations=92802 elapsed=585.48327429999995 total-domain-bits=4851513.1637403192 remaining-domain-bits=2891972.9489357956 log2-remaining-fraction=-1959540.2148045236 eliminated-bits=1959540.2148045236 remaining-nodes=87339 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.219000000 budget=600.0`
  `[Program deadline] overshoot=0.0344772`
  `[SearchEngine] New best cost: 75.0097 at iteration 85093 event-epoch=1791838384.8340604 program-deadline=1791838834.5562315`
- Target 2425756.5818701596; remaining/total 0.59609710441478059; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 150.277828932 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.
- R129 build PASS, R130 oracle 26/26, R131 native PASS. R132 H22+H23: bits identical (1959540.2148045236), first incumbent 150.277828932 s (O4 169.2-172.7), costs unchanged, overshoot 0.0344772. Confirmations R133/R134.

### 133_H22_H23_confirm_decode
- In progress: 2026-10-12T17:00:43.677892-04:00
- Complete metadata: `logs/search_improvements/133_H22_H23_confirm_decode.json`; log: `logs/search_improvements/133_H22_H23_confirm_decode.log`; patch: `logs/search_improvements/133_H22_H23_confirm_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T17:10:44.475172-04:00; exit 0; runner elapsed 600.7810000001919 s.
  `[SearchEngine volume] iterations=92919 elapsed=585.64608950000002 total-domain-bits=4851513.1637403192 remaining-domain-bits=2891972.9489357956 log2-remaining-fraction=-1959540.2148045236 eliminated-bits=1959540.2148045236 remaining-nodes=87456 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.735000000 budget=600.0`
  `[Program deadline] overshoot=0.0532975`
  `[SearchEngine] New best cost: 75.0097 at iteration 85093 event-epoch=1791838990.8358829 program-deadline=1791839443.7141564`
- Target 2425756.5818701596; remaining/total 0.59609710441478059; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 147.121726513 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.

### 134_H22_H23_confirm_decode
- In progress: 2026-10-12T17:10:44.692453-04:00
- Complete metadata: `logs/search_improvements/134_H22_H23_confirm_decode.json`; log: `logs/search_improvements/134_H22_H23_confirm_decode.log`; patch: `logs/search_improvements/134_H22_H23_confirm_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T17:20:45.201439-04:00; exit 0; runner elapsed 600.5 s.
  `[SearchEngine volume] iterations=92912 elapsed=585.4933393 total-domain-bits=4851513.1637403192 remaining-domain-bits=2891972.9489357956 log2-remaining-fraction=-1959540.2148045236 eliminated-bits=1959540.2148045236 remaining-nodes=87449 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.438000000 budget=600.0`
  `[Program deadline] overshoot=0.240171`
  `[SearchEngine] New best cost: 75.0097 at iteration 85093 event-epoch=1791839592.3198316 program-deadline=1791840044.7456603`
- Target 2425756.5818701596; remaining/total 0.59609710441478059; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 147.574171305 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.
- O5 OPTIMIZATION retained: H22+H23 on O4. R132/R133/R134 identical bits 1959540.2148045236, first incumbent 150.277828932/147.121726513/147.574171305 s (O4 169.2-172.7), costs unchanged, overshoots 0.0344772/0.0532975/0.240171; oracle R130 26/26, native R131 PASS. Next H20c = H20b window file (source prefix + entry-domain check) on O5.

### 135_build_H20c
- In progress: 2026-10-12T17:21:01.503607-04:00
- Complete metadata: `logs/search_improvements/135_build_H20c.json`; log: `logs/search_improvements/135_build_H20c.log`; patch: `logs/search_improvements/135_build_H20c.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-12T17:21:59.111099-04:00; exit 0; runner elapsed 57.60900000017136 s.
- Oracle verdict: not applicable.

### 136_oracle_H20c
- In progress: 2026-10-12T17:21:59.392918-04:00
- Complete metadata: `logs/search_improvements/136_oracle_H20c.json`; log: `logs/search_improvements/136_oracle_H20c.log`; patch: `logs/search_improvements/136_oracle_H20c.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T17:22:01.736622-04:00; exit 0; runner elapsed 2.343000000109896 s.
- Oracle verdict: 26/26 cases passed.

### 137_native__private_layout
- In progress: 2026-10-12T17:22:01.974944-04:00
- Complete metadata: `logs/search_improvements/137_native__private_layout.json`; log: `logs/search_improvements/137_native__private_layout.log`; patch: `logs/search_improvements/137_native__private_layout.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --private-layout`
- Completed: 2026-10-12T17:22:02.058995-04:00; exit 0; runner elapsed 0.0779999999795109 s.
- Oracle verdict: not applicable.

### 137_native__write_after_read_cache
- In progress: 2026-10-12T17:22:02.276173-04:00
- Complete metadata: `logs/search_improvements/137_native__write_after_read_cache.json`; log: `logs/search_improvements/137_native__write_after_read_cache.log`; patch: `logs/search_improvements/137_native__write_after_read_cache.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --write-after-read-cache`
- Completed: 2026-10-12T17:22:02.669109-04:00; exit 0; runner elapsed 0.39000000013038516 s.
- Oracle verdict: not applicable.

### 137_native__incremental_propagators
- In progress: 2026-10-12T17:22:02.883889-04:00
- Complete metadata: `logs/search_improvements/137_native__incremental_propagators.json`; log: `logs/search_improvements/137_native__incremental_propagators.log`; patch: `logs/search_improvements/137_native__incremental_propagators.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --incremental-propagators`
- Completed: 2026-10-12T17:22:02.933499-04:00; exit 0; runner elapsed 0.0470000000204891 s.
- Oracle verdict: not applicable.

### 138_H20c_decode
- In progress: 2026-10-12T17:22:06.798954-04:00
- Complete metadata: `logs/search_improvements/138_H20c_decode.json`; log: `logs/search_improvements/138_H20c_decode.log`; patch: `logs/search_improvements/138_H20c_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T17:32:07.104583-04:00; exit 0; runner elapsed 600.280999999959 s.
  `[SearchEngine volume] iterations=91897 elapsed=585.58020039999997 total-domain-bits=4851513.1637403192 remaining-domain-bits=2817502.5194660774 log2-remaining-fraction=-2034010.6442742418 eliminated-bits=2034010.6442742418 remaining-nodes=86726 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.204000000 budget=600.0`
  `[Program deadline] overshoot=0.0329556`
  `[SearchEngine] New best cost: 75.0097 at iteration 84209 event-epoch=1791840278.3652029 program-deadline=1791840726.8586617`
- Target 2425756.5818701596; remaining/total 0.58074716575516772; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 151.506541252 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.
- R135 build H20c PASS, R136 oracle 26/26, R137 native PASS. R138 H20c: eliminated 2034010.6442742418 (+74470.4295 vs O5), first incumbent 151.506541252 s (O5 147.1-150.3; original gate 231.18), costs unchanged, overshoot 0.0329556. Confirmations R139/R140.

### 139_H20c_confirm_decode
- In progress: 2026-10-12T17:32:22.574416-04:00
- Complete metadata: `logs/search_improvements/139_H20c_confirm_decode.json`; log: `logs/search_improvements/139_H20c_confirm_decode.log`; patch: `logs/search_improvements/139_H20c_confirm_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T17:42:22.856632-04:00; exit 0; runner elapsed 600.280999999959 s.
  `[SearchEngine volume] iterations=91953 elapsed=585.57496819999994 total-domain-bits=4851513.1637403192 remaining-domain-bits=2817502.5194660774 log2-remaining-fraction=-2034010.6442742418 eliminated-bits=2034010.6442742418 remaining-nodes=86782 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.204000000 budget=600.0`
  `[Program deadline] overshoot=0.000706911`
  `[SearchEngine] New best cost: 75.0097 at iteration 84209 event-epoch=1791840890.570708 program-deadline=1791841342.6354482`
- Target 2425756.5818701596; remaining/total 0.58074716575516772; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 147.935259819 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.

### 140_H20c_confirm_decode
- In progress: 2026-10-12T17:42:23.080218-04:00
- Complete metadata: `logs/search_improvements/140_H20c_confirm_decode.json`; log: `logs/search_improvements/140_H20c_confirm_decode.log`; patch: `logs/search_improvements/140_H20c_confirm_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T17:52:23.634222-04:00; exit 0; runner elapsed 600.530999999959 s.
  `[SearchEngine volume] iterations=91932 elapsed=585.58194270000001 total-domain-bits=4851513.1637403192 remaining-domain-bits=2817502.5194660774 log2-remaining-fraction=-2034010.6442742418 eliminated-bits=2034010.6442742418 remaining-nodes=86761 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.469000000 budget=600.0`
  `[Program deadline] overshoot=0.0326686`
  `[SearchEngine] New best cost: 75.0097 at iteration 84209 event-epoch=1791841492.8542275 program-deadline=1791841943.1325941`
- Target 2425756.5818701596; remaining/total 0.58074716575516772; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 149.721633434 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.
- O6 OPTIMIZATION retained: H20c on O5. R138/R139/R140 identical eliminated 2034010.6442742418 (+74470.43 vs O5; +2031320.54 vs R002), first incumbent 151.506541252/147.935259819/149.721633434 s (O5 147.1-150.3; within original gate), costs unchanged, overshoots 0.0329556/0.000706911/0.0326686; oracle R136 26/26, native R137 PASS. Remaining gap to target 391746 bits. Next: TEMP DIAG D9 (root box bits by type/selection status/space/cacheable/bucket16), 60 s, then revert.

### 141_build_diag_D9
- In progress: 2026-10-12T17:52:47.145438-04:00
- Complete metadata: `logs/search_improvements/141_build_diag_D9.json`; log: `logs/search_improvements/141_build_diag_D9.log`; patch: `logs/search_improvements/141_build_diag_D9.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings --no-lint --log-level INFO`
- Completed: 2026-10-12T17:53:40.255909-04:00; exit 0; runner elapsed 53.10899999993853 s.
- Oracle verdict: not applicable.

### 142_diag_D9_60s
- In progress: 2026-10-12T17:53:40.512436-04:00
- Complete metadata: `logs/search_improvements/142_diag_D9_60s.json`; log: `logs/search_improvements/142_diag_D9_60s.log`; patch: `logs/search_improvements/142_diag_D9_60s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 60`
- Completed: 2026-10-12T17:54:40.963105-04:00; exit 0; runner elapsed 60.45300000021234 s.
  `[SearchEngine volume] iterations=26888 elapsed=45.625855000000001 total-domain-bits=4851513.1637403192 remaining-domain-bits=2818657.0700338297 log2-remaining-fraction=-2032856.0937064895 eliminated-bits=2032856.0937064895 remaining-nodes=26858 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=60.375000000 budget=60.0`
  `[Program deadline] overshoot=0.00029707`
- Target 2425756.5818701596; remaining/total 0.58098514317144712; target met: False.
- Incumbent: no incumbent within budget; time and both costs unavailable (+infinity).
- Oracle verdict: not applicable.

### 143_build_O6_clean
- In progress: 2026-10-12T17:55:53.301186-04:00
- Complete metadata: `logs/search_improvements/143_build_O6_clean.json`; log: `logs/search_improvements/143_build_O6_clean.log`; patch: `logs/search_improvements/143_build_O6_clean.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-12T17:56:50.669229-04:00; exit 0; runner elapsed 57.35900000017136 s.
- Oracle verdict: not applicable.

### 144_oracle_O6_clean
- In progress: 2026-10-12T17:56:50.944564-04:00
- Complete metadata: `logs/search_improvements/144_oracle_O6_clean.json`; log: `logs/search_improvements/144_oracle_O6_clean.log`; patch: `logs/search_improvements/144_oracle_O6_clean.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T17:56:53.316047-04:00; exit 0; runner elapsed 2.375 s.
- Oracle verdict: 26/26 cases passed.
- R141 build / R142 60 s DIAGNOSTIC D9 (temporary root-box breakdown, reverted in R143): after root propagation only the 17 bucket roots are mandatory; every other class is optional (the root e-class has ~5 alternatives routing through different device paths). Root bits: START ~1.567M (CPP 625k+39k b16, CPP cacheable 183k+11k, OPENCL 527k+33k, OPENCL cacheable 139k+9k; STORAGE starts 0 = sources prefix), OPENCL offsets 1.059M (plain 791k+49k b16, cacheable 206k+13k), CPP offsets 14k (unresolved views), SELECTED 177k, CACHED 1.8k. Bucket 16 (weight 0) ~164k.
- R143 clean O6 rebuild (diagnostic removed), R144 oracle 26/26 PASS. Extension == O6 source.

### Session 4 comparison (timing origin: evaluation start, reconstructed from native event epoch and supervisor deadline)
| Run | Implementation | Eliminated / remaining bits | First incumbent s | First / deadline cost | Gate |
|---|---|---:|---:|---:|---|
| R002 / R046 | original baseline | 2690.1009323587641 / 4848823.0628079604 | R046 230.178876638 | 75.0097 / 75.009705 | permanent anchor (limit 231.18 s) |
| R104 | O2 = O1 + H17 repair | 1945410.9553037602 / 2906102.208436559 | 219.044572115 | 75.0097 / 75.009705 | retained (correctness repair) |
| R108/R109/R110 | O3 = O2 + H19 rank windows | 1959540.2148045236 / 2891972.9489357956 | 196.898 / 197.047 / 202.939 | unchanged | retained |
| R114 | H20 source prefix (O3) | 2029613.0292716608 / 2821900.1344686584 | 245.667168856 | unchanged | FAIL timing (StartUnique rebuilds) |
| R120/R121/R122 | O4 = O3 + H21 private-space skip | 1959540.2148045236 | 172.658 / 169.185 / 172.281 | unchanged | retained (speed) |
| R132/R133/R134 | O5 = O4 + H22 + H23 | 1959540.2148045236 | 150.278 / 147.122 / 147.574 | unchanged | retained (speed) |
| R138/R139/R140 | O6 = O5 + H20c source prefix | 2034010.6442742418 / 2817502.5194660774 | 151.507 / 147.935 / 149.722 | unchanged | retained (+74470.43 bits) |

All overshoots <= 0.27 s. Target 2425756.5818701596 eliminated bits; O6 reaches 83.85% of it (gap 391745.94). Deltas vs R046: +2031320.54 bits, first incumbent -78.7 to -82.2 s, costs 0.

### Remaining-gap analysis (session 4)
- START under rank compression: for a class c the representable window is at least N - minbefore(c) - minafter(c), and minbefore+minafter+1 <= the smallest plan containing c (a full plan has ~2870-6958 of N=7559 classes). Even exact minimum-extraction counts give at most ~log2(7559/4700) = 0.69 bit per START variable (~90k bits total); the measured necessary sets are far smaller because every class is optional. Not enough for 391k.
- Bucket 16 (weight 0) holds ~164k bits but is coupled to the shared cache set (CacheFreshness forces every cached base to be selected in every bucket), so it cannot be fixed at the root; even fully fixed it would not close the gap.
- OPENCL offsets (1.059M) are the only reservoir large enough. A private layout needs cacheable slots (1.58M pages) + plain temporaries (53,866 pages/bucket) > 524,288 pages. Any root layout certificate for plain temporaries must leave room for an arbitrary cache set, and the cache budget equals the full capacity (feasible plans can cache up to cap minus a few activation pages). The canonical forms examined (caches packed at the bottom per CachedOffsetAllocation or collapsed to the top, temporaries left-justified, mutual-exclusion slot sharing H10) all leave root offset windows unchanged because the cached total is undecided at the root. Search-time closure of near-root right siblings would need LB >= incumbent (root LB 0.53 / LP 10.9 per bucket vs 75.0).
- Conclusion: no sound candidate found that reaches the target under the fixed encoding and evaluation contract; the work does not prove impossibility. Possible future directions: exact minimum-extraction START bounds (<= ~90k), a cost-based cache-set dominance that bounds the OPENCL cached total below cap - P (would enable an OPENCL private layout), stronger LBs.

## Resume checkpoint (session 4, current)
- Retained implementation O6 (uncommitted working tree, staged files plus new `tensor_graphs_cpp/core/plan/propagators/start_rank_window.hpp` and oracle cases 25/26): O1 + H17 + H19/H20c StartRankWindow + H21/H23 private-space memory skip + H22 per-bucket StartUnique. Exact source: `logs/search_improvements/144_oracle_O6_clean.patch`. Extension built from it (R143). No running processes.
- Next concrete step: minimum-extraction lower bound for START (replace |ND| with a min-size sub-plan count), expected <= ~90k bits; or research a cache-set dominance for OPENCL.

### 145_build_diag_D10
- In progress: 2026-10-12T17:59:16.904558-04:00
- Complete metadata: `logs/search_improvements/145_build_diag_D10.json`; log: `logs/search_improvements/145_build_diag_D10.log`; patch: `logs/search_improvements/145_build_diag_D10.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings --no-lint --log-level INFO`
- Completed: 2026-10-12T18:00:09.009416-04:00; exit 0; runner elapsed 52.10899999993853 s.
- Oracle verdict: not applicable.

### 146_diag_D10_200s
- In progress: 2026-10-12T18:00:12.629521-04:00
- Complete metadata: `logs/search_improvements/146_diag_D10_200s.json`; log: `logs/search_improvements/146_diag_D10_200s.log`; patch: `logs/search_improvements/146_diag_D10_200s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 200`
- Completed: 2026-10-12T18:03:32.921226-04:00; exit 0; runner elapsed 200.28100000019185 s.
  `[SearchEngine volume] iterations=85488 elapsed=185.74000040000001 total-domain-bits=4851513.1637403192 remaining-domain-bits=2817502.5194660774 log2-remaining-fraction=-2034010.6442742418 eliminated-bits=2034010.6442742418 remaining-nodes=80317 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=200.219000000 budget=200.0`
  `[Program deadline] overshoot=0.026854`
  `[SearchEngine] New best cost: 75.0097 at iteration 84209 event-epoch=1791842559.327399 program-deadline=1791842612.6809568`
- Target 2425756.5818701596; remaining/total 0.58074716575516772; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 146.646442175 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.
- R145 build / R146 200 s DIAGNOSTIC D10 (temporary, removed): OPENCL cacheable classes are {CACHE, COPY_TO from CPP} (403 classes, 919,634 pages/bucket) or {CACHE, FUSED} (162, 656,742 pages); plain OPENCL: FUSED x2 (1117, 34,598 pages), COPY_TO from CPP (1231, 16,126 pages). First incumbent caches 82 OPENCL bases (54,481 pages) and 173 CPP bases; its OPENCL temporaries end at page 54,577. H24 (candidate): StartRankWindow counts disjoint class GROUPS instead of classes; groups = union-find over COPY_TO edges of remaining alternatives (same tensor in different memory spaces). Soundness: for any partition, ND groups each hold an active member strictly before c, distinct classes because groups are disjoint (similarly NA after c); source prefix counts only groups without source members. O6 window file backed up at logs/search_improvements/O6_start_rank_window.hpp.

### 147_build_H24
- In progress: 2026-10-12T18:05:23.971957-04:00
- Complete metadata: `logs/search_improvements/147_build_H24.json`; log: `logs/search_improvements/147_build_H24.log`; patch: `logs/search_improvements/147_build_H24.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-12T18:06:19.910804-04:00; exit 0; runner elapsed 55.937999999849126 s.
- Oracle verdict: not applicable.

### 148_diag_H24_60s
- In progress: 2026-10-12T18:06:20.263182-04:00
- Complete metadata: `logs/search_improvements/148_diag_H24_60s.json`; log: `logs/search_improvements/148_diag_H24_60s.log`; patch: `logs/search_improvements/148_diag_H24_60s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 60`
- Completed: 2026-10-12T18:07:20.455127-04:00; exit 0; runner elapsed 60.188000000081956 s.
  `[SearchEngine volume] iterations=28051 elapsed=45.645602699999998 total-domain-bits=4851513.1637403192 remaining-domain-bits=2810748.2149112178 log2-remaining-fraction=-2040764.9488291014 eliminated-bits=2040764.9488291014 remaining-nodes=28005 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=60.109000000 budget=60.0`
  `[Program deadline] overshoot=0.0102093`
- Target 2425756.5818701596; remaining/total 0.57935495999855136; target met: False.
- Incumbent: no incumbent within budget; time and both costs unavailable (+infinity).
- Oracle verdict: not applicable.

### 149_oracle_H24
- In progress: 2026-10-12T18:07:58.796421-04:00
- Complete metadata: `logs/search_improvements/149_oracle_H24.json`; log: `logs/search_improvements/149_oracle_H24.log`; patch: `logs/search_improvements/149_oracle_H24.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T18:08:01.235889-04:00; exit 0; runner elapsed 2.4379999998491257 s.
- Oracle verdict: 26/26 cases passed.

### 150_native__private_layout
- In progress: 2026-10-12T18:08:01.465438-04:00
- Complete metadata: `logs/search_improvements/150_native__private_layout.json`; log: `logs/search_improvements/150_native__private_layout.log`; patch: `logs/search_improvements/150_native__private_layout.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --private-layout`
- Completed: 2026-10-12T18:08:01.540139-04:00; exit 0; runner elapsed 0.0779999999795109 s.
- Oracle verdict: not applicable.

### 150_native__write_after_read_cache
- In progress: 2026-10-12T18:08:01.774822-04:00
- Complete metadata: `logs/search_improvements/150_native__write_after_read_cache.json`; log: `logs/search_improvements/150_native__write_after_read_cache.log`; patch: `logs/search_improvements/150_native__write_after_read_cache.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --write-after-read-cache`
- Completed: 2026-10-12T18:08:01.822235-04:00; exit 0; runner elapsed 0.0470000000204891 s.
- Oracle verdict: not applicable.

### 150_native__incremental_propagators
- In progress: 2026-10-12T18:08:02.064886-04:00
- Complete metadata: `logs/search_improvements/150_native__incremental_propagators.json`; log: `logs/search_improvements/150_native__incremental_propagators.log`; patch: `logs/search_improvements/150_native__incremental_propagators.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --incremental-propagators`
- Completed: 2026-10-12T18:08:02.117956-04:00; exit 0; runner elapsed 0.06300000008195639 s.
- Oracle verdict: not applicable.

### 151_H24_decode
- In progress: 2026-10-12T18:08:07.887480-04:00
- Complete metadata: `logs/search_improvements/151_H24_decode.json`; log: `logs/search_improvements/151_H24_decode.log`; patch: `logs/search_improvements/151_H24_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`

### 152_build_test_H24
- In progress: 2026-10-12T18:08:58.824558-04:00
- Complete metadata: `logs/search_improvements/152_build_test_H24.json`; log: `logs/search_improvements/152_build_test_H24.log`; patch: `logs/search_improvements/152_build_test_H24.patch`
- Command: `.venv/Scripts/python.exe build.py --targets test --no-lint --log-level INFO`
- Completed: 2026-10-12T18:09:49.321178-04:00; exit 0; runner elapsed 50.5 s.
- Oracle verdict: not applicable.

### 153_native_start_rank_window_H24
- In progress: 2026-10-12T18:09:49.678387-04:00
- Complete metadata: `logs/search_improvements/153_native_start_rank_window_H24.json`; log: `logs/search_improvements/153_native_start_rank_window_H24.log`; patch: `logs/search_improvements/153_native_start_rank_window_H24.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --start-rank-window`
- Completed: 2026-10-12T18:09:49.761324-04:00; exit 0; runner elapsed 0.0779999999795109 s.
- Oracle verdict: not applicable.
- Completed: 2026-10-12T18:18:08.176982-04:00; exit 0; runner elapsed 600.2810000001919 s.
  `[SearchEngine volume] iterations=92073 elapsed=585.67039850000003 total-domain-bits=4851513.1637403192 remaining-domain-bits=2809595.6766788443 log2-remaining-fraction=-2041917.4870614749 eliminated-bits=2041917.4870614749 remaining-nodes=86902 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.203000000 budget=600.0`
  `[Program deadline] overshoot=0.0430758`
  `[SearchEngine] New best cost: 75.0097 at iteration 84209 event-epoch=1791843032.534013 program-deadline=1791843487.9465737`
- Target 2425756.5818701596; remaining/total 0.57911739736737322; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 144.587439299 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.
- R148 60 s DIAGNOSTIC H24: start bits per bucket 92184.4 -> 91716.5 (bucket 16 91844.4 -> 91425.3). R149 oracle 26/26 (oracle schema has no COPY_TO, so grouping is not exercised there); R150 native PASS; new native regression tests/start_rank_window_regression.hpp (--start-rank-window, R152 build, R153 PASS): root reaches x_cpp or its COPY_TO peer x_ocl; group-based window gives root min 2 (class-based gives 1) and contains all three canonical witnesses. R151 H24: eliminated 2041917.4870614749 (+7906.8428 vs O6), first incumbent 144.587439299 s, costs unchanged, overshoot 0.0430758. Confirmations R154/R155.

### 154_H24_confirm_decode
- In progress: 2026-10-12T18:20:54.117816-04:00
- Complete metadata: `logs/search_improvements/154_H24_confirm_decode.json`; log: `logs/search_improvements/154_H24_confirm_decode.log`; patch: `logs/search_improvements/154_H24_confirm_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T18:30:54.364301-04:00; exit 0; runner elapsed 600.25 s.
  `[SearchEngine volume] iterations=92266 elapsed=585.7929388 total-domain-bits=4851513.1637403192 remaining-domain-bits=2809595.6766788443 log2-remaining-fraction=-2041917.4870614749 eliminated-bits=2041917.4870614749 remaining-nodes=87095 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.172000000 budget=600.0`
  `[Program deadline] overshoot=0.035394`
  `[SearchEngine] New best cost: 75.0097 at iteration 84209 event-epoch=1791843789.6621237 program-deadline=1791844254.1716998`
- Target 2425756.5818701596; remaining/total 0.57911739736737322; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 135.490423918 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.

### 155_H24_confirm_decode
- In progress: 2026-10-12T18:30:54.585356-04:00
- Complete metadata: `logs/search_improvements/155_H24_confirm_decode.json`; log: `logs/search_improvements/155_H24_confirm_decode.log`; patch: `logs/search_improvements/155_H24_confirm_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T18:40:55.236575-04:00; exit 0; runner elapsed 600.6400000001304 s.
  `[SearchEngine volume] iterations=92049 elapsed=585.78448119999996 total-domain-bits=4851513.1637403192 remaining-domain-bits=2809595.6766788443 log2-remaining-fraction=-2041917.4870614749 eliminated-bits=2041917.4870614749 remaining-nodes=86878 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.578000000 budget=600.0`
  `[Program deadline] overshoot=6.65188e-05`
  `[SearchEngine] New best cost: 75.0097 at iteration 84209 event-epoch=1791844389.1835928 program-deadline=1791844854.6325967`
- Target 2425756.5818701596; remaining/total 0.57911739736737322; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 134.550996065 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.
- O7 OPTIMIZATION retained: H24 on O6. R151/R154/R155 identical eliminated 2041917.4870614749 (+7906.84 vs O6), first incumbent 144.587439299/135.490423918/134.550996065 s, costs 75.0097/75.009705 unchanged, overshoots 0.0430758/0.035394/6.65188e-05; oracle R149 26/26, native R150 + R153 PASS. Gap to target: 383839.09 bits.

## Resume checkpoint (session 4, after O7)
- Retained implementation O7 = O6 + H24 (group-based rank windows). Exact source: `logs/search_improvements/155_H24_confirm_decode.patch` (includes untracked start_rank_window.hpp and tests/start_rank_window_regression.hpp). Extension built from it (R147), test.exe from R152. No running processes.
- Best measurement: eliminated 2041917.4870614749 / remaining 2809595.6766788443 (ratio 0.57911), target 2425756.58 not met (gap 383839.09). First incumbent 134.6-144.6 s (original 230.18 s), costs unchanged.

### Concrete blocker (why no further candidate is pending)
Post-root box (D9, before H24): START ~1.567M, OPENCL offsets 1.059M, SELECTED 0.177M, CPP view offsets 0.014M. Only the 17 roots are mandatory.
1. START: rank windows are at least N - minbefore - minafter; the smallest complete plan has ~2870 classes, so even perfect necessity counting caps START savings near 0.7 bit/variable (~90k bits). H19/H20c/H24 already took ~82k of the START reductions; the class/group necessary sets measured are small because FUSED alternatives legitimately skip intermediates.
2. OPENCL offsets (the only reservoir larger than the gap): every canonical layout argument (private slots, cache packing bottom/top, left-justification, exclusive-slot sharing) places temporaries relative to the cached OPENCL total C, which ranges from 0 to the full capacity at the root (CACHE alternatives are never slower than their COPY_TO/FUSED alternatives, so no cost argument bounds C). With interval domains, a root window valid for every C is the full range. The incumbent itself caches 54,481 OPENCL pages, but no sound rule can assume that.
3. SELECTED/bucket 16: everything is optional at the root; bucket 16 (weight 0) is tied to the shared cache set by CacheFreshness, so it cannot be canonicalized independently.
4. Frontier: near-root right siblings keep ~root volume; closing them by bound needs LB >= incumbent (root LB 0.53, LP 10.9 vs 75.0).
Reaching the target therefore appears to require changing the encoding or metric (e.g. not counting irrelevant coordinates of unselected classes, or conditional/cache-relative offset variables), which the contract forbids without separate review and a new baseline.
- H25 (candidate): StaticPrivateLayout narrows the offset of each view-only class with several candidate base addresses (not fixable) to the hull [min, max] of those addresses (a selected view shares one of them; all were already checked to lie in its domain). UnselectedStartOffset now fixes an unselected offset to the preallocation end only if it is in the domain, else to the domain minimum. Expected <= 14k bits (CPP offsets left after H6).

### 156_build_H25
- In progress: 2026-10-12T18:42:59.761033-04:00
- Complete metadata: `logs/search_improvements/156_build_H25.json`; log: `logs/search_improvements/156_build_H25.log`; patch: `logs/search_improvements/156_build_H25.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-12T18:43:59.167042-04:00; exit 0; runner elapsed 59.40599999995902 s.
- Oracle verdict: not applicable.

### 157_oracle_H25
- In progress: 2026-10-12T18:43:59.516851-04:00
- Complete metadata: `logs/search_improvements/157_oracle_H25.json`; log: `logs/search_improvements/157_oracle_H25.log`; patch: `logs/search_improvements/157_oracle_H25.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T18:44:02.158262-04:00; exit 0; runner elapsed 2.6399999998975545 s.
- Oracle verdict: 26/26 cases passed.

### 158_native__private_layout
- In progress: 2026-10-12T18:44:02.400445-04:00
- Complete metadata: `logs/search_improvements/158_native__private_layout.json`; log: `logs/search_improvements/158_native__private_layout.log`; patch: `logs/search_improvements/158_native__private_layout.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --private-layout`
- Completed: 2026-10-12T18:44:02.637166-04:00; exit 0; runner elapsed 0.23500000010244548 s.
- Oracle verdict: not applicable.

### 158_native__write_after_read_cache
- In progress: 2026-10-12T18:44:02.851291-04:00
- Complete metadata: `logs/search_improvements/158_native__write_after_read_cache.json`; log: `logs/search_improvements/158_native__write_after_read_cache.log`; patch: `logs/search_improvements/158_native__write_after_read_cache.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --write-after-read-cache`
- Completed: 2026-10-12T18:44:02.908114-04:00; exit 0; runner elapsed 0.04699999978765845 s.
- Oracle verdict: not applicable.

### 158_native__incremental_propagators
- In progress: 2026-10-12T18:44:03.142246-04:00
- Complete metadata: `logs/search_improvements/158_native__incremental_propagators.json`; log: `logs/search_improvements/158_native__incremental_propagators.log`; patch: `logs/search_improvements/158_native__incremental_propagators.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --incremental-propagators`
- Completed: 2026-10-12T18:44:03.455556-04:00; exit 0; runner elapsed 0.3119999999180436 s.
- Oracle verdict: not applicable.

### 158_native__start_rank_window
- In progress: 2026-10-12T18:44:03.683142-04:00
- Complete metadata: `logs/search_improvements/158_native__start_rank_window.json`; log: `logs/search_improvements/158_native__start_rank_window.log`; patch: `logs/search_improvements/158_native__start_rank_window.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --start-rank-window`
- Completed: 2026-10-12T18:44:03.725305-04:00; exit 0; runner elapsed 0.0470000000204891 s.
- Oracle verdict: not applicable.

### 159_H25_decode
- In progress: 2026-10-12T18:44:07.216632-04:00
- Complete metadata: `logs/search_improvements/159_H25_decode.json`; log: `logs/search_improvements/159_H25_decode.log`; patch: `logs/search_improvements/159_H25_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T18:54:07.524039-04:00; exit 0; runner elapsed 600.311999999918 s.
  `[SearchEngine volume] iterations=92006 elapsed=585.12538649999999 total-domain-bits=4851513.1637403192 remaining-domain-bits=2795360.1473746491 log2-remaining-fraction=-2056153.0163656701 eliminated-bits=2056153.0163656701 remaining-nodes=86835 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.219000000 budget=600.0`
  `[Program deadline] overshoot=0.0624912`
  `[SearchEngine] New best cost: 75.0097 at iteration 84209 event-epoch=1791845198.1146891 program-deadline=1791845647.2756672`
- Target 2425756.5818701596; remaining/total 0.576183152148667; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 150.839021921 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.
- R156 build H25, R157 oracle 26/26, R158 native PASS (4 suites). R159 H25: eliminated 2056153.0163656701 (+14235.53 vs O7), first incumbent 150.839021921 s, costs unchanged, overshoot 0.0624912. Confirmations pending. H26 (candidate, prepared): rank upper bound uses active capacity = max over root alternatives reaching c of |reach(alt)| (non-sources + prefix) instead of N (active classes are reachable from the selected root alternative).

### 160_build_H26
- In progress: 2026-10-12T18:54:19.103267-04:00
- Complete metadata: `logs/search_improvements/160_build_H26.json`; log: `logs/search_improvements/160_build_H26.log`; patch: `logs/search_improvements/160_build_H26.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-12T18:55:15.864259-04:00; exit 0; runner elapsed 56.75 s.
- Oracle verdict: not applicable.

### 161_diag_H26_60s
- In progress: 2026-10-12T18:55:16.198627-04:00
- Complete metadata: `logs/search_improvements/161_diag_H26_60s.json`; log: `logs/search_improvements/161_diag_H26_60s.log`; patch: `logs/search_improvements/161_diag_H26_60s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 60`
- Completed: 2026-10-12T18:56:16.372326-04:00; exit 0; runner elapsed 60.17200000002049 s.
  `[SearchEngine volume] iterations=25199 elapsed=42.726531100000003 total-domain-bits=4851513.1637403192 remaining-domain-bits=2796428.991522945 log2-remaining-fraction=-2055084.1722173742 eliminated-bits=2055084.1722173742 remaining-nodes=25200 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=60.094000000 budget=60.0`
  `[Program deadline] overshoot=0.00193167`
- Target 2425756.5818701596; remaining/total 0.5764034636498877; target met: False.
- Incumbent: no incumbent within budget; time and both costs unavailable (+infinity).
- Oracle verdict: not applicable.
- R160 build / R161 60 s DIAGNOSTIC H26: each bucket's root has 5-6 alternatives and the largest reach set is 7556 of 7559 classes (7555 of 7557 in bucket 16); start bits 91716.5 -> 91711.5 per bucket (~90 bits total). H26 REJECTED (negligible), source reverted to O7+H25.

### 162_build_H25_restore
- In progress: 2026-10-12T18:56:30.477515-04:00
- Complete metadata: `logs/search_improvements/162_build_H25_restore.json`; log: `logs/search_improvements/162_build_H25_restore.log`; patch: `logs/search_improvements/162_build_H25_restore.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-12T18:57:28.197460-04:00; exit 0; runner elapsed 57.70299999997951 s.
- Oracle verdict: not applicable.

### 163_H25_confirm_decode
- In progress: 2026-10-12T18:57:28.451675-04:00
- Complete metadata: `logs/search_improvements/163_H25_confirm_decode.json`; log: `logs/search_improvements/163_H25_confirm_decode.log`; patch: `logs/search_improvements/163_H25_confirm_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T19:07:28.714706-04:00; exit 0; runner elapsed 600.25 s.
  `[SearchEngine volume] iterations=92048 elapsed=585.65631710000002 total-domain-bits=4851513.1637403192 remaining-domain-bits=2795360.1473746491 log2-remaining-fraction=-2056153.0163656701 eliminated-bits=2056153.0163656701 remaining-nodes=86877 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.171000000 budget=600.0`
  `[Program deadline] overshoot=0.0051682`
  `[SearchEngine] New best cost: 75.0097 at iteration 84209 event-epoch=1791845989.8136792 program-deadline=1791846448.5137634`
- Target 2425756.5818701596; remaining/total 0.576183152148667; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 141.299915791 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.

### 164_H25_confirm_decode
- In progress: 2026-10-12T19:07:28.924222-04:00
- Complete metadata: `logs/search_improvements/164_H25_confirm_decode.json`; log: `logs/search_improvements/164_H25_confirm_decode.log`; patch: `logs/search_improvements/164_H25_confirm_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T19:17:29.223881-04:00; exit 0; runner elapsed 600.2970000000205 s.
  `[SearchEngine volume] iterations=92242 elapsed=585.81617429999994 total-domain-bits=4851513.1637403192 remaining-domain-bits=2795360.1473746491 log2-remaining-fraction=-2056153.0163656701 eliminated-bits=2056153.0163656701 remaining-nodes=87071 exhausted=false incomplete-branches=5943`
  `[Program evaluation] elapsed=600.219000000 budget=600.0`
  `[Program deadline] overshoot=0.0533314`
  `[SearchEngine] New best cost: 75.0097 at iteration 84209 event-epoch=1791846584.6477933 program-deadline=1791847048.9836771`
- Target 2425756.5818701596; remaining/total 0.576183152148667; target met: False.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 135.664116144 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.
- O8 OPTIMIZATION retained: H25 on O7. R159/R163/R164 identical eliminated 2056153.0163656701 (+14235.53 vs O7), first incumbent 150.839021921/141.299915791/135.664116144 s, costs unchanged, overshoots 0.0624912/0.0051682/0.0533314; oracle R157 26/26, native R158 PASS (R162 rebuild after H26 revert). Gap 369603.57. Next (user request): OR-tools/CP-SAT experiments and solver-technique research, written to docs/reviews.

### 165_cpsat_relaxation_bucket0_300s
- In progress: 2026-10-12T19:18:28.805710-04:00
- Complete metadata: `logs/search_improvements/165_cpsat_relaxation_bucket0_300s.json`; log: `logs/search_improvements/165_cpsat_relaxation_bucket0_300s.log`; patch: `logs/search_improvements/165_cpsat_relaxation_bucket0_300s.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/.venvx64/Scripts/python.exe logs/search_improvements/cpsat_bucket_relaxation.py logs/search_improvements/096_native_initial_state.json 0 300 8`
- Completed: 2026-10-12T19:23:35.216214-04:00; exit 0; runner elapsed 306.405999999959 s.
- Oracle verdict: not applicable.

### 166_cpsat_engine_load_bucket0_120s
- In progress: 2026-10-12T19:24:05.810845-04:00
- Complete metadata: `logs/search_improvements/166_cpsat_engine_load_bucket0_120s.json`; log: `logs/search_improvements/166_cpsat_engine_load_bucket0_120s.log`; patch: `logs/search_improvements/166_cpsat_engine_load_bucket0_120s.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/.venvx64/Scripts/python.exe logs/search_improvements/cpsat_engine_load_bound.py logs/search_improvements/096_native_initial_state.json 0 120 8`
- Completed: 2026-10-12T19:25:02.580448-04:00; exit 0; runner elapsed 56.765000000130385 s.
- Oracle verdict: not applicable.

### 167_cpsat_engine_load_bucket0_cp
- In progress: 2026-10-12T19:25:36.868486-04:00
- Complete metadata: `logs/search_improvements/167_cpsat_engine_load_bucket0_cp.json`; log: `logs/search_improvements/167_cpsat_engine_load_bucket0_cp.log`; patch: `logs/search_improvements/167_cpsat_engine_load_bucket0_cp.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/.venvx64/Scripts/python.exe logs/search_improvements/cpsat_engine_load_bound.py logs/search_improvements/096_native_initial_state.json 0 120 8`
- Completed: 2026-10-12T19:27:42.674613-04:00; exit 0; runner elapsed 125.79699999978766 s.
- Oracle verdict: not applicable.

### 168_cpsat_load_path_bucket0_300s
- In progress: 2026-10-12T19:27:54.552676-04:00
- Complete metadata: `logs/search_improvements/168_cpsat_load_path_bucket0_300s.json`; log: `logs/search_improvements/168_cpsat_load_path_bucket0_300s.log`; patch: `logs/search_improvements/168_cpsat_load_path_bucket0_300s.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/.venvx64/Scripts/python.exe logs/search_improvements/cpsat_engine_load_bound.py logs/search_improvements/096_native_initial_state.json 0 300 8 path`
- Completed: 2026-10-12T19:33:00.868177-04:00; exit 0; runner elapsed 306.31300000008196 s.
- Oracle verdict: not applicable.
- R165-R168 CP-SAT experiments (offline, x64 OR-Tools 9.15, bucket 0 of the R096 export, memory omitted => lower bounds): time-indexed optional-interval model bound 0.705 / best 187.7 after 300 s; min-max engine load OPTIMAL 30.361 (53 s), its selection has critical path 43.07; load+critical-path relaxation best 41.71, bound 30.33 (300 s). Root native LB is 0.53. Findings and borrowable techniques written to docs/reviews/ortools-cpsat-lessons-2026-10-12.md. Key consequence: with 16 equal-weight buckets, closing a single-bucket restriction needs a bucket bound increase of 16 x (incumbent - total LB) (~714 now), so bound-based bit elimination needs near-proven optimality; next useful step is a root engine-load LP bound + LP-guided incumbent (gate metrics), not bits.

### 169_cpsat_max_coactive_ocl_cacheable
- In progress: 2026-10-12T19:35:38.580626-04:00
- Complete metadata: `logs/search_improvements/169_cpsat_max_coactive_ocl_cacheable.json`; log: `logs/search_improvements/169_cpsat_max_coactive_ocl_cacheable.log`; patch: `logs/search_improvements/169_cpsat_max_coactive_ocl_cacheable.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/.venvx64/Scripts/python.exe logs/search_improvements/cpsat_max_coactive.py logs/search_improvements/096_native_initial_state.json 0 120 ocl_cacheable`
- Completed: 2026-10-12T19:35:43.259980-04:00; exit 0; runner elapsed 4.672000000020489 s.
- Oracle verdict: not applicable.

### 169_cpsat_max_coactive_ocl_all
- In progress: 2026-10-12T19:35:43.544461-04:00
- Complete metadata: `logs/search_improvements/169_cpsat_max_coactive_ocl_all.json`; log: `logs/search_improvements/169_cpsat_max_coactive_ocl_all.log`; patch: `logs/search_improvements/169_cpsat_max_coactive_ocl_all.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/.venvx64/Scripts/python.exe logs/search_improvements/cpsat_max_coactive.py logs/search_improvements/096_native_initial_state.json 0 120 ocl_all`
- Completed: 2026-10-12T19:35:48.375042-04:00; exit 0; runner elapsed 4.8280000002123415 s.
- Oracle verdict: not applicable.

### 170_proto_ocl_exclusive_layout
- In progress: 2026-10-12T19:37:08.188428-04:00
- Complete metadata: `logs/search_improvements/170_proto_ocl_exclusive_layout.json`; log: `logs/search_improvements/170_proto_ocl_exclusive_layout.log`; patch: `logs/search_improvements/170_proto_ocl_exclusive_layout.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/.venvx64/Scripts/python.exe logs/search_improvements/ocl_exclusive_layout_proto.py logs/search_improvements/096_native_initial_state.json 0`
- Completed: 2026-10-12T19:37:17.492232-04:00; exit 0; runner elapsed 9.297000000020489 s.
- Oracle verdict: not applicable.
- H27 (candidate): StaticPrivateLayout falls back to an exclusivity-shared layout when private slots exceed capacity. Motivation R169 (CP-SAT, bucket 0): a single plan can hold at most 467,559 co-active OPENCL pages (cacheable alone 430,590) vs cap 524,288 and 1,631,336 total. R170 offline prototype (logs/search_improvements/ocl_exclusive_layout_proto.py): necessary-condition certificates (avg 7 literals/class) + first-fit place all OPENCL classes in 467,784 pages (bases first: 430,590 then 467,784; same for bucket 16). Soundness: certificates are reachability consequences (x active => class P selects from S); disjoint allowed sets on a common class => never co-active; every possibly co-active pair gets disjoint pages, so moving any feasible plan's allocations to their slots keeps all memory constraints vacuous with identical selections/starts/cost. Cached bases share a slot only if exclusive in every common bucket (and have one), so two overlapping bases can never both be cached (a cached base is selected in every bucket). New oracle cases 27 (sharing required) and 28 (hidden co-activity, wrong certificate would undercut the optimum).

### 171_build_H27
- In progress: 2026-10-12T19:40:37.638510-04:00
- Complete metadata: `logs/search_improvements/171_build_H27.json`; log: `logs/search_improvements/171_build_H27.log`; patch: `logs/search_improvements/171_build_H27.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-12T19:41:34.036876-04:00; exit 0; runner elapsed 56.39100000006147 s.
- Oracle verdict: not applicable.

### 172_oracle_H27
- In progress: 2026-10-12T19:41:38.312437-04:00
- Complete metadata: `logs/search_improvements/172_oracle_H27.json`; log: `logs/search_improvements/172_oracle_H27.log`; patch: `logs/search_improvements/172_oracle_H27.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T19:41:41.102192-04:00; exit 1; runner elapsed 2.780999999959022 s.
- Oracle verdict: 27/28 cases passed.

### 173_native__private_layout
- In progress: 2026-10-12T19:41:41.341019-04:00
- Complete metadata: `logs/search_improvements/173_native__private_layout.json`; log: `logs/search_improvements/173_native__private_layout.log`; patch: `logs/search_improvements/173_native__private_layout.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --private-layout`
- Completed: 2026-10-12T19:41:41.422330-04:00; exit 0; runner elapsed 0.0779999999795109 s.
- Oracle verdict: not applicable.

### 173_native__write_after_read_cache
- In progress: 2026-10-12T19:41:41.671222-04:00
- Complete metadata: `logs/search_improvements/173_native__write_after_read_cache.json`; log: `logs/search_improvements/173_native__write_after_read_cache.log`; patch: `logs/search_improvements/173_native__write_after_read_cache.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --write-after-read-cache`
- Completed: 2026-10-12T19:41:41.723404-04:00; exit 0; runner elapsed 0.0470000000204891 s.
- Oracle verdict: not applicable.

### 173_native__incremental_propagators
- In progress: 2026-10-12T19:41:41.970032-04:00
- Complete metadata: `logs/search_improvements/173_native__incremental_propagators.json`; log: `logs/search_improvements/173_native__incremental_propagators.log`; patch: `logs/search_improvements/173_native__incremental_propagators.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --incremental-propagators`
- Completed: 2026-10-12T19:41:42.025898-04:00; exit 0; runner elapsed 0.0470000000204891 s.
- Oracle verdict: not applicable.

### 173_native__start_rank_window
- In progress: 2026-10-12T19:41:42.256711-04:00
- Complete metadata: `logs/search_improvements/173_native__start_rank_window.json`; log: `logs/search_improvements/173_native__start_rank_window.log`; patch: `logs/search_improvements/173_native__start_rank_window.patch`
- Command: `C:/Users/aaron/CODING/tensor_graphs/tensor_graphs_cpp/test.exe --start-rank-window`
- Completed: 2026-10-12T19:41:42.306704-04:00; exit 0; runner elapsed 0.04699999978765845 s.
- Oracle verdict: not applicable.

### 174_oracle_H27_case28
- In progress: 2026-10-12T19:41:55.634836-04:00
- Complete metadata: `logs/search_improvements/174_oracle_H27_case28.json`; log: `logs/search_improvements/174_oracle_H27_case28.log`; patch: `logs/search_improvements/174_oracle_H27_case28.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T19:41:58.335445-04:00; exit 0; runner elapsed 2.702999999979511 s.
- Oracle verdict: 28/28 cases passed.
- R171 build H27 PASS; R172 oracle 27/28 (my case-28 draft had a wrong expected optimum; enumeration showed reuse makes it feasible at 4); R174 oracle 28/28 with corrected case 28 (m reads a and b together). Direct worker runs: case 27 shared layout fires (3/3 pages, fixed 3 offsets); case 28 shared layout skips (6 pages, hidden co-activity respected); case 17 skips (4 pages). R173 native 4 suites PASS.

### 175_H27_decode
- In progress: 2026-10-12T19:42:22.428724-04:00
- Complete metadata: `logs/search_improvements/175_H27_decode.json`; log: `logs/search_improvements/175_H27_decode.log`; patch: `logs/search_improvements/175_H27_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T19:52:22.673928-04:00; exit 0; runner elapsed 600.2339999999385 s.
  `[SearchEngine volume] iterations=86671 elapsed=585.66298670000003 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737137.6591798724 log2-remaining-fraction=-3114375.5045604468 eliminated-bits=3114375.5045604468 remaining-nodes=82463 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=600.172000000 budget=600.0`
  `[Program deadline] overshoot=0.000876904`
  `[SearchEngine] New best cost: 75.0097 at iteration 78266 event-epoch=1791848677.775656 program-deadline=1791849142.4882531`
- Target 2425756.5818701596; remaining/total 0.35806100087763337; target met: True.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 135.287402868 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.
- R175 H27: eliminated 3114375.5045604468 (+1058222.49 vs O8) => TARGET MET (>= 2425756.58), first incumbent 135.287402868 s, costs 75.0097/75.009705 unchanged, overshoot 0.000876904, incomplete-branches 0 (no offset branching left). OPENCL shared layout: 677 bases, end 467784 of 524288 pages, fixed 55726 of 55726 offsets. Confirmations R176/R177.

### 176_H27_confirm_decode
- In progress: 2026-10-12T19:52:41.857427-04:00
- Complete metadata: `logs/search_improvements/176_H27_confirm_decode.json`; log: `logs/search_improvements/176_H27_confirm_decode.log`; patch: `logs/search_improvements/176_H27_confirm_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T20:02:42.082635-04:00; exit 0; runner elapsed 600.219000000041 s.
  `[SearchEngine volume] iterations=86772 elapsed=585.79183090000004 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737137.6591798724 log2-remaining-fraction=-3114375.5045604468 eliminated-bits=3114375.5045604468 remaining-nodes=82564 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=600.140000000 budget=600.0`
  `[Program deadline] overshoot=0.0141485`
  `[SearchEngine] New best cost: 75.0097 at iteration 78266 event-epoch=1791849293.1461008 program-deadline=1791849761.9101622`
- Target 2425756.5818701596; remaining/total 0.35806100087763337; target met: True.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 131.235938549 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.

### 177_H27_confirm_decode
- In progress: 2026-10-12T20:02:42.292170-04:00
- Complete metadata: `logs/search_improvements/177_H27_confirm_decode.json`; log: `logs/search_improvements/177_H27_confirm_decode.log`; patch: `logs/search_improvements/177_H27_confirm_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T20:12:42.591498-04:00; exit 0; runner elapsed 600.2960000000894 s.
  `[SearchEngine volume] iterations=86794 elapsed=585.44371880000006 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737137.6591798724 log2-remaining-fraction=-3114375.5045604468 eliminated-bits=3114375.5045604468 remaining-nodes=82586 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=600.235000000 budget=600.0`
  `[Program deadline] overshoot=0.084131`
  `[SearchEngine] New best cost: 75.0097 at iteration 78266 event-epoch=1791849894.844842 program-deadline=1791850362.3429468`
- Target 2425756.5818701596; remaining/total 0.35806100087763337; target met: True.
- Incumbent: events above; timing measured from evaluation start via deadline minus 600 seconds.
- First incumbent program elapsed 132.501895189 s, logged first cost 75.0097, logged best cost 75.009705.
- Oracle verdict: not applicable.
- O9 OPTIMIZATION retained: H27 on O8. R175/R176/R177 identical eliminated 3114375.5045604468 (remaining 1737137.6591798724, ratio 0.35806), TARGET MET (target 2425756.5818701596; margin 688618.92 bits). First incumbent 135.287402868/131.235938549/132.501895189 s (original 230.18), costs 75.0097/75.009705 unchanged, overshoots 0.000876904/0.0141485/0.084131, incomplete-branches 0. Oracle R174 28/28 (cases 27/28 exercise and reject sharing as intended), native R173 PASS.

## Resume checkpoint (session 4, final: target met)
- Retained implementation O9 = O8 + H27 (exclusivity-shared static layout). Exact source: `logs/search_improvements/177_H27_confirm_decode.patch`. Extension and test.exe built from it (R171). No running processes.
- Best measurement (3 identical runs): eliminated 3,114,375.50 / remaining 1,737,137.66 of 4,851,513.16 total bits; target `remaining <= total/2` met with 688,619 bits margin. Gate vs original baseline R002/R046: bits +3,111,685, first incumbent -95 to -99 s, costs unchanged.
- The earlier "concrete blocker" analysis (OPENCL capacity) is superseded: R169 showed per-plan OPENCL co-activity is bounded by 467,559 pages, and reachability-implied exclusivity certificates realise a fitting layout (see docs/reviews/ortools-cpsat-lessons-2026-10-12.md section 5).
- Possible next steps (not needed for the target): root engine-load lower bound (R166: 30.36/bucket vs native 0.53) and an LP-guided incumbent (R168 relaxation 41.7 vs 75.0).
