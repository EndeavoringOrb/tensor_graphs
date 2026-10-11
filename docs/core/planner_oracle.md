# Adversarial planner correctness cases

```powershell
.venv/Scripts/python.exe -m tests.test_planner_oracle
```

This is a fast empirical correctness gate for search optimization. Python
independently enumerates a small finite problem, then compares its optimum with
production C++ propagation and exhaustive search. It compares **objective
values**, so deleting individual optima is allowed when an equal-cost
representative survives. It also validates the returned native incumbent rather
than accepting a claimed cost alone.

The gate does not prove correctness on arbitrary Gemma instances. Reviewers can
add counterexamples, permanently expanding the cases against which future
changes are checked. An adversarial process improves coverage; passing a finite
suite remains empirical evidence rather than a mathematical proof.

## Add a case without writing test code

Drop a JSON file anywhere under `tests/planner_oracle_cases/`. Discovery is
recursive and requires no test registry. For example:

```json
{
  "name": "expensive child of an optional alternative",
  "description": "Try to break a bound that ignores optional descendants.",
  "root": "out",
  "nodes": {
    "out": {
      "alternatives": [
        {"children": ["slow"], "cost": 1},
        {"children": ["fast"], "cost": 2}
      ]
    },
    "slow": {"alternatives": [{"cost": 20}]},
    "fast": {"alternatives": [{"cost": 3}]}
  },
  "expected_optimum": 5
}
```

`expected_optimum` is optional. The checker computes it independently; a supplied
value is an extra assertion, never the source of truth. Use `null` to assert
infeasibility. A failing run prints the original oracle optimum and a feasible
counterexample assignment where relevant.

Run a single case by filename stem, its name, or an explicit path:

```powershell
.venv/Scripts/python.exe -m tests.test_planner_oracle --case 02_optional_expensive_child
.venv/Scripts/python.exe -m tests.test_planner_oracle --list
.venv/Scripts/python.exe -m tests.test_planner_oracle --json-report .cache/planner_oracle/report.json
```

Each case runs in a subprocess with a hard timeout (20 seconds by default).
Crashes, timeouts, exhausted enumeration budgets, schema errors, and an empty
suite all fail the gate. `--max-steps` defaults to 250000 per enumeration; use
small counterexamples rather than raising this indefinitely.

## Supported case model

- One to six named e-classes per bucket, one to four buckets, one CPP memory
  space, page size four bytes, FLOAT32 one-dimensional buffers.
- `alternatives`: list of implementations; fields are `children` (node names),
  `cost` (default 1), `engines` (default `[0]`), `op` (default `compute`), and
  `view` (default false). Engine IDs 0 through 3 model distinct unary CPU engines.
  An empty engine list uses engine 0, matching production execution semantics.
- `op`: `compute`, `input`, `cache`, or `scatter`. INPUT and CACHE have zero
  duration. Views have zero duration and exactly one child, sharing its address.
  Other operations reserve all declared engines simultaneously for their cost.
- `size_pages` defaults to 1. `base` defaults to the node name and identifies a
  shared cache across buckets. Distinct nodes in one bucket need distinct bases;
  use a view to represent an alias.
- `selected` lists allowed choices: 0 means unselected; 1..N index alternatives.
  The root is mandatory by default and must exclude 0. Other nodes allow 0..N.
  Selected nodes must be reachable from the root through chosen alternatives.
- `start` lists dispatch ranks, defaulting to 0..number-of-nodes-minus-one.
  Active nodes have distinct ranks and producers precede consumers strictly.
- `offset` lists page addresses, defaulting to the node's insertion index.
  Set explicit addresses for larger buffers, aliasing, or memory-reuse cases.
- `capacity_pages` defaults to the largest possible declared buffer extent.
  INPUT, CACHE, cached classes, and bucket roots cannot share physical ranges
  with distinct allocations. Other buffers can reuse memory strictly after all
  readers finish in dispatch order. No kernels in this schema are safe in-place.
- Unselected nodes' ranks/addresses are irrelevant and represented by zero.
  Changing those coordinates does not constitute losing a solution.
- `caches` maps base names to `{ "domain": [0, 1], "size_pages": 1 }`.
  Cached copies must be selected and use the same persistent address. Declared
  cache size must cover every copy. CACHE/SCATTER alternatives require caching.
- For multiple buckets, replace top-level `nodes`/`root` with `buckets`, whose
  objects have `nodes`, `root`, and `weight` (default 1). Objective is weighted
  simulated makespan. Weight zero does not exempt a bucket from feasibility.
- Costs are in [0,256], weights in [0,8], both multiples of 1/16. Together with
  the case-size limits, these bounds make every simulated objective exactly
  representable in native FLOAT32 arithmetic. Reference costs use exact rational
  arithmetic; optimal costs and returned incumbent costs must match exactly.
  Lower-bound diagnostics allow an absolute tolerance of 0.00001.

Unknown fields are rejected. This first model excludes preallocated buffers,
multiple memory spaces, kernel eligibility, safe in-place operations, and dirty
region evolution. Those need explicit independent semantics before adding their
fields. Tiny synthetic egraphs are constructed directly, bypassing saturation
and hardware benchmarking, and use the production propagator registration and
brancher/selector.

## What is checked

For each case:

1. Independently compute feasibility and the exact optimal cost.
2. Run production root propagation; enumerate its surviving domains and require
   the same optimum to remain. Check its lower bound does not exceed the oracle
   optimum.
3. Repeat propagation with cutoffs above, at, and below the oracle optimum.
   An optimum must survive whenever its cost is strictly below the cutoff.
   Equal-to-cutoff solutions may be eliminated legitimately.
4. Run native search to exhaustion; compare its result with the oracle and
   independently validate its returned implementation choices, order, cache
   decisions, addresses, and cost. This exercises branching and restoration.

Optional `probes` add parent-domain restrictions and repeat all these checks:

```json
"probes": [{
  "name": "after choosing the slow root alternative",
  "fixes": [{"bucket": 0, "node": "out", "field": "selected", "domain": [1]}]
}]
```

Node fix fields are `selected`, `start`, or `offset`. Cache fixes use
`{"cache": 0, "domain": [1]}`, indexing the cache insertion order. Fixes must
narrow original domains. `cutoffs` can optionally replace the automatic cutoff
values. Existing cases cover shared children, duplicates, views, cycles, engine
overlap, empty engine sets, rank overload, memory reuse, sparse domains, zero
weights, and persistent cache addresses.

## Reference-result cache

Independent enumeration results are cached by default in
`.cache/planner_oracle/`, which is ignored by git. Keys hash the complete
normalized problem and the source of `planner_oracle_reference.py`. Editing the
case's actual constraints or the oracle's implementation invalidates its cached
truth. Stored results include exact rational costs, witnesses, work counts, and a
checksum, and are written atomically for concurrent runs.

**Native propagation, search, incumbent validation, and pass/fail decisions are
never cached.** A newly rebuilt candidate is always checked against reference
truth. Returned incumbents are checked directly for feasibility and cost, without
enumerating the whole problem again. Arbitrary witness-filter queries, if used
through the reference API, are enumerated afresh.

```powershell
.venv/Scripts/python.exe -m tests.test_planner_oracle --no-cache
.venv/Scripts/python.exe -m tests.test_planner_oracle --cache-dir .cache/reviewer-oracles
.venv/Scripts/python.exe -m tests.test_planner_oracle --reference-only
```

The report shows cache hits and misses. A hit still respects the current
enumeration-work limit. Cache corruption fails explicitly; rerun with
`--no-cache` or an empty cache directory. Description/expected-value/probe edits
are rechecked even when the underlying reference problem can reuse its optimum.

## Protect the evaluator in the agent loop

Treat the Python reference, gate runner, native adapter, and baseline case corpus
as trusted evaluation inputs. The optimizing agent should edit planner code;
reviewers add cases. Do not accept a change that disables checks, deletes cases,
alters expected semantics, or spoofs the adapter. Keep private additional cases
if optimizing repeatedly against the public corpus. Cache checksums detect
accidental corruption, not malicious edits to trusted evaluation files.

The native adapter is `_run_planner_oracle_case` in the rebuilt extension. Build:

```powershell
.venv/Scripts/python.exe build.py --targets bindings --no-lint --log-level INFO
```

Gate self-checks deliberately inject a prune-everything result, an invalid bound,
and a fabricated incumbent, while accepting an equal-cost representative:

```powershell
.venv/Scripts/python.exe -m unittest tests.test_planner_oracle_harness
```
