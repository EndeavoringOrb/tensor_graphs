# tensor_graphs

computes DAGs with caching on subsequent runs

## build
```
uv venv
uv sync
source .venv/bin/activate # or .venv/Scripts/activate on windows
python build.py
```

## run
LLM completion: `python main.py`

Pass `--use-ortools-full` to solve cache selection, e-graph extraction, engine
scheduling, bufferization and allocation together in one OR-Tools CP-SAT model:

```
uv sync --extra ortools
python build.py --targets bindings main bench_model
python main.py --use-ortools-full
```

The same flag works with the native executables. Python callers can set
`use_ortools_full=True` on `Session` or `LLMSession`. This mode takes precedence
over `--use-ortools` and uses separate default compiled-cache filenames.
Explicit `--cache-file` paths still reuse existing plans; choose a new path to
force a solve.

`ortools_full.OrtoolsSolver` models persistent cache allocations across all
buckets, fixed input reservations, view aliases, safe in-place choices, and
4096-byte-aligned offsets under each memory cap. Dispatch minimizes weighted
bucket makespan using the exported kernel engines and costs (rounded up to
microseconds). Views retain their backing storage; in-place reuse requires a
non-view, non-persistent input whose other readers have finished. The full
bucket initializes cached values before other buckets may read them.

The solver uses one solve with a default 30-second limit (increased by
`--min-compile-time`). A feasible solution is accepted at the limit; infeasible
or timed-out searches without a solution fail explicitly. Exported JSON also
accepts `max_time_seconds`, `num_workers`, and `print_progress`.
The resulting choices, order and offsets are consumed directly by C++.
The existing `--use-ortools` mode retains its native finalization path.

Regression tests: build `test_ortools_full`, run that executable, then run
`python -m unittest test_ortools_full`. If OR-Tools needs a separate interpreter
(for example on Windows ARM64), the tests can use the native solver process
bridge. `TENSOR_GRAPHS_ORTOOLS_PYTHON` selects that interpreter.

## utils
- [utils/download_hf_meta.py](utils/download_hf_meta.py) use to allow compilation without downloading full model
