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

## Benchmarks

Head-to-head comparison on Gemma 3 270M BF16 (`gemma-3-270m-it-bf16` / `gemma-3-270m`), measuring prompt processing and token generation up to sequence length 16 (prompt: "Hi, my name", 5 initial tokens; generating 11 tokens up to seq len 16).

Run all or individual benchmarks using [`utils/run_gemma_benchmarks.py`](utils/run_gemma_benchmarks.py):
```bash
# Run all benchmarks
python utils/run_gemma_benchmarks.py --target all

# Target specific framework
python utils/run_gemma_benchmarks.py --target tensor_graphs
python utils/run_gemma_benchmarks.py --target llama_cpp
python utils/run_gemma_benchmarks.py --target transformers
```
Outputs are streamed to console and saved in `./logs/` (`tensor_graphs.log`, `llama_cpp.log`, `transformers.log`).

| Framework | Prompt Processing (seq_len 5) | Text Generation (seq_len 6..16) | Overall Generation (tokens 5..15) |
| :--- | :--- | :--- | :--- |
| **tensor_graphs** (`--config decode`) | 518.20 ms (9.65 t/s) | 63.32 ms / tok (15.79 t/s) | 104.66 ms / tok (9.55 t/s) |
| **llama.cpp** (CUDA / Quadro M4000) | 24.18 ms (206.80 ± 0.27 t/s) | 13.83 ms / tok (72.28 ± 0.02 t/s) | ~14.77 ms / tok (~67.7 t/s) |
| **llama.cpp** (CPU / 8 threads) | 20.79 ms (240.48 ± 2.82 t/s) | 18.18 ms / tok (55.00 ± 0.18 t/s) | ~18.42 ms / tok (~54.3 t/s) |
| **Hugging Face Transformers** (CPU / BF16) | 57.40 ms (87.11 t/s) | 37.32 ms / tok (26.80 t/s) | 39.14 ms / tok (25.55 t/s) |
| **Hugging Face Transformers** (CUDA / BF16) | 69.22 ms (72.23 t/s) | 37.12 ms / tok (26.94 t/s) | 40.04 ms / tok (24.98 t/s) |

### Detailed Results

#### 1. tensor_graphs (`--config decode`)
- **Initial prompt processing (seq len 5)**: 518.20 ms
- **Decode step timings**:
  - `seq_len=6`: 61.99 ms
  - `seq_len=7`: 66.26 ms
  - `seq_len=8`: 62.38 ms
  - `seq_len=9`: 62.34 ms
  - `seq_len=10`: 63.59 ms
  - `seq_len=11`: 63.11 ms
  - `seq_len=12`: 63.36 ms
  - `seq_len=13`: 62.95 ms
  - `seq_len=14`: 63.46 ms
  - `seq_len=15`: 63.67 ms
- **Steady-state decode avg**: ~63.3 ms/tok (~15.8 t/s)
- **Full sequence average (including initial step)**: 104.66 ms/tok (9.55 t/s)

#### 2. llama.cpp (`llama-bench` on `/mnt/bigdrive/models/gguf-org/gemma-3-270m-it-bf16.gguf`)
- **GPU (CUDA / Quadro M4000, ngl -1)**:
  - `pp5`: 206.80 ± 0.27 t/s (24.18 ms)
  - `tg11`: 72.28 ± 0.02 t/s (13.83 ms / tok)
- **CPU (8 threads, ngl 0)**:
  - `pp5`: 240.48 ± 2.82 t/s (20.79 ms)
  - `tg11`: 55.00 ± 0.18 t/s (18.18 ms / tok)

#### 3. Hugging Face Transformers (`models/google/gemma-3-270m`, BF16, PyTorch 2.7.1)
- **CPU (8 threads)**:
  - `pp5` (seq_len 5): 57.40 ms (87.11 t/s)
  - `tg10` (seq_len 6..16 steady-state decode): 37.32 ms / tok (26.80 t/s)
  - Total generation (11 tokens): 39.14 ms / tok (25.55 t/s)
- **GPU (CUDA / Quadro M4000)**:
  - `pp5` (seq_len 5): 69.22 ms (72.23 t/s)
  - `tg10` (seq_len 6..16 steady-state decode): 37.12 ms / tok (26.94 t/s)
  - Total generation (11 tokens): 40.04 ms / tok (24.98 t/s)
