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
gemma-3-270m text completion: `python main.py`

## utils
- [utils/download_hf_meta.py](utils/download_hf_meta.py) use to allow compilation without downloading full model
- [utils/download_hf.py](utils/download_hf.py) use to download full model

## benchmarks

Comparison of Gemma 3 270M BF16, measuring prompt processing and token generation up to sequence length 16 (prompt 5 initial tokens: "Hi, my name", then generate 11 tokens up to seq len 16).

Run benchmarks using [`utils/run_gemma_benchmarks.py`](utils/run_gemma_benchmarks.py)

| Framework | Prompt Processing (seq_len 5) | Text Generation (seq_len 6..16) | Overall Generation (tokens 5..15) |
| :--- | :--- | :--- | :--- |
| **tensor_graphs** (CUDA) | 518.20 ms (9.65 t/s) | 63.32 ms / tok (15.79 t/s) | 104.66 ms / tok (9.55 t/s) |
| **llama.cpp** (CUDA) | 24.18 ms (206.80 ± 0.27 t/s) | 13.83 ms / tok (72.28 ± 0.02 t/s) | ~14.77 ms / tok (~67.7 t/s) |
| **Hugging Face Transformers** (CPU) | 57.40 ms (87.11 t/s) | 37.32 ms / tok (26.80 t/s) | 39.14 ms / tok (25.55 t/s) |
| **Hugging Face Transformers** (CUDA) | 69.22 ms (72.23 t/s) | 37.12 ms / tok (26.94 t/s) | 40.04 ms / tok (24.98 t/s) |