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

Comparison of Gemma 3 270M BF16 tokens per second (t/s), for prompt processing and token generation up to sequence length 16 (prompt 5 initial tokens: "Hi, my name", then generate 11 tokens up to seq len 16). Run on computer with NVIDIA Quadro M4000 + Intel Xeon E5-1680 v3. Reproduction steps coming soon :)

| Framework | Prompt Processing (seq_len 5) | Text Generation (seq_len 6..16) |
| :--- | :--- | :--- |
| **tensor_graphs** (CUDA) | 9.65 t/s | 15.79 t/s |
| **llama.cpp** (CUDA) | 206.80 t/s | 72.28 t/s |
| **Hugging Face Transformers** (CPU) | 87.11 t/s | 26.80 t/s |
| **Hugging Face Transformers** (CUDA) | 72.23 t/s | 26.94 t/s |
