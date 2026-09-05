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

## utils
- [utils/download_hf_meta.py](utils/download_hf_meta.py) use to allow compilation without downloading full model
