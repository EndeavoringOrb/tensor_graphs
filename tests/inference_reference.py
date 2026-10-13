"""Precompute the HF transformers reference for the inference benchmark.

Usage (once, offline):
  .venv/Scripts/python.exe -m tests.inference_reference

Builds the fixed token sequence S (BOS + the model card text, first
SEQUENCE_LENGTH tokens), runs one float32 forward pass with eager attention,
and stores the FULL logits row (every vocabulary entry, float32) of every
position checked by tests/inference_cases.py:

- tests/inference_reference/<model>.logits.npy  float32 [positions, vocab]
  (about 202 MiB; git-ignored)
- tests/inference_reference/<model>.json        manifest: token ids, the row
  order of the .npy, its SHA-256, versions (committed)

The benchmark verifies the .npy against the manifest's SHA-256 before use.
"""
import argparse
import datetime
import hashlib
import json
import pathlib

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from tests.inference_cases import CASES, MODEL_NAME, MODEL_PATH, REFERENCE_PATH, SEQUENCE_LENGTH, all_checked_positions


def build_tokens(tokenizer, model_path):
    text = (pathlib.Path(model_path) / "README.md").read_text(encoding="utf-8")
    ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    if len(ids) < SEQUENCE_LENGTH:
        raise SystemExit(f"Source text has only {len(ids)} tokens; need {SEQUENCE_LENGTH}")
    if ids[0] != tokenizer.bos_token_id:
        raise SystemExit("Tokenizer did not prepend BOS")
    return ids[:SEQUENCE_LENGTH], hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--out", default=REFERENCE_PATH, help="manifest path (.json); logits go next to it")
    args = parser.parse_args()

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokens, text_sha256 = build_tokens(tokenizer, args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32,
                                                 attn_implementation="eager")
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor([tokens], dtype=torch.long), use_cache=False).logits[0]
    if logits.dtype != torch.float32 or logits.shape != (SEQUENCE_LENGTH, model.config.vocab_size):
        raise SystemExit(f"Unexpected logits {logits.dtype} {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise SystemExit("Reference logits contain non-finite values")

    positions = all_checked_positions()
    rows = logits[positions].numpy().astype(np.float32, copy=False)

    manifest_path = pathlib.Path(args.out)
    logits_path = manifest_path.with_suffix(".logits.npy")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(logits_path, rows)

    manifest = {
        "model": MODEL_NAME,
        "model_path": args.model_path,
        "created": datetime.datetime.now().astimezone().isoformat(),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "dtype": "float32",
        "attn_implementation": "eager",
        "source": "README.md of the model directory, tokenized with BOS",
        "source_sha256": text_sha256,
        "sequence_length": SEQUENCE_LENGTH,
        "vocab_size": int(model.config.vocab_size),
        "cases": CASES,
        "tokens": [int(t) for t in tokens],
        "positions": positions,
        "logits_file": logits_path.name,
        "logits_shape": list(rows.shape),
        "logits_sha256": sha256_file(logits_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"Wrote {manifest_path} and {logits_path} ({rows.shape[0]} positions x {rows.shape[1]} logits, "
          f"{logits_path.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
