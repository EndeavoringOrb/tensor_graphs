"""Validate the deployed 16 decode buckets plus zero-weight full Gemma goal."""
import argparse
import math
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import tensor_graphs
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from goal_validation import writeGoalReport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--min-compile-time", type=float, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--plan-only", action="store_true", help="Compile and report selection without correctness or timing runs")
    args = parser.parse_args()
    if not 0 <= args.warmup <= 100 or not 8 <= args.iterations <= 1000:
        raise ValueError("warmup must be 0..100 and iterations 8..1000")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokens = tokenizer("The quick brown fox jumps over the lazy dog. A quiet evening follows.",
                       return_tensors="pt", truncation=True, max_length=16, padding="max_length")["input_ids"]
    reference_config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    # Preserve the existing logical workload: its graph uses theta=10000 for
    # every layer, including full attention (the upstream model uses 1e6 there).
    # This independent reference evaluates that explicit graph semantics.
    reference_config.rope_parameters = {
        layer_type: {"rope_type": "default", "rope_theta": 10000.0}
        for layer_type in set(reference_config.layer_types)}
    if not args.plan_only:
        reference_model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
            config=reference_config, dtype=torch.float32).eval()
        with torch.inference_mode():
            expected = reference_model(input_ids=tokens).logits.detach().cpu()
        del reference_model
    # LLMSession appends its own .dirty-cache suffix.
    cache_base = str(args.cache)
    if cache_base.endswith(".dirty-cache"):
        cache_base = cache_base[:-len(".dirty-cache")]
    owner = tensor_graphs.LLMSession("gemma-3-270m", str(args.model_path),
        cache_file=cache_base, token_buckets=[(i, i + 1, i, 1.0) for i in range(16)], max_sequence_length=16,
        bucket_weights=[1 / 16] * 16 + [0], min_compile_time=args.min_compile_time,
        log_cost_calls=False)
    actual_cache = Path(cache_base + ".dirty-cache")
    if actual_cache != args.cache:
        shutil.copyfile(actual_cache, args.cache)
    session = owner.native_session
    session.write_input_int32(owner.input_id, tokens.reshape(-1).tolist())

    def checkBucket(values, bucket):
        native = torch.tensor(values, dtype=torch.float32).reshape(expected.shape)
        regions = bucket["outputNeededRegion"]
        for region in regions:
            bounds = region["region"]
            slices = tuple(slice(bound["start"], bound["stop"]) for bound in bounds)
            actual, reference = native[slices].flatten().double(), expected[slices].flatten().double()
            cosine = torch.nn.functional.cosine_similarity(actual, reference, dim=0).item()
            error = torch.linalg.vector_norm(actual - reference).item() / torch.linalg.vector_norm(reference).item()
            if not torch.isfinite(actual).all() or not math.isfinite(cosine) or not math.isfinite(error) or cosine < 0.9999 or error > 0.02:
                raise AssertionError(f"Gemma bucket failed: cosine={cosine}, relative error={error}")
            print(f"Bucket correctness: cosine={cosine:.7f}, relative_error={error:.7f}", flush=True)

    writeGoalReport(session, owner.root_id, args.cache, args.validation_report, checkBucket,
                    args.warmup, args.iterations, plan_only=args.plan_only)


if __name__ == "__main__":
    main()
