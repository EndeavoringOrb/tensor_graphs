import argparse
from pathlib import Path

import tensor_graphs

from main import decode_tokens, encode_text
from utils.decode import load_tokenizer


CONFIGS = {
    "full": {
        "compile_decode_buckets": False,
        "compile_no_weights_bucket": False,
    },
    "no-weights": {
        "compile_decode_buckets": False,
        "compile_no_weights_bucket": True,
    },
    "decode": {
        "compile_decode_buckets": True,
        "compile_no_weights_bucket": False,
    },
}


def run_gemma_output(config_name, cache_file="", min_compile_time=0.0):
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown Gemma test config: {config_name}")

    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "google" / "gemma-3-270m"
    tokenizer = load_tokenizer([str(model_path), "gemma-3-270m"])
    conversation_tokens = encode_text(tokenizer, "Hi, my name", is_first=True)
    expected_string = " is <strong>Jasmine"

    disable_compilation_caching = not bool(cache_file)
    session = tensor_graphs.LLMSession(
        "gemma-3-270m",
        str(model_path),
        tensor_graphs.HeuristicSearchDelegate(),
        min_compile_time=min_compile_time,
        max_sequence_length=16,
        **CONFIGS[config_name],
        cache_file=cache_file,
        disable_node_caching=False,
        disable_compilation_caching=disable_compilation_caching,
    )

    generated_string = ""
    while len(generated_string) <= len(expected_string):
        next_token = session.generate_step(conversation_tokens)
        decoded = decode_tokens(tokenizer, [next_token]) if next_token != -1 else ""
        print(f"[DEBUG_TOKEN] next_token={next_token} decoded={decoded!r}", flush=True)
        assert next_token != -1, (
            f"{config_name}: generation stopped at sequence length "
            f"{len(conversation_tokens)} after producing {generated_string!r}"
        )
        conversation_tokens.append(next_token)
        generated_string += decoded
        assert generated_string.startswith(expected_string[: len(generated_string)]), (
            f"{config_name}: generated {generated_string!r}, expected prefix of "
            f"{expected_string!r}"
        )


def test_gemma_output(cache_file="", min_compile_time=0.0):
    for config_name in CONFIGS:
        run_gemma_output(
            config_name,
            cache_file=cache_file,
            min_compile_time=min_compile_time,
        )


def main():
    parser = argparse.ArgumentParser(description="Check Gemma output for bucket configurations")
    parser.add_argument(
        "--config",
        choices=["all", *CONFIGS],
        default="all",
        help="Configuration to test (default: all)",
    )
    parser.add_argument(
        "--cache-file",
        "--cache",
        dest="cache_file",
        type=str,
        default="",
        help="Path to compiled cache file. If specified, enables compilation caching to/from this file.",
    )
    parser.add_argument(
        "--min-compile-time",
        type=float,
        default=0.0,
        help="Search time budget in seconds (0 stops at the first feasible plan)",
    )
    parser.add_argument(
        "--optimal",
        action="store_true",
        help="Search until the planner exhausts its search space",
    )
    args = parser.parse_args()

    config_names = CONFIGS if args.config == "all" else [args.config]
    for config_name in config_names:
        print(f"\n=== Gemma output config: {config_name} ===", flush=True)
        run_gemma_output(
            config_name,
            cache_file=args.cache_file,
            min_compile_time=-1.0 if args.optimal else args.min_compile_time,
        )


if __name__ == "__main__":
    main()
