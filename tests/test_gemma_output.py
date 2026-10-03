import argparse
from pathlib import Path

import tensor_graphs

from main import decode_tokens, encode_text
from utils.decode import load_tokenizer


CONFIGS = {
    "current": {
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


def run_gemma_output(config_name):
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown Gemma test config: {config_name}")

    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "google" / "gemma-3-270m"
    tokenizer = load_tokenizer([str(model_path), "gemma-3-270m"])
    conversation_tokens = encode_text(tokenizer, "Hi, my name", is_first=True)
    expected_string = " is <strong>Jasmine"

    session = tensor_graphs.LLMSession(
        "gemma-3-270m",
        str(model_path),
        tensor_graphs.HeuristicSearchDelegate(),
        max_sequence_length=16,
        **CONFIGS[config_name],
        disable_node_caching=False,
        disable_compilation_caching=True,
    )

    generated_string = ""
    while len(generated_string) <= len(expected_string):
        next_token = session.generate_step(conversation_tokens)
        assert next_token != -1, (
            f"{config_name}: generation stopped at sequence length "
            f"{len(conversation_tokens)} after producing {generated_string!r}"
        )
        conversation_tokens.append(next_token)
        generated_string += decode_tokens(tokenizer, [next_token])
        assert generated_string.startswith(expected_string[: len(generated_string)]), (
            f"{config_name}: generated {generated_string!r}, expected prefix of "
            f"{expected_string!r}"
        )


def test_gemma_output():
    for config_name in CONFIGS:
        run_gemma_output(config_name)


def main():
    parser = argparse.ArgumentParser(description="Check Gemma output for bucket configurations")
    parser.add_argument(
        "--config",
        choices=["all", *CONFIGS],
        default="all",
        help="Configuration to test (default: all)",
    )
    args = parser.parse_args()

    config_names = CONFIGS if args.config == "all" else [args.config]
    for config_name in config_names:
        print(f"\n=== Gemma output config: {config_name} ===", flush=True)
        run_gemma_output(config_name)


if __name__ == "__main__":
    main()
