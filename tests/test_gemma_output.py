import argparse
import os
import math
import re
import subprocess
import sys
import time
from pathlib import Path


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
    import tensor_graphs
    from main import decode_tokens, encode_text
    from utils.decode import load_tokenizer

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
    parser.add_argument(
        "--search-volume",
        action="store_true",
        help="Report remaining frontier hyperbox volume relative to initial domains, in bits",
    )
    parser.add_argument("--timeout", type=float,
                        help="Whole worker process budget in seconds; report frontier and stop")
    args = parser.parse_args()
    if args.timeout is not None:
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            parser.error("--timeout must be finite and positive")
        if args.config == "all" or not args.optimal or args.cache_file:
            parser.error("--timeout requires one --config, --optimal, and no compiled cache")
        if "TG_PROGRAM_DEADLINE" not in os.environ:
            start_time = time.monotonic()
            environment = os.environ.copy()
            environment["TG_PROGRAM_DEADLINE"] = str(time.time() + args.timeout)
            environment["TG_SEARCH_VOLUME"] = "1"
            command = [sys.executable, "-u", "-m", "tests.test_gemma_output", *sys.argv[1:]]
            worker = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True)
            try:
                output, _ = worker.communicate(timeout=max(0.001, args.timeout - (time.monotonic() - start_time)) + 2.0)
            except subprocess.TimeoutExpired:
                worker.kill()
                output, _ = worker.communicate()
                print(output, end="", flush=True)
                raise SystemExit("Evaluation failed: program exceeded deadline and reporting grace")
            elapsed = time.monotonic() - start_time
            print(output, end="", flush=True)
            print(f"[Program evaluation] elapsed={elapsed:.9f} budget={args.timeout}", flush=True)
            if worker.returncode != 0 or "[SearchEngine volume]" not in output:
                raise SystemExit("Evaluation failed: worker failed or did not report frontier volume")
            overshoots = re.findall(r"\[Program deadline\] overshoot=([0-9.eE+\-]+)", output)
            overshoot = float(overshoots[-1]) if overshoots else max(0.0, elapsed - args.timeout)
            if not math.isfinite(overshoot) or overshoot > 1.0:
                raise SystemExit("Evaluation failed: deadline overshoot exceeds 1 second")
            return
    if args.search_volume:
        os.environ["TG_SEARCH_VOLUME"] = "1"

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
