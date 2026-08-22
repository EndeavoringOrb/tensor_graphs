import argparse
from pathlib import Path

import tensor_graphs
from safetensors.torch import load_file

from train_models import AlphaZeroTransformer
from train_shared import ActorDelegate, HeuristicDelegate, TrainConfig
from utils.decode import load_tokenizer


def decode_tokens(tokenizer_obj, token_ids: list[int]) -> str:
    if isinstance(tokenizer_obj, tuple):
        tokenizer_obj = tokenizer_obj[0]
    if hasattr(tokenizer_obj, "decode"):
        return tokenizer_obj.decode(token_ids)
    raise ValueError("Unsupported tokenizer type")


def encode_text(tokenizer_obj, text: str, is_first: bool = False):
    if isinstance(tokenizer_obj, tuple):
        tokenizer_obj = tokenizer_obj[0]
    if hasattr(tokenizer_obj, "encode"):
        res = tokenizer_obj.encode(text)
        if hasattr(res, "ids"):
            tokens = list(res.ids)
        elif isinstance(res, list):
            tokens = list(res)
        elif hasattr(res, "tolist"):
            tokens = list(res.tolist())
        else:
            tokens = list(res)

        if (
            is_first
            and hasattr(tokenizer_obj, "bos_token_id")
            and tokenizer_obj.bos_token_id is not None
        ):
            if not tokens or tokens[0] != tokenizer_obj.bos_token_id:
                tokens = [tokenizer_obj.bos_token_id] + tokens
        return tokens
    raise ValueError("Unsupported tokenizer type")


def main():
    parser = argparse.ArgumentParser(
        description="Run AutoRegressive LLM Interactive Chat / Completion"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-3-270m",
        help="gemma-3-270m, qwen-3.6-35b-a3b, deepseek-v4",
    )
    parser.add_argument("--model-path", type=str, default="models/google/gemma-3-270m")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to runs/N to load the agent from",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=20,
        help="Max tokens to generate per response turn",
    )
    parser.add_argument(
        "--min-compile-time",
        type=float,
        default=0.0,
        help="Minimum required compile time per bucket in seconds",
    )
    parser.add_argument(
        "--compile-decode-buckets",
        action="store_true",
        help="Compile decode buckets in addition to the single full bucket",
    )
    parser.add_argument(
        "--disable-caching",
        action="store_true",
        help="Disable dirty region session caching",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Number of C++ threads (0 = auto-detect based on hardware)",
    )
    args = parser.parse_args()

    if args.threads > 0:
        tensor_graphs.set_num_threads(args.threads)

    cfg = TrainConfig()

    run_dir_path = Path(args.run_dir) if args.run_dir else None
    if run_dir_path:
        config_file = run_dir_path / "config.json"
        try:
            cfg = TrainConfig.load(config_file)
        except FileNotFoundError as e:
            print(f"Warning: Failed to load config from {config_file}: {e}")

        agent = AlphaZeroTransformer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            max_feat_dim=cfg.max_feat_dim,
        )

        model_file = run_dir_path / "model.safetensors"
        if model_file.exists():
            agent.load_state_dict(load_file(model_file))
            print(f"Loaded trained delegate agent from {model_file}")

        agent.eval()
        delegate = ActorDelegate(agent=agent, exploration_noise=0.0)
    else:
        print("No --run-dir specified. Using HeuristicDelegate (caching disabled).")
        delegate = HeuristicDelegate()
        args.disable_caching = True

    print(f"Loading {args.model} via LLMSession...")
    session = tensor_graphs.LLMSession(
        args.model,
        args.model_path,
        delegate,
        min_compile_time=args.min_compile_time,
        compile_decode_buckets=args.compile_decode_buckets,
        disable_caching=args.disable_caching,
        threads=args.threads,
    )

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = load_tokenizer([args.model_path, args.model])
    raw_tokenizer = tokenizer[0] if isinstance(tokenizer, tuple) else tokenizer

    eos_token_id = getattr(raw_tokenizer, "eos_token_id", None)
    has_chat_template = hasattr(raw_tokenizer, "apply_chat_template")

    if has_chat_template:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        print("\nChat initialized. Type 'exit' or 'quit' to end.\n")
    else:
        messages = []
        print(
            "\nCompletion mode initialized (chat template not available). Type 'exit' or 'quit' to end.\n"
        )

    while True:
        try:
            user_input = input("User: " if has_chat_template else "Enter text: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interaction.")
            break

        if user_input.strip().lower() in ["exit", "quit"]:
            print("Exiting interaction.")
            break

        if not user_input.strip():
            continue

        if has_chat_template:
            messages.append({"role": "user", "content": user_input})
            try:
                conversation_tokens = list(
                    raw_tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True
                    )
                )
            except AttributeError:
                has_chat_template = False
                messages.clear()

        if not has_chat_template:
            conversation_tokens = encode_text(tokenizer, user_input, is_first=True)

        if has_chat_template:
            print("Bot: ", end="", flush=True)
        generated_tokens = []
        prev_text_len = 0

        for _ in range(args.tokens):
            next_token = session.generate_step(conversation_tokens)
            if next_token == -1:
                print("\n[Max sequence length reached]")
                break
            conversation_tokens.append(next_token)
            generated_tokens.append(next_token)

            full_text = decode_tokens(tokenizer, generated_tokens)
            new_text = full_text[prev_text_len:]
            print(new_text, end="", flush=True)
            prev_text_len = len(full_text)

            if eos_token_id is not None and next_token == eos_token_id:
                break
        print("\n")

        if has_chat_template:
            assistant_response = decode_tokens(tokenizer, generated_tokens)
            messages.append({"role": "assistant", "content": assistant_response})


if __name__ == "__main__":
    main()