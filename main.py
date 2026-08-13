import argparse
import os

import tensor_graphs
from safetensors.torch import load_file

from train_shared import ActorDelegate, AlphaZeroAgent
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
    args = parser.parse_args()

    agent = AlphaZeroAgent(hidden_dim=64)
    if args.run_dir:
        model_file = os.path.join(args.run_dir, "model.safetensors")
        if os.path.exists(model_file):
            agent.load_state_dict(load_file(model_file))
            print(f"Loaded trained delegate agent from {model_file}")

    agent.eval()
    delegate = ActorDelegate(agent, exploration_noise=0.0)

    print(f"Loading {args.model} via LLMSession...")
    session = tensor_graphs.LLMSession(
        args.model,
        args.model_path,
        delegate,
        min_compile_time=args.min_compile_time,
        compile_decode_buckets=args.compile_decode_buckets,
        disable_caching=args.disable_caching,
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
        print("\nCompletion mode initialized (chat template not available). Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            user_input = input("User: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interaction.")
            break

        if user_input.strip().lower() in ["exit", "quit"]:
            print("Exiting interaction.")
            break

        if not user_input.strip():
            continue

        if has_chat_template:
            # Append user input to history and try applying chat template
            messages.append({"role": "user", "content": user_input})
            try:
                conversation_tokens = list(
                    raw_tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True
                    )
                )
            except AttributeError:
                # Fallback to completion mode if apply_chat_template fails at runtime
                has_chat_template = False
                messages.clear()

        if not has_chat_template:
            # Completion mode encoding
            conversation_tokens = encode_text(tokenizer, user_input, is_first=True)

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
            # Save assistant turn content to history only in chat mode
            assistant_response = decode_tokens(tokenizer, generated_tokens)
            messages.append({"role": "assistant", "content": assistant_response})


if __name__ == "__main__":
    main()