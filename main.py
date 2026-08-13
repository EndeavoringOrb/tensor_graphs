# main.py
import argparse
import os

import tensor_graphs
from safetensors.torch import load_file

from train_shared import ActorDelegate, AlphaZeroAgent


def load_tokenizer(model_path: str, model_name: str):
    tokenizer = None
    try:
        from tokenizers import Tokenizer

        for path in [model_path, model_name]:
            try:
                if os.path.isdir(path):
                    json_path = os.path.join(path, "tokenizer.json")
                    if os.path.exists(json_path):
                        tokenizer = Tokenizer.from_file(json_path)
                        break
                else:
                    tokenizer = Tokenizer.from_pretrained(path)
                    break
            except Exception:
                pass
    except ImportError:
        pass

    if tokenizer is None:
        try:
            from transformers import AutoTokenizer

            for path in [model_path, model_name]:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(path)
                    break
                except Exception:
                    pass
        except ImportError:
            pass

    if tokenizer is None:
        raise RuntimeError(
            f"Could not load tokenizer for '{model_path}' or '{model_name}'. "
            "Please install 'transformers' or 'tokenizers'."
        )
    return tokenizer


def encode_text(tokenizer, text: str, is_first: bool = False):
    if hasattr(tokenizer, "encode"):
        res = tokenizer.encode(text)
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
            and hasattr(tokenizer, "bos_token_id")
            and tokenizer.bos_token_id is not None
        ):
            if not tokens or tokens[0] != tokenizer.bos_token_id:
                tokens = [tokenizer.bos_token_id] + tokens
        return tokens
    raise ValueError("Unsupported tokenizer type")


def decode_tokens(tokenizer, token_ids: list[int]) -> str:
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids)
    raise ValueError("Unsupported tokenizer type")


def main():
    parser = argparse.ArgumentParser(
        description="Run AutoRegressive LLM Interactive Chat"
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
        args.model, args.model_path, delegate, min_compile_time=args.min_compile_time
    )

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = load_tokenizer(args.model_path, args.model)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    conversation_tokens = []
    print("\nChat initialized. Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            user_input = input("User: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break

        if user_input.strip().lower() in ["exit", "quit"]:
            print("Exiting chat.")
            break

        if not user_input.strip():
            continue

        is_first = len(conversation_tokens) == 0
        user_tokens = encode_text(tokenizer, user_input, is_first=is_first)
        conversation_tokens.extend(user_tokens)
        print(f"Bot: ", end="")

        for _ in range(args.tokens):
            next_token = session.generate_step(conversation_tokens)
            if next_token == -1:
                print("\n[Max sequence length reached]")
                break
            conversation_tokens.append(next_token)
            if eos_token_id is not None and next_token == eos_token_id:
                break
            print(decode_tokens(tokenizer, [next_token]), end="")


if __name__ == "__main__":
    main()
