#!/usr/bin/env python3
import argparse
import os
import sys


def load_tokenizer(model_name_or_path: str):
    """
    Attempts to load the tokenizer first using the standard 'transformers' library,
    and falls back to the native 'tokenizers' library if 'transformers' is not available.
    Supports local paths as well as remote Hugging Face Hub repository IDs.
    """
    tokenizer = None
    tokenizer_type = None

    # Method 1: Try using native tokenizers (fast Rust-backed library)
    try:
        from tokenizers import Tokenizer

        print(f"Loading '{model_name_or_path}' via tokenizers.Tokenizer...")

        # Check if model_name_or_path points to a local directory or file
        if os.path.exists(model_name_or_path):
            if os.path.isdir(model_name_or_path):
                json_path = os.path.join(model_name_or_path, "tokenizer.json")
                if os.path.exists(json_path):
                    tokenizer = Tokenizer.from_file(json_path)
                else:
                    raise FileNotFoundError(
                        f"tokenizer.json not found in directory '{model_name_or_path}'"
                    )
            else:
                tokenizer = Tokenizer.from_file(model_name_or_path)
        else:
            # Fallback to downloading directly from Hugging Face Hub
            tokenizer = Tokenizer.from_pretrained(model_name_or_path)

        tokenizer_type = "tokenizers"
        return tokenizer, tokenizer_type
    except ImportError:
        pass
    except Exception as e:
        print(f"Could not load via tokenizers: {e}")

    # Method 2: Try using transformers (recommended for online model IDs)
    try:
        from transformers import AutoTokenizer

        print(f"Loading '{model_name_or_path}' via transformers.AutoTokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer_type = "transformers"
        return tokenizer, tokenizer_type
    except ImportError:
        print(
            "Error: Neither 'transformers' nor 'tokenizers' libraries are installed.",
            file=sys.stderr,
        )
        print(
            "Please install at least one: 'pip install tokenizers' or 'pip install transformers'",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(
            f"Failed to load tokenizer using native tokenizers library: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def parse_token_list(value: str) -> list[int]:
    """
    Validates and parses a comma-separated string of integers into a Python list of ints.
    """
    try:
        return [int(t.strip()) for t in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Token list must be comma-separated integers (e.g. 24227,220,16)."
        )


def main():
    # 1. Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Decode token IDs and encode text using 'transformers' or native 'tokenizers' libraries."
    )

    # We use nargs='?' to make these arguments optional positional,
    # preserving the sequential syntax of your original script.
    parser.add_argument(
        "model",
        nargs="?",
        default="Qwen/Qwen3.6-35B-A3B",
        help="Hugging Face model ID or path to a local directory/file (default: Qwen/Qwen3.6-35B-A3B)",
    )
    parser.add_argument(
        "tokens",
        nargs="?",
        type=parse_token_list,
        default=[24227, 220, 16, 198],
        help="Comma-separated integer token IDs to decode (default: 24227,220,16,198)",
    )
    parser.add_argument(
        "text",
        nargs="?",
        default="Chapter 1\n",
        help="Test text to encode into token IDs (default: 'Chapter 1\\n')",
    )

    args = parser.parse_args()

    # 2. Load tokenizer
    tokenizer, tok_type = load_tokenizer(args.model)
    print(f"Successfully loaded tokenizer using backend: {tok_type}\n")

    # 3. Decode the full sequence of tokens
    try:
        output_text = tokenizer.decode(args.tokens)
        print(f"Tokens to decode: {args.tokens}")
        print(
            f"Decoded Text: {output_text!r}"
        )  # repr() helps visually identify newlines (\n) or trailing spaces
    except Exception as e:
        print(f"Error decoding tokens: {e}", file=sys.stderr)

    # 4. Decode individual tokens
    print("\nIndividual Token Mapping:")
    for tok in args.tokens:
        try:
            if tok_type == "transformers":
                decoded_tok = tokenizer.decode([tok], skip_special_tokens=False)
            else:
                decoded_tok = tokenizer.decode([tok], False)
            print(f"  {tok} = {decoded_tok!r}")
        except Exception as e:
            print(f"  {tok} = <Error decoding: {e}>")

    # 5. Encode the test text
    print(f"\nEncoding test text: {args.text!r}")
    try:
        encoded_obj = tokenizer.encode(args.text)
        # Check if the returned object has an .ids attribute (native Tokenizer)
        # or is a raw list (transformers AutoTokenizer)
        if hasattr(encoded_obj, "ids"):
            input_ids = encoded_obj.ids
        else:
            input_ids = encoded_obj
        print(f"Generated Token IDs: {input_ids}")
    except Exception as e:
        print(f"Error encoding test text: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""
Qwen/Qwen3.6-35B-A3B 24227 -> 220,16,25,27416,310,279,10776
google/gemma-3-270m 2,9259 -> 236888,564,236789,236757,9775,531
"""
