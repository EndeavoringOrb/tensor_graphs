# txt2img.py
import argparse
import time
from pathlib import Path

import numpy as np
import tensor_graphs
import torch
from PIL import Image
from safetensors.torch import load_file

from train import CostPredictorDelegate, CostPredictorRNN, TrainConfig
from utils.decode import load_tokenizer

PROMPT_TEMPLATE_ENCODE_PREFIX = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
PROMPT_TEMPLATE_ENCODE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


def encode_prompt(tokenizer_obj, prompt: str, max_seq_len: int = 128) -> list[int]:
    raw_tok = tokenizer_obj[0] if isinstance(tokenizer_obj, tuple) else tokenizer_obj
    full_prompt = PROMPT_TEMPLATE_ENCODE_PREFIX + prompt

    if hasattr(raw_tok, "encode"):
        enc_prefix = raw_tok.encode(PROMPT_TEMPLATE_ENCODE_PREFIX)
        prefix_ids = enc_prefix.ids if hasattr(enc_prefix, "ids") else enc_prefix
        prefix_len = len(prefix_ids)

        enc = raw_tok.encode(full_prompt)
        token_ids = enc.ids if hasattr(enc, "ids") else enc
    else:
        raise ValueError("Unsupported tokenizer object")

    token_ids = token_ids[prefix_len:]

    enc_suf = raw_tok.encode(PROMPT_TEMPLATE_ENCODE_SUFFIX)
    suffix_ids = enc_suf.ids if hasattr(enc_suf, "ids") else enc_suf
    token_ids = list(token_ids) + list(suffix_ids)

    if len(token_ids) < max_seq_len:
        token_ids = token_ids + [0] * (max_seq_len - len(token_ids))
    else:
        token_ids = token_ids[:max_seq_len]

    return token_ids


def main():
    parser = argparse.ArgumentParser(
        description="Krea 2 Turbo Text-to-Image Generation (Unified Unrolled Pipeline)"
    )
    parser.add_argument(
        "--prompt_file",
        type=Path,
        default=Path("prompt.txt"),
        help="Path to .txt file containing the prompt",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/krea/Krea-2-Turbo/krea.safetensors",
        help="Path to Krea-2-Turbo model directory or checkpoint",
    )
    parser.add_argument(
        "--text-encoder-path",
        type=str,
        default="models/krea/Krea-2-Turbo/qwen3vl_4b_bf16.safetensors",
        help="Path to Qwen3-VL text encoder checkpoint",
    )
    parser.add_argument(
        "--vae-path",
        type=str,
        default="models/krea/Krea-2-Turbo/qwen_image_vae.safetensors",
        help="Path to Qwen Image VAE checkpoint",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to run directory (e.g. runs/0) to load the cost predictor model from",
    )
    parser.add_argument(
        "--min-compile-time",
        type=float,
        default=0.0,
        help="Minimum required compile time per bucket in seconds",
    )
    parser.add_argument(
        "--disable-caching",
        action="store_true",
        help="Disable dirty region session caching",
    )
    parser.add_argument(
        "--log-cost-calls",
        action="store_true",
        default=True,
        help="Log cost model calls to benchmarks/calls.bin (default: True)",
    )
    parser.add_argument(
        "--no-log-cost-calls",
        dest="log_cost_calls",
        action="store_false",
        help="Disable logging cost model calls to benchmarks/calls.bin",
    )
    parser.add_argument(
        "--height", type=int, default=512, help="Output image height (divisible by 16)"
    )
    parser.add_argument(
        "--width", type=int, default=512, help="Output image width (divisible by 16)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Number of flow-matching inference steps (default: 8)",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=1.15,
        help="Timestep schedule shift parameter (default: 1.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for latent noise initialization",
    )
    parser.add_argument(
        "--output", type=str, default="output.png", help="Path to save generated image"
    )
    parser.add_argument(
        "--threads", type=int, default=0, help="C++ execution threads (0 = auto-detect)"
    )
    args = parser.parse_args()

    if args.threads > 0:
        tensor_graphs.set_num_threads(args.threads)

    if not args.prompt_file.is_file():
        raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")
    prompt = args.prompt_file.read_text(encoding="utf-8").strip()

    print("=========================================================")
    print(" Krea 2 Turbo (Unified Pipeline: Qwen3-VL + 12B DiT + VAE)")
    print(f" Prompt File: {args.prompt_file}")
    print(f" Prompt: {prompt!r}")
    print(
        f" Resolution: {args.width}x{args.height} | Steps: {args.steps} | Shift mu: {args.mu}"
    )
    print("=========================================================")

    # Initialize search agent delegate
    if args.run_dir:
        run_dir_path = Path(args.run_dir)
        config_file = run_dir_path / "config.json"
        cfg = TrainConfig()
        if config_file.exists():
            try:
                cfg = TrainConfig.load(config_file)
                print(f"[Txt2Img] Loaded config from {config_file}")
            except Exception as e:
                print(
                    f"[Txt2Img] Warning: Failed to load config from {config_file}: {e}"
                )

        model_file = run_dir_path / "model.safetensors"
        if model_file.exists():
            state_dict = load_file(model_file)
            model = CostPredictorRNN(hidden_dim=cfg.hidden_dim)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            delegate = CostPredictorDelegate(
                model=model,
                epsilon=0.0,
                is_training=False,
            )
            print(f"[Txt2Img] Loaded trained CostPredictorRNN agent from {model_file}")
        else:
            print(
                f"[Txt2Img] Warning: Model file not found at {model_file}, using HeuristicSearchDelegate."
            )
            delegate = tensor_graphs.HeuristicSearchDelegate()
    else:
        print(
            "[Txt2Img] No --run-dir specified. Using HeuristicSearchDelegate (node caching disabled)."
        )
        delegate = tensor_graphs.HeuristicSearchDelegate()
        args.disable_caching = True

    session = tensor_graphs.Krea2Session(
        model_path=args.model_path,
        text_encoder_path=args.text_encoder_path,
        vae_path=args.vae_path,
        height=args.height,
        width=args.width,
        text_seq_len=128,
        steps=args.steps,
        mu=args.mu,
        delegate=delegate,
        min_compile_time=args.min_compile_time,
        disable_caching=args.disable_caching,
        threads=args.threads,
        log_cost_calls=args.log_cost_calls,
    )

    tok_path = args.text_encoder_path if args.text_encoder_path else args.model_path
    tokenizer = load_tokenizer([tok_path, "Qwen/Qwen3-VL-4B-Instruct"])
    token_ids = encode_prompt(tokenizer, prompt, max_seq_len=128)

    latent_h = args.height // 8
    latent_w = args.width // 8
    torch.manual_seed(args.seed)
    latent = torch.randn((1, 16, latent_h, latent_w), dtype=torch.float32)

    print(f"\nGenerating image with {args.steps} unrolled flow-matching steps...")
    t_start = time.perf_counter()
    pixels = session.generate_image(token_ids, latent.flatten().tolist())
    t_total = time.perf_counter() - t_start
    print(f"End-to-end generation complete in {t_total * 1000:.2f} ms")

    image_tensor = torch.tensor(pixels, dtype=torch.float32).reshape(
        1, 3, args.height, args.width
    )
    image_tensor = torch.clamp((image_tensor + 1.0) / 2.0, 0.0, 1.0)
    image_np = (image_tensor[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

    img = Image.fromarray(image_np)
    img.save(args.output)
    print(f"\nImage successfully saved to: {args.output}")


if __name__ == "__main__":
    main()
