"""Run the local EmbeddingGemma 2 reference model on each supported modality."""

import argparse
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
DEFAULT_MODEL_PATH = Path("/mnt/bigdrive/models/google/embeddinggemma-2")
TEXT = "task: search result | query: A small orange animal in a sunny landscape."
MODALITIES = ("text", "image", "video", "audio")


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "modality",
        nargs="?",
        choices=("all", *MODALITIES),
        default="all",
        help="Modality to run (default: all)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Local model directory (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument("--image", type=Path, default=ASSETS / "sample.png")
    parser.add_argument("--video", type=Path, default=ASSETS / "sample.mp4")
    parser.add_argument("--audio", type=Path, default=ASSETS / "sample.wav")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device (default: CUDA when available, otherwise CPU)",
    )
    return parser.parse_args()


def encodeModality(model: SentenceTransformer, modality: str, args: argparse.Namespace) -> torch.Tensor:
    if modality == "text":
        return model.encode(TEXT, prompt_name="SearchQuery", convert_to_tensor=True)

    media_path = getattr(args, modality)
    if not media_path.is_file():
        raise FileNotFoundError(
            f"Missing {modality} sample {media_path}. "
            "Run .venv/bin/python tests/embeddinggemma-2/make_samples.py first."
        )
    return model.encode({modality: str(media_path)}, convert_to_tensor=True)


def main() -> None:
    args = parseArgs()
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {args.model_path}")

    selected_modalities = MODALITIES if args.modality == "all" else (args.modality,)
    model = SentenceTransformer(str(args.model_path), device=args.device)

    for modality in selected_modalities:
        embedding = encodeModality(model, modality, args)
        print(
            f"{modality}: shape={tuple(embedding.shape)}, "
            f"norm={embedding.float().norm().item():.6f}, "
            f"first_values={embedding.float().flatten()[:8].tolist()}"
        )


if __name__ == "__main__":
    main()
