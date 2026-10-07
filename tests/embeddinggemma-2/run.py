"""Compare native TensorGraphs and Sentence Transformers EmbeddingGemma 2 outputs."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import soundfile as sf
import tensor_graphs
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision.io import read_video
from transformers import AutoProcessor


HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "google" / "embeddinggemma-2"
TEXT = "A small orange animal in a sunny landscape."
SEARCH_TEXT = f"task: search result | query: {TEXT}"
MODALITIES = ("text", "image", "video", "audio")
CONFIGS = {
    "full": {"modality": None, "compile_no_weights_bucket": False, "compile_dirty_input_bucket": False},
    "no-weights": {"modality": None, "compile_no_weights_bucket": True, "compile_dirty_input_bucket": False},
    "text-dirty": {"modality": "text", "compile_no_weights_bucket": False, "compile_dirty_input_bucket": True},
    "image-dirty": {"modality": "image", "compile_no_weights_bucket": False, "compile_dirty_input_bucket": True},
    "video-dirty": {"modality": "video", "compile_no_weights_bucket": False, "compile_dirty_input_bucket": True},
    "audio-dirty": {"modality": "audio", "compile_no_weights_bucket": False, "compile_dirty_input_bucket": True},
}
MEDIA_TOKEN_IDS = {"image": 258880, "video": 258884, "audio": 258881}


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modality", nargs="?", choices=("all", *MODALITIES), default="all")
    parser.add_argument("--config", choices=("all", *CONFIGS), default="all")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image", type=Path, default=ASSETS / "sample.png")
    parser.add_argument("--video", type=Path, default=ASSETS / "sample.mp4")
    parser.add_argument("--audio", type=Path, default=ASSETS / "sample.wav")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cosine-tolerance", type=float, default=0.99)
    return parser.parse_args()


def mediaPositions(input_ids: torch.Tensor, modality: str) -> list[int]:
    token_id = MEDIA_TOKEN_IDS[modality]
    return torch.nonzero(input_ids.reshape(-1) == token_id, as_tuple=False).reshape(-1).tolist()


def prepareText(processor: AutoProcessor) -> dict:
    batch = processor(text=[SEARCH_TEXT], return_tensors="pt")
    return {"token_ids": batch["input_ids"][0].to(torch.int32), "positions": []}


def prepareImage(processor: AutoProcessor, image_path: Path) -> dict:
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as image:
        batch = processor(images=[image.convert("RGB")], return_tensors="pt")
    positions = batch["image_position_ids"][0]
    valid = (positions >= 0).all(dim=-1)
    pixels = batch["pixel_values"][0][valid]
    valid_positions = positions[valid]
    grid_width = int(valid_positions[:, 0].max().item()) + 1
    return {
        "token_ids": batch["input_ids"][0].to(torch.int32),
        "positions": mediaPositions(batch["input_ids"][0], "image"),
        "media": pixels.float().contiguous(),
        "patch_count": int(pixels.shape[0]),
        "patch_grid_width": grid_width,
    }


def prepareAudio(processor: AutoProcessor, audio_path: Path) -> dict:
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    waveform, sample_rate = sf.read(audio_path, dtype="float32")
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if sample_rate != 16000:
        raise ValueError(f"Audio sample rate must be 16 kHz; got {sample_rate} Hz")
    batch = processor(audio=[waveform], sampling_rate=sample_rate, return_tensors="pt")
    valid_frames = int(batch["input_features_mask"][0].sum().item())
    features = batch["input_features"][0, :valid_frames]
    return {
        "token_ids": batch["input_ids"][0].to(torch.int32),
        "positions": mediaPositions(batch["input_ids"][0], "audio"),
        "media": features.float().contiguous(),
        "mel_frames": valid_frames,
    }


def prepareVideo(processor: AutoProcessor, video_path: Path) -> dict:
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    decoded, _, metadata = read_video(str(video_path), pts_unit="sec")
    fps = float(metadata.get("video_fps", 0.0))
    if fps <= 0.0:
        raise ValueError("Could not determine video frame rate")
    # The checkpoint's default video sampling rate is 1 frame per second.
    stride = max(1, round(fps))
    frames = decoded[::stride]
    batch = processor(videos=[frames], return_tensors="pt")
    pixels = batch["pixel_values_videos"]
    positions = batch["video_position_ids"]
    valid = (positions >= 0).all(dim=-1)
    per_frame_counts = valid.sum(dim=-1)
    if not torch.all(per_frame_counts == per_frame_counts[0]):
        raise ValueError("Native video embedding currently requires equal patch counts for sampled frames")
    pixels = torch.stack([pixels[i][valid[i]] for i in range(pixels.shape[0])])
    valid_positions = positions[0][valid[0]]
    grid_width = int(valid_positions[:, 0].max().item()) + 1
    return {
        "token_ids": batch["input_ids"][0].to(torch.int32),
        "positions": mediaPositions(batch["input_ids"][0], "video"),
        "media": pixels.float().contiguous(),
        "patch_count": int(pixels.shape[1]),
        "patch_grid_width": grid_width,
        "video_frames": int(pixels.shape[0]),
    }


def prepareNativeInput(processor: AutoProcessor, modality: str, args: argparse.Namespace) -> dict:
    if modality == "text":
        return prepareText(processor)
    if modality == "image":
        return prepareImage(processor, args.image)
    if modality == "video":
        return prepareVideo(processor, args.video)
    return prepareAudio(processor, args.audio)


def encodeReference(model: SentenceTransformer, modality: str, args: argparse.Namespace) -> torch.Tensor:
    if modality == "text":
        return model.encode(TEXT, prompt_name="SearchQuery", convert_to_tensor=True)
    media_path = getattr(args, modality)
    return model.encode({modality: str(media_path)}, convert_to_tensor=True)


def compareEmbedding(native_values: list[float], reference: torch.Tensor, modality: str,
                     config_name: str, cosine_tolerance: float) -> None:
    native = torch.tensor(native_values, dtype=torch.float32).reshape(1, -1)
    expected = reference.detach().float().cpu().reshape(1, -1)
    if native.shape != expected.shape:
        raise AssertionError(f"{config_name}/{modality}: shape {tuple(native.shape)} != {tuple(expected.shape)}")
    cosine = torch.nn.functional.cosine_similarity(native, expected).item()
    max_abs = (native - expected).abs().max().item()
    print(f"{modality}: cosine={cosine:.7f}, max_abs={max_abs:.7f}", flush=True)
    if cosine < cosine_tolerance:
        raise AssertionError(
            f"{config_name}/{modality}: cosine {cosine:.7f} is below {cosine_tolerance:.7f}"
        )


def runEmbeddingGemma2(config_name: str, modality: str, args: argparse.Namespace,
                       reference_model: SentenceTransformer, processor: AutoProcessor) -> None:
    options = CONFIGS[config_name]
    native_input = prepareNativeInput(processor, modality, args)
    reference = encodeReference(reference_model, modality, args)
    token_ids = native_input["token_ids"].tolist()
    kwargs = {
        "sequence_length": len(token_ids),
        "patch_count": native_input.get("patch_count", 2520),
        "patch_grid_width": native_input.get("patch_grid_width", 60),
        "mel_frames": native_input.get("mel_frames", 280),
        "video_frames": native_input.get("video_frames", 1),
        "media_placeholder_positions": native_input["positions"],
        "compile_no_weights_bucket": options["compile_no_weights_bucket"],
        "compile_dirty_input_bucket": options["compile_dirty_input_bucket"],
    }
    session = tensor_graphs.EmbeddingGemma2(str(args.model_path), modality, **kwargs)
    if modality == "text":
        output = session.embed_text(token_ids)
    else:
        media = native_input["media"].reshape(-1).tolist()
        output = session.embed_media(media, token_ids)
    compareEmbedding(output, reference, modality, config_name, args.cosine_tolerance)


def main() -> None:
    args = parseArgs()
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {args.model_path}")
    selected_modalities = MODALITIES if args.modality == "all" else (args.modality,)
    config_names = list(CONFIGS) if args.config == "all" else [args.config]
    reference_model = SentenceTransformer(str(args.model_path), device=args.device)
    processor = AutoProcessor.from_pretrained(args.model_path)

    for config_name in config_names:
        configured_modality = CONFIGS[config_name]["modality"]
        modalities = (configured_modality,) if configured_modality else selected_modalities
        for modality in modalities:
            if modality not in selected_modalities:
                continue
            print(f"\n=== EmbeddingGemma 2 {config_name}: {modality} ===", flush=True)
            runEmbeddingGemma2(config_name, modality, args, reference_model, processor)


if __name__ == "__main__":
    main()
