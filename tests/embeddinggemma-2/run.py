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
    parser.add_argument(
        "--min-compile-time",
        type=float,
        default=1.0,
        help="Minimum required compile time per bucket in seconds",
    )
    parser.add_argument(
        "--saturate-only",
        action="store_true",
        help="Compile the selected modality/configuration graphs without running embeddings",
    )
    parser.add_argument(
        "--cache-file",
        "--cache",
        dest="cache_file",
        type=str,
        default="",
        help="Path to compiled cache file. If specified, enables compilation caching to/from this file.",
    )
    return parser.parse_args()


def prepareText(processor: AutoProcessor) -> dict:
    batch = processor(text=[SEARCH_TEXT], return_tensors="pt")
    return {"token_ids": batch["input_ids"][0].to(torch.int32)}


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
        "media": features.float().contiguous(),
        "mel_frames": valid_frames,
    }


def prepareVideo(processor: AutoProcessor, video_path: Path) -> dict:
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    try:
        import av
    except ImportError as error:
        raise RuntimeError("Video mode requires PyAV; install the embeddinggemma dependencies") from error
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"Video file has no video stream: {video_path}")
        stream = container.streams.video[0]
        source_fps = stream.average_rate or stream.base_rate or stream.guessed_rate
        if source_fps is None or float(source_fps) <= 0.0:
            raise ValueError("Could not determine video frame rate")
        target_fps = float(processor.video_processor.fps or 1.0)
        stride = max(1, round(float(source_fps) / target_fps))
        frames = [
            torch.from_numpy(frame.to_ndarray(format="rgb24"))
            for frame_index, frame in enumerate(container.decode(video=0))
            if frame_index % stride == 0
        ]
    if not frames:
        raise ValueError(f"Video file contains no decodable frames: {video_path}")
    max_frames = getattr(processor.video_processor, "max_frames", None)
    if max_frames and len(frames) > max_frames:
        selected = torch.linspace(0, len(frames) - 1, max_frames).round().to(torch.int64).tolist()
        frames = [frames[index] for index in selected]
    frames = torch.stack(frames)
    batch = processor(videos=[frames], do_sample_frames=False, return_tensors="pt")
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
        "media": pixels.float().contiguous(),
        "patch_count": int(pixels.shape[1]),
        "patch_grid_width": grid_width,
        "video_frames": int(pixels.shape[0]),
        "video": frames,
    }


def prepareNativeInput(processor: AutoProcessor, modality: str, args: argparse.Namespace) -> dict:
    if modality == "text":
        return prepareText(processor)
    if modality == "image":
        return prepareImage(processor, args.image)
    if modality == "video":
        return prepareVideo(processor, args.video)
    return prepareAudio(processor, args.audio)


def encodeReference(model: SentenceTransformer, modality: str, args: argparse.Namespace,
                    native_input: dict) -> torch.Tensor:
    if modality == "text":
        return model.encode(TEXT, prompt_name="SearchQuery", convert_to_tensor=True)
    if modality == "video":
        return model.encode(
            {"video": native_input["video"]},
            processing_kwargs={"video": {"do_sample_frames": False}},
            convert_to_tensor=True,
        )
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
                       reference_model: SentenceTransformer | None, processor: AutoProcessor) -> None:
    options = CONFIGS[config_name]
    native_input = prepareNativeInput(processor, modality, args)
    token_ids = native_input["token_ids"].tolist()
    kwargs = {
        "token_ids": token_ids,
        "cache_file": args.cache_file,
        "compile_no_weights_bucket": options["compile_no_weights_bucket"],
        "compile_dirty_input_bucket": options["compile_dirty_input_bucket"],
        "disable_compilation_caching": args.saturate_only or not bool(args.cache_file),
        "min_compile_seconds": args.min_compile_time,
    }
    for shape_key in ("patch_count", "patch_grid_width", "mel_frames", "video_frames"):
        if shape_key in native_input:
            kwargs[shape_key] = native_input[shape_key]
    session = tensor_graphs.EmbeddingGemma2(str(args.model_path), modality, **kwargs)
    if args.saturate_only:
        return

    reference = encodeReference(reference_model, modality, args, native_input)
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
    reference_model = (
        None if args.saturate_only
        else SentenceTransformer(str(args.model_path), device=args.device)
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    for config_name in config_names:
        configured_modality = CONFIGS[config_name]["modality"]
        modalities = (configured_modality,) if configured_modality else selected_modalities
        for modality in modalities:
            if modality not in selected_modalities:
                continue
            print(f"\n=== EmbeddingGemma 2 {config_name}: {modality} ===", flush=True)
            runEmbeddingGemma2(config_name, modality, args, reference_model, processor)

    if args.saturate_only:
        print("Saturate-only complete. Selected graphs compiled without running embeddings.", flush=True)


if __name__ == "__main__":
    main()
