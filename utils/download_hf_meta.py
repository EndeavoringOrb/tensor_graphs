import json
import struct
import argparse
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import (
    HfApi,
    hf_hub_download,
    get_token,
    ModelInfo,
    get_safetensors_metadata,
)


def format_size(size_bytes):
    """Formats bytes into a human-readable string."""
    if size_bytes is None:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def serialize_safetensors_header(file_metadata) -> bytes:
    """
    Constructs the exact binary prefix for a .safetensors file.
    """
    header_dict = {}

    # 1. Reconstruct global metadata if it exists
    if file_metadata.metadata:
        header_dict["__metadata__"] = file_metadata.metadata

    # 2. Populate the tensor definitions
    for tensor_name, tensor_info in file_metadata.tensors.items():
        header_dict[tensor_name] = {
            "dtype": tensor_info.dtype,
            "shape": tensor_info.shape,
            "data_offsets": list(tensor_info.data_offsets),
        }

    # 3. Serialize to a compact UTF-8 JSON payload
    json_bytes = json.dumps(header_dict, separators=(",", ":")).encode("utf-8")

    # 4. Apply standard padding so the header size aligns with 8-byte boundaries
    remainder = len(json_bytes) % 8
    if remainder > 0:
        padding_size = 8 - remainder
        json_bytes += b" " * padding_size

    # 5. Pack the final JSON size as a 64-bit little-endian uint64
    header_size_bytes = struct.pack("<Q", len(json_bytes))

    # 6. Combine the 8-byte prefix and the padded JSON block
    return header_size_bytes + json_bytes


def main():
    parser = argparse.ArgumentParser(
        description="Download Hugging Face model metadata and headers of .safetensors files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="Hugging Face repository ID (e.g., Qwen/Qwen3.6-35B-A3B)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="models_meta",
        help="Base directory to save the model",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Model revision / branch / commit hash",
    )
    parser.add_argument(
        "--download-other-files",
        action="store_true",
        help="Fully download non-safetensors metadata files (configs, tokenizers, etc.)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without saving files or downloading actual data",
    )
    args = parser.parse_args()

    token = get_token()
    api = HfApi(token=token)

    print(
        f"Retrieving file list and metadata for '{args.repo_id}' ({args.revision})..."
    )
    repo_info = api.repo_info(
        repo_id=args.repo_id,
        revision=args.revision,
        files_metadata=True,
        token=token,
    )
    assert isinstance(repo_info, ModelInfo), "Repo type must be model"
    siblings = repo_info.siblings or []
    files = [s.rfilename for s in siblings]
    file_sizes = {s.rfilename: (s.size or 0) for s in siblings}

    local_dir = Path(args.dir) / args.repo_id
    print(f"Target local directory: {local_dir}")
    print(f"Found {len(files)} files in repository.")

    # Categorize files
    safetensors_files = [f for f in files if f.endswith(".safetensors")]
    other_files = [f for f in files if not f.endswith(".safetensors")]

    print(f"  - .safetensors files (headers only): {len(safetensors_files)}")
    print(
        f"  - Other files (full download): {len(other_files)} "
        f"{'(will be downloaded)' if args.download_other_files else '(will be skipped)'}"
    )
    to_process_files = safetensors_files + (
        other_files if args.download_other_files else []
    )

    # Fetch metadata for all safetensors files upfront in a single call
    safetensors_metadata = None
    if safetensors_files:
        print("Fetching safetensors metadata...")
        safetensors_metadata = get_safetensors_metadata(
            args.repo_id, revision=args.revision, token=token
        )

    if args.dry_run:
        print("\n--- DRY RUN ACTIVE: No files will be modified or fully downloaded ---")

    total_downloaded_bytes = 0

    for filename in tqdm(to_process_files, desc="Processing model files", unit="file"):
        local_path = local_dir / filename

        # Skip if file already exists locally
        if local_path.exists():
            tqdm.write(
                f"Already exists: {filename} ({format_size(local_path.stat().st_size)})"
            )
            continue

        is_safetensors = filename.endswith(".safetensors")
        file_type = "safetensors header" if is_safetensors else "file"

        if is_safetensors:
            if (
                safetensors_metadata is None
                or filename not in safetensors_metadata.files_metadata
            ):
                tqdm.write(f"Warning: No metadata found for {filename}. Skipping.")
                continue

            file_meta = safetensors_metadata.files_metadata[filename]
            header_bytes = serialize_safetensors_header(file_meta)
            size_bytes = len(header_bytes)

            if args.dry_run:
                tqdm.write(
                    f"[DRY RUN] Would generate {file_type}: {filename} "
                    f"({format_size(size_bytes)})"
                )
            else:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(header_bytes)
                tqdm.write(
                    f"Generated {file_type}: {filename} ({format_size(size_bytes)})"
                )
        else:
            size_bytes = file_sizes.get(filename, 0)
            if args.dry_run:
                tqdm.write(
                    f"[DRY RUN] Would download {file_type}: {filename} "
                    f"({format_size(size_bytes)})"
                )
            else:
                hf_hub_download(
                    repo_id=args.repo_id,
                    filename=filename,
                    local_dir=local_dir,
                    revision=args.revision,
                    token=token,
                )
                size_bytes = local_path.stat().st_size
                tqdm.write(
                    f"Downloaded {file_type}: {filename} ({format_size(size_bytes)})"
                )

        total_downloaded_bytes += size_bytes

    print("\nDownload process completed.")
    status_label = "simulated download" if args.dry_run else "downloaded"
    print(f"Total {status_label} size: {format_size(total_downloaded_bytes)}")


if __name__ == "__main__":
    main()
