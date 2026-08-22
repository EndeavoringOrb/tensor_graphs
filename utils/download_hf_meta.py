import argparse
import json
import re
import struct
from pathlib import Path
from urllib.parse import unquote, urlparse

from huggingface_hub import (
    HfApi,
    ModelInfo,
    get_safetensors_metadata,
    get_token,
    hf_hub_download,
)
from tqdm import tqdm


def format_size(size_bytes):
    """Formats bytes into a human-readable string."""
    if size_bytes is None:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def parse_hf_target(target: str):
    """
    Parses a Hugging Face repo_id, repo URL, or direct file URL.

    Returns:
        tuple: (repo_id, revision, filename)
    """
    target = target.strip()

    # Check if target is a URL
    if target.startswith("http://") or target.startswith("https://"):
        parsed = urlparse(target)
        path = unquote(parsed.path.strip("/"))

        # Remove 'datasets/' or 'spaces/' prefix if present, default assumption is model repo
        if path.startswith("datasets/"):
            path = path[len("datasets/") :]
        elif path.startswith("spaces/"):
            path = path[len("spaces/") :]

        # Match patterns like:
        # - org/repo/blob/revision/path/to/file.safetensors
        # - org/repo/resolve/revision/path/to/file.safetensors
        # - org/repo/tree/revision
        match = re.match(r"^(.+?)/(blob|resolve|raw|tree)/([^/]+)(?:/(.+))?$", path)
        if match:
            repo_id, action, revision, filename = match.groups()
            if action == "tree":
                return repo_id, revision, None
            return repo_id, revision, filename
        else:
            # URL is just the repo root: https://huggingface.co/org/repo
            return path, None, None

    # Plain repo_id (e.g., "meta-llama/Llama-3-8B")
    return target, None, None


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


def download_model_meta(
    target: str,
    base_dir: str = "models",
    revision: str = None,
    download_other_files: bool = True,
    dry_run: bool = False,
):
    parsed_repo_id, parsed_revision, target_filename = parse_hf_target(target)

    # Priority: Explicit CLI revision > URL revision > Default "main"
    effective_revision = revision or parsed_revision or "main"
    repo_id = parsed_repo_id

    token = get_token()
    api = HfApi(token=token)

    local_dir = Path(base_dir) / repo_id
    print(f"Target repository: '{repo_id}' ({effective_revision})")
    print(f"Target local directory: {local_dir}")

    # Case 1: Single file target
    if target_filename:
        print(f"Target file specified: {target_filename}")
        is_safetensors = target_filename.endswith(".safetensors")
        safetensors_files = [target_filename] if is_safetensors else []
        other_files = [target_filename] if not is_safetensors else []
        to_process_files = [target_filename]
        file_sizes = {}
    # Case 2: Entire repository target
    else:
        print(f"Retrieving file list and metadata for '{repo_id}'...")
        repo_info = api.repo_info(
            repo_id=repo_id,
            revision=effective_revision,
            files_metadata=True,
            token=token,
        )
        assert isinstance(repo_info, ModelInfo), "Repo type must be model"
        siblings = repo_info.siblings or []
        files = [s.rfilename for s in siblings]
        file_sizes = {s.rfilename: (s.size or 0) for s in siblings}

        print(f"Found {len(files)} files in repository.")
        safetensors_files = [f for f in files if f.endswith(".safetensors")]
        other_files = [f for f in files if not f.endswith(".safetensors")]

        print(f"  - .safetensors files (headers only): {len(safetensors_files)}")
        print(
            f"  - Other files (full download): {len(other_files)} "
            f"{'(will be downloaded)' if download_other_files else '(will be skipped)'}"
        )
        to_process_files = safetensors_files + (
            other_files if download_other_files else []
        )

    # Fetch safetensors metadata if needed
    safetensors_metadata = None
    if safetensors_files:
        print("Fetching safetensors metadata...")
        try:
            safetensors_metadata = get_safetensors_metadata(
                repo_id, revision=effective_revision, token=token
            )
        except Exception as e:
            print(f"Error fetching safetensors metadata: {e}")

    if dry_run:
        print("\n--- DRY RUN ACTIVE: No files will be modified or fully downloaded ---")

    total_downloaded_bytes = 0

    for filename in tqdm(to_process_files, desc="Processing files", unit="file"):
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

            if dry_run:
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
            if dry_run:
                tqdm.write(
                    f"[DRY RUN] Would download {file_type}: {filename} "
                    f"({format_size(size_bytes)})"
                )
            else:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=local_dir,
                    revision=effective_revision,
                    token=token,
                )
                size_bytes = local_path.stat().st_size
                tqdm.write(
                    f"Downloaded {file_type}: {filename} ({format_size(size_bytes)})"
                )

        total_downloaded_bytes += size_bytes

    print("\nProcess completed.")
    status_label = "simulated download" if dry_run else "downloaded"
    print(f"Total {status_label} size: {format_size(total_downloaded_bytes)}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Hugging Face model metadata or headers for .safetensors files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "target",
        type=str,
        help="Hugging Face repo_id (e.g. 'Qwen/Qwen2.5-7B'), repo URL, or direct file URL",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="models",
        help="Base directory to save the model",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision / branch / commit hash (overrides URL revision)",
    )
    parser.add_argument(
        "--download-other-files",
        action="store_true",
        help="Fully download non-safetensors metadata files (configs, tokenizers, etc.) when processing a full repo",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without saving files or downloading actual data",
    )
    args = parser.parse_args()

    download_model_meta(
        target=args.target,
        base_dir=args.dir,
        revision=args.revision,
        download_other_files=args.download_other_files,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
