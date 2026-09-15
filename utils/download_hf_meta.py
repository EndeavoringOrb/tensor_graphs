# File: utils/download_hf_meta.py
import argparse
import json
import os
import re
import struct
import sys
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import format_size


def parse_hf_target(target: str):
    target = target.strip()
    if target.startswith(("http://", "https://")):
        path = unquote(urlparse(target).path.strip("/"))
        for prefix in ["datasets/", "spaces/"]:
            path = path.removeprefix(prefix)
        match = re.match(r"^(.+?)/(blob|resolve|raw|tree)/([^/]+)(?:/(.+))?$", path)
        if match:
            repo_id, action, revision, filename = match.groups()
            return repo_id, revision, (None if action == "tree" else filename)
        return path, None, None
    return target, None, None


def serialize_safetensors_header(file_metadata) -> bytes:
    header_dict = (
        {"__metadata__": file_metadata.metadata} if file_metadata.metadata else {}
    )
    for tensor_name, tensor_info in file_metadata.tensors.items():
        header_dict[tensor_name] = {
            "dtype": tensor_info.dtype,
            "shape": tensor_info.shape,
            "data_offsets": list(tensor_info.data_offsets),
        }
    json_bytes = json.dumps(header_dict, separators=(",", ":")).encode("utf-8")
    remainder = len(json_bytes) % 8
    if remainder > 0:
        json_bytes += b" " * (8 - remainder)
    return struct.pack("<Q", len(json_bytes)) + json_bytes


def download_model_meta(
    target: str,
    base_dir: str = "models",
    revision: str = None,
    download_other_files: bool = True,
    dry_run: bool = False,
):
    repo_id, parsed_rev, target_filename = parse_hf_target(target)
    effective_revision = revision or parsed_rev or "main"
    token = get_token()
    api = HfApi(token=token)
    local_dir = Path(base_dir) / repo_id

    print(f"Target repository: '{repo_id}' ({effective_revision})")
    print(f"Target local directory: {local_dir}")

    if target_filename:
        safetensors_files = (
            [target_filename] if target_filename.endswith(".safetensors") else []
        )
        to_process_files = [target_filename]
        file_sizes = {}
    else:
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
        safetensors_files = [f for f in files if f.endswith(".safetensors")]
        other_files = [f for f in files if not f.endswith(".safetensors")]
        to_process_files = safetensors_files + (
            other_files if download_other_files else []
        )

    safetensors_metadata = None
    if safetensors_files:
        try:
            safetensors_metadata = get_safetensors_metadata(
                repo_id, revision=effective_revision, token=token
            )
        except Exception as e:
            print(f"Error fetching safetensors metadata: {e}")

    total_downloaded_bytes = 0
    for filename in tqdm(to_process_files, desc="Processing files", unit="file"):
        local_path = local_dir / filename
        if local_path.exists():
            tqdm.write(
                f"Already exists: {filename} ({format_size(local_path.stat().st_size)})"
            )
            continue

        is_safetensors = filename.endswith(".safetensors")
        if is_safetensors:
            if (
                not safetensors_metadata
                or filename not in safetensors_metadata.files_metadata
            ):
                tqdm.write(f"Warning: No metadata found for {filename}. Skipping.")
                continue
            header_bytes = serialize_safetensors_header(
                safetensors_metadata.files_metadata[filename]
            )
            size_bytes = len(header_bytes)
            if not dry_run:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(header_bytes)
            tqdm.write(
                f"{'[DRY RUN] Would generate' if dry_run else 'Generated'} safetensors header: {filename} ({format_size(size_bytes)})"
            )
        else:
            size_bytes = file_sizes.get(filename, 0)
            if not dry_run:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=local_dir,
                    revision=effective_revision,
                    token=token,
                )
                size_bytes = local_path.stat().st_size
            tqdm.write(
                f"{'[DRY RUN] Would download' if dry_run else 'Downloaded'} file: {filename} ({format_size(size_bytes)})"
            )

        total_downloaded_bytes += size_bytes

    print(
        f"\nProcess completed. Total {'simulated' if dry_run else 'downloaded'} size: {format_size(total_downloaded_bytes)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Download Hugging Face model metadata or headers for .safetensors files."
    )
    parser.add_argument("target", type=str, help="Hugging Face repo_id or URL")
    parser.add_argument("--dir", type=str, default="models", help="Base directory")
    parser.add_argument(
        "--revision", type=str, default=None, help="Model revision / branch / commit"
    )
    parser.add_argument(
        "--download-other-files",
        action="store_true",
        help="Download non-safetensors metadata files",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Simulate without saving files"
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
