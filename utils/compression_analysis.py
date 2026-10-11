#!/usr/bin/env python3
# File: utils/compression_analysis.py
import argparse
import concurrent.futures
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path

import torch
from safetensors import safe_open
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import format_size, natural_sort_key


# Force instant exit on Ctrl+C across all threads and C extensions
def force_exit_handler(signum, frame):
    # Print clean newline and exit immediately without waiting for thread joins
    try:
        sys.stderr.write(
            "\n\nAnalysis cancelled by user (Ctrl+C). Exiting immediately...\n"
        )
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(130)


signal.signal(signal.SIGINT, force_exit_handler)

progress_lock = threading.Lock()


def get_raw_u8_tensor(tensor: torch.Tensor) -> torch.Tensor:
    t = tensor.contiguous()
    if t.dtype == torch.uint8:
        return t.reshape(-1)
    try:
        return t.view(torch.uint8).reshape(-1)
    except Exception:
        # Robust fallback for custom scalar types
        return torch.frombuffer(t.untyped_storage().bytes(), dtype=torch.uint8).reshape(
            -1
        )


def quantize_frequencies(
    counts: torch.Tensor, table_log: int
) -> tuple[torch.Tensor, int]:
    K = counts.numel()
    if K == 0:
        return torch.empty(0, dtype=torch.int64), table_log

    M = 1 << table_log
    if K > M:
        table_log = int(math.ceil(math.log2(K)))
        M = 1 << table_log

    freqs = torch.ones(K, dtype=torch.int64)
    remaining = M - K
    if remaining > 0:
        total = counts.sum().double()
        if total > 0:
            proportions = (counts.double() / total) * remaining
            freqs += torch.round(proportions).to(torch.int64)

    diff = M - int(freqs.sum().item())
    if diff != 0:
        sorted_indices = torch.argsort(counts, descending=True)
        for idx in sorted_indices.tolist():
            if diff == 0:
                break
            if diff > 0:
                freqs[idx] += 1
                diff -= 1
            elif diff < 0 and freqs[idx] > 1:
                freqs[idx] -= 1
                diff += 1
    return freqs, table_log


def analyze_tensor(
    tensor_name: str,
    tensor: torch.Tensor,
    unpack_4bit: bool = True,
    freq_bytes: int = 2,
) -> list[dict]:
    dtype_str = str(tensor.dtype).split(".")[-1]
    shape = list(tensor.shape)
    num_elements = tensor.numel()
    elem_size = tensor.element_size()
    orig_size = num_elements * elem_size

    if num_elements == 0:
        return [
            {
                "name": tensor_name,
                "mode": "Raw / Packed",
                "shape": shape,
                "dtype": dtype_str,
                "num_params": 0,
                "original_size_bytes": 0,
                "unique_symbols": 0,
                "bpp": 0.0,
                "codebook_bytes": 0,
                "encoded_size_bytes": 0,
                "savings_pct": 0.0,
            }
        ]

    raw_u8 = get_raw_u8_tensor(tensor)

    # Fast symbol extraction using native PyTorch operations
    if elem_size == 1:
        bincount = torch.bincount(raw_u8, minlength=256)
        counts_raw = bincount[bincount > 0]
    elif elem_size == 2:
        u16 = raw_u8[0::2].to(torch.int64) | (raw_u8[1::2].to(torch.int64) << 8)
        bincount = torch.bincount(u16, minlength=65536)
        counts_raw = bincount[bincount > 0]
    elif elem_size == 4:
        u32 = (
            raw_u8[0::4].to(torch.int64)
            | (raw_u8[1::4].to(torch.int64) << 8)
            | (raw_u8[2::4].to(torch.int64) << 16)
            | (raw_u8[3::4].to(torch.int64) << 24)
        )
        _, counts_raw = torch.unique(u32, return_counts=True)
    else:
        flat = tensor.contiguous().reshape(-1)
        try:
            _, counts_raw = torch.unique(flat.view(torch.int64), return_counts=True)
        except Exception:
            _, counts_raw = torch.unique(flat, return_counts=True)

    unique_symbols = counts_raw.numel()
    quant_freqs, actual_table_log = quantize_frequencies(counts_raw, freq_bytes * 8)
    actual_freq_bytes = int(math.ceil(actual_table_log / 8))

    counts_double = counts_raw.double()
    p = counts_double / num_elements
    bpp_raw = float(
        actual_table_log - torch.sum(p * torch.log2(quant_freqs.double())).item()
    )
    bpp_raw = max(0.0, bpp_raw)

    raw_codebook = unique_symbols * (elem_size + actual_freq_bytes)
    raw_enc_size = raw_codebook + (bpp_raw * num_elements) / 8.0

    results = [
        {
            "name": tensor_name,
            "mode": "Raw / Packed",
            "shape": shape,
            "dtype": dtype_str,
            "num_params": num_elements,
            "original_size_bytes": orig_size,
            "unique_symbols": unique_symbols,
            "bpp": bpp_raw,
            "codebook_bytes": raw_codebook,
            "encoded_size_bytes": raw_enc_size,
            "savings_pct": (1.0 - (raw_enc_size / orig_size)) * 100.0
            if orig_size > 0
            else 0.0,
        }
    ]

    if (
        unpack_4bit
        and "packed" in tensor_name.lower()
        and dtype_str in ["uint8", "uint8_t", "u8", "int8"]
    ):
        low = (raw_u8 & 0x0F).to(torch.int64)
        high = ((raw_u8 >> 4) & 0x0F).to(torch.int64)
        u_bincount = torch.bincount(low, minlength=16) + torch.bincount(
            high, minlength=16
        )
        u_counts = u_bincount[u_bincount > 0]
        num_u = num_elements * 2
        u_unique_syms = u_counts.numel()

        u_freqs, u_log = quantize_frequencies(u_counts, freq_bytes * 8)
        u_p = u_counts.double() / num_u
        u_bpp = float(u_log - torch.sum(u_p * torch.log2(u_freqs.double())).item())
        u_bpp = max(0.0, u_bpp)

        u_codebook = u_unique_syms * (1 + int(math.ceil(u_log / 8)))
        u_enc_size = u_codebook + (u_bpp * num_u) / 8.0

        results.append(
            {
                "name": tensor_name,
                "mode": "Unpacked (4-bit)",
                "shape": [shape[0], shape[1] * 2] if len(shape) == 2 else [num_u],
                "dtype": "int4",
                "num_params": num_u,
                "original_size_bytes": orig_size,
                "unique_symbols": u_unique_syms,
                "bpp": u_bpp,
                "codebook_bytes": u_codebook,
                "encoded_size_bytes": u_enc_size,
                "savings_pct": (1.0 - (u_enc_size / orig_size)) * 100.0
                if orig_size > 0
                else 0.0,
            }
        )

    return results


def find_safetensors_files(path_str: str) -> list[Path]:
    p = Path(path_str)
    if not p.exists():
        print(f"Error: Path '{path_str}' does not exist.")
        sys.exit(1)
    if p.is_file():
        return [p]
    files = sorted(p.rglob("*.safetensors"), key=lambda f: natural_sort_key(str(f)))
    if not files:
        print(f"Error: No .safetensors files found under '{path_str}'.")
        sys.exit(1)
    return files


def format_row(res: dict) -> str:
    shape_str = f"[{' × '.join(map(str, res['shape']))}]"
    return (
        f"{res['name']:<45} | "
        f"{res['mode']:<18} | "
        f"{res['dtype']:<10} | "
        f"{shape_str:<20} | "
        f"{format_size(res['original_size_bytes']):>12} | "
        f"{res['unique_symbols']:>12} | "
        f"{res['bpp']:>14.3f} | "
        f"{format_size(res['encoded_size_bytes']):>14} | "
        f"{res['savings_pct']:>9.2f}%"
    )


def process_file_worker(
    file_path_str: str,
    unpack_4bit: bool,
    freq_bytes: int,
    on_tensor_completed,
):
    try:
        with safe_open(file_path_str, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                res = analyze_tensor(
                    key, tensor, unpack_4bit=unpack_4bit, freq_bytes=freq_bytes
                )
                if on_tensor_completed:
                    on_tensor_completed(res)
    except Exception as e:
        print(f"\nError processing '{file_path_str}': {e}", file=sys.stderr)


def is_decode_step_tensor(tensor_name: str) -> bool:
    """Return whether a tensor participates in a standard text decode step."""
    if tensor_name == "lm_head.weight":
        return True
    prefix = "model.language_model.layers."
    if not tensor_name.startswith(prefix):
        return False
    try:
        layer = int(tensor_name[len(prefix) :].split(".", 1)[0])
    except (ValueError, IndexError):
        return False
    # Layers 0–44 are the decoder. Layer 45 is the optional MTP head.
    return 0 <= layer <= 44


def analyze_decode_step(
    files: list[Path], top_k: int, freq_bytes: int, expert_samples: int
) -> None:
    """Estimate bytes read for one batch-1 decode token from active weights."""
    def process_decode_file(file_path: Path) -> dict:
        result = {
            "always_original": 0,
            "always_encoded": 0.0,
            "expert_original": 0,
            "expert_encoded": 0.0,
            "expert_tensors": 0,
            "sampled_expert_original": 0,
            "sampled_expert_encoded": 0.0,
            "expert_layers": set(),
            "always_tensors": 0,
            "skipped_embedding": False,
        }
        with safe_open(str(file_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key == "model.language_model.embed_tokens.weight":
                    # Decode reads one embedding row, not the full embedding matrix.
                    tensor_slice = f.get_slice(key)
                    dtype_bytes = {
                        "F8_E4M3": 1,
                        "BF16": 2,
                        "F16": 2,
                        "F32": 4,
                    }.get(tensor_slice.get_dtype(), 0)
                    row_bytes = tensor_slice.get_shape()[1] * dtype_bytes
                    result["always_original"] += row_bytes
                    result["always_encoded"] += row_bytes
                    result["always_tensors"] += 1
                    result["skipped_embedding"] = True
                    continue
                if not is_decode_step_tensor(key):
                    continue

                if ".mlp.experts." in key:
                    expert_id = int(key.split(".mlp.experts.")[1].split(".")[0])
                    tensor_slice = f.get_slice(key)
                    dtype_bytes = {
                        "F8_E4M3": 1,
                        "BF16": 2,
                        "F16": 2,
                        "F32": 4,
                    }.get(tensor_slice.get_dtype(), 0)
                    expert_size = math.prod(tensor_slice.get_shape()) * dtype_bytes
                    result["expert_original"] += expert_size
                    result["expert_tensors"] += 1
                    expert_layer = int(key.split(".layers.")[1].split(".")[0])
                    result["expert_layers"].add(expert_layer)
                    if expert_id not in sample_expert_ids:
                        continue

                tensor = f.get_tensor(key)
                results = analyze_tensor(
                    key, tensor, unpack_4bit=False, freq_bytes=freq_bytes
                )
                original_size = results[0]["original_size_bytes"]
                encoded_size = min(r["encoded_size_bytes"] for r in results)
                effective_size = min(original_size, encoded_size)

                if ".mlp.experts." in key:
                    result["sampled_expert_original"] += original_size
                    result["sampled_expert_encoded"] += effective_size
                else:
                    result["always_original"] += original_size
                    result["always_encoded"] += effective_size
                    result["always_tensors"] += 1
        return result

    sample_expert_ids = {
        round(i * 287 / (expert_samples - 1)) if expert_samples > 1 else 0
        for i in range(expert_samples)
    }
    results = []
    max_workers = min(len(files), os.cpu_count() or 4, 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_decode_file, file_path) for file_path in files]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Analyzing decode-step weights",
            leave=False,
        ):
            results.append(future.result())

    always_original = sum(r["always_original"] for r in results)
    always_encoded = sum(r["always_encoded"] for r in results)
    expert_original = sum(r["expert_original"] for r in results)
    sampled_expert_encoded = sum(r["sampled_expert_encoded"] for r in results)
    expert_tensors = sum(r["expert_tensors"] for r in results)
    expert_layers = set().union(*(r["expert_layers"] for r in results))
    always_tensors = sum(r["always_tensors"] for r in results)
    skipped_embedding = any(r["skipped_embedding"] for r in results)

    # Each token activates top_k distinct routed experts in every MoE layer.
    expert_count = 288
    active_expert_fraction = top_k / expert_count
    active_expert_original = expert_original * active_expert_fraction
    active_expert_encoded = sampled_expert_encoded * top_k / len(sample_expert_ids)
    total_original = always_original + active_expert_original
    total_encoded = always_encoded + active_expert_encoded
    savings_pct = (1.0 - total_encoded / total_original) * 100.0

    print("\nDecode-step weight traffic estimate (batch size 1):")
    print(f"  Decoder layers analyzed: 0–44 ({len(expert_layers)} MoE layers)")
    print(f"  Routed experts per MoE layer: {top_k} of {expert_count} (average over experts)")
    print(f"  Expert IDs sampled for compression: {sorted(sample_expert_ids)}")
    print(f"  Always-used tensors: {always_tensors}")
    print(f"  Routed expert tensors in checkpoint: {expert_tensors}")
    print(f"  Input embedding: {'one row counted' if skipped_embedding else 'not present'}")
    print(
        f"  Uncompressed active-weight bytes: {total_original:.0f} "
        f"({total_original / 1e9:.3f} GB, {total_original / (1 << 30):.3f} GiB)"
    )
    print(
        f"  Estimated lossless encoded bytes: {total_encoded:.0f} "
        f"({total_encoded / 1e9:.3f} GB, {total_encoded / (1 << 30):.3f} GiB)"
    )
    print(f"  Estimated savings: {savings_pct:.2f}%")
    print("  Note: Uses this script's per-tensor ANS estimate; the selected-expert value is an average.")


def main():
    parser = argparse.ArgumentParser(
        description="Check compressibility of tensors in .safetensors with quantized ANS."
    )
    parser.add_argument(
        "file_path", type=str, help="Path to .safetensors file or directory of shards."
    )
    parser.add_argument(
        "--no-unpack", action="store_true", help="Disable 4-bit unpacking of packed U8."
    )
    parser.add_argument(
        "--freq-bytes",
        type=int,
        default=2,
        help="Bytes for frequency counts in codebook (default: 2).",
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=None,
        help="Number of worker threads (default: CPU core count).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only display summary report without per-tensor table.",
    )
    parser.add_argument(
        "--decode-step",
        action="store_true",
        help="Estimate active weight bytes read for one standard text decode token.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Routed experts selected per MoE layer for --decode-step (default: 8).",
    )
    parser.add_argument(
        "--expert-samples",
        type=int,
        default=4,
        help="Evenly spaced routed experts per layer to encode in --decode-step (default: 4).",
    )
    args = parser.parse_args()

    files = find_safetensors_files(args.file_path)

    if args.decode_step:
        if not 1 <= args.top_k <= 288:
            parser.error("--top-k must be between 1 and 288")
        if not 1 <= args.expert_samples <= 288:
            parser.error("--expert-samples must be between 1 and 288")
        torch.set_num_threads(1)
        analyze_decode_step(files, args.top_k, args.freq_bytes, args.expert_samples)
        return

    # Fast header-only inspection to obtain total tensor count
    total_tensors = 0
    for file_path in files:
        with safe_open(str(file_path), framework="pt", device="cpu") as f:
            total_tensors += len(f.keys())

    target_desc = (
        str(files[0]) if len(files) == 1 else f"{args.file_path} ({len(files)} files)"
    )
    print(f"Analyzing {total_tensors} tensors from {target_desc}...\n")

    max_workers = args.threads or min(len(files), os.cpu_count() or 4)
    torch.set_num_threads(1)  # Prevent CPU context-switch oversubscription

    hdr_str = (
        f"{'Tensor Name':<45} | "
        f"{'Analysis Mode':<18} | "
        f"{'Dtype':<10} | "
        f"{'Shape':<20} | "
        f"{'Orig. Size':>12} | "
        f"{'Unique Syms':>12} | "
        f"{'Quantized BPP':>14} | "
        f"{'Est. Encoded':>14} | "
        f"{'Savings':>10}"
    )

    if not args.summary_only:
        banner = "TENSOR COMPRESSIBILITY REPORT (ANS ESTIMATED)".center(len(hdr_str))
        print(
            f"{'=' * len(hdr_str)}\n{banner}\n{'=' * len(hdr_str)}\n{hdr_str}\n{'-' * len(hdr_str)}"
        )

    total_original_bytes = 0
    total_encoded_bytes = 0
    tensors_stored_raw = 0

    # Summary bar (position 0) pinned right above the progress bar (position 1)
    summary_bar = tqdm(
        total=0,
        position=0,
        bar_format="{desc}",
        leave=False,
    )
    pbar = tqdm(
        total=total_tensors,
        position=1,
        desc="Analyzing tensors",
        leave=False,
    )

    summary_bar.set_description_str(
        "Running Summary: Orig: 0 B | Enc: 0 B | Savings: 0.00% (Saved: 0 B)"
    )

    def on_tensor_completed(res_list: list[dict]):
        nonlocal total_original_bytes, total_encoded_bytes, tensors_stored_raw
        with progress_lock:
            orig_size = res_list[0]["original_size_bytes"]

            # Select the best compression mode for this tensor
            best_mode_enc_size = min(item["encoded_size_bytes"] for item in res_list)

            # Selective encoding: store raw if compression inflates the size
            if best_mode_enc_size < orig_size:
                effective_enc_size = best_mode_enc_size
            else:
                effective_enc_size = orig_size
                tensors_stored_raw += 1

            total_original_bytes += orig_size
            total_encoded_bytes += effective_enc_size

            if not args.summary_only:
                for item in res_list:
                    tqdm.write(format_row(item))

            saved = total_original_bytes - total_encoded_bytes
            savings_pct = (
                ((saved / total_original_bytes) * 100)
                if total_original_bytes > 0
                else 0.0
            )
            summary_str = (
                f"Running Summary: Orig: {format_size(total_original_bytes)} | "
                f"Enc: {format_size(total_encoded_bytes)} | "
                f"Savings: {savings_pct:.2f}% (Saved: {format_size(saved)})"
            )
            summary_bar.set_description_str(summary_str)
            pbar.update(1)

    with summary_bar, pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    process_file_worker,
                    str(file_path),
                    not args.no_unpack,
                    args.freq_bytes,
                    on_tensor_completed,
                )
                for file_path in files
            ]

            # Poll with timeout to ensure SIGINT (Ctrl+C) triggers immediately in the main thread
            while True:
                unfinished = [f for f in futures if not f.done()]
                if not unfinished:
                    break
                time.sleep(0.1)

            for f in futures:
                f.result()

    saved = total_original_bytes - total_encoded_bytes
    savings_pct = (
        ((saved / total_original_bytes) * 100) if total_original_bytes > 0 else 0.0
    )
    hdr_len = 80
    print(f"\n{'=' * hdr_len}\nSummary:")
    if len(files) > 1:
        print(f"  Total Files: {len(files)}")
    print(
        f"  Total Tensors: {total_tensors} (Stored uncompressed/raw: {tensors_stored_raw})"
    )
    print(f"  Total Original Footprint: {format_size(total_original_bytes)}")
    print(
        f"  Total Theoretical Lossless Footprint: {format_size(total_encoded_bytes)}\n  Potential Savings: {savings_pct:.2f}% (Saved {format_size(saved)})\n{'=' * hdr_len}"
    )


if __name__ == "__main__":
    main()
