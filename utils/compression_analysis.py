#!/usr/bin/env python3
import argparse
import os

import numpy as np
import torch
from safetensors import safe_open

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, *args, **kwargs):
        return iterable


def quantize_frequencies(counts, table_log):
    """
    Quantizes and normalizes empirical counts so that they sum to exactly 2^table_log,
    ensuring every present symbol has a frequency of at least 1.
    """
    K = len(counts)
    M = 1 << table_log

    # If the number of unique symbols is greater than the table size,
    # we must scale up the table log to accommodate them.
    if K > M:
        table_log = int(np.ceil(np.log2(K)))
        M = 1 << table_log

    # Initialize each symbol with a minimum frequency of 1
    freqs = np.ones(K, dtype=np.int64)
    remaining = M - K

    if remaining > 0:
        total_counts = np.sum(counts)
        proportions = counts.astype(np.float64) / total_counts
        added = np.round(proportions * remaining).astype(np.int64)
        freqs += added

    # Resolve rounding discrepancies so the sum is exactly 2^table_log
    diff = M - np.sum(freqs)
    if diff != 0:
        # Adjust the most frequent elements first to minimize relative entropy distortion
        indices = np.argsort(counts)[::-1]
        for idx in indices:
            if diff == 0:
                break
            if diff > 0:
                freqs[idx] += 1
                diff -= 1
            elif diff < 0 and freqs[idx] > 1:
                freqs[idx] -= 1
                diff += 1

    return freqs, table_log


def analyze_tensor(tensor_name, tensor, unpack_4bit=True, freq_bytes=2):
    """
    Analyzes a PyTorch tensor, estimating its Shannon cross-entropy under
    quantized frequencies, accounting for codebook and stream overhead.
    """
    dtype_str = str(tensor.dtype).split(".")[-1]
    shape = list(tensor.shape)
    num_elements = tensor.nelement()
    original_size_bytes = num_elements * tensor.element_size()

    # Safely convert to NumPy for distribution analysis (handling bf16 -> fp32 conversion)
    if tensor.dtype == torch.bfloat16:
        arr = tensor.to(torch.float32).cpu().numpy()
    else:
        arr = tensor.cpu().numpy()

    results = []

    # 1. Raw / Packed representation analysis
    flat_raw = arr.ravel()
    unique_raw, counts_raw = np.unique(flat_raw, return_counts=True)

    # Quantize frequencies according to target frequency precision
    target_table_log = freq_bytes * 8
    quant_freqs_raw, actual_table_log = quantize_frequencies(
        counts_raw, target_table_log
    )
    actual_freq_bytes = int(np.ceil(actual_table_log / 8))

    # Calculate bits-per-parameter (BPP) using cross-entropy: H(p, q) = W - sum(p_i * log2(f_i))
    probs_raw = counts_raw / num_elements
    bpp_raw = actual_table_log - np.sum(probs_raw * np.log2(quant_freqs_raw))

    # Codebook overhead: unique_symbols * (symbol_size_bytes + actual_frequency_bytes)
    raw_codebook_size = len(unique_raw) * (tensor.element_size() + actual_freq_bytes)
    raw_encoded_size_bytes = raw_codebook_size + (bpp_raw * num_elements) / 8

    results.append(
        {
            "name": tensor_name,
            "mode": "Raw / Packed",
            "shape": shape,
            "dtype": dtype_str,
            "num_params": num_elements,
            "original_size_bytes": original_size_bytes,
            "unique_symbols": len(unique_raw),
            "bpp": bpp_raw,
            "codebook_bytes": raw_codebook_size,
            "encoded_size_bytes": raw_encoded_size_bytes,
            "savings_pct": (
                (1.0 - (raw_encoded_size_bytes / original_size_bytes)) * 100
                if original_size_bytes > 0
                else 0.0
            ),
        }
    )

    # 2. Unpacked 4-bit representation analysis (e.g., MXFP4 weight_packed tensors)
    is_packed_u8 = "packed" in tensor_name.lower() and dtype_str in [
        "uint8",
        "uint8_t",
        "u8",
    ]

    if unpack_4bit and is_packed_u8:
        # Unpack each uint8 element into two 4-bit values (nibbles)
        low = arr & 0x0F
        high = (arr >> 4) & 0x0F

        # Interleave elements to preserve sequential block layouts
        unpacked = np.empty(num_elements * 2, dtype=np.uint8)
        unpacked[0::2] = low.ravel()
        unpacked[1::2] = high.ravel()

        num_params_unpacked = unpacked.size
        unique_unpacked, counts_unpacked = np.unique(unpacked, return_counts=True)

        quant_freqs_unpacked, actual_table_log_unpacked = quantize_frequencies(
            counts_unpacked, target_table_log
        )
        actual_freq_bytes_unpacked = int(np.ceil(actual_table_log_unpacked / 8))

        probs_unpacked = counts_unpacked / num_params_unpacked
        bpp_unpacked = actual_table_log_unpacked - np.sum(
            probs_unpacked * np.log2(quant_freqs_unpacked)
        )

        # 4-bit symbol size is estimated as 1 byte in the lookup mapping table
        unpacked_codebook_size = len(unique_unpacked) * (1 + actual_freq_bytes_unpacked)
        unpacked_encoded_size_bytes = (
            unpacked_codebook_size + (bpp_unpacked * num_params_unpacked) / 8
        )

        unpacked_shape = (
            [shape[0], shape[1] * 2] if len(shape) == 2 else [num_params_unpacked]
        )

        results.append(
            {
                "name": tensor_name,
                "mode": "Unpacked (4-bit)",
                "shape": unpacked_shape,
                "dtype": "int4",
                "num_params": num_params_unpacked,
                "original_size_bytes": original_size_bytes,
                "unique_symbols": len(unique_unpacked),
                "bpp": bpp_unpacked,
                "codebook_bytes": unpacked_codebook_size,
                "encoded_size_bytes": unpacked_encoded_size_bytes,
                "savings_pct": (
                    (1.0 - (unpacked_encoded_size_bytes / original_size_bytes)) * 100
                    if original_size_bytes > 0
                    else 0.0
                ),
            }
        )

    return results


def format_size(bytes_val):
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.2f} KB"
    elif bytes_val < 1024 * 1024 * 1024:
        return f"{bytes_val / (1024 * 1024):.2f} MB"
    else:
        return f"{bytes_val / (1024 * 1024 * 1024):.2f} GB"


def print_table(rows, headers):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_str = " | ".join(f"{h!s:<{col_widths[i]}}" for i, h in enumerate(headers))
    print(header_str)
    print("-" * len(header_str))
    for row in rows:
        print(" | ".join(f"{val!s:<{col_widths[i]}}" for i, val in enumerate(row)))


def main():
    parser = argparse.ArgumentParser(
        description="Check the compressibility of tensors in a .safetensors file with quantized ANS frequencies."
    )
    parser.add_argument("file_path", type=str, help="Path to the .safetensors file.")
    parser.add_argument(
        "--no-unpack",
        action="store_true",
        help="Disable automatic 4-bit unpacking of packed U8 tensors.",
    )
    parser.add_argument(
        "--freq-bytes",
        type=int,
        default=2,
        help="Bytes used to store frequency counts in codebook estimation (typically 2 for 16-bit counts).",
    )

    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: File not found at {args.file_path}")
        return

    table_rows = []
    headers = [
        "Tensor Name",
        "Analysis Mode",
        "Shape",
        "Orig. Size",
        "Unique Syms",
        "Quantized BPP",
        "Est. Encoded",
        "Savings",
    ]

    total_original_bytes = 0
    total_encoded_bytes = 0

    with safe_open(args.file_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"Analyzing {len(keys)} tensors from {args.file_path}...")

        for key in tqdm(keys):
            tensor = f.get_tensor(key)
            results = analyze_tensor(
                key, tensor, unpack_4bit=not args.no_unpack, freq_bytes=args.freq_bytes
            )

            for res in results:
                # Accumulate raw original footprint. Avoid double counting if both raw and unpacked exist.
                has_raw_and_unpacked = len(results) > 1
                if res["mode"] == "Raw / Packed" or (
                    res["mode"] == "Unpacked (4-bit)" and not has_raw_and_unpacked
                ):
                    total_original_bytes += res["original_size_bytes"]
                    total_encoded_bytes += res["encoded_size_bytes"]

                shape_str = " × ".join(map(str, res["shape"]))
                table_rows.append(
                    [
                        res["name"],
                        res["mode"],
                        f"[{shape_str}]",
                        format_size(res["original_size_bytes"]),
                        res["unique_symbols"],
                        f"{res['bpp']:.3f}",
                        format_size(res["encoded_size_bytes"]),
                        f"{res['savings_pct']:.2f}%",
                    ]
                )

    print("\n" + "=" * 90)
    print("                      TENSOR COMPRESSIBILITY REPORT (ANS ESTIMATED)")
    print("=" * 90 + "\n")
    print_table(table_rows, headers)

    print("\n" + "=" * 90)
    print("Summary:")
    print(f"  Total Tensors: {len(keys)}")
    print(f"  Total Original Footprint: {format_size(total_original_bytes)}")
    print(
        f"  Total Theoretical Lossless Encoded Footprint: {format_size(total_encoded_bytes)}"
    )
    if total_original_bytes > 0:
        overall_savings = (1.0 - (total_encoded_bytes / total_original_bytes)) * 100
        print(
            f"  Potential Savings: {overall_savings:.2f}% (Saved {format_size(total_original_bytes - total_encoded_bytes)})"
        )
    print("=" * 90)


if __name__ == "__main__":
    main()
