#!/usr/bin/env python3
# File: utils/read_benchmarks.py
import argparse
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binary import BinaryReader, load_cache_file
from common import format_op_name, load_uids_from_cpp


def format_constants(raw_bytes, dtype):
    if not raw_bytes:
        return ""
    dtypes = ["FLOAT32", "INT32", "BF16", "UINT8"]
    dt_str = (
        dtypes[dtype] if isinstance(dtype, int) and dtype < len(dtypes) else str(dtype)
    )
    count = len(raw_bytes) // 4
    if dt_str == "FLOAT32":
        formatted = list(struct.unpack(f"<{count}f", raw_bytes))
    elif dt_str == "INT32":
        formatted = list(struct.unpack(f"<{count}i", raw_bytes))
    else:
        formatted = list(raw_bytes)
    return f"{formatted[:10]}..." if len(formatted) > 10 else str(formatted)


def load_uid_map(cache_path, header_path=None):
    uid_map = {k: format_op_name(v) for k, v in load_uids_from_cpp(header_path).items()}
    if cache_path and os.path.exists(cache_path):
        for entry in load_cache_file(cache_path):
            if entry.get("type") == "compiled_bucket":
                for inst in entry["graph"]["instructions"]:
                    uid = str(inst["kernelId"])
                    uid_map.setdefault(uid, f"Kernel_{hex(inst['kernelId'])}")
    return uid_map


def print_record(idx, name, r):
    dtypes = ["FLOAT32", "INT32", "BF16", "UINT8"]
    backends_map = ["STORAGE", "CPU", "CUDA"]
    backends = (
        ",".join(
            backends_map[b] if b < len(backends_map) else "???"
            for b in r.get("backends", [])
        )
        or "CPU"
    )

    print(f"[{idx + 1}][{backends}] {name} ({hex(r['kernelId'])})")
    for j, sh in enumerate(r.get("inputShapes", [])):
        dt_raw = (
            r.get("inputDTypes", [])[j] if j < len(r.get("inputDTypes", [])) else -1
        )
        dt = dtypes[dt_raw] if 0 <= dt_raw < len(dtypes) else "???"
        st = r.get("inputStrides", [])[j] if j < len(r.get("inputStrides", [])) else []
        const = format_constants(
            r.get("inputConstants", [])[j]
            if j < len(r.get("inputConstants", []))
            else b"",
            dt_raw,
        )
        print(
            f"  In  #{j}: dtype={dt}, shape={sh}, strides={st}{f', constants={const}' if const else ''}"
        )

    for j, sh in enumerate(r.get("outputShapes", [])):
        dt_raw = (
            r.get("outputDTypes", [])[j] if j < len(r.get("outputDTypes", [])) else -1
        )
        dt = dtypes[dt_raw] if 0 <= dt_raw < len(dtypes) else "???"
        st = (
            r.get("outputStrides", [])[j] if j < len(r.get("outputStrides", [])) else []
        )
        print(f"  Out #{j}: dtype={dt}, shape={sh}, strides={st}")

    print(f"  Benchmarking... -> {r.get('runTime', 0.0):.6f} ms\n")


def main():
    parser = argparse.ArgumentParser(
        description="Read and filter TensorGraph benchmarks."
    )
    parser.add_argument(
        "--records", default="benchmarks/records.bin", help="Path to records file"
    )
    parser.add_argument(
        "--cache",
        default="dirty_region_caches/gemma-3-270m-cpp.bin",
        help="Path to cache file",
    )
    parser.add_argument(
        "--header",
        default="tensor_graphs_cpp/generated/kernel_uids.gen.hpp",
        help="Path to header",
    )
    parser.add_argument("--op", help="Regex/Substring filter for OpName")
    parser.add_argument("--shape", help="String filter for OutputShape")
    args = parser.parse_args()

    if not os.path.exists(args.records):
        print(f"Error: {args.records} not found.")
        return

    uid_map = load_uid_map(args.cache, args.header)
    matched_idx = 0

    with open(args.records, "rb") as f:
        br = BinaryReader(f)
        idx = 0
        while True:
            idx += 1
            print(idx, end="\r")
            r = br.read_record()
            if r is None:
                break

            uid = str(r["kernelId"])
            name = uid_map.get(uid, uid_map.get(hex(r["kernelId"])))
            if not name:
                continue

            if args.op and not (
                re.search(args.op, name, re.IGNORECASE)
                or re.search(args.op, uid, re.IGNORECASE)
            ):
                continue

            shapes_str = str(r.get("outputShapes", [])) + str(r.get("inputShapes", []))
            if args.shape and args.shape not in shapes_str:
                continue

            print_record(matched_idx, name, r)
            matched_idx += 1


if __name__ == "__main__":
    main()
