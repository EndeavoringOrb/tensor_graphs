#!/usr/bin/env python3
import os
import sys
import argparse
import re
import struct

# Robust import to support running directly or as a module
try:
    from utils.binary import load_cache_file, load_records_file
except ModuleNotFoundError:
    from binary import load_cache_file, load_records_file


def _format_constants(raw_bytes, dtype):
    if not raw_bytes:
        return ""
    dtypes = ["FLOAT32", "INT32", "BF16", "UINT8"]
    dt_str = (
        dtypes[dtype] if isinstance(dtype, int) and dtype < len(dtypes) else str(dtype)
    )

    if dt_str == "FLOAT32":
        count = len(raw_bytes) // 4
        return list(struct.unpack(f"<{count}f", raw_bytes))
    elif dt_str == "INT32":
        count = len(raw_bytes) // 4
        return list(struct.unpack(f"<{count}i", raw_bytes))
    return list(raw_bytes)


def format_constants(raw_bytes, dtype):
    formatted = _format_constants(raw_bytes, dtype)
    if len(formatted) > 10:
        formatted = str(formatted[:10]) + "..."
    else:
        formatted = str(formatted)
    return formatted


def load_uids_from_cpp(header_path):
    uid_to_name = {}
    if not header_path or not os.path.exists(header_path):
        return uid_to_name
    pattern = re.compile(r"constexpr uint64_t\s+(\w+)\s+=\s+(0x[0-9a-fA-F]+)ULL;")
    with open(header_path, "r") as f:
        content = f.read()
        matches = pattern.findall(content)
        for name, hex_val in matches:
            val_int = int(hex_val, 16)
            uid_to_name[str(val_int)] = name
            uid_to_name[hex_val.lower()] = name
    return uid_to_name


def load_uid_map(cache_path):
    uid_to_name = {}
    if not cache_path or not os.path.exists(cache_path):
        return uid_to_name
    entries = load_cache_file(cache_path)
    for entry in entries:
        if entry.get("type") == "compiled_bucket":
            graph = entry["graph"]
            nodes = graph["nodesMap"]
            for inst in graph["instructions"]:
                uid = str(inst["fullKernelId"])
                node = nodes.get(str(inst["nodeId"]))
                if node:
                    op_name = node["opType"]
                    if op_name == "FUSED":
                        op_name = f"FUSED_{node.get('opName', 'UNKNOWN')}"
                    uid_to_name[uid] = op_name
    return uid_to_name


def main():
    parser = argparse.ArgumentParser(
        description="Read and filter TensorGraph benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        help="Path to generated header",
    )
    parser.add_argument("--op", help="Regex/Substring filter for OpName")
    parser.add_argument("--shape", help="String filter for OutputShape")
    args = parser.parse_args()

    if not os.path.exists(args.records):
        print(f"Error: {args.records} not found.")
        return

    uid_map = load_uid_map(args.cache)
    header_map = load_uids_from_cpp(args.header)
    uid_map.update(header_map)

    records = load_records_file(args.records)

    filtered = []
    for r in records:
        uid = str(r["kernelUid"])
        name = uid_map.get(uid, uid_map.get(hex(r["kernelUid"]), None))
        if not name:
            continue

        if (
            args.op
            and not re.search(args.op, name, re.IGNORECASE)
            and not re.search(args.op, uid, re.IGNORECASE)
        ):
            continue

        out_shapes_str = str(r.get("outputShapes", [])) + str(r.get("inputShapes", []))
        if args.shape and args.shape not in out_shapes_str:
            continue

        filtered.append((name, r))

    total = len(filtered)
    dtypes = ["FLOAT32", "INT32", "BF16", "UINT8"]
    backends_map = ["STORAGE", "CPU", "CUDA"]

    for i, (name, r) in enumerate(filtered):
        uid = hex(r["kernelUid"])
        b_list = [
            backends_map[b] if b < len(backends_map) else "???"
            for b in r.get("backends", [])
        ]
        backends = ",".join(b_list) if b_list else "CPU"

        print(f"[{i+1}/{total}][{backends}] {name} ({uid})")

        in_shapes = r.get("inputShapes", [])
        in_dtypes = r.get("inputDTypes", [])
        in_strides = r.get("inputStrides", [])
        in_consts = r.get("inputConstants", [])

        for idx in range(len(in_shapes)):
            dt_raw = in_dtypes[idx] if idx < len(in_dtypes) else -1
            dt = dtypes[dt_raw] if dt_raw >= 0 and dt_raw < len(dtypes) else "???"
            sh = in_shapes[idx]
            st = in_strides[idx] if idx < len(in_strides) else []
            ic_raw = in_consts[idx] if idx < len(in_consts) else []
            ic_formatted = format_constants(ic_raw, dt_raw)
            const_str = f", constants={ic_formatted}" if ic_formatted else ""
            print(f"  In  #{idx}: dtype={dt}, shape={sh}, strides={st}{const_str}")

        out_shapes = r.get("outputShapes", [])
        out_dtypes = r.get("outputDTypes", [])
        out_strides = r.get("outputStrides", [])
        for idx in range(len(out_shapes)):
            dt_raw = out_dtypes[idx] if idx < len(out_dtypes) else -1
            dt = dtypes[dt_raw] if dt_raw >= 0 and dt_raw < len(dtypes) else "???"
            sh = out_shapes[idx]
            st = out_strides[idx] if idx < len(out_strides) else []
            print(f"  Out #{idx}: dtype={dt}, shape={sh}, strides={st}")

        runtime = r.get("runTime", 0.0)
        print(f"  Benchmarking... -> {runtime:.6f} ms\n")


if __name__ == "__main__":
    main()
