#!/usr/bin/env python3
import argparse
import json
import os
import re
import struct

from binary import BinaryReader, load_cache_file


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
    json_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "tensor_graphs_cpp",
        "generated",
        "kernel_uids.json",
    )
    uid_to_name = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for key, info in data.items():
                    if isinstance(info, dict):
                        k_name = info.get("name", "")
                        k_path = info.get("path", "")
                        label = f"{k_name} [{k_path}]" if k_path else k_name
                        uid_to_name[str(key)] = label
                        uid_to_name[key] = label
            return uid_to_name
        except Exception:
            pass

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
            for inst in graph["instructions"]:
                uid = str(inst["kernelId"])
                uid_to_name[uid] = f"Kernel_{hex(inst['kernelId'])}"
    return uid_to_name


def print_record(idx, name, r):
    dtypes = ["FLOAT32", "INT32", "BF16", "UINT8"]
    backends_map = ["STORAGE", "CPU", "CUDA"]

    uid = hex(r["kernelId"])
    b_list = [
        backends_map[b] if b < len(backends_map) else "???"
        for b in r.get("backends", [])
    ]
    backends = ",".join(b_list) if b_list else "CPU"

    print(f"[{idx + 1}][{backends}] {name} ({uid})")

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

    idx = 0
    matched_idx = 0
    if os.path.exists(args.records):
        with open(args.records, "rb") as f:
            br = BinaryReader(f)
            while True:
                idx += 1
                print(idx, end="\r")
                r = br.read_record()
                if r is None:
                    break

                uid = str(r["kernelId"])
                name = uid_map.get(uid, uid_map.get(hex(r["kernelId"]), None))
                if not name:
                    continue

                if (
                    args.op
                    and not re.search(args.op, name, re.IGNORECASE)
                    and not re.search(args.op, uid, re.IGNORECASE)
                ):
                    continue

                out_shapes_str = str(r.get("outputShapes", [])) + str(
                    r.get("inputShapes", [])
                )
                if args.shape and args.shape not in out_shapes_str:
                    continue

                print_record(matched_idx, name, r)
                matched_idx += 1


if __name__ == "__main__":
    main()
