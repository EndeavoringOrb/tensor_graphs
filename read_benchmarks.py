#!/usr/bin/env python3
import json
import os
import argparse
import re


def load_uids_from_cpp(header_path):
    """
    Parses the C++ header file to map Hex UIDs to their Constant Names.
    """
    uid_to_name = {}
    if not header_path or not os.path.exists(header_path):
        return uid_to_name

    # Regex to find: constexpr uint64_t NAME = 0xHEXULL;
    pattern = re.compile(r"constexpr uint64_t\s+(\w+)\s+=\s+(0x[0-9a-fA-F]+)ULL;")

    with open(header_path, "r") as f:
        content = f.read()
        matches = pattern.findall(content)
        for name, hex_val in matches:
            # Convert hex to decimal integer
            val_int = int(hex_val, 16)
            # Store both string decimal and hex representation to be safe
            uid_to_name[str(val_int)] = name
            uid_to_name[hex_val.lower()] = name

    return uid_to_name


def load_uid_map(cache_path):
    """
    Scans the compiled bucket cache to associate kernel UIDs with human names.
    """
    uid_to_name = {}
    if not cache_path or not os.path.exists(cache_path):
        return uid_to_name

    with open(cache_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if entry.get("type") == "compiled_bucket":
                    nodes = entry["graph"]["nodesMap"]
                    for inst in entry["graph"]["instructions"]:
                        uid = str(inst["fullKernelId"])
                        node = nodes[str(inst["nodeId"])]
                        op_name = node["opType"]
                        if op_name == "FUSED":
                            op_name = f"FUSED_{node.get('opName', 'UNKNOWN')}"
                        uid_to_name[uid] = op_name
            except (json.JSONDecodeError, KeyError):
                continue
    return uid_to_name


def main():
    parser = argparse.ArgumentParser(
        description="Read and filter TensorGraph benchmarks."
    )
    parser.add_argument(
        "--records", default="benchmarks/records.jsonl", help="Path to records.jsonl"
    )
    parser.add_argument(
        "--cache",
        default="dirty_region_caches/gemma-3-270m-cpp.jsonl",
        help="Path to cache file to resolve OpNames",
    )
    parser.add_argument(
        "--header",
        default="tensor_graphs_cpp/generated/kernel_uids.gen.hpp",
        help="Path to the C++ header file containing Kernel IDs",
    )
    parser.add_argument(
        "--op", help="Regex/Substring filter for OpName (e.g. 'MUL' or 'RMS')"
    )
    parser.add_argument(
        "--shape", help="String filter for OutputShape (e.g. '[1, 8, 640]')"
    )
    args = parser.parse_args()

    if not os.path.exists(args.records):
        print(f"Error: {args.records} not found.")
        return

    # 1. Load mapping from Cache (OpTypes)
    uid_map = load_uid_map(args.cache)

    # 2. Load mapping from C++ Header (Specific Kernel Names)
    # This will overwrite the generic Cache names with specific kernel names if available
    header_map = load_uids_from_cpp(args.header)
    uid_map.update(header_map)

    records = []
    with open(args.records, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Apply Filters
    filtered = []
    for r in records:
        # Normalize uid to string for lookup
        uid = str(r["kernelUid"])
        name = uid_map.get(uid, "UNKNOWN")

        # Filter by OpName/Type
        if (
            args.op
            and not re.search(args.op, name, re.IGNORECASE)
            and not re.search(args.op, uid, re.IGNORECASE)
        ):
            continue

        # Filter by (Output + Input) Shape string matching
        out_shapes_str = str(r.get("outputShapes", [])) + str(r.get("inputShapes", []))
        if args.shape and args.shape not in out_shapes_str:
            continue

        filtered.append((name, r))

    # Print in bench.cpp format
    total = len(filtered)
    for i, (name, r) in enumerate(filtered):
        uid = r["kernelUid"]
        backends = ",".join(r.get("backends", ["CPU"]))

        print(f"[{i+1}/{total}][{backends}] {name} ({uid})")

        # Inputs
        in_shapes = r.get("inputShapes", [])
        in_dtypes = r.get("inputDTypes", [])
        in_strides = r.get("inputStrides", [])
        for idx in range(len(in_shapes)):
            dt = in_dtypes[idx] if idx < len(in_dtypes) else "???"
            sh = in_shapes[idx]
            st = in_strides[idx] if idx < len(in_strides) else []
            print(f"  In  #{idx}: dtype={dt}, shape={sh}, strides={st}")

        # Outputs
        out_shapes = r.get("outputShapes", [])
        out_dtypes = r.get("outputDTypes", [])
        out_strides = r.get("outputStrides", [])
        for idx in range(len(out_shapes)):
            dt = out_dtypes[idx] if idx < len(out_dtypes) else "???"
            sh = out_shapes[idx]
            st = out_strides[idx] if idx < len(out_strides) else []
            print(f"  Out #{idx}: dtype={dt}, shape={sh}, strides={st}")

        runtime = r.get("runTime", 0.0)
        print(f"  Benchmarking... -> {runtime:.6f} ms")
        print()


if __name__ == "__main__":
    main()
