# File: utils/analyze_performance.py
import argparse
import json
import os
import re
from collections import defaultdict

from utils.binary import load_cache_file


def load_uids_from_cpp():
    json_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "tensor_graphs_cpp",
        "generated",
        "kernel_uids.json",
    )
    uid_map = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for key, info in data.items():
                    uid_map[key] = info
                    try:
                        uid_map[int(key)] = info
                    except ValueError:
                        pass
            return uid_map
        except Exception as e:
            print(f"Warning: Failed to load {json_path}: {e}")

    # Fallback to header file
    header_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "tensor_graphs_cpp",
        "generated",
        "kernel_uids.gen.hpp",
    )
    if os.path.exists(header_path):
        pattern = re.compile(r"constexpr uint64_t\s+(\w+)\s+=\s+(0x[0-9a-fA-F]+)ULL;")
        with open(header_path, "r") as f:
            content = f.read()
            matches = pattern.findall(content)
            for name, hex_val in matches:
                val_int = int(hex_val, 16)
                info = {"name": name, "path": "", "hex_uid": hex_val}
                uid_map[val_int] = info
                uid_map[hex_val.lower()] = info
                uid_map[str(val_int)] = info
    return uid_map


def format_ms(ms):
    return f"{ms:.4f} ms"


def analyze(cache_file, top_n=20, chain_len=1, bucket_idx=None):
    print(f"Loading compiled buckets from: {cache_file}")
    cache_entries = load_cache_file(cache_file)

    # Load physical UID mapping
    uid_map = load_uids_from_cpp()

    compiled_buckets = [
        entry for entry in cache_entries if entry.get("type") == "compiled_bucket"
    ]
    num_buckets = len(compiled_buckets)

    if bucket_idx is not None:
        if bucket_idx < 0 or bucket_idx >= num_buckets:
            if num_buckets == 0:
                print(f"Error: No compiled buckets found in {cache_file}.")
            elif num_buckets == 1:
                print(
                    f"Error: Bucket index {bucket_idx} is out of range. Available index: 0"
                )
            else:
                print(
                    f"Error: Bucket index {bucket_idx} is out of range. Available range: 0 to {num_buckets - 1}"
                )
            return
        buckets_to_analyze = [compiled_buckets[bucket_idx]]
    else:
        buckets_to_analyze = compiled_buckets

    chain_stats = defaultdict(lambda: {"time": 0.0, "count": 0})
    op_type_stats = defaultdict(float)
    total_estimated_time = 0.0

    bucket_count = 0
    for entry in buckets_to_analyze:
        bucket_count += 1
        graph = entry["graph"]
        node_views = graph.get("nodeViews", {})
        instructions = graph["instructions"]
        node_costs = graph.get("nodeCosts", {})

        bucket_sequence = []
        for inst in instructions:
            eclass_id = inst["eclassId"]
            node_view = node_views.get(eclass_id, {})

            runtime = node_costs.get(eclass_id, 0.0)
            if runtime == float("inf"):
                runtime = 0.0

            kernel_uid = inst.get("kernelId", 0)
            info = (
                uid_map.get(kernel_uid)
                or uid_map.get(str(kernel_uid))
                or uid_map.get(hex(kernel_uid).lower())
            )

            if info and isinstance(info, dict):
                k_name = info.get("name", f"Kernel_{hex(kernel_uid)}")
                k_path = info.get("path", "")
                op_name = f"{k_name} [{k_path}]" if k_path else k_name
            elif isinstance(info, str):
                op_name = info
            else:
                op_name = f"Kernel_{hex(kernel_uid)}"

            input_shapes = []
            for child_eclass in inst.get("children", []):
                if child_eclass in node_views:
                    input_shapes.append(node_views[child_eclass].get("shape", []))

            shape = node_view.get("shape", [])
            debug_origin = inst.get("debugOrigin", "UNKNOWN")

            display_identity = (
                op_name,
                hex(kernel_uid),
                tuple(shape),
                json.dumps(input_shapes),
                debug_origin,
            )
            bucket_sequence.append({"identity": display_identity, "runtime": runtime})

            op_type_stats[op_name] += runtime
            total_estimated_time += runtime

        if len(bucket_sequence) >= chain_len:
            for i in range(len(bucket_sequence) - chain_len + 1):
                window = bucket_sequence[i : i + chain_len]
                chain_key = tuple(k["identity"] for k in window)
                chain_time = sum(k["runtime"] for k in window)
                chain_stats[chain_key]["time"] += chain_time
                chain_stats[chain_key]["count"] += 1

    chain_label = "Kernels" if chain_len == 1 else f"Chain of {chain_len} Kernels"
    if bucket_idx is not None:
        print(
            f"\nAnalyzed compiled bucket index {bucket_idx} with chain length {chain_len}."
        )
    else:
        print(
            f"\nAnalyzed {bucket_count} compiled buckets with chain length {chain_len}."
        )
    print(f"Total Estimated Execution Time: {format_ms(total_estimated_time)}")

    top_n = min(len(chain_stats), top_n)
    sorted_chains = sorted(
        chain_stats.items(), key=lambda x: x[1]["time"], reverse=True
    )

    formatted_chains = []
    for identities, stats in sorted_chains[:top_n]:
        parts = []
        for op_name, uid, shape, in_shapes, debug_origin in identities:
            parts.append(
                f"{op_name}({json.loads(in_shapes)}->{list(shape)}) [{debug_origin}]"
            )
        label = " -> ".join(parts)
        formatted_chains.append((label, stats))

    label_len = max((len(label) for label, _ in formatted_chains), default=0)
    col1_header = f"Top {top_n} {chain_label}"
    col1_width = max(label_len, len(col1_header))

    col2_header = "Count"
    col2_width = max(
        len(col2_header),
        max((len(str(stats["count"])) for _, stats in formatted_chains), default=0),
    )

    col3_header = "Total Time"
    col3_width = max(
        len(col3_header),
        max(
            (len(format_ms(stats["time"])) for _, stats in formatted_chains), default=0
        ),
    )

    col4_header = "Avg"
    col4_width = max(
        len(col4_header),
        max(
            (
                len(format_ms(stats["time"] / stats["count"]))
                for _, stats in formatted_chains
            ),
            default=0,
        ),
    )

    header = f"{col1_header:<{col1_width}} | {col2_header:<{col2_width}} | {col3_header:<{col3_width}} | {col4_header:<{col4_width}}"
    total_width = len(header)

    print("\n" + "=" * total_width)
    print(header)
    print("-" * total_width)

    for label, stats in formatted_chains:
        avg = stats["time"] / stats["count"]
        print(
            f"{label:<{col1_width}} | {stats['count']:<{col2_width}} | {format_ms(stats['time']):<{col3_width}} | {format_ms(avg):<{col4_width}}"
        )

    sorted_ops = sorted(op_type_stats.items(), key=lambda x: x[1], reverse=True)

    op_col1_header = "Operation Type"
    op_col1_width = max(
        len(op_col1_header), max((len(op) for op, _ in sorted_ops), default=0)
    )

    op_col2_header = "Total Time"
    op_col2_width = max(
        len(op_col2_header),
        max((len(format_ms(time)) for _, time in sorted_ops), default=0),
    )

    op_header = f"{op_col1_header:<{op_col1_width}} | {op_col2_header:<{op_col2_width}}"
    op_total_width = len(op_header)

    print("\n" + "=" * op_total_width)
    print(op_header)
    print("-" * op_total_width)
    for op, time in sorted_ops:
        print(f"{op:<{op_col1_width}} | {format_ms(time):<{op_col2_width}}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze TensorGraph performance.")
    parser.add_argument("--graph", default="dirty_region_caches/jina-v5-928x1376.bin")
    parser.add_argument("--top_n", "-n", type=int, default=20)
    parser.add_argument("--chain_len", "-c", type=int, default=1)
    parser.add_argument("--bucket", nargs="?", const="show_range", default=None)
    args = parser.parse_args()

    if args.bucket == "show_range":
        if not os.path.exists(args.graph):
            print(f"Error: Cache file '{args.graph}' does not exist.")
        else:
            cache_entries = load_cache_file(args.graph)
            compiled_buckets = [
                entry
                for entry in cache_entries
                if entry.get("type") == "compiled_bucket"
            ]
            num_buckets = len(compiled_buckets)
            if num_buckets == 0:
                print(f"No compiled buckets found in {args.graph}.")
            elif num_buckets == 1:
                print("Available bucket index: 0 (total 1 bucket)")
            else:
                print(
                    f"Available bucket range: 0 to {num_buckets - 1} (total {num_buckets} buckets)"
                )
    else:
        bucket_idx = None
        if args.bucket is not None:
            try:
                bucket_idx = int(args.bucket)
            except ValueError:
                print(
                    f"Error: Invalid bucket index '{args.bucket}'. Please specify an integer."
                )
                exit(1)
        analyze(args.graph, args.top_n, args.chain_len, bucket_idx)
