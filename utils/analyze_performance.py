# File: utils/analyze_performance.py
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binary import load_cache_file
from common import format_ms, format_op_name, load_uids_from_cpp


def analyze(cache_file, top_n=20, chain_len=1, bucket_idx=None):
    print(f"Loading compiled buckets from: {cache_file}")
    cache_entries = load_cache_file(cache_file)
    uid_map = load_uids_from_cpp()

    compiled_buckets = [
        entry for entry in cache_entries if entry.get("type") == "compiled_bucket"
    ]
    num_buckets = len(compiled_buckets)

    if bucket_idx is not None:
        if bucket_idx < 0 or bucket_idx >= num_buckets:
            print(
                f"Error: Bucket index {bucket_idx} is out of range. "
                f"Available: 0 to {max(0, num_buckets - 1)}"
            )
            return
        buckets_to_analyze = [compiled_buckets[bucket_idx]]
    else:
        buckets_to_analyze = compiled_buckets

    chain_stats = defaultdict(lambda: {"time": 0.0, "count": 0})
    op_type_stats = defaultdict(float)
    total_estimated_time = 0.0

    for entry in buckets_to_analyze:
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
            op_name = format_op_name(info, f"Kernel_{hex(kernel_uid)}")

            input_shapes = [
                node_views[c].get("shape", [])
                for c in inst.get("children", [])
                if c in node_views
            ]
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
                chain_stats[chain_key]["time"] += sum(k["runtime"] for k in window)
                chain_stats[chain_key]["count"] += 1

    chain_label = "Kernels" if chain_len == 1 else f"Chain of {chain_len} Kernels"
    count_str = (
        f"index {bucket_idx}"
        if bucket_idx is not None
        else f"{len(buckets_to_analyze)} compiled buckets"
    )
    print(f"\nAnalyzed {count_str} with chain length {chain_len}.")
    print(f"Total Estimated Execution Time: {format_ms(total_estimated_time)}")

    sorted_chains = sorted(
        chain_stats.items(), key=lambda x: x[1]["time"], reverse=True
    )[:top_n]
    formatted_chains = [
        (
            " -> ".join(
                f"{op}({json.loads(in_sh)}->{list(sh)}) [{dbg}]"
                for op, _, sh, in_sh, dbg in identities
            ),
            stats,
        )
        for identities, stats in sorted_chains
    ]

    col1_w = max(
        [len(l) for l, _ in formatted_chains] + [len(f"Top {top_n} {chain_label}")],
        default=0,
    )
    col2_w = max([len(str(s["count"])) for _, s in formatted_chains] + [5], default=5)
    col3_w = max(
        [len(format_ms(s["time"])) for _, s in formatted_chains] + [10], default=10
    )
    col4_w = max(
        [len(format_ms(s["time"] / s["count"])) for _, s in formatted_chains] + [8],
        default=8,
    )

    header = f"{f'Top {top_n} {chain_label}':<{col1_w}} | {'Count':<{col2_w}} | {'Total Time':<{col3_w}} | {'Avg':<{col4_w}}"
    print("\n" + "=" * len(header) + "\n" + header + "\n" + "-" * len(header))
    for label, stats in formatted_chains:
        print(
            f"{label:<{col1_w}} | {stats['count']:<{col2_w}} | "
            f"{format_ms(stats['time']):<{col3_w}} | {format_ms(stats['time'] / stats['count']):<{col4_w}}"
        )

    sorted_ops = sorted(op_type_stats.items(), key=lambda x: x[1], reverse=True)
    op_w1 = max([len(op) for op, _ in sorted_ops] + [14], default=14)
    op_w2 = max([len(format_ms(t)) for _, t in sorted_ops] + [10], default=10)
    op_hdr = f"{'Operation Type':<{op_w1}} | {'Total Time':<{op_w2}}"
    print("\n" + "=" * len(op_hdr) + "\n" + op_hdr + "\n" + "-" * len(op_hdr))
    for op, time in sorted_ops:
        print(f"{op:<{op_w1}} | {format_ms(time):<{op_w2}}")


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
            num = len(
                [
                    e
                    for e in load_cache_file(args.graph)
                    if e.get("type") == "compiled_bucket"
                ]
            )
            print(
                f"Available bucket range: 0 to {num - 1} (total {num} buckets)"
                if num > 1
                else "Available index: 0"
                if num == 1
                else "No buckets found."
            )
    else:
        b_idx = int(args.bucket) if args.bucket is not None else None
        analyze(args.graph, args.top_n, args.chain_len, b_idx)
