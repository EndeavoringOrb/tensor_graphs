import json
import os
import argparse
from collections import defaultdict


def format_ms(ms):
    return f"{ms:.4f} ms"


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def analyze(cache_file, records_file):
    print(f"Loading benchmark records from: {records_file}")
    records = load_jsonl(records_file)

    # Create a lookup map: (kernelUid, output_shape_tuple, output_strides_tuple) -> runTime
    # We use tuples for keys because lists aren't hashable.
    bench_map = {}
    for r in records:
        key = (
            r["kernelUid"],
            tuple(r["outputShapes"][0]),
            tuple(r["outputStrides"][0]),
        )
        bench_map[key] = r["runTime"]

    print(f"Loading compiled buckets from: {cache_file}")
    cache_entries = load_jsonl(cache_file)

    # Aggregators
    kernel_stats = defaultdict(lambda: {"time": 0.0, "count": 0})
    op_type_stats = defaultdict(float)
    total_estimated_time = 0.0
    missing_benchmarks = set()

    bucket_count = 0
    for entry in cache_entries:
        if entry.get("type") != "compiled_bucket":
            continue

        bucket_count += 1
        graph = entry["graph"]
        nodes = graph["nodesMap"]
        instructions = graph["instructions"]

        for inst in instructions:
            node_id = str(inst["nodeId"])
            node = nodes[node_id]

            # Identify the operation name
            op_name = node["opType"]
            if op_name == "FUSED":
                op_name = f"FUSED_{node.get('opName', 'UNKNOWN')}"

            # Get physical metadata
            shape = tuple(node["shape"])
            strides = tuple(node["strides"])
            uid = inst["fullKernelId"]

            # Try to find timing data
            bench_key = (uid, shape, strides)
            runtime = bench_map.get(bench_key, 0.0)

            if bench_key not in bench_map:
                missing_benchmarks.add(f"{op_name} (UID: {uid}, Shape: {shape})")

            # Update aggregators
            stats_key = (op_name, uid, shape)
            kernel_stats[stats_key]["time"] += runtime
            kernel_stats[stats_key]["count"] += 1
            op_type_stats[op_name] += runtime
            total_estimated_time += runtime

    # --- Print Results ---
    print(f"\nAnalyzed {bucket_count} compiled buckets.")
    print(f"Total Estimated Execution Time: {format_ms(total_estimated_time)}")

    # 1. Top 20 Most Expensive Kernels
    print("\n" + "=" * 85)
    print(
        f"{'Top 20 Kernels (Kernel ID + Shape)':<50} | {'Count':<6} | {'Total Time':<12} | {'Avg'}"
    )
    print("-" * 85)

    sorted_kernels = sorted(
        kernel_stats.items(), key=lambda x: x[1]["time"], reverse=True
    )
    for (op_name, uid, shape), stats in sorted_kernels[:20]:
        label = f"{op_name} ({uid[:8]}...) {list(shape)}"
        avg = stats["time"] / stats["count"]
        print(
            f"{label:<50} | {stats['count']:<6} | {format_ms(stats['time']):<12} | {format_ms(avg)}"
        )

    # 2. Breakdown by OpType
    print("\n" + "=" * 40)
    print(f"{'Operation Type':<25} | {'Total Time'}")
    print("-" * 40)
    sorted_ops = sorted(op_type_stats.items(), key=lambda x: x[1], reverse=True)
    for op, time in sorted_ops:
        print(f"{op:<25} | {format_ms(time)}")

    if missing_benchmarks:
        print(
            f"\n[Warning] {len(missing_benchmarks)} kernel configurations were missing from records.jsonl."
        )
        print("Run bench.cpp to gather timing data for these kernels.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze TensorGraph performance.")
    parser.add_argument(
        "--graph",
        default="dirty_region_caches/gemma-3-270m-cpp.jsonl",
        help="Path to the cache file (JSONL with compiled_bucket entries)",
    )
    parser.add_argument(
        "--records",
        default="benchmarks/records.jsonl",
        help="Path to the benchmark records file",
    )
    args = parser.parse_args()

    try:
        analyze(args.graph, args.records)
    except Exception as e:
        print(f"\n[Error] Analysis failed: {e}")
        import traceback

        traceback.print_exc()
