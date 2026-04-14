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


def analyze(cache_file, records_file, top_n=20, chain_len=1):
    print(f"Loading benchmark records from: {records_file}")
    records = load_jsonl(records_file)

    # Create a lookup map: (kernelUid, output_shape_tuple, output_strides_tuple) -> runTime
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
    # Key: Tuple of (op_name, uid, shape) entries of length chain_len
    chain_stats = defaultdict(lambda: {"time": 0.0, "count": 0})
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

        # 1. First, build a linear list of kernel metadata and times for this bucket
        bucket_sequence = []
        for inst in instructions:
            node_id = str(inst["nodeId"])
            node = nodes[node_id]

            op_name = node["opType"]
            if op_name == "FUSED":
                op_name = f"FUSED_{node.get('opName', 'UNKNOWN')}"

            input_shapes = [nodes[str(pid)]["shape"] if str(pid) in nodes else [] for pid in node["parentIds"]]
            shape = tuple(node["shape"])
            strides = tuple(node["strides"])
            uid = inst["fullKernelId"]

            bench_key = (uid, shape, strides)
            runtime = bench_map.get(bench_key, 0.0)

            if bench_key not in bench_map:
                missing_benchmarks.add(f"{op_name} (UID: {uid}, Shape: {shape})")

            # The identity of a kernel in a chain
            identity = (op_name, uid, shape, json.dumps(input_shapes))
            bucket_sequence.append({"identity": identity, "runtime": runtime})

            # Still update global op_type stats regardless of chain length
            op_type_stats[op_name] += runtime
            total_estimated_time += runtime

        # 2. Extract N-grams (chains) from the sequence
        if len(bucket_sequence) >= chain_len:
            for i in range(len(bucket_sequence) - chain_len + 1):
                window = bucket_sequence[i : i + chain_len]

                # Composite key: ((op1, uid1, shp1), (op2, uid2, shp2), ...)
                chain_key = tuple(k["identity"] for k in window)

                # The "cost" of a chain is the sum of its component runtimes
                chain_time = sum(k["runtime"] for k in window)

                chain_stats[chain_key]["time"] += chain_time
                chain_stats[chain_key]["count"] += 1

    # --- Print Results ---
    chain_label = "Kernels" if chain_len == 1 else f"Chain of {chain_len} Kernels"
    print(f"\nAnalyzed {bucket_count} compiled buckets with chain length {chain_len}.")
    print(f"Total Estimated Execution Time: {format_ms(total_estimated_time)}")

    # 1. Top top_n Most Expensive Chains
    top_n = min(len(chain_stats), top_n)
    print("\n" + "=" * 100)
    print(
        f"{f'Top {top_n} {chain_label}':<105} | {'Count':<6} | {'Total Time':<12} | {'Avg'}"
    )
    print("-" * 100)

    sorted_chains = sorted(
        chain_stats.items(), key=lambda x: x[1]["time"], reverse=True
    )

    label_len = 0
    for identities, stats in sorted_chains[:top_n]:
        # Format the chain identity for display
        parts = []
        for op_name, uid, shape, input_shapes in identities:
            parts.append(f"{op_name}({input_shapes}->{list(shape)})")

        label = " -> ".join(parts)
        label_len = max(len(label), label_len)

    for identities, stats in sorted_chains[:top_n]:
        # Format the chain identity for display
        parts = []
        for op_name, uid, shape, input_shapes in identities:
            parts.append(f"{op_name}({input_shapes}->{list(shape)})")

        label = " -> ".join(parts)

        avg = stats["time"] / stats["count"]
        print(
            f"{label:<{label_len}} | {stats['count']:<6} | {format_ms(stats['time']):<12} | {format_ms(avg)}"
        )

    # 2. Breakdown by OpType (Always per-operation)
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
    parser.add_argument(
        "--top_n",
        "-n",
        type=int,
        default=20,
        help="Number of items to print",
    )
    parser.add_argument(
        "--chain_len",
        "-c",
        type=int,
        default=1,
        help="Number of kernels in a sequence (N-gram) to analyze",
    )
    args = parser.parse_args()

    try:
        analyze(args.graph, args.records, args.top_n, args.chain_len)
    except Exception as e:
        print(f"\n[Error] Analysis failed: {e}")
        import traceback

        traceback.print_exc()
