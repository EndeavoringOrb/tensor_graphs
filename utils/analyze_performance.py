# File: analyze_performance.py
import argparse
import json
from collections import defaultdict
from binary import load_cache_file


def format_ms(ms):
    return f"{ms:.4f} ms"


def analyze(cache_file, top_n=20, chain_len=1):
    print(f"Loading compiled buckets from: {cache_file}")
    cache_entries = load_cache_file(cache_file)

    chain_stats = defaultdict(lambda: {"time": 0.0, "count": 0})
    op_type_stats = defaultdict(float)
    total_estimated_time = 0.0

    bucket_count = 0
    for entry in cache_entries:
        if entry.get("type") != "compiled_bucket":
            continue

        bucket_count += 1
        graph = entry["graph"]
        nodes = graph["nodesMap"]
        instructions = graph["instructions"]
        node_costs = graph.get("nodeCosts", {})

        bucket_sequence = []
        for inst in instructions:
            node_id = inst["nodeId"]
            node = nodes[str(node_id)]

            # Fetch the runtime exactly as the C++ planner saw it
            runtime = node_costs[node_id] # I want this to error if node_id isn't present


            op_name = node["opType"]
            if op_name == "FUSED":
                op_name = f"FUSED_{node.get('opName', 'UNKNOWN')}"

            input_shapes = []
            for pid in inst["inputNodeIds"]:
                p_node = nodes[str(pid)]
                input_shapes.append(p_node["shape"])

            shape = node["shape"]

            # Identity for display purposes in the report
            display_identity = (
                op_name,
                hex(inst["fullKernelId"]),
                tuple(shape),
                json.dumps(input_shapes),
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
    print(f"\nAnalyzed {bucket_count} compiled buckets with chain length {chain_len}.")
    print(f"Total Estimated Execution Time: {format_ms(total_estimated_time)}")

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
        parts = []
        for op_name, uid, shape, in_shapes in identities:
            parts.append(f"{op_name}({json.loads(in_shapes)}->{list(shape)})")
        label = " -> ".join(parts)
        label_len = max(len(label), label_len)

    for identities, stats in sorted_chains[:top_n]:
        parts = []
        for op_name, uid, shape, in_shapes in identities:
            parts.append(f"{op_name}({json.loads(in_shapes)}->{list(shape)})")
        label = " -> ".join(parts)
        avg = stats["time"] / stats["count"]
        print(
            f"{label:<{label_len}} | {stats['count']:<6} | {format_ms(stats['time']):<12} | {format_ms(avg)}"
        )

    print("\n" + "=" * 40)
    print(f"{'Operation Type':<25} | {'Total Time'}")
    print("-" * 40)
    sorted_ops = sorted(op_type_stats.items(), key=lambda x: x[1], reverse=True)
    for op, time in sorted_ops:
        print(f"{op:<25} | {format_ms(time)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze TensorGraph performance.")
    parser.add_argument("--graph", default="dirty_region_caches/flux-trans.bin")
    parser.add_argument("--top_n", "-n", type=int, default=20)
    parser.add_argument("--chain_len", "-c", type=int, default=1)
    args = parser.parse_args()

    analyze(args.graph, args.top_n, args.chain_len)
