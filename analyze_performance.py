import collections
import argparse
import json
import math
import os


def count_elements(shape):
    return math.prod(shape) if shape else 1


def dtype_size(dtype):
    sizes = {"FLOAT32": 4, "INT32": 4, "BF16": 2, "BOOL": 1}
    return sizes.get(dtype, 4)


def format_bytes(b):
    if b < 1024:
        return f"{b} B"
    if b < 1024**2:
        return f"{b/1024:.2f} KB"
    if b < 1024**3:
        return f"{b/(1024**2):.2f} MB"
    return f"{b/(1024**3):.2f} GB"


def estimate_flops(op_type, target_elems, in_shapes):
    if op_type == "DOT":
        if len(in_shapes) < 2:
            return 0
        s0, s1 = in_shapes[0], in_shapes[1]
        if len(s0) == 3 and len(s1) == 3:
            return 2 * s0[0] * s0[1] * s1[2] * s0[2]
        elif len(s0) == 2 and len(s1) == 2:
            return 2 * s0[0] * s1[1] * s0[1]
    elif op_type in {
        "ADD",
        "MUL",
        "DIVIDE",
        "POWER",
        "SIN",
        "COS",
        "NEGATE",
        "CAST",
        "TRIU",
        "FUSED",
    }:
        return target_elems
    elif op_type in {"SUM", "MAX"}:
        return count_elements(in_shapes[0]) if in_shapes else 0
    return 0


def load_records(filepath):
    records = collections.defaultdict(list)
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Runtime data will be 0.")
        return records
    with open(filepath, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            records[rec["kernelUid"]].append(rec)
    return records


def load_compiled_graphs(filepath):
    """Load all compiled_bucket entries from the cache file.

    Returns a list of (key, compiled_graph, bucket) tuples.
    The cache format is JSONL with entries of type 'compiled_bucket' and 'constants'.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find {filepath}")

    graphs = []
    with open(filepath, "r") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("type") == "compiled_bucket":
                key = entry["key"]
                graph_data = entry["graph"]
                bucket_data = entry["bucket"]
                graphs.append((key, graph_data, bucket_data))

    if not graphs:
        raise ValueError(f"No compiled_bucket entries found in {filepath}")

    return graphs


def print_dynamic_table(data_rows, headers, col_widths=None):
    if not data_rows:
        print("No data available for this category.")
        return
    if not col_widths:
        col_widths = []
        for i in range(len(headers)):
            col_data = [str(r[i]) for r in data_rows]
            max_val = max([len(headers[i])] + [len(d) for d in col_data])
            col_widths.append(max_val + 2)
    header_str = " | ".join([f"{h:<{col_widths[i]}}" for i, h in enumerate(headers)])
    print(header_str)
    print("-" * len(header_str))
    for row in data_rows:
        print(" | ".join([f"{str(row[i]):<{col_widths[i]}}" for i in range(len(row))]))


def analyze(graph_file, records_file):
    records = load_records(records_file)
    try:
        graphs = load_compiled_graphs(graph_file)
    except Exception as e:
        print(f"Error loading graph: {e}")
        return

    if not graphs:
        print("No compiled graphs found in cache file")
        return

    print(f"Analyzing {len(graphs)} compiled bucket(s)...")

    # Aggregate trace data from all compiled buckets
    trace = []

    for key, cg, bucket in graphs:
        nodes_map = cg["nodesMap"]
        instructions = cg["instructions"]
        ref_counts = cg["refCounts"]

        for inst in instructions:
            out_node_id = inst["nodeId"]
            out_node = nodes_map[str(out_node_id)]
            in_nodes = [nodes_map[str(nid)] for nid in inst["inputNodeIds"]]

            target_elems = count_elements(out_node["shape"])
            out_bytes = target_elems * dtype_size(out_node["dtype"])
            in_bytes = sum(
                count_elements(n["shape"]) * dtype_size(n["dtype"]) for n in in_nodes
            )

            is_inplace_candidate = False
            for nid in inst["inputNodeIds"]:
                in_node = nodes_map[str(nid)]
                count = int(ref_counts.get(str(nid), 0))
                if count == 1 and in_node["shape"] == out_node["shape"]:
                    is_inplace_candidate = True
                    break

            est_time = 0.0
            # Collect all kernel IDs: fullKernelId + all cachedKernelIds
            all_k_ids = []
            cached_k_ids = inst.get("cachedKernelIds", [])
            all_k_ids.extend(cached_k_ids)
            if not all_k_ids:
                full_k_id = inst.get("fullKernelId")
                if full_k_id:
                    all_k_ids.append(full_k_id)

            for k_id in all_k_ids:
                recs = records.get(k_id, [])
                # Try to find an exact matching record across all kernel IDs
                best_rec = None
                matched_k_id = None

                for rec in recs:
                    # Check for exact shape, dtype, and strides match
                    if (
                        rec["outputShapes"][0] == out_node["shape"]
                        and rec["outputDTypes"][0] == out_node["dtype"]
                        and rec["outputStrides"][0] == out_node["view"]["strides"]
                    ):
                        best_rec = rec
                        matched_k_id = k_id
                        break
                if not best_rec:
                    continue

                if best_rec:
                    est_time = best_rec["runTime"]

                op_type = out_node["opType"]
                in_shapes = [n["shape"] for n in in_nodes]
                flops = estimate_flops(op_type, target_elems, in_shapes)
                name = (
                    op_type
                    if op_type != "FUSED"
                    else f'FUSED_{out_node.get("opName", "UNKNOWN")}'
                )

                trace.append(
                    {
                        "name": name,
                        "kernelId": matched_k_id,
                        "shape": tuple(out_node["shape"]),
                        "in_shapes": tuple(tuple(s) for s in in_shapes),
                        "out_node_id": out_node_id,
                        "in_node_ids": inst["inputNodeIds"],
                        "time": est_time,
                        "bytes_read": in_bytes,
                        "bytes_written": out_bytes,
                        "flops": flops,
                        "inplace": is_inplace_candidate,
                    }
                )

    print(f"Collected {len(trace):,} traces.")

    # 1. Arithmetic Intensity
    print("\n" + "=" * 80 + "\n1. ARITHMETIC INTENSITY ANALYSIS\n" + "=" * 80)
    ai_stats = collections.defaultdict(lambda: {"flops": 0, "bytes": 0, "time": 0.0})
    for t in trace:
        ai_stats[t["name"]]["flops"] += t["flops"]
        ai_stats[t["name"]]["bytes"] += t["bytes_read"] + t["bytes_written"]
        ai_stats[t["name"]]["time"] += t["time"]

    ai_list = []
    for name, stats in ai_stats.items():
        intensity = stats["flops"] / stats["bytes"] if stats["bytes"] > 0 else 0
        bound = "Memory Bound" if intensity < 2.0 else "Compute Bound"
        if intensity == 0:
            bound = "Memory (Data Move)"
        ai_list.append((name, f"{intensity:.4f}", f"{stats['time']:.2f}", bound))
    ai_list.sort(key=lambda x: float(x[1]))
    print_dynamic_table(
        ai_list, ["Operation", "Intensity (FLOPs/Byte)", "Time (ms)", "Bound"]
    )

    # 2 & 3. N-Grams
    print("\n" + "=" * 80 + "\n2 & 3. N-GRAM ANALYSIS\n" + "=" * 80)
    ngram_data = {
        True: collections.defaultdict(lambda: {"mem": 0, "time": 0, "count": 0}),
        False: collections.defaultdict(lambda: {"mem": 0, "time": 0, "count": 0}),
    }
    for n in range(2, 7):
        for i in range(len(trace) - n + 1):
            window = trace[i : i + n]
            key = " -> ".join(t["name"] for t in window)
            is_chain_inplace = all(t["inplace"] for t in window)
            runtime = sum(t["time"] for t in window)
            win_nodes = {t["out_node_id"] for t in window}
            saved = sum(
                2 * t["bytes_written"]
                for t in window
                for in_id in t["in_node_ids"]
                if in_id in win_nodes
            )
            target = ngram_data[is_chain_inplace][key]
            target["mem"] += saved
            target["time"] += runtime
            target["count"] += 1

    for is_inplace in [True, False]:
        label = "INPLACE" if is_inplace else "NOT INPLACE"
        print(f"\n--- {label} CHAINS ---")
        top_time = sorted(
            ngram_data[is_inplace].items(), key=lambda x: x[1]["time"], reverse=True
        )[:10]
        rows_time = [(k, v["count"], f"{v['time']:.2f} ms") for k, v in top_time]
        print_dynamic_table(rows_time, ["Chain", "Count", "Total Time"])

    # 4. Shape-Preserving
    print("\n" + "=" * 80 + "\n4. SHAPE-PRESERVING SUBGRAPHS\n" + "=" * 80)
    ew_ops = {"ADD", "MUL", "DIVIDE", "POWER", "SIN", "COS", "NEGATE", "CAST", "TRIU"}
    subgraphs, current_sg = [], []
    for t in trace:
        is_sp = (t["name"] in ew_ops or t["name"].startswith("FUSED_")) and all(
            in_s == t["shape"] for in_s in t["in_shapes"]
        )
        if is_sp:
            current_sg.append(t)
        else:
            if len(current_sg) > 1:
                subgraphs.append(list(current_sg))
            current_sg = []
    if len(current_sg) > 1:
        subgraphs.append(list(current_sg))

    for is_inplace in [True, False]:
        label = "INPLACE" if is_inplace else "NOT INPLACE"
        print(f"\n--- {label} SHAPE-PRESERVING SUBGRAPHS ---")
        agg_shape = collections.defaultdict(lambda: {"time": 0.0, "count": 0})
        for sg in subgraphs:
            if all(n["inplace"] for n in sg) == is_inplace:
                chain_shape = f"{' -> '.join(n['name'] for n in sg)} | {sg[0]['shape']}"
                agg_shape[chain_shape]["time"] += sum(n["time"] for n in sg)
                agg_shape[chain_shape]["count"] += 1
        shape_rows = sorted(
            [(k, v["count"], f"{v['time']:.2f}") for k, v in agg_shape.items()],
            key=lambda x: float(x[2]),
            reverse=True,
        )[:10]
        print_dynamic_table(
            shape_rows, ["Chain | Shape", "Occurrences", "Total Time (ms)"]
        )

    # 5. Top Kernels (Grouped by Kernel ID + Shape)
    print("\n" + "=" * 80 + "\n5. TOP KERNELS BY SHAPE & ID\n" + "=" * 80)

    # We group by (Operation Name, KernelId, Shape)
    k_shape_stats = collections.defaultdict(
        lambda: {"time": 0.0, "count": 0, "inplace_count": 0}
    )

    for t in trace:
        # Construct a tuple key to ensure unique grouping
        group_key = (t["name"], t["kernelId"], t["shape"])
        k_shape_stats[group_key]["time"] += t["time"]
        k_shape_stats[group_key]["count"] += 1
        if t["inplace"]:
            k_shape_stats[group_key]["inplace_count"] += 1

    kernel_shape_rows = []
    for (name, kid, shape), stats in k_shape_stats.items():
        kernel_label = f"{name} ({kid[:8]}...)" if kid else f"{name} (None)"
        kernel_shape_rows.append(
            (
                kernel_label,
                str(shape),
                stats["count"],
                f"{stats['time']:.2f}",
                f"{stats['time']/stats['count']:.3f}",
                f"{stats['inplace_count']}/{stats['count']}",
            )
        )

    # Sort by total time descending
    kernel_shape_rows.sort(key=lambda x: float(x[3]), reverse=True)

    print_dynamic_table(
        kernel_shape_rows[:50],
        ["Kernel", "Shape", "Count", "Total (ms)", "Avg (ms)", "Inplace/Total"],
    )

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph",
        default="dirty_region_caches/gemma-3-270m-cpp.jsonl",
        help="Path to the cache file (JSONL format with compiled_bucket entries)",
    )
    parser.add_argument(
        "--records",
        default="benchmarks/records.jsonl",
        help="Path to the benchmark records file (JSONL format)",
    )
    args = parser.parse_args()
    analyze(args.graph, args.records)
