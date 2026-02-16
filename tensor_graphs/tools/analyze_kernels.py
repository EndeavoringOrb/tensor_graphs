#!/usr/bin/env python3
"""
Analyze kernel launch recordings from .jsonl files.
Usage: python tensor_graphs/tools/analyze_kernels.py [file]
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

def load_jsonl(filepath):
    """Load all records from a .jsonl file."""
    records = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def analyze_records(records):
    """Analyze kernel execution data."""
    # Group by op_type
    by_op_type = defaultdict(list)
    for rec in records:
        by_op_type[rec["op_type"]].append(rec)

    analysis = {}

    for op_type, ops in by_op_type.items():
        # Total time and count
        total_time = sum(op["compute_time_ms"] for op in ops)
        count = len(ops)
        avg_time = total_time / count if count > 0 else 0
        max_time = max((op["compute_time_ms"] for op in ops), default=0)
        min_time = min((op["compute_time_ms"] for op in ops), default=0)

        analysis[op_type] = {
            "total_time_ms": round(total_time, 3),
            "count": count,
            "avg_time_ms": round(avg_time, 3),
            "max_time_ms": round(max_time, 3),
            "min_time_ms": round(min_time, 3),
            "median_time_ms": round(sorted([op["compute_time_ms"] for op in ops])[count // 2], 3) if count > 0 else 0,
            "mode": max(set(op["mode"] for op in ops), key=lambda m: ops.count({"op_type": op_type, "mode": m})),
            "backend": ops[0]["backend"] if ops else None,
            "by_shape": {},
            "by_attrs": defaultdict(int)
        }

        # Group by shape
        shape_data = defaultdict(lambda: {"count": 0, "total_time": 0, "times": []})
        for op in ops:
            shape_key = json.dumps(op["input_shapes"])
            shape_data[shape_key]["count"] += 1
            shape_data[shape_key]["total_time"] += op["compute_time_ms"]
            shape_data[shape_key]["times"].append(op["compute_time_ms"])

        # Add shape analysis sorted by total time
        analysis[op_type]["by_shape_sorted"] = [
            {
                "shape": shape_key,
                "count": d["count"],
                "total_time_ms": round(d["total_time"], 3),
                "avg_time_ms": round(d["total_time"] / d["count"], 3)
            }
            for shape_key, d in shape_data.items()
        ]
        analysis[op_type]["by_shape"] = shape_data  # Keep original for count-based view

        # Group by attributes
        for op in ops:
            attrs_key = json.dumps(op["attrs"])
            analysis[op_type]["by_attrs"][attrs_key] += 1

    return analysis

def print_analysis(analysis):
    """Print formatted analysis."""
    print("\n" + "=" * 80)
    print("KERNEL EXECUTION ANALYSIS")
    print("=" * 80)

    if not analysis:
        print("No records found.")
        return

    # Sort by total time (descending)
    sorted_ops = sorted(analysis.items(), key=lambda x: x[1]["total_time_ms"], reverse=True)

    for op_type, data in sorted_ops:
        print(f"\n{op_type}:")
        print(f"  Total executions: {data['count']}")
        print(f"  Total time: {data['total_time_ms']:.3f} ms")
        print(f"  Average time: {data['avg_time_ms']:.3f} ms")
        print(f"  Min/Max time: {data['min_time_ms']:.3f} ms / {data['max_time_ms']:.3f} ms")
        print(f"  Median time: {data['median_time_ms']:.3f} ms")
        print(f"  Mode: {data['mode']}")
        print(f"  Backend: {data['backend']}")

        if data["by_shape_sorted"]:
            print(f"  Top input shapes BY TIME (total):")
            for item in sorted(data["by_shape_sorted"], key=lambda x: x["total_time_ms"], reverse=True)[:5]:
                print(f"    {item['shape']} - {item['total_time_ms']:.3f} ms (avg: {item['avg_time_ms']:.3f} ms, count: {item['count']})")
            if len(data["by_shape_sorted"]) > 5:
                print(f"    ... and {len(data['by_shape_sorted']) - 5} more")

        if data["by_shape"]:
            print(f"  Top input shapes BY COUNT:")
            for shape, shape_data in sorted(data["by_shape"].items(), key=lambda x: x[1]["count"], reverse=True)[:5]:
                print(f"    {shape} x {shape_data['count']}")

        if data["by_attrs"]:
            print(f"  Top attribute combinations:")
            for attrs, count in sorted(data["by_attrs"].items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {json.loads(attrs)} x {count}")

    print("\n" + "=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Analyze kernel launch recordings.")
    parser.add_argument("--file", nargs="?", help="Path to .jsonl file. Defaults to newest file.", default="kernel_launches/0.jsonl")
    args = parser.parse_args()

    if args.file:
        filepath = Path(args.file)
    else:
        # Find newest .jsonl file in tensor_graphs/tools
        tools_dir = Path(__file__).parent.parent
        jsonl_files = sorted(tools_dir.glob("*.jsonl"))
        if not jsonl_files:
            print("No .jsonl files found. Run with RECORD_KERNEL_LAUNCHES=True first.")
            return
        filepath = jsonl_files[-1]

    if not filepath.exists():
        print(f"File not found: {filepath}")
        return

    print(f"Analyzing: {filepath}")
    records = load_jsonl(filepath)
    analysis = analyze_records(records)
    print_analysis(analysis)

if __name__ == "__main__":
    main()
