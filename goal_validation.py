"""Shared report contract for end-to-end goal runners."""
import json
import math
import statistics
from pathlib import Path


def writeGoalReport(session, root_id, cache_path, report_path, check_bucket, warmup=1, iterations=8, plan_only=False):
    from utils.binary import loadCacheFile
    entries = loadCacheFile(cache_path, lazy=True)
    metadata = next(entry for entry in entries if entry["type"] == "metadata")
    graphs = [entry["graph"] for entry in entries if entry["type"] == "compiled_bucket"]
    if not plan_only:
        for index, graph in enumerate(graphs):
            check_bucket(session.run_float32(root_id, index), graph["bucket"])
    samples = [[] for graph in graphs] if plan_only else session.benchmark_buckets(warmup, iterations)
    if len(samples) != len(graphs) or len(samples) != len(metadata["bucketWeights"]):
        raise RuntimeError("Measured buckets do not match the cache metadata")
    buckets = [{"bucket": graph["bucket"], "weight": weight, "samples_ms": values,
                "median_ms": statistics.median(values) if values else 0.0,
                "kernel_ids": [str(inst["kernelId"]) for inst in graph["instructions"]]}
               for graph, weight, values in zip(graphs, metadata["bucketWeights"], samples)]
    registry_path = Path("tensor_graphs_cpp/generated/kernel_uids.json")
    registry = json.loads(registry_path.read_text()) if registry_path.is_file() else {}
    for bucket, graph in zip(buckets, graphs):
        bucket["executed_workloads"] = [{"kernel_id": str(inst["kernelId"]),
            "kernel_name": registry.get(str(inst["kernelId"]), {}).get("name"),
            "logical_id": inst["logicalId"], "output": graph["nodeViews"].get(inst["eclassId"]),
            "inputs": [graph["nodeViews"].get(child) for child in inst["children"]],
            "output_memory": inst["outBuffer"], "input_memory": inst["inBuffers"], "engines": inst["engines"],
            "estimated_ms": graph["nodeCosts"].get(inst["eclassId"]) if math.isfinite(graph["nodeCosts"].get(inst["eclassId"], math.inf)) else None}
            for inst in graph["instructions"] if inst["kernelId"]]
    report = {"correctness_passed": not plan_only, "plan_only": plan_only, "logical_graph": session.logical_graph_definition(),
              "buckets": buckets, "timing_scope": "synchronized compiled bucket execution"}
    Path(report_path).write_text(json.dumps(report, allow_nan=False))
    return report
