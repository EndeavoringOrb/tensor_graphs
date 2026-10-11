"""Shared report contract for end-to-end goal runners."""
import json
import statistics
from pathlib import Path


def writeGoalReport(session, root_id, cache_path, report_path, check_bucket, warmup=1, iterations=8):
    from utils.binary import loadCacheFile
    entries = loadCacheFile(cache_path, lazy=True)
    metadata = next(entry for entry in entries if entry["type"] == "metadata")
    graphs = [entry["graph"] for entry in entries if entry["type"] == "compiled_bucket"]
    for index, graph in enumerate(graphs):
        check_bucket(session.run_float32(root_id, index), graph["bucket"])
    samples = session.benchmark_buckets(warmup, iterations)
    if len(samples) != len(graphs) or len(samples) != len(metadata["bucketWeights"]):
        raise RuntimeError("Measured buckets do not match the cache metadata")
    buckets = [{"bucket": graph["bucket"], "weight": weight, "samples_ms": values,
                "median_ms": statistics.median(values) if values else 0.0,
                "kernel_ids": [str(inst["kernelId"]) for inst in graph["instructions"]]}
               for graph, weight, values in zip(graphs, metadata["bucketWeights"], samples)]
    report = {"correctness_passed": True, "logical_graph": session.logical_graph_definition(),
              "buckets": buckets, "timing_scope": "synchronized compiled bucket execution"}
    Path(report_path).write_text(json.dumps(report, allow_nan=False))
    return report
