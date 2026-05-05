# File: kernel_bench/app.py
from flask import Flask, jsonify, request, render_template
import os
import json
import re
import struct
from pathlib import Path
from collections import defaultdict
from .jobs import (
    create_job,
    jobs,
    load_job_history,
    get_hw_info,
    start_worker,
    PROJECT_ROOT,
)

app = Flask(__name__)
start_worker()


def format_constants(data, dtype):
    """Reinterprets a list of integers (bytes) into the specified dtype."""
    if not data:
        return data
    raw_bytes = bytes(data)
    if dtype == "FLOAT32":
        count = len(raw_bytes) // 4
        return list(struct.unpack(f"<{count}f", raw_bytes))
    elif dtype == "INT32":
        count = len(raw_bytes) // 4
        return list(struct.unpack(f"<{count}i", raw_bytes))
    return data


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/api/kernels/test")
def submit_kernel():
    data = request.get_json(force=True, silent=True)
    if not data or not data.get("source") or not data.get("opname"):
        return jsonify({"error": "Missing 'source' or 'opname'"}), 400

    backend = data.get("backend", "cpu")
    target_model = data.get("target_model", "gemma-3-270m")

    job_id = create_job(data["source"], data["opname"], backend, target_model)
    return jsonify({"job_id": job_id, "status": "queued"}), 202


@app.get("/api/kernels/file/<job_id>")
def read_kernel_file(job_id):
    history = load_job_history()
    job = next((j for j in history if j["job_id"] == job_id), None)
    if not job and job_id in jobs:
        job = jobs[job_id]

    if not job or not job.get("kernel_file"):
        return jsonify({"error": "File not found"}), 404

    content = Path(job["kernel_file"]).read_text()
    return jsonify({"content": content})


@app.get("/api/jobs/<job_id>")
def get_job(job_id):
    job = jobs.get(job_id)
    if not job:
        history = load_job_history()
        job = next((j for j in history if j["job_id"] == job_id), None)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.get("/api/history")
def get_history():
    return jsonify(load_job_history())


@app.get("/api/hwinfo")
def get_hardware_info():
    return jsonify({"hwinfo": get_hw_info()})


@app.get("/api/read_benchmarks")
def get_read_benchmarks():
    op_filter = request.args.get("op", "")
    shape_filter = request.args.get("shape", "")
    target_model = request.args.get("target_model", "gemma-3-270m")

    records_path = PROJECT_ROOT / "benchmarks" / "records.jsonl"
    cache_path = PROJECT_ROOT / "dirty_region_caches" / f"{target_model}-cpp.jsonl"
    header_path = (
        PROJECT_ROOT / "tensor_graphs_cpp" / "generated" / "kernel_uids.gen.hpp"
    )

    if not records_path.exists():
        return jsonify({"records": []})

    # Build Map for resolving human-readable Names
    uid_map = {}

    # 1. Load UIDs from Cache
    if cache_path.exists():
        with open(cache_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("type") == "compiled_bucket":
                    nodes = entry["graph"]["nodesMap"]
                    for inst in entry["graph"]["instructions"]:
                        uid = str(inst["fullKernelId"])
                        node = nodes[str(inst["nodeId"])]
                        op_name = node["opType"]
                        if op_name == "FUSED":
                            op_name = f"FUSED_{node.get('opName', 'UNKNOWN')}"
                        uid_map[uid] = op_name

    # 2. Load UIDs from C++ Header
    if header_path.exists():
        pattern = re.compile(r"constexpr uint64_t\s+(\w+)\s+=\s+(0x[0-9a-fA-F]+)ULL;")

        with open(header_path, "r") as f:
            for name, hex_val in pattern.findall(f.read()):
                val_int = int(hex_val, 16)
                uid_map[str(val_int)] = name
                uid_map[hex_val.lower()] = name

    records = []
    with open(records_path, "r") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                uid = str(r.get("kernelUid", ""))

                # Resolve the proper human-readable opName
                opname = uid_map.get(uid, r.get("opName", "UNKNOWN"))
                r["opName"] = opname  # Replace so the agent gets the proper name

                shapes = str(r.get("outputShapes", [])) + str(r.get("inputShapes", []))

                # Apply Regex for OpName (against Name and UID)
                if op_filter:
                    if not re.search(
                        op_filter, opname, re.IGNORECASE
                    ) and not re.search(op_filter, uid, re.IGNORECASE):
                        continue

                # Apply Regex for Shapes
                if shape_filter:
                    if not re.search(shape_filter, shapes):
                        continue

                # Format Constants (from raw byte array to Float/Int arrays)
                in_consts = r.get("inputConstants", [])
                in_dtypes = r.get("inputDTypes", [])
                if in_consts and in_dtypes:
                    formatted_consts = []
                    for idx, data in enumerate(in_consts):
                        dt = in_dtypes[idx] if idx < len(in_dtypes) else ""
                        formatted_consts.append(format_constants(data, dt))
                    r["inputConstants"] = formatted_consts

                records.append(r)

    return jsonify({"records": records})


@app.get("/api/analyze")
def get_analyze():
    target_model = request.args.get("target_model", "gemma-3-270m")
    records_path = PROJECT_ROOT / "benchmarks" / "records.jsonl"
    cache_path = PROJECT_ROOT / "dirty_region_caches" / f"{target_model}-cpp.jsonl"

    if not records_path.exists() or not cache_path.exists():
        return jsonify({"error": "No benchmark or cache data available yet."}), 404

    bench_map = {}
    with open(records_path, "r") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                key = (
                    r["kernelUid"],
                    tuple(r["outputShapes"][0]),
                    tuple(r["outputStrides"][0]),
                )
                bench_map[key] = r["runTime"]

    total_estimated_time = 0.0
    extracted_uids = set()
    chain_stats = defaultdict(lambda: {"time": 0.0, "count": 0})
    op_type_stats = defaultdict(float)

    with open(cache_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("type") != "compiled_bucket":
                continue

            graph = entry["graph"]
            nodes = graph["nodesMap"]

            for inst in graph["instructions"]:
                node_id = str(inst["nodeId"])
                node = nodes[node_id]
                uid = inst["fullKernelId"]
                extracted_uids.add(uid)

                op_name = node["opType"]
                if op_name == "FUSED":
                    op_name = f"FUSED_{node.get('opName', 'UNKNOWN')}"

                shape = tuple(node["shape"])
                strides = tuple(node["strides"])
                bench_key = (uid, shape, strides)

                runtime = bench_map.get(bench_key, 0.0)
                total_estimated_time += runtime
                op_type_stats[op_name] += runtime

                # Length-1 chain registration
                input_shapes = [
                    nodes[str(pid)]["shape"] if str(pid) in nodes else []
                    for pid in node["parentIds"]
                ]
                identity = f"{op_name}({input_shapes}->{list(shape)})"
                chain_stats[identity]["time"] += runtime
                chain_stats[identity]["count"] += 1

    top_chains = sorted(
        [
            {"chain": k, "time": v["time"], "count": v["count"]}
            for k, v in chain_stats.items()
        ],
        key=lambda x: x["time"],
        reverse=True,
    )[:20]

    top_ops = sorted(
        [{"op": k, "time": v} for k, v in op_type_stats.items()],
        key=lambda x: x["time"],
        reverse=True,
    )

    return jsonify(
        {
            "total_estimated_time_ms": total_estimated_time,
            "extracted_uids": list(extracted_uids),
            "top_chains": top_chains,
            "top_ops": top_ops,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
