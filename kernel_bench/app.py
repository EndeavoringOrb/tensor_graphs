# File: kernel_bench/app.py
from flask import Flask, jsonify, request, render_template
import os
import json
import re
import struct
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from .jobs import (
    create_job,
    jobs,
    load_job_history,
    get_hw_info,
    start_worker,
    PROJECT_ROOT,
    KERNELS_DIR,
    BinaryReader,
    save_report,
    load_reports,
)

app = Flask(__name__)
start_worker()


def format_constants(raw_bytes, dtype):
    if not raw_bytes:
        return ""
    dtypes = ["FLOAT32", "INT32", "BF16", "UINT8"]
    dt_str = (
        dtypes[dtype] if isinstance(dtype, int) and dtype < len(dtypes) else str(dtype)
    )
    if dt_str == "FLOAT32":
        count = len(raw_bytes) // 4
        return list(struct.unpack(f"<{count}f", raw_bytes))
    elif dt_str == "INT32":
        count = len(raw_bytes) // 4
        return list(struct.unpack(f"<{count}i", raw_bytes))
    return list(raw_bytes)


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
    history = load_job_history()
    return jsonify(
        {"history": history, "message": f"{len(history)} item(s) in history"}
    )


@app.get("/api/hwinfo")
def get_hardware_info():
    return jsonify({"hwinfo": get_hw_info()})


@app.get("/api/read_benchmarks")
def get_read_benchmarks():
    op_filter = request.args.get("op", "")
    shape_filter = request.args.get("shape", "")
    target_model = request.args.get("target_model", "gemma-3-270m")

    records_path = PROJECT_ROOT / "benchmarks" / "records.bin"
    cache_path = PROJECT_ROOT / "dirty_region_caches" / f"{target_model}-cpp.bin"
    header_path = (
        PROJECT_ROOT / "tensor_graphs_cpp" / "generated" / "kernel_uids.gen.hpp"
    )

    if not records_path.exists():
        return jsonify({"records": []})

    uid_map = {}
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            br = BinaryReader(f)
            while True:
                t = br.read_u8()
                if t is None:
                    break
                if t == 1:  # Compiled Bucket
                    br.read_string()  # key
                    graph = br.read_compiled_graph()
                    nodes = graph["nodesMap"]
                    for inst in graph["instructions"]:
                        uid = str(inst["fullKernelId"])
                        node = nodes[str(inst["nodeId"])]
                        op_name = node["opType"]
                        if op_name == "FUSED":
                            op_name = f"FUSED_{node.get('opName', 'UNKNOWN')}"
                        uid_map[uid] = op_name
                elif t == 0:  # Metadata
                    br.read_u32()
                    br.read_u32()
                    br.read_map(br.read_u32, br.read_backend)
                elif t == 2:  # Constants
                    count = br.read_u32()
                    for _ in range(count):
                        br.read_u32()
                        f.read(br.read_u32())
                else:
                    break

    if header_path.exists():
        pattern = re.compile(r"constexpr uint64_t\s+(\w+)\s+=\s+(0x[0-9a-fA-F]+)ULL;")
        with open(header_path, "r") as f:
            for name, hex_val in pattern.findall(f.read()):
                val_int = int(hex_val, 16)
                uid_map[str(val_int)] = name
                uid_map[hex_val.lower()] = name

    records = []
    with open(records_path, "rb") as f:
        br = BinaryReader(f)
        while True:
            r = br.read_record()
            if r is None:
                break

            uid = str(r.get("kernelUid", ""))
            opname = uid_map.get(
                uid, uid_map.get(hex(r.get("kernelUid", 0)), r.get("opName", "UNKNOWN"))
            )
            r["opName"] = opname
            r["kernelUid"] = hex(r["kernelUid"])

            shapes = str(r.get("outputShapes", [])) + str(r.get("inputShapes", []))
            if op_filter:
                if not re.search(op_filter, opname, re.IGNORECASE) and not re.search(
                    op_filter, uid, re.IGNORECASE
                ):
                    continue
            if shape_filter:
                if not re.search(shape_filter, shapes):
                    continue

            in_consts = r.get("inputConstants", [])
            in_dtypes = r.get("inputDTypes", [])
            if in_consts and in_dtypes:
                formatted_consts = []
                for idx, data in enumerate(in_consts):
                    dt = in_dtypes[idx] if idx < len(in_dtypes) else -1
                    formatted_consts.append(format_constants(data, dt))
                r["inputConstants"] = formatted_consts
            records.append(r)

    return jsonify({"records": records})


@app.get("/api/analyze")
def get_analyze():
    target_model = request.args.get("target_model", "gemma-3-270m")
    records_path = PROJECT_ROOT / "benchmarks" / "records.bin"
    if target_model == "gemma-3-270m":
        cache_paths = [PROJECT_ROOT / "dirty_region_caches" / f"{target_model}-cpp.bin"]
    elif target_model == "flux-klein-4b":
        cache_paths = [
            PROJECT_ROOT / "dirty_region_caches" / f"flux-text.bin",
            PROJECT_ROOT / "dirty_region_caches" / f"flux-trans.bin",
            PROJECT_ROOT / "dirty_region_caches" / f"flux-vae.bin",
        ]
    else:
        return jsonify({"error": f'Unrecognized target model "{target_model}"'}), 404

    if not records_path.exists() or any(
        not cache_path.exists() for cache_path in cache_paths
    ):
        return jsonify({"error": "No benchmark or cache data available yet."}), 404

    bench_map = {}
    with open(records_path, "rb") as f:
        br = BinaryReader(f)
        while True:
            r = br.read_record()
            if r is None:
                break
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

    for cache_path in cache_paths:
        with open(cache_path, "rb") as f:
            br = BinaryReader(f)
            while True:
                t = br.read_u8()
                if t is None:
                    break
                if t == 1:  # Compiled Bucket
                    br.read_string()  # key
                    graph = br.read_compiled_graph()
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
                        input_shapes = [
                            nodes[str(pid)]["shape"] if str(pid) in nodes else []
                            for pid in node["parentIds"]
                        ]
                        identity = f"{op_name}({input_shapes}->{list(shape)})"
                        chain_stats[identity]["time"] += runtime
                        chain_stats[identity]["count"] += 1
                elif t == 0:  # Metadata
                    br.read_u32()
                    br.read_u32()
                    br.read_map(br.read_u32, br.read_backend)
                elif t == 2:  # Constants
                    count = br.read_u32()
                    for _ in range(count):
                        br.read_u32()
                        f.read(br.read_u32())
                else:
                    break

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
            "extracted_uids": [hex(u) for u in extracted_uids],
            "top_chains": top_chains,
            "top_ops": top_ops,
        }
    )


@app.get("/api/kernels/list")
def list_kernel_files():
    """Recursively lists all kernel files in the kernels directory."""
    try:
        files = []
        for path in KERNELS_DIR.rglob("*"):
            if path.is_file():
                files.append(str(path.relative_to(KERNELS_DIR)))
        return jsonify({"files": sorted(files)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/kernels/read_model")
def read_model_source():
    target_model = request.args.get("target_model", "gemma-3-270m")
    model_files = {
        "gemma-3-270m": ["gemma-3-270m.hpp"],
        "flux-klein-4b": [
            "flux-klein-4b.hpp",
            "flux-klein-4b-text_encoder.hpp",
            "flux-klein-4b-transformer.hpp",
            "flux-klein-4b-vae.hpp",
        ],
    }

    files_to_read = model_files.get(target_model, [])
    if not files_to_read:
        return jsonify({"error": f"No source files mapped for {target_model}"}), 404

    content = ""
    for fname in files_to_read:
        path = PROJECT_ROOT / "tensor_graphs_cpp" / "models" / fname
        if path.exists():
            content += f"// --- {fname} ---\n{path.read_text()}\n\n"
        else:
            content += f"// --- {fname} (NOT FOUND) ---\n\n"

    return jsonify({"content": content, "target_model": target_model})


@app.get("/api/kernels/read_source")
def read_kernel_source():
    """Reads the content of a specific kernel file."""
    rel_path = request.args.get("path")
    if not rel_path:
        return jsonify({"error": "Missing 'path' parameter"}), 400

    try:
        safe_path = (KERNELS_DIR / rel_path).resolve()
        if not str(safe_path).startswith(str(KERNELS_DIR.resolve())):
            return (
                jsonify({"error": "Access denied: Path outside kernels directory"}),
                403,
            )

        if not safe_path.exists():
            return jsonify({"error": "File not found"}), 404

        content = safe_path.read_text()
        return jsonify({"content": content, "path": rel_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/reports")
def add_report():
    data = request.get_json(force=True, silent=True)
    if not data or not data.get("issue_description"):
        return jsonify({"error": "Missing 'issue_description'"}), 400

    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    save_report(data)
    return jsonify({"status": "success", "message": "Issue recorded"})


@app.get("/api/reports")
def get_reports_api():
    return jsonify({"reports": load_reports()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)
