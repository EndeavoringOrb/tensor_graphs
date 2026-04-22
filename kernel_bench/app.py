from flask import Flask, jsonify, request

from .jobs import create_job, jobs, load_benchmark_records, run_cmd, start_worker, TIMEOUTS

app = Flask(__name__)

# Start background worker thread
start_worker()


@app.route("/")
def index():
    return jsonify({
        "message": "Kernel Bench API",
        "endpoints": [
            "POST /api/kernels/test",
            "GET  /api/jobs/<job_id>",
            "GET  /api/benchmarks",
            "GET  /api/analyze",
        ],
    })


@app.post("/api/kernels/test")
def submit_kernel():
    data = request.get_json(force=True, silent=True)
    if not data or not data.get("source"):
        return jsonify({"error": "Missing 'source' field in request body"}), 400

    source = data["source"]
    backend = data.get("backend", "cpu")
    if backend not in ("cpu", "cuda"):
        return jsonify({"error": "backend must be 'cpu' or 'cuda'"}), 400

    job_id = create_job(source, backend)
    return jsonify({"job_id": job_id, "status": "queued"}), 202


@app.get("/api/jobs/<job_id>")
def get_job(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.get("/api/benchmarks")
def get_benchmarks():
    limit = request.args.get("limit", type=int)
    kernel_uid = request.args.get("kernel_uid")
    sort = request.args.get("sort")

    records = load_benchmark_records()

    if kernel_uid:
        records = [r for r in records if r.get("kernelUid") == kernel_uid]
    if sort == "runtime":
        records.sort(key=lambda r: r.get("runTime", 0), reverse=True)
    if limit is not None:
        records = records[:limit]

    return jsonify({"count": len(records), "records": records})


@app.get("/api/analyze")
def get_analyze():
    from .jobs import PROJECT_ROOT, CACHE_DIR, BENCHMARKS_DIR, parse_analyze_output

    graph = request.args.get("graph", str(CACHE_DIR / "gemma-3-270m-cpp.jsonl"))
    records = request.args.get("records", str(BENCHMARKS_DIR / "records.jsonl"))
    top_n = request.args.get("top_n", 20, type=int)
    chain_len = request.args.get("chain_len", 1, type=int)

    res = run_cmd(
        ["python", "analyze_performance.py",
         "--graph", graph,
         "--records", records,
         "--top_n", str(top_n),
         "--chain_len", str(chain_len)],
        TIMEOUTS["analyze"],
    )

    if res["exit_code"] != 0:
        return jsonify({
            "error": "analyze_performance.py failed",
            "exit_code": res["exit_code"],
            "stdout": res["stdout"],
            "stderr": res["stderr"],
        }), 500

    return jsonify(parse_analyze_output(res["stdout"]))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
