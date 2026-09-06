# File: kernel_bench/app.py
import json
import os
import re
import struct
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

from .jobs import (
    BENCHMARKS_DIR,
    CACHE_DIR,
    CORE_DIR,
    GENERATED_DIR,
    KERNELS_DIR,
    LLAMA_CPP_TARGETS,
    PROJECT_ROOT,
    VERSIONS_DIR,
    BinaryReader,
    createJob,
    getAllVersions,
    getFallbackCause,
    getHwInfo,
    getNextVersionId,
    getVersionDetails,
    jobs,
    load_cache_file,
    loadJobHistory,
    loadReports,
    loadSuggestions,
    parseBenchModelMetrics,
    parseCacheAnalysis,
    saveReport,
    saveSuggestion,
    startWorker,
    updateSuggestionStatus,
)
from utils.common import format_op_name, load_uids_from_cpp, natural_sort_key

app = Flask(__name__)
startWorker()


def formatConstants(raw_bytes: bytes, dtype: int):
    if not raw_bytes:
        return ""
    dtypes = ["FLOAT32", "INT32", "INT64", "BF16", "BOOL", "ANY", "INT8", "E2M1_PACKED_INT8", "E2M1", "F8_E8M0", "F8_E4M3"]
    dt_str = dtypes[dtype] if isinstance(dtype, int) and dtype < len(dtypes) else str(dtype)
    if dt_str == "FLOAT32":
        count = len(raw_bytes) // 4
        return list(struct.unpack(f"<{count}f", raw_bytes))
    elif dt_str == "INT32":
        count = len(raw_bytes) // 4
        return list(struct.unpack(f"<{count}i", raw_bytes))
    return list(raw_bytes)


def buildAgentIndexData(target_model: str = "gemma-3-270m") -> dict:
    versions = getAllVersions()
    llama_targets = LLAMA_CPP_TARGETS.get(target_model, {})

    best_pp = 0.0
    best_tg = 0.0
    latest_version = versions[-1] if versions else None

    for v in versions:
        metrics = v.get("metrics", {})
        if "pp512" in metrics:
            best_pp = max(best_pp, metrics["pp512"].get("tps", 0.0))
        if "tg128" in metrics:
            best_tg = max(best_tg, metrics["tg128"].get("tps", 0.0))

    pp_target = llama_targets.get("pp512", {}).get("tps", 3439.99)
    tg_target = llama_targets.get("tg128", {}).get("tps", 69.16)
    target_beaten = (best_pp > pp_target) and (best_tg > tg_target)

    top_fallbacks = []
    top_ops = []
    if latest_version:
        top_fallbacks = latest_version.get("cpu_fallbacks", [])[:5]
        top_ops = latest_version.get("top_ops", [])[:5]

    hw_summary = "NVIDIA Quadro M4000 (1664 CUDA cores, 8GB VRAM) | Intel Xeon E5-1680 v3 (16 vCPUs)"

    system_prompt = (
        f"You are an elite CUDA and C++ high-performance optimization AI agent.\n"
        f"Target Model: {target_model}\n"
        f"Goal: Optimize Tensor Graphs kernels to beat llama.cpp for pp512 ({pp_target} t/s) and tg128 ({tg_target} t/s).\n"
        f"Current Best: pp512={best_pp:.2f} t/s ({round((best_pp / pp_target) * 100, 2) if pp_target > 0 else 0}%), "
        f"tg128={best_tg:.2f} t/s ({round((best_tg / tg_target) * 100, 2) if tg_target > 0 else 0}%).\n"
        f"Target Beaten: {target_beaten}.\n\n"
        "Operating Rules:\n"
        "1. NEVER modify or overwrite existing kernel files; only add new files in tensor_graphs_cpp/kernels/.\n"
        "2. Keep a running list of changes across versions/0, versions/1, versions/N using get_versions_history.\n"
        "3. Analyze bottlenecks via get_bottleneck_analysis. Eliminate CPU reference fallbacks (e.g. REF_MUL, REF_DIVIDE, REF_NEGATE) by providing CUDA kernels supporting non-contiguous/broadcast strides.\n"
        "4. Always register kernels using REGISTER_KERNEL or REGISTER_KERNEL_VIEW macros.\n"
        "5. Check get_system_status to monitor progress against llama.cpp targets.\n"
        "6. Iterate continuously until targets are beaten.\n"
    )

    initial_user_prompt = (
        f"Begin optimizing Tensor Graphs for {target_model}. Step 1: Call get_system_status and get_bottleneck_analysis to review current progress and identify CPU fallback bottlenecks (such as REF_MUL). Step 2: Formulate an optimization idea, author your CUDA kernel, and submit via submit_iteration."
    )

    autonomous_loop = [
        {
            "step": 1,
            "title": "Orientation & Benchmark Targets",
            "description": "Call get_system_status or get_agent_index to read the required throughput targets (pp512 and tg128) and current best achieved speeds.",
            "endpoint": "GET /api/status",
            "tool": "get_system_status",
            "example": "curl -s http://localhost:8080/api/status",
        },
        {
            "step": 2,
            "title": "Bottleneck & Fallback Analysis",
            "description": "Call get_bottleneck_analysis with filter='fallback' or sort_by='time' to locate the slowest operations. CPU reference fallbacks (REF_*) trigger expensive CPU-GPU copies and are the highest ROI optimization opportunities.",
            "endpoint": "GET /api/analyze?filter=fallback",
            "tool": "get_bottleneck_analysis",
            "example": "curl -s 'http://localhost:8080/api/analyze?filter=fallback'",
        },
        {
            "step": 3,
            "title": "Inspect Existing Kernels, Core Headers & Graph Definition",
            "description": "Review existing kernel implementations, core engine definitions (TensorNode, TensorView, KernelContext, REGISTER_KERNEL macros in types.hpp, kernels.hpp), and model computation graph to understand memory layouts, data types, and operation signatures.",
            "endpoint": "GET /api/kernels/list, GET /api/kernels/read_source, GET /api/core/list, GET /api/core/read_source, GET /api/model/source",
            "tool": "list_kernel_files, read_kernel_source, list_core_headers, read_core_header, read_model_source",
            "example": "curl -s 'http://localhost:8080/api/core/read_source?path=types.hpp'",
        },
        {
            "step": 4,
            "title": "Review Historical Iterations",
            "description": "Query get_versions_history and get_version_details to review what previous iterations attempted, preventing duplicate mistakes or repeating failed compiler flags.",
            "endpoint": "GET /api/versions, GET /api/versions/{version_id}",
            "tool": "get_versions_history, get_version_details",
            "example": "curl -s http://localhost:8080/api/versions",
        },
        {
            "step": 5,
            "title": "Author & Submit New Kernel",
            "description": "Write high-performance CUDA kernel code in a new unique file under tensor_graphs_cpp/kernels/. Register the kernel with REGISTER_KERNEL. Submit via submit_iteration to automatically build, test, and benchmark.",
            "endpoint": "POST /api/iteration/submit",
            "tool": "submit_iteration",
            "example": "curl -X POST http://localhost:8080/api/iteration/submit -H 'Content-Type: application/json' -d '{\"idea\": \"Strided Mul CUDA kernel\", \"filename\": \"kernels/cuda/mul/NC_F32_ND.cu\", \"source\": \"...\", \"backend\": \"cuda\"}'",
        },
        {
            "step": 6,
            "title": "Poll Job Status & Inspect Logs",
            "description": "Poll get_job_status until completed or failed. If build or test fails, inspect compiler errors or test assertions using getJobLogs.",
            "endpoint": "GET /api/jobs/{job_id}, GET /api/jobs/{job_id}/logs",
            "tool": "get_job_status",
            "example": "curl -s http://localhost:8080/api/jobs/{job_id}",
        },
        {
            "step": 7,
            "title": "Evaluate Progress & Iterate Autonomously",
            "description": "Check if target_beaten is true. If not, analyze the new cache and bottlenecks, formulate your next hypothesis, and submit your next kernel.",
            "endpoint": "GET /api/status",
            "tool": "get_system_status",
            "example": "curl -s http://localhost:8080/api/status",
        },
    ]

    rules = [
        {
            "rule": 1,
            "name": "NEVER Overwrite Existing Kernels",
            "description": "Existing kernel files in tensor_graphs_cpp/kernels/ are immutable baselines. You must always create a new file with a distinct path (e.g. kernels/cuda/mul/NC_F32_ND.cu).",
            "severity": "CRITICAL",
        },
        {
            "rule": 2,
            "name": "Eliminate CPU Reference Fallbacks First",
            "description": "CPU reference ops (REF_MUL, REF_DIVIDE, REF_NEGATE, etc.) cause massive CPU-GPU synchronization and data copies. Supporting non-contiguous and broadcast strides in CUDA kernels provides the highest speedups.",
            "severity": "HIGH_PRIORITY",
        },
        {
            "rule": 3,
            "name": "Register Every New Kernel",
            "description": "Every kernel must be registered with REGISTER_KERNEL or REGISTER_KERNEL_VIEW so the e-graph optimizer can select and dispatch it.",
            "severity": "CRITICAL",
        },
        {
            "rule": 4,
            "name": "Learn From Version History",
            "description": "Always check prior versions before writing kernels. Do not repeat failed hypotheses or re-introduce broken compile patterns.",
            "severity": "RECOMMENDED",
        },
        {
            "rule": 5,
            "name": "Iterate Autonomously Until Target Beaten",
            "description": "Do not stop after a single run. Continue optimizing until target_beaten is true.",
            "severity": "GOAL",
        },
        {
            "rule": 6,
            "name": "Report Harness Bugs & Suggest Improvements",
            "description": "If you encounter environment failures outside your control, call POST /api/reports. If you need new API endpoints or tooling to assist you, call POST /api/suggestions.",
            "severity": "INFO",
        },
    ]

    api_endpoints = [
        {"method": "GET", "path": "/agent", "description": "Agent Home Page & Index (HTML, Markdown, or JSON based on Accept header/format param)"},
        {"method": "GET", "path": "/api/agent", "description": "Operational index and guidance for agents (JSON)"},
        {"method": "GET", "path": "/api/status", "description": "Live system status, benchmark targets, and current best throughput"},
        {"method": "GET", "path": "/api/tools", "description": "Function calling schemas for LLM agent harnesses"},
        {"method": "GET", "path": "/api/openapi.json", "description": "Complete OpenAPI 3.0 specification"},
        {"method": "GET", "path": "/api/analyze", "description": "Performance cache analysis, CPU fallbacks, kernel runtimes, search and filter"},
        {"method": "GET", "path": "/api/versions", "description": "List all previous iteration versions and metrics"},
        {"method": "GET", "path": "/api/versions/{version_id}", "description": "Full logs (idea.md, build.log, bench_model.log, cache_analysis.log)"},
        {"method": "GET", "path": "/api/kernels/list", "description": "List all existing C++ and CUDA kernel source files"},
        {"method": "GET", "path": "/api/kernels/read_source", "description": "Read source code of any kernel file or core header (?path=...)"},
        {"method": "GET", "path": "/api/core/list", "description": "List all core C++ engine headers and types (e.g. types.hpp, kernels.hpp)"},
        {"method": "GET", "path": "/api/core/read_source", "description": "Read source code of core engine headers (?path=types.hpp)"},
        {"method": "GET", "path": "/api/model/source", "description": "Read C++ model graph definition (?target_model=...)"},
        {"method": "GET", "path": "/api/benchmarks/records", "description": "Query recorded benchmarks from records.bin (?op=...&shape=...)"},
        {"method": "GET", "path": "/api/benchmarks/calls", "description": "Query benchmark invocation calls from calls.bin"},
        {"method": "POST", "path": "/api/iteration/submit", "description": "Queue an optimization job (build -> test -> bench -> cache analysis)"},
        {"method": "GET", "path": "/api/jobs/{job_id}", "description": "Check job status and current pipeline step"},
        {"method": "GET", "path": "/api/jobs/{job_id}/logs", "description": "Inspect job build, test, and benchmark logs"},
        {"method": "GET", "path": "/api/history", "description": "List history of all queued/executed jobs"},
        {"method": "POST", "path": "/api/reports", "description": "Report harness or environment issues"},
        {"method": "GET", "path": "/api/reports", "description": "List reported issues"},
        {"method": "POST", "path": "/api/suggestions", "description": "Submit a suggestion for API improvements or harness tools"},
        {"method": "GET", "path": "/api/suggestions", "description": "List all suggestions (?category=...&status=...)"},
        {"method": "POST", "path": "/api/suggestions/{id}/resolve", "description": "Mark a suggestion as resolved"},
        {"method": "PATCH", "path": "/api/suggestions/{id}", "description": "Update suggestion status"}
    ]

    return {
        "title": "KernelBench Agent Home Page & Index",
        "subtitle": "Autonomous Optimization Entrypoint for Tensor Graphs",
        "version": "2.0.0",
        "target_model": target_model,
        "targets": llama_targets,
        "state": {
            "best_pp512_tps": best_pp,
            "best_pp512_pct": round((best_pp / pp_target) * 100, 2) if pp_target > 0 else 0,
            "best_tg128_tps": best_tg,
            "best_tg128_pct": round((best_tg / tg_target) * 100, 2) if tg_target > 0 else 0,
            "target_beaten": target_beaten,
            "total_versions": len(versions),
            "latest_version": latest_version,
            "next_version_id": getNextVersionId(),
            "hwinfo_summary": hw_summary,
        },
        "mission": {
            "objective": f"Surpass llama.cpp inference throughput targets for {target_model} (pp512: {pp_target} t/s, tg128: {tg_target} t/s).",
            "context": "Tensor Graphs performs e-graph rewriting and kernel generation for high-performance deep learning. The harness measures token throughput on prompt processing (pp512) and token generation (tg128).",
            "target_beaten": target_beaten,
        },
        "rules": rules,
        "autonomous_loop": autonomous_loop,
        "primary_bottlenecks": top_fallbacks,
        "top_operations": top_ops,
        "api_endpoints": api_endpoints,
        "system_prompt": system_prompt,
        "initial_user_prompt": initial_user_prompt,
        "links": {
            "home": "/",
            "agent_page": "/agent",
            "agent_api": "/api/agent",
            "status": "/api/status",
            "tools": "/api/tools",
            "openapi": "/api/openapi.json",
            "analyze": "/api/analyze",
            "versions": "/api/versions",
            "suggestions": "/api/suggestions",
        },
    }


def formatAgentIndexMarkdown(data: dict) -> str:
    target_model = data.get("target_model", "gemma-3-270m")
    state = data.get("state", {})
    targets = data.get("targets", {})
    pp_target = targets.get("pp512", {}).get("tps", 3439.99)
    tg_target = targets.get("tg128", {}).get("tps", 69.16)
    best_pp = state.get("best_pp512_tps", 0.0)
    best_tg = state.get("best_tg128_tps", 0.0)
    pp_pct = state.get("best_pp512_pct", 0.0)
    tg_pct = state.get("best_tg128_pct", 0.0)
    target_beaten = state.get("target_beaten", False)

    lines = [
        "# 🤖 KernelBench: Agent Home Page & Operational Guidance Index",
        "",
        f"> **Mission**: Surpass llama.cpp inference throughput targets for `{target_model}`.",
        f"> **Status**: {'🎉 TARGET BEATEN!' if target_beaten else '⚡ OPTIMIZATION IN PROGRESS'}",
        "",
        "---",
        "",
        "## 1. Live Targets & Current Best Metrics",
        "",
        "| Metric | llama.cpp Target | Current Best | % Achieved | Remaining Gap |",
        "| :--- | :--- | :--- | :--- | :--- |",
        f"| **Prompt Processing (pp512)** | `{pp_target} t/s` | `{best_pp:.2f} t/s` | `{pp_pct}%` | `{(pp_target - best_pp):.2f} t/s` |",
        f"| **Text Generation (tg128)** | `{tg_target} t/s` | `{best_tg:.2f} t/s` | `{tg_pct}%` | `{(tg_target - best_tg):.2f} t/s` |",
        "",
        f"- **Completed Iterations**: {state.get('total_versions', 0)}",
        f"- **Next Iteration ID**: Version {state.get('next_version_id', 0)}",
        f"- **Hardware**: {state.get('hwinfo_summary', 'N/A')}",
        "",
        "---",
        "",
        "## 2. Priority 1 Bottlenecks (Eliminate CPU Fallbacks)",
        "",
        "CPU reference kernels (`REF_*`) execute on the host CPU and cause catastrophic synchronization and memory transfer delays. Providing CUDA kernels that handle non-contiguous and broadcast strides will eliminate these bottlenecks.",
        "",
    ]

    fallbacks = data.get("primary_bottlenecks", [])
    if fallbacks:
        lines.append("| Operation | File Path | Time (ms) | % Total Time | Cause |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for fb in fallbacks:
            lines.append(f"| `{fb.get('op')}` | `{fb.get('path')}` | `{fb.get('time_ms')} ms` | `{fb.get('percentage')}%` | {fb.get('cause', 'Stride/broadcast mismatch')} |")
        lines.append("")
    else:
        lines.append("No active fallbacks detected in current profile.\n")

    lines.extend([
        "---",
        "",
        "## 3. Strict Operating Rules (The Guardrails)",
        "",
    ])

    for r in data.get("rules", []):
        lines.append(f"{r['rule']}. **{r['name']}** [{r.get('severity', 'RULE')}]")
        lines.append(f"   {r['description']}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 4. The 7-Step Autonomous Loop",
        "",
    ])

    for s in data.get("autonomous_loop", []):
        lines.append(f"### Step {s['step']}: {s['title']}")
        lines.append(f"{s['description']}")
        lines.append(f"- **Endpoint**: `{s['endpoint']}`")
        lines.append(f"- **Tool**: `{s['tool']}`")
        lines.append(f"```bash\n{s['example']}\n```")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 5. Complete API Catalog",
        "",
        "| Method | Path | Description |",
        "| :--- | :--- | :--- |",
    ])

    for ep in data.get("api_endpoints", []):
        lines.append(f"| `{ep['method']}` | `{ep['path']}` | {ep['description']} |")

    lines.extend([
        "",
        "---",
        "",
        "## 6. Recommended Agent System Prompt",
        "",
        "```text",
        data.get("system_prompt", ""),
        "```",
        "",
        "---",
        "",
        "## 7. Recommended Initial User Prompt",
        "",
        "```text",
        data.get("initial_user_prompt", ""),
        "```",
    ])

    return "\n".join(lines)


@app.route("/")
def index():
    accept = request.headers.get("Accept", "")
    fmt = request.args.get("format", "").lower()
    is_agent = request.args.get("agent", "").lower() in ("1", "true", "yes")

    if fmt == "json" or is_agent or ("application/json" in accept and "text/html" not in accept):
        target_model = request.args.get("target_model", "gemma-3-270m")
        return jsonify(buildAgentIndexData(target_model))
    if fmt in ("md", "markdown", "text") or ("text/markdown" in accept and "text/html" not in accept):
        target_model = request.args.get("target_model", "gemma-3-270m")
        return Response(formatAgentIndexMarkdown(buildAgentIndexData(target_model)), mimetype="text/markdown; charset=utf-8")

    return render_template("index.html")


@app.get("/agent")
@app.get("/agent/")
@app.get("/agent/index")
def getAgentIndex():
    target_model = request.args.get("target_model", "gemma-3-270m")
    fmt = request.args.get("format", "").lower()
    accept = request.headers.get("Accept", "")

    data = buildAgentIndexData(target_model)

    if fmt == "json" or ("application/json" in accept and "text/html" not in accept):
        return jsonify(data)
    if fmt in ("md", "markdown", "text") or ("text/markdown" in accept and "text/html" not in accept):
        return Response(formatAgentIndexMarkdown(data), mimetype="text/markdown; charset=utf-8")
    if fmt == "html" or "text/html" in accept:
        return render_template("agent.html", data=data, markdown_text=formatAgentIndexMarkdown(data))

    # Default to markdown response for CLI agents (e.g. curl with */*)
    return Response(formatAgentIndexMarkdown(data), mimetype="text/markdown; charset=utf-8")


@app.get("/api/agent")
@app.get("/api/agent/index")
@app.get("/api/home")
def getAgentIndexApi():
    target_model = request.args.get("target_model", "gemma-3-270m")
    fmt = request.args.get("format", "").lower()
    accept = request.headers.get("Accept", "")

    data = buildAgentIndexData(target_model)

    if fmt in ("md", "markdown", "text") or ("text/markdown" in accept and "application/json" not in accept):
        return Response(formatAgentIndexMarkdown(data), mimetype="text/markdown; charset=utf-8")

    return jsonify(data)



@app.get("/api/status")
def getSystemStatus():
    target_model = request.args.get("target_model", "gemma-3-270m")
    versions = getAllVersions()
    llama_targets = LLAMA_CPP_TARGETS.get(target_model, {})

    best_pp = 0.0
    best_tg = 0.0
    latest_version = versions[-1] if versions else None

    for v in versions:
        metrics = v.get("metrics", {})
        if "pp512" in metrics:
            best_pp = max(best_pp, metrics["pp512"].get("tps", 0.0))
        if "tg128" in metrics:
            best_tg = max(best_tg, metrics["tg128"].get("tps", 0.0))

    pp_target = llama_targets.get("pp512", {}).get("tps", 3439.99)
    tg_target = llama_targets.get("tg128", {}).get("tps", 69.16)

    target_beaten = (best_pp > pp_target) and (best_tg > tg_target)

    return jsonify({
        "target_model": target_model,
        "targets": llama_targets,
        "current_best": {
            "pp512_tps": best_pp,
            "tg128_tps": best_tg,
            "pp512_pct": round((best_pp / pp_target) * 100, 2) if pp_target > 0 else 0,
            "tg128_pct": round((best_tg / tg_target) * 100, 2) if tg_target > 0 else 0,
        },
        "target_beaten": target_beaten,
        "total_versions": len(versions),
        "latest_version": latest_version,
        "next_version_id": getNextVersionId(),
        "hwinfo": getHwInfo(),
    })


@app.get("/api/tools")
def getAgentTools():
    return jsonify({
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_agent_index",
                    "description": "Fetch the agent home page and comprehensive guidance, including mission objective, operating rules, target benchmarks, autonomous workflow instructions, and complete API directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_model": {"type": "string", "default": "gemma-3-270m", "description": "Target model name."}
                        }
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_system_status",
                    "description": "Get system hardware specs, llama.cpp target benchmarks, current best achieved metrics (pp512 and tg128 t/s), and latest version.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_model": {"type": "string", "default": "gemma-3-270m", "description": "Model to benchmark (e.g. gemma-3-270m)."}
                        }
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_bottleneck_analysis",
                    "description": "Analyze performance cache to retrieve total execution time, all operations, and CPU reference fallback bottlenecks with search and filtering.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_model": {"type": "string", "default": "gemma-3-270m", "description": "Target model name."},
                            "search": {"type": "string", "default": "", "description": "Search query for kernel name, path, or UID."},
                            "filter": {"type": "string", "enum": ["all", "fallback", "cuda", "cpu"], "default": "all", "description": "Filter by category."},
                            "sort_by": {"type": "string", "enum": ["time", "count", "name", "percentage"], "default": "time", "description": "Field to sort by."},
                            "limit": {"type": "integer", "default": 0, "description": "Limit number of returned kernels (0 for all)."}
                        }
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_versions_history",
                    "description": "List all previous iteration versions, their ideas, metrics (pp512, tg128), and status to avoid repeated mistakes.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_version_details",
                    "description": "Get detailed logs (idea.md, build.log, bench_model.log, cache_analysis.log, README.md) for a specific version.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "version_id": {"type": "integer", "description": "Version index (e.g. 0, 1, 2)."}
                        },
                        "required": ["version_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_benchmark_records",
                    "description": "Query recorded kernel benchmarks from benchmarks/records.bin filtered by op name or tensor shapes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string", "description": "Regex pattern to filter operation name (e.g. 'CuBLAS' or 'Dot')."},
                            "shape": {"type": "string", "description": "Regex pattern to filter shapes (e.g. '512, 640')."}
                        }
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_kernel_files",
                    "description": "List all existing C++ and CUDA kernel files to see what is already implemented.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_kernel_source",
                    "description": "Read the source code of any kernel file (e.g. 'cuda/mul/F32_ND.cu') or core header (e.g. 'core/types.hpp').",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Relative path under tensor_graphs_cpp/kernels (e.g. 'cuda/mul/F32_ND.cu') or core header (e.g. 'core/types.hpp')."}
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_core_headers",
                    "description": "List all engine core header and source files under tensor_graphs_cpp/core (e.g. types.hpp, kernels.hpp, graph.hpp, ops/*.hpp).",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_core_header",
                    "description": "Read engine core header or source file from tensor_graphs_cpp/core (e.g. types.hpp, kernels.hpp, graph.hpp, ops/add.hpp).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path under tensor_graphs_cpp/core (e.g. 'types.hpp', 'kernels.hpp', 'ops/add.hpp')."
                            }
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_model_source",
                    "description": "Read the C++ model graph definition (e.g. tensor_graphs_cpp/models/gemma-3-270m.hpp).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_model": {"type": "string", "default": "gemma-3-270m", "description": "Model name."}
                        }
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "submit_iteration",
                    "description": "Submit an iterative improvement. Writes kernel code, runs build -> clear caches -> populate calls -> bench -> final bench_model -> cache analysis, logs everything to versions/N, and returns the resulting metrics.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "idea": {"type": "string", "description": "Clear explanation of the idea/hypothesis being tested."},
                            "source": {"type": "string", "description": "C++ or CUDA source code for the new kernel."},
                            "filename": {"type": "string", "description": "Relative path for the new kernel (e.g. 'kernels/cublas/gemm_f32.cu' or 'kernels/cuda/mul/NC_F32_ND.cu'). Must NOT overwrite existing files."},
                            "backend": {"type": "string", "enum": ["cuda", "cpu"], "default": "cuda", "description": "Target backend."},
                            "target_model": {"type": "string", "default": "gemma-3-270m", "description": "Target model."},
                            "pp": {"type": "integer", "default": 512, "description": "Prompt processing sequence length."},
                            "tg": {"type": "integer", "default": 128, "description": "Text generation target token position."},
                            "min_compile_time": {"type": "number", "default": 90.0, "description": "Search compile time budget in seconds (max 90.0)."},
                            "kernel_name": {"type": "string", "description": "Optional registered opName (e.g. 'CuBLAS_Dot_F32') to run fused kernel testing on shapes in calls.bin. Automatically extracted from source if omitted."}
                        },
                        "required": ["idea"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_job_status",
                    "description": "Poll the progress and status of a running or completed optimization job.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "job_id": {"type": "string", "description": "Job ID returned by submit_iteration."}
                        },
                        "required": ["job_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "report_issue",
                    "description": "Report an environment or harness error outside the agent's control.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "issue_description": {"type": "string", "description": "Detailed explanation of the issue."}
                        },
                        "required": ["issue_description"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "suggest_api_change",
                    "description": "Suggest an addition, improvement, or change to the KernelBench REST API or harness. Use this whenever you identify a missing endpoint, schema gap, or enhancement that would facilitate agentic optimization.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Concise summary of the proposed change."},
                            "description": {"type": "string", "description": "Detailed explanation of the motivation and rationale for the change."},
                            "suggested_endpoint": {"type": "string", "description": "Optional HTTP method and path (e.g. 'POST /api/batch_bench')."},
                            "proposed_changes": {"type": "string", "description": "Optional details regarding request/response schemas or implementation notes."},
                            "category": {"type": "string", "enum": ["new_endpoint", "schema_change", "performance", "developer_experience", "other"], "default": "new_endpoint", "description": "Category of the suggestion."}
                        },
                        "required": ["title", "description"],
                    },
                },
            },
        ]
    })


@app.get("/api/openapi.json")
def getOpenApiSpec():
    return jsonify({
        "openapi": "3.0.0",
        "info": {
            "title": "KernelBench Iterative Performance Optimization API",
            "version": "2.0.0",
            "description": "API for autonomous agentic harnesses to optimize tensor graph operations and surpass llama.cpp on LLM benchmarks.",
        },
        "servers": [{"url": "http://localhost:8080"}],
        "paths": {
            "/agent": {
                "get": {
                    "summary": "Agent home page and operational index (HTML, Markdown, or JSON)",
                    "parameters": [
                        {"name": "target_model", "in": "query", "schema": {"type": "string", "default": "gemma-3-270m"}},
                        {"name": "format", "in": "query", "schema": {"type": "string", "enum": ["html", "json", "markdown"]}}
                    ],
                    "responses": {"200": {"description": "Agent index and operational guidance"}}
                }
            },
            "/api/agent": {
                "get": {
                    "summary": "Agent operational index and guidance in JSON format",
                    "parameters": [
                        {"name": "target_model", "in": "query", "schema": {"type": "string", "default": "gemma-3-270m"}}
                    ],
                    "responses": {"200": {"description": "Structured JSON agent guidance"}}
                }
            },
            "/api/status": {
                "get": {
                    "summary": "System status and target benchmarks",
                    "responses": {"200": {"description": "Current status, targets, and latest version"}}
                }
            },
            "/api/tools": {
                "get": {
                    "summary": "Function calling tool schemas for agents",
                    "responses": {"200": {"description": "List of available agent tools"}}
                }
            },
            "/api/versions": {
                "get": {
                    "summary": "List all iteration versions",
                    "responses": {"200": {"description": "Array of version objects with metrics"}}
                }
            },
            "/api/versions/{version_id}": {
                "get": {
                    "summary": "Get version details and logs",
                    "parameters": [{"name": "version_id", "in": "path", "required": True, "schema": {"type": "integer"}}],
                    "responses": {"200": {"description": "Full logs and metrics for version"}}
                }
            },
            "/api/analyze": {
                "get": {
                    "summary": "Bottleneck and cache analysis with searching and filtering",
                    "parameters": [
                        {"name": "target_model", "in": "query", "schema": {"type": "string", "default": "gemma-3-270m"}},
                        {"name": "pp", "in": "query", "schema": {"type": "string", "default": "512"}},
                        {"name": "tg", "in": "query", "schema": {"type": "string", "default": "128"}},
                        {"name": "search", "in": "query", "schema": {"type": "string"}},
                        {"name": "filter", "in": "query", "schema": {"type": "string", "enum": ["all", "fallback", "cuda", "cpu"]}},
                        {"name": "sort_by", "in": "query", "schema": {"type": "string", "enum": ["time", "count", "name", "percentage"]}},
                        {"name": "sort_order", "in": "query", "schema": {"type": "string", "enum": ["desc", "asc"]}},
                        {"name": "limit", "in": "query", "schema": {"type": "integer"}}
                    ],
                    "responses": {"200": {"description": "Kernel operations, bottlenecks, and CPU fallbacks"}}
                }
            },
            "/api/core/list": {
                "get": {
                    "summary": "List all core engine header and source files",
                    "responses": {"200": {"description": "List of core files"}}
                }
            },
            "/api/core/read_source": {
                "get": {
                    "summary": "Read source code of a core engine header",
                    "parameters": [
                        {"name": "path", "in": "query", "required": True, "schema": {"type": "string"}}
                    ],
                    "responses": {
                        "200": {"description": "Core header source code and path"},
                        "400": {"description": "Missing path parameter"},
                        "404": {"description": "File not found"}
                    }
                }
            },
            "/api/kernels/list": {
                "get": {
                    "summary": "List all existing C++ and CUDA kernel files",
                    "responses": {"200": {"description": "List of kernel files"}}
                }
            },
            "/api/kernels/read_source": {
                "get": {
                    "summary": "Read source code of a kernel file or core header",
                    "parameters": [
                        {"name": "path", "in": "query", "required": True, "schema": {"type": "string"}}
                    ],
                    "responses": {
                        "200": {"description": "Kernel source code and path"},
                        "400": {"description": "Missing path parameter"},
                        "404": {"description": "File not found"}
                    }
                }
            },
            "/api/model/source": {
                "get": {
                    "summary": "Read C++ model graph definition",
                    "parameters": [
                        {"name": "target_model", "in": "query", "schema": {"type": "string", "default": "gemma-3-270m"}}
                    ],
                    "responses": {
                        "200": {"description": "Model source code and path"},
                        "404": {"description": "Model not found"}
                    }
                }
            },
            "/api/kernels/test": {
                "post": {
                    "summary": "Submit a kernel or run an iteration",
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {"type": "object"}}}
                    },
                    "responses": {"202": {"description": "Job queued"}}
                }
            },
            "/api/jobs/{job_id}": {
                "get": {
                    "summary": "Check job status",
                    "parameters": [{"name": "job_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                    "responses": {"200": {"description": "Job status and metrics"}}
                }
            },
            "/api/suggestions": {
                "get": {
                    "summary": "List all agent-submitted API suggestions",
                    "parameters": [
                        {"name": "category", "in": "query", "schema": {"type": "string"}},
                        {"name": "status", "in": "query", "schema": {"type": "string"}}
                    ],
                    "responses": {"200": {"description": "List of suggestions"}}
                },
                "post": {
                    "summary": "Submit a suggestion for API changes or improvements",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["title", "description"],
                                    "properties": {
                                        "title": {"type": "string"},
                                        "description": {"type": "string"},
                                        "suggested_endpoint": {"type": "string"},
                                        "proposed_changes": {"type": "string"},
                                        "category": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "201": {"description": "Suggestion created successfully"},
                        "400": {"description": "Missing required fields"}
                    }
                }
            },
            "/api/suggestions/{suggestion_id}": {
                "patch": {
                    "summary": "Update suggestion status",
                    "parameters": [{"name": "suggestion_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                    "requestBody": {
                        "content": {"application/json": {"schema": {"type": "object", "properties": {"status": {"type": "string"}}}}}
                    },
                    "responses": {"200": {"description": "Suggestion updated"}}
                }
            },
            "/api/suggestions/{suggestion_id}/resolve": {
                "post": {
                    "summary": "Mark a suggestion as resolved",
                    "parameters": [{"name": "suggestion_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                    "responses": {"200": {"description": "Suggestion resolved"}}
                }
            }
        }
    })


@app.get("/api/versions")
def getVersionsApi():
    return jsonify({"versions": getAllVersions()})


@app.get("/api/versions/<int:v_id>")
def getVersionDetailsApi(v_id: int):
    details = getVersionDetails(v_id)
    if "error" in details:
        return jsonify(details), 404
    return jsonify(details)


def filterAndSortKernels(
    kernels: list,
    search: str = "",
    filter_type: str = "all",
    sort_by: str = "time",
    sort_order: str = "desc",
    limit: int = 0,
) -> list:
    search = search.strip().lower()
    filter_type = filter_type.strip().lower()
    sort_by = sort_by.strip().lower()
    sort_order = sort_order.strip().lower()

    filtered = []
    for k in kernels:
        is_fallback = k.get("is_fallback", False)
        category = k.get("category", "")

        if filter_type in ("fallback", "cpu_fallback", "reference", "ref"):
            if not is_fallback:
                continue
        elif filter_type in ("cuda", "gpu"):
            if category != "cuda":
                continue
        elif filter_type == "cpu":
            if category not in ("cpu", "fallback"):
                continue
        elif filter_type in ("general", "optimized_cpu"):
            if is_fallback or category != "cpu":
                continue

        if search:
            searchable = (
                f"{k.get('name', '')} {k.get('op', '')} {k.get('path', '')} "
                f"{k.get('kernel_id', '')} {k.get('device', '')} {k.get('cause', '')}"
            ).lower()
            if search not in searchable:
                continue

        filtered.append(k)

    reverse = (sort_order != "asc")
    if sort_by in ("time", "time_ms"):
        sorted_list = sorted(filtered, key=lambda x: x.get("time_ms", 0.0), reverse=reverse)
    elif sort_by in ("count", "invocations"):
        sorted_list = sorted(filtered, key=lambda x: x.get("count", 0), reverse=reverse)
    elif sort_by in ("name", "op"):
        sorted_list = sorted(
            filtered,
            key=lambda x: natural_sort_key(x.get("name", "")),
            reverse=(sort_order == "desc"),
        )
    elif sort_by in ("percentage", "pct"):
        sorted_list = sorted(filtered, key=lambda x: x.get("percentage", 0.0), reverse=reverse)
    else:
        sorted_list = sorted(filtered, key=lambda x: x.get("time_ms", 0.0), reverse=reverse)

    if limit > 0:
        sorted_list = sorted_list[:limit]

    return sorted_list


@app.get("/api/analyze")
def getAnalyze():
    target_model = request.args.get("target_model", "gemma-3-270m")
    pp = request.args.get("pp", "512")
    tg = request.args.get("tg", "128")
    search_query = request.args.get("search", request.args.get("q", request.args.get("query", ""))).strip()
    filter_type = request.args.get("filter", request.args.get("type", "all")).strip()
    sort_by = request.args.get("sort_by", "time").strip()
    sort_order = request.args.get("sort_order", request.args.get("order", "desc")).strip()
    try:
        limit = int(request.args.get("limit", 0))
    except (ValueError, TypeError):
        limit = 0

    cache_path = CACHE_DIR / f"bench_{target_model}-pp{pp}-tg{tg}.bin"
    if not cache_path.exists():
        candidates = list(CACHE_DIR.glob(f"*{target_model}*.bin"))
        if candidates:
            cache_path = candidates[0]

    if not cache_path.exists():
        versions = getAllVersions()
        if versions:
            latest_v = versions[-1]["version"]
            v_details = getVersionDetails(latest_v)
            if v_details.get("cache_analysis"):
                analysis_val = v_details["cache_analysis"]
                parsed = parseCacheAnalysis(analysis_val) if isinstance(analysis_val, str) else dict(analysis_val)
                filtered = filterAndSortKernels(
                    parsed.get("kernels", []),
                    search=search_query,
                    filter_type=filter_type,
                    sort_by=sort_by,
                    sort_order=sort_order,
                    limit=limit,
                )
                parsed["kernels"] = filtered
                parsed["source"] = f"versions/{latest_v}/cache_analysis.log"
                parsed["cache_analysis"] = v_details["cache_analysis"]
                parsed["filters_applied"] = {
                    "search": search_query,
                    "filter": filter_type,
                    "sort_by": sort_by,
                    "sort_order": sort_order,
                    "limit": limit,
                }
                return jsonify(parsed)
        return jsonify({"error": "No cache file found. Run bench_model first."}), 404

    try:
        cache_entries = load_cache_file(cache_path)
    except Exception as e:
        return jsonify({"error": f"Failed loading cache file: {e}"}), 500

    uid_map = load_uids_from_cpp()
    kernel_stats = defaultdict(lambda: {
        "kernel_id": "",
        "name": "",
        "path": "",
        "time_ms": 0.0,
        "count": 0,
        "is_fallback": False,
        "device": "",
        "category": "",
        "cause": "",
    })
    total_time = 0.0
    chain_stats = defaultdict(lambda: {"time": 0.0, "count": 0})

    for entry in cache_entries:
        if entry.get("type") == "compiled_bucket":
            graph = entry["graph"]
            node_views = graph.get("nodeViews", {})
            node_costs = graph.get("nodeCosts", {})
            for inst in graph.get("instructions", []):
                eclass_id = inst.get("eclassId")
                uid = inst.get("kernelId", 0)

                runtime = node_costs.get(eclass_id, 0.0)
                if runtime == float("inf"):
                    runtime = 0.0

                total_time += runtime

                info = (
                    uid_map.get(uid)
                    or uid_map.get(str(uid))
                    or uid_map.get(hex(uid).lower())
                ) if uid else None

                if info and isinstance(info, dict):
                    kernel_name = info.get("name") or f"Kernel_{hex(uid)}"
                    kernel_path = info.get("path", "")
                    hex_uid = info.get("hex_uid", hex(uid))
                else:
                    kernel_name = inst.get("kernelName") or (f"Kernel_{hex(uid)}" if uid else "unknown")
                    kernel_path = ""
                    hex_uid = hex(uid) if uid else "0x0"

                is_fallback = "REF_" in kernel_name or "reference" in kernel_path.lower()
                if is_fallback:
                    device = "CPU (Reference)"
                    category = "fallback"
                elif (
                    "cuda" in kernel_path.lower()
                    or "cublas" in kernel_path.lower()
                    or kernel_path.endswith(".cu")
                    or "CUDA" in kernel_name
                    or "CuBLAS" in kernel_name
                ):
                    device = "CUDA"
                    category = "cuda"
                else:
                    device = "CPU"
                    category = "cpu"

                k_entry = kernel_stats[hex_uid]
                k_entry["kernel_id"] = hex_uid
                k_entry["name"] = kernel_name
                k_entry["path"] = kernel_path
                k_entry["time_ms"] += runtime
                k_entry["count"] += 1
                k_entry["is_fallback"] = is_fallback
                k_entry["device"] = device
                k_entry["category"] = category
                if is_fallback:
                    k_entry["cause"] = getFallbackCause(kernel_name, kernel_path)

                in_shapes = [node_views[cid]["shape"] for cid in inst.get("children", []) if cid in node_views]
                out_shape = node_views[eclass_id]["shape"] if eclass_id in node_views else []
                debug_origin = inst.get("debugOrigin", "")
                ident = f"{kernel_name}({in_shapes}->{out_shape}) [{debug_origin}]"
                chain_stats[ident]["time"] += runtime
                chain_stats[ident]["count"] += 1

    name_counts = defaultdict(int)
    for k_entry in kernel_stats.values():
        name_counts[k_entry["name"]] += 1

    all_kernels = []
    for k_entry in kernel_stats.values():
        if name_counts[k_entry["name"]] > 1 and k_entry["path"]:
            k_entry["op"] = f"{k_entry['name']} ({Path(k_entry['path']).stem})"
        else:
            k_entry["op"] = k_entry["name"]
        k_entry["percentage"] = round((k_entry["time_ms"] / total_time * 100), 2) if total_time > 0 else 0.0
        k_entry["time_ms"] = round(k_entry["time_ms"], 4)
        all_kernels.append(k_entry)

    all_kernels.sort(key=lambda x: x["time_ms"], reverse=True)

    fallbacks_all = [k for k in all_kernels if k["is_fallback"]]
    cuda_all = [k for k in all_kernels if k["category"] == "cuda"]
    cpu_all = [k for k in all_kernels if k["category"] == "cpu"]

    fallback_time = sum(k["time_ms"] for k in fallbacks_all)
    cuda_time = sum(k["time_ms"] for k in cuda_all)
    cpu_time = sum(k["time_ms"] for k in cpu_all)

    summary = {
        "total_time_ms": round(total_time, 4),
        "total_kernels": len(all_kernels),
        "fallback_count": len(fallbacks_all),
        "fallback_time_ms": round(fallback_time, 4),
        "fallback_percentage": round(fallback_time / total_time * 100, 2) if total_time > 0 else 0.0,
        "cuda_count": len(cuda_all),
        "cuda_time_ms": round(cuda_time, 4),
        "cuda_percentage": round(cuda_time / total_time * 100, 2) if total_time > 0 else 0.0,
        "cpu_general_count": len(cpu_all),
        "cpu_general_time_ms": round(cpu_time, 4),
        "cpu_general_percentage": round(cpu_time / total_time * 100, 2) if total_time > 0 else 0.0,
    }

    filtered_kernels = filterAndSortKernels(
        all_kernels,
        search=search_query,
        filter_type=filter_type,
        sort_by=sort_by,
        sort_order=sort_order,
        limit=limit,
    )
    summary["matching_kernels"] = len(filtered_kernels)

    sorted_chains = sorted(
        [{"chain": k, "time_ms": v["time"], "count": v["count"]} for k, v in chain_stats.items()],
        key=lambda x: x["time_ms"],
        reverse=True,
    )[:20]

    return jsonify({
        "cache_file": str(cache_path.relative_to(PROJECT_ROOT)),
        "total_estimated_time_ms": round(total_time, 4),
        "top_ops": all_kernels[:20],
        "kernels": filtered_kernels,
        "all_kernels": all_kernels,
        "cpu_fallbacks": fallbacks_all,
        "top_chains": sorted_chains,
        "summary": summary,
        "filters_applied": {
            "search": search_query,
            "filter": filter_type,
            "sort_by": sort_by,
            "sort_order": sort_order,
            "limit": limit,
        },
    })


@app.get("/api/benchmarks/records")
@app.get("/api/read_benchmarks")
def getBenchmarkRecords():
    op_filter = request.args.get("op", "")
    shape_filter = request.args.get("shape", "")

    records_path = BENCHMARKS_DIR / "records.bin"
    if not records_path.exists():
        return jsonify({"records": []})

    header_path = GENERATED_DIR / "kernel_uids.gen.hpp"
    uid_map = {}
    if header_path.exists():
        pattern = re.compile(r"constexpr uint64_t\s+(\w+)\s+=\s+(0x[0-9a-fA-F]+)ULL;")
        with open(header_path, "r", encoding="utf-8") as f:
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

            uid = str(r.get("kernelId", ""))
            opname = uid_map.get(uid, r.get("opName", "UNKNOWN"))
            r["opName"] = opname
            r["kernelId"] = hex(r["kernelId"])

            shapes = str(r.get("outputShapes", [])) + str(r.get("inputShapes", []))
            if op_filter and not re.search(op_filter, opname, re.IGNORECASE) and not re.search(op_filter, uid, re.IGNORECASE):
                continue
            if shape_filter and not re.search(shape_filter, shapes):
                continue

            in_consts = r.get("inputConstants", [])
            in_dtypes = r.get("inputDTypes", [])
            formatted_consts = []
            if in_consts and in_dtypes:
                for idx, data in enumerate(in_consts):
                    dt = in_dtypes[idx] if idx < len(in_dtypes) else -1
                    formatted_consts.append(formatConstants(data, dt))
            r["inputConstants"] = formatted_consts

            records.append(r)

    return jsonify({"records": records, "total_records": len(records)})


@app.get("/api/benchmarks/calls")
def getBenchmarkCalls():
    calls_path = BENCHMARKS_DIR / "calls.bin"
    if not calls_path.exists():
        return jsonify({"calls": []})

    calls = []
    with open(calls_path, "rb") as f:
        br = BinaryReader(f)
        while True:
            r = br.read_record()
            if r is None:
                break

            in_consts = r.get("inputConstants", [])
            in_dtypes = r.get("inputDTypes", [])
            formatted_consts = []
            if in_consts and in_dtypes:
                for idx, data in enumerate(in_consts):
                    dt = in_dtypes[idx] if idx < len(in_dtypes) else -1
                    formatted_consts.append(formatConstants(data, dt))
            r["inputConstants"] = formatted_consts

            calls.append(r)

    return jsonify({"calls": calls, "total_calls": len(calls)})


@app.get("/api/core/list")
@app.get("/api/core/headers")
def listCoreFiles():
    try:
        files = []
        for path in CORE_DIR.rglob("*"):
            if path.is_file() and path.suffix in (".hpp", ".h", ".cpp", ".cu", ".cuh"):
                files.append(str(path.relative_to(CORE_DIR)))
        return jsonify({"files": sorted(files)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/core/read_source")
@app.get("/api/core/source")
def readCoreSource():
    rel_path = request.args.get("path")
    if not rel_path:
        return jsonify({"error": "Missing 'path' parameter"}), 400

    cpp_dir = (PROJECT_ROOT / "tensor_graphs_cpp").resolve()
    core_dir_resolved = CORE_DIR.resolve()
    clean_path = rel_path.strip().lstrip("/")
    if clean_path.startswith("tensor_graphs_cpp/core/"):
        clean_path = clean_path[len("tensor_graphs_cpp/core/"):]
    elif clean_path.startswith("tensor_graphs_cpp/"):
        clean_path = clean_path[len("tensor_graphs_cpp/"):]
    elif clean_path.startswith("core/"):
        clean_path = clean_path[len("core/"):]

    candidates = [
        (CORE_DIR / clean_path).resolve(),
        (cpp_dir / "core" / clean_path).resolve(),
    ]

    target_path = None
    for cand in candidates:
        if cand.is_file() and str(cand).startswith(str(core_dir_resolved)):
            target_path = cand
            break

    if not target_path:
        return jsonify({"error": f"Core file '{rel_path}' not found"}), 404

    try:
        content = target_path.read_text(encoding="utf-8", errors="ignore")
        return jsonify({
            "content": content,
            "path": rel_path,
            "resolved_path": str(target_path.relative_to(cpp_dir)),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/kernels/list")
def listKernelFiles():
    try:
        files = []
        for path in KERNELS_DIR.rglob("*"):
            if path.is_file() and path.suffix in (".hpp", ".cu"):
                files.append(str(path.relative_to(KERNELS_DIR)))
        return jsonify({"files": sorted(files)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/kernels/read_source")
def readKernelSource():
    rel_path = request.args.get("path")
    if not rel_path:
        return jsonify({"error": "Missing 'path' parameter"}), 400

    cpp_dir = (PROJECT_ROOT / "tensor_graphs_cpp").resolve()
    clean_path = rel_path.strip().lstrip("/")

    if clean_path.startswith("tensor_graphs_cpp/"):
        clean_path = clean_path[len("tensor_graphs_cpp/"):]

    candidates = []
    if clean_path.startswith("core/"):
        candidates.append((CORE_DIR / clean_path[5:]).resolve())
        candidates.append((cpp_dir / clean_path).resolve())
    elif clean_path.startswith("kernels/"):
        candidates.append((KERNELS_DIR / clean_path[8:]).resolve())
        candidates.append((cpp_dir / clean_path).resolve())
    else:
        candidates.append((KERNELS_DIR / clean_path).resolve())
        candidates.append((CORE_DIR / clean_path).resolve())
        candidates.append((cpp_dir / clean_path).resolve())

    target_path = None
    for cand in candidates:
        if cand.is_file() and str(cand).startswith(str(cpp_dir)):
            target_path = cand
            break

    if not target_path:
        return jsonify({"error": f"File '{rel_path}' not found"}), 404

    try:
        content = target_path.read_text(encoding="utf-8", errors="ignore")
        return jsonify({
            "content": content,
            "path": rel_path,
            "resolved_path": str(target_path.relative_to(cpp_dir)),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/kernels/read_model")
@app.get("/api/model/source")
def readModelSource():
    target_model = request.args.get("target_model", "gemma-3-270m")
    fname = f"{target_model}.hpp"
    model_path = PROJECT_ROOT / "tensor_graphs_cpp" / "models" / fname
    if not model_path.exists():
        return jsonify({"error": f"Model file {fname} not found"}), 404
    content = model_path.read_text(encoding="utf-8", errors="ignore")
    return jsonify({"content": content, "target_model": target_model, "path": str(model_path.relative_to(PROJECT_ROOT))})


@app.post("/api/kernels/test")
@app.post("/api/iteration/submit")
def submitIteration():
    data = request.get_json(force=True, silent=True) or {}
    idea = data.get("idea", "")
    source = data.get("source", "")
    opname = data.get("opname", "")
    backend = data.get("backend", "cuda")
    target_model = data.get("target_model", "gemma-3-270m")
    filename = data.get("filename", "")
    pp = int(data.get("pp", 512))
    tg = int(data.get("tg", 128))
    min_compile_time = float(data.get("min_compile_time", 90.0))
    version = data.get("version")

    if not idea and not source and not opname:
        return jsonify({"error": "Provide at least 'idea' or 'source'"}), 400

    kernel_name = data.get("kernel_name", "")

    job_id = createJob(
        source=source,
        opname=opname,
        backend=backend,
        target_model=target_model,
        idea=idea,
        filename=filename,
        pp=pp,
        tg=tg,
        min_compile_time=min_compile_time,
        version=version,
        kernel_name=kernel_name,
    )
    job = jobs[job_id]
    return jsonify({
        "job_id": job_id,
        "version": job["version"],
        "status": "queued",
        "message": f"Queued iteration job {job_id} for Version {job['version']}",
    }), 202


@app.get("/api/jobs/<job_id>")
def getJob(job_id: str):
    job = jobs.get(job_id)
    if not job:
        history = loadJobHistory()
        job = next((j for j in history if j.get("job_id") == job_id), None)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.get("/api/jobs/<job_id>/logs")
def getJobLogs(job_id: str):
    job = jobs.get(job_id)
    if not job:
        history = loadJobHistory()
        job = next((j for j in history if j.get("job_id") == job_id), None)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    v_id = job.get("version")
    if v_id is None:
        return jsonify({"logs": "No version assigned yet."})

    v_dir = VERSIONS_DIR / str(v_id)
    logs = {}
    for fname in ["build.log", "test.log", "bench_model.log", "cache_analysis.log", "README.md"]:
        f = v_dir / fname
        logs[fname] = f.read_text(encoding="utf-8", errors="ignore") if f.exists() else None

    return jsonify({"version": v_id, "logs": logs})


@app.get("/api/history")
def getHistory():
    history = loadJobHistory()
    return jsonify({"history": history, "count": len(history)})


@app.post("/api/reports")
def addReport():
    data = request.get_json(force=True, silent=True)
    if not data or not data.get("issue_description"):
        return jsonify({"error": "Missing 'issue_description'"}), 400

    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    saveReport(data)
    return jsonify({"status": "success", "message": "Issue recorded"})


@app.get("/api/reports")
def getReportsApi():
    return jsonify({"reports": loadReports()})


@app.post("/api/suggestions")
def addSuggestion():
    data = request.get_json(force=True, silent=True)
    if not data or not data.get("title") or not data.get("description"):
        return jsonify({"error": "Missing required fields: 'title' and 'description'"}), 400

    suggestion_id = saveSuggestion(data)
    return jsonify({
        "status": "success",
        "suggestion_id": suggestion_id,
        "message": "Suggestion recorded successfully"
    }), 201


@app.get("/api/suggestions")
def getSuggestionsApi():
    category = request.args.get("category")
    status = request.args.get("status")
    suggestions = loadSuggestions()
    if category:
        suggestions = [s for s in suggestions if s.get("category") == category]
    if status:
        suggestions = [s for s in suggestions if s.get("status") == status]
    return jsonify({"suggestions": suggestions, "count": len(suggestions)})


@app.patch("/api/suggestions/<suggestion_id>")
def patchSuggestion(suggestion_id: str):
    data = request.get_json(force=True, silent=True) or {}
    new_status = data.get("status", "resolved")
    success = updateSuggestionStatus(suggestion_id, new_status)
    if not success:
        return jsonify({"error": f"Suggestion '{suggestion_id}' not found"}), 404
    return jsonify({
        "status": "success",
        "suggestion_id": suggestion_id,
        "new_status": new_status,
        "message": f"Suggestion {suggestion_id} updated to {new_status}"
    })


@app.post("/api/suggestions/<suggestion_id>/resolve")
def resolveSuggestionApi(suggestion_id: str):
    success = updateSuggestionStatus(suggestion_id, "resolved")
    if not success:
        return jsonify({"error": f"Suggestion '{suggestion_id}' not found"}), 404
    return jsonify({
        "status": "success",
        "suggestion_id": suggestion_id,
        "new_status": "resolved",
        "message": f"Suggestion {suggestion_id} marked as resolved"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)
