# File: kernel_bench/app.py
import json
import os
import re
import struct
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from .jobs import (
    BENCHMARKS_DIR,
    CACHE_DIR,
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


@app.route("/")
def index():
    return render_template("index.html")


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
                    "description": "Read the source code of any kernel file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Relative path under tensor_graphs_cpp/kernels (e.g. 'cuda/mul/F32_ND.cu' or 'cublas/dot_f32.cu')."}
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

    try:
        full_path = (KERNELS_DIR / rel_path).resolve()
        if not str(full_path).startswith(str(KERNELS_DIR.resolve())):
            full_path = (PROJECT_ROOT / "tensor_graphs_cpp" / rel_path).resolve()

        if not full_path.exists():
            return jsonify({"error": f"File '{rel_path}' not found"}), 404

        content = full_path.read_text(encoding="utf-8", errors="ignore")
        return jsonify({"content": content, "path": rel_path})
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
