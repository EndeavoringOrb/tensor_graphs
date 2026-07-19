import os
from pathlib import Path
from flask import Flask, jsonify, request, render_template

from utils.binary import load_cache_file

app = Flask(__name__)

# Global variable to cache the currently parsed session metadata
CURRENT_SESSION = None


def load_cache_file_lazy(path):
    """Loads cache file lazily using the unified helper in the utils module."""
    entries = load_cache_file(path, lazy=True, string_enums=True)

    session = {"metadata": None, "buckets": [], "file_size": os.path.getsize(path)}

    for entry in entries:
        if entry["type"] == "metadata":
            session["metadata"] = {
                "cacheVersion": entry["cacheVersion"],
                "rootId": entry["rootId"],
                "selectedCachedNodes": entry["selectedCachedNodes"],
            }
        elif entry["type"] == "compiled_bucket":
            session["buckets"].append(entry["graph"])

    return session


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/files")
def list_files():
    """Lists eligible cached files in the workspace directory."""
    directories = [Path("./dirty_region_caches"), Path(".")]
    bin_files = []
    for d in directories:
        if d.exists():
            for p in d.glob("*.bin"):
                bin_files.append(
                    {
                        "name": p.name,
                        "path": str(p),
                        "size_mb": round(os.path.getsize(p) / (1024 * 1024), 2),
                    }
                )
    return jsonify({"files": bin_files})


@app.post("/api/load")
def load_file():
    global CURRENT_SESSION
    data = request.get_json(force=True, silent=True) or {}
    path = data.get("path")
    if not path or not os.path.exists(path):
        return jsonify({"error": "Cache file path is invalid or missing."}), 400

    try:
        CURRENT_SESSION = load_cache_file_lazy(path)

        # Build index helper: identify dependent instructions for quick parent/child tracing
        for b_idx, bucket in enumerate(CURRENT_SESSION["buckets"]):
            dependents = {}
            for inst in bucket["instructions"]:
                node_id = inst["nodeId"]
                for p_id in inst["inputNodeIds"]:
                    if p_id not in dependents:
                        dependents[p_id] = []
                    dependents[p_id].append(node_id)
            bucket["dependentsMap"] = dependents

        return jsonify(
            {
                "status": "success",
                "file_size_mb": round(CURRENT_SESSION["file_size"] / (1024 * 1024), 2),
                "buckets_count": len(CURRENT_SESSION["buckets"]),
                "root_id": (
                    CURRENT_SESSION["metadata"]["rootId"]
                    if CURRENT_SESSION["metadata"]
                    else "N/A"
                ),
            }
        )
    except Exception as e:
        return jsonify({"error": f"Failed to parse binary file: {str(e)}"}), 500


@app.get("/api/summary")
def get_summary():
    if not CURRENT_SESSION:
        return jsonify({"error": "No file loaded."}), 400

    buckets_summary = []
    for idx, b in enumerate(CURRENT_SESSION["buckets"]):
        total_cost = sum(b["nodeCosts"].values())
        buckets_summary.append(
            {
                "index": idx,
                "instructions_count": len(b["instructions"]),
                "nodes_count": len(b["nodesMap"]),
                "total_estimated_time_ms": round(total_cost, 4),
                "dirty_inputs": list(b["bucket"]["inputDirtyRegions"].keys()),
            }
        )

    return jsonify(
        {"metadata": CURRENT_SESSION["metadata"], "buckets": buckets_summary}
    )


@app.get("/api/bucket/<int:b_idx>/instructions")
def get_instructions(b_idx):
    if not CURRENT_SESSION or b_idx >= len(CURRENT_SESSION["buckets"]):
        return jsonify({"error": "Invalid bucket index."}), 400

    bucket = CURRENT_SESSION["buckets"][b_idx]

    # Query Parameters for Server-side Pagination & Filtering
    search = request.args.get("search", "").lower()
    backend_filter = request.args.get("backend", "")
    page = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 100))

    filtered_list = []
    for inst in bucket["instructions"]:
        node_id = str(inst["nodeId"])
        node = bucket["nodesMap"].get(node_id, {})
        op_type = node.get("opType", "UNKNOWN")
        op_name = node.get("opName", "")

        display_name = f"FUSED_{op_name}" if op_type == "FUSED" else op_type

        # Apply Filters
        if backend_filter and inst["backend"] != backend_filter:
            continue
        if search:
            if search not in display_name.lower() and search not in str(inst["nodeId"]):
                continue

        filtered_list.append(
            {
                "nodeId": inst["nodeId"],
                "logicalNodeId": inst["logicalNodeId"],
                "opType": op_type,
                "opName": op_name,
                "displayName": display_name,
                "backend": inst["backend"],
                "cost_ms": round(bucket["nodeCosts"].get(inst["nodeId"], 0.0), 5),
                "inplaceInputIndex": inst["inplaceInputIndex"],
                "viewInputIndex": inst["viewInputIndex"],
                "outputStorageType": inst["outputStorageType"],
            }
        )

    # Paginate results
    total = len(filtered_list)
    start_idx = (page - 1) * limit
    end_idx = min(start_idx + limit, total)
    paginated_list = filtered_list[start_idx:end_idx]

    return jsonify(
        {"total": total, "page": page, "limit": limit, "instructions": paginated_list}
    )


@app.get("/api/bucket/<int:b_idx>/node/<int:node_id>")
def get_node_details(b_idx, node_id):
    if not CURRENT_SESSION or b_idx >= len(CURRENT_SESSION["buckets"]):
        return jsonify({"error": "Invalid bucket index."}), 400

    bucket = CURRENT_SESSION["buckets"][b_idx]
    node = bucket["nodesMap"].get(str(node_id))
    if not node:
        return jsonify({"error": f"Node {node_id} not found."}), 404

    # Fetch dependent nodes (children that use this node as input)
    dependents = bucket.get("dependentsMap", {}).get(node_id, [])

    # Find if this node has a constant associated with it
    const_meta = next(
        (c for c in bucket.get("constantsMeta", []) if c["nodeId"] == node_id), None
    )

    return jsonify(
        {
            "node": node,
            "cost_ms": round(bucket["nodeCosts"].get(node_id, 0.0), 5),
            "refCount": bucket["refCounts"].get(node_id, 0),
            "dependents": dependents,
            "constant_size_bytes": const_meta["length"] if const_meta else 0,
        }
    )


@app.get("/api/bucket/<int:b_idx>/analytics")
def get_analytics(b_idx):
    """Provides a breakdown of costs grouped by operation type."""
    if not CURRENT_SESSION or b_idx >= len(CURRENT_SESSION["buckets"]):
        return jsonify({"error": "Invalid bucket index."}), 400

    bucket = CURRENT_SESSION["buckets"][b_idx]
    op_costs = {}
    op_counts = {}

    for inst in bucket["instructions"]:
        node_id = inst["nodeId"]
        node = bucket["nodesMap"].get(str(node_id), {})
        op_type = node.get("opType", "UNKNOWN")
        if op_type == "FUSED":
            op_type = f"FUSED_{node.get('opName', 'UNKNOWN')}"

        cost = bucket["nodeCosts"].get(node_id, 0.0)
        op_costs[op_type] = op_costs.get(op_type, 0.0) + cost
        op_counts[op_type] = op_counts.get(op_type, 0) + 1

    analytics = []
    for op in op_costs:
        analytics.append(
            {
                "op": op,
                "total_cost_ms": round(op_costs[op], 4),
                "count": op_counts[op],
                "avg_cost_ms": round(op_costs[op] / op_counts[op], 5),
            }
        )

    analytics.sort(key=lambda x: x["total_cost_ms"], reverse=True)
    return jsonify({"analytics": analytics})


if __name__ == "__main__":
    app.run(host="localhost", port=8081, debug=True)
