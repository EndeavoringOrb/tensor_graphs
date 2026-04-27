import os
import struct
from flask import Flask, render_template, jsonify, request, send_from_directory

app = Flask(__name__)
EGRAPHS_DIR = "egraphs"
STATIC_DIR = "static"

# Cache parsed egraphs to avoid re-parsing
egraph_cache = {}
OP_TYPE_MAP = {
    0: "INPUT",
    1: "ADD",
    2: "MUL",
    3: "DIVIDE",
    4: "DOT",
    5: "SIN",
    6: "COS",
    7: "NEGATE",
    8: "POWER",
    9: "SUM",
    10: "MAX",
    11: "RESHAPE",
    12: "PERMUTE",
    13: "SLICE",
    14: "CONCAT",
    15: "CAST",
    16: "REPEAT",
    17: "ARANGE",
    18: "TRIU",
    19: "GATHER",
    20: "FILL",
    21: "COPY_TO",
    22: "IM2COL",
    23: "CONTIGUOUS",
    24: "SCATTER",
    25: "FUSED",
}


def parse_egraph_bin(filepath):
    """Parse binary egraph file and return structured data."""
    eclasses = {}
    enodes = {}

    with open(filepath, "rb") as f:
        try:
            num_classes = struct.unpack("<I", f.read(4))[0]
            num_enodes = struct.unpack("<I", f.read(4))[0]
            root_eclass_id = struct.unpack("<I", f.read(4))[0]
        except struct.error:
            return {"eclasses": {}, "enodes": {}, "root_eclass": 0}

        for _ in range(num_classes):
            cls_id = struct.unpack("<I", f.read(4))[0]
            shape_size = struct.unpack("<I", f.read(4))[0]
            shape = [struct.unpack("<I", f.read(4))[0] for _ in range(shape_size)]
            strides_size = struct.unpack("<I", f.read(4))[0]
            strides = [struct.unpack("<Q", f.read(8))[0] for _ in range(strides_size)]
            view_offset = struct.unpack("<Q", f.read(8))[0]
            dtype = struct.unpack("<I", f.read(4))[0]
            backend = struct.unpack("<I", f.read(4))[0]
            enodes_count = struct.unpack("<I", f.read(4))[0]
            enode_indices = [
                struct.unpack("<I", f.read(4))[0] for _ in range(enodes_count)
            ]
            eclasses[cls_id] = {
                "shape": shape,
                "backend": backend,
                "dtype": dtype,
                "enodes": enode_indices,
            }

        for enode_idx in range(num_enodes):
            kernel_uid = struct.unpack("<Q", f.read(8))[0]
            op_type = struct.unpack("<I", f.read(4))[0]
            name_len = struct.unpack("<I", f.read(4))[0]
            if name_len > 0:
                op_name = f.read(name_len).decode("utf-8")
            else:
                op_name = OP_TYPE_MAP.get(op_type, f"Unknown({op_type})")
            children_count = struct.unpack("<I", f.read(4))[0]
            children = [
                struct.unpack("<I", f.read(4))[0] for _ in range(children_count)
            ]
            leaf_id = struct.unpack("<I", f.read(4))[0]
            shape_size = struct.unpack("<I", f.read(4))[0]
            [struct.unpack("<I", f.read(4))[0] for _ in range(shape_size)]
            strides_size = struct.unpack("<I", f.read(4))[0]
            [struct.unpack("<Q", f.read(8))[0] for _ in range(strides_size)]
            view_offset = struct.unpack("<Q", f.read(8))[0]
            dtype = struct.unpack("<I", f.read(4))[0]
            backend = struct.unpack("<I", f.read(4))[0]
            sig = struct.unpack("<Q", f.read(8))[0]

            enodes[enode_idx] = {
                "kernel_uid": kernel_uid,
                "op_name": op_name,
                "op_type": op_type,
                "children": children,
                "leaf_id": leaf_id,
                "dtype": dtype,
                "backend": backend,
            }

    return {"eclasses": eclasses, "enodes": enodes, "root_eclass": root_eclass_id}


def get_or_parse_egraph(filename):
    """Get cached egraph or parse and cache it."""
    if filename not in egraph_cache:
        filepath = os.path.join(EGRAPHS_DIR, filename)
        egraph_cache[filename] = parse_egraph_bin(filepath)
    return egraph_cache[filename]


@app.route("/")
def index():
    os.makedirs(EGRAPHS_DIR, exist_ok=True)
    files = sorted([f for f in os.listdir(EGRAPHS_DIR) if f.endswith(".bin")])
    return render_template("index.html", files=files)


@app.route("/api/files")
def list_files():
    os.makedirs(EGRAPHS_DIR, exist_ok=True)
    files = sorted([f for f in os.listdir(EGRAPHS_DIR) if f.endswith(".bin")])
    return jsonify(files)


@app.route("/api/egraph/<filename>")
def get_egraph_meta(filename):
    """Return only metadata, not full egraph data."""
    filepath = os.path.join(EGRAPHS_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    data = get_or_parse_egraph(filename)
    return jsonify(
        {
            "root_eclass": data["root_eclass"],
            "num_eclasses": len(data["eclasses"]),
            "num_enodes": len(data["enodes"]),
        }
    )


@app.route("/api/eclass/<filename>/<int:eclass_id>")
def get_eclass(filename, eclass_id):
    """Get a single eclass with its enodes (lazy loading)."""
    filepath = os.path.join(EGRAPHS_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    data = get_or_parse_egraph(filename)

    if eclass_id not in data["eclasses"]:
        return jsonify({"error": "EClass not found"}), 404

    eclass = data["eclasses"][eclass_id]
    enodes = []
    for enode_id in eclass["enodes"]:
        if enode_id in data["enodes"]:
            enode = data["enodes"][enode_id]
            enodes.append(
                {
                    "id": enode_id,
                    "op_name": enode["op_name"],
                    "op_type": enode["op_type"],
                    "children": enode["children"],
                    "dtype": enode["dtype"],
                    "backend": enode["backend"],
                }
            )

    return jsonify(
        {
            "id": eclass_id,
            "shape": eclass["shape"],
            "backend": eclass["backend"],
            "dtype": eclass["dtype"],
            "enodes": enodes,
        }
    )


@app.route("/api/explore", methods=["POST"])
def explore():
    """Get graph data for visible eclasses only (incremental exploration)."""
    req = request.get_json()
    filename = req.get("filename")
    selection_map = req.get("selection_map", {})
    visible_eclasses = req.get("visible_eclasses", [])

    filepath = os.path.join(EGRAPHS_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    data = get_or_parse_egraph(filename)
    eclasses = data["eclasses"]
    enodes = data["enodes"]

    if not visible_eclasses:
        visible_eclasses = [data["root_eclass"]]

    nodes = []
    edges = []

    for eclass_id in visible_eclasses:
        if eclass_id not in eclasses:
            continue

        nodes.append(
            {
                "id": f"C{eclass_id}",
                "type": "eclass",
                "label": f"EC{eclass_id}",
                "eclass_id": eclass_id,
            }
        )

        selected_enode_id = selection_map.get(str(eclass_id))
        if selected_enode_id is not None and selected_enode_id in enodes:
            enode = enodes[selected_enode_id]
            nodes.append(
                {
                    "id": f"E{selected_enode_id}",
                    "type": "enode",
                    "label": enode["op_name"],
                    "eclass_id": eclass_id,
                    "enode_id": selected_enode_id,
                }
            )
            edges.append(
                {
                    "source": f"C{eclass_id}",
                    "target": f"E{selected_enode_id}",
                    "type": "contains",
                }
            )
            for child_eclass_id in enode["children"]:
                edges.append(
                    {
                        "source": f"E{selected_enode_id}",
                        "target": f"C{child_eclass_id}",
                        "type": "child",
                    }
                )

    return jsonify({"nodes": nodes, "edges": edges})


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)


if __name__ == "__main__":
    os.makedirs(EGRAPHS_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)
    print("🌐 Starting EGraph Viewer at http://localhost:5000")
    app.run(debug=True, port=5000)
