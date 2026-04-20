import os
import struct
from flask import Flask, render_template, jsonify, request, send_from_directory

app = Flask(__name__)
EGRAPHS_DIR = "egraphs"
STATIC_DIR = "static"


def parse_egraph_bin(filepath):
    """Parse binary egraph file and return structured data."""
    eclasses = {}  # eclass_id -> {shape, backend, enodes: [...]}
    enodes = {}  # enode_id -> {kernel_uid, op_name, children: [...], ...}

    with open(filepath, "rb") as f:
        try:
            num_classes = struct.unpack("<I", f.read(4))[0]
            num_enodes = struct.unpack("<I", f.read(4))[0]
            root_eclass_id = struct.unpack("<I", f.read(4))[0]
        except struct.error:
            return {"eclasses": {}, "enodes": {}, "root_eclass": 0}

        # Parse eclasses
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

        # Parse enodes
        for enode_idx in range(num_enodes):
            kernel_uid = struct.unpack("<Q", f.read(8))[0]
            op_type = struct.unpack("<I", f.read(4))[0]
            name_len = struct.unpack("<I", f.read(4))[0]
            op_name = (
                f.read(name_len).decode("utf-8") if name_len > 0 else f"Op{op_type}"
            )
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
                "children": children,  # List of child eclass IDs
                "leaf_id": leaf_id,
                "dtype": dtype,
                "backend": backend,
            }

    return {
        "eclasses": eclasses,
        "enodes": enodes,
        "root_eclass": root_eclass_id,  # Assume eclass 0 is root; customize if needed
    }


def extract_graph(egraph_data, selection_map):
    """
    Extract a single graph from the egraph given a selection map.
    selection_map: {eclass_id: selected_enode_id}
    Returns: {nodes: [...], edges: [...]} for rendering
    """
    eclasses = egraph_data["eclasses"]
    enodes = egraph_data["enodes"]
    root_eclass = egraph_data["root_eclass"]

    if not eclasses:
        return {"nodes": [], "edges": []}

    visited_eclasses = set()
    visited_enodes = set()
    nodes = []
    edges = []

    def extract_eclass(eclass_id, depth=0):
        if eclass_id in visited_eclasses or eclass_id not in eclasses:
            return None

        visited_eclasses.add(eclass_id)
        eclass = eclasses[eclass_id]

        # Determine which enode to use for this eclass
        selected_enode_id = selection_map.get(str(eclass_id))
        if selected_enode_id is None or selected_enode_id not in enodes:
            return None

        if selected_enode_id in visited_enodes:
            return f"C{eclass_id}"  # Return reference to existing eclass node

        visited_enodes.add(selected_enode_id)
        enode = enodes[selected_enode_id]

        # Add enode to graph
        enode_node_id = f"E{selected_enode_id}"
        nodes.append(
            {
                "id": enode_node_id,
                "type": "enode",
                "label": enode["op_name"],
                "eclass_id": eclass_id,
                "enode_id": selected_enode_id,
                "depth": depth,
            }
        )

        # Add edge from eclass to selected enode
        eclass_node_id = f"C{eclass_id}"
        edges.append(
            {"source": eclass_node_id, "target": enode_node_id, "type": "contains"}
        )

        # Process children
        for child_eclass_id in enode["children"]:
            child_ref = extract_eclass(child_eclass_id, depth + 1)
            if child_ref:
                edges.append(
                    {"source": enode_node_id, "target": child_ref, "type": "child"}
                )

        return enode_node_id

    # Start extraction from root
    root_eclass_node = f"C{root_eclass}"
    nodes.append(
        {
            "id": root_eclass_node,
            "type": "eclass",
            "label": f"EC{root_eclass}",
            "eclass_id": root_eclass,
            "depth": 0,
        }
    )

    extract_eclass(root_eclass)

    return {"nodes": nodes, "edges": edges}


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
def get_egraph(filename):
    filepath = os.path.join(EGRAPHS_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    print(f"parsing egraph file {filepath}")
    data = parse_egraph_bin(filepath)
    return jsonify(data)


@app.route("/api/extract", methods=["POST"])
def extract():
    req = request.get_json()
    filename = req.get("filename")
    selection_map = req.get("selection_map", {})

    filepath = os.path.join(EGRAPHS_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    egraph_data = parse_egraph_bin(filepath)
    graph = extract_graph(egraph_data, selection_map)
    return jsonify(graph)


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)


if __name__ == "__main__":
    os.makedirs(EGRAPHS_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)
    print("🌐 Starting EGraph Viewer at http://localhost:5000")
    app.run(debug=True, port=5000)
