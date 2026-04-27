import os
import json
import struct
from flask import Flask, render_template, jsonify, request, send_from_directory

app = Flask(__name__)
EGRAPHS_DIR = "egraphs"
STATIC_DIR = "static"
SETTINGS_FILE = "settings.json"

# Name Maps
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

DTYPE_MAP = {0: "FLOAT32", 1: "INT32", 2: "BF16", 3: "BOOL"}
BACKEND_MAP = {0: "CPU", 1: "CUDA"}


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    return {"constant_limit": 3}


def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f)


def interpret_constant(raw_bytes, dtype_id, limit):
    if not raw_bytes:
        return None

    # Unpack based on DType
    if dtype_id == 0:  # FLOAT32
        count = len(raw_bytes) // 4
        data = struct.unpack(f"<{count}f", raw_bytes)
    elif dtype_id == 1:  # INT32
        count = len(raw_bytes) // 4
        data = struct.unpack(f"<{count}i", raw_bytes)
    elif dtype_id == 2:  # BF16 (Interpret as uint16 and show hex/float approx)
        count = len(raw_bytes) // 2
        raw_vals = struct.unpack(f"<{count}H", raw_bytes)
        data = [f"0x{v:04x}" for v in raw_vals]
    elif dtype_id == 3:  # BOOL
        data = [bool(b) for b in raw_bytes]
    else:
        data = list(raw_bytes)

    return {
        "values": data[:limit],
        "total_count": len(data),
        "is_truncated": len(data) > limit,
    }


def parse_egraph_bin(filepath):
    if filepath in egraph_cache:
        return egraph_cache[filepath]

    eclasses = {}
    enodes = {}
    constants = {}

    with open(filepath, "rb") as f:
        num_classes = struct.unpack("<I", f.read(4))[0]
        num_enodes = struct.unpack("<I", f.read(4))[0]
        root_eclass_id = struct.unpack("<I", f.read(4))[0]

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
                "strides": strides,
                "view_offset": view_offset,
                "backend_name": BACKEND_MAP.get(backend, f"UNK({backend})"),
                "dtype_id": dtype,
                "dtype_name": DTYPE_MAP.get(dtype, f"UNK({dtype})"),
                "enodes": enode_indices,
            }

        for enode_idx in range(num_enodes):
            kernel_uid = struct.unpack("<Q", f.read(8))[0]
            op_type = struct.unpack("<I", f.read(4))[0]
            name_len = struct.unpack("<I", f.read(4))[0]
            op_name = (
                f.read(name_len).decode("utf-8")
                if name_len > 0
                else OP_TYPE_MAP.get(op_type, "UNK")
            )
            children_count = struct.unpack("<I", f.read(4))[0]
            children = [
                struct.unpack("<I", f.read(4))[0] for _ in range(children_count)
            ]
            leaf_id = struct.unpack("<I", f.read(4))[0]

            # Skip unused physical metadata in enode
            sh_sz = struct.unpack("<I", f.read(4))[0]
            f.read(sh_sz * 4)
            st_sz = struct.unpack("<I", f.read(4))[0]
            f.read(st_sz * 8)
            f.read(8 + 4 + 4 + 8)  # view_offset, dtype, backend, sig

            enodes[enode_idx] = {
                "id": enode_idx,
                "kernel_uid": f"0x{kernel_uid:x}",
                "op_name": op_name,
                "children": children,
                "leaf_id": leaf_id,
            }

        # Constants Section
        num_const_bytes = f.read(4)
        if num_const_bytes:
            num_constants = struct.unpack("<I", num_const_bytes)[0]
            for _ in range(num_constants):
                canon_id = struct.unpack("<I", f.read(4))[0]
                data_size = struct.unpack("<Q", f.read(8))[0]
                constants[canon_id] = f.read(data_size)

    egraph_cache[filepath] = {
        "eclasses": eclasses,
        "enodes": enodes,
        "root_eclass": root_eclass_id,
        "constants": constants,
    }
    return egraph_cache[filepath]


@app.route("/api/settings", methods=["GET", "POST"])
def settings_route():
    if request.method == "POST":
        save_settings(request.json)
    return jsonify(load_settings())


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

    data = parse_egraph_bin(filepath)
    return jsonify(
        {
            "root_eclass": data["root_eclass"],
            "num_eclasses": len(data["eclasses"]),
            "num_enodes": len(data["enodes"]),
        }
    )


@app.route("/api/eclass/<filename>/<int:eclass_id>")
def get_eclass(filename, eclass_id):
    data = parse_egraph_bin(os.path.join(EGRAPHS_DIR, filename))
    if eclass_id not in data["eclasses"]:
        return jsonify({"error": "Not found"}), 404

    eclass = data["eclasses"][eclass_id]
    settings = load_settings()

    constant_data = None
    if eclass_id in data["constants"]:
        constant_data = interpret_constant(
            data["constants"][eclass_id], eclass["dtype_id"], settings["constant_limit"]
        )

    return jsonify(
        {
            "id": eclass_id,
            "shape": eclass["shape"],
            "strides": eclass["strides"],
            "dtype": eclass["dtype_name"],
            "backend": eclass["backend_name"],
            "view_offset": eclass["view_offset"],
            "constant": constant_data,
            "enodes": [data["enodes"][eid] for eid in eclass["enodes"]],
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

    data = parse_egraph_bin(filepath)
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


@app.route("/")
def index():
    os.makedirs(EGRAPHS_DIR, exist_ok=True)
    files = sorted([f for f in os.listdir(EGRAPHS_DIR) if f.endswith(".bin")])
    return render_template("index.html", files=files)


if __name__ == "__main__":
    os.makedirs(EGRAPHS_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)
    print("🌐 Starting EGraph Viewer at http://localhost:5000")
    app.run(debug=True, port=5000)
