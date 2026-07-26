# File: utils/binary.py
import torch
import struct
import os

# TODO: make build.py extract these from the C++ code
# Common enums exported for reuse across the codebase (cache viewer, benchmarks, etc.)
OP_TYPES = [
    "INPUT",
    "CACHE",
    "ADD",
    "MUL",
    "DIVIDE",
    "DOT",
    "SIN",
    "COS",
    "NEGATE",
    "POWER",
    "SUM",
    "MAX",
    "RESHAPE",
    "PERMUTE",
    "SLICE",
    "CONCAT",
    "CAST",
    "REPEAT",
    "ARANGE",
    "TRIU",
    "GATHER",
    "FILL",
    "COPY_TO",
    "IM2COL",
    "CONTIGUOUS",
    "SCATTER",
    "LOG",
    "ARGMAX",
    "LT",
    "EQ",
    "AND",
    "OR",
    "NOT",
    "TRANSFER",
    "FUSED",
]

DTYPES = ["FLOAT32", "INT32", "INT64", "BF16", "BOOL", "ANY"]
BACKENDS = ["STORAGE", "CPU", "CUDA"]
STORAGE_TYPES = ["TRANSIENT", "PERSISTENT", "PINNED"]

DTYPE_MAP = {
    torch.float32: 0,  # FLOAT32
    torch.int32: 1,  # INT32
    torch.int64: 2,  # INT64
    torch.bfloat16: 3,  # BF16
    torch.bool: 4,  # BOOL
}


def make_enum_mapper(enum_list):
    def mapper(val):
        if val is None:
            return None
        return enum_list[val] if val < len(enum_list) else f"UNKNOWN({val})"

    return mapper


to_dtype = make_enum_mapper(DTYPES)
to_backend = make_enum_mapper(BACKENDS)
to_storage = make_enum_mapper(STORAGE_TYPES)
to_op_type = make_enum_mapper(OP_TYPES)


class BinaryReader:
    def __init__(self, f, lazy=False, string_enums=False):
        self.f = f
        self.lazy = lazy
        self.string_enums = string_enums

    def read_u8(self):
        buf = self.f.read(1)
        if not buf:
            return None
        return struct.unpack("<B", buf)[0]

    def read_u32(self):
        buf = self.f.read(4)
        if not buf:
            return None
        return struct.unpack("<I", buf)[0]

    def read_u64(self):
        buf = self.f.read(8)
        if not buf:
            return None
        return struct.unpack("<Q", buf)[0]

    def read_i32(self):
        buf = self.f.read(4)
        if not buf:
            return None
        return struct.unpack("<i", buf)[0]

    def read_float(self):
        buf = self.f.read(4)
        if not buf:
            return None
        return struct.unpack("<f", buf)[0]

    def read_string(self):
        size = self.read_u32()
        if size is None:
            return None
        if size == 0:
            return ""
        return self.f.read(size).decode("utf-8", errors="ignore")

    def read_vector(self, read_func):
        size = self.read_u32()
        if size is None:
            return None
        return [read_func() for _ in range(size)]

    def read_dtype(self):
        val = self.read_u32()
        return to_dtype(val) if self.string_enums else val

    def read_backend(self):
        val = self.read_u32()
        return to_backend(val) if self.string_enums else val

    def read_storage_type(self):
        val = self.read_u32()
        return to_storage(val) if self.string_enums else val

    def read_map(self, read_key, read_val):
        size = self.read_u32()
        if size is None:
            return None
        return {read_key(): read_val() for _ in range(size)}

    def read_record(self):
        kernelId = self.read_u64()
        if kernelId is None:
            return None
        buildContextId = self.read_u64()
        hwTag = self.read_string()

        inputShapes = self.read_vector(lambda: self.read_vector(self.read_u32))
        outputShapes = self.read_vector(lambda: self.read_vector(self.read_u32))
        inputStrides = self.read_vector(lambda: self.read_vector(self.read_u64))
        outputStrides = self.read_vector(lambda: self.read_vector(self.read_u64))

        inputDTypes = self.read_vector(self.read_dtype)
        outputDTypes = self.read_vector(self.read_dtype)

        inputConstants = self.read_vector(lambda: self.f.read(self.read_u32()))

        backends = self.read_vector(self.read_backend)
        inputBackends = self.read_vector(lambda: self.read_vector(self.read_backend))

        runTime = self.read_float()

        return {
            "kernelId": kernelId,
            "outputShapes": outputShapes,
            "outputStrides": outputStrides,
            "runTime": runTime,
            "inputShapes": inputShapes,
            "inputStrides": inputStrides,
            "inputDTypes": inputDTypes,
            "outputDTypes": outputDTypes,
            "inputConstants": inputConstants,
            "backends": backends,
            "inputBackends": inputBackends,
            "hwTag": hwTag,
            "buildContextId": buildContextId,
        }

    def read_dim(self):
        return {"start": self.read_u32(), "stop": self.read_u32()}

    def read_region(self):
        return {"region": self.read_vector(self.read_dim)}

    def read_region_list(self):
        return self.read_vector(self.read_region)

    def read_bucket(self):
        return {
            "inputDirtyRegions": self.read_map(self.read_u32, self.read_region_list),
            "outputNeededRegion": self.read_vector(self.read_region),
        }

    def read_op_instruction(self):
        return {
            "nodeId": self.read_u32(),
            "logicalNodeId": self.read_u32(),
            "kernelId": self.read_u64(),
            "inputNodeIds": self.read_vector(self.read_u32),
            "inplaceInputIndex": self.read_i32(),
            "viewInputIndex": self.read_i32(),
            "backend": self.read_backend(),
            "outputStorageType": (
                self.read_storage_type() if self.string_enums else self.read_u32()
            ),
        }

    def read_tensor_node(self):
        return {
            "id": self.read_u32(),
            "opType": to_op_type(self.read_u32()),
            "opName": self.read_string(),
            "dtype": self.read_dtype(),
            "child_ids": self.read_vector(self.read_u32),
            "shape": self.read_vector(self.read_u32),
            "strides": self.read_vector(self.read_u64),
            "backend": self.read_backend(),
            "storageType": (
                self.read_storage_type() if self.string_enums else self.read_u32()
            ),
            "contentHash": self.read_string(),
            "debugOrigin": self.read_string(),
        }

    def read_compiled_graph(self):
        bucket = self.read_bucket()
        instructions = self.read_vector(self.read_op_instruction)
        ref_counts = self.read_map(self.read_u32, self.read_u32)
        nodes_map = self.read_map(self.read_u32, self.read_tensor_node)
        assert nodes_map is not None
        nodes_map = {str(k): v for k, v in nodes_map.items()}
        node_costs = self.read_map(self.read_u32, self.read_float)
        physical_to_logical = self.read_map(self.read_u32, self.read_u32)

        if self.lazy:
            constants_meta = []
            const_size = self.read_u32()
            if const_size is not None:
                for _ in range(const_size):
                    node_id = self.read_u32()
                    data_len = self.read_u32()
                    offset = self.f.tell()
                    self.f.seek(data_len, 1)  # Fast skip
                    constants_meta.append(
                        {"nodeId": node_id, "length": data_len, "offset": offset}
                    )
            return {
                "bucket": bucket,
                "instructions": instructions,
                "nodesMap": nodes_map,
                "nodeCosts": node_costs,
                "physicalToLogicalNodeMap": physical_to_logical,
                "constantsMeta": constants_meta,
            }
        else:
            const_staging = self.read_vector(
                lambda: (self.read_u32(), self.f.read(self.read_u32()))
            )
            return {
                "bucket": bucket,
                "instructions": instructions,
                "nodesMap": nodes_map,
                "nodeCosts": node_costs,
                "physicalToLogicalNodeMap": physical_to_logical,
                "constStaging": const_staging,
            }


class BinaryWriter:
    def __init__(self, f):
        self.f = f

    def write_u8(self, v):
        self.f.write(struct.pack("<B", v))

    def write_u32(self, v):
        self.f.write(struct.pack("<I", v))

    def write_u64(self, v):
        self.f.write(struct.pack("<Q", v))

    def write_i32(self, v):
        self.f.write(struct.pack("<i", v))

    def write_float(self, v):
        self.f.write(struct.pack("<f", v))

    def write_string(self, s):
        b = s.encode("utf-8")
        self.write_u32(len(b))
        self.f.write(b)

    def write_vector(self, v, write_func):
        self.write_u32(len(v))
        for x in v:
            write_func(x)

    def write_record(self, r):
        self.write_u64(r["kernelId"])
        self.write_u64(r["buildContextId"])
        self.write_string(r["hwTag"])
        self.write_vector(
            r["inputShapes"], lambda v: self.write_vector(v, self.write_u32)
        )
        self.write_vector(
            r["outputShapes"], lambda v: self.write_vector(v, self.write_u32)
        )
        self.write_vector(
            r["inputStrides"], lambda v: self.write_vector(v, self.write_u64)
        )
        self.write_vector(
            r["outputStrides"], lambda v: self.write_vector(v, self.write_u64)
        )
        self.write_vector(r["inputDTypes"], self.write_u32)
        self.write_vector(r["outputDTypes"], self.write_u32)
        self.write_vector(
            r["inputConstants"], lambda v: (self.write_u32(len(v)), self.f.write(v))
        )
        self.write_vector(r["backends"], self.write_u32)
        self.write_vector(
            r["inputBackends"], lambda v: self.write_vector(v, self.write_u32)
        )
        self.write_float(r["runTime"])


def load_records_file(path):
    records = []
    if os.path.exists(path):
        with open(path, "rb") as f:
            br = BinaryReader(f)
            while True:
                r = br.read_record()
                if r is None:
                    break
                records.append(r)
    return records


def load_cache_file(path, lazy=False, string_enums=False):
    entries = []
    if os.path.exists(path):
        with open(path, "rb") as f:
            br = BinaryReader(f, lazy=lazy, string_enums=string_enums)
            while True:
                t = br.read_u8()
                if t is None:
                    break
                if t == 0:  # Metadata
                    version = br.read_u32()
                    rootId = br.read_u32()
                    selectedCachedNodes = br.read_map(br.read_u32, br.read_backend)
                    entries.append(
                        {
                            "type": "metadata",
                            "cacheVersion": version,
                            "rootId": rootId,
                            "selectedCachedNodes": selectedCachedNodes,
                        }
                    )
                elif t == 1:  # Compiled Bucket
                    graph = br.read_compiled_graph()
                    entries.append({"type": "compiled_bucket", "graph": graph})
                elif t == 2:  # Constants
                    if lazy:
                        # Skip global constants data
                        count = br.read_u32()
                        assert count is not None
                        for _ in range(count):
                            br.read_u32()
                            data_len = br.read_u32()
                            assert data_len is not None
                            f.seek(data_len, 1)
                    else:
                        constants = {}
                        count = br.read_u32()
                        assert count is not None
                        for _ in range(count):
                            nodeId = br.read_u32()
                            data = f.read(br.read_u32())
                            constants[nodeId] = data
                        entries.append({"type": "constants", "constants": constants})
                else:
                    break
    return entries


def get_record_identity(r):
    """
    Creates a unique hashable signature for a kernel configuration.
    Matches the logic used in C++ CostModel::estimateCost.
    """
    return (
        r["kernelId"],
        tuple(tuple(s) for s in r["inputShapes"]),
        tuple(tuple(s) for s in r["inputStrides"]),
        tuple(r["inputDTypes"]),
        tuple(tuple(s) for s in r["outputShapes"]),
        tuple(tuple(s) for s in r["outputStrides"]),
        tuple(r["outputDTypes"]),
        tuple(r["inputConstants"]),
    )
