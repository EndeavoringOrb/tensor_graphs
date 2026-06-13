# File: binary_utils.py
import struct
import os


class BinaryReader:
    def __init__(self, f):
        self.f = f

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
        return self.read_u32()

    def read_backend(self):
        return self.read_u32()

    def read_map(self, read_key, read_val):
        size = self.read_u32()
        if size is None:
            return None
        return {read_key(): read_val() for _ in range(size)}

    def read_record(self):
        kernelUid = self.read_u64()
        if kernelUid is None:
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
            "kernelUid": kernelUid,
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
            "inputDirtyRegions": self.read_map(
                self.read_u32, self.read_region_list
            ),
            "outputNeededRegion": self.read_vector(self.read_region),
        }

    def read_op_instruction(self):
        return {
            "nodeId": self.read_u32(),
            "logicalNodeId": self.read_u32(),
            "fullKernelId": self.read_u64(),
            "cachedKernelIds": self.read_vector(self.read_u64),
            "inputNodeIds": self.read_vector(self.read_u32),
            "inplaceInputIndex": self.read_i32(),
            "viewInputIndex": self.read_i32(),
            "backend": self.read_backend(),
            "outputStorageType": self.read_u32(),
        }

    def read_tensor_node(self):
        _id = self.read_u32()
        opType = self.read_u32()
        opTypes = [
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
            "FUSED",
        ]  # TODO: make build.py construct this from enum in tensor_graphs_cpp

        return {
            "id": _id,
            "opType": opTypes[opType] if opType < len(opTypes) else "UNKNOWN",
            "opName": self.read_string(),
            "dtype": self.read_dtype(),
            "parentIds": self.read_vector(self.read_u32),
            "shape": self.read_vector(self.read_u32),
            "strides": self.read_vector(self.read_u64),
            "viewOffset": self.read_u64(),
            "backend": self.read_backend(),
            "storageType": self.read_u32(),
            "contentHash": self.read_string(),
        }

    def read_compiled_graph(self):
        return {
            "bucket": self.read_bucket(),
            "instructions": self.read_vector(self.read_op_instruction),
            "refCounts": self.read_map(self.read_u32, self.read_u32),
            "nodesMap": {
                str(k): v
                for k, v in self.read_map(self.read_u32, self.read_tensor_node).items()
            },
            "nodeCosts": self.read_map(self.read_u32, self.read_float),
            "physicalToLogicalNodeMap": self.read_map(self.read_u32, self.read_u32),
            "constStaging": self.read_vector(
                lambda: (self.read_u32(), self.f.read(self.read_u32()))
            ),
        }


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


def load_cache_file(path):
    entries = []
    if os.path.exists(path):
        with open(path, "rb") as f:
            br = BinaryReader(f)
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
                    constants = {}
                    count = br.read_u32()
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
        r["kernelUid"],
        tuple(tuple(s) for s in r["inputShapes"]),
        tuple(tuple(s) for s in r["inputStrides"]),
        tuple(r["inputDTypes"]),
        tuple(tuple(s) for s in r["outputShapes"]),
        tuple(tuple(s) for s in r["outputStrides"]),
        tuple(r["outputDTypes"]),
        tuple(r["inputConstants"]),
    )
