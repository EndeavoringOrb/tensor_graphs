#!/usr/bin/env python3
import struct
import os
import argparse
import re
import json

class BinaryReader:
    def __init__(self, f):
        self.f = f

    def read_u8(self):
        buf = self.f.read(1)
        assert buf
        return struct.unpack("<B", buf)[0]

    def read_u32(self):
        buf = self.f.read(4)
        assert buf
        return struct.unpack("<I", buf)[0]

    def read_u64(self):
        buf = self.f.read(8)
        assert buf
        return struct.unpack("<Q", buf)[0]

    def read_i32(self):
        buf = self.f.read(4)
        assert buf
        return struct.unpack("<i", buf)[0]

    def read_float(self):
        buf = self.f.read(4)
        assert buf
        return struct.unpack("<f", buf)[0]

    def read_string(self):
        size = self.read_u32()
        if size is None: return None
        if size == 0: return ""
        return self.f.read(size).decode('utf-8', errors='ignore')

    def read_vector(self, read_func):
        size = self.read_u32()
        if size is None: return None
        return [read_func() for _ in range(size)]

    def read_dtype(self): return self.read_u32()
    def read_backend(self): return self.read_u32()

    def read_map(self, read_key, read_val):
        size = self.read_u32()
        assert size
        return {read_key(): read_val() for _ in range(size)}

    def read_record(self):
        kernelUid = self.read_u64()
        if kernelUid is None: return None
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
            "buildContextId": buildContextId
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
            "outputStorageType": self.read_u32()
        }

    def read_tensor_node(self):
        _id = self.read_u32()
        opType = self.read_u32()
        opTypes = ["INPUT", "ADD", "MUL", "DIVIDE", "DOT", "SIN", "COS", "NEGATE", "POWER", "SUM", "MAX", "RESHAPE", "PERMUTE", "SLICE", "CONCAT", "CAST", "REPEAT", "ARANGE", "TRIU", "GATHER", "FILL", "COPY_TO", "IM2COL", "CONTIGUOUS", "SCATTER", "FUSED"]
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
            "contentHash": self.read_string()
        }

    def read_compiled_graph(self):
        return {
            "instructions": self.read_vector(self.read_op_instruction),
            "refCounts": self.read_map(self.read_u32, self.read_u32),
            "nodesMap": {str(k): v for k, v in self.read_map(self.read_u32, self.read_tensor_node).items()},
            "nodeCosts": self.read_map(self.read_u32, self.read_float),
            "physicalToLogicalNodeMap": self.read_map(self.read_u32, self.read_u32),
            "constStaging": self.read_vector(lambda: (self.read_u32(), self.f.read(self.read_u32())))
        }

def _format_constants(raw_bytes, dtype):
    if not raw_bytes: return ""
    dtypes = ["FLOAT32", "INT32", "BF16", "UINT8"]
    dt_str = dtypes[dtype] if isinstance(dtype, int) and dtype < len(dtypes) else str(dtype)
    
    if dt_str == "FLOAT32":
        count = len(raw_bytes) // 4
        return list(struct.unpack(f"<{count}f", raw_bytes))
    elif dt_str == "INT32":
        count = len(raw_bytes) // 4
        return list(struct.unpack(f"<{count}i", raw_bytes))
    return list(raw_bytes)

def format_constants(raw_bytes, dtype):
    formatted = _format_constants(raw_bytes, dtype)
    if len(formatted) > 10:
        formatted = str(formatted[:10]) + "..."
    else:
        formatted = str(formatted)
    return formatted

def load_uids_from_cpp(header_path):
    uid_to_name = {}
    if not header_path or not os.path.exists(header_path):
        return uid_to_name
    pattern = re.compile(r"constexpr uint64_t\s+(\w+)\s+=\s+(0x[0-9a-fA-F]+)ULL;")
    with open(header_path, "r") as f:
        content = f.read()
        matches = pattern.findall(content)
        for name, hex_val in matches:
            val_int = int(hex_val, 16)
            uid_to_name[str(val_int)] = name
            uid_to_name[hex_val.lower()] = name
    return uid_to_name

def load_uid_map(cache_path):
    uid_to_name = {}
    if not cache_path or not os.path.exists(cache_path):
        return uid_to_name
    with open(cache_path, "rb") as f:
        br = BinaryReader(f)
        while True:
            t = br.read_u8()
            if t is None: break
            if t == 1: # Compiled Bucket
                br.read_string() # key
                graph = br.read_compiled_graph()
                nodes = graph["nodesMap"]
                for inst in graph["instructions"]:
                    uid = str(inst["fullKernelId"])
                    node = nodes[str(inst["nodeId"])]
                    op_name = node["opType"]
                    if op_name == "FUSED":
                        op_name = f"FUSED_{node.get('opName', 'UNKNOWN')}"
                    uid_to_name[uid] = op_name
            elif t == 0: # Metadata
                br.read_u32(); br.read_u32(); br.read_map(br.read_u32, br.read_backend)
            elif t == 2: # Constants
                count = br.read_u32()
                for _ in range(count): br.read_u32(); f.read(br.read_u32())
            else: break
    return uid_to_name

def main():
    parser = argparse.ArgumentParser(description="Read and filter TensorGraph benchmarks.")
    parser.add_argument("--records", default="benchmarks/records.bin")
    parser.add_argument("--cache", default="dirty_region_caches/gemma-3-270m-cpp.bin")
    parser.add_argument("--header", default="tensor_graphs_cpp/generated/kernel_uids.gen.hpp")
    parser.add_argument("--op", help="Regex/Substring filter for OpName")
    parser.add_argument("--shape", help="String filter for OutputShape")
    args = parser.parse_args()

    if not os.path.exists(args.records):
        print(f"Error: {args.records} not found.")
        return

    uid_map = load_uid_map(args.cache)
    header_map = load_uids_from_cpp(args.header)
    uid_map.update(header_map)

    records = []
    with open(args.records, "rb") as f:
        br = BinaryReader(f)
        while True:
            r = br.read_record()
            if r is None: break
            records.append(r)

    filtered = []
    for r in records:
        uid = str(r["kernelUid"])
        name = uid_map.get(uid, uid_map.get(hex(r["kernelUid"]), None))
        if not name: continue

        if (args.op and not re.search(args.op, name, re.IGNORECASE) and not re.search(args.op, uid, re.IGNORECASE)):
            continue

        out_shapes_str = str(r.get("outputShapes", [])) + str(r.get("inputShapes", []))
        if args.shape and args.shape not in out_shapes_str:
            continue

        filtered.append((name, r))

    total = len(filtered)
    dtypes = ["FLOAT32", "INT32", "BF16", "UINT8"]
    backends_map = ["CPU", "CUDA", "NEON", "METAL"]

    for i, (name, r) in enumerate(filtered):
        uid = hex(r["kernelUid"])
        b_list = [backends_map[b] if b < len(backends_map) else "???" for b in r.get("backends", [])]
        backends = ",".join(b_list) if b_list else "CPU"

        print(f"[{i+1}/{total}][{backends}] {name} ({uid})")

        in_shapes = r.get("inputShapes", [])
        in_dtypes = r.get("inputDTypes", [])
        in_strides = r.get("inputStrides", [])
        in_consts = r.get("inputConstants", [])

        for idx in range(len(in_shapes)):
            dt_raw = in_dtypes[idx] if idx < len(in_dtypes) else -1
            dt = dtypes[dt_raw] if dt_raw >= 0 and dt_raw < len(dtypes) else "???"
            sh = in_shapes[idx]
            st = in_strides[idx] if idx < len(in_strides) else []
            ic_raw = in_consts[idx] if idx < len(in_consts) else []
            ic_formatted = format_constants(ic_raw, dt_raw)
            const_str = f", constants={ic_formatted}" if ic_formatted else ""
            print(f"  In  #{idx}: dtype={dt}, shape={sh}, strides={st}{const_str}")

        out_shapes = r.get("outputShapes", [])
        out_dtypes = r.get("outputDTypes", [])
        out_strides = r.get("outputStrides", [])
        for idx in range(len(out_shapes)):
            dt_raw = out_dtypes[idx] if idx < len(out_dtypes) else -1
            dt = dtypes[dt_raw] if dt_raw >= 0 and dt_raw < len(dtypes) else "???"
            sh = out_shapes[idx]
            st = out_strides[idx] if idx < len(out_strides) else []
            print(f"  Out #{idx}: dtype={dt}, shape={sh}, strides={st}")

        runtime = r.get("runTime", 0.0)
        print(f"  Benchmarking... -> {runtime:.6f} ms\n")

if __name__ == "__main__":
    main()
