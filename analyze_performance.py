# File: analyze_performance.py
import struct
import os
import argparse
import json
from collections import defaultdict
from tqdm import tqdm

class BinaryReader:
    def __init__(self, f):
        self.f = f

    def read_u8(self):
        buf = self.f.read(1)
        if not buf: return None
        return struct.unpack("<B", buf)[0]

    def read_u32(self):
        buf = self.f.read(4)
        if not buf: return None
        return struct.unpack("<I", buf)[0]

    def read_u64(self):
        buf = self.f.read(4) # Wait, C++ writes uint32_t for size but kernelUid is uint64_t.
        # Actually in C++, bw.write(val.kernelUid) where kernelUid is uint64_t will use the generic arithmetic serializer which uses sizeof(T).
        # So it should be 8 bytes for uint64_t.
        buf = self.f.read(8)
        if not buf: return None
        return struct.unpack("<Q", buf)[0]

    def read_i32(self):
        buf = self.f.read(4)
        if not buf: return None
        return struct.unpack("<i", buf)[0]

    def read_float(self):
        buf = self.f.read(4)
        if not buf: return None
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
        if size is None: return None
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

def load_records_file(path):
    records = []
    if os.path.exists(path):
        with open(path, "rb") as f:
            br = BinaryReader(f)
            while True:
                r = br.read_record()
                if r is None: break
                records.append(r)
    return records

def load_cache_file(path):
    entries = []
    if os.path.exists(path):
        with open(path, "rb") as f:
            br = BinaryReader(f)
            while True:
                t = br.read_u8()
                if t is None: break
                if t == 0: # Metadata
                    version = br.read_u32()
                    rootId = br.read_u32()
                    selectedCachedNodes = br.read_map(br.read_u32, br.read_backend)
                    entries.append({"type": "metadata", "cacheVersion": version, "rootId": rootId, "selectedCachedNodes": selectedCachedNodes})
                elif t == 1: # Compiled Bucket
                    key = br.read_string()
                    graph = br.read_compiled_graph()
                    entries.append({"type": "compiled_bucket", "key": key, "graph": graph})
                elif t == 2: # Constants
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

def format_ms(ms):
    return f"{ms:.4f} ms"

def analyze(cache_file, records_file, top_n=20, chain_len=1):
    print(f"Loading benchmark records from: {records_file}")
    records = load_records_file(records_file)
    
    bench_map = {}
    for r in records:
        key = (
            hex(r["kernelUid"]),
            tuple(r["outputShapes"][0]),
            tuple(r["outputStrides"][0]),
        )
        bench_map[key] = r["runTime"]

    print(f"Loading compiled buckets from: {cache_file}")
    cache_entries = load_cache_file(cache_file)

    chain_stats = defaultdict(lambda: {"time": 0.0, "count": 0})
    op_type_stats = defaultdict(float)
    total_estimated_time = 0.0
    missing_benchmarks = set()

    bucket_count = 0
    for entry in cache_entries:
        if entry.get("type") != "compiled_bucket":
            continue

        bucket_count += 1
        graph = entry["graph"]
        nodes = graph["nodesMap"]
        instructions = graph["instructions"]

        bucket_sequence = []
        for inst in instructions:
            node_id = str(inst["nodeId"])
            node = nodes[node_id]

            op_name = node["opType"]
            if op_name == "FUSED":
                op_name = f"FUSED_{node.get('opName', 'UNKNOWN')}"

            input_shapes = [nodes[str(pid)]["shape"] if str(pid) in nodes else [] for pid in node["parentIds"]]
            shape = tuple(node["shape"])
            strides = tuple(node["strides"])
            uid = hex(inst["fullKernelId"])

            bench_key = (uid, shape, strides)
            runtime = bench_map.get(bench_key, 0.0)

            if bench_key not in bench_map:
                missing_benchmarks.add(f"{op_name} (UID: {uid}, Shape: {shape})")

            identity = (op_name, uid, shape, json.dumps(input_shapes))
            bucket_sequence.append({"identity": identity, "runtime": runtime})

            op_type_stats[op_name] += runtime
            total_estimated_time += runtime

        if len(bucket_sequence) >= chain_len:
            for i in range(len(bucket_sequence) - chain_len + 1):
                window = bucket_sequence[i : i + chain_len]
                chain_key = tuple(k["identity"] for k in window)
                chain_time = sum(k["runtime"] for k in window)
                chain_stats[chain_key]["time"] += chain_time
                chain_stats[chain_key]["count"] += 1

    chain_label = "Kernels" if chain_len == 1 else f"Chain of {chain_len} Kernels"
    print(f"\nAnalyzed {bucket_count} compiled buckets with chain length {chain_len}.")
    print(f"Total Estimated Execution Time: {format_ms(total_estimated_time)}")

    top_n = min(len(chain_stats), top_n)
    print("\n" + "=" * 100)
    print(f"{f'Top {top_n} {chain_label}':<105} | {'Count':<6} | {'Total Time':<12} | {'Avg'}")
    print("-" * 100)

    sorted_chains = sorted(chain_stats.items(), key=lambda x: x[1]["time"], reverse=True)

    label_len = 0
    for identities, stats in sorted_chains[:top_n]:
        parts = []
        for op_name, uid, shape, input_shapes in identities:
            parts.append(f"{op_name}({input_shapes}->{list(shape)})")
        label = " -> ".join(parts)
        label_len = max(len(label), label_len)

    for identities, stats in sorted_chains[:top_n]:
        parts = []
        for op_name, uid, shape, input_shapes in identities:
            parts.append(f"{op_name}({input_shapes}->{list(shape)})")
        label = " -> ".join(parts)
        avg = stats["time"] / stats["count"]
        print(f"{label:<{label_len}} | {stats['count']:<6} | {format_ms(stats['time']):<12} | {format_ms(avg)}")

    print("\n" + "=" * 40)
    print(f"{'Operation Type':<25} | {'Total Time'}")
    print("-" * 40)
    sorted_ops = sorted(op_type_stats.items(), key=lambda x: x[1], reverse=True)
    for op, time in sorted_ops:
        print(f"{op:<25} | {format_ms(time)}")

    if missing_benchmarks:
        print(f"\n[Warning] {len(missing_benchmarks)} kernel configurations were missing from records.bin.")
        print("Run bench.cpp to gather timing data for these kernels.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze TensorGraph performance.")
    parser.add_argument("--graph", default="dirty_region_caches/gemma-3-270m-cpp.bin")
    parser.add_argument("--records", default="benchmarks/records.bin")
    parser.add_argument("--top_n", "-n", type=int, default=20)
    parser.add_argument("--chain_len", "-c", type=int, default=1)
    args = parser.parse_args()

    try:
        analyze(args.graph, args.records, args.top_n, args.chain_len)
    except Exception as e:
        print(f"\n[Error] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
