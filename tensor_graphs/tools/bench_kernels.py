import json
import glob
import os
import time
import numpy as np
import torch
from collections import defaultdict
from tensor_graphs.backend.registry import KernelRegistry
from tensor_graphs.benchmark.db import BenchmarkDB
from tensor_graphs.benchmark.data_gen import DataGenerator
from tensor_graphs.ir.dtypes import KernelUnavailableError, Backend, DType, TensorSignature
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.compiler.propagation import GraphPropagator
from tensor_graphs.config import RECORD_KERNEL_LAUNCHES_FOLDER
from tensor_graphs.ops.atomic_types import OpType

def load_launch_records(folder):
    records = defaultdict(list)
    # Check if folder exists
    if not os.path.exists(folder):
        return records
        
    files = glob.glob(os.path.join(folder, "*.jsonl"))
    for f in files:
        with open(f, 'r') as fd:
            for line in fd:
                try:
                    rec = json.loads(line)
                    # Key: (op_type, backend)
                    key = (rec['op_type'], rec['backend'])
                    # Store tuple of (input_shapes, output_shape, attrs)
                    item = (rec['input_shapes'], rec['output_shape'], rec['attrs'])
                    records[key].append(item)
                except:
                    pass
    return records

def bench_all(db_path="benchmarks.db"):
    db = BenchmarkDB(db_path)
    registry = KernelRegistry.get_all_kernels()
    
    # Load recorded launches to prioritize real-world shapes
    launch_records = load_launch_records(RECORD_KERNEL_LAUNCHES_FOLDER)
    
    print("Starting Kernel Benchmarking...")

    for op_type, backends in registry.items():
        for backend, kernels in backends.items():
            print(f"Benchmarking {op_type} on {backend.value}...")
            
            # Check if we have recorded runs
            recorded = launch_records.get((op_type, backend.value))
            
            configs_to_run = []
            
            if recorded:
                print(f"  Found {len(recorded)} recorded launches.")
                seen = set()
                for inp_shapes, out_shape, attrs in recorded:
                    # Create a canonical key for deduplication
                    sig_key = (str(inp_shapes), str(out_shape), json.dumps(attrs, sort_keys=True))
                    if sig_key not in seen:
                        seen.add(sig_key)
                        configs_to_run.append({
                            'source': 'record',
                            'input_shapes': inp_shapes,
                            'recorded_output_shape': out_shape,
                            'attrs': attrs
                        })
            else:
                # Fallback to random generation based on registered signatures
                for entry in kernels:
                    _, sigs, _, inplace, func = entry
                    configs_to_run.append({
                        'source': 'random',
                        'sigs': sigs,
                        'func': func
                    })
            
            for config in configs_to_run:
                try:
                    target_dtype = DType.FP32

                    if config['source'] == 'record':
                        # --- RECORDED CASE ---
                        
                        # Heuristic for input dtypes based on op
                        input_dtypes = [DType.FP32] * len(config['input_shapes'])
                        if op_type == "Where":
                            input_dtypes[0] = DType.BOOL # Condition
                        elif op_type == "Gather":
                            input_dtypes[1] = DType.INT32 # Indices
                        elif op_type == "Reshape":
                            input_dtypes[1] = DType.INT32 # Shape tensor
                            
                        concrete_sigs = []
                        for s, dt in zip(config['input_shapes'], input_dtypes):
                             concrete_sigs.append(TensorSignature(dt, tuple(s), backend))
                        
                        # Find matching kernel
                        kernel_result = KernelRegistry.select_best_kernel(op_type, concrete_sigs, backend, DType.FP32)
                        
                        if not kernel_result:
                             concrete_sigs_int = [TensorSignature(DType.INT32, tuple(s), backend) for s in config['input_shapes']]
                             kernel_result = KernelRegistry.select_best_kernel(op_type, concrete_sigs_int, backend, DType.INT32)
                             if kernel_result:
                                 concrete_sigs = concrete_sigs_int
                                 target_dtype = DType.INT32
                             else:
                                 continue

                        func, _ = kernel_result
                        
                        # Generate Inputs
                        inputs = []
                        for sig in concrete_sigs:
                             inputs.append(DataGenerator.random_tensor(sig.shape, sig.dtype))
                        
                        # Special handling for Reshape: Ensure shape tensor matches input volume
                        if op_type == "Reshape":
                            # Use recorded output shape to construct valid shape tensor
                            rec_out_shape = config['recorded_output_shape']
                            inputs[1] = np.array(rec_out_shape, dtype=np.int32)
                        
                        real_inputs = DataGenerator._prepare_inputs_for_backend(
                            [(inp, backend) for inp in inputs], backend
                        )
                        attrs = config['attrs']
                        
                        # Infer Output Shape from Inputs (crucial for tiles)
                        # Create dummy nodes
                        dummy_parents = []
                        for i, inp in enumerate(real_inputs):
                            s = tuple(inp.shape) if hasattr(inp, 'shape') else (1,)
                            dt = DType.FP32
                            if hasattr(inp, 'dtype'):
                                dt_s = str(inp.dtype)
                                if 'int' in dt_s: dt = DType.INT32
                                elif 'bool' in dt_s: dt = DType.BOOL
                            dummy_parents.append(TensorNode(OpType.INPUT, dt, [], shape=s, name=f"p{i}"))

                        op_node = TensorNode(op_type, target_dtype, dummy_parents, name="bench", attrs=attrs)
                        
                        try:
                            # For Reshape, we need values. Construct dict.
                            known = {}
                            if op_type == "Reshape":
                                known[dummy_parents[1].name] = inputs[1] # Use numpy array
                            
                            GraphPropagator.infer_shapes([op_node], known, disable_pbar=True)
                            out_shape = op_node.shape
                        except:
                            # Fallback to recorded shape if inference fails (e.g. Reshape without value)
                            out_shape = tuple(config['recorded_output_shape'])

                    else:
                        # --- RANDOM CASE ---
                        sigs = config['sigs']
                        func = config['func']
                        real_inputs, attrs = DataGenerator.generate(op_type, tuple(sigs), backend)
                        
                        # Infer Output Shape
                        dummy_parents = []
                        for i, inp in enumerate(real_inputs):
                            s = tuple(inp.shape) if hasattr(inp, 'shape') else (1,)
                            dt = DType.FP32
                            if hasattr(inp, 'dtype'):
                                dt_s = str(inp.dtype)
                                if 'int' in dt_s: dt = DType.INT32
                                elif 'bool' in dt_s: dt = DType.BOOL
                            dummy_parents.append(TensorNode(OpType.INPUT, dt, [], shape=s, name=f"p{i}"))

                        # Guess target dtype
                        out_dtype = DType.FP32
                        if dummy_parents and dummy_parents[0].dtype == DType.INT32:
                             out_dtype = DType.INT32
                        
                        op_node = TensorNode(op_type, out_dtype, dummy_parents, name="bench", attrs=attrs)
                        try:
                            GraphPropagator.infer_shapes([op_node], {}, disable_pbar=True)
                            out_shape = op_node.shape
                            target_dtype = op_node.dtype
                        except:
                            if hasattr(real_inputs[0], 'shape'): out_shape = real_inputs[0].shape
                            else: out_shape = (1,)
                            target_dtype = DType.FP32

                    # --- ALLOCATE OUTPUT ---
                    if backend == Backend.GPU_TORCH:
                        if not torch.cuda.is_available():
                            continue
                        t_dt = torch.float32
                        if target_dtype == DType.INT32: t_dt = torch.int32
                        elif target_dtype == DType.BOOL: t_dt = torch.bool
                        safe_out_shape = tuple(int(d) if d is not None else 1 for d in out_shape)
                        output = torch.zeros(safe_out_shape, dtype=t_dt, device='cuda')
                    else:
                        n_dt = np.float32
                        if target_dtype == DType.INT32: n_dt = np.int32
                        elif target_dtype == DType.BOOL: n_dt = bool
                        safe_out_shape = tuple(int(d) if d is not None else 1 for d in out_shape)
                        output = np.zeros(safe_out_shape, dtype=n_dt)
                    
                    outputs = [output]

                    # --- EXECUTE ---
                    # Warmup
                    for _ in range(5):
                        func(real_inputs, outputs, attrs)

                    # Timed
                    start = time.perf_counter()
                    for _ in range(20):
                        func(real_inputs, outputs, attrs)
                    end = time.perf_counter()
                    
                    avg_ms = ((end - start) / 20.0) * 1000
                    
                    # Log
                    if op_type == "Fill" and attrs and "target_shape" in attrs:
                        primary_shape = list(attrs["target_shape"])
                    elif real_inputs and hasattr(real_inputs[0], 'shape'):
                        primary_shape = list(real_inputs[0].shape)
                    else:
                        primary_shape = list(safe_out_shape)

                    db.add_benchmark(
                        op_type, backend.value, target_dtype.value, primary_shape, attrs, avg_ms
                    )
                    
                    src_tag = "REC" if config['source'] == 'record' else "RND"
                    print(f"  [{src_tag}] Shape {primary_shape}: {avg_ms:.4f} ms")

                except KernelUnavailableError:
                    pass
                except Exception as e:
                    print(f"  Failed {op_type}: {e}")

    print("Benchmarking Complete. DB populated.")

if __name__ == "__main__":
    bench_all()