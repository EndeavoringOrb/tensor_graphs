import numpy as np
import torch
from typing import Dict, Any, List
from ..compiler.compiled_graph import CompiledGraph
from ..ir.buffer import StorageType
from ..ir.dtypes import DType


class StaticExecutor:
    def __init__(self, compiled_graph: CompiledGraph):
        self.graph = compiled_graph
        self.buffers: Dict[str, Any] = {}

        # Calculate size per device
        device_sizes: Dict[str, int] = {}
        for alloc in self.graph.buffer_allocations.values():
            end = alloc.offset + alloc.size_bytes
            device_sizes[alloc.device] = max(device_sizes.get(alloc.device, 0), end)

        for device, size in device_sizes.items():
            if "cuda" in device or "gpu" in device:
                # Align to 256 bytes or more
                self.buffers[device] = torch.empty(
                    size, dtype=torch.uint8, device=device
                )
            else:
                # CPU
                self.buffers[device] = np.zeros(size, dtype=np.uint8)

    def _map_dtype_np(self, dtype: DType):
        if dtype == DType.FP32:
            return np.float32
        if dtype == DType.INT32:
            return np.int32
        if dtype == DType.FP16:
            return np.float16
        if dtype == DType.BOOL:
            return np.bool_
        return np.float32

    def _map_dtype_torch(self, dtype: DType):
        if dtype == DType.FP32:
            return torch.float32
        if dtype == DType.INT32:
            return torch.int32
        if dtype == DType.FP16:
            return torch.float16
        if dtype == DType.BOOL:
            return torch.bool
        return torch.float32

    def _get_view(self, offset: int, name: str, device: str = None) -> Any:
        meta = self.graph.node_metadata[name]
        alloc = self.graph.buffer_allocations[name]
        dev = device or alloc.device
        buf = self.buffers[dev]
        size_bytes = alloc.size_bytes

        if isinstance(buf, np.ndarray):
            np_dtype = self._map_dtype_np(meta.dtype)
            # Create a view into the buffer
            # Note: We must ensure alignment/size match
            return np.ndarray(
                meta.shape, dtype=np_dtype, buffer=buf.data, offset=offset
            )

        elif isinstance(buf, torch.Tensor):
            torch_dtype = self._map_dtype_torch(meta.dtype)
            slice_t = buf[offset : offset + size_bytes]
            return slice_t.view(torch_dtype).view(meta.shape)

        raise RuntimeError(f"Unknown buffer type: {type(buf)}")

    def copy_to_buffer(self, name: str, data: Any):
        if name not in self.graph.buffer_allocations:
            return

        alloc = self.graph.buffer_allocations[name]
        offset = alloc.offset
        buf = self.buffers[alloc.device]
        meta = self.graph.node_metadata[name]

        if isinstance(buf, np.ndarray):
            if not isinstance(data, np.ndarray):
                data = np.array(data)

            np_dtype = self._map_dtype_np(meta.dtype)

            # Flatten and view as uint8 to copy
            data_casted = data.astype(np_dtype)
            data_u8 = data_casted.view(np.uint8).flatten()

            end = offset + len(data_u8)
            buf[offset:end] = data_u8

        elif isinstance(buf, torch.Tensor):
            t_dtype = self._map_dtype_torch(meta.dtype)
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=t_dtype, device=alloc.device)
            else:
                data = data.to(dtype=t_dtype, device=alloc.device)

            data_u8 = data.view(torch.uint8).flatten()
            end = offset + len(data_u8)
            buf[offset:end] = data_u8

    def load_weights(self, weights: Dict[str, Any]):
        for name, data in weights.items():
            if name in self.graph.buffer_allocations:
                alloc = self.graph.buffer_allocations[name]
                if alloc.storage_type in (StorageType.PERSISTENT, StorageType.STATE):
                    self.copy_to_buffer(name, data)

    def run(self, inputs: Dict[str, Any]) -> Any:
        # 1. Copy inputs
        for name, data in inputs.items():
            self.copy_to_buffer(name, data)

        # 2. Execution Loop
        for inst in self.graph.instructions:
            # Prepare args (views)
            args = []
            for i, offset in enumerate(inst.input_offsets):
                input_name = inst.input_node_names[i]
                args.append(self._get_view(offset, input_name))

            # Prepare output (view)
            out_view = self._get_view(inst.output_offset, inst.node_name)

            # Execute Kernel
            # Kernels typically take (*inputs, attrs) and return output
            # OR take (*inputs, output, attrs) -> in-place/out-ptr style
            # The current kernels in `registry.py` return a new value.
            # We need to change kernels to support writing to `out` OR
            # if we can't change kernels easily, we copy the result to `out`.

            # "ideally kernels take raw pointers/offsets"
            # Since we are wrapping existing kernels, they likely return new arrays.
            # We need to copy the result into our buffer.

            result = inst.kernel(args, inst.attrs)

            # Copy result to out_view
            # If result is same shape/dtype, we can assign?
            # out_view[:] = result

            if isinstance(out_view, np.ndarray):
                np.copyto(out_view, result, casting="no")
            elif isinstance(out_view, torch.Tensor):
                out_view.copy_(result)

        # 3. Return Output
        # Assume root is the last instruction output?
        # Or look up output_offsets.
        # But `CompiledGraph` has `output_offsets` map.

        # We return a dict of outputs? Or single output?
        # `evaluate_graph` returns single output.
        # Let's find the root node name.
        # We don't have root node name directly in CompiledGraph field easily,
        # but `output_offsets` keys are likely the outputs.

        outputs = {}
        for name, offset in self.graph.output_offsets.items():
            outputs[name] = self._get_view(offset, name)

        # If single output, return it directly?
        if len(outputs) == 1:
            return list(outputs.values())[0]

        return outputs
