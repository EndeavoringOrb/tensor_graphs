import numpy as np
import torch
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
from ..ir.buffer import StorageType
from ..ir.node import TensorNode
from ..ir.dtypes import DType
from ..config import DEBUG_EXECUTION, DEBUG_DETAILED

DEBUG = DEBUG_EXECUTION and DEBUG_DETAILED


@dataclass
class MemoryBlock:
    offset: int
    size: int
    node_name: str
    is_free: bool = True
    is_locked: bool = False  # Locked by current execution step
    last_used_step: int = 0  # For LRU eviction


class DeviceBuffer:
    """Manages a single linear block of memory (CPU or GPU)."""

    def __init__(self, device: str, size_bytes: int):
        self.device = device
        self.size_bytes = size_bytes
        self.alignment = 256

        # Physical storage
        if "cuda" in device or "gpu" in device:
            self.data = torch.zeros(size_bytes, dtype=torch.uint8, device=device)
            self.is_torch = True
        else:
            self.data = np.zeros(size_bytes, dtype=np.uint8)
            self.is_torch = False

        # Memory Management (Simple Best-Fit Allocator)
        # We track used blocks. Free space is implicit between blocks or explicitly tracked.
        # For simplicity and robustness, we maintain a list of distinct free segments.
        self.allocations: Dict[str, MemoryBlock] = {}  # Active allocations
        self.free_segments: List[Tuple[int, int]] = [(0, size_bytes)]  # (start, size)

        # Views cache to avoid overhead
        self.views: Dict[str, Any] = {}

    def _align(self, val: int) -> int:
        return (val + self.alignment - 1) // self.alignment * self.alignment

    def _defrag(self):
        """Merge adjacent free segments."""
        self.free_segments.sort()
        merged = []
        if not self.free_segments:
            return

        curr_start, curr_size = self.free_segments[0]
        for next_start, next_size in self.free_segments[1:]:
            if curr_start + curr_size == next_start:
                curr_size += next_size
            else:
                merged.append((curr_start, curr_size))
                curr_start, curr_size = next_start, next_size
        merged.append((curr_start, curr_size))
        self.free_segments = merged

    def allocate(self, node_name: str, size: int, step: int) -> Optional[int]:
        """
        Allocate memory for a node.
        Returns offset if successful, None if eviction is required.
        """
        aligned_size = self._align(size)

        # 1. Best-fit strategy
        best_idx = -1
        min_waste = float("inf")

        for i, (start, seg_size) in enumerate(self.free_segments):
            if seg_size >= aligned_size:
                waste = seg_size - aligned_size
                if waste < min_waste:
                    min_waste = waste
                    best_idx = i
                # Optimization: perfect fit
                if waste == 0:
                    break

        if best_idx == -1:
            return None  # Out of memory / needs eviction

        # 2. Carve out memory
        start, seg_size = self.free_segments.pop(best_idx)
        offset = start

        # Return remaining chunk to free list
        remaining = seg_size - aligned_size
        if remaining > 0:
            self.free_segments.append((start + aligned_size, remaining))
            self.free_segments.sort()  # Keep sorted for defrag

        # 3. Record allocation
        block = MemoryBlock(
            offset=offset,
            size=aligned_size,
            node_name=node_name,
            is_free=False,
            is_locked=True,
            last_used_step=step,
        )
        self.allocations[node_name] = block

        # Invalidate specific view cache
        if node_name in self.views:
            del self.views[node_name]

        if DEBUG:
            print(
                f"[DeviceBuffer.allocate] {node_name} {offset}-{offset + aligned_size}"
            )

        return offset

    def free(self, node_name: str):
        """Mark a block as free (used for persistent removal or eviction)."""
        if node_name not in self.allocations:
            if DEBUG:
                print(f"[DeviceBuffer.free] {node_name} not allocated")
            return

        block = self.allocations.pop(node_name)
        if node_name in self.views:
            del self.views[node_name]

        self.free_segments.append((block.offset, block.size))
        self._defrag()

        if DEBUG:
            print(
                f"[DeviceBuffer.free] {node_name} {block.offset}-{block.offset + block.size}"
            )

    def get_view(
        self,
        node_name: str,
        shape: Tuple[int, ...],
        dtype: DType,
        dirty_region: Optional[List] = None,
    ) -> Any:
        if node_name not in self.allocations:
            raise RuntimeError(
                f"Attempted to access unallocated memory for {node_name}"
            )

        # For dirty regions, we don't use the cache since the region might change
        use_cache = dirty_region is None

        # Return cached view if valid and no dirty region
        if use_cache and node_name in self.views:
            if DEBUG:
                print(f"[DeviceBuffer.get_view] {node_name} cached view")
            return self.views[node_name]

        block = self.allocations[node_name]

        if self.is_torch:
            t_dtype = self._map_dtype_torch(dtype)
            # View as bytes then cast
            raw = self.data[block.offset : block.offset + block.size]
            # Ensure we only view the exact number of bytes for the shape
            # (Block might be padded)
            req_bytes = self._calc_bytes(shape, dtype)
            sliced = raw[:req_bytes]
            view = sliced.view(t_dtype).reshape(shape)
        else:
            np_dtype = self._map_dtype_np(dtype)
            view = np.ndarray(
                shape, dtype=np_dtype, buffer=self.data.data, offset=block.offset
            )

        # Apply dirty region slicing if specified
        if dirty_region:
            # Ensure we have enough slices for all dimensions
            full_slices = list(dirty_region) + [slice(None)] * (
                len(shape) - len(dirty_region)
            )
            view = view[tuple(full_slices[: len(shape)])]
            if DEBUG:
                print(
                    f"[DeviceBuffer.get_view] {node_name} dirty region {dirty_region} -> shape {view.shape}"
                )

        # Only cache if no dirty region
        if use_cache:
            self.views[node_name] = view

        if DEBUG:
            print(
                f"[DeviceBuffer.get_view] {node_name} {block.offset}-{block.offset + block.size}"
            )

        return view

    def _calc_bytes(self, shape, dtype):
        import math

        elem_size = 4
        if dtype == DType.FP16:
            elem_size = 2
        elif dtype == DType.BOOL:
            elem_size = 1
        elif dtype == DType.FP8E4M3:
            elem_size = 1
        return math.prod(shape) * elem_size

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


class MemoryManager:
    """
    Unified manager for Static Persistence, Dynamic Execution, and Caching.

    - No copying for cache: Cache is just a block that isn't freed yet.
    - Handles eviction: If dynamic allocation fails, evict LRU cached blocks.
    """

    def __init__(self, max_bytes: int = 4 * 1024**3):
        self.max_bytes = max_bytes
        self.buffers: Dict[str, DeviceBuffer] = {}  # e.g., "cuda:0", "cpu"
        self.current_step = 0

        # Default CPU buffer
        self._ensure_device("cpu")

    def _ensure_device(self, device: str):
        if device not in self.buffers:
            # For now, simplistic split of max_bytes or dedicated size
            # In production, this might be per-device limits.
            self.buffers[device] = DeviceBuffer(device, self.max_bytes)

    def allocate_persistent(self, node: TensorNode, data: Any):
        """Allocate space for weights once. Panic if full."""
        # Handle different data types for size calculation
        if hasattr(data, "nbytes"):
            # NumPy array
            size = data.nbytes
        elif hasattr(data, "numel"):
            # PyTorch tensor
            size = data.numel() * data.element_size()
        elif isinstance(data, (int, float, bool)):
            # Python scalar - determine size from node dtype or default
            dtype_sizes = {
                DType.FP32: 4,
                DType.FP16: 2,
                DType.INT32: 4,
                DType.BOOL: 1,
                DType.FP8E4M3: 1,
            }
            size = dtype_sizes.get(node.dtype, 4)  # Default to 4 bytes
        elif isinstance(data, np.generic):
            # NumPy scalar
            size = data.nbytes
        else:
            # Fallback: try to convert to numpy and get size
            try:
                arr = np.asarray(data)
                size = arr.nbytes
            except:
                raise TypeError(f"Cannot determine size for data of type {type(data)}")

        device = node.backend.value if node.backend else "cpu"
        if "numpy" in device:
            device = "cpu"
        elif "torch" in device and "cpu" in device:
            device = "cpu"

        self._ensure_device(device)
        offset = self.buffers[device].allocate(node.name, size, step=0)

        if offset is None:
            raise MemoryError(f"OOM allocating persistent node {node.name}")

        # Write immediately
        self.write(node, data)
        # Lock forever
        self.buffers[device].allocations[node.name].is_locked = True

        if DEBUG:
            print(
                f"[DeviceBuffer.allocate_persistent] {node.name} {offset}-{offset + size}"
            )

    def prepare_allocation(self, node: TensorNode, size_bytes: int) -> bool:
        """
        Ensure space exists for a transient node.
        If node is already cached, marks it as used (LRU update) and returns True.
        If not, allocates space (evicting if needed).
        """
        device = node.backend.value if node.backend else "cpu"
        if "numpy" in device:
            device = "cpu"
        elif "torch" in device and "cpu" in device:
            device = "cpu"

        self._ensure_device(device)
        buf = self.buffers[device]

        # 1. Cache Hit?
        if node.name in buf.allocations:
            buf.allocations[node.name].last_used_step = self.current_step
            buf.allocations[node.name].is_locked = True
            if DEBUG:
                print(f"[DeviceBuffer.prepare_allocation] {node.name} hit")
            return True

        # 2. Try Allocate
        offset = buf.allocate(node.name, size_bytes, self.current_step)

        # 3. Evict if needed
        if offset is None:
            # Gather evictable candidates (unlocked blocks)
            candidates = [
                b
                for b in buf.allocations.values()
                if not b.is_locked and b.node_name != node.name
            ]
            # Sort by LRU
            candidates.sort(key=lambda b: b.last_used_step)

            freed = 0
            for victim in candidates:
                if DEBUG:
                    print(
                        f"[DeviceBuffer.prepare_allocation] Evicting {victim.node_name}"
                    )
                buf.free(victim.node_name)
                freed += victim.size
                if freed >= size_bytes:  # Heuristic check
                    offset = buf.allocate(node.name, size_bytes, self.current_step)
                    if offset is not None:
                        break

            # Still failed?
            if offset is None:
                raise MemoryError(
                    f"OOM: Could not allocate {size_bytes} bytes for {node.name} after eviction."
                )

        if DEBUG:
            print(f"[DeviceBuffer.prepare_allocation] {node.name} miss")

        return True

    def get_view(self, node: TensorNode, use_dirty: bool = False) -> Any:
        device = node.backend.value if node.backend else "cpu"
        if "numpy" in device:
            device = "cpu"
        elif "torch" in device and "cpu" in device:
            device = "cpu"

        # Ensure shape is concrete
        if not node.shape or any(d is None for d in node.shape):
            # Scalar fallback
            shape = ()
        else:
            shape = tuple(node.shape)

        # Pass dirty_region to the buffer's get_view
        dirty_region = getattr(node, "dirty_region", None) if use_dirty else None
        return self.buffers[device].get_view(node.name, shape, node.dtype, dirty_region)

    def write(self, node: TensorNode, data: Any):
        view = self.get_view(node)

        # Handle different source/dest types
        if isinstance(view, np.ndarray):
            if hasattr(data, "cpu"):
                data = data.cpu().numpy()
            elif isinstance(data, (int, float)):
                data = np.array(data, dtype=view.dtype)

            # Use [...] instead of [:] to handle 0-dimensional arrays (scalars)
            # [...] works for all dimensions, [:] fails for 0-d arrays
            if data.shape != view.shape:
                try:
                    view[...] = data.reshape(view.shape)
                except:
                    view[...] = data  # Hope for broadcast
            else:
                view[...] = data

        elif hasattr(view, "copy_"):
            # Torch
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data)

            data = data.to(view.device, dtype=view.dtype)
            view.copy_(data.reshape(view.shape))

    def has(self, node_name: str, device_hint: str = "cpu") -> bool:
        if "numpy" in device_hint:
            device_hint = "cpu"
        if device_hint not in self.buffers:
            return False
        return node_name in self.buffers[device_hint].allocations

    def lock(self, node: TensorNode):
        """Mark node as needed for current execution."""
        device = node.backend.value if node.backend else "cpu"
        if "numpy" in device:
            device = "cpu"
        elif "torch" in device and "cpu" in device:
            device = "cpu"
        if node.name in self.buffers[device].allocations:
            self.buffers[device].allocations[node.name].is_locked = True
            self.buffers[device].allocations[
                node.name
            ].last_used_step = self.current_step

        if DEBUG:
            print(
                f"[DeviceBuffer.lock] {node.name} {self.buffers[device].allocations[node.name].is_locked}"
            )

    def unlock(self, node: TensorNode):
        """Mark node as done for current execution (can be evicted if needed)."""
        device = node.backend.value if node.backend else "cpu"
        if "numpy" in device:
            device = "cpu"
        elif "torch" in device and "cpu" in device:
            device = "cpu"

        if node.name in self.buffers[device].allocations:
            # Persistent nodes stay locked
            if node.storage_type == StorageType.TRANSIENT:
                self.buffers[device].allocations[node.name].is_locked = False

        if DEBUG:
            print(
                f"[DeviceBuffer.unlock] {node.name} {self.buffers[device].allocations[node.name].is_locked}"
            )

    def step(self):
        """Advance time step (for LRU)."""
        self.current_step += 1

        if DEBUG:
            print(f"[DeviceBuffer.step] {self.current_step}")
