import numpy as np
import torch
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
from ..ir.buffer import StorageType
from ..ir.node import TensorNode
from ..ir.dtypes import DType, get_buffer_size, get_size_bytes
from ..ops.atomic_types import OpType
from ..config import DEBUG_EXECUTION, DEBUG_DETAILED

DEBUG = DEBUG_EXECUTION and DEBUG_DETAILED


@dataclass
class MemoryBlock:
    offset: int
    size: int
    node_name: str
    storage_type: StorageType  # Added to track if it's a weight vs activation
    region_id: int = 0  # 0 = Internal Arena, >0 = External Region
    is_free: bool = True
    is_locked: bool = False
    last_used_step: int = 0
    ref_count: int = 0


class DeviceBuffer:
    """
    Manages memory for a specific device.
    Supports multiple memory regions:
    - Region 0: Internal contiguous arena (allocated at init).
    - Region N: External regions (registered memory, e.g., mmap'd weights).
    """

    def __init__(self, device: str, size_bytes: int):
        self.device = device
        self.size_bytes = size_bytes  # Size of Region 0 only
        self.alignment = 256

        # List of backing arrays.
        # Index 0 is always the internal arena.
        self.regions: List[Any] = []

        # Initialize Region 0 (Internal Arena)
        if "cuda" in device or "gpu" in device:
            self.regions.append(
                torch.zeros(size_bytes, dtype=torch.uint8, device=device)
            )
            self.is_torch = True
        else:
            self.regions.append(np.zeros(size_bytes, dtype=np.uint8))
            self.is_torch = False

        # Allocation tracking (only applies to Region 0)
        self.allocations: Dict[str, MemoryBlock] = {}
        self.free_segments: List[Tuple[int, int]] = [(0, size_bytes)]
        self.views: Dict[str, Any] = {}

    def register_external_region(self, data: Any) -> int:
        """Registers an external array (e.g., mmap) as a new region. Returns region_id."""
        region_id = len(self.regions)
        self.regions.append(data)
        return region_id

    def _align(self, val: int) -> int:
        return (val + self.alignment - 1) // self.alignment * self.alignment

    def _defrag(self):
        """Merge adjacent free segments in Region 0."""
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

    def allocate(
        self,
        node_name: str,
        size: int,
        step: int,
        storage_type: StorageType,
        initial_refs: int,
    ) -> Optional[int]:
        """Allocate memory in Region 0 (Internal Arena)."""
        aligned_size = self._align(size)

        best_idx = -1
        min_waste = float("inf")

        for i, (start, seg_size) in enumerate(self.free_segments):
            if seg_size >= aligned_size:
                waste = seg_size - aligned_size
                if waste < min_waste:
                    min_waste = waste
                    best_idx = i
                if waste == 0:
                    break

        if best_idx == -1:
            return None

        start, seg_size = self.free_segments.pop(best_idx)
        offset = start

        remaining = seg_size - aligned_size
        if remaining > 0:
            self.free_segments.append((start + aligned_size, remaining))
            self.free_segments.sort()

        block = MemoryBlock(
            offset=offset,
            size=aligned_size,
            node_name=node_name,
            storage_type=storage_type,
            region_id=0,
            is_free=False,
            is_locked=True,  # Initially locked while refs > 0 or if persistent
            last_used_step=step,
            ref_count=initial_refs,
        )
        self.allocations[node_name] = block

        if node_name in self.views:
            del self.views[node_name]

        if DEBUG:
            print(
                f"[DeviceBuffer.allocate] {node_name} Region0 {offset}-{offset + aligned_size}"
            )

        return offset

    def allocate_external(
        self, node_name: str, data: Any, step: int, storage_type: StorageType
    ):
        """
        Register an external memory region and create a block pointing to it.
        Used for Zero-Copy weights.
        """
        # Register the external memory region
        region_id = self.register_external_region(data)

        # Determine size (used for tracking, not allocation)
        if hasattr(data, "nbytes"):
            size = data.nbytes
        elif hasattr(data, "numel"):
            size = data.numel() * data.element_size()
        else:
            size = 0  # Scalar?

        # Create block pointing to this region
        # Offset is 0 relative to the region start (the data IS the region)
        block = MemoryBlock(
            offset=0,
            size=size,
            node_name=node_name,
            storage_type=storage_type,
            region_id=region_id,
            is_free=False,
            is_locked=True,
            last_used_step=step,
            ref_count=0,  # Persistent weights typically don't use ref counting
        )
        self.allocations[node_name] = block

        if DEBUG:
            print(
                f"[DeviceBuffer.allocate_external] {node_name} Region{region_id} (Zero-Copy)"
            )

    def free(self, node_name: str):
        """Mark a block as free (used for eviction from Region 0)."""
        if node_name not in self.allocations:
            if DEBUG:
                print(f"[DeviceBuffer.free] {node_name} not allocated")
            return

        block = self.allocations.pop(node_name)
        if node_name in self.views:
            del self.views[node_name]

        # Only Region 0 blocks can be freed back to the arena
        if block.region_id == 0:
            self.free_segments.append((block.offset, block.size))
            self._defrag()
            if DEBUG:
                print(
                    f"[DeviceBuffer.free] {node_name} Region0 {block.offset}-{block.offset + block.size}"
                )
        else:
            # External regions are not 'freed' via the allocator.
            # They are managed by the weight source lifetime.
            if DEBUG:
                print(
                    f"[DeviceBuffer.free] {node_name} Region{block.region_id} (External, just dereferenced)"
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

        use_cache = dirty_region is None
        if use_cache and node_name in self.views:
            if DEBUG:
                print(f"[DeviceBuffer.get_view] {node_name} cached view")
            return self.views[node_name]

        block = self.allocations[node_name]

        # Select backing array based on region_id
        backing_array = self.regions[block.region_id]

        # Check if the backing array for this region is a Torch Tensor
        is_region_torch = isinstance(backing_array, torch.Tensor)

        if is_region_torch:
            t_dtype = self._map_dtype_torch(dtype)
            # For external regions (weights), backing_array IS the tensor.
            # We just need to ensure it matches shape/dtype (view logic).
            if block.region_id > 0:
                # External region: data is the tensor itself.
                # Assume it's contiguous or view-compatible.
                # Force a view to handle slicing/shape adjustments.
                # Note: If data is already the correct shape, view() is cheap.
                view = backing_array.view(t_dtype).reshape(shape)
            else:
                # Region 0 logic (byte slab)
                raw = backing_array[block.offset : block.offset + block.size]
                req_bytes = get_size_bytes(shape, dtype)
                sliced = raw[:req_bytes]
                view = sliced.view(t_dtype).reshape(shape)
        else:
            # NumPy logic
            np_dtype = self._map_dtype_np(dtype)

            if block.region_id > 0:
                # External region: backing_array is the numpy array
                view = np.ndarray(
                    shape, dtype=np_dtype, buffer=backing_array.data, offset=0
                )
            else:
                # Region 0 logic
                view = np.ndarray(
                    shape,
                    dtype=np_dtype,
                    buffer=backing_array.data,
                    offset=block.offset,
                )

        # Apply dirty region slicing if specified
        if dirty_region:
            full_slices = list(dirty_region) + [slice(None)] * (
                len(shape) - len(dirty_region)
            )
            view = view[tuple(full_slices[: len(shape)])]
            if DEBUG:
                print(
                    f"[DeviceBuffer.get_view] {node_name} dirty region {dirty_region} -> shape {view.shape}"
                )

        if use_cache:
            self.views[node_name] = view

        return view

    def _map_dtype_np(self, dtype: DType):
        if dtype == DType.FP32:
            return np.float32
        if dtype == DType.INT32:
            return np.int32
        if dtype == DType.BOOL:
            return np.bool_
        return np.float32

    def _map_dtype_torch(self, dtype: DType):
        if dtype == DType.FP32:
            return torch.float32
        if dtype == DType.INT32:
            return torch.int32
        if dtype == DType.BOOL:
            return torch.bool
        return torch.float32


class MemoryManager:
    def __init__(self, max_bytes: int = 4 * 1024**3):
        self.max_bytes = max_bytes
        self.buffers: Dict[str, DeviceBuffer] = {}
        self.current_step = 0
        self._ensure_device("cpu")

    def _ensure_device(self, device: str):
        if device not in self.buffers:
            self.buffers[device] = DeviceBuffer(device, self.max_bytes)

    def allocate_persistent(self, node: TensorNode, data: Any):
        """
        Allocate space for weights.
        - GPU: Allocates in Region 0 (Arena) and copies data (Memcpy).
        - CPU: Registers data as External Region (Zero-Copy).
        """
        size = get_buffer_size(node.dtype, data=data)

        device = node.backend.value if node.backend else "cpu"
        if "numpy" in device:
            device = "cpu"
        elif "torch" in device and "cpu" in device:
            device = "cpu"

        self._ensure_device(device)
        buf = self.buffers[device]

        # Strategy: Zero-Copy for CPU, Copy for GPU
        if device == "cpu" and node.op_type != OpType.CONSTANT:
            # --- Zero-Copy Path ---
            # Register the external memory region
            # We assume data is passed as a memory-backed object (Torch Tensor or Numpy Array)
            buf.allocate_external(
                node.name, data, step=0, storage_type=StorageType.PERSISTENT
            )
        else:
            # --- Standard Copy Path (GPU) ---
            offset = buf.allocate(
                node.name,
                size,
                step=0,
                storage_type=StorageType.PERSISTENT,
                initial_refs=0,
            )
            if offset is None:
                raise MemoryError(f"OOM allocating persistent node {node.name}")

            # Write data to Region 0
            self.write(node, data)

        # Lock forever
        if node.name in buf.allocations:
            buf.allocations[node.name].is_locked = True

        if DEBUG:
            print(f"[MemoryManager.allocate_persistent] {node.name} on {device}")

    def prepare_allocation(
        self, node: TensorNode, size_bytes: int, initial_refs: int
    ) -> bool:
        """Ensure space exists for a transient node (Region 0)."""
        device = node.backend.value if node.backend else "cpu"
        if "numpy" in device:
            device = "cpu"
        elif "torch" in device and "cpu" in device:
            device = "cpu"

        self._ensure_device(device)
        buf = self.buffers[device]

        if node.name in buf.allocations:
            block = buf.allocations[node.name]
            block.last_used_step = self.current_step
            # Only reset ref_count if we are moving from an unlocked (evictable)
            # state to a locked (active) state for this run.
            if not block.is_locked:
                block.ref_count = initial_refs
                block.is_locked = True
            return True

        offset = buf.allocate(
            node.name,
            size_bytes,
            self.current_step,
            StorageType.TRANSIENT,
            initial_refs,
        )

        if offset is None:
            # Eviction logic: Only candidates with is_locked=False (which implies ref_count=0)
            candidates = [
                b
                for b in buf.allocations.values()
                if not b.is_locked and b.region_id == 0
            ]
            candidates.sort(key=lambda b: b.last_used_step)

            freed = 0
            for victim in candidates:
                if DEBUG:
                    print(
                        f"[MemoryManager.prepare_allocation] Evicting {victim.node_name}"
                    )
                buf.free(victim.node_name)
                freed += victim.size
                if freed >= size_bytes:
                    offset = buf.allocate(
                        node.name,
                        size_bytes,
                        self.current_step,
                        node.storage_type,
                        initial_refs,
                    )
                    if offset is not None:
                        break

            if offset is None:
                raise MemoryError(
                    f"OOM: Could not allocate {size_bytes} bytes for {node.name} after eviction."
                )

        return True

    def release(self, node_name: str):
        """Signals that a consumer is done with this node."""
        for buf in self.buffers.values():
            if node_name in buf.allocations:
                block = buf.allocations[node_name]
                if block.storage_type == StorageType.PERSISTENT:
                    return  # Weights stay locked regardless of refs

                if block.ref_count > 0:
                    block.ref_count -= 1
                    if block.ref_count == 0:
                        block.is_locked = False
                        if DEBUG_DETAILED:
                            print(f"[MemoryManager] Unlocked {node_name} (refs=0)")
                return

    def get_view(self, node: TensorNode, use_dirty: bool = False) -> Any:
        device = node.backend.value if node.backend else "cpu"
        if "numpy" in device:
            device = "cpu"
        elif "torch" in device and "cpu" in device:
            device = "cpu"

        if not node.shape or any(d is None for d in node.shape):
            shape = ()
        else:
            shape = tuple(node.shape)

        dirty_region = getattr(node, "dirty_region", None) if use_dirty else None
        return self.buffers[device].get_view(node.name, shape, node.dtype, dirty_region)

    def write(self, node: TensorNode, data: Any):
        view = self.get_view(node)

        if isinstance(view, np.ndarray):
            if hasattr(data, "cpu"):
                data = data.cpu().numpy()
            elif isinstance(data, (int, float)):
                data = np.array(data, dtype=view.dtype)

            if data.shape != view.shape:
                view[...] = data.reshape(view.shape)
            else:
                view[...] = data

        elif hasattr(view, "copy_"):
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
        device = node.backend.value if node.backend else "cpu"
        if "numpy" in device:
            device = "cpu"
        elif "torch" in device and "cpu" in device:
            device = "cpu"
        if device in self.buffers and node.name in self.buffers[device].allocations:
            self.buffers[device].allocations[node.name].is_locked = True
            self.buffers[device].allocations[
                node.name
            ].last_used_step = self.current_step

        if DEBUG:
            print(f"[MemoryManager.lock] {node.name}")

    def unlock(self, node: TensorNode):
        device = node.backend.value if node.backend else "cpu"
        if "numpy" in device:
            device = "cpu"
        elif "torch" in device and "cpu" in device:
            device = "cpu"

        if device in self.buffers and node.name in self.buffers[device].allocations:
            if node.storage_type == StorageType.TRANSIENT:
                self.buffers[device].allocations[node.name].is_locked = False

        if DEBUG:
            print(f"[MemoryManager.unlock] {node.name}")

    def step(self):
        self.current_step += 1
        if DEBUG:
            print(f"[MemoryManager.step] {self.current_step}")
