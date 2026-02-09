from typing import List, Dict, Tuple, Optional
from ..ir.node import TensorNode
from ..ir.buffer import BufferAllocation, StorageType
from ..ir.dtypes import DType
import math


def get_dtype_size(dtype: DType) -> int:
    if dtype == DType.FP32:
        return 4
    if dtype == DType.INT32:
        return 4
    if dtype == DType.FP16:
        return 2
    if dtype == DType.BOOL:
        return 1
    if dtype == DType.FP8E4M3:
        return 1
    raise ValueError(f"Unknown dtype size: {dtype}")


def calculate_size_bytes(
    shape: Optional[Tuple[Optional[int], ...]], dtype: DType
) -> int:
    if shape is None:
        raise ValueError("Cannot statically allocate buffer for undefined shape.")

    num_elements = 1
    for dim in shape:
        if dim is None:
            raise ValueError("Cannot statically allocate buffer for dynamic shape.")
        num_elements *= dim

    return num_elements * get_dtype_size(dtype)


class MemoryPlanner:
    def __init__(self, alignment: int = 256):
        self.alignment = alignment

    def _align(self, size: int) -> int:
        return math.ceil(size / self.alignment) * self.alignment

    def plan(
        self, nodes: List[TensorNode], liveness: Dict[TensorNode, List[int]]
    ) -> Dict[TensorNode, BufferAllocation]:
        allocations = {}
        persistent_offsets: Dict[str, int] = {}

        # 1. Separate Persistent vs Transient
        for node in nodes:
            if node.storage_type == StorageType.PERSISTENT:
                device = node.backend.value if node.backend else "cpu"
                size = calculate_size_bytes(node.shape, node.dtype)
                size_aligned = self._align(size)

                current_offset = persistent_offsets.get(device, 0)

                allocations[node] = BufferAllocation(
                    node_id=node.name,
                    device=device,
                    storage_type=node.storage_type,
                    size_bytes=size,
                    offset=current_offset,
                )
                persistent_offsets[device] = current_offset + size_aligned

        # 2. Allocate Transient (Greedy First-Fit with Reuse)
        # We model memory as intervals on a timeline.

        # Active allocations per device: device -> List[(offset, end_offset, death_time)]
        active_allocations: Dict[str, List[Tuple[int, int, int]]] = {}

        # Sort nodes by birth time (which is just their index in topo sort)
        sorted_nodes = sorted(nodes, key=lambda n: liveness[n][0])

        for node in sorted_nodes:
            if node.storage_type != StorageType.TRANSIENT:
                continue

            current_time = liveness[node][0]
            death_time = liveness[node][1]
            device = node.backend.value if node.backend else "cpu"
            size = calculate_size_bytes(node.shape, node.dtype)
            size_aligned = self._align(size)

            # Initialize device tracker if needed
            if device not in active_allocations:
                active_allocations[device] = []

            # 1. Expire dead allocations for this device
            active_allocations[device] = [
                a for a in active_allocations[device] if a[2] >= current_time
            ]

            # 2. Find hole
            used_intervals = sorted([(a[0], a[1]) for a in active_allocations[device]])

            best_offset = -1
            candidate_start = 0

            base_offset = persistent_offsets.get(device, 0)
            # Actually, if we use a single buffer, we should offset everything by base_offset?
            # Or just start searching from base_offset.

            candidate_start = max(candidate_start, base_offset)

            for start, end in used_intervals:
                # Need to handle the case where used_intervals are "after" the candidate_start
                # But since we sorted used_intervals, and they are > base_offset (presumably),
                # We check gaps.

                # If the gap between candidate and next usage is big enough
                if start - candidate_start >= size_aligned:
                    best_offset = candidate_start
                    break
                candidate_start = max(candidate_start, end)

            if best_offset == -1:
                best_offset = candidate_start # Couldn't find a gap, put it at the end

            # 3. Alloc
            allocations[node] = BufferAllocation(
                node_id=node.name,
                device=device,
                storage_type=StorageType.TRANSIENT,
                size_bytes=size,
                offset=best_offset,
            )

            active_allocations[device].append(
                (best_offset, best_offset + size_aligned, death_time)
            )

        return allocations
