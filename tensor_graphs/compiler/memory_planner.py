from typing import List, Dict, Tuple, Optional
from ..ir.node import TensorNode
from ..ir.buffer import BufferAllocation, StorageType
from ..ir.dtypes import DType
from ..ops.atomic_types import OpType
import math


def get_dtype_size(dtype: DType) -> int:
    sizes = {
        DType.FP32: 4,
        DType.INT32: 4,
        DType.FP16: 2,
        DType.BOOL: 1,
        DType.FP8E4M3: 1,
    }
    if dtype not in sizes:
        raise ValueError(f"Unknown dtype size: {dtype}")
    return sizes[dtype]


def calculate_size_bytes(
    shape: Optional[Tuple[Optional[int], ...]], dtype: DType
) -> int:
    if shape is None or any(d is None for d in shape):
        raise ValueError(
            "Cannot statically allocate buffer for undefined/dynamic shape."
        )
    return math.prod(shape) * get_dtype_size(dtype)


class MemoryPlanner:
    def __init__(self, alignment: int = 256):
        self.alignment = alignment

    def _align(self, size: int) -> int:
        return math.ceil(size / self.alignment) * self.alignment

    def _analyze_liveness(self, nodes: List[TensorNode]) -> Dict[TensorNode, List[int]]:
        intervals = {node: [i, i] for i, node in enumerate(nodes)}
        has_children = set()

        for i, node in enumerate(nodes):
            if node.op_type in (OpType.INPUT, OpType.CONSTANT):
                intervals[node][0] = 0
            for parent in node.parents:
                if parent in intervals:
                    intervals[parent][1] = max(intervals[parent][1], i)
                has_children.add(parent)

        last_step = len(nodes) - 1
        for node in nodes:
            if node not in has_children:
                intervals[node][1] = last_step

        return intervals

    def plan(self, nodes: List[TensorNode]) -> Dict[TensorNode, BufferAllocation]:
        liveness = self._analyze_liveness(nodes)
        allocations = {}
        persistent_offsets: Dict[str, int] = {}
        active_allocations: Dict[str, List[Tuple[int, int, int]]] = {}

        # Allocate persistent buffers first
        for node in nodes:
            if node.storage_type == StorageType.PERSISTENT:
                device = node.backend.value if node.backend else "cpu"
                size = calculate_size_bytes(node.shape, node.dtype)
                offset = persistent_offsets.get(device, 0)

                allocations[node] = BufferAllocation(
                    node_id=node.name,
                    device=device,
                    storage_type=node.storage_type,
                    size_bytes=size,
                    offset=offset,
                )
                persistent_offsets[device] = offset + self._align(size)

        # Allocate transient buffers with reuse
        for node in sorted(nodes, key=lambda n: liveness[n][0]):
            if node.storage_type != StorageType.TRANSIENT:
                continue

            birth, death = liveness[node]
            device = node.backend.value if node.backend else "cpu"
            size_aligned = self._align(calculate_size_bytes(node.shape, node.dtype))

            if device not in active_allocations:
                active_allocations[device] = []

            # Expire dead allocations
            active_allocations[device] = [
                a for a in active_allocations[device] if a[2] >= birth
            ]

            # Find first-fit hole
            used = sorted((a[0], a[1]) for a in active_allocations[device])
            candidate = persistent_offsets.get(device, 0)

            for start, end in used:
                if start - candidate >= size_aligned:
                    break
                candidate = max(candidate, end)

            allocations[node] = BufferAllocation(
                node_id=node.name,
                device=device,
                storage_type=StorageType.TRANSIENT,
                size_bytes=size_aligned,
                offset=candidate,
            )
            active_allocations[device].append(
                (candidate, candidate + size_aligned, death)
            )

        return allocations
