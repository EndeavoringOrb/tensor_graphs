from dataclasses import dataclass
from typing import List, Dict, Callable, Any, Tuple
from ..ir.buffer import BufferAllocation
from ..ir.dtypes import DType
from ..ir.node import TensorNode


@dataclass
class TensorMetadata:
    shape: Tuple[int, ...]
    dtype: DType


@dataclass
class OpInstruction:
    node_name: str
    kernel: Callable
    input_offsets: List[int]
    input_node_names: List[str]
    output_offsets: List[int]
    attrs: Dict[str, Any]


@dataclass
class CompiledGraph:
    instructions: List[OpInstruction]
    buffer_allocations: Dict[str, BufferAllocation]
    node_metadata: Dict[str, TensorMetadata]
    total_memory_bytes: int
    input_offsets: Dict[str, int]
    output_offsets: Dict[str, int]
    # Map back to original nodes for runtime state management (caching, dirty flags)
    nodes_map: Dict[str, TensorNode]
