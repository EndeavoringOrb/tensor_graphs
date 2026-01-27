from dataclasses import dataclass, field
from typing import List, Dict, Callable, Any, Tuple
from ..ir.buffer import BufferAllocation
from ..ir.dtypes import DType


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
    output_offset: int
    attrs: Dict[str, Any]


@dataclass
class CompiledGraph:
    instructions: List[OpInstruction]
    buffer_allocations: Dict[str, BufferAllocation]  # node_name -> Allocation
    node_metadata: Dict[str, TensorMetadata]
    total_memory_bytes: int
    input_offsets: Dict[str, int]  # name -> offset
    output_offsets: Dict[str, int]  # name -> offset
