from dataclasses import dataclass
from typing import List, Dict, Callable, Any
from ..ir.node import TensorNode


@dataclass
class OpInstruction:
    node_name: str
    kernel: Callable
    input_node_names: List[str]
    attrs: Dict[str, Any]


@dataclass
class CompiledGraph:
    instructions: List[OpInstruction]
    # Liveness: Map node_name -> number of consumers
    ref_counts: Dict[str, int]
    nodes_map: Dict[str, TensorNode]
