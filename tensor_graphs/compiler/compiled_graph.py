from dataclasses import dataclass, field
from typing import List, Dict, Callable, Any, Optional
from ..ir.node import TensorNode
from ..ir.dtypes import Backend
from ..ir.graph import graph_to_json
from ..backend.registry import KernelRegistry


@dataclass
class OpInstruction:
    node_name: str
    kernel: Callable
    input_node_names: List[str]
    attrs: Dict[str, Any]
    # New field for in-place support
    inplace_input_index: Optional[int] = None


@dataclass
class SerializedInstruction:
    """Serialized form of OpInstruction for caching."""

    node_name: str
    op_type: str
    backend: str
    input_node_names: List[str]
    attrs: Dict[str, Any]
    inplace_input_index: Optional[int] = None


@dataclass
class CompiledGraph:
    instructions: List[OpInstruction]
    # Liveness: Map node_name -> number of consumers
    ref_counts: Dict[str, int]
    nodes_map: Dict[str, TensorNode]
    # Cache for instruction lookup
    _instructions_map: Dict[str, OpInstruction] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self):
        if not self._instructions_map:
            for instr in self.instructions:
                self._instructions_map[instr.node_name] = instr

    def get_instruction(self, node_name: str) -> Optional[OpInstruction]:
        return self._instructions_map.get(node_name)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize compiled graph to a dictionary for caching."""
        # Serialize instructions (kernel metadata only, not the callable)
        serialized_instructions = []
        for instr in self.instructions:
            # Get kernel metadata
            op_type = None
            backend_str = None
            for node_name, node in self.nodes_map.items():
                if node_name == instr.node_name:
                    op_type = node.op_type
                    backend_str = node.backend.value if node.backend else None
                    break

            serialized_instr = SerializedInstruction(
                node_name=instr.node_name,
                op_type=op_type or "UNKNOWN",
                backend=backend_str or Backend.CPU_NUMPY.value,
                input_node_names=instr.input_node_names,
                attrs=instr.attrs,
                inplace_input_index=instr.inplace_input_index,
            )
            serialized_instructions.append(serialized_instr)

        return {
            "instructions": [
                {
                    "node_name": si.node_name,
                    "op_type": si.op_type,
                    "backend": si.backend,
                    "input_node_names": si.input_node_names,
                    "attrs": si.attrs,
                    "inplace_input_index": si.inplace_input_index,
                }
                for si in serialized_instructions
            ],
            "ref_counts": self.ref_counts,
            "nodes_json": graph_to_json(self.root),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompiledGraph":
        """Reconstruct compiled graph from a dictionary."""
        from ..ir.graph import graph_from_json

        # Reconstruct nodes from JSON
        root = graph_from_json(data["nodes_json"])
        nodes_map = {}
        all_nodes = []
        from ..ir.graph import topological_sort

        for node in topological_sort(root):
            nodes_map[node.name] = node
            all_nodes.append(node)

        # Reconstruct instructions with kernels from registry
        instructions = []
        for instr_data in data["instructions"]:
            node_name = instr_data["node_name"]
            op_type = instr_data["op_type"]
            backend = Backend(instr_data["backend"])
            input_node_names = instr_data["input_node_names"]
            attrs = instr_data["attrs"]
            inplace_input_index = instr_data.get("inplace_input_index")

            # Look up the kernel from registry
            input_sigs = []
            for parent_name in input_node_names:
                if parent_name in nodes_map:
                    input_sigs.append(nodes_map[parent_name].signature)

            # Get target dtype from the output node
            target_dtype = None
            if node_name in nodes_map:
                target_dtype = nodes_map[node_name].dtype

            # Determine if inplace is allowed
            allow_inplace = inplace_input_index is not None

            kernel_result = KernelRegistry.select_best_kernel(
                op_type,
                input_sigs,
                backend,
                target_dtype=target_dtype,
                allow_inplace=allow_inplace,
            )

            if not kernel_result:
                raise RuntimeError(
                    f"Could not find kernel for {op_type} on {backend} during deserialization"
                )

            kernel, _ = kernel_result

            instructions.append(
                OpInstruction(
                    node_name=node_name,
                    kernel=kernel,
                    input_node_names=input_node_names,
                    attrs=attrs,
                    inplace_input_index=inplace_input_index,
                )
            )

        ref_counts = data["ref_counts"]

        return cls(
            instructions=instructions,
            ref_counts=ref_counts,
            nodes_map=nodes_map,
        )

    @property
    def root(self) -> TensorNode:
        """Get the root node of the graph."""
        # The root is typically the node with no consumers
        # Find the node that's not a parent of any other node in our graph
        all_children = set()
        for node in self.nodes_map.values():
            for parent in node.parents:
                all_children.add(parent.name)

        for name, node in self.nodes_map.items():
            if name not in all_children:
                return node

        # Fallback: return any node (shouldn't happen for valid graphs)
        return list(self.nodes_map.values())[0]
