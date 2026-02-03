from typing import List, Dict, Any, Optional, Tuple, cast
import numpy as np
from ..ir.node import TensorNode
from ..ir.dtypes import DType, Backend, TensorSignature
from ..ops.atomic_types import OpType
from ..ir.graph import topological_sort
from ..backend.registry import KernelRegistry
from .shape_inference import ShapeInference


class ConstantFolding:
    """
    Evaluates constant subgraphs and replaces them with a single OpType.CONSTANT node.
    Uses registered CPU kernels to perform the evaluation, ensuring consistency
    with runtime execution and supporting all registered operations (including Cast).
    """

    @classmethod
    def fold(cls, root: TensorNode, known_values: Dict[str, Any]) -> TensorNode:
        """
        Traverses the graph and folds constants.
        Returns the new root node.
        """
        # 1. Topological Sort
        nodes = topological_sort(root)

        # Map old nodes to new nodes (for parent references)
        node_map = {}

        # 2. Process nodes
        for node in nodes:
            # Re-link parents to their new versions (folded or not)
            if node.parents:
                node.parents = [node_map[p] for p in node.parents]

            if node.op_type == OpType.INPUT or node.op_type == OpType.CONSTANT:
                node_map[node] = node
                continue

            # Check if all parents are constants
            parent_values = []
            all_parents_constant = True

            for p_mapped in node.parents:
                if p_mapped.op_type == OpType.CONSTANT:
                    val = p_mapped.attrs.get("value")
                    if val is None:
                        all_parents_constant = False
                        break
                    parent_values.append(val)
                else:
                    all_parents_constant = False
                    break

            if all_parents_constant and (len(parent_values) == len(node.parents)):
                # Try to fold using kernels
                folded_node = cls._try_fold_node(node, parent_values, known_values)
                if folded_node:
                    node_map[node] = folded_node
                else:
                    # Folding failed (no kernel, shape inference failed, etc.)
                    # Keep original node
                    node_map[node] = node
            else:
                # Keep the node as is
                node_map[node] = node

        # 3. Return the replacement for the original root
        return node_map[root]

    @classmethod
    def _try_fold_node(
        cls, node: TensorNode, parent_values: List[Any], known_values: Dict[str, Any]
    ) -> Optional[TensorNode]:
        """
        Attempts to evaluate the node using CPU kernels.
        Returns a new Constant node if successful, else None.
        """
        # 1. Infer Shape
        # Ensure parents have concrete shapes before attempting inference
        if any(p.shape is None for p in node.parents):
            return None

        ShapeInference.infer([node], known_values)

        if node.shape is None or any(d is None for d in node.shape):
            return None

        # 2. Select Kernel
        # Create input signatures based on parent nodes
        input_sigs = []
        for p in node.parents:
            # Parents are constants, so they should have concrete dtype/shape/backend
            # For folding, we treat them as CPU_NUMPY
            input_sigs.append(TensorSignature(p.dtype, p.shape, Backend.CPU_NUMPY))

        kernel = KernelRegistry.select_best_kernel(
            node.op_type, input_sigs, Backend.CPU_NUMPY, target_dtype=node.dtype
        )

        if not kernel:
            return None

        # 3. Allocate Output
        np_dtype = cls._map_dtype_np(node.dtype)
        try:
            # Type narrowing: We verified above that node.shape and its dims are not None
            # Cast to Tuple[int, ...] to satisfy static analysis for the generator expression below
            concrete_shape = cast(Tuple[int, ...], node.shape)
            shape = tuple(int(d) for d in concrete_shape)
            output_array = np.zeros(shape, dtype=np_dtype)
        except Exception:
            return None

        # 4. Prepare Inputs
        # Ensure inputs are numpy arrays (Constant values might be scalars/lists)
        kernel_inputs = []
        for val in parent_values:
            if not isinstance(val, np.ndarray):
                kernel_inputs.append(np.array(val))
            else:
                kernel_inputs.append(val)

        # 5. Run Kernel
        try:
            # Kernels signature: (inputs, outputs, attrs)
            kernel(kernel_inputs, [output_array], node.attrs)
        except Exception:
            return None

        # 6. Create new Constant Node
        new_node = TensorNode(
            OpType.CONSTANT,
            node.dtype,
            [],  # No parents for constants
            name=f"folded_{node.name}",
            attrs={"value": output_array},
            backend=node.backend,
        )
        new_node.shape = node.shape
        return new_node

    @staticmethod
    def _map_dtype_np(dtype: DType) -> Any:
        if dtype == DType.FP32:
            return np.float32
        if dtype == DType.INT32:
            return np.int32
        if dtype == DType.FP16:
            return np.float16
        if dtype == DType.BOOL:
            return np.bool_
        return np.float32
