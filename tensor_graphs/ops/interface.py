from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, ClassVar, Tuple
from ..ir.node import TensorNode
import numpy as np


class CompositeOp(ABC):
    """
    Abstract Base Class for Composite Operations.
    These are operations that can be decomposed into a graph of atomic operations.
    """

    op_type: ClassVar[str]

    @abstractmethod
    def decompose(
        self, inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
    ) -> TensorNode:
        """
        Returns the output node of a graph constructed entirely
        of Atomic ops (or other Composite ops).
        """
        pass

    def sample_inputs(self) -> List[Tuple[List[np.ndarray], Dict[str, Any]]]:
        """
        Optional: Returns a list of (inputs_list, attrs_dict) tuples for verification.
        Used to verify that kernels matching this OpType produce results
        consistent with the decomposed graph.
        """
        return []
