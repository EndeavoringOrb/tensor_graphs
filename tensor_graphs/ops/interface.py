from abc import ABC, abstractmethod
from typing import List, Tuple, ClassVar
from ..ir.node import TensorNode
from ..ir.dtypes import DType


class CompositeOp(ABC):
    """
    Abstract Base Class for Composite Operations.
    These are operations that can be decomposed into a graph of atomic operations.
    """

    op_type: ClassVar[str]

    @abstractmethod
    def decompose(self, inputs: List[TensorNode]) -> TensorNode:
        """
        Returns the output node of a graph constructed entirely
        of Atomic ops (or other Composite ops).
        """
        pass