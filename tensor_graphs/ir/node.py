from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import uuid

@dataclass
class TensorNode:
    op_type: str                  # e.g., "Add", "Mul", "Dot", "Input"
    shape: Tuple[int, ...]        # e.g., (32, 128)
    parents: List['TensorNode']   # Dependency edges
    name: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Metadata for the future LLM compiler
    dtype: str = "float32"
    
    def __repr__(self):
        parent_names = [p.name for p in self.parents]
        return f"Node({self.name}, op={self.op_type}, parents={parent_names})"

    def __hash__(self):
        # Use object identity for hashing in DAGs
        return id(self)