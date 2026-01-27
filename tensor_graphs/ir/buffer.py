from enum import Enum
from dataclasses import dataclass


class StorageType(Enum):
    TRANSIENT = "transient"  # Activations (recyclable)
    PERSISTENT = "persistent"  # Weights (static)
    STATE = "state"  # KV Cache (persistent but mutable)


@dataclass
class BufferAllocation:
    node_id: str
    device: str  # e.g., "cuda:0" or "cpu"
    storage_type: StorageType
    size_bytes: int
    offset: int = 0  # Assigned later by the Allocator
