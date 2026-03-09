from enum import Enum


class StorageType(Enum):
    TRANSIENT = "transient"  # Activations (recyclable)
    PERSISTENT = "persistent"  # Weights (static)
