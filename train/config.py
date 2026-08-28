# train/config.py
import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class TrainConfig:
    run_dir: str = "runs/0"
    host: str = "127.0.0.1"
    port: int = 5000

    # Model Architecture
    hidden_dim: int = 64

    # Replay Buffer & Eviction Strategy ("fifo", "lowest_loss", "highest_cost")
    buffer_strategy: str = "lowest_loss"
    buffer_size: int = 50_000
    batch_size: int = 64
    lr: float = 1e-3
    save_interval: int = 100

    # Exploration
    epsilon: float = 0.1

    # Worker / Graph Provider Config
    workers: int = 4
    cpp_threads: int = 1
    graph_source: str = "model"  # "model" or "random"
    model_name: str = "gemma-3-270m"
    model_path: str = "models/google/gemma-3-270m"
    seq_len: int = 128

    # Random Graph Generator Settings
    random_min_nodes: int = 10
    random_max_nodes: int = 100
    random_dim: int = 128
    random_seq_len: int = 64
    random_seed: int | None = None
    resample_graph_every: int = 1

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid_fields})

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=4), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
