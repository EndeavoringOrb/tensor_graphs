"""Shared storage and metadata helpers for the GNN tools."""

import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

SAMPLES_ROOT = Path(__file__).resolve().parent / "samples"
PROBLEM_FILE = "problem.json"
SAMPLES_FILE = "samples.jsonl"
PROBLEM_HASH_FILE = "problem_hash"
CONFIG_FILE = "config.json"


@dataclass
class SampleGenerationConfig:
    """Configuration persisted with a generated sample run."""

    sample_count: int = 500
    neighborhood_size: int = 15
    max_neighborhood_size: int = 64
    repair_time: float = 90.0
    seed: int = 0

    def toDict(self):
        values = asdict(self)
        # Keep the command-line spelling in the persisted file while retaining
        # the explicit field name used by Python callers.
        values["samples"] = values["sample_count"]
        return values


def problemHash(problem_data):
    encoded = json.dumps(
        problem_data, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def metadataGraph(metadata):
    """Return stable node keys, features, and edges for a selector graph."""
    groups = metadata.get("groups", {})
    keys = sorted(groups)
    feature_dim = int(metadata.get("feature_dim", 0))
    features = []
    for key in keys:
        values = list(groups[key].get("features", []))[:feature_dim]
        values.extend([0.0] * (feature_dim - len(values)))
        features.append(
            [
                0.0 if not math.isfinite(float(value)) else float(value)
                for value in values
            ]
        )
    key_indices = {key: index for index, key in enumerate(keys)}
    edges = [
        [key_indices[source], key_indices[target]]
        for source in keys
        for target in metadata.get("adjacency", {}).get(source, ())
        if target in key_indices
    ]
    return keys, features, edges


def loadProblem(path):
    with open(path, "r", encoding="utf-8") as handle:
        problem_data = json.load(handle)
    if not isinstance(problem_data, dict):
        raise TypeError("The problem file must contain a JSON object")
    return problem_data


def readJson(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def writeJson(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def nextRunDirectory(samples_root=SAMPLES_ROOT):
    samples_root = Path(samples_root)
    samples_root.mkdir(parents=True, exist_ok=True)
    indices = [
        int(path.name)
        for path in samples_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    run_directory = samples_root / str(max(indices, default=-1) + 1)
    run_directory.mkdir()
    return run_directory


def resolveRunDirectory(samples_root, resume):
    samples_root = Path(samples_root)
    if resume is None:
        return nextRunDirectory(samples_root)
    if resume == "latest":
        indices = (
            [
                int(path.name)
                for path in samples_root.iterdir()
                if path.is_dir() and path.name.isdigit()
            ]
            if samples_root.exists()
            else []
        )
        if not indices:
            raise FileNotFoundError(f"No sample runs found under {samples_root}")
        index = max(indices)
    else:
        try:
            index = int(resume)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "--resume must be omitted, 'latest', or a numeric run index"
            ) from error
    run_directory = samples_root / str(index)
    if not run_directory.is_dir():
        raise FileNotFoundError(f"Sample run does not exist: {run_directory}")
    return run_directory


def initializeRun(problem_path, run_directory, config):
    """Create the immutable problem and persisted configuration for a run."""
    run_directory = Path(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    problem_data = loadProblem(problem_path)
    shutil.copyfile(problem_path, run_directory / PROBLEM_FILE)
    (run_directory / PROBLEM_HASH_FILE).write_text(
        problemHash(problem_data) + "\n", encoding="utf-8"
    )
    (run_directory / SAMPLES_FILE).touch()
    writeJson(run_directory / CONFIG_FILE, config.toDict())
    return problem_data


def loadJsonl(path):
    samples = []
    path = Path(path)
    if not path.exists():
        return samples
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} line {line_number}"
                ) from error
            if not isinstance(sample, dict):
                raise TypeError(f"Sample {line_number} in {path} is not an object")
            samples.append(sample)
    return samples


def compactSample(sample):
    """Remove graph data repeated by every sample in a run."""
    neighborhood = sample.get(
        "neighborhood_indices",
        sample.get("target_indices", sample.get("neighborhood", [])),
    )
    compact = {
        "neighborhood_indices": [int(index) for index in neighborhood],
        "current_runtime": float(sample["current_runtime"]),
        "solution_runtime": float(sample["solution_runtime"]),
        "relative_runtime": float(
            sample.get(
                "relative_runtime",
                sample.get("runtime_ratio", sample.get("target", 1.0)),
            )
        ),
    }
    return compact


class SampleWriter:
    """Append one compact JSONL sample at a time."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = None

    def __enter__(self):
        self.handle = self.path.open("a", encoding="utf-8", buffering=1)
        return self

    def append(self, sample):
        if self.handle is None:
            raise RuntimeError("SampleWriter must be used as a context manager")
        self.handle.write(
            json.dumps(compactSample(sample), separators=(",", ":")) + "\n"
        )
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def loadRun(run_or_samples):
    """Load samples and the copied problem from a run directory or JSONL file."""
    path = Path(run_or_samples)
    run_directory = path if path.is_dir() else path.parent
    samples_path = run_directory / SAMPLES_FILE if path.is_dir() else path
    samples = (
        loadJsonl(samples_path) if samples_path.suffix.lower() == ".jsonl" else None
    )
    if samples is None:
        value = readJson(samples_path)
        if isinstance(value, list):
            samples = value
        elif isinstance(value, dict) and isinstance(value.get("problems"), dict):
            samples = []
            for problem in value["problems"].values():
                samples.extend(
                    problem if isinstance(problem, list) else problem.get("samples", [])
                )
        elif isinstance(value, dict):
            samples = value.get("samples", [])
        else:
            samples = []
    problem_path = run_directory / PROBLEM_FILE
    problem_data = loadProblem(problem_path) if problem_path.exists() else None
    return samples, problem_data
