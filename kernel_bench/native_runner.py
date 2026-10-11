"""Fresh process per rebuilt extension; C++ crashes cannot kill the API server."""
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import kernel_bench_native as native


def evaluate(payload):
    kernels = native.kernel_bench_kernels()
    candidates = [kernel for kernel in kernels if kernel["name"] == payload["kernel_name"]]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one registered candidate with the submitted name")
    candidate = candidates[0]
    if candidate["reference_expression"] != payload["reference_expression"]:
        raise ValueError("Candidate reference expression differs from the requested target")
    if not candidate["has_reference_factory"] or candidate["is_reference"] or candidate["is_view"]:
        raise ValueError("Candidate must be an executable kernel with a reference factory")
    matching = [row for row in payload["configurations"]
                if native.kernel_bench_matches(candidate["kernel_id"], row["configuration"])]
    if not matching:
        raise ValueError("Candidate matches zero goal configurations")
    for row in matching:
        print(f"Testing configuration {row['id']}", flush=True)
        native.kernel_bench_test(candidate["kernel_id"], row["configuration"])
    measurements = []
    comparisons = []
    for row in matching:
        configuration = row["configuration"]
        competitors = [kernel for kernel in kernels
                       if kernel["kernel_id"] != candidate["kernel_id"]
                       and kernel["reference_expression"] == payload["reference_expression"]
                       and not kernel["is_view"]
                       and native.kernel_bench_matches(kernel["kernel_id"], configuration)]
        optimized = [kernel for kernel in competitors if not kernel["is_reference"]]
        competitors = optimized or competitors
        ordered = [candidate] + sorted(competitors, key=lambda kernel: kernel["kernel_id"])
        samples = {kernel["kernel_id"]: [] for kernel in ordered}
        for round_idx in range(2):
            for kernel in ordered if round_idx == 0 else reversed(ordered):
                print(f"Benchmarking {kernel['name']} configuration {row['id']}", flush=True)
                samples[kernel["kernel_id"]].extend(native.kernel_bench_benchmark(
                    kernel["kernel_id"], configuration, payload.get("iterations", 8)))
        medians = {}
        for kernel in ordered:
            values = samples[kernel["kernel_id"]]
            if not all(math.isfinite(value) and value > 0 for value in values):
                raise ValueError("Benchmark returned invalid timing samples")
            median_ms = statistics.median(values)
            medians[kernel["kernel_id"]] = median_ms
            measurements.append({"configuration_id": row["id"], "kernel_id": kernel["kernel_id"],
                                 "kernel_name": kernel["name"], "is_candidate": kernel == candidate,
                                 "samples_ms": values, "median_ms": median_ms})
        best = min(competitors, key=lambda kernel: medians[kernel["kernel_id"]]) if competitors else None
        comparisons.append({"configuration_id": row["id"], "candidate_ms": medians[candidate["kernel_id"]],
                            "best_existing_kernel": best["name"] if best else None,
                            "best_existing_ms": medians[best["kernel_id"]] if best else None,
                            "speedup": medians[best["kernel_id"]] / medians[candidate["kernel_id"]] if best else None})
    return {"candidate": candidate, "tested_configurations": len(matching),
            "total_configurations": len(payload["configurations"]),
            "measurements": measurements, "comparisons": comparisons}


def main():
    payload = json.load(sys.stdin)
    if payload["action"] == "discover":
        result = {"kernels": native.kernel_bench_kernels(), "calls": native.kernel_bench_discover(payload["cache_path"])}
    elif payload["action"] == "evaluate":
        result = evaluate(payload)
    else:
        raise ValueError("Unknown native runner action")
    Path(payload["output_path"]).write_text(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
