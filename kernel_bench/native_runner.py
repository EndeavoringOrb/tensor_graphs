"""Fresh process per rebuilt extension; C++ crashes cannot kill the API server."""
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
if __package__:
    from .policy import baselineEligible
else:
    from policy import baselineEligible

native = None


class NativeFailure(RuntimeError):
    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


def checkpoint(payload, state):
    path = Path(payload["output_path"])
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, allow_nan=False))
    temporary.replace(path)


def evaluate(payload, state):
    state["error_details"] = {"stage": "registration"}
    checkpoint(payload, state)
    capabilities = native.kernel_bench_capabilities()
    state["error_details"]["enabled_capabilities"] = capabilities
    checkpoint(payload, state)
    kernels = native.kernel_bench_kernels()
    candidates = [kernel for kernel in kernels if kernel["name"] == payload["kernel_name"]]
    if len(candidates) != 1:
        raise NativeFailure(
            f"Expected exactly one registered candidate named '{payload['kernel_name']}'; found {len(candidates)}. "
            "Check capability guards and REGISTER_KERNEL reachability.",
            {"kind": "registration_error", "stage": "registration", "submitted_name": payload["kernel_name"],
             "observed_match_count": len(candidates), "enabled_capabilities": capabilities, "registered_names": [kernel["name"] for kernel in kernels]})
    candidate = candidates[0]
    if candidate["reference_expression"] != payload["reference_expression"]:
        raise NativeFailure("Candidate reference expression differs from the requested target",
                            {"kind": "registration_error", "stage": "registration", "submitted_name": payload["kernel_name"],
                             "observed_match_count": len(candidates), "expected_target": payload["reference_expression"],
                             "actual_target": candidate["reference_expression"], "enabled_capabilities": capabilities})
    if not candidate["has_reference_factory"] or candidate["is_reference"] or candidate["is_view"]:
        raise NativeFailure("Candidate must be an executable kernel with a reference factory",
                            {"kind": "registration_error", "stage": "registration", "candidate": candidate,
                             "submitted_name": payload["kernel_name"], "observed_match_count": len(candidates),
                             "enabled_capabilities": capabilities})
    if payload["action"] == "compile":
        return {"candidate": candidate, "measurements": []}
    result = state["result"]
    result.update({"matched_configuration_ids": [], "unmatched_configuration_ids": [], "unmatched_configurations": []})
    matching = []
    for row in payload["configurations"]:
        state["error_details"] = {"stage": "match", "configuration_id": row["id"]}
        checkpoint(payload, state)
        diagnostic = native.kernel_bench_match_diagnostics(candidate["kernel_id"], row["configuration"])
        if diagnostic["matched"]:
            matching.append(row)
            result["matched_configuration_ids"].append(row["id"])
        else:
            result["unmatched_configuration_ids"].append(row["id"])
            result["unmatched_configurations"].append({"configuration_id": row["id"], "reasons": diagnostic["reasons"]})
        checkpoint(payload, state)
    if not matching:
        raise NativeFailure("Candidate matches zero goal configurations", {"kind": "no_matching_configurations", "stage": "match"})
    result = state["result"]
    result.update({"candidate": candidate, "tested_configuration_ids": [], "correctness_results": [],
                   "tested_configurations": 0, "total_configurations": len(payload["configurations"]),
                   "measurements": [], "comparisons": []})
    for row in matching:
        state["error_details"] = {"stage": "test", "configuration_id": row["id"]}
        checkpoint(payload, state)
        print(f"Testing configuration {row['id']}", flush=True)
        diagnostic = native.kernel_bench_test_diagnostics(candidate["kernel_id"], row["configuration"], 42)
        result["correctness_results"].append({"configuration_id": row["id"], **diagnostic})
        if not diagnostic["passed"]:
            raise NativeFailure(f"KernelBench correctness test failed in configuration {row['id']}",
                                {**diagnostic, "stage": "test", "configuration_id": row["id"]})
        result["tested_configuration_ids"].append(row["id"])
        result["tested_configurations"] += 1
        checkpoint(payload, state)
    measurements = result["measurements"]
    comparisons = result["comparisons"]
    for row in matching:
        state["error_details"] = {"stage": "benchmark", "configuration_id": row["id"]}
        checkpoint(payload, state)
        configuration = row["configuration"]
        competitors = [kernel for kernel in kernels
                       if kernel["kernel_id"] != candidate["kernel_id"]
                       and kernel["reference_expression"] == payload["reference_expression"]
                       and baselineEligible(kernel)
                       and native.kernel_bench_matches(kernel["kernel_id"], configuration)]
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
        checkpoint(payload, state)
    return result


def calibrate(payload, state):
    """Same correctness seed, benchmark rounds and competitor policy as evaluation."""
    row = payload["configuration"]
    configuration = row["configuration"]
    kernels = native.kernel_bench_kernels()
    eligible = [kernel for kernel in kernels
                if kernel["reference_expression"] == payload["reference_expression"]
                and baselineEligible(kernel)
                and native.kernel_bench_matches(kernel["kernel_id"], configuration)]
    ordered = sorted(eligible, key=lambda kernel: kernel["kernel_id"])
    if not ordered:
        raise NativeFailure("No eligible baseline kernels", {
            "kind": "no_matching_kernels", "stage": "baseline", "configuration_id": row["id"],
            "reference_expression": payload["reference_expression"],
            "target_kernels": [{**kernel, "matches_configuration": native.kernel_bench_matches(
                kernel["kernel_id"], configuration)} for kernel in kernels
                if kernel["reference_expression"] == payload["reference_expression"]]})
    for kernel in ordered:
        state["error_details"] = {"stage": "test", "configuration_id": row["id"], "kernel_name": kernel["name"]}
        checkpoint(payload, state)
        diagnostic = native.kernel_bench_test_diagnostics(kernel["kernel_id"], configuration, 42)
        if not diagnostic["passed"]:
            raise NativeFailure("Baseline correctness failed", {**state["error_details"], **diagnostic})
    samples = {kernel["kernel_id"]: [] for kernel in ordered}
    for round_idx in range(2):
        for kernel in ordered if round_idx == 0 else reversed(ordered):
            state["error_details"] = {"stage": "benchmark", "configuration_id": row["id"], "kernel_name": kernel["name"]}
            checkpoint(payload, state)
            samples[kernel["kernel_id"]].extend(native.kernel_bench_benchmark(
                kernel["kernel_id"], configuration, payload.get("iterations", 8)))
    measurements = []
    for kernel in ordered:
        values = samples[kernel["kernel_id"]]
        if not values or not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("Baseline benchmark returned invalid samples")
        measurements.append({"kernel_id": kernel["kernel_id"], "kernel_name": kernel["name"],
                             "samples_ms": values, "median_ms": statistics.median(values)})
    return {"measurements": measurements, "winner": min(measurements, key=lambda value: value["median_ms"]),
            "enabled_capabilities": native.kernel_bench_capabilities()}


def main():
    global native
    payload = json.load(sys.stdin)
    state = {"result": {}, "error_details": {"stage": "load"}}
    checkpoint(payload, state)
    try:
        import kernel_bench_native as native_module
        native = native_module
        if payload["action"] == "discover":
            state["result"] = {"kernels": native.kernel_bench_kernels(), "calls": native.kernel_bench_discover(payload["cache_path"]), "enabled_capabilities": native.kernel_bench_capabilities()}
        elif payload["action"] == "baseline":
            state["result"] = calibrate(payload, state)
        elif payload["action"] in ("evaluate", "compile"):
            state["result"] = evaluate(payload, state)
        else:
            raise ValueError("Unknown native runner action")
        state.pop("error_details", None)
        checkpoint(payload, state)
    except Exception as error:  # noqa: BLE001 - persist native exceptions before exiting the child process
        state["error"] = str(error)
        state.setdefault("error_details", {"stage": "native"}).update(getattr(error, "details", {}))
        state["error_details"].setdefault("kind", "native_exception")
        checkpoint(payload, state)
        print(str(error), file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
