"""Regression checks for incidents observed in the production review."""
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from goal_validation import writeGoalReport
from kernel_bench import jobs, native_runner
from kernel_bench.policy import configurationErrors
from kernel_linter import lintCandidate


def testMalformedConstantDetectedBeforeCandidateTesting():
    configuration = {"input_shapes": [[16, 128]], "input_dtypes": [0], "input_constants": [[0] * 4]}
    assert configurationErrors(configuration) == [{"input_index": 0, "expected_constant_bytes": 8192, "actual_constant_bytes": 4}]
    configuration["input_constants"] = [[0] * 8192]
    assert configurationErrors(configuration) == []


def testPlanningReportNeverRunsInferenceOrBenchmark(monkeypatch, tmp_path):
    memory = {"memSpaceIdx": 0, "memSpaceType": "CUDA"}
    graph = {"bucket": {}, "nodeViews": {1: {"shape": [4], "strides": [1]}}, "nodeCosts": {1: float("inf")},
        "instructions": [{"kernelId": 7, "logicalId": 1, "eclassId": 1, "children": [1],
                          "inBuffers": [memory], "outBuffer": memory, "engines": []}]}
    monkeypatch.setattr("utils.binary.loadCacheFile", lambda *args, **kwargs: [
        {"type": "metadata", "bucketWeights": [1]}, {"type": "compiled_bucket", "graph": graph}])
    monkeypatch.chdir(tmp_path)
    def forbidden(*args):
        pytest.fail("Planning executed inference or benchmark")
    session = SimpleNamespace(run_float32=forbidden, benchmark_buckets=forbidden,
                              logical_graph_definition=lambda: {"root": 1})
    evidence = writeGoalReport(session, 1, tmp_path / "cache", tmp_path / "report.json", forbidden, plan_only=True)
    assert evidence["plan_only"] and not evidence["correctness_passed"]
    assert evidence["buckets"][0]["kernel_ids"] == ["7"]
    assert evidence["buckets"][0]["executed_workloads"][0]["estimated_ms"] is None
    json.loads((tmp_path / "report.json").read_text())


def testMissingNativeExportFailsBeforeCalibration(monkeypatch, tmp_path):
    payload = {"action": "preflight", "output_path": str(tmp_path / "output.json")}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    monkeypatch.setitem(sys.modules, "kernel_bench_native", SimpleNamespace())
    with pytest.raises(SystemExit):
        native_runner.main()
    envelope = json.loads((tmp_path / "output.json").read_text())
    assert envelope["error_details"]["kind"] == "incompatible_native_interface"
    assert "kernel_bench_test_diagnostics" in envelope["error_details"]["missing_exports"]


def testInfrastructureFailureStopsRemainingCalibration(monkeypatch, tmp_path):
    calls = []
    store = SimpleNamespace(execute=lambda sql, args: calls.append((sql, args)))
    def incompatible(*args):
        raise jobs.JobFailure("Missing diagnostics export", {"kind": "incompatible_native_interface"})
    monkeypatch.setattr(jobs, "requireNativeInterface", incompatible)
    jobs.calibrateConfiguration(store, {"id": 5, "goal_id": 8, "configuration": {}}, tmp_path)
    assert "c.goal_id = %s" in calls[-1][0]
    assert calls[-1][1][-1] == 8
    assert "Calibration batch stopped" in calls[-1][1][0]


def testRegistrationArityRejectedBeforeBuild(tmp_path):
    path = tmp_path / "candidate.hpp"
    path.write_text('bool matchCandidate(const std::vector<TensorNode>& inputs, const TensorNode& output) { return true; }\n'
        'REGISTER_KERNEL("Candidate", 2, 2, matchCandidate, runCandidate, refCandidate, {}, '
        'MemSpace{1, HandleType::CPP}, {Engine{0, EngineType::CPU}}, {DType::FLOAT32}, {{8}, {8}}, '
        '{true, true}, {MemSpace{1, HandleType::CPP}, MemSpace{1, HandleType::CPP}});\n')
    with pytest.raises(SystemExit):
        lintCandidate(path)


@pytest.mark.skipif(os.name == "nt", reason="systemd updater uses Bash and flock")
def testDeploymentRefusalReportsPathsOnceAndPersistsState(tmp_path):
    script = Path(__file__).resolve().parents[1] / "deploy/kernelbench-update.sh"
    binaries = tmp_path / "bin"
    binaries.mkdir()
    git = binaries / "git"
    git.write_text('''#!/usr/bin/env bash
case "$1" in
 fetch) exit 0 ;;
 rev-parse) if [[ "$2" == origin/main ]]; then echo new; else echo old; fi ;;
 branch) echo main ;;
 status) echo ' M benchmarks/records.bin' ;;
 *) exit 99 ;;
esac
''')
    git.chmod(0o755)
    env = {**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
           "KERNEL_BENCH_CHECKOUT": str(tmp_path), "KERNEL_BENCH_STATE_DIR": str(tmp_path / "state")}
    first = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    second = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert first.returncode == second.returncode == 1
    assert "benchmarks/records.bin" in first.stderr
    assert not second.stderr
    assert (tmp_path / "state/deployment-status").read_text().splitlines()[:2] == ["failed", "refused"]
