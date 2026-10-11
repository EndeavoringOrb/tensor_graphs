"""Failure contracts and artifact access without a GPU or database service."""
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_bench import native_runner
from kernel_bench.app import createApp
from kernel_bench.jobs import JobFailure, evaluateJob, readableOutput, runCmd, runNative


def testSubprocessDrainsBothStreamsAndPreservesDiagnostics(tmp_path):
    script = "import sys; sys.stdout.write('x' * 200000 + '\\rlinting [1/2]: 25%\\r'); sys.stderr.write('y' * 200000 + '\\nerror: std::vector<size_t> [0]\\n'); sys.exit(2)"
    result = runCmd([sys.executable, "-c", script], tmp_path, 10)
    assert result["exit_code"] == 2
    assert result["stdout"].endswith("\rlinting [1/2]: 25%\r")
    assert result["stderr"].endswith("error: std::vector<size_t> [0]\n")
    assert "linting" not in result["output"]
    assert "error: std::vector<size_t> [0]" in result["output"]
    assert readableOutput("\x1b[31merror\x1b[0m\r\n") == "error"


def testSubprocessTimeoutKeepsOutput(tmp_path):
    result = runCmd([sys.executable, "-u", "-c", "import time; print('before timeout'); time.sleep(30)"], tmp_path, 0.2)
    assert result["timed_out"]
    assert "before timeout" in result["stdout"]
    assert "Timed out" in result["output"]


def fakeNative(fail_id=None):
    candidate = {"name": "candidate", "kernel_id": 1, "reference_expression": "ADD", "has_reference_factory": True,
                 "is_reference": False, "is_view": False}
    def test(kernel_id, configuration, seed):
        assert seed == 42
        if configuration["id"] == fail_id:
            return {"passed": False, "kind": "correctness_mismatch", "index": 0, "reference": 2., "actual": 0.,
                    "tolerance": {"absolute": 1e-4}, "reproduction": {"seed": seed}}
        return {"passed": True}
    return SimpleNamespace(kernel_bench_capabilities=lambda: ["TG_USE_CUDA"], kernel_bench_kernels=lambda: [candidate], kernel_bench_matches=lambda *args: True,
                           kernel_bench_match_diagnostics=lambda *args: {"matched": True, "reasons": []}, kernel_bench_test_diagnostics=test, kernel_bench_benchmark=lambda *args: [1.] * 8)


def payloadFor(tmp_path):
    return {"action": "evaluate", "kernel_name": "candidate", "reference_expression": "ADD",
            "configurations": [{"id": idx, "configuration": {"id": idx}} for idx in (10, 11)],
            "output_path": str(tmp_path / "native_result.json")}


def testBaselineFailureReportsExcludedKernels(monkeypatch, tmp_path):
    module = fakeNative()
    module.kernel_bench_matches = lambda *args: False
    monkeypatch.setattr(native_runner, "native", module)
    payload = payloadFor(tmp_path)
    payload["configuration"] = payload["configurations"][0]
    with pytest.raises(native_runner.NativeFailure) as exception:
        native_runner.calibrate(payload, {"result": {}})
    assert exception.value.details["configuration_id"] == 10
    assert exception.value.details["target_kernels"][0]["matches_configuration"] is False


def testNativeFailureEnvelopePreservesCompletedCorrectness(monkeypatch, tmp_path):
    payload = payloadFor(tmp_path)
    monkeypatch.setitem(sys.modules, "kernel_bench_native", fakeNative(11))
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    with pytest.raises(SystemExit) as exception:
        native_runner.main()
    assert exception.value.code == 1
    state = json.loads(Path(payload["output_path"]).read_text())
    assert state["error_details"]["kind"] == "correctness_mismatch"
    assert state["error_details"]["configuration_id"] == 11
    assert state["error_details"]["reference"] == 2.
    assert state["result"]["tested_configuration_ids"] == [10]
    assert len(state["result"]["correctness_results"]) == 2
    assert state["result"]["measurements"] == []


def testRegistrationFailureNamesCandidateAndObservedCount(monkeypatch, tmp_path):
    module = fakeNative()
    module.kernel_bench_kernels = list
    monkeypatch.setattr(native_runner, "native", module)
    with pytest.raises(native_runner.NativeFailure) as exception:
        native_runner.evaluate(payloadFor(tmp_path), {"result": {}})
    assert exception.value.details["submitted_name"] == "candidate"
    assert exception.value.details["observed_match_count"] == 0
    assert "guards" in str(exception.value)


@pytest.mark.skipif(os.name == "nt", reason="POSIX crash signals")
@pytest.mark.parametrize("timeout", [False, True])
def testNativeCrashAndTimeoutUseLastCheckpoint(monkeypatch, tmp_path, timeout):
    source = '''
import json, os, signal, time

def kernel_bench_capabilities():
    return ["TG_USE_CUDA"]
def kernel_bench_kernels():
    return [{"name": "candidate", "kernel_id": 1, "reference_expression": "ADD", "has_reference_factory": True, "is_reference": False, "is_view": False}]
def kernel_bench_matches(*args):
    return True
def kernel_bench_match_diagnostics(*args):
    return {"matched": True, "reasons": []}
def kernel_bench_test_diagnostics(*args):
    if TIMEOUT:
        time.sleep(30)
    os.kill(os.getpid(), signal.SIGABRT)
'''
    (tmp_path / "kernel_bench_native.py").write_text("TIMEOUT = " + repr(timeout) + "\n" + source)
    monkeypatch.setattr("kernel_bench.jobs.getPythonExe", lambda: sys.executable)
    with pytest.raises(JobFailure) as exception:
        runNative(tmp_path, payloadFor(tmp_path), 0.3 if timeout else 10)
    details = exception.value.details
    assert details["kind"] == ("timeout" if timeout else "native_crash")
    assert details["stage"] == "test"
    assert details["configuration_id"] == 10
    assert details["signal"] is not None
    assert (tmp_path / "native.stderr").exists()


@pytest.mark.parametrize("kind", ["compilation_error", "lint_error"])
def testBuildFailureArtifactsSurviveCandidateCleanup(monkeypatch, tmp_path, kind):
    (tmp_path / "build.py").write_text("")
    class FakeStore:
        def __init__(self):
            self.artifacts = {}
            self.events = []
        def getGoal(self, *args):
            return {"metadata": {"native_interface_version": 1}}
        def query(self, *args):
            return [{"source_archive": b""}]
        def addArtifact(self, job_id, name, data, media_type):
            self.artifacts[name] = data
            return name
        def addEvent(self, *args):
            self.events.append(args)
    store = FakeStore()
    (tmp_path / ".baseline_linted").touch()
    diagnostic = "candidate.cu(187): error: no suitable conversion [0]" if kind == "compilation_error" else "REDUNDANT LOGIC DETECTED: manual Input Count Check\nLinter Violation"
    def run(command, cwd, timeout, env):
        if command[1] == "kernel_linter.py" and kind == "compilation_error":
            return {"exit_code": 0, "output": "", "stdout": "", "stderr": "", "timed_out": False, "command": command}
        directory = Path(env["KERNEL_BENCH_DIAGNOSTICS_DIR"])
        if kind == "compilation_error":
            (directory / "compiler-test.json").write_text(json.dumps({"stderr": diagnostic, "stdout": "", "exit_code": 1, "command": "nvcc candidate.cu"}))
        return {"exit_code": 1, "output": diagnostic, "stdout": "linting\r", "stderr": diagnostic,
                "timed_out": False, "command": command}
    monkeypatch.setattr("kernel_bench.jobs.runCmd", run)
    job = {"id": "job", "goal_id": 1, "source": "submitted source", "backend": "cuda"}
    with pytest.raises(JobFailure) as exception:
        evaluateJob(store, job, tmp_path)
    assert exception.value.details["kind"] == kind
    assert diagnostic in exception.value.details["diagnostic"]
    assert diagnostic.splitlines()[0] in str(exception.value)
    assert store.artifacts["submitted_source"] == b"submitted source"
    name = "tensor_graphs_cpp/kernels/generated/kernel_bench/candidate.cu"
    assert store.artifacts[name] == b"submitted source"
    assert name in exception.value.details["artifact_ids"]
    assert not (tmp_path / name).exists()
    assert store.artifacts["build.stderr"].decode() == diagnostic


def diagnosticApp(monkeypatch):
    class FakeStore:
        def __init__(self, *args):
            self.queries = []
        def initialize(self):
            pass
        def getJob(self, job_id):
            return {"id": job_id, "goal_id": 1}
        def query(self, sql, parameters):
            self.queries.append((sql, parameters))
            if "SELECT name, media_type, data" in sql:
                if parameters[1] == "22222222-2222-2222-2222-222222222222":
                    return []
                return [{"name": "candidate.cu", "media_type": "text/plain", "data": b"source"}]
            if "artifacts" in sql:
                return [{"id": "11111111-1111-1111-1111-111111111111", "name": "candidate.cu", "media_type": "text/plain", "size_bytes": 6}]
            return [{"id": 3, "message": "error"}, {"id": 2, "message": "error"}]
    monkeypatch.setattr("kernel_bench.app.Store", FakeStore)
    return createApp(token="test", goal_id=1, start_worker=False)


def testLogFiltersAndScopedArtifacts(monkeypatch):
    app = diagnosticApp(monkeypatch)
    client = app.test_client()
    headers = {"X-Bench-Token": "test"}
    base = "/api/jobs/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    response = client.get(base + "/logs?after_id=1&tail=2&stage=build&grep=%25_", headers=headers)
    assert response.get_json()["next_after_id"] == 3
    assert [row["id"] for row in response.get_json()["events"]] == [2, 3]
    sql, parameters = app.extensions["kernel_bench_store"].queries[-1]
    assert "strpos(message, %s)" in sql
    assert parameters[1:] == (1, "build", "%_", 2)
    for query in ("tail=0", "tail=1001", "after_id=-1", "grep=" + "a" * 201):
        assert client.get(base + "/logs?" + query, headers=headers).status_code == 400
    response = client.get(base + "/artifacts", headers=headers)
    url = response.get_json()["artifacts"][0]["url"]
    assert client.get(url).status_code == 401
    assert client.get(url, headers=headers).data == b"source"
    assert client.get(base + "/artifacts/22222222-2222-2222-2222-222222222222", headers=headers).status_code == 404
    assert app.extensions["kernel_bench_store"].queries[-1][1][0] == base.split("/")[-1]


def testCompilerCaptureAvoidsRichMarkupAndKeepsRawStreams(monkeypatch, tmp_path, capsys):
    import shlex

    import build
    monkeypatch.setenv("KERNEL_BENCH_DIAGNOSTICS_DIR", str(tmp_path))
    monkeypatch.setattr(build, "SERVER_JOB", True)
    toolchain = build.Toolchain(None, SimpleNamespace(is_windows=False))
    source = "import sys; print('compiler stdout'); sys.stderr.write('error: std::vector<size_t> [red]\\n'); sys.exit(1)"
    with pytest.raises(SystemExit):
        toolchain.run_cmd([shlex.quote(sys.executable), "-c", shlex.quote(source)], "test compiler")
    records = list(tmp_path.glob("compiler-*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["exit_code"] == 1
    assert record["stderr"] == "error: std::vector<size_t> [red]\n"
    assert records[0].with_suffix(".stderr").read_bytes() == record["stderr"].encode()
    assert "[red]" in capsys.readouterr().err


def testWorkerPersistsFailureDetailsAndDiagnosticEvent(monkeypatch):
    import threading
    from contextlib import contextmanager

    from kernel_bench.jobs import startWorker

    stop_event = threading.Event()
    class FakeConnection:
        def execute(self, *args):
            return self
        def fetchone(self):
            return {"acquired": True}
    class FakeStore:
        def __init__(self):
            self.events = []
            self.failure = None
        @contextmanager
        def connection(self):
            yield FakeConnection()
        def execute(self, *args):
            pass
        def nextJob(self, validation=False):
            return {"id": "test", "goal_id": 1}
        def addEvent(self, *args):
            self.events.append(args)
        def failJob(self, *args):
            self.failure = args
            stop_event.set()
    def evaluate(*args):
        raise JobFailure("Correctness failed", {"kind": "correctness_mismatch", "stage": "test",
                                               "diagnostic": "actual mismatch", "artifact_ids": ["artifact"]},
                         {"tested_configuration_ids": [1]})
    monkeypatch.setattr("kernel_bench.jobs.evaluateJob", evaluate)
    store = FakeStore()
    worker = startWorker(store, stop_event)
    worker.join(5)
    assert not worker.is_alive()
    assert store.failure[2]["kind"] == "correctness_mismatch"
    assert store.failure[3]["tested_configuration_ids"] == [1]
    assert "actual mismatch" in store.events[-1][2]
    assert "artifact" in store.events[-1][2]


def testCompileValidationNeverExecutesCandidate(monkeypatch, tmp_path):
    from kernel_bench import native_runner
    native = fakeNative()
    def forbidden(*args):
        pytest.fail("Compile validation executed a configuration")
    native.kernel_bench_matches = forbidden
    native.kernel_bench_test_diagnostics = forbidden
    native.kernel_bench_benchmark = forbidden
    monkeypatch.setattr(native_runner, "native", native)
    payload = payloadFor(tmp_path)
    payload["action"] = "compile"
    result = native_runner.evaluate(payload, {"result": {}})
    assert result["candidate"]["name"] == "candidate"
    assert result["measurements"] == []


def testLintModeNeverBuildsOrLoads(monkeypatch, tmp_path):
    (tmp_path / "build.py").touch()
    class FakeStore:
        def query(self, *args):
            return [{"source_archive": b""}]
        def getGoal(self, *args):
            return {"fingerprint": "snapshot", "metadata": {"native_interface_version": 1}}
        def addArtifact(self, *args):
            return "artifact"
        def addEvent(self, *args):
            pass
        def completeJob(self, job_id, result):
            self.result = result
    commands = []
    def run(command, *args, **kwargs):
        commands.append(command)
        assert command[1] == "kernel_linter.py"
        return {"exit_code": 0, "output": "", "stdout": "", "stderr": "", "command": command}
    monkeypatch.setattr("kernel_bench.jobs.runCmd", run)
    store = FakeStore()
    evaluateJob(store, {"id": "job", "goal_id": 1, "backend": "cpu", "source": "candidate", "mode": "lint"}, tmp_path)
    assert len(commands) == 1
    assert store.result["passed_stages"] == ["submission", "lint"]
    assert store.result["measurements"] == []
    assert store.result["build_fingerprint"]
