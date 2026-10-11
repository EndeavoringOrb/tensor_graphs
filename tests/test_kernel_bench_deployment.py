"""Deployment lifecycle checks without PostgreSQL or native builds."""
import os
import signal
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from kernel_bench.app import createApp
from kernel_bench.jobs import startWorker


def testDrainFinishesActiveJobWithoutClaimingAnother(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    stop_event = threading.Event()

    class FakeConnection:
        def execute(self, *args):
            return self

        def fetchone(self):
            return {"acquired": True}

    class FakeStore:
        claims = 0

        @contextmanager
        def connection(self):
            yield FakeConnection()

        def execute(self, *args):
            pass

        def nextJob(self):
            self.claims += 1
            return {"goal_id": 1}

    def evaluateJob(*args):
        entered.set()
        assert release.wait(5)
        completed.set()

    monkeypatch.setattr("kernel_bench.jobs.evaluateJob", evaluateJob)
    store = FakeStore()
    worker = startWorker(store, stop_event)
    try:
        assert entered.wait(5)
        stop_event.set()
        worker.join(0.05)
        assert worker.is_alive()
        assert not completed.is_set()
    finally:
        release.set()
        worker.join(5)
    assert completed.is_set()
    assert not worker.is_alive()
    assert store.claims == 1


def testDrainRejectsRequests(monkeypatch):
    class FakeStore:
        def __init__(self, *args):
            pass

        def initialize(self):
            pass

    monkeypatch.setattr("kernel_bench.app.Store", FakeStore)
    stop_event = threading.Event()
    app = createApp(token="test", goal_id=1, start_worker=False, stop_event=stop_event)
    stop_event.set()
    client = app.test_client()
    for path in ("/welcome", "/api/iteration/submit"):
        response = client.open(path, method="POST" if "submit" in path else "GET")
        assert response.status_code == 503
        assert response.headers["Retry-After"] == "60"


@pytest.mark.skipif(os.name == "nt", reason="Linux systemd signal lifecycle")
@pytest.mark.parametrize("during_startup", [False, True])
def testServiceSignalWaitsAndExits(during_startup):
    script = '''
import argparse
import threading
import time
from flask import Flask
import kernel_bench.app as module
import waitress

original_create_server = waitress.create_server
def createServer(*args, **kwargs):
    server = original_create_server(*args, **kwargs)
    original_run = server.run
    def run():
        print("ready", flush=True)
        original_run()
    server.run = run
    return server
waitress.create_server = createServer

def createApp(goal_cache, stop_event):
    app = Flask(__name__)
    def worker():
        stop_event.wait()
        time.sleep(0.2)
        print("drained", flush=True)
    if STARTUP:
        print("ready", flush=True)
        stop_event.wait()
        time.sleep(0.2)
        print("startup finished", flush=True)
    thread = threading.Thread(target=worker)
    thread.start()
    app.extensions["kernel_bench_worker"] = thread
    return app

module.createApp = createApp
module.serveApp(argparse.Namespace(goal_cache=None, host="127.0.0.1", port=0))
'''
    process = subprocess.Popen([sys.executable, "-c", "STARTUP = " + repr(during_startup) + "\n" + script],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "ready"
        process.send_signal(signal.SIGTERM)
        output, error = process.communicate(timeout=10)
        assert process.returncode == 0, error
        assert "drained" in output
        if during_startup:
            assert "startup finished" in output
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


@pytest.mark.skipif(os.name == "nt", reason="Linux deployment script")
@pytest.mark.parametrize("scenario", ["update", "unchanged", "dirty", "dependency_failure"])
def testUpdaterOrdersDrainBeforeMutation(tmp_path, scenario):
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    log_path = tmp_path / "commands"
    scripts = {
        "git": '''#!/bin/bash
echo "git $*" >> "$COMMAND_LOG"
case "$1 $2" in
    'rev-parse origin/main') echo new ;;
    'rev-parse HEAD') if [[ "$SCENARIO" == unchanged ]]; then echo new; else echo old; fi ;;
    'branch --show-current') echo main ;;
    'status --porcelain') if [[ "$SCENARIO" == dirty ]]; then echo ' M app.py'; fi ;;
esac
exit 0
''',
        "systemctl": '''#!/bin/bash
echo "systemctl $*" >> "$COMMAND_LOG"
''',
        "uv": '''#!/bin/bash
echo "uv $*" >> "$COMMAND_LOG"
if [[ "$SCENARIO" == dependency_failure ]]; then exit 1; fi
''',
        "curl": '''#!/bin/bash
echo "curl" >> "$COMMAND_LOG"
''',
    }
    for name, script in scripts.items():
        path = bin_path / name
        path.write_text(script)
        path.chmod(0o755)
    environment = dict(os.environ, PATH=str(bin_path) + os.pathsep + os.environ["PATH"],
                       KERNEL_BENCH_CHECKOUT=str(checkout), KERNEL_BENCH_STATE_DIR=str(tmp_path / "state"),
                       COMMAND_LOG=str(log_path), SCENARIO=scenario)
    result = subprocess.run(["bash", str(Path(__file__).resolve().parents[1] / "deploy/kernelbench-update.sh")],
                            env=environment, capture_output=True, text=True, timeout=10, check=False)
    commands = log_path.read_text().splitlines()
    stop = "systemctl --user stop kernelbench.service"
    start = "systemctl --user start kernelbench.service"
    merge = "git merge --ff-only new"
    if scenario in ("dirty", "unchanged"):
        assert stop not in commands
        assert merge not in commands
        assert result.returncode == (1 if scenario == "dirty" else 0)
    else:
        assert commands.index(stop) < commands.index(merge) < commands.index("uv sync --locked")
        if scenario == "dependency_failure":
            assert result.returncode == 1
            assert start not in commands
        else:
            assert result.returncode == 0, result.stderr
            assert commands.index("uv sync --locked") < commands.index(start) < commands.index("curl")
