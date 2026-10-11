"""Onboarding works without PostgreSQL, native builds, or a GPU."""
import io
import zipfile

from kernel_bench.app import createApp


def testWelcomeAndStarterDownloads(monkeypatch):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("docs/hwinfo/box/cpu.txt", "snapshot CPU")
        archive.writestr("tensor_graphs_cpp/core/example.hpp", "private source")

    class FakeStore:
        def __init__(self, database_url):
            pass

        def initialize(self):
            pass

        def query(self, sql, params):
            return [{"source_archive": output.getvalue()}]

    monkeypatch.setattr("kernel_bench.app.Store", FakeStore)
    app = createApp(token="test-token", goal_id="test-goal", start_worker=False)
    client = app.test_client()
    welcome = client.get("/welcome")
    assert welcome.status_code == 200
    data = welcome.get_json()
    assert data == client.get("/api/agent").get_json()
    assert data == client.get("/api/agent/index").get_json()
    assert "BENCH_SECRET_TOKEN" in data["instructions"]
    assert client.get("/api/starter.zip").status_code == 401
    assert client.get("/api/status").status_code == 401
    headers = {"X-Bench-Token": "test-token"}
    response = client.get(data["setup"]["download"], headers=headers)
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
        assert set(archive.namelist()) == {"pyproject.toml", "agent.py", "AGENTS.md", ".env.example", "docs/hwinfo/box/cpu.txt"}
        assert archive.read("docs/hwinfo/box/cpu.txt") == b"snapshot CPU"
        assert b"class KernelBenchClient" in archive.read("agent.py")
        for name, url in data["downloads"].items():
            assert client.get(url, headers=headers).data == archive.read(name)
    assert client.get("/api/starter/docs/hwinfo/box/cpu.txt", headers=headers).data == b"snapshot CPU"
    assert client.get("/api/starter/.env", headers=headers).status_code == 404
    assert client.get("/api/starter/tensor_graphs_cpp/core/example.hpp", headers=headers).status_code == 404
