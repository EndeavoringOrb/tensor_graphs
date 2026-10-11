"""Shared agent orientation and portable starter workspace."""
from pathlib import Path

INSTRUCTIONS = """Optimize individual native kernels for this server's compiled goal graph.
Read BENCH_SECRET_TOKEN from your environment and send it as X-Bench-Token on API
requests. Never print the token or write it into submitted source. Set
BENCH_SERVER_URL to setup.base_url. Download setup.download with that header and
extract the ZIP into your working directory; it contains pyproject.toml, agent.py,
AGENTS.md, .env.example, and the goal snapshot's hardware documentation. Run
uv sync, then use .venv/bin/python (Windows: .venv/Scripts/python.exe).
The client reads BENCH_SERVER_URL and BENCH_SECRET_TOKEN from the environment;
.env.example is a configuration example, not an automatically loaded file.

Read /api/status, list /api/targets, and choose a target. Inspect
/api/targets/{target_id} for exact shapes, strides, dtypes, constants, existing
kernels, and source paths. Use /api/source/list and /api/source?path=... to inspect
baseline kernels, reference factories, core headers, and hardware documentation.
Inspect /api/jobs?target_id=... before designing an optimization. Multiple agents
may work on the same target; there are no reservations.

Submit one complete source file with exactly one REGISTER_KERNEL, a unique
registration name, and a reference factory matching the target expression to
POST /api/iteration/submit. JSON fields: target_id, source, idea (optimization
hypothesis), backend (cpu or cuda). Kernels may specialize to a subset of target
configurations. Poll /api/jobs/{job_id} until completed or failed; inspect
/api/jobs/{job_id}/logs and correctness coverage, raw timing samples, medians,
and per-configuration speedups. Zero matches and correctness/build failures fail
the job. Iterate using the measurements; report coverage and evidence for gains.
The server builds, tests, and benchmarks on its hardware, so a local CUDA/compiler
installation is unnecessary. Model inference, compiler selection, and automatic
kernel promotion are outside this service's scope.

Use PascalCase classes, camelCase functions, and snake_case variables. Fail loudly
on invalid assumptions. Keep threaded and unthreaded implementations in separate
kernels; the compiler chooses between them. Capability guards such as
#if defined(TG_HAS_AVX2) must wrap the entire kernel file, with no fallbacks inside
run functions.
"""


def starterFiles(archive):
    files = {
        "pyproject.toml": b'''[project]
name = "kernel-bench-workspace"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = ["requests>=2.32.5"]
''',
        "agent.py": Path(__file__).with_name("agent.py").read_bytes(),
        "AGENTS.md": INSTRUCTIONS.encode(),
        ".env.example": b"BENCH_SERVER_URL=https://your-server\n# BENCH_SECRET_TOKEN must be supplied in your environment.\n",
    }
    for name in archive.namelist():
        if name.startswith("docs/hwinfo/box/"):
            files[name] = archive.read(name)
    return files
