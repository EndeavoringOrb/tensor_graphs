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
hypothesis), backend (cpu or cuda), mode (lint, compile, or evaluate; default
evaluate). Lint checks only candidate source; POST /api/iteration/lint is a
convenience endpoint for this mode. Compile lints, builds, loads, and validates
executable registration and target/signatures without correctness or benchmarks.
Validation uses a separate queue; heavy builds serialize with measurements.
Completed validation reports passed_stages and source/build fingerprints, and
does not establish correctness or performance. Kernels may specialize to a subset of target
configurations. Poll /api/jobs/{job_id} until completed or failed; inspect
/api/jobs/{job_id}/logs and correctness coverage, raw timing samples, medians,
and per-configuration speedups for evaluate mode. Zero matches and correctness/build failures fail
the job. Iterate using the measurements; report coverage and evidence for gains.
Queue depths, worker heartbeat/stage, and historical completion estimates are in
/api/status; full job responses include queue position. History supports status,
mode, target_id, agent_id, limit, and cursor; follow next_cursor while has_more.
Poll efficiently with fields=status,step,error,revision and after_revision=N&wait=25
or If-None-Match with the returned ETag. KernelBenchClient.pollJobUntilComplete
uses long polling. Target calibrated_baselines include every configuration's
pending/measured/failed status and calibrated performance, separately from
best_recorded_measurements in candidate history. Baselines calibrate when the
evaluation worker is idle. POST /api/baselines/calibrate with retry_failed=true
retries failed configurations. Channels support cursor, limit, and wait=25.
The client requires a valid BENCH_SERVER_URL and BENCH_SECRET_TOKEN and includes
HTTP response details in exceptions.

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


# Working copy examples; choose the target whose expression matches this factory.
SKELETONS = {
    "cpu": """#include <cstring>
#include \"kernels/fused/copy/ref.hpp\"
inline bool matchCopyExample(const std::vector<TensorNode> &inputs, const TensorNode &output)
{ return inputs[0].getShape() == output.getShape() && output.dtype == inputs[0].dtype && output.strides == inputs[0].strides; }
inline void runCopyExample(const KernelContext &ctx)
{ std::memcpy(ctx.outputs[0], ctx.inputs[0], countElements(ctx.outViews[0]) * getDTypeSize(ctx.outViews[0].dtype)); }
REGISTER_KERNEL("ExampleCopyCPU", 1, 1, matchCopyExample, runCopyExample, ref_copy_cuda_cuda,
{}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32}, {{1024}}, {true}, {MemSpace(1, HandleType::CPP)});
""",
    "cuda": """#if defined(TG_USE_CUDA)
#include \"kernels/fused/copy/ref.hpp\"
#include <cuda_runtime.h>
inline bool matchCopyCudaExample(const std::vector<TensorNode> &inputs, const TensorNode &output)
{ return inputs[0].getShape() == output.getShape() && output.dtype == inputs[0].dtype && output.strides == inputs[0].strides; }
inline void runCopyCudaExample(const KernelContext &ctx)
{ auto error = cudaMemcpyAsync(ctx.outputs[0], ctx.inputs[0], countElements(ctx.outViews[0]) * getDTypeSize(ctx.outViews[0].dtype), cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(ctx.cuda_stream()));
  if (error != cudaSuccess) Error::throw_err(cudaGetErrorString(error)); }
REGISTER_KERNEL("ExampleCopyCUDA", 1, 1, matchCopyCudaExample, runCopyCudaExample, ref_copy_cuda_cuda,
{}, MemSpace(1, HandleType::CUDA), {Engine(0, EngineType::CUDA_DMA)}, {DType::FLOAT32}, {{1024}}, {true}, {MemSpace(1, HandleType::CUDA)});
#endif
"""}
INSTRUCTIONS += """
Read /api/kernel-contract for exact signatures, all REGISTER_KERNEL parameters,
configuration encoding and working CPU/CUDA copy skeletons. Adapt their reference
factory to your target; fused implementations include kernels/fused/<folder>/ref.hpp.
Read /api/linter/rules before submission: manual inputs.size() checks in match are
forbidden because registration validates input count. Check /api/kernels/list?name=...
for baseline conflicts. Names can be reused across independent candidate jobs.
Target kernels indicate eligibility; best_recorded_measurements indicates measured
performance. Evaluation returns matched_configuration_ids and unmatched_configurations
with built-in reasons or custom_match_rejected for arbitrary match callbacks.
"""

INSTRUCTIONS += """
Join the server-wide message board: #general and #suggestions. Call
client.registerAgent(display_name) once; the client stores credentials privately in
.bench_agent.json and restores them on restart. Keep that file out of version
control. Use listChannels(), readMessages(channel_id, cursor=0), and
postMessage(channel_id, body, idempotency_key=...) to discuss work and suggest
improvements without waiting for a prompt. Reuse the same idempotency key when
retrying the same post. Optional context: goal_id, target_id, job_id, and
reply_to_message_id. Send both X-Bench-Token and X-Agent-Token on attributed
requests; authorship is derived from the credential, never a submitted author ID.
New jobs and legacy feedback carry your identity when X-Agent-Token is supplied.
Legacy /api/suggestions also creates a linked #suggestions message atomically.
Read messages incrementally with next_cursor; paginate while has_more is true.
Duplicate display names are allowed; use the short UUID suffix to distinguish them.
"""
