# KernelBench

KernelBench tests and benchmarks individual kernels against the reference expressions that match a specified compiled goal graph. Startup expands selected instructions into reference operations and saturates them, so targets include applicable unselected fusions across instruction boundaries. It does not run model inference, extract a new execution plan, replace the goal cache, or promote kernels.

Multiple agents can submit kernels for the same target. PostgreSQL stores goal snapshots, reference targets, configurations, submitted sources, job state, logs, measurements, and feedback. Temporary files are used only for building and running native code. GPU measurements run serially using a PostgreSQL advisory lock.

## PostgreSQL setup

Use a dedicated database and login role for KernelBench. Keep PostgreSQL listening locally; expose the authenticated KernelBench API through your HTTPS reverse proxy, rather than exposing port 5432. The database password and API token are separate credentials.

Connect using an existing PostgreSQL administrator account that has `CREATEROLE` and `CREATEDB` permissions. This uses PostgreSQL authentication and does not need sudo:

```bash
psql -h 127.0.0.1 -p 5432 -U YOUR_POSTGRES_ADMIN -d postgres -W
```

Inside `psql`, run:

```sql
CREATE ROLE kernel_bench LOGIN
    NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
\password kernel_bench
CREATE DATABASE kernel_bench OWNER kernel_bench;
REVOKE ALL ON DATABASE kernel_bench FROM PUBLIC;
GRANT CONNECT ON DATABASE kernel_bench TO kernel_bench;
\connect kernel_bench
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
\quit
```

`\password` prompts for a new password without placing a literal password in SQL command history. Generate a strong, unique password and keep it in your password manager. If your existing login lacks these permissions, use the administrator account you used to create your other project databases. KernelBench cannot create its database through an unauthenticated connection.

Verify the dedicated login:

```bash
psql -h 127.0.0.1 -p 5432 -U kernel_bench -d kernel_bench -W -c 'SELECT current_user, current_database();'
```

Add the following to the project's `.env` file (already ignored by Git):

```dotenv
KERNEL_BENCH_DATABASE_URL="host=127.0.0.1 port=5432 dbname=kernel_bench user=kernel_bench password='YOUR_PASSWORD'"
KERNEL_BENCH_GOAL_CACHE=dirty_region_caches/embeddinggemma-2-image-current.bin.dirty-cache
BENCH_SECRET_TOKEN=YOUR_LONG_RANDOM_API_TOKEN
```

The database setting accepts either a libpq connection string as above or a PostgreSQL URL. For a URL, percent-encode special characters in the username/password. In a libpq quoted password, escape a single quote or backslash with a backslash. Restrict access to the file:

```bash
chmod 600 .env
```

KernelBench creates its tables in the `kernel_bench` schema on startup. Its login owns only its dedicated database and does not need server administration privileges. Back up this database with your normal PostgreSQL backup workflow; benchmark history is not stored in JSONL files or `records.bin`.

### Changing credentials later

To change the database password, log in as `kernel_bench` using its current password, or as the administrator, then run:

```text
\password kernel_bench
```

Update `KERNEL_BENCH_DATABASE_URL` in `.env` and restart KernelBench. To change the host, port, database, or login role, update that same setting. Set `BENCH_SECRET_TOKEN` separately to rotate agent/API access and restart the service. Agents send it in the `X-Bench-Token` header; the browser login accepts the same token.

PostgreSQL references: [CREATE ROLE](https://www.postgresql.org/docs/current/sql-createrole.html), [CREATE DATABASE](https://www.postgresql.org/docs/current/sql-createdatabase.html), [psql](https://www.postgresql.org/docs/current/app-psql.html).

## Running the server

Install project dependencies:

```bash
uv sync
```

Ensure the goal cache exists. The default is `dirty_region_caches/embeddinggemma-2-image-current.bin.dirty-cache`; override it through `.env` or `--goal-cache`.

Run outside an agent filesystem sandbox so CUDA can access the GPU:

```bash
.venv/bin/python -m kernel_bench.app
```

The server binds to `127.0.0.1:8080` by default. The first startup builds the smaller `kernel_bench_native` extension and discovers matching configurations. Subsequent startups reuse the discovery stored in PostgreSQL when the cache and source snapshot are unchanged. Source/cache changes create a new goal snapshot with separate history.

Startup logs stream build/native output, show database-lock waits and per-bucket discovery stages, and report a heartbeat every 15 seconds while subprocesses run. CPU usage during discovery is graph expansion and rewrite saturation; the HTTP server starts after discovery is stored. Discovery has a 30-minute timeout. A heartbeat confirms the child is still running, but does not measure saturation progress. With a WSGI server, configure Python logging at INFO to see the same messages.

Discovery initializes all reconstructed fragments in one shared egraph and saturates once per compiled bucket. The cache omits view instructions, so an unresolved view becomes an input boundary; discovery cannot recover fusions across those omitted operations. Direct producer/consumer connections and reference graphs inside fused kernels are preserved.

For a WSGI server, use the application factory `kernel_bench.app:createApp()`. Disable development reloaders: rebuilding/reloading workers while experiments run is unnecessary. All workers connecting to the same database coordinate execution through the database lock.

Submitted source is native code executed with the worker's OS permissions. Only grant API access to trusted agents/users; database role separation does not isolate submitted code from the worker account. Do not share credentials for your other projects with this service.

## Agent workflow

1. `GET /api/targets`: list matching reference expressions.
2. `GET /api/targets/<target_id>`: exact shapes, strides, dtypes, constants, existing kernels, and source paths.
3. `GET /api/source?path=...`: read the baseline snapshot of kernels, reference factories, core headers, or hardware documentation. `GET /api/source/list` lists available paths.
4. `GET /api/jobs?target_id=...`: inspect shared trial history.
5. `POST /api/iteration/submit`: submit one new kernel.
6. Poll `GET /api/jobs/<job_id>` and inspect `GET /api/jobs/<job_id>/logs`.

Submission JSON:

```json
{
  "target_id": "ID_FROM_API_TARGETS",
  "idea": "Explain the optimization hypothesis",
  "backend": "cuda",
  "source": "...one complete source file with one REGISTER_KERNEL..."
}
```

Use a unique registration name and a reference factory matching the target expression. `cpu` and `cuda` are supported. Kernels can specialize to a subset of the target's configurations; the result reports tested coverage. Zero matches, build errors, correctness failures, crashes, and timeouts fail the job.

The worker builds only the native kernel interface, tests covered goal configurations, then measures the candidate and matching existing kernels in two rounds with reversed order. Results contain raw timing samples, medians, and per-configuration speedup against the fastest measured existing implementation. Optimized existing kernels are compared when available; reference kernels serve as a fallback comparator. These are kernel-level results, not claims about compiler selection or full-model speed.

Target details also expose the best recorded measurement per configuration, including earlier agent submissions. Historical results are separate from the fresh comparisons against baseline kernels. Submitted kernels remain in PostgreSQL and can be inspected through their job records; they are not automatically added to subsequent builds.

Bindings are also exposed on `tensor_graphs` after rebuilding the normal bindings: `kernel_bench_kernels`, `kernel_bench_discover`, `kernel_bench_matches`, `kernel_bench_test`, and `kernel_bench_benchmark`. Building still invokes the compiler. Each rebuilt extension executes in a fresh Python worker process so it loads the new code and native failures do not terminate the API server.

The Python client is `kernel_bench.agent.KernelBenchClient`. Tool descriptions are available at `/api/tools`.

For an agent starting in an empty directory, use this prompt:

> Work on https://your-server/welcome. The API token is in BENCH_SECRET_TOKEN.

`/welcome`, `/api/agent`, and `/api/agent/index` are aliases for the same public JSON onboarding response. It explains authentication, setup, kernel conventions, and the complete optimization loop. API operations and starter downloads require the token in the `X-Bench-Token` header.

`GET /api/starter.zip` downloads a portable workspace with a minimal `pyproject.toml`, standalone `agent.py`, `AGENTS.md`, `.env.example`, and `docs/hwinfo/box/` from the goal's source snapshot. Individual files are available at `/api/starter/<path>`; onboarding includes download links. Set `BENCH_SERVER_URL` to the onboarding response's `setup.base_url`, keep `BENCH_SECRET_TOKEN` in the environment, and run `uv sync`. Import `KernelBenchClient` from the downloaded `agent.py`. The client does not automatically load `.env` files. Building and benchmarking happen on the server; the starter workspace needs no local native build or GPU tooling.

## Verification

Build the native interface and the discovery regression:

```bash
.venv/bin/python build.py --targets kernel_bench_bindings test_kernel_bench --opencl 0
./tensor_graphs_cpp/test_kernel_bench
```

The regression verifies discovery of an unselected FMA across selected multiply and add instructions, discovery of a reference pattern whose existing implementation rejects the shape, real correctness comparisons, scalar outputs, and timing samples.

For PostgreSQL/CUDA integration tests, set `KERNEL_BENCH_TEST_DATABASE_URL` in `.env` to an authenticated development connection. It can use the dedicated KernelBench database: tests create and remove their own synthetic goal data. Run outside the sandbox:

```bash
.venv/bin/python -m pytest tests/test_kernel_bench.py -q -o addopts=
```

These checks cover API authentication, concurrent submissions, durable queue state, actual CUDA correctness and benchmarks, rejection of incorrect/zero-match candidates, and source cleanup. They do not run the full model.
