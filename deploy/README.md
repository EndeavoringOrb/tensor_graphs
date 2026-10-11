# Independent KernelBench deployment

These instructions install user systemd services for `~/services/kernel_bench`. Commit and push these changes to GitHub main first. Run commands as `aaron` except where sudo is shown. Git, curl, flock (util-linux), uv, and the project's existing C++/CUDA toolchain must be installed. Ensure uv is in `~/.local/bin` or edit the updater unit's PATH.

## Prepare the checkout

```bash
mkdir -p ~/services
git clone --branch main https://github.com/EndeavoringOrb/tensor_graphs.git ~/services/kernel_bench
cd ~/services/kernel_bench
uv sync --locked --extra embeddinggemma --extra ml
mkdir -p ~/services/kernel_bench_data
cp ~/Projects/tensor_graphs/dirty_region_caches/embeddinggemma-2-image-current.bin.dirty-cache ~/services/kernel_bench_data/goal.bin.dirty-cache
cp ~/Projects/tensor_graphs/.env .env
chmod 600 .env
```

Edit the deployment `.env`, retaining only KernelBench's settings. Preserve `KERNEL_BENCH_DATABASE_URL` and `BENCH_SECRET_TOKEN`, and set:

```dotenv
KERNEL_BENCH_GOAL_CACHE=/home/aaron/services/kernel_bench_data/goal.bin.dirty-cache
```

If the token was only in `.bench_token`, put it in `BENCH_SECRET_TOKEN` in this `.env`. Reusing the database preserves jobs/history. For a new database, follow [PostgreSQL setup](../kernel_bench/README.md#postgresql-setup). Credentials and caches are not fetched from Git; the copied cache is independent of development edits.

## Migrate and start

Let any old server's active startup/build/benchmark finish before stopping it: the old version lacks graceful shutdown. Stop manually launched servers too. If the old system service exists:

```bash
sudo systemctl disable --now kernelbench.service
```

Verify port 8080 is free, then install the new user units:

```bash
cd ~/services/kernel_bench
mkdir -p ~/.config/systemd/user ~/.local/lib/kernelbench
cp kernelbench.service ~/.config/systemd/user/
cp deploy/kernelbench-update.service deploy/kernelbench-update.timer ~/.config/systemd/user/
cp deploy/kernelbench-update.sh ~/.local/lib/kernelbench/update.sh
systemctl --user daemon-reload
systemctl --user enable --now kernelbench.service
journalctl --user -u kernelbench.service -f
```

Startup discovery may take tens of minutes. Once it finishes, check from another terminal, enable updates, and enable operation without an interactive login:

```bash
curl --fail http://127.0.0.1:8080/welcome
systemctl --user enable --now kernelbench-update.timer
sudo loginctl enable-linger aaron
```

Run only one deployed server. SIGTERM sets its drain event: requests receive 503, the worker finishes the active job, and startup discovery also finishes if in progress. Queued jobs stay in PostgreSQL. `KillMode=mixed` sends SIGTERM only to the parent; native children finish normally. `TimeoutStopSec=infinity` prevents forced termination. Existing native timeouts still apply; a genuinely stuck process requires manual intervention.

## Website integration

Point a subdomain such as `kernelbench.YOUR_DOMAIN` at this machine. Configure its TLS virtual host in your existing reverse proxy and add the location block in [kernelbench-nginx.conf](kernelbench-nginx.conf). Add a link to `https://kernelbench.YOUR_DOMAIN/` in `~/endeavoringOrbWebsite` navigation. The website continues running with its existing upstream.

KernelBench listens on loopback port 8080 and trusts one local proxy. The proxy must overwrite forwarded host/protocol headers as shown so onboarding uses HTTPS URLs and login cookies are secure. Keep port 8080 private. Use a subdomain: current UI/API links are root-relative, so a `/kernel_bench` mount needs further route changes.

References: [Nginx proxy module](https://nginx.org/en/docs/http/ngx_http_proxy_module.html), [Waitress API](https://docs.pylonsproject.org/projects/waitress/en/stable/api.html).

## Updates and maintenance

The timer fetches `origin/main` about once a minute after the previous check finishes. When it changes, the updater validates a clean checkout and fast-forward history, drains/stops the service, advances to the fetched commit, runs `uv sync --locked --extra embeddinggemma --extra ml`, starts the service, and checks readiness. Startup builds/discovers changed source snapshots. Queued jobs use their original goal's archived sources. Commits arriving during deployment are picked up on the next check. A file lock serializes updater scripts.

The updater fails loudly on dirty/divergent history, dependency errors, or startup failure. It never force-resets local work and has no automatic rollback. Dependency failure after draining leaves the service stopped. Once HEAD has advanced, startup failures require manual recovery rather than automatically retrying that same commit.

Units and the updater are copied outside the checkout so pulling cannot overwrite a running updater. Recopy them and run `daemon-reload` when deployment files change.

```bash
systemctl --user status kernelbench.service kernelbench-update.timer
journalctl --user -u kernelbench.service -u kernelbench-update.service
systemctl --user start kernelbench-update.service
systemctl --user restart kernelbench.service
```

Restart waits for active work. For manual maintenance, stop the timer, wait for any active updater to finish, then stop the benchmark service:

```bash
systemctl --user stop kernelbench-update.timer
systemctl --user status kernelbench-update.service
systemctl --user stop kernelbench.service
```

After resolving a dependency/startup failure:

```bash
cd ~/services/kernel_bench
uv sync --locked --extra embeddinggemma --extra ml
systemctl --user reset-failed kernelbench.service kernelbench-update.service
systemctl --user start kernelbench.service
```

Wait for discovery, then check `/welcome` and restart the timer. Waitress serves HTTP with four threads and one serial benchmark worker. The checkout is isolated from development files; CPU/GPU resources remain shared with interactive work.


## Configure promotion

Copy `deploy/gemma-decode-promotion.json` to
`~/services/kernel_bench_data/promotion.json`, verify its model path, and add
`KERNEL_BENCH_PROMOTION_CONFIG=/home/aaron/services/kernel_bench_data/promotion.json`
to the deployed `.env`. Authenticate `gh` as the service account and configure Git's
HTTPS credential helper (`gh auth setup-git`). The sample enables PRs in
`EndeavoringOrb/tensor_graphs`; omit `github_repo` for validation-only operation.
Restart after draining active work. Source changes must be committed before deployment:
promotions reject source snapshots that differ from their recorded Git commit.

The updater now retains the `embeddinggemma` and `ml` extras. Recopy the updater script
into `~/.local/lib/kernelbench/update.sh` when installing this version. `/progress`
shows reference kernel benchmark history immediately; logical goal measurements appear
when the queued baseline/full validations finish. All heavy jobs share one FIFO worker.

The initial deployed goal remains the existing 17-bucket Gemma decode cache. `/admin` uses a separate `KERNEL_BENCH_ADMIN_TOKEN` to activate stored snapshots or queue trusted presets from `KERNEL_BENCH_GOAL_CATALOG`. Preset preparation runs on the same worker; see the operator goal switching section in `kernel_bench/README.md` for the catalog format. Keep both tokens in the deployment `.env`, readable only by the service user.
