#!/usr/bin/env bash
set -euo pipefail

checkout="${KERNEL_BENCH_CHECKOUT:-$HOME/services/kernel_bench}"
state_directory="${KERNEL_BENCH_STATE_DIR:-$HOME/.cache/kernelbench}"
mkdir -p "$state_directory"
exec 9>"$state_directory/update.lock"
flock -n 9 || exit 0
phase=fetch
writeState() {
    local update_status="$1"
    printf '%s\n%s\n%s\n' "$update_status" "$phase" "$(date -u +%FT%TZ)" > "$state_directory/deployment-status.tmp"
    mv "$state_directory/deployment-status.tmp" "$state_directory/deployment-status"
}
trap 'writeState failed' ERR
refuseUpdate() {
    local failure_details="$1"
    phase=refused
    writeState failed
    if [[ ! -f "$state_directory/last-refusal" ]] || [[ "$(cat "$state_directory/last-refusal")" != "$failure_details" ]]; then
        printf '%s\n' "$failure_details" >&2
        printf '%s\n' "$failure_details" > "$state_directory/last-refusal"
    fi
    exit 1
}
cd "$checkout"
export GIT_TERMINAL_PROMPT=0
git fetch --no-tags origin main
target_commit=$(git rev-parse origin/main)
current_commit=$(git rev-parse HEAD)
if [[ "$target_commit" == "$current_commit" ]]; then
    writeState current
    exit 0
fi
if [[ "$(git branch --show-current)" != main ]]; then
    refuseUpdate 'Deployment checkout must be on main'
fi
changed_paths=$(git status --porcelain --untracked-files=normal)
if [[ -n "$changed_paths" ]]; then
    refuseUpdate "Deployment checkout has local changes; refusing to update:
$changed_paths"
fi
git merge-base --is-ancestor "$current_commit" "$target_commit"
command -v uv >/dev/null
echo "Draining KernelBench before deploying $target_commit"
phase=draining
writeState updating
# SIGTERM reaches only the main process. Its startup/job children finish before exit.
systemctl --user stop kernelbench.service
git merge --ff-only "$target_commit"
phase=dependencies
uv sync --locked --extra embeddinggemma --extra ml
systemctl --user start kernelbench.service
phase=readiness
for ((attempt = 0; attempt < 2700; attempt++)); do
    if ! systemctl --user is-active --quiet kernelbench.service; then
        refuseUpdate 'KernelBench failed to start; inspect journalctl --user -u kernelbench'
    fi
    if curl --fail --silent --max-time 2 http://127.0.0.1:8080/welcome >/dev/null; then
        echo "KernelBench healthy at $target_commit"
        phase=ready
        writeState current
        : > "$state_directory/last-refusal"
        exit 0
    fi
    sleep 1
done
refuseUpdate 'KernelBench readiness timed out; inspect journalctl --user -u kernelbench'
