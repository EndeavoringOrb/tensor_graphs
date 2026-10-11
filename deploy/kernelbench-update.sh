#!/usr/bin/env bash
set -euo pipefail

checkout="${KERNEL_BENCH_CHECKOUT:-$HOME/services/kernel_bench}"
state_directory="${KERNEL_BENCH_STATE_DIR:-$HOME/.cache/kernelbench}"
mkdir -p "$state_directory"
exec 9>"$state_directory/update.lock"
flock -n 9 || exit 0
cd "$checkout"
export GIT_TERMINAL_PROMPT=0
git fetch --no-tags origin main
target_commit=$(git rev-parse origin/main)
current_commit=$(git rev-parse HEAD)
if [[ "$target_commit" == "$current_commit" ]]; then
    exit 0
fi
if [[ "$(git branch --show-current)" != main ]]; then
    echo 'Deployment checkout must be on main' >&2
    exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
    echo 'Deployment checkout has local changes; refusing to update' >&2
    exit 1
fi
git merge-base --is-ancestor "$current_commit" "$target_commit"
command -v uv >/dev/null
echo "Draining KernelBench before deploying $target_commit"
# SIGTERM reaches only the main process. Its startup/job children finish before exit.
systemctl --user stop kernelbench.service
git merge --ff-only "$target_commit"
uv sync --locked
systemctl --user start kernelbench.service
for ((attempt = 0; attempt < 2700; attempt++)); do
    if ! systemctl --user is-active --quiet kernelbench.service; then
        echo 'KernelBench failed to start; inspect journalctl --user -u kernelbench' >&2
        exit 1
    fi
    if curl --fail --silent --max-time 2 http://127.0.0.1:8080/welcome >/dev/null; then
        echo "KernelBench healthy at $target_commit"
        exit 0
    fi
    sleep 1
done
echo 'KernelBench readiness timed out; inspect journalctl --user -u kernelbench' >&2
exit 1
