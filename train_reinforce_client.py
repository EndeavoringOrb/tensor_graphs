#!/usr/bin/env python3
# File: train_reinforce_client.py
import argparse
import logging
import math
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import psutil
import torch
import torch.multiprocessing as mp

DEFAULT_WORKERS = max(1, (psutil.cpu_count(logical=False) or 4) - 1)
torch.set_float32_matmul_precision("high")

import tensor_graphs

from train_models import PolicyValueRNN
from train_shared import (
    RNNREINFORCEDelegate,
    TrainConfig,
    create_client_socket,
    get_default_model_path,
    get_graph_provider,
    recv_msg,
    send_msg,
)


def reinforce_worker(
    rank: int,
    config: TrainConfig,
    weights_path_str: str,
    weights_event: mp.Event,
    results_queue: mp.Queue,
    temperature: float = 1.0,
    cpp_threads: int = 1,
):
    # 1. Duplicate actual console stdout for clean progress updates
    real_stdout_fd = os.dup(1)
    real_stdout = os.fdopen(real_stdout_fd, "w", buffering=1)

    # 2. Redirect standard OS file descriptors 1 and 2 to per-worker log file
    log_path = Path(f"client_worker_{rank}.log")
    f_log = open(log_path, "w", encoding="utf-8")
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(f_log.fileno(), 1)
    os.dup2(f_log.fileno(), 2)

    # 3. On Windows, redirect underlying Win32 handles used by C++ std::cout / std::cerr
    if os.name == "nt":
        import ctypes
        import msvcrt

        os_handle = msvcrt.get_osfhandle(f_log.fileno())
        ctypes.windll.kernel32.SetStdHandle(-11, os_handle)  # STD_OUTPUT_HANDLE
        ctypes.windll.kernel32.SetStdHandle(-12, os_handle)  # STD_ERROR_HANDLE

    sys.stdout = f_log
    sys.stderr = f_log

    # 4. Logger setup: only lines with LOG_PREFIX get mirrored to real console
    LOG_PREFIX = "[CLIENT]"

    logger = logging.getLogger(f"Worker_{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    class SpecialPrefixFilter(logging.Filter):
        def filter(self, record):
            return record.getMessage().startswith(LOG_PREFIX)

    console_handler = logging.StreamHandler(real_stdout)
    console_handler.addFilter(SpecialPrefixFilter())
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    tensor_graphs.set_num_threads(cpp_threads)

    device = torch.device("cpu")
    model = PolicyValueRNN(hidden_dim=config.d_model, global_dim=8).to(device)
    model.eval()

    weights_path = Path(weights_path_str)
    current_version = -1

    graph_provider = get_graph_provider(config, worker_rank=rank)
    logger.info(f"{LOG_PREFIX} [Worker {rank}] Initialized on graph source: {config.graph_source}")

    episode = 0
    delegate = RNNREINFORCEDelegate(
        model=model,
        temperature=temperature,
        is_training=True,
        device=device,
    )

    while True:
        if weights_event.is_set() or current_version == -1:
            if weights_path.exists():
                try:
                    loaded = torch.load(weights_path, map_location="cpu", weights_only=True)
                    if isinstance(loaded, dict) and "state_dict" in loaded:
                        model.load_state_dict(loaded["state_dict"], strict=False)
                        current_version = loaded.get("version", current_version + 1)
                    else:
                        model.load_state_dict(loaded, strict=False)
                        current_version += 1
                except Exception as e:
                    logger.info(f"{LOG_PREFIX} [Worker {rank}] Weight reload error: {e}")
            if rank == 0:
                weights_event.clear()

        try:
            egraph_context = graph_provider.get_context(config, episode=episode)
        except Exception as e:
            logger.info(f"{LOG_PREFIX} [Worker {rank}] Error obtaining E-Graph context at episode {episode}: {e}")
            traceback.print_exc()
            break

        num_buckets = getattr(egraph_context, "num_buckets", 1)
        bucket_idx = (
            config.bucket_idx if config.bucket_idx >= 0 else (rank % max(1, num_buckets))
        )

        delegate.reset_for_episode(mem_caps={1: 16 * 1024 * 1024 * 1024})

        try:
            costs = tensor_graphs.run_hierarchical_simulations(
                egraph_context,
                bucket_idx,
                delegate,
                config.level_simulations,
                config.log_cost_calls,
            )
        except Exception as e:
            logger.info(f"{LOG_PREFIX} [Worker {rank}] Error in simulation: {e}")
            costs = []

        valid_costs = [c for c in costs if c < float("inf") and not math.isnan(c)]
        best_cost = min(valid_costs) if valid_costs else float("inf")

        if delegate.completed_episodes:
            results_queue.put(list(delegate.completed_episodes))
            logger.info(
                f"{LOG_PREFIX} [Worker {rank}] Ep {episode:03d} | Best Cost: {best_cost:8.4f} ms | "
                f"Completed Paths: {len(delegate.completed_episodes)}"
            )

        episode += 1


def main():
    parser = argparse.ArgumentParser(description="REINFORCE Client for TensorGraph Search")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("-bt", "--use-bluetooth", action="store_true")
    parser.add_argument("--bt-address", type=str, default=None)
    parser.add_argument("--bt-port", type=int, default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--cpp-threads", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--level-sims", nargs="+", type=int, default=[1, 1, 1, 1])
    parser.add_argument("--model", type=str, default="gemma-3-270m")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--graph-source", type=str, default="model", choices=["model", "random"])
    parser.add_argument("--random-min-nodes", type=int, default=10)
    parser.add_argument("--random-max-nodes", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=64)

    args = parser.parse_args()

    net_config = TrainConfig()
    net_config.host = args.host
    net_config.port = args.port
    net_config.use_bluetooth = args.use_bluetooth
    if args.bt_address is not None:
        net_config.bt_host_address = args.bt_address
    if args.bt_port is not None:
        net_config.bt_port = args.bt_port

    client_sock = create_client_socket(net_config)
    sock_lock = threading.Lock()

    print("[Client] Fetching base configuration from server...")
    with sock_lock:
        send_msg(client_sock, {"type": "req_config"})
        resp = recv_msg(client_sock)

    if resp and resp.get("type") == "config" and resp.get("config"):
        config = TrainConfig.from_dict(resp["config"])
    else:
        config = TrainConfig()

    config.host = net_config.host
    config.port = net_config.port
    config.use_bluetooth = net_config.use_bluetooth
    config.level_simulations = args.level_sims
    config.graph_source = args.graph_source
    config.model_name = args.model
    config.d_model = args.hidden_dim
    config.model_path = args.model_path or get_default_model_path(args.model)
    config.random_min_nodes = args.random_min_nodes
    config.random_max_nodes = args.random_max_nodes

    runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    weights_path = runs_dir / "reinforce_client_weights.pt"
    weights_event = mp.Event()
    results_queue = mp.Queue()

    # Weight Sync Thread
    current_version = -1

    def weight_sync_loop():
        nonlocal current_version
        while True:
            try:
                with sock_lock:
                    send_msg(client_sock, {"type": "req_version"})
                    ver_resp = recv_msg(client_sock)

                if ver_resp and ver_resp.get("type") == "version":
                    s_ver = ver_resp.get("version", 0)
                    if s_ver > current_version or current_version == -1:
                        with sock_lock:
                            send_msg(client_sock, {"type": "req_weights"})
                            w_resp = recv_msg(client_sock)

                        if w_resp and w_resp.get("type") == "weights" and w_resp.get("data"):
                            torch.save(
                                {"version": s_ver, "state_dict": w_resp["data"]},
                                weights_path,
                            )
                            current_version = s_ver
                            weights_event.set()
                            print(f"[Client] Updated model weights to version {current_version}")
                time.sleep(5)
            except Exception as e:
                print(f"[Client] Weight sync error: {e}")
                time.sleep(3)

    threading.Thread(target=weight_sync_loop, daemon=True).start()

    # Start Worker Processes
    processes = []
    for rank in range(args.workers):
        p = mp.Process(
            target=reinforce_worker,
            args=(
                rank,
                config,
                str(weights_path),
                weights_event,
                results_queue,
                args.temperature,
                args.cpp_threads,
            ),
        )
        p.start()
        processes.append(p)

    print(f"[Client] Running {args.workers} REINFORCE exploration workers...")

    # Forward completed episodes to server
    while True:
        try:
            episodes = results_queue.get(timeout=1.0)
            with sock_lock:
                send_msg(client_sock, {"type": "episodes", "episodes": episodes})
        except Exception:
            pass


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()