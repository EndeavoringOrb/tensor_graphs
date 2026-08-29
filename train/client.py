import argparse
import logging
import math
import os
import queue
import random
import sys
import threading
import time
from pathlib import Path
import numpy as np
import tensor_graphs
import psutil
import torch
import torch.multiprocessing as mp

from .config import TrainConfig
from .delegate import CostPredictorDelegate
from .graph_provider import get_graph_provider
from .model import CostPredictorRNN
from .net import create_client_socket, recv_msg, send_msg

torch.set_float32_matmul_precision("high")


def client_worker_process(
    rank: int,
    config: TrainConfig,
    weights_path_str: str,
    weights_event: mp.Event,
    traj_queue: mp.Queue,
):
    """In-process search worker performing its own RNN inference."""
    # Seed per worker
    worker_seed = (
        int(time.time() * 1000) ^ (os.getpid() << 16) ^ (rank * 10007)
    ) & 0x7FFFFFFF
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    # Clean per-worker file logging with terminal forwarding for [CLIENT] prefix
    real_stdout_fd = os.dup(1)
    real_stdout = os.fdopen(real_stdout_fd, "w", buffering=1)

    log_path = Path(f"client_worker_{rank}.log")
    f_log = open(log_path, "w", encoding="utf-8")
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(f_log.fileno(), 1)
    os.dup2(f_log.fileno(), 2)

    if os.name == "nt":
        import ctypes
        import msvcrt

        os_handle = msvcrt.get_osfhandle(f_log.fileno())
        ctypes.windll.kernel32.SetStdHandle(-11, os_handle)
        ctypes.windll.kernel32.SetStdHandle(-12, os_handle)

    sys.stdout = f_log
    sys.stderr = f_log

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
    tensor_graphs.set_num_threads(config.cpp_threads)

    device = torch.device("cpu")
    model = CostPredictorRNN(hidden_dim=config.hidden_dim).to(device)
    model.eval()

    weights_path = Path(weights_path_str)
    current_version = -1

    graph_provider = get_graph_provider(config, worker_rank=rank)
    logger.info(
        f"{LOG_PREFIX} [Worker {rank}] Initialized local inference worker (source: {config.graph_source})"
    )

    delegate = CostPredictorDelegate(
        model=model,
        epsilon=config.epsilon,
        is_training=True,
        device=device,
    )

    episode = 0

    while True:
        start = time.perf_counter()
        delegate.epsilon = config.epsilon * (random.random() ** 2)

        # Check and reload latest weights from disk
        if weights_event.is_set() or current_version == -1:
            if weights_path.exists():
                try:
                    loaded = torch.load(
                        weights_path, map_location="cpu", weights_only=True
                    )
                    if isinstance(loaded, dict) and "state_dict" in loaded:
                        model.load_state_dict(loaded["state_dict"], strict=False)
                        current_version = loaded.get("version", current_version + 1)
                    else:
                        model.load_state_dict(loaded, strict=False)
                        current_version += 1
                except Exception as e:
                    logger.info(
                        f"{LOG_PREFIX} [Worker {rank}] Weight reload error: {e}"
                    )
            if rank == 0:
                weights_event.clear()

        try:
            egraph_context = graph_provider.get_context(config, episode=episode)
        except Exception as e:
            logger.info(
                f"{LOG_PREFIX} [Worker {rank}] Error loading graph context: {e}"
            )
            break

        num_buckets = getattr(egraph_context, "num_buckets", 1)
        bucket_idx = rank % max(1, num_buckets)

        delegate.reset()

        try:
            costs = tensor_graphs.run_hierarchical_simulations(
                egraph_context,
                bucket_idx,
                delegate,
                [1, 1, 1, 1],  # Standard single extraction pass
                False,
            )
        except Exception as e:
            logger.info(f"{LOG_PREFIX} [Worker {rank}] Simulation error: {e}")
            costs = []

        valid_positive_costs = [
            c for c in costs if c >= 0.0 and c < float("inf") and not math.isnan(c)
        ]
        best_cost = min(valid_positive_costs) if valid_positive_costs else float("inf")

        # Vectorize, sanitize, and pack completed trajectories on the client worker
        packed_by_phase, leaf_costs = delegate.export_and_reset()

        if packed_by_phase:
            total_transitions = sum(len(v["hiddens"]) for v in packed_by_phase.values())
            traj_queue.put(
                {
                    "by_phase": packed_by_phase,
                    "leaf_costs": leaf_costs,
                    "best_cost": best_cost,
                }
            )

            logger.info(
                f"{LOG_PREFIX} [Worker {rank}] Ep {episode:03d} (v{current_version}, eps {delegate.epsilon:.4e}) | "
                f"Best Cost: {best_cost:.4f} ms | {total_transitions} transitions | took {time.perf_counter() - start:.2f}s"
            )
        else:
            logger.info(f"{LOG_PREFIX} [Worker {rank}] Ep {episode:03d} (v{current_version}, eps {delegate.epsilon:.4e}) | took {time.perf_counter() - start:.2f}s")

        episode += 1
        logger.info(f"{LOG_PREFIX} [Worker {rank}] Ep {episode:03d}")


def main():
    parser = argparse.ArgumentParser(
        description="TensorGraph RNN Cost Predictor Training Client"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (psutil.cpu_count(logical=False) or 4) - 1),
        help="Worker processes",
    )
    parser.add_argument(
        "--cpp-threads", type=int, default=1, help="C++ threads per worker"
    )
    parser.add_argument(
        "--epsilon", type=float, default=None, help="Exploration epsilon"
    )
    parser.add_argument(
        "--graph-source",
        type=str,
        default=None,
        choices=["model", "random"],
        help="Workload graph source",
    )
    parser.add_argument("--model", type=str, default=None, help="Target model")
    parser.add_argument(
        "--model-path", type=str, default=None, help="Target model path"
    )
    args = parser.parse_args()

    # Connect to training server
    print(f"[Client] Connecting to training server at {args.host}:{args.port}...")
    sock = create_client_socket(args.host, args.port)
    sock_lock = threading.Lock()

    # Sync base configuration from server
    with sock_lock:
        send_msg(sock, {"type": "req_config"})
        resp = recv_msg(sock)

    if resp and resp.get("type") == "config" and resp.get("config"):
        config = TrainConfig.from_dict(resp["config"])
        print("[Client] Synchronized config from server.")
    else:
        config = TrainConfig()

    config.host = args.host
    config.port = args.port
    config.workers = args.workers
    config.cpp_threads = args.cpp_threads
    if args.epsilon is not None:
        config.epsilon = args.epsilon
    if args.graph_source is not None:
        config.graph_source = args.graph_source
    if args.model is not None:
        config.model_name = args.model
    if args.model_path is not None:
        config.model_path = args.model_path

    run_dir = Path(config.run_dir) if config.run_dir else Path("runs/0")
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_path = run_dir / "client_weights.pt"
    weights_event = mp.Event()

    # Thread: Periodically fetch updated weights from server
    def weight_sync_thread():
        current_version = -1
        while True:
            try:
                with sock_lock:
                    send_msg(sock, {"type": "req_version"})
                    ver_resp = recv_msg(sock)

                if ver_resp and ver_resp.get("type") == "version":
                    s_ver = ver_resp.get("version", 0)
                    if s_ver > current_version or current_version == -1:
                        with sock_lock:
                            send_msg(sock, {"type": "req_weights"})
                            w_resp = recv_msg(sock)

                        if (
                            w_resp
                            and w_resp.get("type") == "weights"
                            and w_resp.get("data")
                        ):
                            torch.save(
                                {
                                    "version": s_ver,
                                    "state_dict": w_resp["data"],
                                },
                                weights_path,
                            )
                            current_version = s_ver
                            weights_event.set()
                            print(
                                f"[Client] Synced updated model weights (version {current_version})."
                            )
                time.sleep(5)
            except Exception as e:
                print(f"[Client] Weight sync error: {e}")
                time.sleep(3)

    threading.Thread(target=weight_sync_thread, daemon=True).start()

    traj_queue = mp.Queue()
    print("=========================================================")
    print(" TensorGraph RNN Cost-Predictor Client")
    print(f" Starting {config.workers} In-Process Worker(s)")
    print(f" Server: {config.host}:{config.port}")
    print(f" Graph Source: {config.graph_source.upper()}")
    print("=========================================================")

    processes = []
    for rank in range(config.workers):
        p = mp.Process(
            target=client_worker_process,
            args=(
                rank,
                config,
                str(weights_path),
                weights_event,
                traj_queue,
            ),
        )
        p.start()
        processes.append(p)

    # Main process loop: transmits each trajectory to the server as soon as it arrives
    try:
        while True:
            try:
                batch_data = traj_queue.get(timeout=1.0)
                with sock_lock:
                    send_msg(sock, {"type": "transitions_batch", "data": batch_data})
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        print("\n[Client] Shutting down.")
    finally:
        for p in processes:
            p.terminate()
            p.join()
        sock.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
