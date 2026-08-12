import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["KMP_AFFINITY"] = "none"

import argparse
import logging
import sys
import traceback
from pathlib import Path

import psutil
import torch
import torch.multiprocessing as mp

DEFAULT_WORKERS = max(1, (psutil.cpu_count(logical=False) or 4) - 1)

import tensor_graphs

from train_shared import (
    ActorDelegate,
    AlphaZeroAgent,
    TrainConfig,
    create_client_socket,
    recv_msg,
    send_msg,
)


@torch.inference_mode()
def client_worker(rank: int, config: TrainConfig):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # Save original stdout fd before dup2 redirection so we can print filtered logger output to terminal
    real_stdout_fd = os.dup(1)
    real_stdout = os.fdopen(real_stdout_fd, "w", buffering=1)

    log_path = f"client_worker_{rank}.log"
    f_log = open(log_path, "w", encoding="utf-8")

    sys.stdout.flush()
    sys.stderr.flush()

    os.dup2(f_log.fileno(), 1)
    os.dup2(f_log.fileno(), 2)

    if os.name == "nt":
        import ctypes
        import msvcrt

        os_handle = msvcrt.get_osfhandle(f_log.fileno())
        ctypes.windll.kernel32.SetStdHandle(-11, os_handle)  # STD_OUTPUT_HANDLE
        ctypes.windll.kernel32.SetStdHandle(-12, os_handle)  # STD_ERROR_HANDLE

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

    agent = AlphaZeroAgent(hidden_dim=config.hidden_dim)

    # 1. Connect to Server
    conn_type = "Bluetooth" if config.use_bluetooth else "TCP"
    logger.info(
        f"{LOG_PREFIX} [Worker {rank}] Connecting to {conn_type} Server at {config.host}:{config.port}..."
    )
    try:
        client_sock = create_client_socket(
            config.host, config.port, config.use_bluetooth
        )
        logger.info(f"{LOG_PREFIX} [Worker {rank}] Connected successfully!")
    except Exception as e:
        logger.info(f"{LOG_PREFIX} [Worker {rank}] Connection failed: {e}")
        return

    # 2. Fetch Initial Weights from Server
    try:
        send_msg(client_sock, {"type": "req_weights"})
        initial_resp = recv_msg(client_sock)
        if initial_resp and initial_resp.get("type") == "weights":
            agent.load_state_dict(initial_resp["data"])
            logger.info(
                f"{LOG_PREFIX} [Worker {rank}] Loaded initial network weights from server."
            )
    except Exception as e:
        logger.info(f"{LOG_PREFIX} [Worker {rank}] Error fetching initial weights: {e}")
        client_sock.close()
        return

    # 3. Setup and Cache the Saturated E-Graph
    logger.info(
        f"{LOG_PREFIX} [Worker {rank}] Building and caching saturated E-Graph for {config.model_name}..."
    )
    try:
        egraph_context = tensor_graphs.build_and_saturate_egraph(
            config.model_name, config.model_path
        )
    except Exception as e:
        logger.info(
            f"{LOG_PREFIX} [Worker {rank}] Error building saturated E-Graph: {e}"
        )
        traceback.print_exc()
        client_sock.close()
        return

    # 4. Generate Trajectories
    episode = 0
    while True:
        agent.eval()
        best_cost = float("inf")
        state_visits = {}
        extraction_costs = []

        # MCTS Simulations using the already saturated e-graph
        for sim in range(config.num_simulations):
            delegate = ActorDelegate(agent, exploration_noise=0.25)
            try:
                # Fast extraction only
                cost = tensor_graphs.extract_best_from_egraph(
                    egraph_context, delegate, config.log_cost_calls
                )
            except Exception as e:
                logger.info(
                    f"{LOG_PREFIX} [Worker {rank}] Error during extraction: {e}"
                )
                cost = float("inf")

            if cost < float("inf"):
                extraction_costs.append(float(cost))
                best_cost = min(best_cost, cost)

            # Build trajectory MCTS target distribution
            for step in delegate.trajectory:
                h = hash(step["features"].tobytes())
                if h not in state_visits:
                    state_visits[h] = {
                        "counts": torch.zeros_like(torch.tensor(step["P"])),
                        "data": step,
                    }
                state_visits[h]["counts"][step["top_action"]] += 1

        # Calculate Z and normalize targets
        if best_cost < float("inf"):
            Z = 1000.0 / (best_cost + 1.0)
        else:
            Z = -1.0

        trajectory_payload = []
        for h, state_info in state_visits.items():
            data = state_info["data"]
            counts = state_info["counts"]

            if counts.sum() > 0:
                pi = (counts / counts.sum()).numpy()
            else:
                pi = data["P"]

            trajectory_payload.append(
                {
                    "type": data["type"],
                    "global_state": data["global_state"],
                    "features": data["features"],
                    "pi": pi,
                    "Z": Z,
                }
            )

        logger.info(
            f"{LOG_PREFIX} [Worker {rank}] Ep {episode:03d} | Best Cost: {best_cost:8.4f} ms | "
            f"Extractions: {len(extraction_costs)} | Sending {len(trajectory_payload)} transitions..."
        )

        # 5. Stream Trajectory and ALL extraction costs to Server
        try:
            send_msg(
                client_sock,
                {
                    "type": "trajectory",
                    "cost": best_cost,
                    "costs": extraction_costs,
                    "data": trajectory_payload,
                },
            )
        except Exception as e:
            logger.info(
                f"{LOG_PREFIX} [Worker {rank}] Error sending trajectory to server: {e}"
            )
            break

        episode += 1

        # 6. Check for updated weights periodically
        if episode % 5 == 0:
            try:
                send_msg(client_sock, {"type": "req_weights"})
                weight_resp = recv_msg(client_sock)
                if weight_resp and weight_resp.get("type") == "weights":
                    agent.load_state_dict(weight_resp["data"])
                    logger.info(
                        f"{LOG_PREFIX} [Worker {rank}] Synced updated weights from server."
                    )
            except Exception as e:
                logger.info(
                    f"{LOG_PREFIX} [Worker {rank}] Error syncing weights from server: {e}"
                )
                break

    client_sock.close()


def main():
    parser = argparse.ArgumentParser(description="AlphaZero TensorGraph Worker Client")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server address (IP or Bluetooth MAC)",
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Server port or BT channel"
    )
    parser.add_argument(
        "--use-bluetooth", action="store_true", help="Use Bluetooth RFCOMM socket"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=800,
        help="Number of MCTS simulations per episode",
    )
    parser.add_argument("--model", type=str, default="gemma-3-270m", help="Model name")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/google/gemma-3-270m",
        help="Model weights path",
    )
    parser.add_argument("--log-cost-calls", action="store_true", help="Log cost calls")

    args = parser.parse_args()

    config = TrainConfig()
    config.host = args.host
    config.port = args.port
    config.use_bluetooth = args.use_bluetooth
    config.workers = args.workers
    config.num_simulations = args.simulations
    config.model_name = args.model
    config.model_path = args.model_path
    config.log_cost_calls = args.log_cost_calls

    if config.host and ":" in config.host and len(config.host.split(":")) == 6:
        config.use_bluetooth = True

    # Auto-download model safetensors / metadata before launching workers
    try:
        from utils.download_hf_meta import download_model_meta

        p = Path(config.model_path)
        if p.is_file() or p.suffix == ".safetensors":
            p = p.parent
        parts = p.parts
        if parts and parts[0] == "models":
            parts = parts[1:]
        repo_id = "/".join(parts) if parts else config.model_name

        print(f"[Client] Ensuring model files for '{repo_id}' are downloaded...")
        download_model_meta(repo_id, download_other_files=False)
    except Exception as e:
        print(
            f"[Client] Note: Could not auto-download model metadata ({e}). Proceeding assuming local files exist."
        )

    conn_type = "Bluetooth" if config.use_bluetooth else "TCP/IP"
    print("=========================================================")
    print(f" Starting {config.workers} Client Worker Process(es)")
    print(f" Target Server: {config.host}:{config.port} ({conn_type})")
    print("=========================================================")

    processes = []
    for rank in range(config.workers):
        p = mp.Process(target=client_worker, args=(rank, config))
        p.start()
        processes.append(p)

    # Wait for completion (Manual exit via Ctrl+C)
    for p in processes:
        p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
