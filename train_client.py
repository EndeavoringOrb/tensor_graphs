# File: train_client.py
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
    AlphaZeroTransformer,
    TrainConfig,
    create_client_socket,
    get_graph_provider,
    recv_msg,
    send_msg,
)


@torch.inference_mode()
def client_worker(rank: int, config: TrainConfig):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

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

    agent = AlphaZeroTransformer(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        max_feat_dim=config.max_feat_dim,
    )

    # 1. Connect to Server
    conn_type = "Bluetooth" if config.use_bluetooth else "TCP"
    logger.info(
        f"{LOG_PREFIX} [Worker {rank}] Connecting to {conn_type} Server at {config.host}:{config.port}..."
    )
    try:
        client_sock = create_client_socket(config)
        logger.info(f"{LOG_PREFIX} [Worker {rank}] Connected successfully!")
    except Exception as e:
        logger.info(f"{LOG_PREFIX} [Worker {rank}] Connection failed: {e}")
        return

    # 2. Fetch Initial Weights from Server
    try:
        send_msg(client_sock, {"type": "req_weights"})
        initial_resp = recv_msg(client_sock)
        if (
            initial_resp
            and initial_resp.get("type") == "weights"
            and initial_resp.get("data")
        ):
            agent.load_state_dict(initial_resp["data"], strict=False)
            logger.info(
                f"{LOG_PREFIX} [Worker {rank}] Loaded initial network weights from server."
            )
        else:
            logger.info(
                f"{LOG_PREFIX} [Worker {rank}] Using local randomly initialized weights."
            )
    except Exception as e:
        logger.info(f"{LOG_PREFIX} [Worker {rank}] Error fetching initial weights: {e}")
        client_sock.close()
        return

    # 3. Setup Graph Provider
    graph_provider = get_graph_provider(config, worker_rank=rank)
    logger.info(
        f"{LOG_PREFIX} [Worker {rank}] Initializing graph provider (source: {config.graph_source})..."
    )

    episode = 0

    while True:
        try:
            egraph_context = graph_provider.get_context(config, episode=episode)
        except Exception as e:
            logger.info(
                f"{LOG_PREFIX} [Worker {rank}] Error obtaining E-Graph context at episode {episode}: {e}"
            )
            traceback.print_exc()
            break

        num_buckets = getattr(egraph_context, "num_buckets", 1)
        bucket_idx = (
            config.bucket_idx
            if config.bucket_idx >= 0
            else (rank % max(1, num_buckets))
        )

        agent.eval()
        best_cost = float("inf")
        extraction_costs = []
        mcts_tree = {}

        # 4. Run MCTS Simulations
        for sim in range(config.num_simulations):
            delegate = ActorDelegate(
                agent,
                mcts_tree=mcts_tree,
                c_puct=config.c_puct,
                episode=episode,
                decay_episodes=config.decay_episodes,
                base_noise=config.base_noise,
                min_noise=config.min_noise,
                depth_gamma=config.depth_gamma,
            )
            try:
                costs = tensor_graphs.run_hierarchical_simulations(
                    egraph_context,
                    bucket_idx,
                    delegate,
                    config.level_simulations,
                    config.log_cost_calls,
                )
            except Exception as e:
                logger.info(
                    f"{LOG_PREFIX} [Worker {rank}] Error during simulation: {e}"
                )
                costs = []

            for cost in costs:
                if cost < float("inf"):
                    extraction_costs.append(float(cost))
                    best_cost = min(best_cost, cost)

        if best_cost < float("inf"):
            best_Z = 1000.0 / (best_cost + 1.0)
        else:
            best_Z = -1.0

        # 5. Pack Trajectory
        trajectory_transitions = []
        for h, node_data in mcts_tree.items():
            counts = node_data["N"]
            total_counts = counts.sum()
            pi = counts / total_counts if total_counts > 0 else node_data["P"]

            trajectory_transitions.append(
                {
                    "features": node_data["features"],
                    "token_types": node_data["token_types"],
                    "phase_ids": node_data["phase_ids"],
                    "pis": pi,
                    "z": best_Z,
                }
            )

        ep_noise = max(
            config.min_noise,
            config.base_noise * (1.0 - episode / max(1, config.decay_episodes)),
        )
        logger.info(
            f"{LOG_PREFIX} [Worker {rank}] Ep {episode:03d} | Ep Noise: {ep_noise:.4f} | Best Cost: {best_cost:8.4f} ms | "
            f"Extractions: {len(extraction_costs)} | Sending {len(trajectory_transitions)} transitions..."
        )

        # 6. Stream Trajectory to Server
        try:
            send_msg(
                client_sock,
                {
                    "type": "trajectory",
                    "cost": best_cost,
                    "costs": extraction_costs,
                    "data": trajectory_transitions,
                },
            )
        except Exception as e:
            logger.info(
                f"{LOG_PREFIX} [Worker {rank}] Error sending trajectory to server: {e}"
            )
            break

        episode += 1

        # 7. Check for updated weights
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
        "-bt",
        "--use-bluetooth",
        action="store_true",
        help="Use Bluetooth RFCOMM socket",
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
        default=10,
        help="Number of MCTS simulations per episode",
    )
    parser.add_argument(
        "--graph-source",
        type=str,
        default="model",
        choices=["model", "random"],
        help="Graph provider source: 'model' (load from file/path) or 'random'",
    )
    parser.add_argument("--model", type=str, default="gemma-3-270m", help="Model name")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/google/gemma-3-270m",
        help="Model weights path",
    )
    parser.add_argument(
        "--random-min-nodes",
        type=int,
        default=10,
        help="Minimum number of nodes for random graph generation",
    )
    parser.add_argument(
        "--random-max-nodes",
        type=int,
        default=30,
        help="Maximum number of nodes for random graph generation",
    )
    parser.add_argument(
        "--random-dim",
        type=int,
        default=128,
        help="Hidden dimension for random graph nodes",
    )
    parser.add_argument(
        "--random-seq-len",
        type=int,
        default=64,
        help="Sequence length dimension for random graph nodes",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Base seed for deterministic random graph generation",
    )
    parser.add_argument(
        "--resample-graph-every",
        type=int,
        default=0,
        help="Resample a new random graph every N episodes (0 = fixed per worker)",
    )
    parser.add_argument(
        "--compile-decode-buckets",
        action="store_true",
        help="Compile decode buckets in addition to the single full bucket",
    )
    parser.add_argument(
        "--log-lost-calls",
        action="store_true",
        dest="log_cost_calls",
        help="Log cost calls (forces workers=1 when enabled)",
    )
    parser.add_argument(
        "--c-puct", type=float, default=1.25, help="PUCT exploration constant"
    )
    parser.add_argument(
        "--base-noise",
        type=float,
        default=0.25,
        help="Initial exploration noise at episode 0 and depth 0",
    )
    parser.add_argument(
        "--min-noise", type=float, default=0.01, help="Minimum exploration noise floor"
    )
    parser.add_argument(
        "--decay-episodes",
        type=int,
        default=500,
        help="Number of episodes over which to decay episode-level noise",
    )
    parser.add_argument(
        "--depth-gamma", type=float, default=0.7, help="Per-depth noise decay factor"
    )
    parser.add_argument(
        "--level-sims",
        nargs="+",
        type=int,
        default=[1, 1, 1, 1],
        help="Simulations per level: [num_extract, num_dispatch, num_bufferize, num_malloc] or with num_cache",
    )

    args = parser.parse_args()

    config = TrainConfig()
    config.host = args.host
    config.port = args.port
    config.use_bluetooth = args.use_bluetooth
    config.workers = 1 if args.log_cost_calls else args.workers
    config.num_simulations = args.simulations
    config.graph_source = args.graph_source
    config.model_name = args.model
    config.model_path = args.model_path
    config.random_min_nodes = args.random_min_nodes
    config.random_max_nodes = args.random_max_nodes
    config.random_hidden_dim = args.random_dim
    config.random_seq_len = args.random_seq_len
    config.random_seed = args.random_seed
    config.resample_graph_every = args.resample_graph_every
    config.compile_decode_buckets = args.compile_decode_buckets
    config.log_cost_calls = args.log_cost_calls
    config.c_puct = args.c_puct
    config.base_noise = args.base_noise
    config.min_noise = args.min_noise
    config.decay_episodes = args.decay_episodes
    config.depth_gamma = args.depth_gamma
    config.level_simulations = args.level_sims

    if config.host and ":" in config.host and len(config.host.split(":")) == 6:
        config.use_bluetooth = True

    if config.graph_source == "model":
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
    print(f" Graph Source: {config.graph_source.upper()}")
    if config.graph_source == "random":
        print(
            f" Random Node Range: [{config.random_min_nodes}, {config.random_max_nodes}], Dim: {config.random_hidden_dim}, Seq: {config.random_seq_len}"
        )
    print(
        f" MCTS Settings: c_puct={config.c_puct}, base_noise={config.base_noise}, min_noise={config.min_noise}"
    )
    print("=========================================================")

    processes = []
    for rank in range(config.workers):
        p = mp.Process(target=client_worker, args=(rank, config))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
