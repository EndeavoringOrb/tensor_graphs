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
import numpy as np

DEFAULT_WORKERS = max(1, (psutil.cpu_count(logical=False) or 4) - 1)

import tensor_graphs

from train_shared import (
    ActorDelegate,
    AlphaZeroAgent,
    TrainConfig,
    create_client_socket,
    recv_msg,
    send_msg,
    PHASE_MAP,
    MAX_GNN_DIM,
    MAX_OPT_DIM,
)


# Pad helper: ensure arrays are padded to MAX dims for unified buffer
def pad_gnn(arr, target_dim=MAX_GNN_DIM):
    if arr.shape[1] >= target_dim:
        return arr[:, :target_dim]
    pad = np.zeros((arr.shape[0], target_dim - arr.shape[1]), dtype=np.float32)
    return np.concatenate([arr, pad], axis=1)


def pad_opt(arr, target_dim=MAX_OPT_DIM):
    if arr.shape[1] >= target_dim:
        return arr[:, :target_dim]
    pad = np.zeros((arr.shape[0], target_dim - arr.shape[1]), dtype=np.float32)
    return np.concatenate([arr, pad], axis=1)


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

    agent = AlphaZeroAgent(
        hidden_dim=config.hidden_dim,
        transformer_layers=config.transformer_layers,
        transformer_heads=config.transformer_heads,
        dropout=config.transformer_dropout,
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

    # 2. Fetch Initial Weights
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
                f"{LOG_PREFIX} [Worker {rank}] Using local randomly initialized weights (server weights pending)."
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
            config.model_name,
            config.model_path,
            config.log_cost_calls,
            config.compile_decode_buckets,
        )
    except Exception as e:
        logger.info(
            f"{LOG_PREFIX} [Worker {rank}] Error building saturated E-Graph: {e}"
        )
        traceback.print_exc()
        client_sock.close()
        return

    num_buckets = getattr(egraph_context, "num_buckets", 1)
    bucket_idx = (
        config.bucket_idx if config.bucket_idx >= 0 else (rank % max(1, num_buckets))
    )
    logger.info(
        f"{LOG_PREFIX} [Worker {rank}] Assigned to bucket {bucket_idx}/{num_buckets}"
    )

    # 4. Generate Trajectories - optimized for thousands of transitions per episode
    episode = 0

    while True:
        agent.eval()
        best_cost = float("inf")
        extraction_costs = []
        mcts_tree = {}

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

        # 5. Pack unified trajectory - optimized for thousands
        # Pre-allocate lists with expected size
        num_transitions = len(mcts_tree)
        phase_list = []
        nf_list = []
        esrc_list = []
        edst_list = []
        feats_list = []
        pis_list = []
        zs_list = []

        # Reserve capacity hint (Python list doesn't have reserve, but we can extend efficiently)
        # Use local variables for speed
        for node_data in mcts_tree.values():
            counts = node_data["N"]
            total_counts = counts.sum()
            pi = counts / total_counts if total_counts > 0 else node_data["P"]

            # node_data["node_features"] is already padded to MAX_GNN_DIM in new ActorDelegate
            # but ensure padding for safety
            nf = node_data["node_features"]
            if nf.shape[1] < MAX_GNN_DIM:
                nf = pad_gnn(nf, MAX_GNN_DIM)
            else:
                nf = nf[:, :MAX_GNN_DIM]

            feat = node_data["features"]
            if feat.shape[1] < MAX_OPT_DIM:
                feat = pad_opt(feat, MAX_OPT_DIM)

            phase_list.append(
                int(
                    node_data.get(
                        "phase", PHASE_MAP.get(node_data.get("type", "cache_dec"), 0)
                    )
                )
            )
            nf_list.append(nf.astype(np.float32, copy=False))
            esrc_list.append(node_data["edge_src"].astype(np.int64, copy=False))
            edst_list.append(node_data["edge_dst"].astype(np.int64, copy=False))
            feats_list.append(feat.astype(np.float32, copy=False))
            pis_list.append(pi.astype(np.float32, copy=False))
            zs_list.append(best_Z)

        total_transitions = len(zs_list)

        ep_noise = max(
            config.min_noise,
            config.base_noise * (1.0 - episode / max(1, config.decay_episodes)),
        )
        logger.info(
            f"{LOG_PREFIX} [Worker {rank}] Ep {episode:03d} | Ep Noise: {ep_noise:.4f} | Best Cost: {best_cost:8.4f} ms | "
            f"Extractions: {len(extraction_costs)} | Sending {total_transitions} transitions (unified)..."
        )

        trajectory_payload = {
            "phase": phase_list,
            "node_features": nf_list,
            "edge_src": esrc_list,
            "edge_dst": edst_list,
            "features": feats_list,
            "pis": pis_list,
            "Zs": zs_list,
        }

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

        # 6. Check for updated weights
        try:
            send_msg(client_sock, {"type": "req_weights"})
            weight_resp = recv_msg(client_sock)
            if weight_resp and weight_resp.get("type") == "weights":
                agent.load_state_dict(weight_resp["data"], strict=False)
        except Exception as e:
            logger.info(f"{LOG_PREFIX} [Worker {rank}] Error fetching weights: {e}")
            break

    client_sock.close()


def main():
    parser = argparse.ArgumentParser(
        description="AlphaZero TensorGraph Client - Unified MDP"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model", type=str, default="gemma-3-270m")
    parser.add_argument("--model-path", type=str, default="models/google/gemma-3-270m")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--num-simulations", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)

    args = parser.parse_args()

    config = TrainConfig()
    config.host = args.host
    config.port = args.port
    config.model_name = args.model
    config.model_path = args.model_path
    config.num_simulations = args.num_simulations
    config.hidden_dim = args.hidden_dim
    config.transformer_layers = args.transformer_layers
    config.transformer_heads = args.transformer_heads

    # Use spawn for multiprocessing
    mp.set_start_method("spawn", force=True)

    processes = []
    for rank in range(args.workers):
        p = mp.Process(target=client_worker, args=(rank, config))
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Shutting down client workers...")
        for p in processes:
            p.terminate()


if __name__ == "__main__":
    main()
