# File: train_client.py
import argparse
import logging
import os
import queue
import sys
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path

import psutil
import torch
import torch.multiprocessing as mp

DEFAULT_WORKERS = max(1, (psutil.cpu_count(logical=False) or 4) - 1)
torch.set_float32_matmul_precision("high")

import tensor_graphs

from train_shared import (
    ActorDelegate,
    AlphaZeroTransformer,
    TrainConfig,
    TrajectoryCodec,
    create_client_socket,
    get_graph_provider,
    recv_msg,
    send_msg,
)


def inference_worker(config, req_queue, resp_queues, weights_event, run_dir):
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference Server] Started on {device}")

    agent = AlphaZeroTransformer(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        max_feat_dim=config.max_feat_dim,
    ).to(device)
    agent.eval()

    prefix_cache_kv = {}
    prefix_cache_v = {}
    weights_path = os.path.join(run_dir, "client_weights.pt")

    while True:
        if weights_event.is_set():
            if os.path.exists(weights_path):
                try:
                    weights_dict = torch.load(
                        weights_path, map_location="cpu", weights_only=True
                    )
                    agent.load_state_dict(weights_dict, strict=False)
                    prefix_cache_kv.clear()
                    prefix_cache_v.clear()
                    print("[Inference Server] Weights updated, KV cache cleared.")
                except Exception as e:
                    print(f"[Inference Server] Error loading weights: {e}")
            weights_event.clear()

        try:
            reqs = [req_queue.get(timeout=0.05)]
        except queue.Empty:
            continue

        while not req_queue.empty():
            reqs.append(req_queue.get_nowait())

        eval_reqs = []
        for req in reqs:
            if req[0] == "register_prefix":
                _, pkey, pdata = req
                if pkey not in prefix_cache_kv:
                    f = torch.tensor(
                        pdata.features, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    tt = torch.tensor(
                        pdata.token_types, dtype=torch.int64, device=device
                    ).unsqueeze(0)
                    pid = torch.tensor(
                        pdata.phase_ids, dtype=torch.int64, device=device
                    ).unsqueeze(0)

                    with (
                        torch.inference_mode(),
                        torch.autocast(device_type=device.type, dtype=torch.bfloat16),
                    ):
                        v, kv = agent.encode_prefix(f, tt, pid)

                    prefix_cache_kv[pkey] = kv
                    prefix_cache_v[pkey] = v.item() if v is not None else 0.0
            elif req[0] == "evaluate":
                eval_reqs.append(req)

        if not eval_reqs:
            continue

        valid_reqs = []
        for req in eval_reqs:
            _, pkey, a_feats, phase_id, wid = req
            if pkey not in prefix_cache_kv:
                resp_queues[wid].put(("error", "missing_prefix"))
                continue
            valid_reqs.append(req)

        # Group evaluation requests by prefix_key
        groups = defaultdict(list)
        for req in valid_reqs:
            groups[req[1]].append(req)

        for pkey, group_reqs in groups.items():
            B = len(group_reqs)
            max_A = max(req[2].shape[0] for req in group_reqs)
            padded_actions = torch.zeros(
                (B, max_A, 8), dtype=torch.float32, device=device
            )
            padded_pid = torch.zeros((B, max_A), dtype=torch.int64, device=device)

            # For the same pkey, past_kv length L is identical; expand across the group batch
            kv = prefix_cache_kv[pkey]
            batched_past_kv = [
                (k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1)) for (k, v) in kv
            ]

            for i, req in enumerate(group_reqs):
                _, _, a_feats, phase_id, wid = req
                A_len = a_feats.shape[0]
                dim_feat = min(7, a_feats.shape[1])
                padded_actions[i, :A_len, 1 : 1 + dim_feat] = torch.tensor(
                    a_feats[:, :dim_feat], dtype=torch.float32, device=device
                )
                padded_actions[i, :A_len, 0] = torch.arange(
                    A_len, dtype=torch.float32, device=device
                )
                padded_pid[i, :A_len] = phase_id

            with (
                torch.inference_mode(),
                torch.autocast(device_type=device.type, dtype=torch.bfloat16),
            ):
                logits = agent.evaluate_actions(
                    padded_actions, padded_pid, past_kv=batched_past_kv
                )

            for i, req in enumerate(group_reqs):
                _, _, a_feats, _, wid = req
                A_len = a_feats.shape[0]
                resp_logits = logits[i, :A_len].cpu().float().numpy()
                v = prefix_cache_v[pkey]
                resp_queues[wid].put(("ok", resp_logits, v))


@torch.inference_mode()
def client_worker(rank: int, config: TrainConfig, req_queue, resp_queue, traj_queue):
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

        best_cost = float("inf")
        extraction_costs = []
        mcts_tree = {}
        last_delegate = None

        for sim in range(config.num_simulations):
            delegate = ActorDelegate(
                agent=None,
                req_queue=req_queue,
                resp_queue=resp_queue,
                worker_id=rank,
                mcts_tree=mcts_tree,
                c_puct=config.c_puct,
                episode=episode,
                decay_episodes=config.decay_episodes,
                base_noise=config.base_noise,
                min_noise=config.min_noise,
                depth_gamma=config.depth_gamma,
            )
            last_delegate = delegate
            try:
                costs = tensor_graphs.run_hierarchical_simulations(
                    egraph_context,
                    bucket_idx,
                    delegate,
                    config.level_simulations,
                    config.log_cost_calls,
                )
            except Exception:
                logger.info(
                    f"{LOG_PREFIX} [Worker {rank}] Error during simulation: {traceback.format_exc()}"
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

        packed_payload = (
            TrajectoryCodec.pack_episode(
                mcts_tree, best_Z, last_delegate.prefix_registry
            )
            if last_delegate
            else {"prefixes": {}, "transitions": []}
        )

        num_transitions = len(packed_payload["transitions"])
        ep_noise = max(
            config.min_noise,
            config.base_noise * (1.0 - episode / max(1, config.decay_episodes)),
        )
        logger.info(
            f"{LOG_PREFIX} [Worker {rank}] Ep {episode:03d} | Ep Noise: {ep_noise:.4f} | Best Cost: {best_cost:8.4f} ms | "
            f"Extractions: {len(extraction_costs)} | Sending {num_transitions} deduplicated transitions..."
        )

        traj_queue.put(
            {
                "payload": packed_payload,
                "cost": best_cost,
                "costs": extraction_costs,
            }
        )
        episode += 1


def main():
    parser = argparse.ArgumentParser(description="AlphaZero TensorGraph Worker Client")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server address")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument(
        "-bt", "--use-bluetooth", action="store_true", help="Use Bluetooth RFCOMM"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--simulations", type=int, default=10, help="MCTS simulations per episode"
    )
    parser.add_argument(
        "--graph-source", type=str, default="model", choices=["model", "random"]
    )
    parser.add_argument("--model", type=str, default="gemma-3-270m", help="Model name")
    parser.add_argument("--model-path", type=str, default="models/google/gemma-3-270m")
    parser.add_argument("--random-min-nodes", type=int, default=10)
    parser.add_argument("--random-max-nodes", type=int, default=30)
    parser.add_argument("--random-dim", type=int, default=128)
    parser.add_argument("--random-seq-len", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--resample-graph-every", type=int, default=0)
    parser.add_argument("--compile-decode-buckets", action="store_true")
    parser.add_argument("--log-lost-calls", action="store_true", dest="log_cost_calls")
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--base-noise", type=float, default=0.25)
    parser.add_argument("--min-noise", type=float, default=0.01)
    parser.add_argument("--decay-episodes", type=int, default=500)
    parser.add_argument("--depth-gamma", type=float, default=0.7)
    parser.add_argument(
        "--level-sims",
        nargs="+",
        type=int,
        default=[1, 1, 1, 1],
        help="Simulations per level: [num_extract, num_dispatch, num_bufferize, num_malloc]",
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
            print(f"[Client] Note: Proceeding assuming local model files exist ({e}).")

    conn_type = "Bluetooth" if config.use_bluetooth else "TCP/IP"
    print("=========================================================")
    print(f" Starting {config.workers} Client Worker Process(es)")
    print(f" Target Server: {config.host}:{config.port} ({conn_type})")
    print(f" Graph Source: {config.graph_source.upper()}")
    print("=========================================================")

    req_queue = mp.Queue()
    resp_queues = [mp.Queue() for _ in range(config.workers)]
    traj_queue = mp.Queue()
    weights_event = mp.Event()

    inf_process = mp.Process(
        target=inference_worker,
        args=(config, req_queue, resp_queues, weights_event, "runs"),
    )
    inf_process.start()

    processes = []
    for rank in range(config.workers):
        p = mp.Process(
            target=client_worker,
            args=(rank, config, req_queue, resp_queues[rank], traj_queue),
        )
        p.start()
        processes.append(p)

    client_sock = create_client_socket(config)
    sock_lock = threading.Lock()

    # Track weight versions and completed episodes
    current_version = -1
    episodes_completed = 0
    episodes_lock = threading.Lock()

    def weight_sync_thread():
        nonlocal current_version, episodes_completed
        os.makedirs("runs", exist_ok=True)
        weights_path = os.path.join("runs", "client_weights.pt")
        while True:
            try:
                # 1. Ask server only for current version (lightweight)
                with sock_lock:
                    send_msg(client_sock, {"type": "req_version"})
                    resp = recv_msg(client_sock)

                if resp and resp.get("type") == "version":
                    server_version = resp.get("version", 0)

                    # Only fetch weights on initial boot OR if server has newer weights
                    # AND at least 1 episode was completed using current weights
                    with episodes_lock:
                        ready_to_update = (current_version == -1) or (
                            server_version > current_version and episodes_completed > 0
                        )

                    if ready_to_update:
                        with sock_lock:
                            send_msg(client_sock, {"type": "req_weights"})
                            weights_resp = recv_msg(client_sock)

                        if (
                            weights_resp
                            and weights_resp.get("type") == "weights"
                            and weights_resp.get("data")
                        ):
                            torch.save(weights_resp["data"], weights_path)
                            current_version = weights_resp.get(
                                "version", server_version
                            )
                            with episodes_lock:
                                episodes_completed = 0
                            weights_event.set()
                            print(
                                f"[Client] Synced new weights (version {current_version})."
                            )

                time.sleep(10)
            except Exception as e:
                print(f"[Client] Weight sync error: {e}")
                time.sleep(5)

    threading.Thread(target=weight_sync_thread, daemon=True).start()

    buffer_transitions = []
    buffer_prefixes = {}
    extraction_costs = []
    best_cost_in_window = float("inf")
    last_send_time = time.time()

    while True:
        try:
            msg = traj_queue.get(timeout=1.0)
            payload = msg["payload"]
            cost = msg["cost"]
            buffer_transitions.extend(payload["transitions"])
            buffer_prefixes.update(payload["prefixes"])
            extraction_costs.extend(msg.get("costs", []))
            best_cost_in_window = min(best_cost_in_window, cost)

            # Record that an episode completed with the current weights
            with episodes_lock:
                episodes_completed += 1

        except queue.Empty:
            pass

        if time.time() - last_send_time > 2.0 and buffer_transitions:
            packed_payload = {
                "prefixes": buffer_prefixes,
                "transitions": buffer_transitions,
            }
            try:
                with sock_lock:
                    send_msg(
                        client_sock,
                        {
                            "type": "trajectory",
                            "cost": best_cost_in_window,
                            "costs": extraction_costs,
                            "payload": packed_payload,
                        },
                    )
                buffer_transitions.clear()
                buffer_prefixes.clear()
                extraction_costs.clear()
                best_cost_in_window = float("inf")
            except Exception as e:
                print(f"[Client] Error sending trajectory: {e}")
            last_send_time = time.time()

    for p in processes:
        p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
