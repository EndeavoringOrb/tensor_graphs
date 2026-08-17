import argparse
import logging
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.multiprocessing as mp

DEFAULT_WORKERS = max(1, (psutil.cpu_count(logical=False) or 4) - 1)
torch.set_float32_matmul_precision("high")

import tensor_graphs

from train_shared import (
    MAX_ACTIONS,
    MAX_FEATS,
    ActorDelegate,
    AlphaZeroTransformer,
    RadixTreeKVCache,
    ShmRequestSlot,
    ShmResponseSlot,
    ShmSPSCQueue,
    TrainConfig,
    TrajectoryCodec,
    create_client_socket,
    get_graph_provider,
    recv_msg,
    send_msg,
)


# ==============================================================================
# INFERENCE WORKER USING FLASHINFER RADIXATTENTION BATCHING
# ==============================================================================
def inference_worker(
    config: TrainConfig,
    num_workers: int,
    shared_prefix_dict,
    weights_event,
    run_dir,
):
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference Server] Started on {device}")

    head_dim = config.d_model // config.nhead
    agent = AlphaZeroTransformer(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        max_feat_dim=config.max_feat_dim,
    ).to(device)
    agent.eval()

    radix_cache = RadixTreeKVCache(
        num_layers=config.num_layers,
        num_heads=config.nhead,
        head_dim=head_dim,
        device=device,
    )

    current_version = 0
    weights_path = os.path.join(run_dir, "client_weights.pt")

    # Connect to per-worker lock-free shared memory SPSC queues
    req_queues = []
    resp_queues = []
    for wid in range(num_workers):
        q_req = ShmSPSCQueue(
            f"tg_req_w_{wid}", ShmRequestSlot, capacity=32, create=False
        )
        q_resp = ShmSPSCQueue(
            f"tg_resp_w_{wid}", ShmResponseSlot, capacity=32, create=False
        )
        req_queues.append(q_req)
        resp_queues.append(q_resp)

    print(
        f"[Inference Server] Connected to {num_workers} SPSC Shared-Memory Ring Buffers."
    )

    while True:
        # 1. Update weights without purging the Radix KV cache
        if weights_event.is_set():
            if os.path.exists(weights_path):
                try:
                    loaded = torch.load(
                        weights_path, map_location="cpu", weights_only=True
                    )
                    if isinstance(loaded, dict) and "state_dict" in loaded:
                        new_version = loaded.get("version", current_version + 1)
                        weights_dict = loaded["state_dict"]
                    else:
                        new_version = current_version + 1
                        weights_dict = loaded

                    agent.load_state_dict(weights_dict, strict=False)
                    current_version = new_version
                    print(
                        f"[Inference Server] Weights updated to version {current_version}. Persisting Radix KV Tree."
                    )
                except Exception as e:
                    print(f"[Inference Server] Error loading weights: {e}")
            weights_event.clear()

        # 2. Lock-free collection across all worker SPSC queues
        batched_evals = []
        for wid in range(num_workers):
            slot_idx, slot = req_queues[wid].read_slot()
            if slot is not None:
                msg_type = slot.msg_type
                if msg_type == 1:
                    pkey = int(slot.prefix_key)
                    ver = int(slot.version)
                    pid = int(slot.phase_id)
                    num_act = int(slot.num_actions)

                    action_feats_np = (
                        np.frombuffer(slot.action_features, dtype=np.float32)
                        .reshape(MAX_ACTIONS, MAX_FEATS)[:num_act]
                        .copy()
                    )
                    batched_evals.append(
                        (wid, ver, pkey, pid, num_act, action_feats_np)
                    )
                req_queues[wid].commit_read()

        if not batched_evals:
            time.sleep(0.0001)
            continue

        # 3. Ensure Radix Cache has prefix KV encoded
        valid_evals = []
        for wid, ver, pkey, pid, num_act, action_feats_np in batched_evals:
            if not radix_cache.contains(pkey):
                if pkey in shared_prefix_dict:
                    pdata = shared_prefix_dict[pkey]
                    f_t = torch.tensor(
                        pdata.features, dtype=torch.float32, device=device
                    )
                    tt_t = torch.tensor(
                        pdata.token_types, dtype=torch.int64, device=device
                    )
                    p_t = torch.tensor(
                        pdata.phase_ids, dtype=torch.int64, device=device
                    )

                    with (
                        torch.inference_mode(),
                        torch.autocast(device_type=device.type, dtype=torch.bfloat16),
                    ):
                        v_pred, k_layers, v_layers = agent.encode_prefix(f_t, tt_t, p_t)

                    radix_cache.insert(
                        prefix_key=pkey,
                        num_tokens=pdata.features.shape[0],
                        layer_k=k_layers,
                        layer_v=v_layers,
                        value=float(v_pred.item()),
                    )
                else:
                    while True:
                        _, r_slot = resp_queues[wid].write_slot()
                        if r_slot is not None:
                            r_slot.ready = 2
                            resp_queues[wid].commit_write()
                            break
                    continue

            valid_evals.append((wid, ver, pkey, pid, num_act, action_feats_np))

        if not valid_evals:
            continue

        # 4. Batched FlashInfer Execution across heterogeneous requests
        B = len(valid_evals)
        nodes = [radix_cache.get(req[2]) for req in valid_evals]

        q_lens = [req[4] for req in valid_evals]
        total_A = sum(q_lens)

        q_indptr = torch.zeros(B + 1, dtype=torch.int32, device=device)
        q_indptr[1:] = torch.cumsum(
            torch.tensor(q_lens, dtype=torch.int32, device=device), dim=0
        )

        paged_kv_indices_list = [n.page_indices for n in nodes]
        paged_kv_indices = torch.cat(paged_kv_indices_list)

        kv_lens = [n.num_tokens for n in nodes]
        paged_kv_indptr = torch.zeros(B + 1, dtype=torch.int32, device=device)
        paged_kv_indptr[1:] = torch.cumsum(
            torch.tensor(kv_lens, dtype=torch.int32, device=device), dim=0
        )
        paged_kv_last_page_len = torch.ones(B, dtype=torch.int32, device=device)

        ragged_actions = torch.zeros(
            (total_A, MAX_FEATS), dtype=torch.float32, device=device
        )
        ragged_pids = torch.zeros(total_A, dtype=torch.int64, device=device)

        offset = 0
        for i, (wid, ver, pkey, pid, num_act, a_feats) in enumerate(valid_evals):
            dim_f = min(MAX_FEATS, a_feats.shape[1])
            ragged_actions[offset : offset + num_act, :dim_f] = torch.tensor(
                a_feats[:, :dim_f], dtype=torch.float32, device=device
            )
            ragged_pids[offset : offset + num_act] = pid
            offset += num_act

        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type, dtype=torch.bfloat16),
        ):
            all_logits = agent.evaluate_actions_paged(
                ragged_actions,
                ragged_pids,
                radix_cache.paged_kv_data,
                paged_kv_indices,
                paged_kv_indptr,
                paged_kv_last_page_len,
                q_indptr,
            )

        # 5. Write response directly to worker's SPSC shared memory slot
        offset = 0
        for i, (wid, ver, pkey, pid, num_act, a_feats) in enumerate(valid_evals):
            node = nodes[i]
            resp_logits = all_logits[offset : offset + num_act].cpu().float().numpy()
            v_val = node.value
            offset += num_act

            while True:
                _, r_slot = resp_queues[wid].write_slot()
                if r_slot is not None:
                    r_slot.ready = 1
                    r_slot.num_actions = num_act
                    r_slot.value = v_val

                    logits_dest = np.frombuffer(r_slot.logits, dtype=np.float32)[
                        :num_act
                    ]
                    logits_dest[:] = resp_logits

                    resp_queues[wid].commit_write()
                    break


# ==============================================================================
# CLIENT SEARCH WORKER
# ==============================================================================
@torch.inference_mode()
def client_worker(
    rank: int,
    config: TrainConfig,
    shared_prefix_dict,
    traj_queue,
    shared_version,
):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    tensor_graphs.set_num_threads(config.cpp_threads)

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

    shm_req = ShmSPSCQueue(
        f"tg_req_w_{rank}", ShmRequestSlot, capacity=32, create=False
    )
    shm_resp = ShmSPSCQueue(
        f"tg_resp_w_{rank}", ShmResponseSlot, capacity=32, create=False
    )

    graph_provider = get_graph_provider(config, worker_rank=rank)
    logger.info(
        f"{LOG_PREFIX} [Worker {rank}] Initializing graph provider (source: {config.graph_source})..."
    )

    episode = 0

    while True:
        with shared_version.get_lock():
            episode_version = shared_version.value

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
                shm_req_queue=shm_req,
                shm_resp_queue=shm_resp,
                prefix_dict=shared_prefix_dict,
                worker_id=rank,
                mcts_tree=mcts_tree,
                c_puct=config.c_puct,
                episode=episode,
                decay_episodes=config.decay_episodes,
                base_noise=config.base_noise,
                min_noise=config.min_noise,
                depth_gamma=config.depth_gamma,
                version=episode_version,
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
            f"{LOG_PREFIX} [Worker {rank}] Ep {episode:03d} (v{episode_version}) | Ep Noise: {ep_noise:.4f} | Best Cost: {best_cost:8.4f} ms | "
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
        "--cpp-threads",
        type=int,
        default=1,
        help="Number of C++ threads per worker process (default: 1)",
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
    config.cpp_threads = args.cpp_threads
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
    print(f" C++ Threads / Worker: {config.cpp_threads}")
    print(f" Target Server: {config.host}:{config.port} ({conn_type})")
    print(f" Graph Source: {config.graph_source.upper()}")
    print("=========================================================")

    # Pre-create SPSC Shared-Memory Queues
    created_shm_queues = []
    for wid in range(config.workers):
        q_req = ShmSPSCQueue(
            f"tg_req_w_{wid}", ShmRequestSlot, capacity=32, create=True
        )
        q_resp = ShmSPSCQueue(
            f"tg_resp_w_{wid}", ShmResponseSlot, capacity=32, create=True
        )
        created_shm_queues.append((q_req, q_resp))

    manager = mp.Manager()
    shared_prefix_dict = manager.dict()

    traj_queue = mp.Queue()
    weights_event = mp.Event()
    shared_version = mp.Value("i", 0)

    inf_process = mp.Process(
        target=inference_worker,
        args=(
            config,
            config.workers,
            shared_prefix_dict,
            weights_event,
            "runs",
        ),
    )
    inf_process.start()

    processes = []
    for rank in range(config.workers):
        p = mp.Process(
            target=client_worker,
            args=(
                rank,
                config,
                shared_prefix_dict,
                traj_queue,
                shared_version,
            ),
        )
        p.start()
        processes.append(p)

    client_sock = create_client_socket(config)
    sock_lock = threading.Lock()

    current_version = -1
    episodes_completed = 0
    episodes_lock = threading.Lock()

    def weight_sync_thread():
        nonlocal current_version, episodes_completed
        os.makedirs("runs", exist_ok=True)
        weights_path = os.path.join("runs", "client_weights.pt")
        while True:
            try:
                with sock_lock:
                    send_msg(client_sock, {"type": "req_version"})
                    resp = recv_msg(client_sock)

                if resp and resp.get("type") == "version":
                    server_version = resp.get("version", 0)

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
                            torch.save(
                                {
                                    "version": server_version,
                                    "state_dict": weights_resp["data"],
                                },
                                weights_path,
                            )
                            current_version = weights_resp.get(
                                "version", server_version
                            )
                            with shared_version.get_lock():
                                shared_version.value = current_version
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
    server_known_prefixes = set()

    try:
        while True:
            try:
                msg = traj_queue.get(timeout=1.0)
                payload = msg["payload"]
                cost = msg["cost"]
                buffer_transitions.extend(payload["transitions"])
                buffer_prefixes.update(payload["prefixes"])
                extraction_costs.extend(msg.get("costs", []))
                best_cost_in_window = min(best_cost_in_window, cost)

                with episodes_lock:
                    episodes_completed += 1

            except queue.Empty:
                pass

            if time.time() - last_send_time > 2.0 and buffer_transitions:
                new_prefixes = {
                    k: v
                    for k, v in buffer_prefixes.items()
                    if k not in server_known_prefixes
                }
                packed_payload = {
                    "prefixes": new_prefixes,
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
                    server_known_prefixes.update(new_prefixes.keys())
                    buffer_transitions.clear()
                    buffer_prefixes.clear()
                    extraction_costs.clear()
                    best_cost_in_window = float("inf")
                except Exception as e:
                    print(f"[Client] Error sending trajectory: {e}")
                    server_known_prefixes.clear()
                last_send_time = time.time()
    finally:
        for p in processes:
            p.terminate()
        inf_process.terminate()
        for q_req, q_resp in created_shm_queues:
            q_req.unlink()
            q_resp.unlink()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
