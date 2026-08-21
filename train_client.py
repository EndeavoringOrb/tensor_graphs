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

from train_models import AlphaZeroTransformer
from train_shared import (
    ActorDelegate,
    TrainConfig,
    TrajectoryCodec,
    create_client_socket,
    get_default_model_path,
    get_graph_provider,
    recv_msg,
    send_msg,
)


def inference_worker(
    config: TrainConfig,
    req_queue,
    resp_queues,
    weights_event,
    run_dir,
    device_str: str | None = None,
    shared_action_feats=None,
    shared_logits=None,
    shared_v=None,
):
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference Server] Started on {device}")

    agent = AlphaZeroTransformer(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        max_feat_dim=config.max_feat_dim,
    ).to(device)
    agent.eval()

    prefix_cache_ctx = {}
    prefix_cache_v = {}
    current_version = 0
    weights_path = Path(run_dir) / "client_weights.pt"

    while True:
        if weights_event.is_set():
            if weights_path.exists():
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
                        f"[Inference Server] Weights updated to version {current_version}."
                    )
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
                _, ver, pkey, pdata = req
                cache_key = (ver, pkey)
                if cache_key not in prefix_cache_ctx:
                    gf = torch.tensor(
                        pdata.global_feature, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    nf = torch.tensor(
                        pdata.node_features, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    e = torch.tensor(pdata.edge_index, dtype=torch.int64, device=device)
                    pid = torch.tensor(
                        [pdata.phase_id], dtype=torch.int64, device=device
                    )

                    with (
                        torch.inference_mode(),
                        torch.autocast(device_type=device.type, dtype=torch.bfloat16),
                    ):
                        v, ctx = agent.encode_prefix(gf, nf, e, pid)

                    prefix_cache_ctx[cache_key] = ctx
                    prefix_cache_v[cache_key] = v.item() if v is not None else 0.0
            elif req[0] == "evaluate" or req[0] == "evaluate_shm":
                eval_reqs.append(req)

        if not eval_reqs:
            continue

        valid_reqs = []
        for req in eval_reqs:
            # For "evaluate" index 3 is action_features. For "evaluate_shm" index 3 is A_len. 
            ver, pkey = req[1], req[2]
            cache_key = (ver, pkey)
            if cache_key not in prefix_cache_ctx:
                wid = req[5]
                resp_queues[wid].put(("error", "missing_prefix"))
                continue
            valid_reqs.append(req)

        groups = defaultdict(list)
        for req in valid_reqs:
            groups[(req[1], req[2])].append(req)

        for (ver, pkey), group_reqs in groups.items():
            cache_key = (ver, pkey)
            B = len(group_reqs)
            
            max_A = 0
            for req in group_reqs:
                if req[0] == "evaluate":
                    max_A = max(max_A, req[3].shape[0])
                else:  # "evaluate_shm"
                    max_A = max(max_A, req[3])

            padded_actions = torch.zeros(
                (B, max_A, 8), dtype=torch.float32, device=device
            )
            padded_pid = torch.zeros((B, max_A), dtype=torch.int64, device=device)

            ctx = prefix_cache_ctx[cache_key]
            batched_ctx = ctx.expand(B, -1, -1)

            for i, req in enumerate(group_reqs):
                phase_id = req[4]
                if req[0] == "evaluate":
                    a_feats = req[3]
                    A_len = a_feats.shape[0]
                    dim_feat = min(7, a_feats.shape[1])
                    padded_actions[i, :A_len, 1 : 1 + dim_feat] = torch.tensor(
                        a_feats[:, :dim_feat], dtype=torch.float32, device=device
                    )
                else:  # "evaluate_shm"
                    A_len = req[3]
                    wid = req[5]
                    padded_actions[i, :A_len, 1:8] = shared_action_feats[wid, :A_len, :].to(device)

                padded_actions[i, :A_len, 0] = torch.arange(
                    A_len, dtype=torch.float32, device=device
                )
                padded_pid[i, :A_len] = phase_id

            with (
                torch.inference_mode(),
                torch.autocast(device_type=device.type, dtype=torch.bfloat16),
            ):
                logits = agent.evaluate_actions(
                    padded_actions, padded_pid, context=batched_ctx
                )

            for i, req in enumerate(group_reqs):
                wid = req[5]
                A_len = req[3].shape[0] if req[0] == "evaluate" else req[3]
                resp_logits = logits[i, :A_len].cpu().float()
                v = prefix_cache_v[cache_key]

                if req[0] == "evaluate":
                    resp_queues[wid].put(("ok", resp_logits.numpy(), v))
                else:
                    shared_logits[wid, :A_len].copy_(resp_logits)
                    shared_v[wid] = v
                    resp_queues[wid].put(("ok",))


@torch.inference_mode()
def client_worker(
    rank: int,
    config: TrainConfig,
    req_queue,
    resp_queue,
    traj_queue,
    shared_version,
    device_str: str | None = None,
    shared_action_feats=None,
    shared_logits=None,
    shared_v=None,
):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    tensor_graphs.set_num_threads(config.cpp_threads)

    if device_str and "cuda:" in device_str:
        try:
            dev_idx = int(device_str.split(":")[-1])
            torch.cuda.set_device(dev_idx)
        except Exception:
            pass

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
            version=episode_version,
            shared_action_feats=shared_action_feats,
            shared_logits=shared_logits,
            shared_v=shared_v,
        )

        for _ in range(config.num_simulations):
            # 2. Clear the active stack at the start of each simulation
            delegate.active_stack.clear()
            
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

        # 3. Update the pack_episode call to use the preserved delegate
        packed_payload = TrajectoryCodec.pack_episode(
            mcts_tree, best_Z, delegate.prefix_registry
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
        "--device",
        type=str,
        default=None,
        help="PyTorch compute device for client inference worker (e.g. cuda:0, cuda:1, cpu)",
    )
    parser.add_argument(
        "-bt", "--use-bluetooth", action="store_true", help="Use Bluetooth RFCOMM"
    )
    parser.add_argument(
        "--bt-address", type=str, default=None, help="Bluetooth host MAC"
    )
    parser.add_argument(
        "--bt-port", type=int, default=None, help="Bluetooth RFCOMM channel"
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
        "--simulations", type=int, default=None, help="MCTS simulations per episode"
    )
    parser.add_argument(
        "--level-sims",
        nargs="+",
        type=int,
        default=None,
        help="Simulations per level: [num_extract, num_dispatch, num_bufferize, num_malloc]",
    )
    parser.add_argument(
        "--graph-source", type=str, default=None, choices=["model", "random"]
    )
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--model-path", type=str, default=None, help="Model path")
    parser.add_argument("--random-min-nodes", type=int, default=None)
    parser.add_argument("--random-max-nodes", type=int, default=None)
    parser.add_argument("--random-dim", type=int, default=None)
    parser.add_argument("--random-seq-len", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--resample-graph-every", type=int, default=None)
    parser.add_argument("--compile-decode-buckets", action="store_true", default=None)
    parser.add_argument("--log-cost-calls", action="store_true", default=None)
    parser.add_argument("--c-puct", type=float, default=None)
    parser.add_argument("--base-noise", type=float, default=None)
    parser.add_argument("--min-noise", type=float, default=None)
    parser.add_argument("--decay-episodes", type=int, default=None)
    parser.add_argument("--depth-gamma", type=float, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)

    args = parser.parse_args()

    net_config = TrainConfig()
    net_config.host = args.host
    net_config.port = args.port
    net_config.use_bluetooth = args.use_bluetooth
    if args.bt_address is not None:
        net_config.bt_host_address = args.bt_address
    if args.bt_port is not None:
        net_config.bt_port = args.bt_port

    if (
        net_config.host
        and ":" in net_config.host
        and len(net_config.host.split(":")) == 6
    ):
        net_config.use_bluetooth = True

    conn_type = "Bluetooth" if net_config.use_bluetooth else "TCP/IP"
    print(
        f"[Client] Connecting to {net_config.host}:{net_config.port} ({conn_type})..."
    )
    client_sock = create_client_socket(net_config)
    sock_lock = threading.Lock()

    print("[Client] Querying base training configuration from server...")
    with sock_lock:
        send_msg(client_sock, {"type": "req_config"})
        resp = recv_msg(client_sock)

    if resp and resp.get("type") == "config" and resp.get("config"):
        config = TrainConfig.from_dict(resp["config"])
        print("[Client] Synced base TrainConfig from server.")
    else:
        print("[Client] Server did not provide config; falling back to local defaults.")
        config = TrainConfig()

    config.host = net_config.host
    config.port = net_config.port
    config.use_bluetooth = net_config.use_bluetooth
    config.bt_host_address = net_config.bt_host_address
    config.bt_port = net_config.bt_port
    config.workers = (
        1 if (args.log_cost_calls or config.log_cost_calls) else args.workers
    )
    config.cpp_threads = args.cpp_threads

    if args.simulations is not None:
        config.num_simulations = args.simulations
    if args.level_sims is not None:
        config.level_simulations = args.level_sims
    if args.graph_source is not None:
        config.graph_source = args.graph_source
    if args.model is not None:
        config.model_name = args.model
    if args.model_path is not None:
        config.model_path = args.model_path
    if args.random_min_nodes is not None:
        config.random_min_nodes = args.random_min_nodes
    if args.random_max_nodes is not None:
        config.random_max_nodes = args.random_max_nodes
    if args.random_dim is not None:
        config.random_hidden_dim = args.random_dim
    if args.random_seq_len is not None:
        config.random_seq_len = args.random_seq_len
    if args.random_seed is not None:
        config.random_seed = args.random_seed
    if args.resample_graph_every is not None:
        config.resample_graph_every = args.resample_graph_every
    if args.compile_decode_buckets is not None:
        config.compile_decode_buckets = args.compile_decode_buckets
    if args.log_cost_calls is not None:
        config.log_cost_calls = args.log_cost_calls
    if args.c_puct is not None:
        config.c_puct = args.c_puct
    if args.base_noise is not None:
        config.base_noise = args.base_noise
    if args.min_noise is not None:
        config.min_noise = args.min_noise
    if args.decay_episodes is not None:
        config.decay_episodes = args.decay_episodes
    if args.depth_gamma is not None:
        config.depth_gamma = args.depth_gamma
    if args.d_model is not None:
        config.d_model = args.d_model
    if args.nhead is not None:
        config.nhead = args.nhead
    if args.num_layers is not None:
        config.num_layers = args.num_layers

    target_device = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if args.model is not None:
        config.model_name = args.model
        if args.model_path is None:
            config.model_path = get_default_model_path(args.model)
    if args.model_path is not None:
        config.model_path = args.model_path

    # TODO: MAX_ACTIONS is a hardcoded limit for shared memory size to handle worst-case expansion paths. Might need to be dynamic for exceptionally large extraction graphs.
    MAX_ACTIONS = 16384 
    shared_action_feats = torch.zeros((config.workers, MAX_ACTIONS, 7), dtype=torch.float32).share_memory_()
    shared_logits = torch.zeros((config.workers, MAX_ACTIONS), dtype=torch.float32).share_memory_()
    shared_v = torch.zeros((config.workers,), dtype=torch.float32).share_memory_()

    print("=========================================================")
    print(f" Starting {config.workers} Client Worker Process(es)")
    print(f" C++ Threads / Worker: {config.cpp_threads}")
    print(f" Inference Device: {target_device}")
    print(f" Target Server: {config.host}:{config.port} ({conn_type})")
    print(f" Graph Source: {config.graph_source.upper()}")
    print(
        f" Transformer Model: d_model={config.d_model}, nhead={config.nhead}, layers={config.num_layers}"
    )
    print("=========================================================")

    req_queue = mp.Queue()
    resp_queues = [mp.Queue() for _ in range(config.workers)]
    traj_queue = mp.Queue()
    weights_event = mp.Event()
    shared_version = mp.Value("i", 0)

    runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    target_device = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    inf_process = mp.Process(
        target=inference_worker,
        args=(config, req_queue, resp_queues, weights_event, "runs", target_device),
    )
    inf_process.start()

    processes = []
    for rank in range(config.workers):
        p = mp.Process(
            target=client_worker,
            args=(
                rank,
                config,
                req_queue,
                resp_queues[rank],
                traj_queue,
                shared_version,
                target_device,
                shared_action_feats,
                shared_logits,
                shared_v,
            ),
        )
        p.start()
        processes.append(p)

    current_version = -1
    episodes_completed = 0
    episodes_lock = threading.Lock()

    def weight_sync_thread():
        nonlocal current_version, episodes_completed
        weights_path = runs_dir / "client_weights.pt"
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

    for p in processes:
        p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
