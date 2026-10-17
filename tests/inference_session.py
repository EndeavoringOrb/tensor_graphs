"""How the inference benchmark builds and opens the compiled model.

This module is AGENT-EDITABLE (docs/agents/targets/inference_speed/target.md): bucket
choice, search budget, threads and any other session option may change here
or in the native code behind it. The measurement and correctness checks in
tests/inference_benchmark.py are fixed.

Contract:
- open_session(cache_file, compile_deadline) returns an object with
    reset()               forget previous tokens (next step marks all dirty)
    step_logits(tokens)   process tokens (only positions that differ from the
                          previous call are dirty) and return the float32
                          logits row of the last position (numpy, vocab size)
- Compile phase: compile_deadline is the Unix time by which compiling,
  including writing the plan to `cache_file`, must be done; the supervisor
  kills the process at that time.
- Benchmark phase: compile_deadline is None and the session must load the plan
  written by the compile phase without planning again.
"""
import os

from tests.inference_cases import CASES, MODEL_NAME, MODEL_PATH, SEQUENCE_LENGTH

# Seconds reserved after the search for extraction, plan writing and arenas.
SEARCH_MARGIN_SECONDS = 75.0

def tokenBuckets():
    """Build exact prefill and decode buckets for the benchmark workloads."""
    buckets = set()
    for case in CASES:
        context = case["context"]
        count = case["tokens"]
        if context > 0:
            buckets.add((0, context, context - 1, 1.0))
        if case["kind"] == "pp":
            buckets.add((context, context + count, context + count - 1, 1.0))
        else:
            for position in range(context, context + count):
                buckets.add((position, position + 1, position, 1.0))
    return sorted(buckets)


def open_session(cache_file, compile_deadline):
    # The native module's C runtime copies the environment when it loads, so
    # these must be set before the first import of tensor_graphs.
    # Plan for the CPU only (the GPU path only adds copies for this model).
    os.environ["TG_DISABLE_OPENCL"] = "1"
    # The relaxation lower bound only serves tree search and gap measurement;
    # its time is better spent on the root heuristic's local search.
    os.environ["TG_SKIP_RELAXATION_BOUND"] = "1"
    if compile_deadline is not None:
        # The planner's search stops at TG_SEARCH_DEADLINE and keeps its
        # incumbent; leave room to finish compiling afterwards.
        os.environ["TG_SEARCH_DEADLINE"] = repr(compile_deadline - SEARCH_MARGIN_SECONDS)
    import tensor_graphs

    run_dir = os.path.dirname(os.path.abspath(cache_file))
    return tensor_graphs.LLMSession(
        MODEL_NAME,
        MODEL_PATH,
        tensor_graphs.HeuristicSearchDelegate(),
        min_compile_time=-1.0,  # search until exhausted or the deadline
        max_sequence_length=SEQUENCE_LENGTH,
        cache_file=cache_file,
        disable_node_caching=False,
        disable_compilation_caching=False,
        log_cost_calls=False,
        token_buckets=tokenBuckets(),
        # Weights stay resident in memory (written once at load) instead of
        # being copied from storage, or cached by the planner, in every run.
        resident_weights=True,
        # Folded tensors are compile artifacts: keep them in the run directory.
        repo_path=os.path.join(run_dir, "repo"),
    )
