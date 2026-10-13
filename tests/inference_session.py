"""How the inference benchmark builds and opens the compiled model.

This module is AGENT-EDITABLE (docs/agents/improve_inference_speed.md): bucket
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

import tensor_graphs

from tests.inference_cases import MODEL_NAME, MODEL_PATH, SEQUENCE_LENGTH

# Seconds reserved after the search for extraction, plan writing and arenas.
SEARCH_MARGIN_SECONDS = 60.0


def open_session(cache_file, compile_deadline):
    if compile_deadline is not None:
        # The planner's search stops at TG_PROGRAM_DEADLINE (which requires
        # volume tracking); leave room to finish compiling afterwards.
        os.environ["TG_PROGRAM_DEADLINE"] = repr(compile_deadline - SEARCH_MARGIN_SECONDS)
        os.environ["TG_SEARCH_VOLUME"] = "1"
    return tensor_graphs.LLMSession(
        MODEL_NAME,
        MODEL_PATH,
        tensor_graphs.HeuristicSearchDelegate(),
        min_compile_time=-1.0,  # search until exhausted or the deadline
        compile_decode_buckets=False,  # only the full bucket: every step runs the whole graph
        max_sequence_length=SEQUENCE_LENGTH,
        cache_file=cache_file,
        disable_node_caching=False,
        disable_compilation_caching=False,
    )
