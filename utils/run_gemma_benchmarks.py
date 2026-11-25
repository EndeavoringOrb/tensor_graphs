#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def runCommand(command_list, log_file, env=None):
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as out_f:
        process = subprocess.Popen(
            command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            out_f.write(line)
            out_f.flush()
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Command {' '.join(command_list)} failed with exit code {process.returncode}")


def runTensorGraphs(
    model_path="models/google/gemma-3-270m",
    log_file="logs/tensor_graphs.log",
    cache="dirty_region_caches/gemma-decode",
):
    target_cache = Path(f"{cache}.dirty-cache")
    if target_cache.exists():
        raise FileExistsError(f"Cache file already exists: {target_cache}")
    if Path(cache).exists():
        raise FileExistsError(f"Cache file already exists: {cache}")

    print("=" * 60)
    print("Running tensor_graphs benchmark (--config decode)...")
    print(f"Log output: {log_file}")
    print("=" * 60)

    code = f"""
import tensor_graphs
from main import encode_text, decode_tokens
from utils.decode import load_tokenizer
from pathlib import Path
import time

cache_path = Path({repr(cache)} + '.dirty-cache')
if cache_path.exists():
    raise FileExistsError('Cache file already exists: ' + str(cache_path))
if Path({repr(cache)}).exists():
    raise FileExistsError('Cache file already exists: ' + {repr(cache)})

model_path = Path({repr(model_path)})
tokenizer = load_tokenizer([str(model_path), 'gemma-3-270m'])
tokens = encode_text(tokenizer, 'Hi, my name', is_first=True)
print('Initial tokens:', tokens, 'count:', len(tokens), flush=True)

prompt_len = len(tokens)
session = tensor_graphs.LLMSession(
    'gemma-3-270m',
    str(model_path),
    tensor_graphs.HeuristicSearchDelegate(),
    min_compile_time=0.0,
    max_sequence_length=16,
    token_buckets=[(0, prompt_len, prompt_len - 1, 1.0)] +
        [(i, i + 1, i, 1.0) for i in range(prompt_len, 15)],
    compile_no_weights_bucket=False,
    cache_file={repr(cache)},
    disable_node_caching=False,
    disable_compilation_caching=False,
)

step_times = []
while len(tokens) < 16:
    t0 = time.perf_counter()
    nxt = session.generate_step(tokens)
    t1 = time.perf_counter()
    dt = t1 - t0
    step_times.append(dt)
    tok_str = decode_tokens(tokenizer, [nxt]) if nxt != -1 else ''
    print(f'step seq_len={{len(tokens)}} -> next_token={{nxt}} ({{tok_str!r}}) in {{dt*1000:.2f}} ms', flush=True)
    tokens.append(nxt)

print(f'Done 16 tokens! Total tg steps: {{len(step_times)}}', flush=True)
# Output formatting matches the regex patterns in utils/eval_gemma_kernel.py:
#   r"Prompt Processing:\s*([\d\.]+)\s*ms"
#   r"Text Generation:\s*([\d\.]+)\s*ms\s*/\s*tok"
#   r"Overall Generation:\s*([\d\.]+)\s*ms\s*/\s*tok"
# Do not modify these formats without updating both run_gemma_benchmarks.py and eval_gemma_kernel.py.
pp_ms = step_times[0] * 1000
print(f'Prompt Processing: {{pp_ms:.2f}} ms ({{5.0/step_times[0]:.2f}} t/s)', flush=True)
if len(step_times) > 1:
    decode_times = step_times[1:]
    tg_ms = sum(decode_times) / len(decode_times) * 1000
    print(f'Text Generation: {{tg_ms:.2f}} ms / tok ({{len(decode_times)/sum(decode_times):.2f}} t/s)', flush=True)
og_ms = sum(step_times) / len(step_times) * 1000
print(f'Overall Generation: {{og_ms:.2f}} ms / tok ({{len(step_times)/sum(step_times):.2f}} t/s)', flush=True)
"""
    python_bin = sys.executable
    runCommand([python_bin, "-c", code], log_file)


def runLlamaCpp(
    llama_bench_bin=str(Path.home() / "Tools/llama.cpp/build/bin/llama-bench"),
    model_path="/mnt/bigdrive/models/gguf-org/gemma-3-270m-it-bf16.gguf",
    log_file="logs/llama_cpp.log",
):
    print("=" * 60)
    print("Running llama.cpp (llama-bench) benchmark...")
    print(f"Log output: {log_file}")
    print("=" * 60)

    if not Path(llama_bench_bin).exists():
        raise FileNotFoundError(f"llama-bench binary not found at {llama_bench_bin}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"GGUF model not found at {model_path}")

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as out_f:
        # 1. GPU (CUDA, ngl -1)
        print("\n--- Running llama-bench on GPU (ngl -1) ---", flush=True)
        out_f.write("=== llama-bench GPU (ngl -1) ===\n")
        cmd_gpu = [llama_bench_bin, "-m", model_path, "-p", "5", "-n", "11", "-r", "5"]
        proc_gpu = subprocess.Popen(cmd_gpu, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc_gpu.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            out_f.write(line)
            out_f.flush()
        proc_gpu.wait()
        if proc_gpu.returncode != 0:
            raise RuntimeError(f"llama-bench GPU failed with code {proc_gpu.returncode}")

        # 2. CPU (ngl 0)
        print("\n--- Running llama-bench on CPU (ngl 0) ---", flush=True)
        out_f.write("\n=== llama-bench CPU (ngl 0) ===\n")
        cmd_cpu = [llama_bench_bin, "-m", model_path, "-ngl", "0", "-p", "5", "-n", "11", "-r", "5"]
        proc_cpu = subprocess.Popen(cmd_cpu, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc_cpu.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            out_f.write(line)
            out_f.flush()
        proc_cpu.wait()
        if proc_cpu.returncode != 0:
            raise RuntimeError(f"llama-bench CPU failed with code {proc_cpu.returncode}")


def runTransformers(model_path="models/google/gemma-3-270m", log_file="logs/transformers.log"):
    print("=" * 60)
    print("Running Hugging Face transformers benchmark...")
    print(f"Log output: {log_file}")
    print("=" * 60)

    code = f"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

model_path = {repr(model_path)}
tokenizer = AutoTokenizer.from_pretrained(model_path)

devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')

for device in devices:
    print(f'=== Testing Hugging Face transformers on device: {{device}} (dtype=bfloat16) ===', flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to(device)
    model.eval()

    inputs = tokenizer('Hi, my name', return_tensors='pt').to(device)
    input_ids = inputs.input_ids
    print('Initial input_ids shape:', input_ids.shape, flush=True)

    # Warmup
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=2, do_sample=False)

    latencies = []
    with torch.no_grad():
        past_key_values = None
        for step in range(11):
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model(
                input_ids if past_key_values is None else next_token.unsqueeze(0),
                past_key_values=past_key_values,
                use_cache=True,
            )
            if device == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            dt = t1 - t0
            latencies.append(dt)
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            tok_str = tokenizer.decode(next_token[0])
            print(f'[{{device}}] step seq_len={{input_ids.shape[1] + step}} -> next_token={{next_token.item()}} ({{tok_str!r}}) in {{dt*1000:.2f}} ms', flush=True)

    pp_time = latencies[0]
    tg_times = latencies[1:]
    print(f'[{{device}}] PP (seq_len 5): {{pp_time*1000:.2f}} ms ({{5.0/pp_time:.2f}} t/s)', flush=True)
    print(f'[{{device}}] TG (seq_len 6..16): {{sum(tg_times)/len(tg_times)*1000:.2f}} ms/tok ({{len(tg_times)/sum(tg_times):.2f}} t/s)', flush=True)
    print(f'[{{device}}] Overall (all 11 tokens): {{sum(latencies)/len(latencies)*1000:.2f}} ms/tok ({{len(latencies)/sum(latencies):.2f}} t/s)', flush=True)
"""
    python_bin = sys.executable
    runCommand([python_bin, "-c", code], log_file)


def main():
    parser = argparse.ArgumentParser(description="Head-to-head comparison benchmarks (tensor_graphs vs llama.cpp vs Hugging Face transformers)")
    parser.add_argument(
        "--target",
        choices=["all", "tensor_graphs", "llama_cpp", "transformers"],
        default="all",
        help="Target benchmark to run (default: all)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/google/gemma-3-270m",
        help="Path to Hugging Face model directory (default: models/google/gemma-3-270m)",
    )
    parser.add_argument(
        "--gguf-path",
        type=str,
        default="/mnt/bigdrive/models/gguf-org/gemma-3-270m-it-bf16.gguf",
        help="Path to GGUF model for llama.cpp (default: /mnt/bigdrive/models/gguf-org/gemma-3-270m-it-bf16.gguf)",
    )
    parser.add_argument(
        "--llama-bench",
        type=str,
        default=str(Path.home() / "Tools/llama.cpp/build/bin/llama-bench"),
        help="Path to llama-bench executable",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs",
        help="Directory to save log files (default: logs)",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="dirty_region_caches/gemma-decode",
        help="Path to cache file (default: dirty_region_caches/gemma-decode)",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.target in ("all", "tensor_graphs"):
        runTensorGraphs(
            model_path=args.model_path,
            log_file=str(logs_dir / "tensor_graphs.log"),
            cache=args.cache,
        )

    if args.target in ("all", "llama_cpp"):
        runLlamaCpp(
            llama_bench_bin=args.llama_bench,
            model_path=args.gguf_path,
            log_file=str(logs_dir / "llama_cpp.log"),
        )

    if args.target in ("all", "transformers"):
        runTransformers(
            model_path=args.model_path,
            log_file=str(logs_dir / "transformers.log"),
        )

    print("\nAll selected benchmarks completed successfully!")


if __name__ == "__main__":
    main()
