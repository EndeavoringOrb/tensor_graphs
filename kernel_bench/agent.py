# File: kernel_bench/agent.py
import json
import time
import requests

LLM_API_URL = "http://localhost:55554/v1/chat/completions"
BENCH_API_URL = "http://127.0.0.1:8080"
MODEL = "my-model-id"

# Set the optimization target!
TARGET_MODEL = "gemma-3-270m"  # "flux-klein-4b" or "gemma-3-270m"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_hw_info",
            "description": "Get hardware specifications of the machine.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_performance_history",
            "description": "Get the historical performance of all submitted kernels.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_analysis",
            "description": "Get the current total estimated execution time, top heaviest chains, and extracted UIDs.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_benchmarks",
            "description": "Read all recorded benchmarks (shapes/strides) to find targets to optimize.",
            "parameters": {
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "description": "Regex pattern to filter OpName (e.g., 'Dot.*F32' or '^Softmax'). Case-insensitive.",
                    },
                    "shape": {
                        "type": "string",
                        "description": "Regex pattern to filter OutputShape or InputShape (e.g., '128, 768' or '\\[.*, 4096\\]').",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_and_test_kernel",
            "description": "Submit a C++ kernel for compilation, testing, and benchmarking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_code": {
                        "type": "string",
                        "description": "The full C++ source code of the kernel.",
                    },
                    "opname": {
                        "type": "string",
                        "description": "The operation name (e.g. Dot_F32_3D_Optimized).",
                    },
                    "backend": {
                        "type": "string",
                        "enum": ["cpu", "cuda"],
                        "description": "Target backend.",
                    },
                },
                "required": ["source_code", "opname", "backend"],
            },
        },
    },
]


def call_bench_api(path, method="GET", json_data=None):
    url = f"{BENCH_API_URL}{path}"
    if method == "GET":
        res = requests.get(url, params=json_data)
    else:
        res = requests.post(url, json=json_data)
    return res.json()


def handle_tool_call(tool_call):
    name = tool_call["function"]["name"]
    args = json.loads(tool_call["function"]["arguments"])
    print(f"\n[Agent executing tool: {name}]")

    if name == "get_hw_info":
        return call_bench_api("/api/hwinfo")
    elif name == "get_performance_history":
        return call_bench_api("/api/history")
    elif name == "get_analysis":
        return call_bench_api("/api/analyze", json_data={"target_model": TARGET_MODEL})
    elif name == "read_benchmarks":
        return call_bench_api("/api/read_benchmarks", json_data=args)
    elif name == "submit_and_test_kernel":
        args["target_model"] = TARGET_MODEL
        res = call_bench_api("/api/kernels/test", method="POST", json_data=args)
        job_id = res.get("job_id")
        if not job_id:
            return {"error": "Submission failed", "details": res}

        print(f"  -> Job {job_id} queued for {TARGET_MODEL}. Polling for completion...")
        while True:
            time.sleep(5)
            status = call_bench_api(f"/api/jobs/{job_id}")
            if status.get("status") in ["completed", "failed"]:
                return status
            print("  ...still running...")

    return {"error": f"Unknown tool {name}"}


def run_agentic_loop():
    messages = [
        {
            "role": "system",
            "content": (
                f"You are an elite C++ and CUDA/NEON performance optimization AI agent. "
                f"Your target model for optimization is {TARGET_MODEL}. "
                "Your goal is to optimize tensor operations to reduce 'Total Estimated Execution Time'. "
                "You work in a loop: analyze current performance, generate an optimized kernel, submit it, "
                "and learn from the test results and benchmarks. "
                "The test pipeline steps are: Compile -> Test(No Rec) -> Matched in Graph -> Test(Records) -> Benchmark -> Extracted in final graph. "
                "Iterate infinitely. Use the provided tools."
            ),
        },
        {
            "role": "user",
            "content": "Begin optimizing the kernels. First get hardware info and performance analysis.",
        },
    ]

    print("Starting Autonomous Optimization Loop...")

    while True:
        try:
            payload = {
                "model": MODEL,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
            }

            response = requests.post(LLM_API_URL, json=payload)
            response.raise_for_status()
            response_data = response.json()
            message = response_data["choices"][0]["message"]
            messages.append(message)

            if message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    result = handle_tool_call(tool_call)
                    content = json.dumps(result, indent=2)
                    if len(content) > 10000:
                        content = (
                            "Content exceeded maximum length. Please narrow parameters."
                        )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "content": content,
                        }
                    )
            else:
                print(f"\n[Agent says]:\n{message.get('content')}\n")
                messages.append(
                    {
                        "role": "user",
                        "content": "Please continue optimizing. Generate and submit your next kernel.",
                    }
                )

        except Exception as e:
            print(f"Error communicating with LLM server: {e}")
            time.sleep(10)


if __name__ == "__main__":
    run_agentic_loop()
