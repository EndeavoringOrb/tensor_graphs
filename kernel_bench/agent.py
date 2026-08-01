# File: kernel_bench/agent.py
import json
import threading
import time

import requests

BENCH_API_URL = "http://127.0.0.1:8080"

# Configure multiple agents and URLs here!
AGENT_CONFIGS = [
    {
        "url": "http://localhost:11434/v1/chat/completions",
        "model": "qwen3.6:35b",
        "target_model": "flux-klein-4b",
        "instances": 1,
    },
    {
        "url": "http://localhost:11435/v1/chat/completions",
        "model": "qwen3.6:35b",
        "target_model": "flux-klein-4b",
        "instances": 1,
    },
]

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
                    "source": {
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
                "required": ["source", "opname", "backend"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_kernel_files",
            "description": "Recursively list all existing C++ and CUDA kernel files to see what is already implemented.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_target_model_source",
            "description": "Read the C++ source code of the current target model to understand its graph structure.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_kernel_source",
            "description": "Read the source code of an existing kernel file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The relative path of the file (e.g., 'cpu/general/matmul.hpp' or 'cpu/general/generated/00010.hpp.failed').",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "report_issue",
            "description": "Report an issue with the testing harness, codebase, or environment. Use this if a failure seems completely anomalous or outside your control.",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_description": {
                        "type": "string",
                        "description": "Detailed explanation of the issue encountered.",
                    }
                },
                "required": ["issue_description"],
            },
        },
    },
]

print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


class WorkerAgent(threading.Thread):
    def __init__(self, agent_id, config):
        super().__init__(daemon=True)
        self.agent_id = agent_id
        self.api_url = config["url"]
        self.model = config["model"]
        self.target_model = config["target_model"]

    def call_bench_api(self, path, method="GET", json_data=None):
        url = f"{BENCH_API_URL}{path}"
        try:
            if method == "GET":
                res = requests.get(url, params=json_data)
            else:
                res = requests.post(url, json=json_data)
            return res.json()
        except Exception as e:
            return {"error": f"Failed to reach Benchmark API: {e!s}"}

    def handle_tool_call(self, tool_call):
        name = tool_call["function"]["name"]

        try:
            args = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError:
            return {"error": "Invalid JSON arguments generated"}

        safe_print(f"\n[Agent {self.agent_id} executing tool: {name}]")

        if name == "get_hw_info":
            return self.call_bench_api("/api/hwinfo")
        elif name == "get_performance_history":
            return self.call_bench_api("/api/history")
        elif name == "get_analysis":
            return self.call_bench_api(
                "/api/analyze", json_data={"target_model": self.target_model}
            )
        elif name == "read_benchmarks":
            # Pass the target model context
            args["target_model"] = self.target_model
            return self.call_bench_api("/api/read_benchmarks", json_data=args)
        elif name == "read_target_model_source":
            return self.call_bench_api(
                "/api/kernels/read_model", json_data={"target_model": self.target_model}
            )
        elif name == "list_kernel_files":
            return self.call_bench_api("/api/kernels/list")
        elif name == "read_kernel_source":
            return self.call_bench_api("/api/kernels/read_source", json_data=args)
        elif name == "report_issue":
            args["agent_id"] = self.agent_id
            res = self.call_bench_api("/api/reports", method="POST", json_data=args)
            return {"status": "Issue reported successfully.", "details": res}
        elif name == "submit_and_test_kernel":
            args["target_model"] = self.target_model
            res = self.call_bench_api(
                "/api/kernels/test", method="POST", json_data=args
            )
            job_id = res.get("job_id")
            if not job_id:
                return {"error": "Submission failed", "details": res}

            safe_print(
                f"  -> [Agent {self.agent_id}] Job {job_id} queued for {self.target_model}. Polling..."
            )
            while True:
                time.sleep(5)
                status = self.call_bench_api(f"/api/jobs/{job_id}")
                if status.get("status") in ["completed", "failed"]:
                    safe_print(
                        f"  -> [Agent {self.agent_id}] Job finished with status '{status.get('status')}'"
                    )
                    return status

        return {"error": f"Unknown tool {name}"}

    def get_initial_messages(self):
        return [
            {
                "role": "system",
                "content": (
                    f"You are an elite C++ and CUDA/NEON performance optimization AI agent. "
                    f"Your target model for optimization is {self.target_model}. "
                    "Your goal is to optimize tensor operations to reduce 'Total Estimated Execution Time'. "
                    "You work in a loop: analyze current performance, generate an optimized kernel, submit it, "
                    "and learn from the test results and benchmarks. "
                    "The test pipeline steps are: Compile -> Test(No Rec) -> Matched in Graph -> Test(Records) -> Benchmark -> Extracted in final graph. "
                    "Iterate infinitely. Use the provided tools.\n\n"
                    "CRITICAL INSTRUCTIONS:\n"
                    "1. Your conversation history is reset after EVERY kernel submission to keep the prompt context small. You MUST call `get_performance_history` in your first step to remember past tests!\n"
                    "2. To avoid repeating previous mistakes, locate failed jobs in the history and strictly read their error messages (which contain full compiler/test output).\n"
                    "3. If you want to read a failed kernel's code, use `read_kernel_source` and pass the `agent_file_path` provided for it in the history.\n"
                    "4. A kernel is only considered successful if it passes ALL stages (including being extracted in the final graph which means it was faster than previous options). Failures at any stage will mark it as failed.\n"
                    "5. If you encounter persistent bugs or environment problems outside your control, use the `report_issue` tool."
                ),
            },
            {
                "role": "user",
                "content": "Begin optimizing. Step 1: Call `get_performance_history` and `get_analysis`. Step 2: Either read_model to look for new sequences that can be fused (bypassing the need for certain kernels), or choose a specific existing kernel to optimize and read benchmarks to find a target. Step 3: Get hardware info. Step 4: Write and submit your kernel.",
            },
        ]

    def run(self):
        messages = self.get_initial_messages()
        safe_print(
            f"[Agent {self.agent_id}] Started Autonomous Optimization Loop on {self.api_url}..."
        )

        while True:
            try:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                }

                response = requests.post(self.api_url, json=payload)
                response.raise_for_status()
                response_data = response.json()
                message = response_data["choices"][0]["message"]
                messages.append(message)

                if message.get("tool_calls"):
                    reset_context = False
                    for tool_call in message["tool_calls"]:
                        result = self.handle_tool_call(tool_call)

                        if tool_call["function"]["name"] == "submit_and_test_kernel":
                            reset_context = True

                        content = json.dumps(result, indent=2)
                        if len(content) > 10000:
                            content = "Content exceeded maximum length. Please narrow parameters."

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "name": tool_call["function"]["name"],
                                "content": content,
                            }
                        )

                    if reset_context:
                        safe_print(
                            f"\n[Agent {self.agent_id}] Submission complete! Resetting context...\n"
                        )
                        messages = self.get_initial_messages()

                else:
                    safe_print(
                        f"\n[Agent {self.agent_id} says]:\n{message.get('content')}\n"
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": "Please continue optimizing. Generate and submit your next kernel.",
                        }
                    )

            except Exception as e:
                safe_print(
                    f"[Agent {self.agent_id}] Error communicating with LLM server: {e}"
                )
                time.sleep(10)


if __name__ == "__main__":
    threads = []
    agent_counter = 1

    for config in AGENT_CONFIGS:
        num_instances = config.get("instances", 1)
        for _ in range(num_instances):
            agent_id = f"Agent-{agent_counter}"
            t = WorkerAgent(agent_id, config)
            t.start()
            threads.append(t)
            agent_counter += 1

    # Keep main thread alive
    for t in threads:
        t.join()
