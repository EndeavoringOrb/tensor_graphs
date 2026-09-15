# File: kernel_bench/agent.py
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

BENCH_SERVER_URL = os.environ.get("BENCH_SERVER_URL", "http://localhost:8080")

DEFAULT_AGENT_CONFIGS = [
    {
        "url": os.environ.get("LLM_API_URL", "http://localhost:11434/v1/chat/completions"),
        "model": os.environ.get("LLM_MODEL", "qwen3.6:35b"),
        "target_model": "gemma-3-270m",
        "instances": 1,
    }
]

print_lock = threading.Lock()


def safePrint(*args, **kwargs) -> None:
    with print_lock:
        print(*args, **kwargs)


class KernelBenchClient:
    """Client for any agentic harness to interface with the KernelBench server at localhost:8080."""

    def __init__(self, base_url: str = BENCH_SERVER_URL, token: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.token = token or os.environ.get("BENCH_SECRET_TOKEN")
        if not self.token:
            token_path = Path(__file__).resolve().parent.parent / ".bench_token"
            if token_path.exists():
                try:
                    self.token = token_path.read_text(encoding="utf-8").strip()
                except Exception:
                    pass
        self.session = requests.Session()
        if self.token:
            self.session.headers["X-Bench-Token"] = self.token

    def getAgentIndex(self, target_model: str = "gemma-3-270m", format: str = "json") -> Any:
        params = {"target_model": target_model}
        if format != "json":
            params["format"] = format
        headers = {"Accept": "application/json" if format == "json" else "text/markdown"}
        res = self.session.get(f"{self.base_url}/api/agent", params=params, headers=headers)
        res.raise_for_status()
        if format == "json":
            return res.json()
        return res.text


    def getStatus(self, target_model: str = "gemma-3-270m") -> Dict[str, Any]:
        res = self.session.get(f"{self.base_url}/api/status", params={"target_model": target_model})
        res.raise_for_status()
        return res.json()

    def getTools(self) -> List[Dict[str, Any]]:
        res = self.session.get(f"{self.base_url}/api/tools")
        res.raise_for_status()
        return res.json().get("tools", [])

    def getOpenApi(self) -> Dict[str, Any]:
        res = self.session.get(f"{self.base_url}/api/openapi.json")
        res.raise_for_status()
        return res.json()

    def getBottleneckAnalysis(
        self,
        target_model: str = "gemma-3-270m",
        search: str = "",
        filter_type: str = "all",
        sort_by: str = "time",
        sort_order: str = "desc",
        limit: int = 0,
    ) -> Dict[str, Any]:
        """Fetch bottleneck analysis for the current performance cache with optional filtering."""
        params: Dict[str, Any] = {"target_model": target_model}
        if search:
            params["search"] = search
        if filter_type and filter_type != "all":
            params["filter"] = filter_type
        if sort_by != "time":
            params["sort_by"] = sort_by
        if sort_order != "desc":
            params["sort_order"] = sort_order
        if limit > 0:
            params["limit"] = limit
        res = self.session.get(f"{self.base_url}/api/analyze", params=params)
        res.raise_for_status()
        return res.json()

    def getVersions(self) -> List[Dict[str, Any]]:
        res = self.session.get(f"{self.base_url}/api/versions")
        res.raise_for_status()
        return res.json().get("versions", [])

    def getVersionDetails(self, version_id: int) -> Dict[str, Any]:
        res = self.session.get(f"{self.base_url}/api/versions/{version_id}")
        res.raise_for_status()
        return res.json()

    def queryBenchmarkRecords(self, op: str = "", shape: str = "") -> List[Dict[str, Any]]:
        res = self.session.get(f"{self.base_url}/api/benchmarks/records", params={"op": op, "shape": shape})
        res.raise_for_status()
        return res.json().get("records", [])

    def listKernelFiles(self) -> List[str]:
        res = self.session.get(f"{self.base_url}/api/kernels/list")
        res.raise_for_status()
        return res.json().get("files", [])

    def readKernelSource(self, path: str) -> str:
        res = self.session.get(f"{self.base_url}/api/kernels/read_source", params={"path": path})
        res.raise_for_status()
        return res.json().get("content", "")

    def listCoreHeaders(self) -> List[str]:
        res = self.session.get(f"{self.base_url}/api/core/list")
        res.raise_for_status()
        return res.json().get("files", [])

    def readCoreHeader(self, path: str) -> str:
        res = self.session.get(f"{self.base_url}/api/core/read_source", params={"path": path})
        res.raise_for_status()
        return res.json().get("content", "")

    def readModelSource(self, target_model: str = "gemma-3-270m") -> str:
        res = self.session.get(f"{self.base_url}/api/model/source", params={"target_model": target_model})
        res.raise_for_status()
        return res.json().get("content", "")

    def submitIteration(
        self,
        idea: str,
        source: str = "",
        filename: str = "",
        opname: str = "",
        backend: str = "cuda",
        target_model: str = "gemma-3-270m",
        pp: int = 512,
        tg: int = 128,
        kernel_name: str = "",
    ) -> Dict[str, Any]:
        payload = {
            "idea": idea,
            "source": source,
            "filename": filename,
            "opname": opname,
            "backend": backend,
            "target_model": target_model,
            "pp": pp,
            "tg": tg,
            "kernel_name": kernel_name,
        }
        res = self.session.post(f"{self.base_url}/api/iteration/submit", json=payload)
        res.raise_for_status()
        return res.json()

    def getJobStatus(self, job_id: str) -> Dict[str, Any]:
        res = self.session.get(f"{self.base_url}/api/jobs/{job_id}")
        res.raise_for_status()
        return res.json()

    def getJobLogs(self, job_id: str) -> Dict[str, Any]:
        res = self.session.get(f"{self.base_url}/api/jobs/{job_id}/logs")
        res.raise_for_status()
        return res.json()

    def pollJobUntilComplete(self, job_id: str, interval: int = 5, timeout: int = 1800) -> Dict[str, Any]:
        start = time.time()
        while time.time() - start < timeout:
            job = self.getJobStatus(job_id)
            status = job.get("status")
            step = job.get("step")
            safePrint(f"  [Job {job_id}] Status: {status}, Current Step: {step}")
            if status in ("completed", "failed"):
                return job
            time.sleep(interval)
        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")

    def suggestApiChange(
        self,
        title: str,
        description: str,
        suggested_endpoint: str = "",
        proposed_changes: str = "",
        category: str = "new_endpoint",
    ) -> Dict[str, Any]:
        payload = {
            "title": title,
            "description": description,
            "suggested_endpoint": suggested_endpoint,
            "proposed_changes": proposed_changes,
            "category": category,
        }
        res = self.session.post(f"{self.base_url}/api/suggestions", json=payload)
        res.raise_for_status()
        return res.json()

    def getSuggestions(self, category: str = "", status: str = "") -> List[Dict[str, Any]]:
        params = {}
        if category:
            params["category"] = category
        if status:
            params["status"] = status
        res = self.session.get(f"{self.base_url}/api/suggestions", params=params)
        res.raise_for_status()
        return res.json().get("suggestions", [])



class WorkerAgent(threading.Thread):
    """Autonomous agent runner driving iterative optimization against KernelBench."""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(daemon=True)
        self.agent_id = agent_id
        self.api_url = config.get("url", "http://localhost:11434/v1/chat/completions")
        self.model = config.get("model", "qwen3.6:35b")
        self.target_model = config.get("target_model", "gemma-3-270m")
        self.client = KernelBenchClient(config.get("bench_url", BENCH_SERVER_URL))

    def executeTool(self, name: str, args: Dict[str, Any]) -> Any:
        safePrint(f"\n[Agent {self.agent_id} calling tool: {name}]")
        try:
            if name in ("get_agent_index", "get_agent_guidance"):
                return self.client.getAgentIndex(
                    target_model=args.get("target_model", self.target_model),
                    format=args.get("format", "json"),
                )
            elif name == "get_system_status":
                return self.client.getStatus(args.get("target_model", self.target_model))
            elif name == "get_bottleneck_analysis":
                return self.client.getBottleneckAnalysis(
                    target_model=args.get("target_model", self.target_model),
                    search=args.get("search", args.get("query", args.get("q", ""))),
                    filter_type=args.get("filter", args.get("filter_type", "all")),
                    sort_by=args.get("sort_by", "time"),
                    sort_order=args.get("sort_order", "desc"),
                    limit=int(args.get("limit", 0)),
                )
            elif name == "get_versions_history":
                return self.client.getVersions()
            elif name == "get_version_details":
                return self.client.getVersionDetails(args["version_id"])
            elif name == "read_benchmark_records":
                return self.client.queryBenchmarkRecords(args.get("op", ""), args.get("shape", ""))
            elif name == "list_kernel_files":
                return self.client.listKernelFiles()
            elif name == "read_kernel_source":
                return self.client.readKernelSource(args["path"])
            elif name in ("list_core_headers", "list_core_files"):
                return self.client.listCoreHeaders()
            elif name in ("read_core_header", "read_core_source"):
                return self.client.readCoreHeader(args["path"])
            elif name == "read_model_source":
                return self.client.readModelSource(args.get("target_model", self.target_model))
            elif name == "submit_iteration":
                args.setdefault("target_model", self.target_model)
                sub_res = self.client.submitIteration(**args)
                job_id = sub_res.get("job_id")
                if not job_id:
                    return {"error": "Submission failed", "response": sub_res}
                safePrint(f"  [Agent {self.agent_id}] Job {job_id} queued. Polling...")
                return self.client.pollJobUntilComplete(job_id)
            elif name == "get_job_status":
                return self.client.getJobStatus(args["job_id"])
            elif name == "report_issue":
                requests.post(f"{self.client.base_url}/api/reports", json=args)
                return {"status": "success", "message": "Issue recorded"}
            elif name == "suggest_api_change":
                res = self.client.suggestApiChange(**args)
                return {"status": "success", "message": "Suggestion recorded", "details": res}
            return {"error": f"Unknown tool: {name}"}
        except Exception as e:
            return {"error": str(e)}

    def run(self) -> None:
        safePrint(f"[Agent {self.agent_id}] Connecting to KernelBench agent index...")
        tools = self.client.getTools()

        try:
            agent_data = self.client.getAgentIndex(self.target_model)
            if isinstance(agent_data, dict) and agent_data.get("system_prompt"):
                system_prompt = agent_data["system_prompt"]
            else:
                system_prompt = (
                    f"You are an elite CUDA and C++ high-performance optimization AI agent.\n"
                    f"Target Model: {self.target_model}\n"
                    "Goal: Optimize Tensor Graphs kernels to beat llama.cpp for pp512 and tg128.\n"
                    "Rules:\n"
                    "1. NEVER modify existing kernel files; only add new files in tensor_graphs_cpp/kernels/.\n"
                    "2. Keep a running list of changes across versions/0, versions/1, versions/N.\n"
                    "3. Analyze bottlenecks via get_bottleneck_analysis. Eliminate CPU reference fallbacks (e.g. REF_MUL, REF_DIVIDE, REF_NEGATE) by providing CUDA kernels supporting non-contiguous/broadcast strides.\n"
                    "4. Check get_system_status to monitor progress against llama.cpp targets.\n"
                    "5. Iterate continuously until targets are beaten.\n"
                )
        except Exception as err:
            safePrint(f"[Agent {self.agent_id}] Note: Could not fetch agent index ({err}), using default prompt.")
            system_prompt = (
                f"You are an elite CUDA and C++ high-performance optimization AI agent.\n"
                f"Target Model: {self.target_model}\n"
                "Goal: Optimize Tensor Graphs kernels to beat llama.cpp for pp512 and tg128.\n"
                "Rules:\n"
                "1. NEVER modify existing kernel files; only add new files in tensor_graphs_cpp/kernels/.\n"
                "2. Keep a running list of changes across versions/0, versions/1, versions/N.\n"
                "3. Analyze bottlenecks via get_bottleneck_analysis. Eliminate CPU reference fallbacks (e.g. REF_MUL, REF_DIVIDE, REF_NEGATE) by providing CUDA kernels supporting non-contiguous/broadcast strides.\n"
                "4. Check get_system_status to monitor progress against llama.cpp targets.\n"
                "5. Iterate continuously until targets are beaten.\n"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Begin optimizing. Step 1: Call get_system_status and get_bottleneck_analysis. Step 2: Formulate your idea, write your kernel, and submit via submit_iteration.",
            },
        ]

        while True:
            try:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                }
                res = requests.post(self.api_url, json=payload, timeout=120)
                res.raise_for_status()
                data = res.json()
                choice = data["choices"][0]
                message = choice["message"]
                messages.append(message)

                if message.get("tool_calls"):
                    for call in message["tool_calls"]:
                        fn_name = call["function"]["name"]
                        try:
                            fn_args = json.loads(call["function"]["arguments"])
                        except Exception:
                            fn_args = {}
                        tool_result = self.executeTool(fn_name, fn_args)

                        # Truncate large tool content to prevent context overflow
                        result_str = json.dumps(tool_result)
                        if len(result_str) > 8000:
                            result_str = result_str[:8000] + "... [truncated]"

                        messages.append({
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": fn_name,
                            "content": result_str,
                        })

                        if fn_name == "submit_iteration" and tool_result.get("target_beaten"):
                            safePrint(f"\n[Agent {self.agent_id}] TARGET BEATEN! Goal achieved!\n")
                            return

                else:
                    safePrint(f"\n[Agent {self.agent_id} response]:\n{message.get('content')}\n")
                    messages.append({
                        "role": "user",
                        "content": "Analyze the results from the last run, formulate the next hypothesis, and submit your next kernel.",
                    })

            except Exception as err:
                safePrint(f"[Agent {self.agent_id}] Error in optimization loop: {err}")
                time.sleep(10)


if __name__ == "__main__":
    client = KernelBenchClient(BENCH_SERVER_URL)
    try:
        status = client.getStatus()
        print(f"[KernelBench] Connected to server at {BENCH_SERVER_URL}")
        print(f"  Target: {status.get('target_model')}")
        print(f"  Current Best: {status.get('current_best')}")
        print(f"  Llama.cpp Targets: {status.get('targets')}")
    except Exception as e:
        print(f"[KernelBench] Warning: Could not reach server at {BENCH_SERVER_URL}: {e}")
        print("Make sure the server is running: .venv/bin/python -m kernel_bench.app")
        sys.exit(1)

    threads = []
    agent_counter = 1
    for config in DEFAULT_AGENT_CONFIGS:
        for _ in range(config.get("instances", 1)):
            t = WorkerAgent(f"Agent-{agent_counter}", config)
            t.start()
            threads.append(t)
            agent_counter += 1

    for t in threads:
        t.join()
