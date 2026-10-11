"""Small client for any agent harness; agents own hypothesis generation."""
import os
import time

import requests


class KernelBenchClient:
    def __init__(self, base_url=None, token=None):
        self.base_url = (base_url or os.environ.get("BENCH_SERVER_URL", "http://localhost:8080")).rstrip("/")
        self.session = requests.Session()
        self.session.headers["X-Bench-Token"] = token or os.environ.get("BENCH_SECRET_TOKEN", "")

    def get(self, path, **params):
        response = self.session.get(self.base_url + path, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def getAgentIndex(self):
        return self.get("/api/agent")

    def getStatus(self):
        return self.get("/api/status")

    def getTools(self):
        return self.get("/api/tools")["tools"]

    def listTargets(self, search=""):
        return self.get("/api/targets", search=search)["targets"]

    def getTarget(self, target_id):
        return self.get(f"/api/targets/{target_id}")

    def readSource(self, path):
        return self.get("/api/source", path=path)["content"]

    def getJobs(self, target_id=""):
        return self.get("/api/jobs", target_id=target_id)["jobs"]

    def submitIteration(self, target_id, source, idea="", backend="cuda", mode="evaluate"):
        response = self.session.post(self.base_url + "/api/iteration/submit",
                                     json={"target_id": target_id, "source": source, "idea": idea, "backend": backend, "mode": mode}, timeout=30)
        response.raise_for_status()
        return response.json()

    def lintIteration(self, target_id, source, idea="", backend="cuda"):
        return self.submitIteration(target_id, source, idea, backend, mode="lint")

    def getJobStatus(self, job_id):
        return self.get(f"/api/jobs/{job_id}")

    def getJobLogs(self, job_id, after_id=None, tail=None, stage=None, grep=None):
        parameters = {key: value for key, value in {"after_id": after_id, "tail": tail, "stage": stage, "grep": grep}.items()
                      if value is not None}
        return self.get(f"/api/jobs/{job_id}/logs", **parameters)

    def getJobArtifacts(self, job_id):
        return self.get(f"/api/jobs/{job_id}/artifacts")["artifacts"]

    def readJobArtifact(self, job_id, artifact_id):
        response = self.session.get(self.base_url + f"/api/jobs/{job_id}/artifacts/{artifact_id}", timeout=30)
        response.raise_for_status()
        return response.content

    def pollJobUntilComplete(self, job_id, interval=5, timeout=1800):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.getJobStatus(job_id)
            if job["status"] in ("completed", "failed"):
                return job
            time.sleep(interval)
        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")
