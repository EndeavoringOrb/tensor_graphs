"""Small client for any agent harness; agents own hypothesis generation."""
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import requests


class KernelBenchClient:
    def __init__(self, base_url=None, token=None, identity_path=".bench_agent.json"):
        configured_url = base_url if base_url is not None else os.environ.get("BENCH_SERVER_URL", "")
        parsed = urlsplit(configured_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Set BENCH_SERVER_URL to an absolute HTTP(S) server URL")
        try:
            if parsed.port == 0:
                raise ValueError("Port cannot be zero")
        except ValueError:
            raise ValueError("Invalid BENCH_SERVER_URL port") from None
        configured_token = token if token is not None else os.environ.get("BENCH_SECRET_TOKEN", "")
        if not configured_token or not configured_token.strip() or any(character.isspace() for character in configured_token):
            raise ValueError("Set BENCH_SECRET_TOKEN to a nonempty token without whitespace")
        self.base_url = configured_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["X-Bench-Token"] = configured_token

        self.identity_path = Path(identity_path)
        self.identity = None
        if self.identity_path.exists():
            self.identity = json.loads(self.identity_path.read_text())
            if self.identity.get("base_url") != self.base_url:
                raise ValueError("Stored agent identity belongs to another server; use a different identity_path")
            self.session.headers["X-Agent-Token"] = self.identity["agent_token"]

    def registerAgent(self, display_name):
        if self.identity is not None:
            return self.getIdentity()
        response = self.session.post(self.base_url + "/api/agents/register",
                                     json={"display_name": display_name}, timeout=30)
        self.checkResponse(response)
        identity = {**response.json(), "base_url": self.base_url}
        self.identity_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(dir=self.identity_path.parent)
        try:
            with os.fdopen(descriptor, "w") as output:
                json.dump(identity, output)
            os.replace(temporary_path, self.identity_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
        self.identity = identity
        self.session.headers["X-Agent-Token"] = identity["agent_token"]
        return self.getIdentity()

    def getIdentity(self):
        return self.get("/api/agents/me")

    def listChannels(self):
        return self.get("/api/channels")["channels"]

    def readMessages(self, channel_id, cursor=0, limit=50, wait=0):
        return self.get(f"/api/channels/{channel_id}/messages", cursor=cursor, limit=limit, wait=wait)

    def getMessage(self, message_id):
        return self.get(f"/api/messages/{message_id}")

    def postMessage(self, channel_id, body, idempotency_key=None, **context):
        key = idempotency_key or str(uuid.uuid4())
        response = self.session.post(self.base_url + f"/api/channels/{channel_id}/messages",
                                     json={**context, "body": body, "idempotency_key": key}, timeout=30)
        self.checkResponse(response)
        return response.json()

    def checkResponse(self, response):
        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            raise requests.HTTPError(f"{error}: {response.text}", response=response, request=response.request) from error

    def get(self, path, /, params=None, **query):
        combined_params = dict(params or {})
        combined_params.update(query)
        response = self.session.get(self.base_url + path, params={key: value for key, value in combined_params.items() if value is not None and value != ""}, timeout=30)
        self.checkResponse(response)
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

    def getJobs(self, target_id="", **filters):
        return self.getJobsPage(target_id=target_id, **filters)["jobs"]

    def getJobsPage(self, **filters):
        return self.get("/api/jobs", **filters)

    def submitIteration(self, target_id, source, idea="", backend="cuda", mode="evaluate"):
        response = self.session.post(self.base_url + "/api/iteration/submit",
                                     json={"target_id": target_id, "source": source, "idea": idea, "backend": backend, "mode": mode}, timeout=30)
        self.checkResponse(response)
        return response.json()

    def lintIteration(self, target_id, source, idea="", backend="cuda"):
        return self.submitIteration(target_id, source, idea, backend, mode="lint")

    def getJobStatus(self, job_id, **polling):
        return self.get(f"/api/jobs/{job_id}", **polling)

    def getJobLogs(self, job_id, after_id=None, tail=None, stage=None, grep=None):
        parameters = {key: value for key, value in {"after_id": after_id, "tail": tail, "stage": stage, "grep": grep}.items()
                      if value is not None}
        return self.get(f"/api/jobs/{job_id}/logs", **parameters)

    def getJobArtifacts(self, job_id):
        return self.get(f"/api/jobs/{job_id}/artifacts")["artifacts"]

    def readJobArtifact(self, job_id, artifact_id):
        response = self.session.get(self.base_url + f"/api/jobs/{job_id}/artifacts/{artifact_id}", timeout=30)
        self.checkResponse(response)
        return response.content

    def pollJobUntilComplete(self, job_id, interval=5, timeout=1800):
        deadline = time.monotonic() + timeout
        revision = None
        while time.monotonic() < deadline:
            job = self.getJobStatus(job_id, after_revision=revision,
                                    wait=min(25, max(0, deadline - time.monotonic())) if revision is not None else 0)
            revision = job["revision"]
            if job["status"] in ("completed", "failed"):
                return job

        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")
