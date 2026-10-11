"""Server-wide discussion with server-derived attribution."""
import hashlib
import math
import secrets
import time
import uuid

from flask import jsonify, render_template, request
from psycopg.types.json import Jsonb


def credentialHash(credential):
    return hashlib.sha256(credential.encode()).hexdigest()


def boundedInteger(value, name, minimum=1, maximum=2**63 - 1):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer") from None
    if str(number) != str(value) or not minimum <= number <= maximum:
        raise ValueError(f"Invalid {name}")
    return number


def messageData(data):
    if not isinstance(data, dict):
        raise TypeError("Expected a JSON object")
    body = data.get("body")
    if not isinstance(body, str) or not body.strip() or len(body) > 16000:
        raise ValueError("body must contain 1–16000 characters")
    key = data.get("idempotency_key")
    if key is not None and (not isinstance(key, str) or not key.strip() or len(key) > 200):
        raise ValueError("idempotency_key must contain 1–200 characters")
    result = {"body": body, "idempotency_key": key}
    for name in ("reply_to_message_id", "goal_id"):
        result[name] = boundedInteger(data[name], name) if data.get(name) is not None else None
    result["target_id"] = data.get("target_id")
    if result["target_id"] is not None and (not isinstance(result["target_id"], str) or len(result["target_id"]) > 200):
        raise ValueError("Invalid target_id")
    result["job_id"] = str(uuid.UUID(str(data["job_id"]))) if data.get("job_id") is not None else None
    return result


def authorIdentity(store, bench_authorized):
    credential = request.headers.get("X-Agent-Token")
    if credential:
        rows = store.query("SELECT id, display_name, created_at FROM kernel_bench.agents WHERE credential_hash = %s",
                           (credentialHash(credential),))
        if not rows:
            raise PermissionError("Invalid X-Agent-Token")
        return rows[0]
    # Only authenticated browser requests can create operator posts. Header-only
    # API callers must register so the shared bench token cannot impersonate agents.
    if bench_authorized(request.cookies.get("bench_token", "")):
        if request.headers.get("Sec-Fetch-Site") == "cross-site" or (
                request.headers.get("Origin") and request.headers["Origin"] != request.host_url.rstrip("/")):
            raise PermissionError("Cross-origin operator posting is forbidden")
        return None
    raise PermissionError("Posting requires X-Agent-Token; register at /api/agents/register")


MESSAGE_SELECT = """SELECT m.*, COALESCE(a.display_name, CASE WHEN m.author_key = 'legacy' THEN 'Unknown author' ELSE 'Operator' END) AS display_name,
    CASE WHEN m.author_key = 'legacy' THEN 'unknown' WHEN m.author_id IS NULL THEN 'operator' ELSE 'agent' END AS author_type
    FROM kernel_bench.messages m LEFT JOIN kernel_bench.agents a ON a.id = m.author_id"""


def saveMessage(store, channel_id, author, data, feedback=None, legacy=False):
    author_id = author["id"] if author else None
    author_key = str(author_id) if author_id else ("legacy" if legacy else "operator")
    with store.connection() as connection:
        # Lock serializes message IDs with commit order, so cursor readers cannot
        # skip a concurrent insert which commits after a higher ID.
        connection.execute("SELECT pg_advisory_xact_lock(724189004)")
        if not connection.execute("SELECT id FROM kernel_bench.channels WHERE id = %s", (channel_id,)).fetchone():
            raise ValueError("Unknown channel")
        if data["idempotency_key"] is not None:
            existing = connection.execute(MESSAGE_SELECT + " WHERE m.author_key = %s AND m.idempotency_key = %s",
                                          (author_key, data["idempotency_key"])).fetchone()
            if existing:
                if any(str(existing[field]) != str(data[field]) for field in data if field != "idempotency_key") or existing["channel_id"] != channel_id:
                    raise ValueError("Idempotency key already used for a different message")
                if feedback is not None:
                    previous = connection.execute("SELECT kind, data FROM kernel_bench.feedback WHERE id = %s", (existing["feedback_id"],)).fetchone()
                    if not previous or previous["kind"] != feedback[0] or previous["data"] != feedback[1]:
                        raise ValueError("Idempotency key already used for different feedback")
                return existing, False
        if data["reply_to_message_id"] is not None:
            reply = connection.execute("SELECT channel_id FROM kernel_bench.messages WHERE id = %s", (data["reply_to_message_id"],)).fetchone()
            if not reply or reply["channel_id"] != channel_id:
                raise ValueError("Reply must reference a message in this channel")
        for field, table in (("goal_id", "goals"), ("target_id", "targets"), ("job_id", "jobs")):
            if data[field] is not None and not connection.execute(f"SELECT id FROM kernel_bench.{table} WHERE id = %s", (data[field],)).fetchone():
                raise ValueError(f"Unknown {field}")
        if data["job_id"] is not None:
            job = connection.execute("SELECT goal_id, target_id FROM kernel_bench.jobs WHERE id = %s", (data["job_id"],)).fetchone()
            if any(data[field] is not None and str(data[field]) != str(job[field]) for field in ("goal_id", "target_id")):
                raise ValueError("Job context does not match goal/target")
        if (data["goal_id"] is not None and data["target_id"] is not None
                and not connection.execute("SELECT id FROM kernel_bench.configurations WHERE goal_id = %s AND target_id = %s LIMIT 1",
                                           (data["goal_id"], data["target_id"])).fetchone()):
            raise ValueError("Target does not belong to goal")
        feedback_id = None
        if feedback is not None:
            feedback_id = connection.execute("INSERT INTO kernel_bench.feedback(kind, data, agent_id) VALUES (%s, %s, %s) RETURNING id",
                                             (feedback[0], Jsonb(feedback[1]), author_id)).fetchone()["id"]
        row = connection.execute("""INSERT INTO kernel_bench.messages
            (channel_id, author_id, author_key, body, reply_to_message_id, goal_id, target_id, job_id, idempotency_key, feedback_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (channel_id, author_id, author_key, data["body"], data["reply_to_message_id"], data["goal_id"],
             data["target_id"], data["job_id"], data["idempotency_key"], feedback_id)).fetchone()
        return connection.execute(MESSAGE_SELECT + " WHERE m.id = %s", (row["id"],)).fetchone(), True


def installBoard(app, store, bench_authorized):
    @app.errorhandler(PermissionError)
    def identityError(error):
        return jsonify({"error": str(error)}), 401

    @app.post("/api/agents/register")
    def registerAgent():
        data = request.get_json()
        name = data.get("display_name") if isinstance(data, dict) else None
        if not isinstance(name, str) or not name.strip() or len(name) > 80:
            raise ValueError("display_name must contain 1–80 characters")
        credential = secrets.token_urlsafe(32)
        rows = store.query("""INSERT INTO kernel_bench.agents(id, display_name, credential_hash)
            VALUES (%s, %s, %s) RETURNING id, display_name, created_at""",
            (uuid.uuid4(), name.strip(), credentialHash(credential)))
        response = jsonify({**rows[0], "agent_token": credential})
        response.headers["Cache-Control"] = "no-store"
        return response, 201

    @app.get("/api/agents/me")
    def currentAgent():
        if not request.headers.get("X-Agent-Token"):
            raise PermissionError("X-Agent-Token required")
        return jsonify(authorIdentity(store, bench_authorized))

    @app.get("/board")
    def board():
        return render_template("board.html")

    @app.get("/api/channels")
    def channels():
        return jsonify({"channels": store.query("""SELECT c.*, COALESCE(max(m.id), 0) AS latest_message_id
            FROM kernel_bench.channels c LEFT JOIN kernel_bench.messages m ON m.channel_id = c.id
            GROUP BY c.id ORDER BY c.id""")})

    @app.route("/api/channels/<channel_id>/messages", methods=["GET", "POST"])
    def messages(channel_id):
        if request.method == "POST":
            row, created = saveMessage(store, channel_id, authorIdentity(store, bench_authorized), messageData(request.get_json()))
            return jsonify(row), 201 if created else 200
        if not store.query("SELECT id FROM kernel_bench.channels WHERE id = %s", (channel_id,)):
            return jsonify({"error": "Unknown channel"}), 404
        cursor = boundedInteger(request.args.get("cursor", "0"), "cursor", minimum=0)
        limit = boundedInteger(request.args.get("limit", "50"), "limit", maximum=200)
        try:
            wait = float(request.args.get("wait", "0"))
        except ValueError:
            raise ValueError("wait must be from 0 to 25 seconds") from None
        if not math.isfinite(wait) or not 0 <= wait <= 25:
            raise ValueError("wait must be from 0 to 25 seconds")
        deadline = time.monotonic() + wait
        stop_event = app.extensions["kernel_bench_stop"]
        from .polling import waitSlot
        with waitSlot(wait):
            while True:
                rows = store.query(MESSAGE_SELECT + " WHERE m.channel_id = %s AND m.id > %s ORDER BY m.id LIMIT %s", (channel_id, cursor, limit + 1))
                if rows or time.monotonic() >= deadline or stop_event.is_set():
                    break
                stop_event.wait(min(0.25, max(0, deadline - time.monotonic())))
        page = rows[:limit]
        return jsonify({"messages": page, "next_cursor": page[-1]["id"] if page else cursor, "has_more": len(rows) > limit})

    @app.get("/api/messages/<message_id>")
    def message(message_id):
        rows = store.query(MESSAGE_SELECT + " WHERE m.id = %s", (boundedInteger(message_id, "message_id"),))
        return (jsonify(rows[0]), 200) if rows else (jsonify({"error": "Unknown message"}), 404)
