"""Bounded polling and stable history cursors."""
import base64
import hashlib
import json
import math
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime

from flask import abort, current_app, jsonify, make_response, request

from .board import boundedInteger


@contextmanager
def waitSlot(wait):
    # Reserve HTTP capacity for submissions and other immediate requests.
    slots = current_app.extensions.setdefault("kernel_bench_poll_slots", threading.BoundedSemaphore(8))
    acquired = bool(wait) and slots.acquire(blocking=False)
    if wait and not acquired:
        abort(make_response(jsonify({"error": "Long polling capacity reached; retry shortly"}), 429, {"Retry-After": "1"}))
    try:
        yield
    finally:
        if acquired:
            slots.release()


def encodeCursor(row):
    return base64.urlsafe_b64encode(json.dumps([row['created_at'].isoformat(), str(row['job_id'])]).encode()).decode()


def decodeCursor(value):
    if not isinstance(value, str) or len(value) > 200:
        raise ValueError("Invalid job cursor")
    try:
        created_at, job_id = json.loads(base64.b64decode(value, altchars=b'-_', validate=True))
        if not isinstance(created_at, str) or not isinstance(job_id, str):
            raise TypeError("Invalid cursor fields")
        timestamp = datetime.fromisoformat(created_at)
        if timestamp.tzinfo is None:
            raise ValueError('Missing timezone')
        return timestamp, uuid.UUID(job_id)
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        raise ValueError('Invalid job cursor') from error


def listJobs(store, goal_id):
    limit = boundedInteger(request.args.get('limit', '200'), 'limit', maximum=200)
    conditions = ['goal_id = %s']
    parameters = [goal_id]
    for key, allowed in [('status', {'queued', 'running', 'completed', 'failed'}),
                         ('mode', {'lint', 'compile', 'evaluate'}), ('target_id', None), ('agent_id', None)]:
        value = request.args.get(key)
        if value:
            if allowed is not None and value not in allowed:
                raise ValueError(f'Invalid {key}')
            if key == 'agent_id':
                value = uuid.UUID(value)
            conditions.append(f'{key} = %s')
            parameters.append(value)
    if request.args.get('cursor'):
        conditions.append('(created_at, id) < (%s, %s)')
        parameters.extend(decodeCursor(request.args['cursor']))
    parameters.append(limit + 1)
    rows = store.query('''SELECT id AS job_id, agent_id, target_id, idea, kernel_name, backend, mode, status, step,
        revision, stage_started_at, result, error, error_details, created_at, started_at, completed_at
        FROM kernel_bench.jobs WHERE ''' + ' AND '.join(conditions) + ' ORDER BY created_at DESC, id DESC LIMIT %s', parameters)
    has_more = len(rows) > limit
    rows = rows[:limit]
    return jsonify({'jobs': rows, 'has_more': has_more, 'next_cursor': encodeCursor(rows[-1]) if has_more else None})


def pollJob(store, current_job, job_id, stop_event):
    fields = request.args.get('fields')
    projection = fields.split(',') if fields else None
    allowed = {'job_id', 'goal_id', 'target_id', 'agent_id', 'idea', 'source', 'kernel_name', 'backend', 'mode',
               'status', 'step', 'revision', 'stage_started_at', 'result', 'error', 'error_details',
               'created_at', 'started_at', 'completed_at', 'queue'}
    if projection and (len(projection) != len(set(projection)) or any(field not in allowed for field in projection)):
        raise ValueError('Unknown or duplicate job projection field')
    after = request.args.get('after_revision')
    after = boundedInteger(after, 'after_revision', minimum=0) if after is not None else None
    try:
        wait = float(request.args.get('wait', '0'))
    except ValueError:
        raise ValueError('wait must be a number from 0 to 25') from None
    if not math.isfinite(wait) or not 0 <= wait <= 25 or (wait and after is None):
        raise ValueError('wait requires after_revision and must be from 0 to 25 seconds')
    deadline = time.monotonic() + wait
    with waitSlot(wait):
        while True:
            row = current_job(job_id)
            if row is None:
                return jsonify({'error': 'Unknown job for this goal'}), 404
            if after is None or row['revision'] > after or row['status'] in ('completed', 'failed') or time.monotonic() >= deadline:
                break
            if stop_event.wait(min(0.25, max(0, deadline - time.monotonic()))):
                break
    # Include projection in the validator: representations have different bodies.
    etag = f'{job_id}-{row["revision"]}-' + (','.join(projection) if projection else 'full')
    if projection is None or 'queue' in projection:
        row['queue'] = store.queueInfo(row['goal_id'], row)
        etag += '-' + hashlib.sha256(json.dumps(row['queue'], default=str, sort_keys=True).encode()).hexdigest()[:16]
    response = jsonify({})
    if request.if_none_match.contains_weak(etag):
        response.status_code = 304
    else:
        row['job_id'] = row.pop('id')
        response = jsonify({key: row[key] for key in projection} if projection else row)
    response.set_etag(etag)
    response.headers['Cache-Control'] = 'private, no-cache'
    return response
