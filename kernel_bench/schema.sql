CREATE SCHEMA IF NOT EXISTS kernel_bench;

CREATE TABLE IF NOT EXISTS kernel_bench.goals (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    fingerprint text UNIQUE NOT NULL,
    name text NOT NULL,
    cache_sha256 text NOT NULL,
    cache_data bytea NOT NULL,
    source_archive bytea NOT NULL,
    metadata jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS kernel_bench.targets (
    id text PRIMARY KEY,
    expression text NOT NULL
);
-- Long fused expressions can exceed PostgreSQL's B-tree entry limit. Their
-- SHA-256 primary key already supplies identity; do not index the full text.
ALTER TABLE kernel_bench.targets DROP CONSTRAINT IF EXISTS targets_expression_key;
CREATE TABLE IF NOT EXISTS kernel_bench.configurations (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    goal_id bigint NOT NULL REFERENCES kernel_bench.goals(id),
    target_id text NOT NULL REFERENCES kernel_bench.targets(id),
    fingerprint text NOT NULL,
    configuration jsonb NOT NULL,
    matching_kernels jsonb NOT NULL,
    UNIQUE (goal_id, target_id, fingerprint)
);
CREATE INDEX IF NOT EXISTS configurations_target ON kernel_bench.configurations(goal_id, target_id);
CREATE TABLE IF NOT EXISTS kernel_bench.jobs (
    id uuid PRIMARY KEY,
    goal_id bigint NOT NULL REFERENCES kernel_bench.goals(id),
    target_id text NOT NULL REFERENCES kernel_bench.targets(id),
    idea text NOT NULL,
    source text NOT NULL,
    kernel_name text NOT NULL,
    backend text NOT NULL CHECK (backend IN ('cpu', 'cuda')),
    status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    step text NOT NULL DEFAULT 'queued',
    result jsonb NOT NULL DEFAULT '{}',
    error text,
    created_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    completed_at timestamptz
);
CREATE INDEX IF NOT EXISTS jobs_queue ON kernel_bench.jobs(created_at) WHERE status = 'queued';
CREATE INDEX IF NOT EXISTS jobs_goal_history ON kernel_bench.jobs(goal_id, created_at DESC);
CREATE INDEX IF NOT EXISTS jobs_target_history ON kernel_bench.jobs(goal_id, target_id, created_at DESC);
CREATE TABLE IF NOT EXISTS kernel_bench.events (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id uuid NOT NULL REFERENCES kernel_bench.jobs(id),
    step text NOT NULL,
    message text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS events_job ON kernel_bench.events(job_id, id);
CREATE TABLE IF NOT EXISTS kernel_bench.measurements (
    job_id uuid NOT NULL REFERENCES kernel_bench.jobs(id),
    configuration_id bigint NOT NULL REFERENCES kernel_bench.configurations(id),
    kernel_id text NOT NULL,
    kernel_name text NOT NULL,
    is_candidate boolean NOT NULL,
    samples_ms double precision[] NOT NULL,
    median_ms double precision NOT NULL CHECK (median_ms > 0),
    PRIMARY KEY (job_id, configuration_id, kernel_id)
);
CREATE TABLE IF NOT EXISTS kernel_bench.feedback (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    kind text NOT NULL CHECK (kind IN ('report', 'suggestion')),
    data jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE kernel_bench.jobs ADD COLUMN IF NOT EXISTS error_details jsonb;
CREATE TABLE IF NOT EXISTS kernel_bench.artifacts (
    id uuid PRIMARY KEY,
    job_id uuid NOT NULL REFERENCES kernel_bench.jobs(id) ON DELETE CASCADE,
    name text NOT NULL,
    media_type text NOT NULL,
    data bytea NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (job_id, name)
);
CREATE INDEX IF NOT EXISTS artifacts_job ON kernel_bench.artifacts(job_id);
