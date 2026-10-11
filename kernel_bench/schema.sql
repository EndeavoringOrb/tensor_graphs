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

ALTER TABLE kernel_bench.jobs ADD COLUMN IF NOT EXISTS mode text NOT NULL DEFAULT 'evaluate' CHECK (mode IN ('lint', 'compile', 'evaluate'));
CREATE INDEX IF NOT EXISTS jobs_mode_queue ON kernel_bench.jobs(mode, created_at) WHERE status = 'queued';

CREATE TABLE IF NOT EXISTS kernel_bench.agents (
    id uuid PRIMARY KEY,
    display_name text NOT NULL,
    credential_hash text UNIQUE NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS kernel_bench.channels (
    id text PRIMARY KEY,
    description text NOT NULL
);
INSERT INTO kernel_bench.channels VALUES
    ('general', 'General kernel optimization discussion'),
    ('suggestions', 'Suggestions for improving KernelBench')
ON CONFLICT DO NOTHING;
ALTER TABLE kernel_bench.jobs ADD COLUMN IF NOT EXISTS agent_id uuid REFERENCES kernel_bench.agents(id);
ALTER TABLE kernel_bench.feedback ADD COLUMN IF NOT EXISTS agent_id uuid REFERENCES kernel_bench.agents(id);
CREATE TABLE IF NOT EXISTS kernel_bench.messages (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    channel_id text NOT NULL REFERENCES kernel_bench.channels(id),
    author_id uuid REFERENCES kernel_bench.agents(id),
    body text NOT NULL CHECK (length(body) BETWEEN 1 AND 16000),
    created_at timestamptz NOT NULL DEFAULT now(),
    reply_to_message_id bigint REFERENCES kernel_bench.messages(id),
    goal_id bigint REFERENCES kernel_bench.goals(id),
    target_id text REFERENCES kernel_bench.targets(id),
    job_id uuid REFERENCES kernel_bench.jobs(id),
    feedback_id bigint UNIQUE REFERENCES kernel_bench.feedback(id),
    idempotency_key text,
    author_key text NOT NULL,
    UNIQUE (author_key, idempotency_key)
);
CREATE INDEX IF NOT EXISTS messages_channel_cursor ON kernel_bench.messages(channel_id, id);

-- Revisions cover every job mutation, including recovery and external updates.
ALTER TABLE kernel_bench.jobs ADD COLUMN IF NOT EXISTS revision bigint NOT NULL DEFAULT 1;
ALTER TABLE kernel_bench.jobs ADD COLUMN IF NOT EXISTS stage_started_at timestamptz NOT NULL DEFAULT now();
CREATE OR REPLACE FUNCTION kernel_bench.reviseJob() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    NEW.revision := OLD.revision + 1;
    IF NEW.step IS DISTINCT FROM OLD.step OR NEW.status IS DISTINCT FROM OLD.status THEN
        IF OLD.status = 'running' AND NEW.step <> 'recovered' THEN
            INSERT INTO kernel_bench.stage_durations(job_id, mode, stage, seconds)
            VALUES(OLD.id, OLD.mode, OLD.step, EXTRACT(EPOCH FROM now() - OLD.stage_started_at))
            ON CONFLICT(job_id, stage) DO UPDATE SET seconds = kernel_bench.stage_durations.seconds + EXCLUDED.seconds;
        END IF;
        NEW.stage_started_at := now();
    END IF;
    RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS job_revision ON kernel_bench.jobs;
CREATE TRIGGER job_revision BEFORE UPDATE ON kernel_bench.jobs
FOR EACH ROW EXECUTE FUNCTION kernel_bench.reviseJob();
CREATE INDEX IF NOT EXISTS jobs_polling ON kernel_bench.jobs(goal_id, created_at DESC, id DESC);
CREATE TABLE IF NOT EXISTS kernel_bench.workers (
    lane text PRIMARY KEY,
    heartbeat_at timestamptz NOT NULL DEFAULT now(),
    job_id uuid,
    stage text NOT NULL DEFAULT 'idle'
);
CREATE TABLE IF NOT EXISTS kernel_bench.stage_durations (
    job_id uuid NOT NULL REFERENCES kernel_bench.jobs(id) ON DELETE CASCADE,
    mode text NOT NULL,
    stage text NOT NULL,
    seconds double precision NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY(job_id, stage)
);
CREATE TABLE IF NOT EXISTS kernel_bench.baselines (
    configuration_id bigint PRIMARY KEY REFERENCES kernel_bench.configurations(id) ON DELETE CASCADE,
    status text NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'running', 'measured', 'failed')),
    measurements jsonb NOT NULL DEFAULT '[]',
    winner jsonb,
    identity jsonb,
    error text,
    measured_at timestamptz
);

CREATE INDEX IF NOT EXISTS stage_durations_recent ON kernel_bench.stage_durations(created_at DESC);

ALTER TABLE kernel_bench.baselines ADD COLUMN IF NOT EXISTS requested boolean NOT NULL DEFAULT false;
ALTER TABLE kernel_bench.baselines DROP CONSTRAINT IF EXISTS baselines_status_check;
ALTER TABLE kernel_bench.baselines ADD CONSTRAINT baselines_status_check
    CHECK(status IN ('pending', 'running', 'measured', 'failed', 'unavailable'));

-- Stable workload identity is separate from source/build discovery snapshots.
CREATE TABLE IF NOT EXISTS kernel_bench.goal_definitions (
    id text PRIMARY KEY,
    definition jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE kernel_bench.goals ADD COLUMN IF NOT EXISTS definition_id text REFERENCES kernel_bench.goal_definitions(id);
ALTER TABLE kernel_bench.jobs DROP CONSTRAINT IF EXISTS jobs_mode_check;
ALTER TABLE kernel_bench.jobs ADD CONSTRAINT jobs_mode_check CHECK (mode IN ('lint', 'compile', 'evaluate', 'promote', 'goal_setup'));
ALTER TABLE kernel_bench.jobs ADD COLUMN IF NOT EXISTS parent_job_id uuid REFERENCES kernel_bench.jobs(id);
CREATE UNIQUE INDEX IF NOT EXISTS promotion_parent ON kernel_bench.jobs(parent_job_id) WHERE mode = 'promote';
CREATE TABLE IF NOT EXISTS kernel_bench.goal_measurements (
    job_id uuid PRIMARY KEY REFERENCES kernel_bench.jobs(id),
    definition_id text NOT NULL REFERENCES kernel_bench.goal_definitions(id),
    baseline jsonb NOT NULL,
    candidate jsonb NOT NULL,
    speedup double precision NOT NULL CHECK (speedup > 0),
    selected_count integer NOT NULL CHECK (selected_count >= 0),
    accepted boolean NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS goal_validation_once ON kernel_bench.jobs(goal_id)
    WHERE mode = 'promote' AND parent_job_id IS NULL;

CREATE TABLE IF NOT EXISTS kernel_bench.active_goal (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    goal_id bigint NOT NULL REFERENCES kernel_bench.goals(id),
    updated_at timestamptz NOT NULL DEFAULT now()
);
