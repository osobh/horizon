-- Job states enum
CREATE TYPE job_state AS ENUM (
    'queued',
    'scheduled',
    'running',
    'preempted',
    'completed',
    'failed',
    'cancelled'
);

-- Priority levels
CREATE TYPE priority_level AS ENUM ('low', 'normal', 'high');

-- Jobs table
CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    job_name VARCHAR(255),

    -- Resource requirements
    gpu_count INT NOT NULL CHECK (gpu_count > 0),
    gpu_type VARCHAR(50),
    cpu_cores INT,
    memory_gb INT,

    -- Scheduling
    priority priority_level NOT NULL DEFAULT 'normal',
    state job_state NOT NULL DEFAULT 'queued',

    -- Placement (NULL when queued)
    assigned_node_ids UUID[],
    assigned_gpu_ids UUID[],

    -- Timing
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    scheduled_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- Preemption
    checkpoint_path TEXT,
    preempted_count INT DEFAULT 0,
    last_checkpoint_at TIMESTAMPTZ,
    checkpoint_size_bytes BIGINT,

    -- Fair-share
    user_share DECIMAL(10, 4),

    -- Metadata
    command TEXT,
    working_dir TEXT,
    environment JSONB,
    container_id VARCHAR(255),
    image VARCHAR(255),

    CONSTRAINT valid_timing CHECK (
        submitted_at <= COALESCE(scheduled_at, submitted_at + INTERVAL '100 years') AND
        COALESCE(scheduled_at, submitted_at) <= COALESCE(started_at, submitted_at + INTERVAL '100 years') AND
        COALESCE(started_at, submitted_at) <= COALESCE(completed_at, submitted_at + INTERVAL '100 years')
    )
);

-- Indexes for performance
CREATE INDEX idx_jobs_state ON jobs(state);
CREATE INDEX idx_jobs_user ON jobs(user_id, submitted_at DESC);
CREATE INDEX idx_jobs_priority ON jobs(priority, submitted_at);
CREATE INDEX idx_jobs_queued ON jobs(submitted_at) WHERE state = 'queued';
CREATE INDEX idx_jobs_running ON jobs(started_at) WHERE state = 'running';

-- User usage tracking (for fair-share)
CREATE TABLE user_usage (
    user_id VARCHAR(255) PRIMARY KEY,
    gpu_hours_used DECIMAL(15, 2) DEFAULT 0,
    job_count INT DEFAULT 0,
    last_updated TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_user_usage_updated ON user_usage(last_updated);

-- Job history/events
CREATE TABLE job_events (
    id BIGSERIAL PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL,
    event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    details JSONB
);

CREATE INDEX idx_job_events_job_id ON job_events(job_id, event_time DESC);
CREATE INDEX idx_job_events_type ON job_events(event_type, event_time DESC);

-- Trigger to update user usage when job completes
CREATE OR REPLACE FUNCTION update_user_usage()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.state IN ('completed', 'failed', 'cancelled') AND
       OLD.state = 'running' AND
       NEW.started_at IS NOT NULL THEN

        INSERT INTO user_usage (user_id, gpu_hours_used, job_count, last_updated)
        VALUES (
            NEW.user_id,
            NEW.gpu_count * EXTRACT(EPOCH FROM (COALESCE(NEW.completed_at, NOW()) - NEW.started_at)) / 3600.0,
            1,
            NOW()
        )
        ON CONFLICT (user_id) DO UPDATE SET
            gpu_hours_used = user_usage.gpu_hours_used +
                NEW.gpu_count * EXTRACT(EPOCH FROM (COALESCE(NEW.completed_at, NOW()) - NEW.started_at)) / 3600.0,
            job_count = user_usage.job_count + 1,
            last_updated = NOW();
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_user_usage
    AFTER UPDATE ON jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_user_usage();
