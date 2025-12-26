-- Checkpoint status enum
CREATE TYPE checkpoint_status AS ENUM ('active', 'uploaded', 'deleted');

-- Checkpoints table
CREATE TABLE checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,

    -- Storage
    checkpoint_path TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    storage_type VARCHAR(20) NOT NULL DEFAULT 'local',

    -- Timing
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    uploaded_at TIMESTAMPTZ,
    last_accessed_at TIMESTAMPTZ,

    -- Metadata
    epoch BIGINT,
    step BIGINT,
    progress_pct DECIMAL(5, 2),
    metadata JSONB,

    -- Lifecycle
    status checkpoint_status NOT NULL DEFAULT 'active',

    CONSTRAINT valid_progress CHECK (progress_pct >= 0 AND progress_pct <= 100),
    CONSTRAINT valid_size CHECK (size_bytes >= 0)
);

-- Indexes
CREATE INDEX idx_checkpoints_job_id ON checkpoints(job_id, created_at DESC);
CREATE INDEX idx_checkpoints_created_at ON checkpoints(created_at);
CREATE INDEX idx_checkpoints_status ON checkpoints(status) WHERE status = 'active';
