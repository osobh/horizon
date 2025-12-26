-- Quota Manager Schema
-- Hierarchical quota management with optimistic locking

-- Entity types: organization, team, user
-- Resource types: gpu_hours, concurrent_gpus, storage_gb, cpu_hours, memory_gb

CREATE TABLE quotas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type VARCHAR(20) NOT NULL CHECK (entity_type IN ('organization', 'team', 'user')),
    entity_id VARCHAR(255) NOT NULL,
    parent_id UUID REFERENCES quotas(id) ON DELETE CASCADE,
    resource_type VARCHAR(50) NOT NULL,
    limit_value DECIMAL(20, 4) NOT NULL CHECK (limit_value >= 0),
    soft_limit DECIMAL(20, 4) CHECK (soft_limit >= 0 AND soft_limit <= limit_value),
    burst_limit DECIMAL(20, 4) CHECK (burst_limit >= limit_value),
    overcommit_ratio DECIMAL(5, 2) NOT NULL DEFAULT 1.0 CHECK (overcommit_ratio >= 1.0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(entity_type, entity_id, resource_type)
);

-- Allocations track resource usage per job
CREATE TABLE allocations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    quota_id UUID NOT NULL REFERENCES quotas(id) ON DELETE CASCADE,
    job_id UUID NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    allocated_value DECIMAL(20, 4) NOT NULL CHECK (allocated_value > 0),
    allocated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    released_at TIMESTAMPTZ,
    version INTEGER NOT NULL DEFAULT 0,
    metadata JSONB,
    UNIQUE(job_id, resource_type)
);

-- Usage history for auditing and analytics
CREATE TABLE usage_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    quota_id UUID NOT NULL REFERENCES quotas(id) ON DELETE CASCADE,
    entity_type VARCHAR(20) NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    allocated_value DECIMAL(20, 4) NOT NULL,
    operation VARCHAR(20) NOT NULL CHECK (operation IN ('allocate', 'release')),
    job_id UUID,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB
);

-- Indexes for performance
CREATE INDEX idx_quotas_entity ON quotas(entity_type, entity_id);
CREATE INDEX idx_quotas_parent ON quotas(parent_id);
CREATE INDEX idx_quotas_resource ON quotas(resource_type);
CREATE INDEX idx_allocations_quota ON allocations(quota_id) WHERE released_at IS NULL;
CREATE INDEX idx_allocations_job ON allocations(job_id);
CREATE INDEX idx_allocations_active ON allocations(quota_id, released_at) WHERE released_at IS NULL;
CREATE INDEX idx_usage_history_quota ON usage_history(quota_id);
CREATE INDEX idx_usage_history_entity ON usage_history(entity_type, entity_id);
CREATE INDEX idx_usage_history_timestamp ON usage_history(timestamp DESC);
