-- Migration: Multi-Resource Support
-- Transforms GPU-only schema to universal resource support

-- Step 1: Add 'urgent' to priority_level enum
ALTER TYPE priority_level ADD VALUE 'urgent';

-- Step 2: Add new JSONB column for flexible resource specifications
ALTER TABLE jobs ADD COLUMN resource_specs JSONB;

-- Step 3: Migrate existing data to new format
-- Convert existing gpu_count, gpu_type, cpu_cores, memory_gb to JSONB
UPDATE jobs SET resource_specs = jsonb_build_object(
    'resources', jsonb_build_object(
        'Compute', CASE
            WHEN gpu_count > 0 THEN jsonb_build_object(
                'Gpu', jsonb_build_object(
                    'amount', gpu_count,
                    'unit', 'Count',
                    'constraints', CASE
                        WHEN gpu_type IS NOT NULL THEN jsonb_build_object(
                            'model', gpu_type,
                            'vendor', 'Nvidia'
                        )
                        ELSE '{}'::jsonb
                    END
                )
            )
            ELSE NULL
        END
    ) || CASE
        WHEN cpu_cores IS NOT NULL THEN jsonb_build_object(
            'ComputeCpu', jsonb_build_object(
                'amount', cpu_cores,
                'unit', 'Cores'
            )
        )
        ELSE '{}'::jsonb
    END || CASE
        WHEN memory_gb IS NOT NULL THEN jsonb_build_object(
            'Memory', jsonb_build_object(
                'amount', memory_gb,
                'unit', 'Gigabytes'
            )
        )
        ELSE '{}'::jsonb
    END,
    'priority', priority::text,
    'requester_id', user_id
)
WHERE resource_specs IS NULL;

-- Step 4: Create job_allocations table for tracking actual resource assignments
CREATE TABLE job_allocations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    resource_type VARCHAR(50) NOT NULL,
    amount DECIMAL(15, 4) NOT NULL,
    unit VARCHAR(50) NOT NULL,
    asset_ids UUID[] NOT NULL,
    allocated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    released_at TIMESTAMPTZ
);

CREATE INDEX idx_job_allocations_job_id ON job_allocations(job_id);
CREATE INDEX idx_job_allocations_resource_type ON job_allocations(resource_type);
CREATE INDEX idx_job_allocations_active ON job_allocations(allocated_at) WHERE released_at IS NULL;

-- Step 5: Create resource_usage table for multi-resource fair-share tracking
CREATE TABLE resource_usage (
    id BIGSERIAL PRIMARY KEY,
    entity_id VARCHAR(255) NOT NULL,
    entity_type VARCHAR(50) NOT NULL, -- 'user', 'organization', 'team'
    resource_type VARCHAR(50) NOT NULL,
    usage_amount DECIMAL(15, 4) NOT NULL DEFAULT 0,
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(entity_id, entity_type, resource_type)
);

CREATE INDEX idx_resource_usage_entity ON resource_usage(entity_id, entity_type);
CREATE INDEX idx_resource_usage_type ON resource_usage(resource_type);

-- Step 6: Migrate existing user_usage to resource_usage
INSERT INTO resource_usage (entity_id, entity_type, resource_type, usage_amount, last_updated)
SELECT
    user_id,
    'user',
    'gpu',
    gpu_hours_used,
    last_updated
FROM user_usage
WHERE gpu_hours_used > 0;

-- Step 7: Create updated trigger function for multi-resource usage tracking
CREATE OR REPLACE FUNCTION update_resource_usage()
RETURNS TRIGGER AS $$
DECLARE
    gpu_spec JSONB;
    cpu_spec JSONB;
    gpu_amount DECIMAL;
    cpu_amount DECIMAL;
    duration_hours DECIMAL;
BEGIN
    IF NEW.state IN ('completed', 'failed', 'cancelled') AND
       OLD.state = 'running' AND
       NEW.started_at IS NOT NULL THEN

        -- Calculate duration in hours
        duration_hours := EXTRACT(EPOCH FROM (COALESCE(NEW.completed_at, NOW()) - NEW.started_at)) / 3600.0;

        -- Extract GPU resources
        gpu_spec := NEW.resource_specs -> 'resources' -> 'Compute' -> 'Gpu';
        IF gpu_spec IS NOT NULL THEN
            gpu_amount := (gpu_spec ->> 'amount')::DECIMAL;

            INSERT INTO resource_usage (entity_id, entity_type, resource_type, usage_amount, last_updated)
            VALUES (
                NEW.user_id,
                'user',
                'gpu',
                gpu_amount * duration_hours,
                NOW()
            )
            ON CONFLICT (entity_id, entity_type, resource_type) DO UPDATE SET
                usage_amount = resource_usage.usage_amount + (gpu_amount * duration_hours),
                last_updated = NOW();
        END IF;

        -- Extract CPU resources
        cpu_spec := NEW.resource_specs -> 'resources' -> 'ComputeCpu';
        IF cpu_spec IS NOT NULL THEN
            cpu_amount := (cpu_spec ->> 'amount')::DECIMAL;

            INSERT INTO resource_usage (entity_id, entity_type, resource_type, usage_amount, last_updated)
            VALUES (
                NEW.user_id,
                'user',
                'cpu',
                cpu_amount * duration_hours,
                NOW()
            )
            ON CONFLICT (entity_id, entity_type, resource_type) DO UPDATE SET
                usage_amount = resource_usage.usage_amount + (cpu_amount * duration_hours),
                last_updated = NOW();
        END IF;

        -- Legacy user_usage update for backward compatibility
        INSERT INTO user_usage (user_id, gpu_hours_used, job_count, last_updated)
        VALUES (
            NEW.user_id,
            COALESCE(gpu_amount, 0) * duration_hours,
            1,
            NOW()
        )
        ON CONFLICT (user_id) DO UPDATE SET
            gpu_hours_used = user_usage.gpu_hours_used + (COALESCE(gpu_amount, 0) * duration_hours),
            job_count = user_usage.job_count + 1,
            last_updated = NOW();
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Replace the old trigger
DROP TRIGGER IF EXISTS trg_update_user_usage ON jobs;
CREATE TRIGGER trg_update_resource_usage
    AFTER UPDATE ON jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_resource_usage();

-- Step 8: Add backward compatibility view for old API
CREATE OR REPLACE VIEW jobs_legacy_view AS
SELECT
    id,
    user_id,
    job_name,
    COALESCE(
        (resource_specs -> 'resources' -> 'Compute' -> 'Gpu' ->> 'amount')::INT,
        0
    ) AS gpu_count,
    resource_specs -> 'resources' -> 'Compute' -> 'Gpu' -> 'constraints' ->> 'model' AS gpu_type,
    (resource_specs -> 'resources' -> 'ComputeCpu' ->> 'amount')::INT AS cpu_cores,
    (resource_specs -> 'resources' -> 'Memory' ->> 'amount')::INT AS memory_gb,
    priority,
    state,
    assigned_node_ids,
    assigned_gpu_ids,
    submitted_at,
    scheduled_at,
    started_at,
    completed_at,
    checkpoint_path,
    preempted_count,
    last_checkpoint_at,
    checkpoint_size_bytes,
    user_share,
    command,
    working_dir,
    environment,
    container_id,
    image
FROM jobs;

-- Step 9: Mark old columns as deprecated (keep for backward compatibility during transition)
COMMENT ON COLUMN jobs.gpu_count IS 'DEPRECATED: Use resource_specs JSONB column instead';
COMMENT ON COLUMN jobs.gpu_type IS 'DEPRECATED: Use resource_specs JSONB column instead';
COMMENT ON COLUMN jobs.cpu_cores IS 'DEPRECATED: Use resource_specs JSONB column instead';
COMMENT ON COLUMN jobs.memory_gb IS 'DEPRECATED: Use resource_specs JSONB column instead';

-- Optional: After verification period, remove old columns (commented out for safety)
-- ALTER TABLE jobs DROP COLUMN gpu_count;
-- ALTER TABLE jobs DROP COLUMN gpu_type;
-- ALTER TABLE jobs DROP COLUMN cpu_cores;
-- ALTER TABLE jobs DROP COLUMN memory_gb;
