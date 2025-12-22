-- Rollback: Multi-Resource Support Migration

-- Step 1: Drop backward compatibility view
DROP VIEW IF EXISTS jobs_legacy_view;

-- Step 2: Restore old trigger function
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

-- Step 3: Replace trigger
DROP TRIGGER IF EXISTS trg_update_resource_usage ON jobs;
CREATE TRIGGER trg_update_user_usage
    AFTER UPDATE ON jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_user_usage();

-- Step 4: Drop new tables
DROP TABLE IF EXISTS resource_usage;
DROP TABLE IF EXISTS job_allocations;

-- Step 5: Remove resource_specs column
ALTER TABLE jobs DROP COLUMN IF EXISTS resource_specs;

-- Step 6: Remove comments from old columns
COMMENT ON COLUMN jobs.gpu_count IS NULL;
COMMENT ON COLUMN jobs.gpu_type IS NULL;
COMMENT ON COLUMN jobs.cpu_cores IS NULL;
COMMENT ON COLUMN jobs.memory_gb IS NULL;

-- Note: Cannot remove 'urgent' from enum type without dropping and recreating
-- This is a one-way change in PostgreSQL
-- If urgent priority was added, it will remain in the enum
