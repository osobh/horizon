-- Initial schema for inventory service
-- Asset types enum
CREATE TYPE asset_type AS ENUM (
    'node',
    'gpu',
    'cpu',
    'nic',
    'switch',
    'rack'
);

-- Asset status enum
CREATE TYPE asset_status AS ENUM (
    'provisioning',
    'available',
    'allocated',
    'maintenance',
    'degraded',
    'decommissioned'
);

-- Provider type enum
CREATE TYPE provider_type AS ENUM (
    'baremetal',
    'aws',
    'gcp',
    'azure',
    'unknown'
);

-- Change operation enum for history
CREATE TYPE change_operation AS ENUM (
    'created',
    'updated',
    'status_changed',
    'metadata_changed',
    'decommissioned'
);

-- Main assets table
CREATE TABLE assets (
    -- Identity
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_type asset_type NOT NULL,
    provider provider_type NOT NULL,
    provider_id VARCHAR(255),

    -- Hierarchy (GPU -> Node, Node -> Rack)
    parent_id UUID REFERENCES assets(id) ON DELETE SET NULL,

    -- Core attributes
    hostname VARCHAR(255),
    status asset_status NOT NULL DEFAULT 'provisioning',
    location VARCHAR(255),

    -- Flexible metadata (provider-specific, GPU capabilities)
    metadata JSONB NOT NULL DEFAULT '{}',

    -- Temporal tracking
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    decommissioned_at TIMESTAMPTZ,

    -- Audit
    created_by VARCHAR(255) NOT NULL,

    -- Constraints
    CONSTRAINT unique_provider_id UNIQUE (provider, provider_id),
    CONSTRAINT parent_child_type_check CHECK (
        (asset_type = 'gpu' AND parent_id IS NOT NULL) OR
        (asset_type = 'cpu' AND parent_id IS NOT NULL) OR
        (asset_type = 'nic' AND parent_id IS NOT NULL) OR
        asset_type IN ('node', 'switch', 'rack')
    )
);

-- Indexes for common queries
CREATE INDEX idx_assets_type_status ON assets(asset_type, status);
CREATE INDEX idx_assets_provider ON assets(provider, provider_id);
CREATE INDEX idx_assets_parent ON assets(parent_id) WHERE parent_id IS NOT NULL;
CREATE INDEX idx_assets_hostname ON assets(hostname) WHERE hostname IS NOT NULL;
CREATE INDEX idx_assets_location ON assets(location);
CREATE INDEX idx_assets_created_at ON assets(created_at);

-- GIN index for JSONB queries (searching by GPU model, etc.)
CREATE INDEX idx_assets_metadata ON assets USING GIN (metadata jsonb_path_ops);

-- Asset history table for temporal changes
CREATE TABLE asset_history (
    id BIGSERIAL PRIMARY KEY,
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    operation change_operation NOT NULL,

    -- State before change (NULL for 'created')
    previous_status asset_status,
    previous_metadata JSONB,

    -- State after change
    new_status asset_status,
    new_metadata JSONB,

    -- Delta (for efficient queries)
    metadata_delta JSONB,

    -- Context
    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    changed_by VARCHAR(255) NOT NULL,
    reason TEXT,

    CONSTRAINT history_ordering CHECK (id > 0)
);

-- Indexes for history queries
CREATE INDEX idx_asset_history_asset ON asset_history(asset_id, changed_at DESC);
CREATE INDEX idx_asset_history_operation ON asset_history(operation, changed_at DESC);
CREATE INDEX idx_asset_history_changed_at ON asset_history(changed_at DESC);

-- Asset metrics table (recent utilization snapshot)
CREATE TABLE asset_metrics (
    asset_id UUID PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,

    -- GPU Metrics
    gpu_utilization_percent SMALLINT,
    gpu_memory_used_gb NUMERIC(10, 2),
    gpu_memory_total_gb NUMERIC(10, 2),
    gpu_temperature_celsius SMALLINT,
    gpu_power_watts NUMERIC(10, 2),

    -- CPU Metrics
    cpu_utilization_percent SMALLINT,
    cpu_temperature_celsius SMALLINT,

    -- NIC Metrics
    nic_rx_bandwidth_gbps NUMERIC(10, 2),
    nic_tx_bandwidth_gbps NUMERIC(10, 2),

    -- Timestamp
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Check constraints for valid ranges
    CONSTRAINT valid_gpu_utilization CHECK (gpu_utilization_percent BETWEEN 0 AND 100),
    CONSTRAINT valid_cpu_utilization CHECK (cpu_utilization_percent BETWEEN 0 AND 100),
    CONSTRAINT valid_temperature CHECK (
        (gpu_temperature_celsius IS NULL OR gpu_temperature_celsius BETWEEN 0 AND 120) AND
        (cpu_temperature_celsius IS NULL OR cpu_temperature_celsius BETWEEN 0 AND 120)
    )
);

-- Index for finding underutilized resources
CREATE INDEX idx_asset_metrics_gpu_util ON asset_metrics(gpu_utilization_percent)
    WHERE gpu_utilization_percent IS NOT NULL;

-- Asset reservations table (future allocations)
CREATE TABLE asset_reservations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,

    -- Reservation details
    job_id UUID NOT NULL,
    reserved_by VARCHAR(255) NOT NULL,

    -- Time window
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,

    -- Status
    active BOOLEAN NOT NULL DEFAULT TRUE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT valid_time_window CHECK (end_time > start_time)
);

-- Indexes for reservations
CREATE INDEX idx_reservations_asset ON asset_reservations(asset_id, start_time);
CREATE INDEX idx_reservations_time_window ON asset_reservations(start_time, end_time);
CREATE INDEX idx_reservations_job ON asset_reservations(job_id);

-- Function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update updated_at on assets
CREATE TRIGGER assets_updated_at
    BEFORE UPDATE ON assets
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Function to automatically log asset changes to history
CREATE OR REPLACE FUNCTION log_asset_change()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO asset_history (asset_id, operation, new_status, new_metadata, changed_by)
        VALUES (NEW.id, 'created', NEW.status, NEW.metadata, NEW.created_by);
    ELSIF TG_OP = 'UPDATE' THEN
        -- Log only if status or metadata changed
        IF OLD.status != NEW.status OR OLD.metadata != NEW.metadata THEN
            INSERT INTO asset_history (
                asset_id,
                operation,
                previous_status,
                previous_metadata,
                new_status,
                new_metadata,
                metadata_delta,
                changed_by
            ) VALUES (
                NEW.id,
                CASE
                    WHEN OLD.status != NEW.status THEN 'status_changed'
                    ELSE 'metadata_changed'
                END,
                OLD.status,
                OLD.metadata,
                NEW.status,
                NEW.metadata,
                -- Compute delta (simplified - only new fields)
                NEW.metadata,
                'system'
            );
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to log changes to assets
CREATE TRIGGER assets_log_changes
    AFTER INSERT OR UPDATE ON assets
    FOR EACH ROW
    EXECUTE FUNCTION log_asset_change();
