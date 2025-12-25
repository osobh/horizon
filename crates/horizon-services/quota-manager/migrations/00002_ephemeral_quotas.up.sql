-- Ephemeral Quotas Schema
-- Time-bounded quotas for external collaborators with sponsor billing

-- Ephemeral quotas provide time-limited resource access
-- linked to ephemeral identities from hpc-ephemeral-identity
CREATE TABLE ephemeral_quotas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Optional link to base quota definition
    quota_id UUID REFERENCES quotas(id) ON DELETE SET NULL,
    -- The ephemeral identity this quota belongs to
    ephemeral_identity_id UUID NOT NULL,
    -- Organization/tenant context
    tenant_id UUID NOT NULL,
    -- Who is sponsoring/paying for this quota
    sponsor_id VARCHAR(255) NOT NULL,
    -- Who is using this quota (ephemeral user identifier)
    beneficiary_id VARCHAR(255) NOT NULL,
    -- Type of resource being allocated
    resource_type TEXT NOT NULL,
    -- Total limit for this ephemeral period
    limit_value DECIMAL(20, 6) NOT NULL CHECK (limit_value >= 0),
    -- Amount currently used
    used_value DECIMAL(20, 6) NOT NULL DEFAULT 0 CHECK (used_value >= 0),
    -- Amount reserved for pending operations
    reserved_value DECIMAL(20, 6) NOT NULL DEFAULT 0 CHECK (reserved_value >= 0),
    -- When this quota becomes active
    starts_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- When this quota expires
    expires_at TIMESTAMPTZ NOT NULL,
    -- Time window ID for scheduling constraints
    time_window_id UUID,
    -- Whether burst is enabled beyond the limit
    burst_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    -- Multiplier for burst capacity (e.g., 1.5x = 150%)
    burst_multiplier DECIMAL(10, 4) NOT NULL DEFAULT 1.0 CHECK (burst_multiplier >= 1.0),
    -- Resource pool this quota draws from
    pool_id UUID,
    -- Total cost incurred by this quota
    actual_cost DECIMAL(20, 6) NOT NULL DEFAULT 0 CHECK (actual_cost >= 0),
    -- Cost rate per unit of resource
    cost_rate DECIMAL(20, 6) NOT NULL DEFAULT 0 CHECK (cost_rate >= 0),
    -- Current status
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'suspended', 'expired', 'revoked', 'exhausted')),
    -- Reason for current status (especially for revoked/suspended)
    status_reason TEXT,
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT ephemeral_quota_valid_dates CHECK (expires_at > starts_at),
    CONSTRAINT ephemeral_quota_valid_usage CHECK (used_value + reserved_value <= limit_value * burst_multiplier)
);

-- Indexes for ephemeral quotas
CREATE INDEX idx_ephemeral_quotas_identity ON ephemeral_quotas(ephemeral_identity_id);
CREATE INDEX idx_ephemeral_quotas_sponsor ON ephemeral_quotas(sponsor_id);
CREATE INDEX idx_ephemeral_quotas_beneficiary ON ephemeral_quotas(beneficiary_id);
CREATE INDEX idx_ephemeral_quotas_tenant ON ephemeral_quotas(tenant_id);
CREATE INDEX idx_ephemeral_quotas_expires ON ephemeral_quotas(expires_at) WHERE status = 'active';
CREATE INDEX idx_ephemeral_quotas_status ON ephemeral_quotas(status);
CREATE INDEX idx_ephemeral_quotas_pool ON ephemeral_quotas(pool_id) WHERE pool_id IS NOT NULL;
CREATE INDEX idx_ephemeral_quotas_time_window ON ephemeral_quotas(time_window_id) WHERE time_window_id IS NOT NULL;
CREATE INDEX idx_ephemeral_quotas_active ON ephemeral_quotas(ephemeral_identity_id, status) WHERE status = 'active';

-- Ephemeral quota usage history for auditing
CREATE TABLE ephemeral_quota_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ephemeral_quota_id UUID NOT NULL REFERENCES ephemeral_quotas(id) ON DELETE CASCADE,
    operation TEXT NOT NULL CHECK (operation IN ('reserve', 'commit', 'release', 'burst', 'expire')),
    amount DECIMAL(20, 6) NOT NULL,
    cost DECIMAL(20, 6) NOT NULL DEFAULT 0,
    -- Previous values for audit trail
    previous_used DECIMAL(20, 6) NOT NULL,
    previous_reserved DECIMAL(20, 6) NOT NULL,
    -- Metadata about the operation
    job_id UUID,
    description TEXT,
    metadata JSONB,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ephemeral_quota_usage_quota ON ephemeral_quota_usage(ephemeral_quota_id);
CREATE INDEX idx_ephemeral_quota_usage_timestamp ON ephemeral_quota_usage(timestamp DESC);
CREATE INDEX idx_ephemeral_quota_usage_operation ON ephemeral_quota_usage(operation);

-- Trigger to update updated_at
CREATE OR REPLACE FUNCTION update_ephemeral_quota_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER ephemeral_quotas_updated_at
    BEFORE UPDATE ON ephemeral_quotas
    FOR EACH ROW
    EXECUTE FUNCTION update_ephemeral_quota_updated_at();

-- Function to check if ephemeral quota is within time window
CREATE OR REPLACE FUNCTION is_ephemeral_quota_in_window(quota_id UUID)
RETURNS BOOLEAN AS $$
DECLARE
    quota_record RECORD;
    window_record RECORD;
    current_time TIME;
    current_dow INTEGER;
BEGIN
    SELECT * INTO quota_record FROM ephemeral_quotas WHERE id = quota_id;

    -- If no time window, always allowed
    IF quota_record.time_window_id IS NULL THEN
        RETURN TRUE;
    END IF;

    -- Check time window constraints
    SELECT * INTO window_record FROM time_windows WHERE id = quota_record.time_window_id;

    IF NOT FOUND THEN
        RETURN TRUE; -- Window deleted, allow access
    END IF;

    -- Get current time in window's timezone
    current_time := (NOW() AT TIME ZONE window_record.timezone)::TIME;
    current_dow := EXTRACT(DOW FROM NOW() AT TIME ZONE window_record.timezone)::INTEGER;

    -- Check if current day is allowed
    IF NOT (current_dow = ANY(window_record.days_of_week)) THEN
        RETURN FALSE;
    END IF;

    -- Check if current time is within window
    IF current_time < window_record.start_time OR current_time > window_record.end_time THEN
        RETURN FALSE;
    END IF;

    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;
