-- Resource Pools Schema
-- Shared resource pools for bounties, competitions, and managed access

-- Resource pools allow sponsors to pre-allocate resources
-- that can be drawn from by approved users
CREATE TABLE resource_pools (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Human-readable name
    name VARCHAR(255) NOT NULL,
    -- Description of the pool's purpose
    description TEXT,
    -- Organization/tenant this pool belongs to
    tenant_id UUID NOT NULL,
    -- Type of pool
    pool_type TEXT NOT NULL CHECK (pool_type IN ('bounty', 'competition', 'shared', 'reserved', 'burst')),
    -- Type of resource in this pool
    resource_type TEXT NOT NULL,
    -- Total capacity of the pool
    total_limit DECIMAL(20, 6) NOT NULL CHECK (total_limit > 0),
    -- Currently allocated (sum of active allocations)
    allocated DECIMAL(20, 6) NOT NULL DEFAULT 0 CHECK (allocated >= 0),
    -- Reserved for pending requests
    reserved DECIMAL(20, 6) NOT NULL DEFAULT 0 CHECK (reserved >= 0),
    -- Maximum allocation per user
    max_allocation_per_user DECIMAL(20, 6) NOT NULL CHECK (max_allocation_per_user > 0),
    -- Minimum per request
    min_allocation_per_request DECIMAL(20, 6) NOT NULL DEFAULT 1 CHECK (min_allocation_per_request > 0),
    -- Whether allocations require approval
    requires_approval BOOLEAN NOT NULL DEFAULT FALSE,
    -- Domains that get auto-approved (array of email domains)
    auto_approve_domains TEXT[] DEFAULT '{}',
    -- Maximum concurrent users drawing from this pool
    max_concurrent_users INTEGER,
    -- Current number of users with active allocations
    current_users INTEGER NOT NULL DEFAULT 0 CHECK (current_users >= 0),
    -- When the pool becomes active (optional)
    starts_at TIMESTAMPTZ,
    -- When the pool expires (optional)
    expires_at TIMESTAMPTZ,
    -- Current status
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('pending', 'active', 'suspended', 'expired', 'depleted')),
    -- The sponsor funding this pool
    sponsor_id UUID,
    -- Time window constraints (optional)
    time_window_id UUID,
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT pool_capacity_check CHECK (allocated + reserved <= total_limit),
    CONSTRAINT pool_valid_dates CHECK (expires_at IS NULL OR starts_at IS NULL OR expires_at > starts_at),
    CONSTRAINT pool_max_users_check CHECK (max_concurrent_users IS NULL OR current_users <= max_concurrent_users)
);

-- Pool allocations track who is using pool resources
CREATE TABLE pool_allocations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- The pool this allocation draws from
    pool_id UUID NOT NULL REFERENCES resource_pools(id) ON DELETE CASCADE,
    -- The user receiving the allocation
    user_id VARCHAR(255) NOT NULL,
    -- Optional link to ephemeral identity
    ephemeral_identity_id UUID,
    -- Amount allocated
    allocated_amount DECIMAL(20, 6) NOT NULL CHECK (allocated_amount > 0),
    -- Amount used so far
    used_amount DECIMAL(20, 6) NOT NULL DEFAULT 0 CHECK (used_amount >= 0),
    -- Allocation status
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'active', 'completed', 'expired', 'rejected', 'cancelled')),
    -- Purpose/justification for the allocation
    purpose TEXT,
    -- Who approved this allocation (NULL for auto-approved)
    approved_by UUID,
    -- When it was approved
    approved_at TIMESTAMPTZ,
    -- When the allocation was released
    released_at TIMESTAMPTZ,
    -- When this allocation expires
    expires_at TIMESTAMPTZ NOT NULL,
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT allocation_usage_check CHECK (used_amount <= allocated_amount)
);

-- Indexes for resource pools
CREATE INDEX idx_resource_pools_tenant ON resource_pools(tenant_id);
CREATE INDEX idx_resource_pools_type ON resource_pools(pool_type);
CREATE INDEX idx_resource_pools_status ON resource_pools(status);
CREATE INDEX idx_resource_pools_sponsor ON resource_pools(sponsor_id) WHERE sponsor_id IS NOT NULL;
CREATE INDEX idx_resource_pools_active ON resource_pools(status, expires_at) WHERE status = 'active';
CREATE INDEX idx_resource_pools_resource_type ON resource_pools(resource_type);

-- Indexes for pool allocations
CREATE INDEX idx_pool_allocations_pool ON pool_allocations(pool_id);
CREATE INDEX idx_pool_allocations_user ON pool_allocations(user_id);
CREATE INDEX idx_pool_allocations_status ON pool_allocations(status);
CREATE INDEX idx_pool_allocations_ephemeral ON pool_allocations(ephemeral_identity_id) WHERE ephemeral_identity_id IS NOT NULL;
CREATE INDEX idx_pool_allocations_pending ON pool_allocations(pool_id, status) WHERE status = 'pending';
CREATE INDEX idx_pool_allocations_active ON pool_allocations(pool_id, status) WHERE status IN ('approved', 'active');
CREATE INDEX idx_pool_allocations_expires ON pool_allocations(expires_at) WHERE status IN ('approved', 'active');

-- Trigger to update updated_at
CREATE TRIGGER resource_pools_updated_at
    BEFORE UPDATE ON resource_pools
    FOR EACH ROW
    EXECUTE FUNCTION update_ephemeral_quota_updated_at();

CREATE TRIGGER pool_allocations_updated_at
    BEFORE UPDATE ON pool_allocations
    FOR EACH ROW
    EXECUTE FUNCTION update_ephemeral_quota_updated_at();

-- Function to check if allocation is auto-approved
CREATE OR REPLACE FUNCTION check_pool_auto_approve(pool_id UUID, user_email VARCHAR)
RETURNS BOOLEAN AS $$
DECLARE
    pool_record RECORD;
    user_domain VARCHAR;
BEGIN
    SELECT * INTO pool_record FROM resource_pools WHERE id = pool_id;

    IF NOT FOUND THEN
        RETURN FALSE;
    END IF;

    -- If no approval required, auto-approve
    IF NOT pool_record.requires_approval THEN
        RETURN TRUE;
    END IF;

    -- Extract domain from email
    user_domain := substring(user_email from '@(.*)$');

    -- Check if domain is in auto-approve list
    IF user_domain = ANY(pool_record.auto_approve_domains) THEN
        RETURN TRUE;
    END IF;

    RETURN FALSE;
END;
$$ LANGUAGE plpgsql;

-- Function to get pool availability
CREATE OR REPLACE FUNCTION get_pool_availability(pool_id UUID)
RETURNS DECIMAL AS $$
DECLARE
    pool_record RECORD;
BEGIN
    SELECT * INTO pool_record FROM resource_pools WHERE id = pool_id;

    IF NOT FOUND THEN
        RETURN 0;
    END IF;

    RETURN pool_record.total_limit - pool_record.allocated - pool_record.reserved;
END;
$$ LANGUAGE plpgsql;
