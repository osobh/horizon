-- Cost Attributor: Cost Attribution and GPU Pricing Tables
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- GPU Pricing Table
CREATE TABLE gpu_pricing (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    gpu_type VARCHAR(50) NOT NULL,
    region VARCHAR(50),
    pricing_model VARCHAR(20) NOT NULL,  -- on_demand, spot, reserved
    hourly_rate DECIMAL(8,4) NOT NULL,
    effective_start TIMESTAMPTZ NOT NULL,
    effective_end TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_gpu_pricing_type ON gpu_pricing(gpu_type, pricing_model);
CREATE INDEX idx_gpu_pricing_effective ON gpu_pricing(effective_start, effective_end);
CREATE INDEX idx_gpu_pricing_region ON gpu_pricing(region);

-- Cost Attributions Table
CREATE TABLE cost_attributions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID,  -- Optional: references scheduler jobs
    user_id VARCHAR(255) NOT NULL,
    team_id VARCHAR(255),
    customer_id VARCHAR(255),
    gpu_cost DECIMAL(10,4) DEFAULT 0,
    cpu_cost DECIMAL(10,4) DEFAULT 0,
    network_cost DECIMAL(10,4) DEFAULT 0,
    storage_cost DECIMAL(10,4) DEFAULT 0,
    total_cost DECIMAL(10,4) NOT NULL,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_cost_attr_job ON cost_attributions(job_id);
CREATE INDEX idx_cost_attr_user ON cost_attributions(user_id);
CREATE INDEX idx_cost_attr_team ON cost_attributions(team_id);
CREATE INDEX idx_cost_attr_customer ON cost_attributions(customer_id);
CREATE INDEX idx_cost_attr_period ON cost_attributions(period_start, period_end);
CREATE INDEX idx_cost_attr_created ON cost_attributions(created_at);

-- Add check constraints
ALTER TABLE gpu_pricing
    ADD CONSTRAINT check_hourly_rate_positive CHECK (hourly_rate > 0);

ALTER TABLE gpu_pricing
    ADD CONSTRAINT check_pricing_model CHECK (pricing_model IN ('on_demand', 'spot', 'reserved'));

ALTER TABLE cost_attributions
    ADD CONSTRAINT check_costs_non_negative CHECK (
        gpu_cost >= 0 AND
        cpu_cost >= 0 AND
        network_cost >= 0 AND
        storage_cost >= 0 AND
        total_cost >= 0
    );

ALTER TABLE cost_attributions
    ADD CONSTRAINT check_period_valid CHECK (period_end > period_start);

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for cost_attributions
CREATE TRIGGER update_cost_attributions_updated_at
    BEFORE UPDATE ON cost_attributions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
