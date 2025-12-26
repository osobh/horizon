-- Cost Ingestor: Raw Billing Records Table
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE raw_billing_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider VARCHAR(50) NOT NULL,
    account_id VARCHAR(255),
    service VARCHAR(100),
    resource_id VARCHAR(255),
    usage_start TIMESTAMPTZ NOT NULL,
    usage_end TIMESTAMPTZ NOT NULL,
    amount DECIMAL(12,4) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    raw_data JSONB,
    ingested_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_billing_provider ON raw_billing_records(provider);
CREATE INDEX idx_billing_period ON raw_billing_records(usage_start, usage_end);
CREATE INDEX idx_billing_resource ON raw_billing_records(resource_id);
CREATE INDEX idx_billing_account ON raw_billing_records(account_id);
CREATE INDEX idx_billing_service ON raw_billing_records(service);
CREATE INDEX idx_billing_ingested_at ON raw_billing_records(ingested_at);

-- Add check constraints
ALTER TABLE raw_billing_records
    ADD CONSTRAINT check_amount_non_negative CHECK (amount >= 0);

ALTER TABLE raw_billing_records
    ADD CONSTRAINT check_usage_period CHECK (usage_end > usage_start);
