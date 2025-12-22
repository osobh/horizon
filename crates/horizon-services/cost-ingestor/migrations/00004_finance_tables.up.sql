-- Cost Reporter Finance Tables
-- Adds tables for budgets, chargeback reports, cost optimizations, and alert management

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ==================== Budgets ====================

CREATE TABLE budgets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    team_id VARCHAR(255) NOT NULL,
    team_name VARCHAR(255) NOT NULL,
    amount DECIMAL(12,2) NOT NULL,
    spent DECIMAL(12,2) NOT NULL DEFAULT 0,
    remaining DECIMAL(12,2) GENERATED ALWAYS AS (amount - spent) STORED,
    period VARCHAR(20) NOT NULL,
    alert_threshold DECIMAL(5,2) NOT NULL DEFAULT 80.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT budgets_amount_check CHECK (amount >= 0),
    CONSTRAINT budgets_spent_check CHECK (spent >= 0),
    CONSTRAINT budgets_alert_threshold_check CHECK (alert_threshold >= 0 AND alert_threshold <= 100),
    CONSTRAINT budgets_unique_team_period UNIQUE (team_id, period)
);

CREATE INDEX idx_budgets_team ON budgets(team_id);
CREATE INDEX idx_budgets_period ON budgets(period);
CREATE INDEX idx_budgets_updated ON budgets(updated_at DESC);

-- ==================== Chargeback Reports ====================

CREATE TABLE chargeback_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    team_id VARCHAR(255) NOT NULL,
    team_name VARCHAR(255) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    total_amount DECIMAL(12,2) NOT NULL,
    line_items JSONB NOT NULL DEFAULT '[]'::jsonb,
    status VARCHAR(20) NOT NULL DEFAULT 'draft',
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT chargeback_reports_status_check
        CHECK (status IN ('draft', 'pending', 'approved', 'rejected', 'paid')),
    CONSTRAINT chargeback_reports_amount_check CHECK (total_amount >= 0),
    CONSTRAINT chargeback_reports_date_check CHECK (end_date >= start_date)
);

CREATE INDEX idx_chargeback_reports_team ON chargeback_reports(team_id);
CREATE INDEX idx_chargeback_reports_status ON chargeback_reports(status);
CREATE INDEX idx_chargeback_reports_dates ON chargeback_reports(start_date, end_date);
CREATE INDEX idx_chargeback_reports_generated ON chargeback_reports(generated_at DESC);

-- ==================== Cost Optimizations ====================

CREATE TABLE cost_optimizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    category VARCHAR(50) NOT NULL,
    potential_savings DECIMAL(12,2) NOT NULL,
    effort VARCHAR(20) NOT NULL,
    priority VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'identified',
    implementation_steps JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    implemented_at TIMESTAMPTZ,
    rejected_at TIMESTAMPTZ,
    rejection_reason TEXT,

    CONSTRAINT cost_optimizations_effort_check
        CHECK (effort IN ('low', 'medium', 'high')),
    CONSTRAINT cost_optimizations_priority_check
        CHECK (priority IN ('low', 'medium', 'high', 'critical')),
    CONSTRAINT cost_optimizations_status_check
        CHECK (status IN ('identified', 'in_progress', 'implemented', 'rejected')),
    CONSTRAINT cost_optimizations_savings_check CHECK (potential_savings >= 0)
);

CREATE INDEX idx_cost_optimizations_category ON cost_optimizations(category);
CREATE INDEX idx_cost_optimizations_status ON cost_optimizations(status);
CREATE INDEX idx_cost_optimizations_priority ON cost_optimizations(priority);
CREATE INDEX idx_cost_optimizations_savings ON cost_optimizations(potential_savings DESC);
CREATE INDEX idx_cost_optimizations_created ON cost_optimizations(created_at DESC);

-- ==================== Cost Alerts ====================

CREATE TABLE cost_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    severity VARCHAR(20) NOT NULL,
    team_id VARCHAR(255),
    team_name VARCHAR(255),
    threshold_amount DECIMAL(12,2) NOT NULL,
    current_amount DECIMAL(12,2) NOT NULL,
    active BOOLEAN NOT NULL DEFAULT true,
    acknowledged BOOLEAN NOT NULL DEFAULT false,
    acknowledged_by VARCHAR(255),
    acknowledged_at TIMESTAMPTZ,
    resolved BOOLEAN NOT NULL DEFAULT false,
    resolved_at TIMESTAMPTZ,
    resolution TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT cost_alerts_severity_check
        CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    CONSTRAINT cost_alerts_threshold_check CHECK (threshold_amount >= 0),
    CONSTRAINT cost_alerts_current_check CHECK (current_amount >= 0)
);

CREATE INDEX idx_cost_alerts_severity ON cost_alerts(severity);
CREATE INDEX idx_cost_alerts_team ON cost_alerts(team_id);
CREATE INDEX idx_cost_alerts_active ON cost_alerts(active);
CREATE INDEX idx_cost_alerts_acknowledged ON cost_alerts(acknowledged);
CREATE INDEX idx_cost_alerts_resolved ON cost_alerts(resolved);
CREATE INDEX idx_cost_alerts_created ON cost_alerts(created_at DESC);

-- ==================== Alert Configurations ====================

CREATE TABLE alert_configurations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    condition_type VARCHAR(50) NOT NULL,
    threshold DECIMAL(12,2) NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT true,
    notification_channels JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT alert_configurations_condition_check
        CHECK (condition_type IN ('budget_exceeded', 'cost_spike', 'utilization_low', 'cost_anomaly')),
    CONSTRAINT alert_configurations_threshold_check CHECK (threshold >= 0)
);

CREATE INDEX idx_alert_configurations_condition ON alert_configurations(condition_type);
CREATE INDEX idx_alert_configurations_enabled ON alert_configurations(enabled);

-- ==================== Functions ====================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_finance_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_budgets_updated_at
    BEFORE UPDATE ON budgets
    FOR EACH ROW
    EXECUTE FUNCTION update_finance_updated_at();

CREATE TRIGGER update_alert_configurations_updated_at
    BEFORE UPDATE ON alert_configurations
    FOR EACH ROW
    EXECUTE FUNCTION update_finance_updated_at();

-- ==================== Comments ====================

COMMENT ON TABLE budgets IS 'Team budgets with spending tracking and alert thresholds';
COMMENT ON TABLE chargeback_reports IS 'Detailed chargeback reports for teams with line items';
COMMENT ON TABLE cost_optimizations IS 'Cost optimization recommendations with implementation tracking';
COMMENT ON TABLE cost_alerts IS 'Budget and cost alerts with acknowledgment and resolution workflow';
COMMENT ON TABLE alert_configurations IS 'Reusable alert configurations for automated cost monitoring';
COMMENT ON COLUMN budgets.remaining IS 'Computed column: amount - spent';
COMMENT ON COLUMN budgets.alert_threshold IS 'Percentage threshold (0-100) for triggering budget alerts';
COMMENT ON COLUMN chargeback_reports.line_items IS 'Array of line items with description, resource_type, quantity, unit_price, total';
COMMENT ON COLUMN cost_optimizations.implementation_steps IS 'Array of implementation steps for the optimization';
COMMENT ON COLUMN alert_configurations.notification_channels IS 'Array of notification channels (email, slack, etc.)';
