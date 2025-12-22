-- Executive Intelligence Enhancements
-- Adds tables for strategic KPIs, alerts, and investment recommendations

-- ==================== Strategic KPIs ====================

CREATE TABLE strategic_kpis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    category VARCHAR(50) NOT NULL,
    current_value DECIMAL(12,4) NOT NULL,
    target_value DECIMAL(12,4) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'on_track',
    trend VARCHAR(20) NOT NULL DEFAULT 'stable',
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT strategic_kpis_status_check
        CHECK (status IN ('on_track', 'at_risk', 'critical', 'achieved')),
    CONSTRAINT strategic_kpis_trend_check
        CHECK (trend IN ('up', 'down', 'stable'))
);

CREATE INDEX idx_strategic_kpis_category ON strategic_kpis(category);
CREATE INDEX idx_strategic_kpis_status ON strategic_kpis(status);
CREATE INDEX idx_strategic_kpis_last_updated ON strategic_kpis(last_updated DESC);

-- ==================== Strategic Alerts ====================

CREATE TABLE strategic_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    severity VARCHAR(20) NOT NULL,
    category VARCHAR(50) NOT NULL,
    impact TEXT NOT NULL,
    resolved BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,

    CONSTRAINT strategic_alerts_severity_check
        CHECK (severity IN ('low', 'medium', 'high', 'critical'))
);

CREATE INDEX idx_strategic_alerts_severity ON strategic_alerts(severity);
CREATE INDEX idx_strategic_alerts_category ON strategic_alerts(category);
CREATE INDEX idx_strategic_alerts_resolved ON strategic_alerts(resolved);
CREATE INDEX idx_strategic_alerts_created ON strategic_alerts(created_at DESC);

-- ==================== Investment Recommendations ====================

CREATE TABLE investment_recommendations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    category VARCHAR(50) NOT NULL,
    priority VARCHAR(20) NOT NULL,
    estimated_cost DECIMAL(12,2) NOT NULL,
    expected_roi DECIMAL(5,2) NOT NULL,
    payback_period_months INT NOT NULL,
    rationale TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT investment_recommendations_priority_check
        CHECK (priority IN ('low', 'medium', 'high', 'critical')),
    CONSTRAINT investment_recommendations_cost_check
        CHECK (estimated_cost >= 0),
    CONSTRAINT investment_recommendations_roi_check
        CHECK (expected_roi >= 0),
    CONSTRAINT investment_recommendations_payback_check
        CHECK (payback_period_months > 0)
);

CREATE INDEX idx_investment_recommendations_category ON investment_recommendations(category);
CREATE INDEX idx_investment_recommendations_priority ON investment_recommendations(priority);
CREATE INDEX idx_investment_recommendations_roi ON investment_recommendations(expected_roi DESC);
CREATE INDEX idx_investment_recommendations_created ON investment_recommendations(created_at DESC);

-- ==================== Enhance Initiatives Table ====================

-- Add new columns to existing initiatives table for executive dashboard compatibility
ALTER TABLE initiatives ADD COLUMN IF NOT EXISTS progress DECIMAL(5,2) DEFAULT 0 CHECK (progress >= 0 AND progress <= 100);
ALTER TABLE initiatives ADD COLUMN IF NOT EXISTS budget DECIMAL(12,2) DEFAULT 0;
ALTER TABLE initiatives ADD COLUMN IF NOT EXISTS spent DECIMAL(12,2) DEFAULT 0;
ALTER TABLE initiatives ADD COLUMN IF NOT EXISTS expected_roi DECIMAL(5,2) DEFAULT 0;

-- Rename 'name' to 'title' to match API model (if it doesn't break existing code)
-- ALTER TABLE initiatives RENAME COLUMN name TO title;

-- Add index for progress tracking
CREATE INDEX IF NOT EXISTS idx_initiatives_progress ON initiatives(progress);

-- ==================== Comments ====================

COMMENT ON TABLE strategic_kpis IS 'Strategic key performance indicators tracked at executive level';
COMMENT ON TABLE strategic_alerts IS 'High-level strategic alerts for executive attention';
COMMENT ON TABLE investment_recommendations IS 'Investment recommendations with ROI analysis for executive decision-making';
COMMENT ON COLUMN initiatives.progress IS 'Progress percentage (0-100) of initiative completion';
COMMENT ON COLUMN initiatives.budget IS 'Total budget allocated for the initiative';
COMMENT ON COLUMN initiatives.spent IS 'Amount spent to date on the initiative';
COMMENT ON COLUMN initiatives.expected_roi IS 'Expected return on investment percentage';
