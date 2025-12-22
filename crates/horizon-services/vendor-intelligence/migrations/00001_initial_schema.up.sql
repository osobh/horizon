-- ==================================================
-- UNIFIED INTELLIGENCE DATABASE SCHEMA
-- All intelligence services share this database
-- ==================================================

-- MARGIN INTELLIGENCE TABLES
-- Customer profitability profiles
CREATE TABLE IF NOT EXISTS customer_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id VARCHAR(255) UNIQUE NOT NULL,
    segment VARCHAR(50) NOT NULL,
    total_revenue DECIMAL(12,2) NOT NULL DEFAULT 0,
    total_cost DECIMAL(12,2) NOT NULL DEFAULT 0,
    gross_margin DECIMAL(5,2),
    contribution_margin DECIMAL(12,2),
    lifetime_value DECIMAL(12,2),
    first_usage_at TIMESTAMPTZ,
    last_usage_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pricing_simulations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id VARCHAR(255) NOT NULL,
    scenario_name VARCHAR(255) NOT NULL,
    current_price DECIMAL(10,4),
    simulated_price DECIMAL(10,4),
    estimated_margin_impact DECIMAL(10,2),
    elasticity_factor DECIMAL(5,4),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS margin_recommendations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id VARCHAR(255) NOT NULL,
    recommendation_type VARCHAR(50) NOT NULL,
    current_margin DECIMAL(5,2),
    projected_margin DECIMAL(5,2),
    impact_amount DECIMAL(12,2),
    confidence DECIMAL(3,2),
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- EFFICIENCY INTELLIGENCE TABLES
CREATE TABLE IF NOT EXISTS waste_detections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    detection_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(255) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL,
    idle_duration INTERVAL,
    cost_impact_monthly DECIMAL(10,2),
    root_cause TEXT,
    status VARCHAR(20) DEFAULT 'detected',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS efficiency_proposals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    detection_id UUID REFERENCES waste_detections(id),
    proposal_type VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,
    estimated_savings_monthly DECIMAL(10,2),
    roi_months DECIMAL(4,1),
    implementation_cost DECIMAL(10,2) DEFAULT 0,
    status VARCHAR(20) DEFAULT 'draft',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS savings_realized (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    proposal_id UUID REFERENCES efficiency_proposals(id),
    actual_savings_monthly DECIMAL(10,2),
    measured_at TIMESTAMPTZ NOT NULL,
    variance_percent DECIMAL(5,2)
);

-- INITIATIVE TRACKER TABLES
CREATE TABLE IF NOT EXISTS initiatives (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    owner VARCHAR(255) NOT NULL,
    category VARCHAR(50) NOT NULL,
    projected_impact DECIMAL(12,2),
    actual_impact DECIMAL(12,2),
    status VARCHAR(20) DEFAULT 'draft',
    priority VARCHAR(20) DEFAULT 'medium',
    start_date DATE,
    target_date DATE,
    completion_date DATE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS milestones (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    initiative_id UUID REFERENCES initiatives(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    target_date DATE NOT NULL,
    completion_date DATE,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS initiative_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    initiative_id UUID REFERENCES initiatives(id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    baseline_value DECIMAL(12,4),
    target_value DECIMAL(12,4),
    current_value DECIMAL(12,4),
    measured_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- VENDOR INTELLIGENCE TABLES
CREATE TABLE IF NOT EXISTS vendors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    type VARCHAR(50) NOT NULL,
    website VARCHAR(255),
    primary_contact VARCHAR(255),
    email VARCHAR(255),
    performance_score DECIMAL(3,2),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS contracts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_id UUID REFERENCES vendors(id),
    contract_number VARCHAR(100) UNIQUE,
    type VARCHAR(50) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    renewal_date DATE,
    committed_amount DECIMAL(12,2),
    total_value DECIMAL(12,2),
    status VARCHAR(20) DEFAULT 'active',
    auto_renew BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS commitment_utilization (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contract_id UUID REFERENCES contracts(id),
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    committed_hours DECIMAL(10,2),
    utilized_hours DECIMAL(10,2),
    utilization_percent DECIMAL(5,2),
    waste_amount DECIMAL(10,2),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- EXECUTIVE INTELLIGENCE TABLES
CREATE TABLE IF NOT EXISTS executive_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_type VARCHAR(50) NOT NULL,
    report_period DATE NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL,
    content JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS scenarios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    base_assumptions JSONB,
    simulated_assumptions JSONB,
    impact_analysis JSONB,
    created_by VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS benchmarks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_name VARCHAR(100) NOT NULL,
    our_value DECIMAL(12,4),
    industry_median DECIMAL(12,4),
    industry_p75 DECIMAL(12,4),
    industry_p90 DECIMAL(12,4),
    period DATE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ==================================================
-- INDEXES FOR PERFORMANCE
-- ==================================================

-- Margin Intelligence indexes
CREATE INDEX IF NOT EXISTS idx_customer_profiles_segment ON customer_profiles(segment);
CREATE INDEX IF NOT EXISTS idx_customer_profiles_margin ON customer_profiles(gross_margin DESC);
CREATE INDEX IF NOT EXISTS idx_customer_profiles_updated ON customer_profiles(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_pricing_simulations_customer ON pricing_simulations(customer_id);
CREATE INDEX IF NOT EXISTS idx_pricing_simulations_created ON pricing_simulations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_recommendations_customer ON margin_recommendations(customer_id);
CREATE INDEX IF NOT EXISTS idx_recommendations_status ON margin_recommendations(status);

-- Efficiency Intelligence indexes
CREATE INDEX IF NOT EXISTS idx_waste_detections_type ON waste_detections(detection_type);
CREATE INDEX IF NOT EXISTS idx_waste_detections_status ON waste_detections(status);
CREATE INDEX IF NOT EXISTS idx_proposals_roi ON efficiency_proposals(roi_months);

-- Initiative Tracker indexes
CREATE INDEX IF NOT EXISTS idx_initiatives_status ON initiatives(status);
CREATE INDEX IF NOT EXISTS idx_initiatives_category ON initiatives(category);
CREATE INDEX IF NOT EXISTS idx_milestones_status ON milestones(status);

-- Vendor Intelligence indexes
CREATE INDEX IF NOT EXISTS idx_contracts_vendor ON contracts(vendor_id);
CREATE INDEX IF NOT EXISTS idx_contracts_status ON contracts(status);
CREATE INDEX IF NOT EXISTS idx_contracts_renewal ON contracts(renewal_date);

-- Executive Intelligence indexes
CREATE INDEX IF NOT EXISTS idx_reports_type_period ON executive_reports(report_type, report_period DESC);
CREATE INDEX IF NOT EXISTS idx_scenarios_created ON scenarios(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_benchmarks_metric ON benchmarks(metric_name, period DESC);
