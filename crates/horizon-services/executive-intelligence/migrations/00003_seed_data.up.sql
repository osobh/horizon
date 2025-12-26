-- Executive Intelligence Seed Data
-- Populates tables with realistic sample data for executive dashboard

-- ==================== Strategic KPIs ====================

INSERT INTO strategic_kpis (id, name, category, current_value, target_value, unit, status, trend, last_updated) VALUES
-- Efficiency KPIs
('30eebc99-9c0b-4ef8-bb6d-6bb9bd380a01', 'GPU Utilization', 'efficiency', 78.5, 85.0, '%', 'on_track', 'up', NOW() - INTERVAL '1 hour'),
('30eebc99-9c0b-4ef8-bb6d-6bb9bd380a02', 'Job Success Rate', 'efficiency', 96.3, 95.0, '%', 'achieved', 'stable', NOW() - INTERVAL '2 hours'),
('30eebc99-9c0b-4ef8-bb6d-6bb9bd380a03', 'Average Job Duration', 'efficiency', 142.5, 120.0, 'min', 'at_risk', 'up', NOW() - INTERVAL '3 hours'),

-- Financial KPIs
('30eebc99-9c0b-4ef8-bb6d-6bb9bd380a04', 'Gross Margin', 'financial', 52.0, 55.0, '%', 'on_track', 'up', NOW() - INTERVAL '1 day'),
('30eebc99-9c0b-4ef8-bb6d-6bb9bd380a05', 'Revenue Growth', 'financial', 24.8, 25.0, '%', 'on_track', 'up', NOW() - INTERVAL '1 day'),
('30eebc99-9c0b-4ef8-bb6d-6bb9bd380a06', 'Cost per GPU Hour', 'financial', 3.45, 3.20, '$', 'at_risk', 'down', NOW() - INTERVAL '2 days'),
('30eebc99-9c0b-4ef8-bb6d-6bb9bd380a07', 'Customer Acquisition Cost', 'financial', 12500.0, 10000.0, '$', 'at_risk', 'up', NOW() - INTERVAL '1 day'),

-- Customer KPIs
('30eebc99-9c0b-4ef8-bb6d-6bb9bd380a08', 'Active Customers', 'customer', 47.0, 50.0, 'count', 'on_track', 'up', NOW() - INTERVAL '1 hour'),
('30eebc99-9c0b-4ef8-bb6d-6bb9bd380a09', 'Customer Satisfaction', 'customer', 4.6, 4.5, 'stars', 'achieved', 'stable', NOW() - INTERVAL '1 week'),
('30eebc99-9c0b-4ef8-bb6d-6bb9bd380a10', 'Churn Rate', 'customer', 3.2, 2.5, '%', 'at_risk', 'up', NOW() - INTERVAL '2 weeks'),

-- Infrastructure KPIs
('30eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', 'System Availability', 'infrastructure', 99.95, 99.9, '%', 'achieved', 'stable', NOW() - INTERVAL '30 minutes'),
('30eebc99-9c0b-4ef8-bb6d-6bb9bd380a12', 'Mean Time to Recovery', 'infrastructure', 18.5, 15.0, 'min', 'at_risk', 'down', NOW() - INTERVAL '1 day'),
('30eebc99-9c0b-4ef8-bb6d-6bb9bd380a13', 'Cluster Health Score', 'infrastructure', 92.0, 95.0, 'score', 'on_track', 'up', NOW() - INTERVAL '1 hour');

-- ==================== Strategic Alerts ====================

INSERT INTO strategic_alerts (id, title, description, severity, category, impact, resolved, created_at, resolved_at) VALUES
-- Open critical alerts
('40eebc99-9c0b-4ef8-bb6d-6bb9bd380a01', 'Revenue Target at Risk for Q4',
 'Current revenue trajectory is 8% below target for Q4. Primary factors: reduced utilization from top 3 customers and delayed enterprise deals.',
 'critical', 'financial',
 'Projected $420K shortfall in Q4 revenue if current trend continues. May impact year-end financial targets and investor expectations.',
 false, NOW() - INTERVAL '3 days', NULL),

('40eebc99-9c0b-4ef8-bb6d-6bb9bd380a02', 'Major Customer Churn Risk',
 'TechCorp (15% of revenue) has reduced GPU usage by 40% over past 2 weeks. Sales team reports they are evaluating competitive offerings.',
 'critical', 'customer',
 'At risk of losing $375K annual revenue. Customer accounts for significant portion of H100 utilization. Churn would impact Q1 2025 significantly.',
 false, NOW() - INTERVAL '2 days', NULL),

-- High severity alerts
('40eebc99-9c0b-4ef8-bb6d-6bb9bd380a03', 'GPU Utilization Below Target',
 'Overall GPU utilization has been trending below 80% target for 3 consecutive weeks, currently at 78.5%.',
 'high', 'efficiency',
 'Underutilized capacity represents $85K/month in opportunity cost. Affects gross margin and overall profitability.',
 false, NOW() - INTERVAL '1 week', NULL),

('40eebc99-9c0b-4ef8-bb6d-6bb9bd380a04', 'Infrastructure Capacity Constraint',
 'Forecasted demand for Q1 2025 exceeds current H100 capacity by 35%. Current acquisition timeline shows 6-week delivery delay.',
 'high', 'infrastructure',
 'May need to turn away new customers or limit growth of existing accounts. Estimated $250K revenue at risk in Q1.',
 false, NOW() - INTERVAL '5 days', NULL),

-- Medium severity alerts
('40eebc99-9c0b-4ef8-bb6d-6bb9bd380a05', 'Cost Per GPU Hour Increasing',
 'Cost per GPU hour has increased from $3.20 to $3.45 over past month due to higher cloud egress and storage costs.',
 'medium', 'financial',
 'Margin compression of 7.8%. Need to optimize data transfer patterns or pass costs to customers.',
 false, NOW() - INTERVAL '1 week', NULL),

-- Recently resolved
('40eebc99-9c0b-4ef8-bb6d-6bb9bd380a06', 'System Availability SLA Breach',
 'System availability dropped to 99.85% in previous week due to storage subsystem issues, breaching 99.9% SLA.',
 'high', 'infrastructure',
 'SLA breach triggered $12K in service credits. Customer satisfaction impact minimal as issues occurred during off-peak hours.',
 true, NOW() - INTERVAL '2 weeks', NOW() - INTERVAL '1 week'),

('40eebc99-9c0b-4ef8-bb6d-6bb9bd380a07', 'Delayed Enterprise Contract',
 'DataCo enterprise contract (projected $500K ARR) delayed due to security compliance review.',
 'medium', 'customer',
 'Revenue recognition pushed from Q4 to Q1 2025. No long-term impact expected.',
 true, NOW() - INTERVAL '3 weeks', NOW() - INTERVAL '1 week');

-- ==================== Investment Recommendations ====================

INSERT INTO investment_recommendations (id, title, description, category, priority, estimated_cost, expected_roi, payback_period_months, rationale, created_at) VALUES
-- Critical priority
('50eebc99-9c0b-4ef8-bb6d-6bb9bd380a01', 'Expand H100 GPU Capacity',
 'Acquire additional 40 H100 GPUs to meet Q1 2025 forecasted demand and enable new customer acquisition.',
 'infrastructure', 'critical', 1200000.0, 145.0,
 8,
 'Current capacity constraints are limiting growth and forcing us to turn away prospective customers. Market analysis shows strong demand for H100 capacity with premium pricing. Investment will generate $290K/month in additional revenue at 80% utilization, with 8-month payback period. Risk: Demand forecasts may not materialize; Mitigation: Staggered acquisition with customer commitments.',
 NOW() - INTERVAL '1 week'),

('50eebc99-9c0b-4ef8-bb6d-6bb9bd380a02', 'Implement Advanced Job Scheduling',
 'Deploy intelligent job scheduling system to improve GPU utilization and reduce job queue times.',
 'efficiency', 'critical', 180000.0, 220.0,
 6,
 'Current utilization of 78.5% leaves 21.5% capacity unused, representing $85K/month opportunity cost. Advanced scheduling with bin-packing algorithms can increase utilization to 88-92%, generating additional $120K/month revenue. ROI calculation based on conservative 10% utilization improvement. Implementation includes ML-based demand forecasting and dynamic pricing.',
 NOW() - INTERVAL '5 days'),

-- High priority
('50eebc99-9c0b-4ef8-bb6d-6bb9bd380a03', 'Customer Self-Service Portal',
 'Build comprehensive self-service portal for customers to manage resources, view usage analytics, and handle billing.',
 'customer', 'high', 250000.0, 180.0,
 10,
 'Support costs are $45K/month with 60% of tickets being routine requests that could be self-served. Portal will reduce support costs by 40% ($18K/month savings) and improve customer satisfaction. Additional benefits include faster customer onboarding (reducing sales cycle by 2 weeks) and better transparency driving upsells. Portal includes usage dashboards, quota management, and automated billing.',
 NOW() - INTERVAL '3 days'),

('50eebc99-9c0b-4ef8-bb6d-6bb9bd380a04', 'Multi-Region Deployment',
 'Deploy infrastructure in EU and APAC regions to serve international customers and improve latency.',
 'infrastructure', 'high', 2500000.0, 165.0,
 14,
 'International customers represent 30% growth opportunity ($750K ARR) but require local data residency and lower latency. EU deployment addresses $400K pipeline currently blocked by data sovereignty requirements. APAC deployment reduces latency from 280ms to <50ms for key customers. Phased deployment: EU in Q1 2025, APAC in Q2 2025.',
 NOW() - INTERVAL '1 week'),

-- Medium priority
('50eebc99-9c0b-4ef8-bb6d-6bb9bd380a05', 'ML-Powered Cost Optimization',
 'Implement ML models to predict resource usage patterns and automatically optimize infrastructure costs.',
 'financial', 'medium', 120000.0, 195.0,
 7,
 'Current cloud costs are $180K/month with significant waste from over-provisioned storage and inefficient data transfer. ML optimization can reduce costs by 12-15% ($22K/month). System will analyze historical patterns, predict demand, and auto-scale resources. Includes automated data lifecycle management and intelligent caching.',
 NOW() - INTERVAL '2 days'),

('50eebc99-9c0b-4ef8-bb6d-6bb9bd380a06', 'Enhanced Monitoring & Observability',
 'Upgrade monitoring stack with distributed tracing, advanced anomaly detection, and predictive alerting.',
 'infrastructure', 'medium', 95000.0, 175.0,
 9,
 'Current incident detection averages 12 minutes, resulting in monthly cost of $8K in service credits and customer dissatisfaction. Enhanced monitoring will reduce MTTD to <2 minutes and enable predictive alerting before incidents occur. Expected reduction in downtime-related costs of $14K/month. Includes integration with existing ticketing and automated remediation for common issues.',
 NOW() - INTERVAL '4 days');

-- ==================== Enhanced Initiatives Data ====================

-- Update existing initiatives with progress, budget, and ROI data
UPDATE initiatives SET
    progress = 75.0,
    budget = 150000.0,
    spent = 112500.0,
    expected_roi = 180.0
WHERE name LIKE '%Cost Optimization%' OR description LIKE '%optimization%';

UPDATE initiatives SET
    progress = 45.0,
    budget = 80000.0,
    spent = 36000.0,
    expected_roi = 220.0
WHERE name LIKE '%Efficiency%' OR category = 'efficiency';

UPDATE initiatives SET
    progress = 90.0,
    budget = 200000.0,
    spent = 185000.0,
    expected_roi = 165.0
WHERE status = 'completed' OR completion_date IS NOT NULL;

UPDATE initiatives SET
    progress = 30.0,
    budget = 120000.0,
    spent = 32000.0,
    expected_roi = 145.0
WHERE status = 'in_progress' AND priority = 'high';

-- Add some sample executive-focused initiatives
INSERT INTO initiatives (id, name, description, owner, category, status, priority, progress, budget, spent, expected_roi,
                         projected_impact, actual_impact, start_date, target_date, created_at) VALUES
('60eebc99-9c0b-4ef8-bb6d-6bb9bd380a01', 'Q4 Revenue Acceleration Program',
 'Multi-faceted initiative to accelerate Q4 revenue through customer expansion, new logo acquisition, and pricing optimization',
 'alice.cfo@example.com', 'revenue', 'in_progress', 'critical',
 62.0, 350000.0, 215000.0, 285.0,
 950000.0, 580000.0,
 NOW() - INTERVAL '2 months', NOW() + INTERVAL '1 month', NOW() - INTERVAL '2 months'),

('60eebc99-9c0b-4ef8-bb6d-6bb9bd380a02', 'Infrastructure Modernization',
 'Comprehensive upgrade of infrastructure including networking, storage, and orchestration to improve reliability and efficiency',
 'bob.cto@example.com', 'infrastructure', 'in_progress', 'high',
 38.0, 1200000.0, 456000.0, 195.0,
 450000.0, 85000.0,
 NOW() - INTERVAL '3 months', NOW() + INTERVAL '4 months', NOW() - INTERVAL '3 months'),

('60eebc99-9c0b-4ef8-bb6d-6bb9bd380a03', 'Customer Experience Enhancement',
 'Improve customer satisfaction through better tooling, documentation, and support processes',
 'carol.vpcx@example.com', 'customer_success', 'in_progress', 'high',
 55.0, 280000.0, 154000.0, 165.0,
 120000.0, 45000.0,
 NOW() - INTERVAL '4 months', NOW() + INTERVAL '2 months', NOW() - INTERVAL '4 months');

-- ==================== Comments ====================

COMMENT ON TABLE strategic_kpis IS 'Comprehensive strategic KPIs with realistic targets and trend data';
COMMENT ON TABLE strategic_alerts IS 'Strategic alerts covering financial, customer, efficiency, and infrastructure concerns';
COMMENT ON TABLE investment_recommendations IS 'Investment recommendations with detailed ROI analysis and rationale';
