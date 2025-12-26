-- Cost Reporter Finance Seed Data
-- Populates finance tables with realistic sample data for testing

-- ==================== Budgets ====================

INSERT INTO budgets (id, team_id, team_name, amount, spent, period, alert_threshold, created_at) VALUES
-- Current month budgets (active monitoring)
('70eebc99-9c0b-4ef8-bb6d-6bb9bd380a01', 'team-research', 'ML Research Team', 50000.00, 42500.50, '2024-10', 80.0, NOW() - INTERVAL '1 month'),
('70eebc99-9c0b-4ef8-bb6d-6bb9bd380a02', 'team-product', 'Product Engineering', 35000.00, 28750.25, '2024-10', 85.0, NOW() - INTERVAL '1 month'),
('70eebc99-9c0b-4ef8-bb6d-6bb9bd380a03', 'team-data-sci', 'Data Science', 28000.00, 32500.00, '2024-10', 75.0, NOW() - INTERVAL '1 month'), -- Over budget!
('70eebc99-9c0b-4ef8-bb6d-6bb9bd380a04', 'team-infra', 'Infrastructure', 45000.00, 23400.75, '2024-10', 90.0, NOW() - INTERVAL '1 month'),
('70eebc99-9c0b-4ef8-bb6d-6bb9bd380a05', 'team-cv', 'Computer Vision', 38000.00, 31850.40, '2024-10', 80.0, NOW() - INTERVAL '1 month'),

-- Next month budgets (planning)
('70eebc99-9c0b-4ef8-bb6d-6bb9bd380a06', 'team-research', 'ML Research Team', 55000.00, 0.00, '2024-11', 80.0, NOW()),
('70eebc99-9c0b-4ef8-bb6d-6bb9bd380a07', 'team-product', 'Product Engineering', 40000.00, 0.00, '2024-11', 85.0, NOW()),
('70eebc99-9c0b-4ef8-bb6d-6bb9bd380a08', 'team-data-sci', 'Data Science', 30000.00, 0.00, '2024-11', 75.0, NOW()),

-- Historical budgets
('70eebc99-9c0b-4ef8-bb6d-6bb9bd380a09', 'team-research', 'ML Research Team', 48000.00, 46200.00, '2024-09', 80.0, NOW() - INTERVAL '2 months'),
('70eebc99-9c0b-4ef8-bb6d-6bb9bd380a10', 'team-product', 'Product Engineering', 32000.00, 29850.00, '2024-09', 85.0, NOW() - INTERVAL '2 months');

-- ==================== Chargeback Reports ====================

INSERT INTO chargeback_reports (id, team_id, team_name, start_date, end_date, total_amount, line_items, status, generated_at) VALUES
-- Approved and paid reports (September)
('80eebc99-9c0b-4ef8-bb6d-6bb9bd380a01', 'team-research', 'ML Research Team', '2024-09-01', '2024-09-30', 46200.00,
 '[
    {"description": "H100 GPU Hours", "resource_type": "compute", "quantity": 2400.0, "unit_price": 12.50, "total": 30000.00},
    {"description": "A100 GPU Hours", "resource_type": "compute", "quantity": 1200.0, "unit_price": 8.00, "total": 9600.00},
    {"description": "Object Storage", "resource_type": "storage", "quantity": 5000.0, "unit_price": 0.023, "total": 115.00},
    {"description": "Block Storage", "resource_type": "storage", "quantity": 12000.0, "unit_price": 0.10, "total": 1200.00},
    {"description": "Network Egress", "resource_type": "network", "quantity": 8500.0, "unit_price": 0.087, "total": 739.50},
    {"description": "Data Transfer", "resource_type": "network", "quantity": 15000.0, "unit_price": 0.05, "total": 750.00},
    {"description": "Support Services", "resource_type": "support", "quantity": 1.0, "unit_price": 3795.50, "total": 3795.50}
 ]'::jsonb,
 'paid', NOW() - INTERVAL '3 weeks'),

('80eebc99-9c0b-4ef8-bb6d-6bb9bd380a02', 'team-product', 'Product Engineering', '2024-09-01', '2024-09-30', 29850.00,
 '[
    {"description": "A100 GPU Hours", "resource_type": "compute", "quantity": 2800.0, "unit_price": 8.00, "total": 22400.00},
    {"description": "Object Storage", "resource_type": "storage", "quantity": 3500.0, "unit_price": 0.023, "total": 80.50},
    {"description": "Block Storage", "resource_type": "storage", "quantity": 8000.0, "unit_price": 0.10, "total": 800.00},
    {"description": "Network Egress", "resource_type": "network", "quantity": 6200.0, "unit_price": 0.087, "total": 539.40},
    {"description": "Support Services", "resource_type": "support", "quantity": 1.0, "unit_price": 6030.10, "total": 6030.10}
 ]'::jsonb,
 'paid', NOW() - INTERVAL '3 weeks'),

-- Pending approval (October partial)
('80eebc99-9c0b-4ef8-bb6d-6bb9bd380a03', 'team-data-sci', 'Data Science', '2024-10-01', '2024-10-15', 16250.00,
 '[
    {"description": "H100 GPU Hours (Mid-month)", "resource_type": "compute", "quantity": 980.0, "unit_price": 12.50, "total": 12250.00},
    {"description": "A100 GPU Hours (Mid-month)", "resource_type": "compute", "quantity": 450.0, "unit_price": 8.00, "total": 3600.00},
    {"description": "Storage (Mid-month)", "resource_type": "storage", "quantity": 4000.0, "unit_price": 0.10, "total": 400.00}
 ]'::jsonb,
 'pending', NOW() - INTERVAL '2 days'),

-- Draft reports (current period)
('80eebc99-9c0b-4ef8-bb6d-6bb9bd380a04', 'team-research', 'ML Research Team', '2024-10-01', '2024-10-31', 42500.50,
 '[
    {"description": "H100 GPU Hours", "resource_type": "compute", "quantity": 2650.0, "unit_price": 12.50, "total": 33125.00},
    {"description": "A100 GPU Hours", "resource_type": "compute", "quantity": 850.0, "unit_price": 8.00, "total": 6800.00},
    {"description": "Storage", "resource_type": "storage", "quantity": 18500.0, "unit_price": 0.10, "total": 1850.00},
    {"description": "Network", "resource_type": "network", "quantity": 8300.0, "unit_price": 0.087, "total": 722.10},
    {"description": "Other", "resource_type": "support", "quantity": 1.0, "unit_price": 3.40, "total": 3.40}
 ]'::jsonb,
 'draft', NOW() - INTERVAL '1 hour'),

('80eebc99-9c0b-4ef8-bb6d-6bb9bd380a05', 'team-cv', 'Computer Vision', '2024-10-01', '2024-10-31', 31850.40,
 '[
    {"description": "H100 GPU Hours", "resource_type": "compute", "quantity": 2200.0, "unit_price": 12.50, "total": 27500.00},
    {"description": "Storage", "resource_type": "storage", "quantity": 15000.0, "unit_price": 0.10, "total": 1500.00},
    {"description": "Network", "resource_type": "network", "quantity": 6200.0, "unit_price": 0.087, "total": 539.40},
    {"description": "Support", "resource_type": "support", "quantity": 1.0, "unit_price": 2311.00, "total": 2311.00}
 ]'::jsonb,
 'draft', NOW() - INTERVAL '30 minutes');

-- ==================== Cost Optimizations ====================

INSERT INTO cost_optimizations (id, title, description, category, potential_savings, effort, priority, status, implementation_steps, created_at) VALUES
-- Identified optimizations (high value)
('90eebc99-9c0b-4ef8-bb6d-6bb9bd380a01', 'Implement GPU Auto-Scaling',
 'Current GPU allocation is static. Implementing auto-scaling based on queue depth and job patterns can reduce idle time by 15-20% while maintaining performance.',
 'compute', 8500.00, 'medium', 'high', 'identified',
 '[
    "Analyze historical GPU usage patterns and job queue metrics",
    "Define scaling policies based on queue depth and utilization thresholds",
    "Implement auto-scaling rules in orchestrator",
    "Test scaling behavior with production workload simulation",
    "Monitor and tune scaling parameters for optimal performance"
 ]'::jsonb,
 NOW() - INTERVAL '1 week'),

('90eebc99-9c0b-4ef8-bb6d-6bb9bd380a02', 'Optimize Storage Lifecycle',
 'Analysis shows 35% of object storage contains data not accessed in 90+ days. Implementing tiered storage with automatic archival can reduce costs.',
 'storage', 4200.00, 'low', 'medium', 'identified',
 '[
    "Audit storage usage and identify cold data patterns",
    "Configure lifecycle policies for automatic tiering",
    "Set up archival to glacier storage for 180+ day old data",
    "Implement retrieval SLAs for archived data",
    "Monitor cost savings and access patterns"
 ]'::jsonb,
 NOW() - INTERVAL '5 days'),

-- In progress
('90eebc99-9c0b-4ef8-bb6d-6bb9bd380a03', 'Right-Size Instance Types',
 'Several teams are using GPU instances larger than their workload requires. Analysis shows 12 instances can be downsized to save costs.',
 'compute', 6800.00, 'medium', 'high', 'in_progress',
 '[
    "✓ Complete resource utilization analysis across all teams",
    "✓ Identify underutilized instances and recommend alternatives",
    "→ Coordinate with teams to validate downsize recommendations",
    "Schedule migrations during maintenance windows",
    "Verify performance after migration"
 ]'::jsonb,
 NOW() - INTERVAL '2 weeks'),

-- Implemented (success stories)
('90eebc99-9c0b-4ef8-bb6d-6bb9bd380a04', 'Implement Network Cost Optimization',
 'Configured VPC peering and private endpoints to reduce data egress charges by routing traffic through internal networks.',
 'network', 3200.00, 'low', 'medium', 'implemented',
 '[
    "✓ Analyze network traffic patterns and egress costs",
    "✓ Set up VPC peering between regions",
    "✓ Configure private endpoints for S3 and ECR",
    "✓ Update application configurations to use internal endpoints",
    "✓ Validate cost reduction in billing reports"
 ]'::jsonb,
 NOW() - INTERVAL '6 weeks'),

-- Rejected
('90eebc99-9c0b-4ef8-bb6d-6bb9bd380a05', 'Migrate to Spot Instances',
 'Use spot instances for non-critical batch jobs to reduce compute costs by up to 70%.',
 'compute', 15000.00, 'high', 'critical', 'rejected',
 '[]'::jsonb,
 NOW() - INTERVAL '3 weeks'),

-- Additional opportunities
('90eebc99-9c0b-4ef8-bb6d-6bb9bd380a06', 'Reserved Instance Purchase',
 'Analysis shows consistent baseline usage of 40 H100 GPUs. Purchasing 1-year reserved instances can save 30% compared to on-demand.',
 'compute', 12500.00, 'low', 'high', 'identified',
 '[
    "Analyze 6-month usage trends to confirm baseline",
    "Calculate ROI for 1-year vs 3-year reserved instances",
    "Get budget approval for upfront commitment",
    "Purchase reserved instances",
    "Monitor utilization to ensure ROI targets are met"
 ]'::jsonb,
 NOW() - INTERVAL '3 days');

-- Update rejection reason for rejected optimization
UPDATE cost_optimizations SET
    rejected_at = NOW() - INTERVAL '2 weeks',
    rejection_reason = 'After evaluation, spot instance interruptions would negatively impact job completion rates and user experience. The 70% cost savings are offset by 25% job failure rate increase and engineering overhead for retry logic. Team recommends focusing on auto-scaling instead.'
WHERE id = '90eebc99-9c0b-4ef8-bb6d-6bb9bd380a05';

UPDATE cost_optimizations SET
    implemented_at = NOW() - INTERVAL '4 weeks'
WHERE status = 'implemented';

-- ==================== Cost Alerts ====================

INSERT INTO cost_alerts (id, title, description, severity, team_id, team_name, threshold_amount, current_amount, active, acknowledged, acknowledged_by, acknowledged_at, resolved, resolved_at, resolution, created_at) VALUES
-- Active critical alerts
('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a01', 'Data Science Team Over Budget',
 'The Data Science team has exceeded their monthly budget by 16%. Current spending: $32,500 vs budget: $28,000.',
 'critical', 'team-data-sci', 'Data Science', 28000.00, 32500.00,
 true, true, 'manager@example.com', NOW() - INTERVAL '1 day',
 false, NULL, NULL,
 NOW() - INTERVAL '2 days'),

('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a02', 'ML Research Approaching Budget Limit',
 'ML Research team has used 85% of monthly budget ($42,500 / $50,000) with 5 days remaining in the period.',
 'high', 'team-research', 'ML Research Team', 50000.00, 42500.50,
 true, true, 'lead-research@example.com', NOW() - INTERVAL '3 hours',
 false, NULL, NULL,
 NOW() - INTERVAL '1 day'),

-- Active medium alerts
('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a03', 'Product Team Budget Alert',
 'Product Engineering has reached 82% of monthly budget ($28,750 / $35,000).',
 'medium', 'team-product', 'Product Engineering', 35000.00, 28750.25,
 true, false, NULL, NULL,
 false, NULL, NULL,
 NOW() - INTERVAL '6 hours'),

('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a04', 'Computer Vision Budget Alert',
 'Computer Vision team has used 83.8% of monthly budget ($31,850 / $38,000).',
 'medium', 'team-cv', 'Computer Vision', 38000.00, 31850.40,
 true, false, NULL, NULL,
 false, NULL, NULL,
 NOW() - INTERVAL '4 hours'),

-- Resolved alerts (historical)
('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a05', 'Infrastructure Unexpected Cost Spike',
 'Infrastructure team experienced 40% cost increase week-over-week due to misconfigured storage replication.',
 'critical', 'team-infra', 'Infrastructure', 10000.00, 14000.00,
 false, true, 'sre@example.com', NOW() - INTERVAL '2 weeks',
 true, NOW() - INTERVAL '1 week',
 'Issue resolved by correcting storage replication configuration. Replication was set to 3x instead of 2x. Excess costs: $4,000. Configuration has been corrected and monitoring added to prevent recurrence.',
 NOW() - INTERVAL '3 weeks'),

('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a06', 'Product Team Budget Exceeded (Sept)',
 'Product team exceeded September budget by 8% due to increased testing activity for new release.',
 'high', 'team-product', 'Product Engineering', 32000.00, 34560.00,
 false, true, 'product-manager@example.com', NOW() - INTERVAL '3 weeks',
 true, NOW() - INTERVAL '2 weeks',
 'Budget overrun was planned and approved for Q3 release testing. Additional budget allocated for October to cover ongoing validation work.',
 NOW() - INTERVAL '4 weeks');

-- ==================== Alert Configurations ====================

INSERT INTO alert_configurations (id, name, description, condition_type, threshold, enabled, notification_channels, created_at) VALUES
('b0eebc99-9c0b-4ef8-bb6d-6bb9bd380a01', 'Budget 80% Threshold',
 'Alert when any team reaches 80% of their monthly budget',
 'budget_exceeded', 80.0, true,
 '["email:finance-team@example.com", "slack:#finance-alerts"]'::jsonb,
 NOW() - INTERVAL '3 months'),

('b0eebc99-9c0b-4ef8-bb6d-6bb9bd380a02', 'Budget 90% Threshold',
 'High priority alert when any team reaches 90% of their monthly budget',
 'budget_exceeded', 90.0, true,
 '["email:finance-team@example.com", "email:cfo@example.com", "slack:#finance-alerts", "pagerduty:budget-alerts"]'::jsonb,
 NOW() - INTERVAL '3 months'),

('b0eebc99-9c0b-4ef8-bb6d-6bb9bd380a03', 'Budget 100% Threshold',
 'Critical alert when any team exceeds their monthly budget',
 'budget_exceeded', 100.0, true,
 '["email:finance-team@example.com", "email:cfo@example.com", "email:ceo@example.com", "slack:#finance-alerts", "pagerduty:budget-critical"]'::jsonb,
 NOW() - INTERVAL '3 months'),

('b0eebc99-9c0b-4ef8-bb6d-6bb9bd380a04', 'Cost Spike Detection',
 'Alert when week-over-week costs increase by more than 25%',
 'cost_spike', 25.0, true,
 '["email:finance-team@example.com", "slack:#cost-anomalies"]'::jsonb,
 NOW() - INTERVAL '2 months'),

('b0eebc99-9c0b-4ef8-bb6d-6bb9bd380a05', 'Low GPU Utilization',
 'Alert when GPU utilization drops below 60% for 3+ hours',
 'utilization_low', 60.0, true,
 '["email:sre-team@example.com", "slack:#efficiency-alerts"]'::jsonb,
 NOW() - INTERVAL '1 month'),

('b0eebc99-9c0b-4ef8-bb6d-6bb9bd380a06', 'Cost Anomaly Detection',
 'ML-based anomaly detection for unusual cost patterns',
 'cost_anomaly', 2.5, true,
 '["email:finance-team@example.com", "slack:#cost-anomalies"]'::jsonb,
 NOW() - INTERVAL '1 month');

-- ==================== Comments ====================

COMMENT ON TABLE budgets IS 'Team budgets with comprehensive tracking and alert thresholds';
COMMENT ON TABLE chargeback_reports IS 'Detailed chargeback reports with realistic line items and approval workflow';
COMMENT ON TABLE cost_optimizations IS 'Cost optimization opportunities with implementation tracking';
COMMENT ON TABLE cost_alerts IS 'Budget and cost alerts with full acknowledgment and resolution workflow';
COMMENT ON TABLE alert_configurations IS 'Reusable alert configurations for proactive cost monitoring';
