-- Rollback Cost Reporter Finance Seed Data

-- Remove seed alert configurations
DELETE FROM alert_configurations WHERE id IN (
    'b0eebc99-9c0b-4ef8-bb6d-6bb9bd380a01',
    'b0eebc99-9c0b-4ef8-bb6d-6bb9bd380a02',
    'b0eebc99-9c0b-4ef8-bb6d-6bb9bd380a03',
    'b0eebc99-9c0b-4ef8-bb6d-6bb9bd380a04',
    'b0eebc99-9c0b-4ef8-bb6d-6bb9bd380a05',
    'b0eebc99-9c0b-4ef8-bb6d-6bb9bd380a06'
);

-- Remove seed cost alerts
DELETE FROM cost_alerts WHERE id IN (
    'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a01',
    'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a02',
    'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a03',
    'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a04',
    'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a05',
    'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a06'
);

-- Remove seed cost optimizations
DELETE FROM cost_optimizations WHERE id IN (
    '90eebc99-9c0b-4ef8-bb6d-6bb9bd380a01',
    '90eebc99-9c0b-4ef8-bb6d-6bb9bd380a02',
    '90eebc99-9c0b-4ef8-bb6d-6bb9bd380a03',
    '90eebc99-9c0b-4ef8-bb6d-6bb9bd380a04',
    '90eebc99-9c0b-4ef8-bb6d-6bb9bd380a05',
    '90eebc99-9c0b-4ef8-bb6d-6bb9bd380a06'
);

-- Remove seed chargeback reports
DELETE FROM chargeback_reports WHERE id IN (
    '80eebc99-9c0b-4ef8-bb6d-6bb9bd380a01',
    '80eebc99-9c0b-4ef8-bb6d-6bb9bd380a02',
    '80eebc99-9c0b-4ef8-bb6d-6bb9bd380a03',
    '80eebc99-9c0b-4ef8-bb6d-6bb9bd380a04',
    '80eebc99-9c0b-4ef8-bb6d-6bb9bd380a05'
);

-- Remove seed budgets
DELETE FROM budgets WHERE id IN (
    '70eebc99-9c0b-4ef8-bb6d-6bb9bd380a01',
    '70eebc99-9c0b-4ef8-bb6d-6bb9bd380a02',
    '70eebc99-9c0b-4ef8-bb6d-6bb9bd380a03',
    '70eebc99-9c0b-4ef8-bb6d-6bb9bd380a04',
    '70eebc99-9c0b-4ef8-bb6d-6bb9bd380a05',
    '70eebc99-9c0b-4ef8-bb6d-6bb9bd380a06',
    '70eebc99-9c0b-4ef8-bb6d-6bb9bd380a07',
    '70eebc99-9c0b-4ef8-bb6d-6bb9bd380a08',
    '70eebc99-9c0b-4ef8-bb6d-6bb9bd380a09',
    '70eebc99-9c0b-4ef8-bb6d-6bb9bd380a10'
);
