-- Rollback Executive Intelligence Seed Data

-- Remove seed investment recommendations
DELETE FROM investment_recommendations WHERE id IN (
    '50eebc99-9c0b-4ef8-bb6d-6bb9bd380a01',
    '50eebc99-9c0b-4ef8-bb6d-6bb9bd380a02',
    '50eebc99-9c0b-4ef8-bb6d-6bb9bd380a03',
    '50eebc99-9c0b-4ef8-bb6d-6bb9bd380a04',
    '50eebc99-9c0b-4ef8-bb6d-6bb9bd380a05',
    '50eebc99-9c0b-4ef8-bb6d-6bb9bd380a06'
);

-- Remove seed strategic alerts
DELETE FROM strategic_alerts WHERE id IN (
    '40eebc99-9c0b-4ef8-bb6d-6bb9bd380a01',
    '40eebc99-9c0b-4ef8-bb6d-6bb9bd380a02',
    '40eebc99-9c0b-4ef8-bb6d-6bb9bd380a03',
    '40eebc99-9c0b-4ef8-bb6d-6bb9bd380a04',
    '40eebc99-9c0b-4ef8-bb6d-6bb9bd380a05',
    '40eebc99-9c0b-4ef8-bb6d-6bb9bd380a06',
    '40eebc99-9c0b-4ef8-bb6d-6bb9bd380a07'
);

-- Remove seed strategic KPIs
DELETE FROM strategic_kpis WHERE id IN (
    '30eebc99-9c0b-4ef8-bb6d-6bb9bd380a01',
    '30eebc99-9c0b-4ef8-bb6d-6bb9bd380a02',
    '30eebc99-9c0b-4ef8-bb6d-6bb9bd380a03',
    '30eebc99-9c0b-4ef8-bb6d-6bb9bd380a04',
    '30eebc99-9c0b-4ef8-bb6d-6bb9bd380a05',
    '30eebc99-9c0b-4ef8-bb6d-6bb9bd380a06',
    '30eebc99-9c0b-4ef8-bb6d-6bb9bd380a07',
    '30eebc99-9c0b-4ef8-bb6d-6bb9bd380a08',
    '30eebc99-9c0b-4ef8-bb6d-6bb9bd380a09',
    '30eebc99-9c0b-4ef8-bb6d-6bb9bd380a10',
    '30eebc99-9c0b-4ef8-bb6d-6bb9bd380a11',
    '30eebc99-9c0b-4ef8-bb6d-6bb9bd380a12',
    '30eebc99-9c0b-4ef8-bb6d-6bb9bd380a13'
);

-- Remove seed initiatives
DELETE FROM milestones WHERE initiative_id IN (
    '60eebc99-9c0b-4ef8-bb6d-6bb9bd380a01',
    '60eebc99-9c0b-4ef8-bb6d-6bb9bd380a02',
    '60eebc99-9c0b-4ef8-bb6d-6bb9bd380a03'
);

DELETE FROM initiative_metrics WHERE initiative_id IN (
    '60eebc99-9c0b-4ef8-bb6d-6bb9bd380a01',
    '60eebc99-9c0b-4ef8-bb6d-6bb9bd380a02',
    '60eebc99-9c0b-4ef8-bb6d-6bb9bd380a03'
);

DELETE FROM initiatives WHERE id IN (
    '60eebc99-9c0b-4ef8-bb6d-6bb9bd380a01',
    '60eebc99-9c0b-4ef8-bb6d-6bb9bd380a02',
    '60eebc99-9c0b-4ef8-bb6d-6bb9bd380a03'
);

-- Reset enhanced initiative columns (only for initiatives that weren't seed data)
UPDATE initiatives SET
    progress = NULL,
    budget = NULL,
    spent = NULL,
    expected_roi = NULL
WHERE id NOT IN (
    '60eebc99-9c0b-4ef8-bb6d-6bb9bd380a01',
    '60eebc99-9c0b-4ef8-bb6d-6bb9bd380a02',
    '60eebc99-9c0b-4ef8-bb6d-6bb9bd380a03'
);
