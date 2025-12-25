-- Rollback resource pools schema

DROP TRIGGER IF EXISTS pool_allocations_updated_at ON pool_allocations;
DROP TRIGGER IF EXISTS resource_pools_updated_at ON resource_pools;
DROP FUNCTION IF EXISTS get_pool_availability(UUID);
DROP FUNCTION IF EXISTS check_pool_auto_approve(UUID, VARCHAR);
DROP TABLE IF EXISTS pool_allocations;
DROP TABLE IF EXISTS resource_pools;
