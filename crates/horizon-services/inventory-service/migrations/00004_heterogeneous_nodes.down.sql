-- Rollback: Remove heterogeneous node support

-- Drop function
DROP FUNCTION IF EXISTS refresh_uptime_stats();

-- Drop materialized view
DROP MATERIALIZED VIEW IF EXISTS node_uptime_stats;

-- Drop uptime tracking table
DROP TABLE IF EXISTS node_uptime_history;

-- Drop indexes
DROP INDEX IF EXISTS idx_assets_availability_schedule;
DROP INDEX IF EXISTS idx_assets_power_profile;
DROP INDEX IF EXISTS idx_assets_has_battery;
DROP INDEX IF EXISTS idx_assets_device_type;

-- Remove columns from assets
ALTER TABLE assets DROP COLUMN IF EXISTS availability_schedule;
ALTER TABLE assets DROP COLUMN IF EXISTS power_profile;
ALTER TABLE assets DROP COLUMN IF EXISTS has_battery;
ALTER TABLE assets DROP COLUMN IF EXISTS device_type;

-- Drop enum
DROP TYPE IF EXISTS device_type;
