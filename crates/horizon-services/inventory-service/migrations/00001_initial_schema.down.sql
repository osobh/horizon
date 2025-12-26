-- Rollback initial schema

-- Drop triggers
DROP TRIGGER IF EXISTS assets_log_changes ON assets;
DROP TRIGGER IF EXISTS assets_updated_at ON assets;

-- Drop functions
DROP FUNCTION IF EXISTS log_asset_change();
DROP FUNCTION IF EXISTS update_updated_at();

-- Drop tables (in reverse dependency order)
DROP TABLE IF EXISTS asset_reservations;
DROP TABLE IF EXISTS asset_metrics;
DROP TABLE IF EXISTS asset_history;
DROP TABLE IF EXISTS assets;

-- Drop enums
DROP TYPE IF EXISTS change_operation;
DROP TYPE IF EXISTS provider_type;
DROP TYPE IF EXISTS asset_status;
DROP TYPE IF EXISTS asset_type;
