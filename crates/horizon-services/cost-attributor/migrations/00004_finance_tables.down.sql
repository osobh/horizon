-- Rollback Cost Reporter Finance Tables

-- Drop triggers
DROP TRIGGER IF EXISTS update_alert_configurations_updated_at ON alert_configurations;
DROP TRIGGER IF EXISTS update_budgets_updated_at ON budgets;

-- Drop functions
DROP FUNCTION IF EXISTS update_finance_updated_at();

-- Drop tables
DROP TABLE IF EXISTS alert_configurations CASCADE;
DROP TABLE IF EXISTS cost_alerts CASCADE;
DROP TABLE IF EXISTS cost_optimizations CASCADE;
DROP TABLE IF EXISTS chargeback_reports CASCADE;
DROP TABLE IF EXISTS budgets CASCADE;
