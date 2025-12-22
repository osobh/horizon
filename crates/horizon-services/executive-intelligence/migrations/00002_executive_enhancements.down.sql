-- Rollback Executive Intelligence Enhancements

-- Remove enhanced columns from initiatives table
ALTER TABLE initiatives DROP COLUMN IF EXISTS expected_roi;
ALTER TABLE initiatives DROP COLUMN IF EXISTS spent;
ALTER TABLE initiatives DROP COLUMN IF EXISTS budget;
ALTER TABLE initiatives DROP COLUMN IF EXISTS progress;

-- Drop indexes
DROP INDEX IF EXISTS idx_investment_recommendations_created;
DROP INDEX IF EXISTS idx_investment_recommendations_roi;
DROP INDEX IF EXISTS idx_investment_recommendations_priority;
DROP INDEX IF EXISTS idx_investment_recommendations_category;
DROP INDEX IF EXISTS idx_strategic_alerts_created;
DROP INDEX IF EXISTS idx_strategic_alerts_resolved;
DROP INDEX IF EXISTS idx_strategic_alerts_category;
DROP INDEX IF EXISTS idx_strategic_alerts_severity;
DROP INDEX IF EXISTS idx_strategic_kpis_last_updated;
DROP INDEX IF EXISTS idx_strategic_kpis_status;
DROP INDEX IF EXISTS idx_strategic_kpis_category;
DROP INDEX IF EXISTS idx_initiatives_progress;

-- Drop tables
DROP TABLE IF EXISTS investment_recommendations CASCADE;
DROP TABLE IF EXISTS strategic_alerts CASCADE;
DROP TABLE IF EXISTS strategic_kpis CASCADE;
