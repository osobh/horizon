-- Drop materialized views and function
DROP FUNCTION IF EXISTS refresh_cost_summaries();
DROP MATERIALIZED VIEW IF EXISTS monthly_cost_summary;
DROP MATERIALIZED VIEW IF EXISTS daily_cost_summary;
