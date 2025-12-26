-- Cost Reporter: Materialized Views for Performance

-- Daily cost summary materialized view
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_cost_summary AS
SELECT
    date_trunc('day', period_start) as day,
    team_id,
    user_id,
    SUM(total_cost) as total_cost,
    SUM(gpu_cost) as gpu_cost,
    SUM(cpu_cost) as cpu_cost,
    SUM(network_cost) as network_cost,
    SUM(storage_cost) as storage_cost,
    COUNT(DISTINCT job_id) as job_count
FROM cost_attributions
GROUP BY date_trunc('day', period_start), team_id, user_id;

-- Create indices on daily summary
CREATE INDEX IF NOT EXISTS idx_daily_summary_day ON daily_cost_summary(day);
CREATE INDEX IF NOT EXISTS idx_daily_summary_team ON daily_cost_summary(team_id);
CREATE INDEX IF NOT EXISTS idx_daily_summary_user ON daily_cost_summary(user_id);

-- Monthly cost summary materialized view
CREATE MATERIALIZED VIEW IF NOT EXISTS monthly_cost_summary AS
SELECT
    date_trunc('month', period_start) as month,
    team_id,
    user_id,
    customer_id,
    SUM(total_cost) as total_cost,
    SUM(gpu_cost) as gpu_cost,
    SUM(cpu_cost) as cpu_cost,
    SUM(network_cost) as network_cost,
    SUM(storage_cost) as storage_cost,
    COUNT(DISTINCT job_id) as job_count
FROM cost_attributions
GROUP BY date_trunc('month', period_start), team_id, user_id, customer_id;

-- Create indices on monthly summary
CREATE INDEX IF NOT EXISTS idx_monthly_summary_month ON monthly_cost_summary(month);
CREATE INDEX IF NOT EXISTS idx_monthly_summary_team ON monthly_cost_summary(team_id);
CREATE INDEX IF NOT EXISTS idx_monthly_summary_user ON monthly_cost_summary(user_id);
CREATE INDEX IF NOT EXISTS idx_monthly_summary_customer ON monthly_cost_summary(customer_id);

-- Function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_cost_summaries() RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY daily_cost_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY monthly_cost_summary;
END;
$$ LANGUAGE plpgsql;
