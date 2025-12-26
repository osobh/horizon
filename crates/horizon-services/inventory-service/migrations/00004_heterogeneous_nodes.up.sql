-- Migration: Add heterogeneous node support
-- Extends inventory schema to track device types (server, desktop, laptop, Raspberry Pi)
-- and device-specific attributes (battery, power profile, availability schedule)

-- Add device_type enum
CREATE TYPE device_type AS ENUM (
    'server',
    'desktop',
    'laptop',
    'raspberry_pi'
);

-- Add new columns to assets table
ALTER TABLE assets ADD COLUMN device_type device_type;
ALTER TABLE assets ADD COLUMN has_battery BOOLEAN DEFAULT FALSE;
ALTER TABLE assets ADD COLUMN power_profile VARCHAR(50); -- 'performance', 'balanced', 'power_saver'
ALTER TABLE assets ADD COLUMN availability_schedule JSONB; -- When device is typically online

-- Add indexes for new columns
CREATE INDEX idx_assets_device_type ON assets(device_type) WHERE device_type IS NOT NULL;
CREATE INDEX idx_assets_has_battery ON assets(has_battery) WHERE has_battery = TRUE;
CREATE INDEX idx_assets_power_profile ON assets(power_profile) WHERE power_profile IS NOT NULL;

-- GIN index for availability_schedule queries
CREATE INDEX idx_assets_availability_schedule ON assets USING GIN (availability_schedule jsonb_path_ops)
    WHERE availability_schedule IS NOT NULL;

-- Create uptime tracking table for intermittent nodes (laptops, desktops)
CREATE TABLE node_uptime_history (
    id BIGSERIAL PRIMARY KEY,
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,

    -- Uptime window
    online_at TIMESTAMPTZ NOT NULL,
    offline_at TIMESTAMPTZ,

    -- Duration in minutes (computed when offline_at is set)
    duration_minutes INTEGER,

    -- Battery metrics at the time (for laptops)
    battery_percent SMALLINT,
    charging BOOLEAN,

    -- Thermal state at the time
    thermal_state VARCHAR(20), -- 'normal', 'warm', 'hot', 'critical'

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_uptime_window CHECK (offline_at IS NULL OR offline_at > online_at),
    CONSTRAINT valid_battery CHECK (battery_percent IS NULL OR battery_percent BETWEEN 0 AND 100)
);

-- Indexes for uptime queries
CREATE INDEX idx_uptime_asset ON node_uptime_history(asset_id, online_at DESC);
CREATE INDEX idx_uptime_window ON node_uptime_history(online_at, offline_at);
CREATE INDEX idx_uptime_thermal ON node_uptime_history(thermal_state) WHERE thermal_state IS NOT NULL;

-- Create materialized view for uptime statistics
CREATE MATERIALIZED VIEW node_uptime_stats AS
SELECT
    asset_id,
    COUNT(*) as total_sessions,
    AVG(duration_minutes) as avg_session_duration_minutes,
    SUM(duration_minutes) as total_uptime_minutes,

    -- Calculate uptime percentage over last 30 days
    (SUM(duration_minutes) / (30.0 * 24.0 * 60.0) * 100.0) as uptime_percent_30d,

    -- Reliability score (0.0-1.0)
    LEAST(
        (SUM(duration_minutes) / (30.0 * 24.0 * 60.0)),
        1.0
    ) as reliability_score,

    -- Last online time
    MAX(online_at) as last_online_at,

    -- Typical online hours (array of hour-of-day when online most often)
    ARRAY_AGG(DISTINCT EXTRACT(HOUR FROM online_at)::INTEGER ORDER BY EXTRACT(HOUR FROM online_at)::INTEGER) as typical_online_hours,

    NOW() as updated_at
FROM node_uptime_history
WHERE
    duration_minutes IS NOT NULL AND
    online_at >= NOW() - INTERVAL '30 days'
GROUP BY asset_id;

-- Create unique index for concurrent refresh
CREATE UNIQUE INDEX idx_uptime_stats_asset ON node_uptime_stats(asset_id);

-- Create function to refresh uptime stats
CREATE OR REPLACE FUNCTION refresh_uptime_stats()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY node_uptime_stats;
END;
$$ LANGUAGE plpgsql;

-- Comment the tables
COMMENT ON TABLE node_uptime_history IS 'Tracks historical uptime windows for intermittent nodes (laptops, desktops)';
COMMENT ON MATERIALIZED VIEW node_uptime_stats IS 'Aggregated uptime statistics over last 30 days for reliability scoring';
COMMENT ON COLUMN assets.device_type IS 'Type of device: server, desktop, laptop, raspberry_pi';
COMMENT ON COLUMN assets.has_battery IS 'Whether device has battery (indicates laptop)';
COMMENT ON COLUMN assets.power_profile IS 'Power profile: performance, balanced, power_saver';
COMMENT ON COLUMN assets.availability_schedule IS 'JSONB: expected availability schedule (e.g., work hours for desktops)';
