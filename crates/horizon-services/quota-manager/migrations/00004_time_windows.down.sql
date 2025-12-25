-- Rollback time windows schema

DROP TRIGGER IF EXISTS time_windows_updated_at ON time_windows;
DROP FUNCTION IF EXISTS get_next_window_start(UUID, TIMESTAMPTZ);
DROP FUNCTION IF EXISTS is_time_in_window(UUID, TIMESTAMPTZ);
DROP TABLE IF EXISTS time_window_exceptions;
DROP TABLE IF EXISTS time_windows;
