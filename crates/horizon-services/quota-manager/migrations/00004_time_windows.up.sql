-- Time Windows Schema
-- Scheduling constraints for ephemeral access

-- Time windows define when ephemeral access is allowed
CREATE TABLE time_windows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Human-readable name
    name VARCHAR(255) NOT NULL,
    -- Description of the time window
    description TEXT,
    -- Organization/tenant this window belongs to
    tenant_id UUID NOT NULL,
    -- Daily start time (in the specified timezone)
    start_time TIME NOT NULL,
    -- Daily end time (in the specified timezone)
    end_time TIME NOT NULL,
    -- Timezone for interpreting times (IANA timezone name)
    timezone VARCHAR(50) NOT NULL DEFAULT 'UTC',
    -- Days of week when window is active (0=Sunday, 6=Saturday)
    days_of_week INTEGER[] NOT NULL DEFAULT '{0,1,2,3,4,5,6}',
    -- Specific dates when access is blocked regardless of schedule
    blackout_dates DATE[] DEFAULT '{}',
    -- Specific dates when access is allowed regardless of schedule
    override_dates DATE[] DEFAULT '{}',
    -- Whether this window is currently enabled
    is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT time_window_valid_times CHECK (end_time > start_time),
    CONSTRAINT time_window_valid_days CHECK (
        days_of_week <@ ARRAY[0, 1, 2, 3, 4, 5, 6]
    )
);

-- Time window exceptions for specific date overrides
CREATE TABLE time_window_exceptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- The time window this exception applies to
    time_window_id UUID NOT NULL REFERENCES time_windows(id) ON DELETE CASCADE,
    -- The specific date
    exception_date DATE NOT NULL,
    -- Type of exception
    exception_type TEXT NOT NULL CHECK (exception_type IN ('blackout', 'extended', 'reduced', 'override')),
    -- Optional custom hours for this date (overrides regular schedule)
    custom_start_time TIME,
    custom_end_time TIME,
    -- Reason for the exception
    reason TEXT,
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT exception_custom_times CHECK (
        (custom_start_time IS NULL AND custom_end_time IS NULL) OR
        (custom_start_time IS NOT NULL AND custom_end_time IS NOT NULL AND custom_end_time > custom_start_time)
    ),
    -- Unique constraint per window per date
    UNIQUE(time_window_id, exception_date)
);

-- Indexes for time windows
CREATE INDEX idx_time_windows_tenant ON time_windows(tenant_id);
CREATE INDEX idx_time_windows_enabled ON time_windows(is_enabled) WHERE is_enabled = TRUE;
CREATE INDEX idx_time_windows_timezone ON time_windows(timezone);

-- Indexes for time window exceptions
CREATE INDEX idx_time_window_exceptions_window ON time_window_exceptions(time_window_id);
CREATE INDEX idx_time_window_exceptions_date ON time_window_exceptions(exception_date);
CREATE INDEX idx_time_window_exceptions_upcoming ON time_window_exceptions(exception_date)
    WHERE exception_date >= CURRENT_DATE;

-- Trigger to update updated_at
CREATE TRIGGER time_windows_updated_at
    BEFORE UPDATE ON time_windows
    FOR EACH ROW
    EXECUTE FUNCTION update_ephemeral_quota_updated_at();

-- Function to check if current time is within a time window
CREATE OR REPLACE FUNCTION is_time_in_window(window_id UUID, check_time TIMESTAMPTZ DEFAULT NOW())
RETURNS BOOLEAN AS $$
DECLARE
    window_record RECORD;
    exception_record RECORD;
    local_time TIME;
    local_date DATE;
    local_dow INTEGER;
BEGIN
    -- Get the time window
    SELECT * INTO window_record FROM time_windows WHERE id = window_id;

    IF NOT FOUND THEN
        RETURN FALSE;
    END IF;

    -- If window is disabled, deny access
    IF NOT window_record.is_enabled THEN
        RETURN FALSE;
    END IF;

    -- Convert to window's timezone
    local_time := (check_time AT TIME ZONE window_record.timezone)::TIME;
    local_date := (check_time AT TIME ZONE window_record.timezone)::DATE;
    local_dow := EXTRACT(DOW FROM check_time AT TIME ZONE window_record.timezone)::INTEGER;

    -- Check for exceptions on this date
    SELECT * INTO exception_record
    FROM time_window_exceptions
    WHERE time_window_id = window_id AND exception_date = local_date;

    IF FOUND THEN
        CASE exception_record.exception_type
            WHEN 'blackout' THEN
                RETURN FALSE;
            WHEN 'override' THEN
                RETURN TRUE;
            WHEN 'extended', 'reduced' THEN
                -- Use custom times if provided
                IF exception_record.custom_start_time IS NOT NULL THEN
                    RETURN local_time >= exception_record.custom_start_time
                       AND local_time <= exception_record.custom_end_time;
                END IF;
        END CASE;
    END IF;

    -- Check if date is in blackout_dates
    IF local_date = ANY(window_record.blackout_dates) THEN
        RETURN FALSE;
    END IF;

    -- Check if date is in override_dates
    IF local_date = ANY(window_record.override_dates) THEN
        RETURN TRUE;
    END IF;

    -- Check if current day is allowed
    IF NOT (local_dow = ANY(window_record.days_of_week)) THEN
        RETURN FALSE;
    END IF;

    -- Check if current time is within window
    RETURN local_time >= window_record.start_time
       AND local_time <= window_record.end_time;
END;
$$ LANGUAGE plpgsql;

-- Function to get next available time in a window
CREATE OR REPLACE FUNCTION get_next_window_start(window_id UUID, from_time TIMESTAMPTZ DEFAULT NOW())
RETURNS TIMESTAMPTZ AS $$
DECLARE
    window_record RECORD;
    local_time TIME;
    local_date DATE;
    local_dow INTEGER;
    check_date DATE;
    days_checked INTEGER := 0;
    next_dow INTEGER;
BEGIN
    SELECT * INTO window_record FROM time_windows WHERE id = window_id;

    IF NOT FOUND OR NOT window_record.is_enabled THEN
        RETURN NULL;
    END IF;

    -- Start from the given time in window's timezone
    local_time := (from_time AT TIME ZONE window_record.timezone)::TIME;
    local_date := (from_time AT TIME ZONE window_record.timezone)::DATE;
    local_dow := EXTRACT(DOW FROM from_time AT TIME ZONE window_record.timezone)::INTEGER;

    -- Check if we're already in the window today
    IF local_dow = ANY(window_record.days_of_week)
       AND local_time < window_record.end_time
       AND NOT (local_date = ANY(window_record.blackout_dates))
    THEN
        IF local_time >= window_record.start_time THEN
            RETURN from_time; -- Already in window
        ELSE
            -- Return start time today
            RETURN (local_date || ' ' || window_record.start_time)::TIMESTAMP AT TIME ZONE window_record.timezone;
        END IF;
    END IF;

    -- Look for next available day (up to 14 days ahead)
    WHILE days_checked < 14 LOOP
        days_checked := days_checked + 1;
        check_date := local_date + days_checked;
        next_dow := EXTRACT(DOW FROM check_date)::INTEGER;

        IF next_dow = ANY(window_record.days_of_week)
           AND NOT (check_date = ANY(window_record.blackout_dates))
        THEN
            RETURN (check_date || ' ' || window_record.start_time)::TIMESTAMP AT TIME ZONE window_record.timezone;
        END IF;
    END LOOP;

    RETURN NULL; -- No available window in next 14 days
END;
$$ LANGUAGE plpgsql;

-- Common time window presets (inserted as examples, can be removed)
-- INSERT INTO time_windows (name, tenant_id, start_time, end_time, timezone, days_of_week, description)
-- VALUES
--     ('Business Hours PST', '00000000-0000-0000-0000-000000000000', '09:00', '17:00', 'America/Los_Angeles', '{1,2,3,4,5}', 'Monday-Friday 9am-5pm Pacific'),
--     ('Extended Hours UTC', '00000000-0000-0000-0000-000000000000', '06:00', '22:00', 'UTC', '{0,1,2,3,4,5,6}', 'Extended hours every day'),
--     ('Weekdays Only EST', '00000000-0000-0000-0000-000000000000', '08:00', '20:00', 'America/New_York', '{1,2,3,4,5}', 'Weekdays 8am-8pm Eastern');
