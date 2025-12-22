-- Drop trigger and function
DROP TRIGGER IF EXISTS trg_update_user_usage ON jobs;
DROP FUNCTION IF EXISTS update_user_usage();

-- Drop tables
DROP TABLE IF EXISTS job_events;
DROP TABLE IF EXISTS user_usage;
DROP TABLE IF EXISTS jobs;

-- Drop types
DROP TYPE IF EXISTS priority_level;
DROP TYPE IF EXISTS job_state;
