-- Rollback ephemeral quotas schema

DROP TRIGGER IF EXISTS ephemeral_quotas_updated_at ON ephemeral_quotas;
DROP FUNCTION IF EXISTS update_ephemeral_quota_updated_at();
DROP FUNCTION IF EXISTS is_ephemeral_quota_in_window(UUID);
DROP TABLE IF EXISTS ephemeral_quota_usage;
DROP TABLE IF EXISTS ephemeral_quotas;
