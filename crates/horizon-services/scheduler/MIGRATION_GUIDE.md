# Scheduler Database Migration Guide

## Overview
This migration transforms the scheduler from a GPU-only model to a universal multi-resource model.

## What Changed

### Database Schema
- **Added**: `resource_specs` JSONB column for flexible resource specifications
- **Added**: `urgent` priority level to `priority_level` enum
- **Added**: `job_allocations` table for tracking resource assignments
- **Added**: `resource_usage` table for multi-resource fair-share tracking
- **Deprecated**: `gpu_count`, `gpu_type`, `cpu_cores`, `memory_gb` columns (kept for backward compatibility)

### Migration Files
- **Up**: `migrations/00003_multi_resource_support.up.sql`
- **Down**: `migrations/00003_multi_resource_support.down.sql`

## Migration Steps

### 1. Backup Your Database
```bash
pg_dump -h localhost -U postgres -d scheduler > scheduler_backup_$(date +%Y%m%d).sql
```

### 2. Run the Migration
```bash
cd services/scheduler
sqlx migrate run --database-url "postgres://user:pass@localhost/scheduler"
```

### 3. Verify Migration
```sql
-- Check that resource_specs column exists
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'jobs' AND column_name = 'resource_specs';

-- Check that urgent priority was added
SELECT enumlabel FROM pg_enum
WHERE enumtypid = 'priority_level'::regtype;

-- Check that new tables exist
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
AND table_name IN ('job_allocations', 'resource_usage');
```

### 4. Switch to New Repository
Update your scheduler initialization:

```rust
// OLD (deprecated, still works)
use crate::db::JobRepository;
let repo = JobRepository::new(pool);

// NEW (recommended)
use crate::db::JobRepositoryV2;
let repo = JobRepositoryV2::new(pool);
```

## Data Migration

The migration automatically converts existing data:

### Before (GPU-only)
```sql
gpu_count: 4
gpu_type: "H100"
cpu_cores: 64
memory_gb: 512
```

### After (Multi-resource JSONB)
```json
{
  "resources": {
    "Compute": {
      "Gpu": {
        "amount": 4,
        "unit": "Count",
        "constraints": {
          "model": "H100",
          "vendor": "Nvidia"
        }
      }
    },
    "ComputeCpu": {
      "amount": 64,
      "unit": "Cores"
    },
    "Memory": {
      "amount": 512,
      "unit": "Gigabytes"
    }
  },
  "priority": "normal",
  "requester_id": "user123"
}
```

## Backward Compatibility

### Old Columns (Deprecated)
The old columns (`gpu_count`, `gpu_type`, `cpu_cores`, `memory_gb`) are kept for backward compatibility:
- They continue to be populated during INSERT operations (via `JobRepository`)
- They are marked with database comments indicating deprecation
- Can be safely removed after transition period

### Legacy View
A `jobs_legacy_view` is provided for applications that still query old columns:
```sql
SELECT gpu_count, gpu_type, cpu_cores, memory_gb
FROM jobs_legacy_view
WHERE id = $1;
```

### Dual Repository Pattern
Both repositories coexist during transition:
- `JobRepository` - Uses old columns, maintains compatibility
- `JobRepositoryV2` - Uses JSONB, recommended for new code

## Rollback

If issues occur, rollback the migration:

```bash
sqlx migrate revert --database-url "postgres://user:pass@localhost/scheduler"
```

**Note**: The `urgent` priority enum value **cannot** be removed by rollback due to PostgreSQL limitations. It will remain in the enum but won't be used.

## Testing

### Test Data Integrity
```sql
-- Compare old and new representations
SELECT
    id,
    gpu_count AS old_gpu_count,
    (resource_specs -> 'resources' -> 'Compute' -> 'Gpu' ->> 'amount')::INT AS new_gpu_count
FROM jobs
WHERE gpu_count != COALESCE((resource_specs -> 'resources' -> 'Compute' -> 'Gpu' ->> 'amount')::INT, 0);
```

### Test Resource Usage Tracking
```sql
-- Check multi-resource usage is being tracked
SELECT entity_id, resource_type, usage_amount
FROM resource_usage
WHERE entity_type = 'user'
ORDER BY usage_amount DESC
LIMIT 10;
```

## Performance Considerations

### JSONB Indexing
For queries on specific resource types, consider adding GIN indexes:

```sql
-- Index for GPU-related queries
CREATE INDEX idx_jobs_resource_gpu
ON jobs USING GIN ((resource_specs -> 'resources' -> 'Compute' -> 'Gpu'));

-- Index for CPU-related queries
CREATE INDEX idx_jobs_resource_cpu
ON jobs USING GIN ((resource_specs -> 'resources' -> 'ComputeCpu'));
```

### Query Patterns
```sql
-- Efficient: Query using GIN index
SELECT * FROM jobs
WHERE resource_specs -> 'resources' -> 'Compute' -> 'Gpu' IS NOT NULL;

-- Less efficient: Extracting values in WHERE clause
SELECT * FROM jobs
WHERE (resource_specs -> 'resources' -> 'Compute' -> 'Gpu' ->> 'amount')::INT > 4;
```

## Transition Timeline

### Phase 1: Migration (Week 1)
- Run database migration
- Deploy code with both repositories
- Monitor for issues

### Phase 2: Validation (Week 2-3)
- Verify data integrity
- Test new features (CPU-only jobs, TPU support)
- Monitor performance metrics

### Phase 3: Deprecation (Week 4-6)
- Update all code to use `JobRepositoryV2`
- Stop writing to old columns
- Announce deprecation to API consumers

### Phase 4: Cleanup (Week 7+)
- Remove old columns: `gpu_count`, `gpu_type`, `cpu_cores`, `memory_gb`
- Remove `JobRepository` (keep only V2)
- Remove legacy view

## FAQ

### Q: Will existing jobs continue to work?
**A**: Yes. The migration converts all existing data to the new format, and backward compatibility layers ensure old API calls still work.

### Q: Can I submit CPU-only jobs immediately?
**A**: Yes, once the migration completes. Use the new `ResourceRequest` API:
```rust
Job::builder()
    .user_id("user")
    .cpu_cores(128)
    .memory_gb(1024)
    .build()
```

### Q: What happens to fair-share calculations?
**A**: They are enhanced! The old `user_usage` table tracking GPU-hours is preserved, and a new `resource_usage` table tracks CPU-hours, memory-hours, etc. separately.

### Q: How do I query TPU jobs?
**A**: Use JSONB operators:
```sql
SELECT * FROM jobs
WHERE resource_specs -> 'resources' -> 'Compute' -> 'Tpu' IS NOT NULL;
```

### Q: Can I still use the old API endpoints?
**A**: Yes. The REST API maintains backward compatibility. Old requests are automatically converted to the new format internally.

## Monitoring

### Key Metrics to Track
- Query performance on `jobs` table
- Size of `resource_specs` JSONB column
- Fair-share calculation accuracy
- API response times

### Alerts to Configure
- Migration duration > 5 minutes
- Data integrity mismatches
- Failed JSONB deserialization errors

## Support

For issues or questions:
1. Check migration logs: `tail -f /var/log/scheduler/migration.log`
2. Review application logs for serialization errors
3. Contact: scheduler-team@horizon.com

---

**Migration Version**: 00003
**Created**: 2025-10-27
**Status**: Ready for deployment
