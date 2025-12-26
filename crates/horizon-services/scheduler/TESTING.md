# Horizon GPU Scheduler - Testing Guide

## Quick Test Commands

### Run All Tests (86 tests)

```bash
# Unit tests (43 tests) - ~0.01s
cargo test --lib

# API Integration tests (23 tests) - ~1.5s
cargo test --test api_job_tests -- --test-threads=1
cargo test --test api_validation_tests -- --test-threads=1
cargo test --test api_queue_tests -- --test-threads=1
cargo test --test api_concurrency_tests -- --test-threads=1
cargo test --test api_health_tests -- --test-threads=1

# End-to-End tests (20 tests) - ~1.5s
cargo test --test end_to_end_tests -- --test-threads=1
```

### Run Performance Benchmarks

```bash
cargo bench
```

### Quick Verification (One-liner)

```bash
# Run all tests sequentially
cargo test --lib && \
cargo test --test api_job_tests -- --test-threads=1 && \
cargo test --test api_validation_tests -- --test-threads=1 && \
cargo test --test api_queue_tests -- --test-threads=1 && \
cargo test --test api_concurrency_tests -- --test-threads=1 && \
cargo test --test api_health_tests -- --test-threads=1 && \
cargo test --test end_to_end_tests -- --test-threads=1
```

## Prerequisites

### Database Setup

Integration and E2E tests require PostgreSQL:

```bash
# Default connection string (override with DATABASE_URL env var)
postgres://postgres:postgres@localhost:5433/scheduler_test
```

**Docker Setup (Recommended):**

```bash
docker run -d \
  --name scheduler-test-db \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=scheduler_test \
  -p 5433:5432 \
  postgres:16-alpine
```

### Environment Variables

```bash
# Optional: Override database URL
export DATABASE_URL="postgres://user:pass@localhost:5433/scheduler_test"

# Optional: Set log level for tests
export RUST_LOG=debug

# Optional: Show backtrace on test failures
export RUST_BACKTRACE=1
```

## Test Organization

### Unit Tests (43 tests)
- **Location:** Inline in `src/**/*.rs` files
- **Scope:** Individual functions, modules, logic
- **Dependencies:** None (pure Rust)
- **Speed:** Very fast (~0.01s total)

### API Integration Tests (23 tests)
- **Location:** `tests/api_*.rs` files
- **Scope:** HTTP API endpoints, request/response handling
- **Dependencies:** PostgreSQL database
- **Speed:** Fast (~1.5s total)
- **Note:** Must run with `--test-threads=1`

### End-to-End Tests (20 tests)
- **Location:** `tests/end_to_end_tests.rs`
- **Scope:** Complete workflows, multi-component interactions
- **Dependencies:** PostgreSQL database, full scheduler stack
- **Speed:** Moderate (~1.5s total)
- **Note:** Must run with `--test-threads=1`

## Test Categories

### By Functionality

```bash
# Job operations
cargo test job -- --test-threads=1

# Queue operations
cargo test queue -- --test-threads=1

# Priority handling
cargo test priority

# Concurrency
cargo test concurrent -- --test-threads=1

# State transitions
cargo test state

# Error handling
cargo test error
cargo test invalid
```

### By Type

```bash
# All unit tests
cargo test --lib

# All integration tests
cargo test --tests -- --test-threads=1

# Specific test file
cargo test --test end_to_end_tests -- --test-threads=1

# Specific test by name
cargo test test_complete_job_lifecycle -- --test-threads=1
```

## Benchmarks

### Run All Benchmarks

```bash
cargo bench
```

### Run Specific Benchmark Group

```bash
# Queue operations only
cargo bench priority_queue

# Job creation only
cargo bench job_creation

# Scheduler operations only
cargo bench scheduler_operations

# Concurrent operations only
cargo bench concurrent_operations
```

### Benchmark Output

Results are saved to:
- `target/criterion/` - Detailed reports
- Console output shows timing comparisons

## Continuous Integration

### Pre-commit Checks

```bash
# Format check
cargo fmt -- --check

# Linting
cargo clippy -- -D warnings

# Tests
cargo test --lib
cargo test --tests -- --test-threads=1

# Build
cargo build --release
```

### Full CI Pipeline

```bash
#!/bin/bash
set -e

# Formatting
cargo fmt -- --check

# Linting
cargo clippy -- -D warnings

# Unit tests
cargo test --lib

# Integration tests
for test in api_job_tests api_validation_tests api_queue_tests api_concurrency_tests api_health_tests end_to_end_tests; do
    cargo test --test $test -- --test-threads=1
done

# Benchmarks (optional, long-running)
# cargo bench

# Release build
cargo build --release

echo "✓ All checks passed!"
```

## Troubleshooting

### Database Connection Issues

```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Check connection manually
psql postgres://postgres:postgres@localhost:5433/scheduler_test -c "SELECT 1"

# Reset test database
psql postgres://postgres:postgres@localhost:5433/postgres -c "DROP DATABASE IF EXISTS scheduler_test; CREATE DATABASE scheduler_test;"
```

### Test Failures

```bash
# Run with verbose output
cargo test --test end_to_end_tests -- --test-threads=1 --nocapture

# Run with backtrace
RUST_BACKTRACE=1 cargo test --test end_to_end_tests -- --test-threads=1

# Run single failing test
cargo test test_name -- --test-threads=1 --exact --nocapture
```

### Performance Issues

```bash
# Check if database is slow
time psql $DATABASE_URL -c "SELECT COUNT(*) FROM jobs"

# Run benchmarks to establish baseline
cargo bench

# Profile tests (requires flamegraph)
cargo test --test end_to_end_tests --profile release -- --test-threads=1
```

## Test Development

### Writing New Tests

```rust
// Unit test (in src file)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_something() {
        // Arrange
        let input = 42;

        // Act
        let result = do_something(input);

        // Assert
        assert_eq!(result, expected);
    }
}

// Integration test (in tests/ directory)
mod common;
use common::test_app::TestApp;

#[tokio::test]
async fn test_feature() {
    let test_app = TestApp::new().await;

    // Test code here

    test_app.cleanup().await;
}
```

### Best Practices

1. **Use TestApp:** All integration tests should use `TestApp` fixture
2. **Cleanup:** Always call `test_app.cleanup().await` at end
3. **Serial Execution:** Database tests must use `--test-threads=1`
4. **Descriptive Names:** Use `test_<feature>_<scenario>` naming
5. **Arrange-Act-Assert:** Structure tests clearly
6. **No Mocks:** Use real database and scheduler (TDD principle)

## Performance Targets

### Queue Operations
- Enqueue: < 10 µs ✓ (actual: ~158 ns)
- Dequeue: < 10 µs ✓ (actual: ~1.4 µs)

### Database Operations
- Submit job: < 5 ms ✓ (actual: ~0.96 ms)
- Get job: < 1 ms ✓ (actual: ~45 µs)

### Concurrency
- 100 concurrent jobs: < 10s ✓ (actual: ~1.5s)

## Test Coverage

Current coverage: **86 tests** covering:
- ✓ Core scheduler logic
- ✓ Queue operations (priority, FIFO)
- ✓ Fair-share calculations
- ✓ Job state transitions
- ✓ API endpoints
- ✓ Concurrency scenarios
- ✓ Error handling
- ✓ Edge cases

Target: Maintain > 80 tests with 100% critical path coverage

---

**Last Updated:** 2025-10-06
**Test Count:** 86 (all passing)
**Build Status:** ✓ Passing
