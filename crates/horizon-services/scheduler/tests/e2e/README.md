# End-to-End Integration Test Suite

This directory contains the comprehensive end-to-end (E2E) and multi-service integration test suite for the Horizon project. These tests verify that all Phase 4 components (API Gateway, Governor, Quota Manager, Scheduler) work correctly together.

## Overview

### Test Categories

1. **Integration Tests** (`integration_tests.rs`) - 10 tests
   - API Gateway ↔ Governor integration (2 tests)
   - API Gateway ↔ Quota Manager integration (3 tests)
   - API Gateway ↔ Scheduler integration (2 tests)
   - Scheduler ↔ Quota Manager integration (3 tests)

2. **End-to-End Flow Tests** (`e2e_flows.rs`) - 6 tests
   - Complete job submission flow (happy path)
   - Policy denial flow
   - Quota exceeded flow
   - Concurrent job submissions
   - Concurrent quota allocations
   - Service health checks

3. **Performance Tests** (`performance_tests.rs`) - 5 tests
   - E2E job submission latency benchmark
   - Multi-service request latency benchmark
   - Concurrent throughput benchmark
   - Governor policy evaluation benchmark
   - Quota Manager check benchmark

4. **Helper Unit Tests** - 7 tests
   - Service endpoint configuration
   - HTTP client creation
   - Request/response DTOs

**Total: 28 tests** (7 unit tests + 21 integration/E2E tests)

## Prerequisites

### Required Services

The integration and E2E tests require running instances of:

1. **PostgreSQL Database**
   - Used by: scheduler, governor, quota-manager
   - Default connection: `postgres://postgres:postgres@localhost:5433/scheduler_test`
   - Set via: `DATABASE_URL` environment variable

2. **Scheduler Service**
   - Default port: 8080
   - Set via: `SCHEDULER_PORT` environment variable
   - Health endpoint: `http://localhost:8080/health`

3. **Governor Service**
   - Default port: 8081
   - Set via: `GOVERNOR_PORT` environment variable
   - Health endpoint: `http://localhost:8081/health`

4. **Quota Manager Service**
   - Default port: 8082
   - Set via: `QUOTA_MANAGER_PORT` environment variable
   - Health endpoint: `http://localhost:8082/health`

5. **API Gateway Service**
   - Default port: 8000
   - Set via: `API_GATEWAY_PORT` environment variable
   - Health endpoint: `http://localhost:8000/health`

### Starting Services

Before running E2E tests, start all required services:

```bash
# Terminal 1: Start PostgreSQL (if not running)
docker run --name horizon-postgres -p 5433:5432 \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=scheduler_test \
  -d postgres:15

# Terminal 2: Start Scheduler
cd services/scheduler
DATABASE_URL=postgres://postgres:postgres@localhost:5433/scheduler_test \
  cargo run

# Terminal 3: Start Governor
cd services/governor
DATABASE_URL=postgres://postgres:postgres@localhost:5433/scheduler_test \
  cargo run

# Terminal 4: Start Quota Manager
cd services/quota-manager
DATABASE_URL=postgres://postgres:postgres@localhost:5433/scheduler_test \
  cargo run

# Terminal 5: Start API Gateway
cd services/api-gateway
cargo run
```

## Running Tests

### Run All Unit Tests (No Services Required)

```bash
cargo test --test multi_service_e2e_tests
```

This runs only the helper unit tests (7 tests) that don't require running services.

### Run All Integration Tests (Requires Services)

```bash
cargo test --test multi_service_e2e_tests -- --ignored
```

This runs all 21 integration and E2E tests. **All services must be running.**

### Run Specific Test Categories

#### Integration Tests Only
```bash
cargo test --test multi_service_e2e_tests integration_tests -- --ignored
```

#### E2E Flow Tests Only
```bash
cargo test --test multi_service_e2e_tests e2e_flows -- --ignored
```

#### Performance Tests Only
```bash
cargo test --test multi_service_e2e_tests performance_tests -- --ignored --nocapture
```

The `--nocapture` flag shows performance statistics output.

### Run Individual Tests

```bash
# Run specific integration test
cargo test --test multi_service_e2e_tests test_gateway_to_governor_policy_evaluation -- --ignored

# Run specific E2E test
cargo test --test multi_service_e2e_tests test_complete_job_submission_flow -- --ignored --nocapture

# Run specific performance test
cargo test --test multi_service_e2e_tests bench_e2e_job_submission_latency -- --ignored --nocapture
```

## Test Structure

```
tests/e2e/
├── mod.rs                      # Module exports
├── README.md                   # This file
├── helpers/
│   ├── mod.rs                  # Helper module exports
│   ├── services.rs             # Service health checking (180 lines)
│   └── clients.rs              # HTTP client helpers (550 lines)
├── integration_tests.rs        # 2-service integration tests (750 lines)
├── e2e_flows.rs                # Full E2E flows (650 lines)
└── performance_tests.rs        # Performance benchmarks (650 lines)
```

**Total Lines**: ~2,780 lines of test code

## Test Details

### Integration Tests (10 tests)

#### API Gateway ↔ Governor (2 tests)
- `test_gateway_to_governor_policy_evaluation` - Policy evaluation through gateway
- `test_gateway_to_governor_policy_denial` - Policy denial enforcement

#### API Gateway ↔ Quota Manager (3 tests)
- `test_gateway_to_quota_manager_check` - Quota checking through gateway
- `test_gateway_to_quota_manager_exceeded` - Quota exceeded handling
- `test_gateway_to_quota_manager_allocation_flow` - Complete allocation flow

#### API Gateway ↔ Scheduler (2 tests)
- `test_gateway_to_scheduler_job_submission` - Job submission through gateway
- `test_gateway_to_scheduler_job_lifecycle` - Job lifecycle management

#### Scheduler ↔ Quota Manager (3 tests)
- `test_scheduler_to_quota_manager_check` - Quota checking for job submission
- `test_scheduler_quota_allocation_on_job_start` - Quota allocation when job starts
- `test_scheduler_quota_release_on_job_completion` - Quota release when job completes

### End-to-End Tests (6 tests)

- `test_complete_job_submission_flow` - Full flow: policy → quota → job submission
- `test_policy_denial_flow` - Policy denies job submission
- `test_quota_exceeded_flow` - Quota limit prevents job submission
- `test_concurrent_job_submissions` - 10 concurrent job submissions
- `test_concurrent_quota_allocations` - 10 concurrent quota allocations
- `test_all_services_health` - Verify all services are healthy

### Performance Tests (5 tests)

Each benchmark runs 100 iterations with warmup and reports:
- Min, Max, Mean latencies
- P50, P95, P99 percentiles

#### Benchmarks
- `bench_e2e_job_submission_latency` - Target: P99 < 50ms
- `bench_multi_service_request_latency` - Target: P99 < 100ms
- `bench_concurrent_throughput` - Target: > 100 req/s
- `bench_governor_policy_evaluation` - Target: P99 < 5ms
- `bench_quota_manager_check` - Target: P99 < 2ms

## Environment Variables

All services can be configured via environment variables:

```bash
# Service ports
export SCHEDULER_PORT=8080
export GOVERNOR_PORT=8081
export QUOTA_MANAGER_PORT=8082
export API_GATEWAY_PORT=8000

# Database
export DATABASE_URL=postgres://postgres:postgres@localhost:5433/scheduler_test

# Run tests with custom configuration
cargo test --test multi_service_e2e_tests -- --ignored
```

## Test Design Principles

1. **No Mocks**: All tests use real HTTP communication with actual services
2. **Idempotent**: Tests can be run multiple times without interference
3. **Independent**: Tests don't depend on each other
4. **Self-Cleaning**: Tests create and clean up their own test data
5. **Descriptive Failures**: Clear error messages when services aren't available
6. **TDD**: Tests were written following strict TDD methodology

## Troubleshooting

### Tests Are Ignored

If you see "21 ignored" when running tests, it means the services aren't running. The tests are marked with `#[ignore]` to prevent failures when services aren't available.

**Solution**: Start all required services (see "Starting Services" above).

### Service Not Available

If a test fails with "Service X is not available at http://localhost:YYYY":

1. Check if the service is running: `curl http://localhost:YYYY/health`
2. Check if the port is correct (set via environment variables)
3. Check service logs for startup errors

### Connection Refused

If you see "Connection refused" errors:

1. Verify PostgreSQL is running: `psql postgres://postgres:postgres@localhost:5433/scheduler_test`
2. Run database migrations: `cd services/scheduler && cargo sqlx migrate run`
3. Check that services are binding to the correct ports

### Performance Tests Failing

Performance tests may fail on slower machines or under load:

1. Run performance tests individually: `cargo test bench_e2e_job_submission_latency -- --ignored --nocapture`
2. Check system load: `top` or `htop`
3. Adjust targets in `performance_tests.rs` if running on constrained hardware

## Test Coverage

### Integration Tests Coverage

| Service Pair | Tests | Coverage |
|--------------|-------|----------|
| Gateway ↔ Governor | 2 | Policy evaluation, denial |
| Gateway ↔ Quota Manager | 3 | Check, exceeded, allocation flow |
| Gateway ↔ Scheduler | 2 | Job submission, lifecycle |
| Scheduler ↔ Quota Manager | 3 | Check, allocation, release |

### End-to-End Scenarios

- ✅ Happy path: Complete job submission
- ✅ Policy denial
- ✅ Quota exceeded
- ✅ Concurrent requests (10 concurrent)
- ✅ Concurrent allocations (10 concurrent)
- ✅ Service health checks

### Performance Targets

| Metric | Target | Benchmark |
|--------|--------|-----------|
| E2E job submission | P99 < 50ms | `bench_e2e_job_submission_latency` |
| Multi-service request | P99 < 100ms | `bench_multi_service_request_latency` |
| Concurrent throughput | > 100 req/s | `bench_concurrent_throughput` |
| Policy evaluation | P99 < 5ms | `bench_governor_policy_evaluation` |
| Quota check | P99 < 2ms | `bench_quota_manager_check` |

## Example Output

### Successful Test Run

```bash
$ cargo test --test multi_service_e2e_tests -- --ignored

running 21 tests
test e2e::e2e_flows::test_all_services_health ... ok
test e2e::e2e_flows::test_complete_job_submission_flow ... ok
test e2e::e2e_flows::test_concurrent_job_submissions ... ok
test e2e::e2e_flows::test_concurrent_quota_allocations ... ok
test e2e::e2e_flows::test_policy_denial_flow ... ok
test e2e::e2e_flows::test_quota_exceeded_flow ... ok
test e2e::integration_tests::test_gateway_to_governor_policy_denial ... ok
test e2e::integration_tests::test_gateway_to_governor_policy_evaluation ... ok
test e2e::integration_tests::test_gateway_to_quota_manager_allocation_flow ... ok
test e2e::integration_tests::test_gateway_to_quota_manager_check ... ok
test e2e::integration_tests::test_gateway_to_quota_manager_exceeded ... ok
test e2e::integration_tests::test_gateway_to_scheduler_job_lifecycle ... ok
test e2e::integration_tests::test_gateway_to_scheduler_job_submission ... ok
test e2e::integration_tests::test_scheduler_quota_allocation_on_job_start ... ok
test e2e::integration_tests::test_scheduler_quota_release_on_job_completion ... ok
test e2e::integration_tests::test_scheduler_to_quota_manager_check ... ok
test e2e::performance_tests::bench_concurrent_throughput ... ok
test e2e::performance_tests::bench_e2e_job_submission_latency ... ok
test e2e::performance_tests::bench_governor_policy_evaluation ... ok
test e2e::performance_tests::bench_multi_service_request_latency ... ok
test e2e::performance_tests::bench_quota_manager_check ... ok

test result: ok. 21 passed; 0 failed; 0 ignored; 0 measured; 7 filtered out
```

### Performance Test Output

```bash
$ cargo test --test multi_service_e2e_tests bench_e2e_job_submission_latency -- --ignored --nocapture

E2E Job Submission Performance Statistics:
  Min:  12.345ms
  Mean: 23.456ms
  P50:  21.234ms
  P95:  34.567ms
  P99:  42.890ms
  Max:  55.678ms

✓ E2E latency target met: P99 = 42.890ms < 50ms
```

## Contributing

When adding new E2E tests:

1. Follow the existing structure (services.rs, clients.rs, test files)
2. Mark integration tests with `#[ignore = "Requires services: ..."]`
3. Use descriptive test names: `test_<component>_<scenario>`
4. Include cleanup code to remove test data
5. Add documentation comments explaining test purpose
6. Keep test files under 900 lines (split if needed)

## References

- Phase 4 Plan: `/home/osobh/projects/horizon/.ai/plans/phase4_gateway_governor_plan.md`
- Scheduler Service: `services/scheduler/`
- Governor Service: `services/governor/`
- Quota Manager: `services/quota-manager/`
- API Gateway: `services/api-gateway/`
