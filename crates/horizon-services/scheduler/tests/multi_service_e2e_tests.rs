// Multi-service end-to-end integration tests
//
// This test suite verifies multi-service interactions and complete end-to-end flows.
// It includes:
// - Integration tests (2-service interactions)
// - E2E flow tests (complete request flows)
// - Performance benchmarks
//
// To run these tests, ensure all required services are running:
// - scheduler (port 8080)
// - governor (port 8081)
// - quota-manager (port 8082)
// - api-gateway (port 8000)
// - PostgreSQL database
//
// Run specific test categories:
//   cargo test --test multi_service_e2e_tests -- --ignored
//   cargo test --test multi_service_e2e_tests bench_e2e -- --ignored --nocapture
//
// Set custom ports via environment variables:
//   SCHEDULER_PORT=9080 GOVERNOR_PORT=9081 cargo test --test multi_service_e2e_tests

// Include the e2e test module
mod e2e;

// Re-export for convenience
pub use e2e::helpers::*;
