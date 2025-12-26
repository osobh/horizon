//! Test-Driven Development for GPU Agents
//!
//! Following rust.md TDD principles:
//! - Write failing tests first
//! - Implement minimal code to pass tests
//! - Refactor while keeping tests green
//! - Target >80% test coverage

#[cfg(test)]
mod module_resolution_tests;

#[cfg(test)]
mod storage_integration_tests;

#[cfg(test)]
mod cuda_device_tests;

#[cfg(test)]
mod streaming_api_tests;

#[cfg(test)]
mod benchmark_integration_tests;
