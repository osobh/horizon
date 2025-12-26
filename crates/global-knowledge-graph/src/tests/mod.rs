//! Comprehensive test suite for Global Knowledge Graph
//!
//! This module provides comprehensive test coverage for all components of the global
//! knowledge graph system, following Test-Driven Development (TDD) principles.
//!
//! Target: 139+ tests across 7 modules with 85%+ coverage
//! Modules tested: graph_manager, replication, query_engine, compliance_handler,
//! consistency_manager, region_router, cache_layer

#[cfg(test)]
mod cache_layer_tests;
#[cfg(test)]
mod compliance_tests;
#[cfg(test)]
mod consistency_tests;
#[cfg(test)]
mod error_tests;
#[cfg(test)]
mod graph_manager_tests;
#[cfg(test)]
mod integration_tests;
#[cfg(test)]
mod performance_tests;
#[cfg(test)]
mod query_engine_tests;
#[cfg(test)]
mod region_router_tests;
#[cfg(test)]
mod replication_tests;
