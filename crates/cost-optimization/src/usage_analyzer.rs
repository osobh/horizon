//! Usage analysis module for resource patterns, optimization recommendations, and waste detection
//!
//! This module has been refactored using TDD methodology to split the original 1671-line file
//! into logical, maintainable modules. The refactoring maintains complete backward compatibility
//! while organizing code into focused modules under 750 lines each.
//!
//! ## TDD Refactoring Process
//!
//! The refactoring followed the Test-Driven Development red-green-refactor cycle:
//!
//! 1. **RED**: Created comprehensive validation tests to establish expected behavior
//! 2. **GREEN**: Split the monolithic file into focused modules
//! 3. **REFACTOR**: Ensured all tests pass and functionality is preserved
//!
//! ## Refactored Module Structure
//!
//! The original monolithic file has been split into:
//!
//! - **`mod.rs`**: Main coordinator and public API (647 lines)
//! - **`types.rs`**: All data structures and enums (419 lines)  
//! - **`statistics.rs`**: Statistical analysis utilities (574 lines)
//! - **`pattern_detector.rs`**: Pattern detection algorithms (348 lines)
//! - **`temporal_analyzer.rs`**: Time-based analysis (453 lines)
//! - **`opportunity_generator.rs`**: Optimization recommendations (686 lines)
//! - **`waste_analyzer.rs`**: Waste detection and analysis (629 lines)
//!
//! All modules are well under the 750-line target, with comprehensive test coverage
//! that validates the refactoring preserves all original functionality.
//!
//! ## Backward Compatibility
//!
//! This file now serves as a compatibility layer, re-exporting all public types and functions
//! from the refactored modules. Existing code will continue to work without modification.
//!
//! ## Usage
//!
//! ```rust
//! use crate::usage_analyzer::{UsageAnalyzer, AnalysisRequest, UsageAnalyzerConfig};
//! use std::time::Duration;
//!
//! // Create analyzer with default configuration
//! let config = UsageAnalyzerConfig::default();
//! let analyzer = UsageAnalyzer::new(config)?;
//!
//! // Analyze resource usage
//! let request = AnalysisRequest {
//!     resource_id: "my-resource".to_string(),
//!     resource_type: ResourceType::Cpu,
//!     period: Duration::from_secs(86400 * 7), // 7 days
//!     include_recommendations: true,
//!     confidence_threshold: 0.8,
//!     cost_per_hour: Some(1.0),
//! };
//!
//! let result = analyzer.analyze_usage(request, snapshots).await?;
//! println!("Pattern: {}, Confidence: {:.2}", result.pattern, result.confidence);
//! ```

// Import the refactored modules
pub use self::usage_analyzer_impl::*;

// Private module containing the actual implementation
#[path = "usage_analyzer_impl/mod.rs"]
mod usage_analyzer_impl;
