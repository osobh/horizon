//! Module resolution tests following rust.md TDD guidelines
//!
//! These tests will fail until we fix the module ambiguity errors

use anyhow::Result;

/// Test that evolution module can be imported without ambiguity
/// This test will fail until we resolve evolution.rs vs evolution/mod.rs conflict
#[test]
fn test_evolution_module_resolution() -> Result<()> {
    // Following rust.md: Test imports should be explicit and well-organized
    use crate::evolution::{
        GpuEvolutionConfig, GpuEvolutionEngine, GpuFitnessEvaluator, GpuMutationEngine,
        GpuPopulation, GpuSelectionStrategy,
    };

    // Test that we can create default configs (basic functionality)
    let config = GpuEvolutionConfig::default();
    assert!(config.population_size > 0);
    assert!(config.genome_size > 0);

    Ok(())
}

/// Test that knowledge module can be imported without ambiguity
/// This test will fail until we resolve knowledge.rs vs knowledge/mod.rs conflict
#[test]
fn test_knowledge_module_resolution() -> Result<()> {
    // Following rust.md: Import organization - group related imports
    use crate::knowledge::{
        CsrGraph, EnhancedGpuKnowledgeGraph, GraphQuery, KnowledgeEdge, KnowledgeGraph,
        KnowledgeNode, SpatialIndex,
    };

    // Test basic functionality
    let query = GraphQuery::default();
    assert!(!query.query_text.is_empty());

    Ok(())
}

/// Test that streaming module exports are properly structured
/// Following cuda.md: GPU-specific modules should have clear interfaces
#[test]
fn test_streaming_module_exports() -> Result<()> {
    // These imports should work once we fix re-exports
    use crate::streaming::{
        CompressionAlgorithm, GpuBufferPool, GpuCompressor, GpuStreamConfig, PipelineBuilder,
        TransformType,
    };

    // Test enum variants are accessible
    let _compression = CompressionAlgorithm::Lz4;
    let _transform = TransformType::JsonParse;

    // Test config creation
    let config = GpuStreamConfig::default();
    assert!(config.chunk_size > 0);

    Ok(())
}

/// Integration test: All modules should be importable together
/// This comprehensive test validates our module structure
#[test]
fn test_comprehensive_module_integration() -> Result<()> {
    // Following rules.md: ALWAYS write comprehensive tests

    // Evolution imports
    use crate::evolution::GpuEvolutionConfig;

    // Knowledge imports
    use crate::knowledge::GraphQuery;

    // Streaming imports
    use crate::streaming::GpuStreamConfig;

    // Storage imports (local module)
    use crate::storage::GpuStorageConfig;

    // All should create successfully
    let _evo_config = GpuEvolutionConfig::default();
    let _query = GraphQuery::default();
    let _stream_config = GpuStreamConfig::default();
    let _storage_config = GpuStorageConfig::default();

    println!("✅ All modules resolved successfully");
    Ok(())
}
