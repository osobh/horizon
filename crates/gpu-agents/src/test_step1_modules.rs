//! TDD Step 1: Test module resolution after fixing conflicts
//! Following rust.md TDD principles - verify fixes work before proceeding

#[cfg(test)]
mod step1_module_tests {
    use anyhow::Result;

    /// Test evolution module resolution (should now work)
    #[test]
    fn test_evolution_imports_work() -> Result<()> {
        // These imports should now work after merging modules
        use crate::evolution::{
            FitnessObjective, GpuEvolutionConfig, GpuEvolutionEngine, MutationStrategy,
            SelectionStrategy,
        };

        let config = GpuEvolutionConfig::default();
        assert!(config.population_size > 0);

        let _objective = FitnessObjective::Performance;
        let _selection = SelectionStrategy::Tournament;
        let _mutation = MutationStrategy::Adaptive;

        println!("✅ Evolution module imports working");
        Ok(())
    }

    /// Test knowledge module resolution (should now work)
    #[test]
    fn test_knowledge_imports_work() -> Result<()> {
        use crate::knowledge::{
            CsrGraph, EnhancedGpuKnowledgeGraph, GraphQuery, KnowledgeEdge, KnowledgeNode,
        };

        let query = GraphQuery::default();
        assert!(!query.query_text.is_empty());

        println!("✅ Knowledge module imports working");
        Ok(())
    }

    /// Test streaming module resolution (should now work)
    #[test]
    fn test_streaming_imports_work() -> Result<()> {
        use crate::streaming::{
            CompressionAlgorithm, GpuCompressor, GpuStreamConfig, GpuTransformer, PipelineBuilder,
            TransformType,
        };

        let config = GpuStreamConfig::default();
        assert!(config.chunk_size > 0);

        let _compression = CompressionAlgorithm::Lz4;
        let _transform = TransformType::JsonParse;

        println!("✅ Streaming module imports working");
        Ok(())
    }

    /// Test storage module imports (should now work)
    #[test]
    fn test_storage_imports_work() -> Result<()> {
        use crate::storage::{GpuAgentStorage, GpuStorageConfig};

        let config = GpuStorageConfig::default();
        assert!(!config.base_path.as_os_str().is_empty());

        println!("✅ Storage module imports working");
        Ok(())
    }

    /// Integration test - all modules should work together
    #[test]
    fn test_all_modules_integration() -> Result<()> {
        // Import from all modules
        use crate::evolution::GpuEvolutionConfig;
        use crate::knowledge::GraphQuery;
        use crate::storage::GpuStorageConfig;
        use crate::streaming::GpuStreamConfig;

        // Create instances
        let _evo_config = GpuEvolutionConfig::default();
        let _query = GraphQuery::default();
        let _stream_config = GpuStreamConfig::default();
        let _storage_config = GpuStorageConfig::default();

        println!("✅ Step 1 Complete: All module conflicts resolved");
        Ok(())
    }
}
