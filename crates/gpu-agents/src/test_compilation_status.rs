#[cfg(test)]
mod test_compilation_status {
    #[test]
    fn test_main_types_compile() {
        // Test that main types compile
        use crate::knowledge::{KnowledgeEdge, KnowledgeGraph, KnowledgeNode};
        use crate::{GpuEvolutionConfig, GpuEvolutionEngine, GpuSwarm, GpuSwarmConfig};

        // This test just ensures the types exist and can be imported
        let _ = std::mem::size_of::<GpuSwarmConfig>();
        let _ = std::mem::size_of::<GpuEvolutionConfig>();
        let _ = std::mem::size_of::<KnowledgeNode>();
        let _ = std::mem::size_of::<KnowledgeEdge>();
    }
}
