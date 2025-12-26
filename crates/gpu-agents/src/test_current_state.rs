#[cfg(test)]
mod test_current_state {
    #[test]
    fn test_module_imports() {
        // Test that we can access types from evolution module
        use crate::evolution::EvolutionConfig;
        let _ = std::mem::size_of::<EvolutionConfig>();

        // Test that we can access types from knowledge module
        use crate::knowledge::KnowledgeGraph;
        let _ = std::mem::size_of::<KnowledgeGraph>();

        println!("Module imports work correctly!");
    }
}
