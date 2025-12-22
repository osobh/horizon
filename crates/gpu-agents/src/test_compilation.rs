//! Test to verify module compilation

#[cfg(test)]
mod tests {
    #[test]
    fn test_basic_compilation() {
        // Just testing that basic modules compile
        use crate::error::GpuAgentError;
        use crate::types::GpuSwarmConfig;
        
        let config = GpuSwarmConfig::default();
        assert!(config.max_agents > 0);
    }
    
    #[test] 
    fn test_module_exists() {
        // Test that our key modules exist
        // This will fail compilation if modules don't exist
        
        // Test evolution module (this should use evolution/ directory)
        use crate::evolution;
        let _ = evolution::GpuEvolutionConfig::default();
        
        // Test knowledge module (this should use knowledge/ directory) 
        use crate::knowledge;
        // Just test it exists
    }
}