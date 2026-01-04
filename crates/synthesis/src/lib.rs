//! Agent synthesis pipeline for GPU kernel generation
//!
//! This crate implements the intelligent synthesis pipeline that transforms
//! high-level goals into optimized GPU kernels through:
//! - Goal interpretation using LLMs
//! - Template-based kernel generation
//! - Runtime compilation with NVRTC
//! - Performance monitoring and optimization
//! - Continuous improvement through learning

#![warn(missing_docs)]

pub mod compiler;
pub mod error;
pub mod executor;
pub mod improvement;
pub mod intent_orchestrator;
pub mod interpreter;
pub mod pipeline;
pub mod synthesizer;
pub mod templates;

pub use compiler::{CompiledKernel, RuntimeCompiler};
pub use error::{SynthesisError, SynthesisResult};
pub use executor::{ExecutionEngine, ExecutionResult};
pub use improvement::{ImprovedKernel, ImprovementEngine};
pub use intent_orchestrator::{
    ActionPlan, ActionStep, ActionType, Entity, EntityType, ExecutionRecord, ExecutionStatus,
    Intent, IntentOrchestrator, IntentType, OrchestrationMetrics,
};
pub use interpreter::{GoalInterpreter, KernelSpecification};
pub use pipeline::{PipelineConfig, SynthesisPipeline};
pub use synthesizer::{KernelSynthesizer, SynthesizedKernel};

/// Re-export common types
pub mod prelude {
    pub use crate::{
        ActionPlan, CompiledKernel, Entity, ExecutionEngine, ExecutionRecord, ExecutionResult,
        GoalInterpreter, ImprovementEngine, Intent, IntentOrchestrator, IntentType,
        KernelSpecification, KernelSynthesizer, OrchestrationMetrics, PipelineConfig,
        RuntimeCompiler, SynthesisError, SynthesisPipeline, SynthesisResult, SynthesizedKernel,
    };
}

/// Initialize synthesis system
pub async fn init() -> SynthesisResult<()> {
    // Initialize subsystems
    templates::init_template_engine()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_synthesis_init() -> Result<(), Box<dyn std::error::Error>> {
        assert!(init().await.is_ok());
    }

    #[test]
    fn test_prelude_exports() -> Result<(), Box<dyn std::error::Error>> {
        // Verify all types are accessible through prelude
        use crate::prelude::*;

        // Just ensure types exist and can be referenced
        let _config = PipelineConfig::default();
        let _error = SynthesisError::Other("test".to_string());
    }

    #[test]
    fn test_module_exports() -> Result<(), Box<dyn std::error::Error>> {
        // Verify all modules are accessible
        use crate::{
            compiler, error, executor, improvement, interpreter, pipeline, synthesizer, templates,
        };

        // This test just ensures the modules are accessible
        let _ = error::SynthesisError::Other("test".to_string());
    }

    #[test]
    fn test_direct_type_exports() -> Result<(), Box<dyn std::error::Error>> {
        // Test that types can be used directly from crate root
        let config = PipelineConfig::default();
        assert!(config.max_retries > 0);
    }

    #[tokio::test]
    async fn test_init_idempotency() -> Result<(), Box<dyn std::error::Error>> {
        // Initialize multiple times should be safe
        assert!(init().await.is_ok());
        assert!(init().await.is_ok());
        assert!(init().await.is_ok());
    }

    #[test]
    fn test_synthesis_result_type() -> Result<(), Box<dyn std::error::Error>> {
        fn example_function() -> SynthesisResult<String> {
            Ok("success".to_string())
        }

        fn example_error_function() -> SynthesisResult<String> {
            Err(SynthesisError::Other("example error".to_string()))
        }

        assert!(example_function().is_ok());
        assert!(example_error_function().is_err());
    }

    #[test]
    fn test_crate_documentation() -> Result<(), Box<dyn std::error::Error>> {
        // This test just ensures the crate-level documentation compiles
        // The actual test is that the code compiles with #![warn(missing_docs)]
    }

    #[test]
    fn test_type_aliases() -> Result<(), Box<dyn std::error::Error>> {
        // Ensure type aliases work correctly
        type MySynthesisResult = SynthesisResult<()>;

        fn test_alias() -> MySynthesisResult {
            Ok(())
        }

        assert!(test_alias().is_ok());
    }

    #[test]
    fn test_crate_reexports() -> Result<(), Box<dyn std::error::Error>> {
        // Test that main types are accessible from crate root
        let _error = SynthesisError::CompilationFailed {
            reason: "test".to_string(),
        };
    }

    #[test]
    fn test_prelude_completeness() -> Result<(), Box<dyn std::error::Error>> {
        // Import everything from prelude
        use crate::prelude::*;

        // Create instances to verify types are exported correctly
        let config = PipelineConfig {
            max_retries: 3,
            timeout_ms: 5000,
            enable_caching: true,
            optimization_level: 2,
        };

        assert_eq!(config.max_retries, 3);
        assert_eq!(config.timeout_ms, 5000);
        assert!(config.enable_caching);
        assert_eq!(config.optimization_level, 2);
    }

    #[test]
    fn test_error_conversion() -> Result<(), Box<dyn std::error::Error>> {
        // Test that errors can be converted to Box<dyn Error>
        let error: Box<dyn std::error::Error> =
            Box::new(SynthesisError::Other("Test error".to_string()));

        assert!(!error.to_string().is_empty());
    }

    #[tokio::test]
    async fn test_concurrent_init() -> Result<(), Box<dyn std::error::Error>> {
        use tokio::task;

        let mut handles = vec![];

        // Spawn multiple init tasks
        for _ in 0..10 {
            handles.push(task::spawn(async { init().await }));
        }

        // All should succeed
        for handle in handles {
            assert!(handle.await?.is_ok());
        }
    }

    #[test]
    fn test_synthesis_pipeline_types() -> Result<(), Box<dyn std::error::Error>> {
        // Test that we can create type instances
        let kernel_spec = KernelSpecification {
            name: "test_kernel".to_string(),
            goal_description: "Test kernel for unit tests".to_string(),
            input_types: vec!["float*".to_string()],
            output_types: vec!["float*".to_string()],
            constraints: vec![],
            performance_targets: Default::default(),
        };

        assert_eq!(kernel_spec.name, "test_kernel");
        assert_eq!(kernel_spec.goal_description, "Test kernel for unit tests");
        assert_eq!(kernel_spec.input_types.len(), 1);
        assert_eq!(kernel_spec.output_types.len(), 1);
    }

    #[test]
    fn test_execution_result_variants() -> Result<(), Box<dyn std::error::Error>> {
        let success_result = ExecutionResult {
            success: true,
            output: vec![1, 2, 3, 4],
            performance_metrics: Default::default(),
            error_message: None,
        };

        assert!(success_result.success);
        assert_eq!(success_result.output.len(), 4);
        assert!(success_result.error_message.is_none());

        let failure_result = ExecutionResult {
            success: false,
            output: vec![],
            performance_metrics: Default::default(),
            error_message: Some("Kernel execution failed".to_string()),
        };

        assert!(!failure_result.success);
        assert!(failure_result.output.is_empty());
        assert!(failure_result.error_message.is_some());
    }

    #[test]
    fn test_synthesized_kernel_structure() -> Result<(), Box<dyn std::error::Error>> {
        let kernel = SynthesizedKernel {
            kernel_id: uuid::Uuid::new_v4(),
            name: "test_kernel".to_string(),
            source_code: "__global__ void test() {}".to_string(),
            template_used: Some("basic_template".to_string()),
            parameters: std::collections::HashMap::new(),
            metadata: Default::default(),
        };

        assert_eq!(kernel.name, "test_kernel");
        assert!(kernel.source_code.contains("__global__"));
        assert_eq!(kernel.template_used, Some("basic_template".to_string()));
        assert!(kernel.parameters.is_empty());
    }

    #[test]
    fn test_compiled_kernel_fields() -> Result<(), Box<dyn std::error::Error>> {
        let kernel = CompiledKernel {
            kernel_id: uuid::Uuid::new_v4(),
            name: "compiled_test".to_string(),
            ptx_code: vec![0x00, 0x01, 0x02, 0x03],
            cubin: None,
            compile_options: vec!["-O3".to_string()],
            metadata: Default::default(),
        };

        assert_eq!(kernel.name, "compiled_test");
        assert_eq!(kernel.ptx_code.len(), 4);
        assert!(kernel.cubin.is_none());
        assert_eq!(kernel.compile_options.len(), 1);
        assert_eq!(kernel.compile_options[0], "-O3");
    }

    #[test]
    fn test_improved_kernel_tracking() -> Result<(), Box<dyn std::error::Error>> {
        use std::collections::HashMap;

        let mut improvements = HashMap::new();
        improvements.insert(
            "optimization".to_string(),
            "Added loop unrolling".to_string(),
        );
        improvements.insert("memory".to_string(), "Improved coalescing".to_string());

        let improved = ImprovedKernel {
            original_id: uuid::Uuid::new_v4(),
            improved_id: uuid::Uuid::new_v4(),
            improvements,
            performance_gain: 1.5,
            confidence: 0.85,
        };

        assert_eq!(improved.improvements.len(), 2);
        assert_eq!(improved.performance_gain, 1.5);
        assert_eq!(improved.confidence, 0.85);
        assert!(improved.improvements.contains_key("optimization"));
        assert!(improved.improvements.contains_key("memory"));
    }

    #[test]
    fn test_pipeline_config_builder() -> Result<(), Box<dyn std::error::Error>> {
        // Test building pipeline config
        let config = PipelineConfig {
            max_retries: 5,
            timeout_ms: 10000,
            enable_caching: false,
            optimization_level: 3,
        };

        assert_eq!(config.max_retries, 5);
        assert_eq!(config.timeout_ms, 10000);
        assert!(!config.enable_caching);
        assert_eq!(config.optimization_level, 3);
    }

    #[test]
    fn test_error_types() -> Result<(), Box<dyn std::error::Error>> {
        // Test all error variants
        let errors = vec![
            SynthesisError::CompilationFailed {
                reason: "Invalid syntax".to_string(),
            },
            SynthesisError::ExecutionFailed {
                reason: "Out of memory".to_string(),
            },
            SynthesisError::InterpretationFailed {
                reason: "Goal unclear".to_string(),
            },
            SynthesisError::TemplateError {
                reason: "Template not found".to_string(),
            },
            SynthesisError::Other("Unknown error".to_string()),
        ];

        for error in errors {
            // Ensure all errors can be formatted
            let _formatted = format!("{error}");
            let _debug = format!("{:?}", error);
        }
    }

    #[test]
    fn test_kernel_specification_validation() -> Result<(), Box<dyn std::error::Error>> {
        let mut spec = KernelSpecification {
            name: "".to_string(), // Empty name
            goal_description: "Test".to_string(),
            input_types: vec![],
            output_types: vec![],
            constraints: vec![],
            performance_targets: Default::default(),
        };

        // Test with empty name
        assert!(spec.name.is_empty());

        // Test with valid name
        spec.name = "valid_kernel_name".to_string();
        assert!(!spec.name.is_empty());

        // Test with multiple constraints
        spec.constraints = vec![
            "memory_limit:1GB".to_string(),
            "time_limit:100ms".to_string(),
            "precision:fp32".to_string(),
        ];
        assert_eq!(spec.constraints.len(), 3);
    }

    #[test]
    fn test_synthesis_result_propagation() -> Result<(), Box<dyn std::error::Error>> {
        fn inner_function() -> SynthesisResult<i32> {
            Ok(42)
        }

        fn outer_function() -> SynthesisResult<String> {
            let value = inner_function()?;
            Ok(format!("Value: {value}"))
        }

        let result = outer_function();
        assert!(result.is_ok());
        assert_eq!(result?, "Value: 42");
    }

    #[test]
    fn test_synthesis_error_propagation() -> Result<(), Box<dyn std::error::Error>> {
        fn inner_function() -> SynthesisResult<i32> {
            Err(SynthesisError::Other("Inner error".to_string()))
        }

        fn outer_function() -> SynthesisResult<String> {
            let value = inner_function()?;
            Ok(format!("Value: {value}"))
        }

        let result = outer_function();
        assert!(result.is_err());
        assert!(result
            .expect_err("Expected error")
            .to_string()
            .contains("Inner error"));
    }
}
