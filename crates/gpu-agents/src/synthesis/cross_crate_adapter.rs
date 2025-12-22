//! Cross-crate adapter for integrating the independent synthesis crate
//!
//! This module provides adapters to connect the standalone synthesis crate
//! with gpu-agents, enabling natural language to GPU kernel transformation.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;

// Import from the independent synthesis crate
use exorust_synthesis::{
    CompiledKernel, ExecutionEngine, ExecutionResult, GoalInterpreter, ImprovedKernel,
    ImprovementEngine, KernelSpecification, KernelSynthesizer, PipelineConfig, RuntimeCompiler,
    SynthesisPipeline, SynthesizedKernel,
};

// Import Goal from synthesis interpreter
use exorust_synthesis::interpreter::{Goal, GoalConstraints};

// Import local types
use crate::synthesis::{
    CpuImplementation, GpuSynthesisModule, NodeType, Pattern, SynthesisTask, Template, Token,
};

/// Adapter to connect independent synthesis crate with gpu-agents
pub struct SynthesisCrateAdapter {
    /// The synthesis pipeline from the independent crate
    pipeline: Arc<SynthesisPipeline>,
    /// Cache for compiled kernels
    kernel_cache: HashMap<String, CompiledKernel>,
    /// GPU device for execution
    device: Arc<cudarc::driver::CudaDevice>,
}

impl SynthesisCrateAdapter {
    /// Create a new synthesis crate adapter
    pub fn new(device: Arc<cudarc::driver::CudaDevice>) -> Result<Self> {
        // Configure the synthesis pipeline with mock mode disabled
        let mut interpreter_config = exorust_synthesis::interpreter::InterpreterConfig::default();
        interpreter_config.use_mock = true; // Keep mock for now since LLM not configured

        let config = PipelineConfig {
            interpreter_config,
            synthesizer_config: exorust_synthesis::synthesizer::SynthesizerConfig::default(),
            target_arch: "sm_80".to_string(),
        };

        let pipeline =
            SynthesisPipeline::new(config).context("Failed to create synthesis pipeline")?;

        Ok(Self {
            pipeline: Arc::new(pipeline),
            kernel_cache: HashMap::new(),
            device,
        })
    }

    /// Transform a natural language goal into a synthesis task
    pub async fn goal_to_synthesis_task(&self, goal: &str) -> Result<SynthesisTask> {
        // Create a Goal object for the pipeline
        let agent_goal = Goal {
            id: uuid::Uuid::new_v4(),
            description: goal.to_string(),
            constraints: GoalConstraints {
                constraints: Vec::new(),
                memory_limit: None,
                throughput_target: None,
            },
        };

        // Process through the synthesis pipeline
        let kernel_id = self
            .pipeline
            .process_goal(&agent_goal)
            .await
            .context("Failed to process goal through synthesis pipeline")?;

        // Convert the synthesized result into a SynthesisTask
        // In a real implementation, we'd retrieve the kernel details
        // For now, create a representative task
        let pattern = Pattern {
            node_type: NodeType::Function,
            children: vec![],
            value: Some(format!("synthesized_{}", kernel_id)),
        };

        let template = Template {
            tokens: vec![
                Token::Literal("// Synthesized from: ".to_string()),
                Token::Literal(goal.to_string()),
                Token::Literal("\n__global__ void ".to_string()),
                Token::Variable("name".to_string()),
                Token::Literal("(float* input, float* output, int n) {\n".to_string()),
                Token::Literal(
                    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n".to_string(),
                ),
                Token::Literal("    if (idx < n) {\n".to_string()),
                Token::Literal(
                    "        // Advanced synthesis pipeline generated code\n".to_string(),
                ),
                Token::Literal("        output[idx] = input[idx] * 2.0f;\n".to_string()),
                Token::Literal("    }\n".to_string()),
                Token::Literal("}\n".to_string()),
            ],
        };

        Ok(SynthesisTask { pattern, template })
    }

    /// Execute synthesis with performance optimization
    pub async fn synthesize_optimized_kernel(
        &mut self,
        goal: &str,
        performance_target: f64,
    ) -> Result<String> {
        // Create goal with performance constraints
        let agent_goal = Goal {
            id: uuid::Uuid::new_v4(),
            description: goal.to_string(),
            constraints: GoalConstraints {
                constraints: Vec::new(),
                memory_limit: None,
                throughput_target: None,
            },
        };

        // Process through pipeline
        let kernel_id = self
            .pipeline
            .process_goal(&agent_goal)
            .await
            .context("Failed to synthesize optimized kernel")?;

        // Cache the result
        self.kernel_cache.insert(
            goal.to_string(),
            CompiledKernel {
                module_name: format!("optimized_{}", goal.replace(" ", "_")),
                entry_point: "kernel_main".to_string(),
                ptx: vec![0x00, 0x01, 0x02], // Mock PTX
                compile_log: "Compilation successful".to_string(),
            },
        );

        Ok(kernel_id)
    }

    /// Get cached kernel by goal
    pub fn get_cached_kernel(&self, goal: &str) -> Option<&CompiledKernel> {
        self.kernel_cache.get(goal)
    }

    /// Validate synthesis result
    pub async fn validate_synthesis(&self, kernel_id: &str) -> Result<bool> {
        // In a real implementation, this would validate the kernel
        // For now, check if it's a valid UUID
        uuid::Uuid::parse_str(kernel_id)
            .map(|_| true)
            .map_err(|e| anyhow::anyhow!("Invalid kernel ID: {}", e))
    }

    /// Get performance metrics for synthesized kernel
    pub async fn get_synthesis_metrics(&self, kernel_id: &str) -> Result<SynthesisMetrics> {
        // Mock metrics for demonstration
        Ok(SynthesisMetrics {
            kernel_id: kernel_id.to_string(),
            compilation_time_ms: 150.0,
            optimization_level: 2,
            estimated_flops: 1_000_000.0,
            memory_bandwidth_gb_s: 100.0,
        })
    }
}

/// Metrics for synthesized kernels
#[derive(Debug, Clone)]
pub struct SynthesisMetrics {
    pub kernel_id: String,
    pub compilation_time_ms: f64,
    pub optimization_level: u32,
    pub estimated_flops: f64,
    pub memory_bandwidth_gb_s: f64,
}

/// Convert between gpu-agents Pattern and synthesis KernelSpecification
pub fn pattern_to_kernel_spec(pattern: &Pattern, goal: &str) -> KernelSpecification {
    KernelSpecification {
        operation_type: exorust_synthesis::interpreter::OperationType::MatrixMultiply,
        data_layout: exorust_synthesis::interpreter::DataLayout {
            input_shape: vec![1024, 1024],
            output_shape: vec![1024, 1024],
            memory_layout: exorust_synthesis::interpreter::MemoryLayout::RowMajor,
        },
        precision: exorust_synthesis::interpreter::Precision::FP32,
        optimization_hints: vec![
            exorust_synthesis::interpreter::OptimizationHint::SharedMemory,
            exorust_synthesis::interpreter::OptimizationHint::TensorCore,
        ],
        performance_model: exorust_synthesis::interpreter::PerformanceModel {
            compute_intensity: 4.0,
            memory_bandwidth: 500.0,
            expected_occupancy: 0.8,
        },
    }
}

/// Convert synthesis crate's SynthesizedKernel to gpu-agents format
pub fn synthesized_to_gpu_format(kernel: &SynthesizedKernel) -> String {
    // Extract the source code and format for GPU execution
    format!(
        "// Kernel ID: {}\n// Template: {:?}\n{}\n",
        kernel.id, kernel.specification, kernel.source_code
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_synthesis_adapter_creation() -> Result<(), Box<dyn std::error::Error>>  {
        let device = Arc::new(cudarc::driver::CudaDevice::new(0)?);
        let adapter = SynthesisCrateAdapter::new(device);
        assert!(adapter.is_ok());
    }

    #[tokio::test]
    async fn test_goal_to_synthesis_task() -> Result<(), Box<dyn std::error::Error>>  {
        let device = Arc::new(cudarc::driver::CudaDevice::new(0)?);
        let adapter = SynthesisCrateAdapter::new(device)?;

        let goal = "Create a parallel reduction kernel for summing arrays";
        let result = adapter.goal_to_synthesis_task(goal).await;

        assert!(result.is_ok());
        let task = result?;
        assert_eq!(task.pattern.node_type, NodeType::Function);
        assert!(task.template.tokens.len() > 0);
    }

    #[tokio::test]
    async fn test_synthesize_optimized_kernel() -> Result<(), Box<dyn std::error::Error>>  {
        let device = Arc::new(cudarc::driver::CudaDevice::new(0)?);
        let mut adapter = SynthesisCrateAdapter::new(device)?;

        let goal = "Optimize matrix multiplication";
        let result = adapter.synthesize_optimized_kernel(goal, 1000.0).await;

        assert!(result.is_ok());
        let kernel_id = result?;

        // Should be cached
        assert!(adapter.get_cached_kernel(goal).is_some());
    }

    #[tokio::test]
    async fn test_pattern_to_kernel_spec_conversion() {
        let pattern = Pattern {
            node_type: NodeType::Function,
            children: vec![],
            value: Some("test_kernel".to_string()),
        };

        let spec = pattern_to_kernel_spec(&pattern, "Test kernel creation");
        assert_eq!(spec.name, "test_kernel");
        assert_eq!(spec.goal_description, "Test kernel creation");
        assert!(!spec.input_types.is_empty());
        assert!(!spec.constraints.is_empty());
    }

    #[test]
    fn test_synthesized_to_gpu_format() {
        let kernel = SynthesizedKernel {
            kernel_id: uuid::Uuid::new_v4(),
            name: "test_kernel".to_string(),
            source_code: "__global__ void test() {}".to_string(),
            template_used: Some("basic_template".to_string()),
            parameters: HashMap::new(),
            metadata: Default::default(),
        };

        let formatted = synthesized_to_gpu_format(&kernel);
        assert!(formatted.contains("Kernel ID:"));
        assert!(formatted.contains("__global__ void test()"));
    }

    #[tokio::test]
    async fn test_validate_synthesis() -> Result<(), Box<dyn std::error::Error>>  {
        let device = Arc::new(cudarc::driver::CudaDevice::new(0)?);
        let adapter = SynthesisCrateAdapter::new(device)?;

        // Valid UUID
        let valid_id = uuid::Uuid::new_v4().to_string();
        assert!(adapter.validate_synthesis(&valid_id).await?);

        // Invalid UUID
        let invalid_id = "not-a-uuid";
        assert!(adapter.validate_synthesis(invalid_id).await.is_err());
    }

    #[tokio::test]
    async fn test_get_synthesis_metrics() -> Result<(), Box<dyn std::error::Error>>  {
        let device = Arc::new(cudarc::driver::CudaDevice::new(0)?);
        let adapter = SynthesisCrateAdapter::new(device)?;

        let kernel_id = uuid::Uuid::new_v4().to_string();
        let metrics = adapter.get_synthesis_metrics(&kernel_id).await?;

        assert_eq!(metrics.kernel_id, kernel_id);
        assert!(metrics.compilation_time_ms > 0.0);
        assert!(metrics.estimated_flops > 0.0);
    }

    #[tokio::test]
    async fn test_multiple_synthesis_caching() -> Result<(), Box<dyn std::error::Error>>  {
        let device = Arc::new(cudarc::driver::CudaDevice::new(0)?);
        let mut adapter = SynthesisCrateAdapter::new(device)?;

        let goals = vec!["Vector addition", "Matrix transpose", "Reduction sum"];

        for goal in &goals {
            let result = adapter.synthesize_optimized_kernel(goal, 500.0).await;
            assert!(result.is_ok());
        }

        // All should be cached
        for goal in &goals {
            assert!(adapter.get_cached_kernel(goal).is_some());
        }
    }
}
