//! Goal interpretation engine using LLMs

use crate::error::{SynthesisError, SynthesisResult};
use std::error::Error;
use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage, CreateChatCompletionRequestArgs,
};
use async_openai::{config::OpenAIConfig, Client};
// use exorust_agent_core::{Goal, GoalConstraints};
// Mock for testing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Goal {
    pub id: uuid::Uuid,
    pub description: String,
    pub constraints: GoalConstraints,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GoalConstraints {
    pub constraints: Vec<String>,
    pub throughput_target: Option<f64>,
    pub memory_limit: Option<u64>,
}
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Operation types for kernels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationType {
    /// Matrix multiplication
    MatrixMultiply,
    /// Reduction operation
    Reduction,
    /// Convolution
    Convolution,
    /// Element-wise operation
    Elementwise,
    /// Custom operation
    Custom,
}

/// Data layout specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLayout {
    /// Input tensor shape
    pub input_shape: Vec<usize>,
    /// Output tensor shape
    pub output_shape: Vec<usize>,
    /// Memory layout
    pub memory_layout: MemoryLayout,
}

/// Memory layout types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryLayout {
    /// Row-major (C-style)
    RowMajor,
    /// Column-major (Fortran-style)
    ColumnMajor,
    /// Custom layout
    Custom,
}

/// Numeric precision
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    /// 32-bit floating point
    FP32,
    /// 16-bit floating point
    FP16,
    /// Brain floating point 16
    BF16,
    /// 8-bit integer
    INT8,
}

/// Optimization hints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationHint {
    /// Use tensor cores
    TensorCore,
    /// Use shared memory
    SharedMemory,
    /// Unroll loops
    Unrolling,
    /// Vectorize operations
    Vectorization,
    /// Fuse operations
    Fusion,
}

/// Performance model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceModel {
    /// Compute intensity (FLOPS/byte)
    pub compute_intensity: f32,
    /// Memory bandwidth requirement (GB/s)
    pub memory_bandwidth: f32,
    /// Expected occupancy (0.0-1.0)
    pub expected_occupancy: f32,
}

/// Kernel specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelSpecification {
    /// Operation type
    pub operation_type: OperationType,
    /// Data layout
    pub data_layout: DataLayout,
    /// Numeric precision
    pub precision: Precision,
    /// Optimization hints
    pub optimization_hints: Vec<OptimizationHint>,
    /// Performance model
    pub performance_model: PerformanceModel,
}

/// Context for interpretation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpretationContext {
    /// Historical successful kernels
    pub historical_kernels: Vec<KernelSpecification>,
    /// Device capabilities
    pub device_capabilities: HashMap<String, serde_json::Value>,
    /// Previous interpretations
    pub previous_interpretations: Vec<(String, KernelSpecification)>,
}

/// Goal interpreter configuration
#[derive(Debug, Clone)]
pub struct InterpreterConfig {
    /// LLM model to use
    pub model: String,
    /// Temperature for generation
    pub temperature: f32,
    /// Maximum tokens
    pub max_tokens: u16,
    /// Use mock LLM
    pub use_mock: bool,
}

impl Default for InterpreterConfig {
    fn default() -> Self {
        Self {
            model: "gpt-4".to_string(),
            temperature: 0.3,
            max_tokens: 2000,
            use_mock: cfg!(feature = "mock-llm"),
        }
    }
}

/// Goal interpretation engine
pub struct GoalInterpreter {
    config: InterpreterConfig,
    #[cfg(not(feature = "mock-llm"))]
    llm_client: Client<OpenAIConfig>,
    context_cache: Arc<RwLock<HashMap<Uuid, InterpretationContext>>>,
}

impl GoalInterpreter {
    /// Create new interpreter
    pub fn new(config: InterpreterConfig) -> Self {
        Self {
            config,
            #[cfg(not(feature = "mock-llm"))]
            llm_client: Client::new(),
            context_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Interpret a goal into kernel specification
    pub async fn interpret(&self, goal: &Goal) -> SynthesisResult<KernelSpecification> {
        // Build context
        let context = self.build_context(goal).await?;

        // Create prompt
        let prompt = self.create_interpretation_prompt(goal, &context)?;

        // Get interpretation
        let spec_json = if self.config.use_mock {
            self.mock_interpret(goal, &prompt).await?
        } else {
            self.llm_interpret(&prompt).await?
        };

        // Parse specification
        let spec: KernelSpecification =
            serde_json::from_str(&spec_json).map_err(|e| SynthesisError::InvalidSpecification {
                message: format!("Failed to parse specification: {e}"),
            })?;

        // Validate specification
        self.validate_specification(&spec, &goal.constraints)?;

        // Cache context
        self.cache_context(goal.id, context);

        Ok(spec)
    }

    /// Build interpretation context
    async fn build_context(&self, _goal: &Goal) -> SynthesisResult<InterpretationContext> {
        // In real implementation, would gather historical data
        Ok(InterpretationContext {
            historical_kernels: vec![],
            device_capabilities: HashMap::new(),
            previous_interpretations: vec![],
        })
    }

    /// Create interpretation prompt
    fn create_interpretation_prompt(
        &self,
        goal: &Goal,
        context: &InterpretationContext,
    ) -> SynthesisResult<String> {
        let examples = self.get_examples();

        Ok(format!(
            r#"Given the following goal, generate a formal kernel specification:

Goal: {}
Constraints: {:?}
Historical Context: {} previous successful kernels

Generate a JSON specification with the following structure:
{{
    "operation_type": "matrix_multiply|reduction|convolution|elementwise|custom",
    "data_layout": {{
        "input_shape": [dimensions],
        "output_shape": [dimensions],
        "memory_layout": "row_major|column_major|custom"
    }},
    "precision": "fp32|fp16|bf16|int8",
    "optimization_hints": ["tensor_core", "shared_memory", "unrolling", "vectorization", "fusion"],
    "performance_model": {{
        "compute_intensity": float,
        "memory_bandwidth": float,
        "expected_occupancy": float
    }}
}}

Consider the following optimization strategies:
1. Use Tensor Cores for matrix operations when applicable
2. Optimize for memory coalescing
3. Balance compute vs memory bandwidth
4. Consider power efficiency constraints

Examples:
{}

Generate only valid JSON, no explanations."#,
            goal.description,
            goal.constraints,
            context.historical_kernels.len(),
            examples
        ))
    }

    /// Get example specifications
    fn get_examples(&self) -> String {
        r#"Example 1 - Matrix Multiplication:
{
    "operation_type": "matrix_multiply",
    "data_layout": {
        "input_shape": [1024, 1024],
        "output_shape": [1024, 1024],
        "memory_layout": "row_major"
    },
    "precision": "fp16",
    "optimization_hints": ["tensor_core", "shared_memory"],
    "performance_model": {
        "compute_intensity": 40.5,
        "memory_bandwidth": 900.0,
        "expected_occupancy": 0.85
    }
}"#
        .to_string()
    }

    /// Mock interpretation for testing
    async fn mock_interpret(&self, goal: &Goal, _prompt: &str) -> SynthesisResult<String> {
        // Simple mock logic based on goal description
        let operation_type = if goal.description.contains("matrix") {
            "matrix_multiply"
        } else if goal.description.contains("reduce") || goal.description.contains("sum") {
            "reduction"
        } else if goal.description.contains("conv") {
            "convolution"
        } else {
            "elementwise"
        };

        Ok(format!(
            r#"{{
    "operation_type": "{}",
    "data_layout": {{
        "input_shape": [1024, 1024],
        "output_shape": [1024, 1024],
        "memory_layout": "row_major"
    }},
    "precision": "fp32",
    "optimization_hints": ["shared_memory"],
    "performance_model": {{
        "compute_intensity": 10.0,
        "memory_bandwidth": 500.0,
        "expected_occupancy": 0.75
    }}
}}"#,
            operation_type
        ))
    }

    /// LLM interpretation
    async fn llm_interpret(&self, prompt: &str) -> SynthesisResult<String> {
        #[cfg(feature = "mock-llm")]
        {
            // If mock-llm is enabled but use_mock is false, return a mock response
            return self
                .mock_interpret(
                    &Goal::new(
                        "Mock goal".to_string(),
                        GoalPriority::Normal,
                    ),
                    prompt,
                )
                .await;
        }
        #[cfg(not(feature = "mock-llm"))]
        {
            let request = CreateChatCompletionRequestArgs::default()
            .model(&self.config.model)
            .messages([
                ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                    content: async_openai::types::ChatCompletionRequestSystemMessageContent::Text("You are an expert GPU kernel optimization assistant. Generate only valid JSON specifications.".to_string()),
                    ..Default::default()
                }),
                ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                    content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(prompt.to_string()),
                    ..Default::default()
                }),
            ])
            .temperature(self.config.temperature)
            .max_tokens(self.config.max_tokens as u32)
            .build()
            .map_err(|e| SynthesisError::LlmApiError {
                message: format!("Failed to build request: {e}"),
            })?;

            let response = self.llm_client.chat().create(request).await.map_err(|e| {
                SynthesisError::LlmApiError {
                    message: format!("LLM request failed: {e}"),
                }
            })?;

            response
                .choices
                .first()
                .and_then(|choice| choice.message.content.clone())
                .ok_or_else(|| SynthesisError::LlmApiError {
                    message: "Empty response from LLM".to_string(),
                })
        }
    }

    /// Validate specification against constraints
    fn validate_specification(
        &self,
        spec: &KernelSpecification,
        constraints: &GoalConstraints,
    ) -> SynthesisResult<()> {
        // Check memory constraints
        if let Some(memory_limit) = constraints.memory_limit {
            let estimated_memory = self.estimate_memory_usage(spec);
            if estimated_memory > memory_limit as usize {
                return Err(SynthesisError::InvalidSpecification {
                    message: format!(
                        "Memory usage {} exceeds limit {}",
                        estimated_memory, memory_limit
                    ),
                });
            }
        }

        // Validate performance model
        if spec.performance_model.expected_occupancy > 1.0
            || spec.performance_model.expected_occupancy < 0.0
        {
            return Err(SynthesisError::InvalidSpecification {
                message: "Invalid occupancy value".to_string(),
            });
        }

        Ok(())
    }

    /// Estimate memory usage for specification
    fn estimate_memory_usage(&self, spec: &KernelSpecification) -> usize {
        let element_size = match spec.precision {
            Precision::FP32 | Precision::INT8 => 4,
            Precision::FP16 | Precision::BF16 => 2,
        };

        let input_elements: usize = spec.data_layout.input_shape.iter().product();
        let output_elements: usize = spec.data_layout.output_shape.iter().product();

        (input_elements + output_elements) * element_size
    }

    /// Cache interpretation context
    fn cache_context(&self, goal_id: Uuid, context: InterpretationContext) {
        let mut cache = self.context_cache.write();
        cache.insert(goal_id, context);

        // Limit cache size
        if cache.len() > 1000 {
            // Remove oldest entries (simple FIFO for now)
            let to_remove: Vec<_> = cache.keys().take(100).cloned().collect();
            for key in to_remove {
                cache.remove(&key);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[derive(Clone, Debug)]
    pub enum GoalPriority {
        Low,
        Normal,
        High,
    }
    
    impl Goal {
        fn new(description: String, _priority: GoalPriority) -> Self {
            Goal {
                id: uuid::Uuid::new_v4(),
                description,
                constraints: GoalConstraints {
                    constraints: vec![],
                    throughput_target: None,
                    memory_limit: None,
                },
            }
        }
    }
    
    impl Default for GoalConstraints {
        fn default() -> Self {
            GoalConstraints {
                constraints: vec![],
                throughput_target: None,
                memory_limit: None,
            }
        }
    }

    #[test]
    fn test_operation_type_serialization() -> Result<(), Box<dyn std::error::Error>> {
        let op = OperationType::MatrixMultiply;
        let json = serde_json::to_string(&op)?;
        assert_eq!(json, r#""matrix_multiply""#);

        let parsed: OperationType = serde_json::from_str(&json)?;
        assert_eq!(parsed, op);
    }

    #[test]
    fn test_kernel_specification_serialization() -> Result<(), Box<dyn std::error::Error>> {
        let spec = KernelSpecification {
            operation_type: OperationType::Reduction,
            data_layout: DataLayout {
                input_shape: vec![1024, 512],
                output_shape: vec![1024],
                memory_layout: MemoryLayout::RowMajor,
            },
            precision: Precision::FP16,
            optimization_hints: vec![OptimizationHint::SharedMemory],
            performance_model: PerformanceModel {
                compute_intensity: 5.0,
                memory_bandwidth: 300.0,
                expected_occupancy: 0.8,
            },
        };

        let json = serde_json::to_string(&spec)?;
        let parsed: KernelSpecification = serde_json::from_str(&json)?;

        assert_eq!(parsed.operation_type, spec.operation_type);
        assert_eq!(parsed.precision, spec.precision);
        assert_eq!(parsed.data_layout.input_shape, spec.data_layout.input_shape);
    }

    #[tokio::test]
    async fn test_goal_interpretation_mock() -> Result<(), Box<dyn std::error::Error>> {
        let config = InterpreterConfig {
            use_mock: true,
            ..Default::default()
        };

        let interpreter = GoalInterpreter::new(config);

        let goal = Goal::new(
            "Optimize matrix multiplication for transformer model".to_string(),
            GoalPriority::High,
        );

        let spec = interpreter.interpret(&goal).await?;

        assert_eq!(spec.operation_type, OperationType::MatrixMultiply);
        assert_eq!(spec.precision, Precision::FP32);
        assert!(!spec.optimization_hints.is_empty());
    }

    #[tokio::test]
    async fn test_memory_estimation() -> Result<(), Box<dyn std::error::Error>> {
        let interpreter = GoalInterpreter::new(InterpreterConfig::default());

        let spec = KernelSpecification {
            operation_type: OperationType::Elementwise,
            data_layout: DataLayout {
                input_shape: vec![1000, 1000],  // 1M elements
                output_shape: vec![1000, 1000], // 1M elements
                memory_layout: MemoryLayout::RowMajor,
            },
            precision: Precision::FP32, // 4 bytes per element
            optimization_hints: vec![],
            performance_model: PerformanceModel {
                compute_intensity: 1.0,
                memory_bandwidth: 100.0,
                expected_occupancy: 0.5,
            },
        };

        let memory = interpreter.estimate_memory_usage(&spec);
        assert_eq!(memory, 2_000_000 * 4); // 2M elements * 4 bytes
    }

    #[tokio::test]
    async fn test_constraint_validation() -> Result<(), Box<dyn std::error::Error>> {
        let interpreter = GoalInterpreter::new(InterpreterConfig::default());

        let spec = KernelSpecification {
            operation_type: OperationType::MatrixMultiply,
            data_layout: DataLayout {
                input_shape: vec![10000, 10000], // Large matrices
                output_shape: vec![10000, 10000],
                memory_layout: MemoryLayout::RowMajor,
            },
            precision: Precision::FP32,
            optimization_hints: vec![],
            performance_model: PerformanceModel {
                compute_intensity: 40.0,
                memory_bandwidth: 900.0,
                expected_occupancy: 0.85,
            },
        };

        let mut constraints = GoalConstraints::default();
        constraints.memory_limit = Some(1024 * 1024); // 1MB limit

        let result = interpreter.validate_specification(&spec, &constraints);
        assert!(result.is_err());
        assert!(result.expect_err("Expected error").to_string().contains("Memory usage"));
    }

    #[tokio::test]
    async fn test_different_operation_types() -> Result<(), Box<dyn std::error::Error>> {
        let config = InterpreterConfig {
            use_mock: true,
            ..Default::default()
        };

        let interpreter = GoalInterpreter::new(config);

        // Test reduction
        let goal = Goal::new(
            "Calculate sum reduction over large tensor".to_string(),
            GoalPriority::Normal,
        );

        let spec = interpreter.interpret(&goal).await?;
        assert_eq!(spec.operation_type, OperationType::Reduction);

        // Test convolution
        let goal = Goal::new(
            "Implement 2D convolution for image processing".to_string(),
            GoalPriority::Normal,
        );

        let spec = interpreter.interpret(&goal).await?;
        assert_eq!(spec.operation_type, OperationType::Convolution);
    }

    #[test]
    fn test_memory_layout_serialization() -> Result<(), Box<dyn std::error::Error>> {
        let layouts = vec![
            MemoryLayout::RowMajor,
            MemoryLayout::ColumnMajor,
            MemoryLayout::Custom,
        ];

        for layout in layouts {
            let json = serde_json::to_string(&layout)?;
            let parsed: MemoryLayout = serde_json::from_str(&json)?;
            assert_eq!(parsed, layout);
        }
    }

    #[test]
    fn test_precision_serialization() -> Result<(), Box<dyn std::error::Error>> {
        let precisions = vec![
            Precision::FP32,
            Precision::FP16,
            Precision::BF16,
            Precision::INT8,
        ];

        for precision in precisions {
            let json = serde_json::to_string(&precision)?;
            let parsed: Precision = serde_json::from_str(&json)?;
            assert_eq!(parsed, precision);
        }
    }

    #[test]
    fn test_optimization_hint_serialization() -> Result<(), Box<dyn std::error::Error>> {
        let hints = vec![
            OptimizationHint::TensorCore,
            OptimizationHint::SharedMemory,
            OptimizationHint::Unrolling,
            OptimizationHint::Vectorization,
            OptimizationHint::Fusion,
        ];

        for hint in hints {
            let json = serde_json::to_string(&hint)?;
            let parsed: OptimizationHint = serde_json::from_str(&json)?;
            assert_eq!(parsed, hint);
        }
    }

    #[test]
    fn test_data_layout_validation() -> Result<(), Box<dyn std::error::Error>> {
        let layout = DataLayout {
            input_shape: vec![2, 3, 4],
            output_shape: vec![2, 12],
            memory_layout: MemoryLayout::RowMajor,
        };

        // Check total elements match
        let input_elements: usize = layout.input_shape.iter().product();
        let output_elements: usize = layout.output_shape.iter().product();
        assert_eq!(input_elements, 24);
        assert_eq!(output_elements, 24);
    }

    #[test]
    fn test_performance_model_bounds() -> Result<(), Box<dyn std::error::Error>> {
        let model = PerformanceModel {
            compute_intensity: 50.0,
            memory_bandwidth: 1000.0,
            expected_occupancy: 0.9,
        };

        assert!(model.compute_intensity > 0.0);
        assert!(model.memory_bandwidth > 0.0);
        assert!(model.expected_occupancy >= 0.0);
        assert!(model.expected_occupancy <= 1.0);
    }

    #[test]
    fn test_interpreter_config_default() -> Result<(), Box<dyn std::error::Error>> {
        let config = InterpreterConfig::default();
        assert_eq!(config.model, "gpt-4");
        assert_eq!(config.temperature, 0.3);
        assert_eq!(config.max_tokens, 2000);
        assert_eq!(config.use_mock, cfg!(feature = "mock-llm"));
    }

    #[tokio::test]
    async fn test_interpretation_context_building() -> Result<(), Box<dyn std::error::Error>> {
        let interpreter = GoalInterpreter::new(InterpreterConfig::default());
        let goal = Goal::new("Test goal".to_string(), GoalPriority::Normal);

        let context = interpreter.build_context(&goal).await?;
        assert!(context.historical_kernels.is_empty());
        assert!(context.device_capabilities.is_empty());
        assert!(context.previous_interpretations.is_empty());
    }

    #[test]
    fn test_kernel_specification_cloning() -> Result<(), Box<dyn std::error::Error>> {
        let spec = KernelSpecification {
            operation_type: OperationType::Custom,
            data_layout: DataLayout {
                input_shape: vec![100, 200],
                output_shape: vec![100, 100],
                memory_layout: MemoryLayout::Custom,
            },
            precision: Precision::BF16,
            optimization_hints: vec![OptimizationHint::TensorCore, OptimizationHint::Fusion],
            performance_model: PerformanceModel {
                compute_intensity: 25.0,
                memory_bandwidth: 750.0,
                expected_occupancy: 0.7,
            },
        };

        let cloned = spec.clone();
        assert_eq!(cloned.operation_type, spec.operation_type);
        assert_eq!(cloned.precision, spec.precision);
        assert_eq!(
            cloned.optimization_hints.len(),
            spec.optimization_hints.len()
        );
    }

    #[test]
    fn test_interpretation_context_serialization() -> Result<(), Box<dyn std::error::Error>> {
        let context = InterpretationContext {
            historical_kernels: vec![],
            device_capabilities: {
                let mut caps = HashMap::new();
                caps.insert("compute_capability".to_string(), serde_json::json!(8.6));
                caps.insert("memory_gb".to_string(), serde_json::json!(80));
                caps
            },
            previous_interpretations: vec![],
        };

        let json = serde_json::to_string(&context)?;
        let parsed: InterpretationContext = serde_json::from_str(&json)?;

        assert_eq!(parsed.device_capabilities.len(), 2);
        assert!(parsed
            .device_capabilities
            .contains_key("compute_capability"));
    }

    #[tokio::test]
    async fn test_prompt_generation() -> Result<(), Box<dyn std::error::Error>> {
        let interpreter = GoalInterpreter::new(InterpreterConfig::default());
        let goal = Goal::new(
            "Optimize GEMM for large matrices".to_string(),
            GoalPriority::High,
        );
        let context = InterpretationContext {
            historical_kernels: vec![],
            device_capabilities: HashMap::new(),
            previous_interpretations: vec![],
        };

        let prompt = interpreter
            .create_interpretation_prompt(&goal, &context)
            .unwrap();
        assert!(prompt.contains("Goal: Optimize GEMM for large matrices"));
        assert!(prompt.contains("matrix_multiply"));
        assert!(prompt.contains("JSON"));
    }

    #[test]
    fn test_memory_estimation_different_precisions() -> Result<(), Box<dyn std::error::Error>> {
        let interpreter = GoalInterpreter::new(InterpreterConfig::default());

        // Test FP32
        let spec_fp32 = KernelSpecification {
            operation_type: OperationType::Elementwise,
            data_layout: DataLayout {
                input_shape: vec![100],
                output_shape: vec![100],
                memory_layout: MemoryLayout::RowMajor,
            },
            precision: Precision::FP32,
            optimization_hints: vec![],
            performance_model: PerformanceModel {
                compute_intensity: 1.0,
                memory_bandwidth: 100.0,
                expected_occupancy: 0.5,
            },
        };

        let memory_fp32 = interpreter.estimate_memory_usage(&spec_fp32);
        assert_eq!(memory_fp32, 200 * 4); // 200 elements * 4 bytes

        // Test FP16
        let mut spec_fp16 = spec_fp32.clone();
        spec_fp16.precision = Precision::FP16;
        let memory_fp16 = interpreter.estimate_memory_usage(&spec_fp16);
        assert_eq!(memory_fp16, 200 * 2); // 200 elements * 2 bytes
    }

    #[tokio::test]
    async fn test_validation_invalid_occupancy() -> Result<(), Box<dyn std::error::Error>> {
        let interpreter = GoalInterpreter::new(InterpreterConfig::default());

        let spec = KernelSpecification {
            operation_type: OperationType::MatrixMultiply,
            data_layout: DataLayout {
                input_shape: vec![100, 100],
                output_shape: vec![100, 100],
                memory_layout: MemoryLayout::RowMajor,
            },
            precision: Precision::FP32,
            optimization_hints: vec![],
            performance_model: PerformanceModel {
                compute_intensity: 10.0,
                memory_bandwidth: 500.0,
                expected_occupancy: 1.5, // Invalid: > 1.0
            },
        };

        let constraints = GoalConstraints::default();
        let result = interpreter.validate_specification(&spec, &constraints);
        assert!(result.is_err());
        assert!(result
            .expect_err("Expected error")
            .to_string()
            .contains("Invalid occupancy"));
    }

    #[tokio::test]
    async fn test_context_caching() -> Result<(), Box<dyn std::error::Error>> {
        let interpreter = GoalInterpreter::new(InterpreterConfig::default());

        let goal_id = Uuid::new_v4();
        let context = InterpretationContext {
            historical_kernels: vec![],
            device_capabilities: HashMap::new(),
            previous_interpretations: vec![],
        };

        interpreter.cache_context(goal_id, context);

        let cache = interpreter.context_cache.read();
        assert!(cache.contains_key(&goal_id));
    }

    #[tokio::test]
    async fn test_cache_size_limit() -> Result<(), Box<dyn std::error::Error>> {
        let interpreter = GoalInterpreter::new(InterpreterConfig::default());

        // Add more than 1000 contexts
        for i in 0..1100 {
            let goal_id = Uuid::new_v4();
            let context = InterpretationContext {
                historical_kernels: vec![],
                device_capabilities: HashMap::new(),
                previous_interpretations: vec![(
                    format!("test_{i}"),
                    KernelSpecification {
                        operation_type: OperationType::Custom,
                        data_layout: DataLayout {
                            input_shape: vec![i],
                            output_shape: vec![i],
                            memory_layout: MemoryLayout::RowMajor,
                        },
                        precision: Precision::FP32,
                        optimization_hints: vec![],
                        performance_model: PerformanceModel {
                            compute_intensity: 1.0,
                            memory_bandwidth: 100.0,
                            expected_occupancy: 0.5,
                        },
                    },
                )],
            };
            interpreter.cache_context(goal_id, context);
        }

        let cache = interpreter.context_cache.read();
        assert!(cache.len() <= 1000); // Should have removed oldest entries
    }

    #[tokio::test]
    async fn test_elementwise_operation_detection() -> Result<(), Box<dyn std::error::Error>> {
        let config = InterpreterConfig {
            use_mock: true,
            ..Default::default()
        };

        let interpreter = GoalInterpreter::new(config);

        let goal = Goal::new(
            "Apply ReLU activation to tensor".to_string(),
            GoalPriority::Normal,
        );

        let spec = interpreter.interpret(&goal).await?;
        assert_eq!(spec.operation_type, OperationType::Elementwise);
    }

    #[test]
    fn test_examples_formatting() -> Result<(), Box<dyn std::error::Error>> {
        let interpreter = GoalInterpreter::new(InterpreterConfig::default());
        let examples = interpreter.get_examples();

        assert!(examples.contains("matrix_multiply"));
        assert!(examples.contains("tensor_core"));
        assert!(examples.contains("fp16"));

        // Verify it's valid JSON by parsing part of it
        assert!(examples.contains(r#""operation_type": "matrix_multiply""#));
    }

    #[test]
    fn test_large_tensor_memory_estimation() -> Result<(), Box<dyn std::error::Error>> {
        let interpreter = GoalInterpreter::new(InterpreterConfig::default());

        let spec = KernelSpecification {
            operation_type: OperationType::MatrixMultiply,
            data_layout: DataLayout {
                input_shape: vec![4096, 4096, 2], // Two large matrices
                output_shape: vec![4096, 4096],
                memory_layout: MemoryLayout::RowMajor,
            },
            precision: Precision::FP16,
            optimization_hints: vec![OptimizationHint::TensorCore],
            performance_model: PerformanceModel {
                compute_intensity: 40.0,
                memory_bandwidth: 900.0,
                expected_occupancy: 0.85,
            },
        };

        let memory = interpreter.estimate_memory_usage(&spec);
        let expected = (4096 * 4096 * 2 + 4096 * 4096) * 2; // FP16 = 2 bytes
        assert_eq!(memory, expected);
    }

    #[test]
    fn test_edge_case_shapes() -> Result<(), Box<dyn std::error::Error>> {
        let layout = DataLayout {
            input_shape: vec![],   // Empty shape
            output_shape: vec![1], // Scalar output
            memory_layout: MemoryLayout::RowMajor,
        };

        let input_elements: usize = layout.input_shape.iter().product();
        let output_elements: usize = layout.output_shape.iter().product();

        assert_eq!(input_elements, 1); // Empty product is 1
        assert_eq!(output_elements, 1);
    }

    #[tokio::test]
    async fn test_concurrent_interpretations() -> Result<(), Box<dyn std::error::Error>> {
        use tokio::task;

        let config = InterpreterConfig {
            use_mock: true,
            ..Default::default()
        };

        let interpreter = std::sync::Arc::new(GoalInterpreter::new(config));
        let mut handles = vec![];

        for i in 0..10 {
            let interpreter_clone = interpreter.clone();
            let handle = task::spawn(async move {
                let goal = Goal::new(format!("Optimize kernel {i}"), GoalPriority::Normal);
                interpreter_clone.interpret(&goal).await
            });
            handles.push(handle);
        }

        for handle in handles {
            let result = handle.await?;
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_constraint_validation_edge_cases() -> Result<(), Box<dyn std::error::Error>> {
        let interpreter = GoalInterpreter::new(InterpreterConfig::default());

        let spec = KernelSpecification {
            operation_type: OperationType::Reduction,
            data_layout: DataLayout {
                input_shape: vec![1],
                output_shape: vec![1],
                memory_layout: MemoryLayout::RowMajor,
            },
            precision: Precision::INT8,
            optimization_hints: vec![],
            performance_model: PerformanceModel {
                compute_intensity: 0.1,
                memory_bandwidth: 10.0,
                expected_occupancy: 0.0, // Edge case: zero occupancy
            },
        };

        let constraints = GoalConstraints::default();
        let result = interpreter.validate_specification(&spec, &constraints);
        assert!(result.is_ok()); // Zero occupancy is valid
    }

    #[test]
    fn test_all_operation_types() -> Result<(), Box<dyn std::error::Error>> {
        let operations = vec![
            OperationType::MatrixMultiply,
            OperationType::Reduction,
            OperationType::Convolution,
            OperationType::Elementwise,
            OperationType::Custom,
        ];

        for op in operations {
            let spec = KernelSpecification {
                operation_type: op,
                data_layout: DataLayout {
                    input_shape: vec![10],
                    output_shape: vec![10],
                    memory_layout: MemoryLayout::RowMajor,
                },
                precision: Precision::FP32,
                optimization_hints: vec![],
                performance_model: PerformanceModel {
                    compute_intensity: 1.0,
                    memory_bandwidth: 100.0,
                    expected_occupancy: 0.5,
                },
            };

            let json = serde_json::to_string(&spec)?;
            let parsed: KernelSpecification = serde_json::from_str(&json)?;
            assert_eq!(parsed.operation_type, op);
        }
    }
}
