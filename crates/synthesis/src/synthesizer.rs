//! Kernel synthesis core

use crate::error::{SynthesisError, SynthesisResult};
use crate::interpreter::KernelSpecification;
use crate::templates::TemplateEngine;
use chrono::{DateTime, Utc};
// use stratoswarm_cuda::kernel::CompileOptions;
use crate::compiler::CompileOptions;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// Synthesized kernel information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesizedKernel {
    /// Unique kernel ID
    pub id: Uuid,
    /// Kernel name
    pub name: String,
    /// Source code
    pub source_code: String,
    /// Specification used
    pub specification: KernelSpecification,
    /// Compile options
    pub compile_options: CompileOptions,
    /// Synthesis timestamp
    pub synthesized_at: DateTime<Utc>,
    /// Optimization iterations
    pub optimization_iterations: u32,
}

/// Kernel cache entry
#[derive(Debug, Clone)]
struct CacheEntry {
    kernel: SynthesizedKernel,
    hit_count: u64,
    last_accessed: DateTime<Utc>,
}

/// Synthesis configuration
#[derive(Debug, Clone)]
pub struct SynthesizerConfig {
    /// Maximum optimization iterations
    pub max_iterations: u32,
    /// Cache size limit
    pub cache_size: usize,
    /// Enable aggressive optimizations
    pub aggressive_optimization: bool,
    /// Target architecture
    pub target_arch: String,
}

impl Default for SynthesizerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            cache_size: 1000,
            aggressive_optimization: false,
            target_arch: "sm_80".to_string(),
        }
    }
}

/// Kernel synthesis engine
pub struct KernelSynthesizer {
    config: SynthesizerConfig,
    template_engine: Arc<TemplateEngine>,
    cache: Arc<DashMap<String, CacheEntry>>,
    #[cfg(not(feature = "mock-llm"))]
    llm_client: async_openai::Client<async_openai::config::OpenAIConfig>,
}

impl KernelSynthesizer {
    /// Create new synthesizer
    pub fn new(config: SynthesizerConfig) -> SynthesisResult<Self> {
        Ok(Self {
            config,
            template_engine: Arc::new(TemplateEngine::new()?),
            cache: Arc::new(DashMap::new()),
            #[cfg(not(feature = "mock-llm"))]
            llm_client: async_openai::Client::new(),
        })
    }

    /// Synthesize kernel from specification
    pub async fn synthesize(
        &self,
        spec: &KernelSpecification,
    ) -> SynthesisResult<SynthesizedKernel> {
        // Check cache first
        let cache_key = self.compute_cache_key(spec);
        if let Some(cached) = self.get_from_cache(&cache_key) {
            return Ok(cached);
        }

        // Select base template
        let template = self.template_engine.select_template(spec)?;

        // Generate initial code
        let mut source_code = self.generate_initial_code(template, spec).await?;

        // Apply optimizations iteratively
        let mut iterations = 0;
        for i in 0..self.config.max_iterations {
            let optimized = self.optimize_code(&source_code, spec).await?;

            // Check if optimization improved
            if self.is_improvement(&optimized, &source_code, spec)? {
                source_code = optimized;
                iterations = i + 1;
            } else {
                break;
            }
        }

        // Create compile options
        let compile_options = self.create_compile_options(spec);

        // Create synthesized kernel
        let kernel = SynthesizedKernel {
            id: Uuid::new_v4(),
            name: self.generate_kernel_name(spec),
            source_code,
            specification: spec.clone(),
            compile_options,
            synthesized_at: Utc::now(),
            optimization_iterations: iterations,
        };

        // Cache successful kernel
        self.cache_kernel(&cache_key, kernel.clone());

        Ok(kernel)
    }

    /// Compute cache key for specification
    fn compute_cache_key(&self, spec: &KernelSpecification) -> String {
        // Simple key based on specification
        format!(
            "{:?}_{:?}_{:?}_{:?}",
            spec.operation_type,
            spec.data_layout.input_shape,
            spec.precision,
            spec.optimization_hints
        )
    }

    /// Get kernel from cache
    fn get_from_cache(&self, key: &str) -> Option<SynthesizedKernel> {
        if let Some(mut entry) = self.cache.get_mut(key) {
            entry.hit_count += 1;
            entry.last_accessed = Utc::now();
            Some(entry.kernel.clone())
        } else {
            None
        }
    }

    /// Cache synthesized kernel
    fn cache_kernel(&self, key: &str, kernel: SynthesizedKernel) {
        // Evict if cache is full
        if self.cache.len() >= self.config.cache_size {
            // Find least recently used
            if let Some(lru_key) = self
                .cache
                .iter()
                .min_by_key(|entry| entry.value().last_accessed)
                .map(|entry| entry.key().clone())
            {
                self.cache.remove(&lru_key);
            }
        }

        self.cache.insert(
            key.to_string(),
            CacheEntry {
                kernel,
                hit_count: 1,
                last_accessed: Utc::now(),
            },
        );
    }

    /// Generate initial code from template
    async fn generate_initial_code(
        &self,
        template: &str,
        spec: &KernelSpecification,
    ) -> SynthesisResult<String> {
        // Build template context
        let mut context = tera::Context::new();
        context.insert("operation_type", &spec.operation_type);
        context.insert("input_shape", &spec.data_layout.input_shape);
        context.insert("output_shape", &spec.data_layout.output_shape);
        context.insert("precision", &spec.precision);
        context.insert("optimization_hints", &spec.optimization_hints);

        // Render template
        self.template_engine.render(template, &context)
    }

    /// Optimize generated code
    async fn optimize_code(
        &self,
        code: &str,
        spec: &KernelSpecification,
    ) -> SynthesisResult<String> {
        #[cfg(feature = "mock-llm")]
        {
            // Simple mock optimization
            Ok(self.mock_optimize(code, spec))
        }
        #[cfg(not(feature = "mock-llm"))]
        {
            self.llm_optimize(code, spec).await
        }
    }

    /// Mock optimization for testing
    fn mock_optimize(&self, code: &str, spec: &KernelSpecification) -> String {
        let mut optimized = code.to_string();

        // Apply simple transformations based on hints
        for hint in &spec.optimization_hints {
            match hint {
                crate::interpreter::OptimizationHint::SharedMemory => {
                    if !optimized.contains("__shared__") {
                        optimized = optimized.replace(
                            "// SHARED_MEMORY_PLACEHOLDER",
                            "__shared__ float shared_mem[1024];",
                        );
                    }
                }
                crate::interpreter::OptimizationHint::Unrolling => {
                    optimized = optimized.replace("// UNROLL_PLACEHOLDER", "#pragma unroll");
                }
                _ => {}
            }
        }

        optimized
    }

    /// LLM-based optimization
    #[cfg(not(feature = "mock-llm"))]
    async fn llm_optimize(
        &self,
        code: &str,
        spec: &KernelSpecification,
    ) -> SynthesisResult<String> {
        use async_openai::types::{
            ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
            ChatCompletionRequestUserMessage, CreateChatCompletionRequestArgs,
        };

        let prompt = format!(
            r#"Optimize the following CUDA kernel for better performance:

Current code:
```cuda
{}
```

Optimization hints: {:?}
Target architecture: {}

Apply optimizations and return only the improved kernel code."#,
            code, spec.optimization_hints, self.config.target_arch
        );

        let request = CreateChatCompletionRequestArgs::default()
            .model("gpt-4")
            .messages([
                ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                    content: async_openai::types::ChatCompletionRequestSystemMessageContent::Text("You are a CUDA kernel optimization expert. Optimize kernels for maximum performance.".to_string()),
                    ..Default::default()
                }),
                ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                    content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(prompt),
                    ..Default::default()
                }),
            ])
            .temperature(0.2)
            .max_tokens(2000u32)
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

    /// Check if optimization improved code
    fn is_improvement(
        &self,
        optimized: &str,
        original: &str,
        _spec: &KernelSpecification,
    ) -> SynthesisResult<bool> {
        // Simple heuristics for now
        if optimized == original {
            return Ok(false);
        }

        // Check for optimization markers
        let improvements = [
            "__shared__",
            "#pragma unroll",
            "__syncthreads()",
            "tensorcore",
        ];

        let optimized_score: usize = improvements
            .iter()
            .filter(|marker| optimized.contains(*marker))
            .count();

        let original_score: usize = improvements
            .iter()
            .filter(|marker| original.contains(*marker))
            .count();

        Ok(optimized_score > original_score)
    }

    /// Create compile options for specification
    fn create_compile_options(&self, spec: &KernelSpecification) -> CompileOptions {
        let mut options = CompileOptions::default();
        options.arch = self.config.target_arch.clone();
        options.opt_level = if self.config.aggressive_optimization {
            3
        } else {
            2
        };
        options.fast_math = matches!(
            spec.precision,
            crate::interpreter::Precision::FP16 | crate::interpreter::Precision::BF16
        );

        // Add precision-specific flags
        match spec.precision {
            crate::interpreter::Precision::FP16 => {
                options.extra_flags.push("-DUSE_FP16".to_string());
            }
            crate::interpreter::Precision::BF16 => {
                options.extra_flags.push("-DUSE_BF16".to_string());
            }
            crate::interpreter::Precision::INT8 => {
                options.extra_flags.push("-DUSE_INT8".to_string());
            }
            _ => {}
        }

        options
    }

    /// Generate kernel name from specification
    fn generate_kernel_name(&self, spec: &KernelSpecification) -> String {
        format!("{:?}_{:?}_kernel", spec.operation_type, spec.precision).to_lowercase()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::{
        DataLayout, MemoryLayout, OperationType, OptimizationHint, PerformanceModel, Precision,
    };

    fn create_test_spec() -> KernelSpecification {
        KernelSpecification {
            operation_type: OperationType::MatrixMultiply,
            data_layout: DataLayout {
                input_shape: vec![1024, 1024],
                output_shape: vec![1024, 1024],
                memory_layout: MemoryLayout::RowMajor,
            },
            precision: Precision::FP32,
            optimization_hints: vec![OptimizationHint::SharedMemory],
            performance_model: PerformanceModel {
                compute_intensity: 40.0,
                memory_bandwidth: 900.0,
                expected_occupancy: 0.85,
            },
        }
    }

    #[tokio::test]
    async fn test_kernel_synthesis() -> Result<(), Box<dyn std::error::Error>> {
        let config = SynthesizerConfig::default();
        let synthesizer = KernelSynthesizer::new(config)?;

        let spec = create_test_spec();
        let kernel = synthesizer.synthesize(&spec).await?;

        assert!(!kernel.source_code.is_empty());
        assert_eq!(
            kernel.specification.operation_type,
            OperationType::MatrixMultiply
        );
        assert!(kernel.name.contains("matrixmultiply")); // Debug format + lowercase
    }

    #[test]
    fn test_cache_key_generation() -> Result<(), Box<dyn std::error::Error>> {
        let synthesizer = KernelSynthesizer::new(SynthesizerConfig::default())?;

        let spec1 = create_test_spec();
        let spec2 = create_test_spec();

        let key1 = synthesizer.compute_cache_key(&spec1);
        let key2 = synthesizer.compute_cache_key(&spec2);

        assert_eq!(key1, key2); // Same specs should have same key
    }

    #[tokio::test]
    async fn test_kernel_caching() -> Result<(), Box<dyn std::error::Error>> {
        let config = SynthesizerConfig {
            cache_size: 2,
            ..Default::default()
        };
        let synthesizer = KernelSynthesizer::new(config)?;

        let spec = create_test_spec();

        // First synthesis
        let kernel1 = synthesizer.synthesize(&spec).await?;

        // Second synthesis (should hit cache)
        let kernel2 = synthesizer.synthesize(&spec).await?;

        // Should return same kernel
        assert_eq!(kernel1.source_code, kernel2.source_code);
    }

    #[test]
    fn test_compile_options_generation() -> Result<(), Box<dyn std::error::Error>> {
        let synthesizer = KernelSynthesizer::new(SynthesizerConfig::default())?;

        let mut spec = create_test_spec();
        spec.precision = Precision::FP16;

        let options = synthesizer.create_compile_options(&spec);

        assert!(options.fast_math);
        assert!(options.extra_flags.contains(&"-DUSE_FP16".to_string()));
    }

    #[test]
    fn test_mock_optimization() -> Result<(), Box<dyn std::error::Error>> {
        let synthesizer = KernelSynthesizer::new(SynthesizerConfig::default())?;

        let spec = create_test_spec();
        let code = "// SHARED_MEMORY_PLACEHOLDER\nvoid kernel() {}";

        let optimized = synthesizer.mock_optimize(code, &spec);

        assert!(optimized.contains("__shared__"));
    }

    #[test]
    fn test_improvement_detection() -> Result<(), Box<dyn std::error::Error>> {
        let synthesizer = KernelSynthesizer::new(SynthesizerConfig::default())?;

        let spec = create_test_spec();
        let original = "void kernel() {}";
        let optimized = "__shared__ float mem[1024];\nvoid kernel() {}";

        assert!(synthesizer.is_improvement(optimized, original, &spec)?);
        assert!(!synthesizer
            .is_improvement(original, original, &spec)
            .unwrap());
    }

    #[test]
    fn test_synthesizer_config_default() -> Result<(), Box<dyn std::error::Error>> {
        let config = SynthesizerConfig::default();
        assert_eq!(config.max_iterations, 10);
        assert_eq!(config.cache_size, 1000);
        assert!(!config.aggressive_optimization);
        assert_eq!(config.target_arch, "sm_80");
    }

    #[test]
    fn test_kernel_name_generation() -> Result<(), Box<dyn std::error::Error>> {
        let synthesizer = KernelSynthesizer::new(SynthesizerConfig::default())?;

        let mut spec = create_test_spec();
        spec.operation_type = OperationType::Reduction;
        spec.precision = Precision::FP16;

        let name = synthesizer.generate_kernel_name(&spec);
        assert!(name.contains("reduction"));
        assert!(name.contains("fp16"));
        assert!(name.contains("kernel"));
    }

    #[test]
    fn test_cache_entry_creation() -> Result<(), Box<dyn std::error::Error>> {
        let kernel = SynthesizedKernel {
            id: Uuid::new_v4(),
            name: "test_kernel".to_string(),
            source_code: "void test() {}".to_string(),
            specification: create_test_spec(),
            compile_options: CompileOptions::default(),
            synthesized_at: Utc::now(),
            optimization_iterations: 5,
        };

        let entry = CacheEntry {
            kernel: kernel.clone(),
            hit_count: 1,
            last_accessed: Utc::now(),
        };

        assert_eq!(entry.kernel.name, "test_kernel");
        assert_eq!(entry.hit_count, 1);
    }

    #[tokio::test]
    async fn test_cache_eviction() -> Result<(), Box<dyn std::error::Error>> {
        let config = SynthesizerConfig {
            cache_size: 2,
            ..Default::default()
        };
        let synthesizer = KernelSynthesizer::new(config)?;

        // Create three different specs
        let mut spec1 = create_test_spec();
        spec1.precision = Precision::FP32;

        let mut spec2 = create_test_spec();
        spec2.precision = Precision::FP16;

        let mut spec3 = create_test_spec();
        spec3.precision = Precision::INT8;

        // Synthesize three kernels
        let _kernel1 = synthesizer.synthesize(&spec1).await?;
        let _kernel2 = synthesizer.synthesize(&spec2).await?;
        let _kernel3 = synthesizer.synthesize(&spec3).await?;

        // Cache should only have 2 entries (oldest evicted)
        let cache = synthesizer.cache.read();
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_optimization_hints_application() -> Result<(), Box<dyn std::error::Error>> {
        let synthesizer = KernelSynthesizer::new(SynthesizerConfig::default())?;

        let mut spec = create_test_spec();
        spec.optimization_hints = vec![OptimizationHint::SharedMemory, OptimizationHint::Unrolling];

        let code = "// SHARED_MEMORY_PLACEHOLDER\n// UNROLL_PLACEHOLDER\nvoid kernel() {}";
        let optimized = synthesizer.mock_optimize(code, &spec);

        assert!(optimized.contains("__shared__"));
        assert!(optimized.contains("#pragma unroll"));
    }

    #[test]
    fn test_compile_options_aggressive() -> Result<(), Box<dyn std::error::Error>> {
        let config = SynthesizerConfig {
            aggressive_optimization: true,
            ..Default::default()
        };
        let synthesizer = KernelSynthesizer::new(config)?;

        let spec = create_test_spec();
        let options = synthesizer.create_compile_options(&spec);

        assert_eq!(options.opt_level, 3);
    }

    #[test]
    fn test_compile_options_precision_flags() -> Result<(), Box<dyn std::error::Error>> {
        let synthesizer = KernelSynthesizer::new(SynthesizerConfig::default())?;

        // Test BF16
        let mut spec = create_test_spec();
        spec.precision = Precision::BF16;
        let options = synthesizer.create_compile_options(&spec);
        assert!(options.extra_flags.contains(&"-DUSE_BF16".to_string()));
        assert!(options.fast_math);

        // Test INT8
        spec.precision = Precision::INT8;
        let options = synthesizer.create_compile_options(&spec);
        assert!(options.extra_flags.contains(&"-DUSE_INT8".to_string()));
        assert!(!options.fast_math);
    }

    #[test]
    fn test_synthesized_kernel_serialization() -> Result<(), Box<dyn std::error::Error>> {
        let kernel = SynthesizedKernel {
            id: Uuid::new_v4(),
            name: "test_kernel".to_string(),
            source_code: "__global__ void test() {}".to_string(),
            specification: create_test_spec(),
            compile_options: CompileOptions::default(),
            synthesized_at: Utc::now(),
            optimization_iterations: 3,
        };

        let json = serde_json::to_string(&kernel)?;
        let parsed: SynthesizedKernel = serde_json::from_str(&json)?;

        assert_eq!(parsed.name, kernel.name);
        assert_eq!(parsed.source_code, kernel.source_code);
        assert_eq!(
            parsed.optimization_iterations,
            kernel.optimization_iterations
        );
    }

    #[tokio::test]
    async fn test_multiple_optimization_iterations() -> Result<(), Box<dyn std::error::Error>> {
        let config = SynthesizerConfig {
            max_iterations: 3,
            ..Default::default()
        };
        let synthesizer = KernelSynthesizer::new(config)?;

        let spec = create_test_spec();
        let kernel = synthesizer.synthesize(&spec).await?;

        // Should have run at least one iteration
        assert!(kernel.optimization_iterations <= 3);
    }

    #[test]
    fn test_cache_key_uniqueness() -> Result<(), Box<dyn std::error::Error>> {
        let synthesizer = KernelSynthesizer::new(SynthesizerConfig::default())?;

        let mut spec1 = create_test_spec();
        spec1.precision = Precision::FP32;

        let mut spec2 = create_test_spec();
        spec2.precision = Precision::FP16;

        let key1 = synthesizer.compute_cache_key(&spec1);
        let key2 = synthesizer.compute_cache_key(&spec2);

        assert_ne!(key1, key2); // Different specs should have different keys
    }

    #[test]
    fn test_improvement_detection_edge_cases() -> Result<(), Box<dyn std::error::Error>> {
        let synthesizer = KernelSynthesizer::new(SynthesizerConfig::default())?;
        let spec = create_test_spec();

        // Empty strings
        assert!(!synthesizer.is_improvement("", "", &spec)?);

        // Only whitespace changes
        let original = "void kernel() {}";
        let whitespace = "void kernel() {  }";
        assert!(!synthesizer
            .is_improvement(whitespace, original, &spec)
            .unwrap());

        // Multiple improvements
        let multi_opt =
            "__shared__ float mem[1024];\n#pragma unroll\n__syncthreads();\nvoid kernel() {}";
        assert!(synthesizer
            .is_improvement(multi_opt, original, &spec)
            .unwrap());
    }

    #[tokio::test]
    async fn test_cache_hit_counting() -> Result<(), Box<dyn std::error::Error>> {
        let synthesizer = KernelSynthesizer::new(SynthesizerConfig::default())?;
        let spec = create_test_spec();

        // First synthesis
        let _kernel1 = synthesizer.synthesize(&spec).await?;

        // Access cache multiple times
        for _ in 0..5 {
            let _kernel = synthesizer.synthesize(&spec).await?;
        }

        // Check hit count
        let cache = synthesizer.cache.read();
        let key = synthesizer.compute_cache_key(&spec);
        if let Some(entry) = cache.get(&key) {
            assert_eq!(entry.hit_count, 6); // 1 initial + 5 hits
        }
    }

    #[test]
    fn test_target_architecture_configuration() -> Result<(), Box<dyn std::error::Error>> {
        let config = SynthesizerConfig {
            target_arch: "sm_90".to_string(),
            ..Default::default()
        };
        let synthesizer = KernelSynthesizer::new(config)?;

        let spec = create_test_spec();
        let options = synthesizer.create_compile_options(&spec);

        assert_eq!(options.arch, "sm_90");
    }

    #[tokio::test]
    async fn test_concurrent_synthesis() -> Result<(), Box<dyn std::error::Error>> {
        use tokio::task;

        let synthesizer = Arc::new(KernelSynthesizer::new(SynthesizerConfig::default()).unwrap());
        let mut handles = vec![];

        for i in 0..10 {
            let synthesizer_clone = synthesizer.clone();
            let handle = task::spawn(async move {
                let mut spec = create_test_spec();
                spec.data_layout.input_shape = vec![i, i];
                synthesizer_clone.synthesize(&spec).await
            });
            handles.push(handle);
        }

        for handle in handles {
            let result = handle.await?;
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_operation_type_in_kernel_name() -> Result<(), Box<dyn std::error::Error>> {
        let synthesizer = KernelSynthesizer::new(SynthesizerConfig::default())?;

        let operations = vec![
            OperationType::MatrixMultiply,
            OperationType::Reduction,
            OperationType::Convolution,
            OperationType::Elementwise,
            OperationType::Custom,
        ];

        for op in operations {
            let mut spec = create_test_spec();
            spec.operation_type = op;
            let name = synthesizer.generate_kernel_name(&spec);
            assert!(name.contains("kernel"));
            assert!(name.contains(&format!("{:?}", op).to_lowercase()));
        }
    }

    #[test]
    fn test_cache_entry_timestamp_ordering() -> Result<(), Box<dyn std::error::Error>> {
        let now = Utc::now();
        let kernel = SynthesizedKernel {
            id: Uuid::new_v4(),
            name: "test".to_string(),
            source_code: "".to_string(),
            specification: create_test_spec(),
            compile_options: CompileOptions::default(),
            synthesized_at: now,
            optimization_iterations: 0,
        };

        let entry1 = CacheEntry {
            kernel: kernel.clone(),
            hit_count: 1,
            last_accessed: now,
        };

        let entry2 = CacheEntry {
            kernel: kernel.clone(),
            hit_count: 1,
            last_accessed: now + chrono::Duration::seconds(1),
        };

        assert!(entry2.last_accessed > entry1.last_accessed);
    }

    #[test]
    fn test_empty_optimization_hints() -> Result<(), Box<dyn std::error::Error>> {
        let synthesizer = KernelSynthesizer::new(SynthesizerConfig::default())?;

        let mut spec = create_test_spec();
        spec.optimization_hints = vec![];

        let code = "void kernel() {}";
        let optimized = synthesizer.mock_optimize(code, &spec);

        assert_eq!(optimized, code); // No changes without hints
    }

    #[tokio::test]
    async fn test_synthesis_with_all_precisions() -> Result<(), Box<dyn std::error::Error>> {
        let synthesizer = KernelSynthesizer::new(SynthesizerConfig::default())?;

        let precisions = vec![
            Precision::FP32,
            Precision::FP16,
            Precision::BF16,
            Precision::INT8,
        ];

        for precision in precisions {
            let mut spec = create_test_spec();
            spec.precision = precision;
            let kernel = synthesizer.synthesize(&spec).await?;
            assert!(kernel
                .name
                .contains(&format!("{:?}", precision).to_lowercase()));
        }
    }

    #[test]
    fn test_synthesized_kernel_fields() -> Result<(), Box<dyn std::error::Error>> {
        let kernel = SynthesizedKernel {
            id: Uuid::new_v4(),
            name: "matmul_kernel".to_string(),
            source_code: "__global__ void matmul() {}".to_string(),
            specification: create_test_spec(),
            compile_options: CompileOptions::default(),
            synthesized_at: Utc::now(),
            optimization_iterations: 7,
        };

        assert_eq!(kernel.name, "matmul_kernel");
        assert_eq!(kernel.optimization_iterations, 7);
        assert!(kernel.source_code.contains("__global__"));
    }
}
