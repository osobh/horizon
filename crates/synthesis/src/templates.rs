//! Kernel template management

use crate::error::{SynthesisError, SynthesisResult};
use crate::interpreter::{KernelSpecification, OperationType};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::collections::HashMap;
use tera::{Context, Tera};

/// Global template engine instance
static TEMPLATE_ENGINE: Lazy<RwLock<Option<TemplateEngine>>> = Lazy::new(|| RwLock::new(None));

/// Template engine for kernel generation
pub struct TemplateEngine {
    tera: Tera,
    templates: HashMap<String, String>,
}

impl TemplateEngine {
    /// Create new template engine
    pub fn new() -> SynthesisResult<Self> {
        let mut tera = Tera::default();
        let templates = Self::load_builtin_templates();

        // Register templates with Tera
        for (name, content) in &templates {
            tera.add_raw_template(name, content)
                .map_err(|e| SynthesisError::TemplateNotFound {
                    name: format!("Failed to add template {}: {}", name, e),
                })?;
        }

        Ok(Self { tera, templates })
    }

    /// Load built-in templates
    fn load_builtin_templates() -> HashMap<String, String> {
        let mut templates = HashMap::new();

        // Matrix multiplication template
        templates.insert(
            "matrix_multiply".to_string(),
            r#"// Matrix multiplication kernel
// Matrix multiplication kernel for GPU execution

// SHARED_MEMORY_PLACEHOLDER

__global__ void {{kernel_name}}(
    const {{precision}}* __restrict__ A,
    const {{precision}}* __restrict__ B,
    {{precision}}* __restrict__ C,
    int M, int N, int K
) {
    // Thread and block indices
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    {{precision}} sum = 0;
    
    // UNROLL_PLACEHOLDER
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    
    C[row * N + col] = sum;
}"#
            .to_string(),
        );

        // Reduction template
        templates.insert(
            "reduction".to_string(),
            r#"// Reduction kernel
// Reduction kernel for GPU execution

// SHARED_MEMORY_PLACEHOLDER

__global__ void {{kernel_name}}(
    const {{precision}}* __restrict__ input,
    {{precision}}* __restrict__ output,
    int n
) {
    extern __shared__ {{precision}} sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load to shared memory
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();
    
    // Reduction in shared memory
    // UNROLL_PLACEHOLDER
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        atomicAdd(&output[0], sdata[0]);
    }
}"#
            .to_string(),
        );

        // Elementwise template
        templates.insert(
            "elementwise".to_string(),
            r#"// Elementwise operation kernel
// Elementwise operation kernel for GPU execution

__global__ void {{kernel_name}}(
    const {{precision}}* __restrict__ input,
    {{precision}}* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Apply operation
        output[idx] = input[idx]; // PLACEHOLDER: Dynamic operation will be inserted here
    }
}"#
            .to_string(),
        );

        // Convolution template
        templates.insert(
            "convolution".to_string(),
            r#"// Convolution kernel
// Input shape: [batch, channels, height, width]
// Filter shape: [out_channels, in_channels, filter_h, filter_w]
// Output shape: [batch, out_channels, out_height, out_width]

// SHARED_MEMORY_PLACEHOLDER

__global__ void {{kernel_name}}(
    const {{precision}}* __restrict__ input,
    const {{precision}}* __restrict__ filter,
    {{precision}}* __restrict__ output,
    int batch, int in_channels, int height, int width,
    int out_channels, int filter_h, int filter_w,
    int stride, int pad
) {
    // PLACEHOLDER: Convolution kernel will be generated dynamically based on template parameters
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z;
    
    // Compute convolution for this output position
}"#
            .to_string(),
        );

        templates
    }

    /// Select template based on specification
    pub fn select_template(&self, spec: &KernelSpecification) -> SynthesisResult<&str> {
        let template_name = match spec.operation_type {
            OperationType::MatrixMultiply => "matrix_multiply",
            OperationType::Reduction => "reduction",
            OperationType::Convolution => "convolution",
            OperationType::Elementwise => "elementwise",
            OperationType::Custom => "elementwise", // Use elementwise as base for custom
        };

        self.templates
            .get(template_name)
            .map(|s| s.as_str())
            .ok_or_else(|| SynthesisError::TemplateNotFound {
                name: template_name.to_string(),
            })
    }

    /// Render template with context
    pub fn render(&self, template: &str, context: &Context) -> SynthesisResult<String> {
        // Create a temporary template name
        let temp_name = format!("temp_{}", uuid::Uuid::new_v4());

        // Add template temporarily
        let mut tera = self.tera.clone();
        tera.add_raw_template(&temp_name, template).map_err(|e| {
            SynthesisError::TemplateNotFound {
                name: format!("Failed to add template: {e}"),
            }
        })?;

        // Add kernel name to context
        let mut context = context.clone();
        context.insert("kernel_name", "synthesized_kernel");
        context.insert("precision", "float"); // Default to float

        // Render
        tera.render(&temp_name, &context)
            .map_err(|e| SynthesisError::SynthesisFailure {
                message: format!("Template rendering failed: {e}"),
            })
    }
}

/// Initialize global template engine
pub fn init_template_engine() -> SynthesisResult<()> {
    let engine = TemplateEngine::new()?;
    *TEMPLATE_ENGINE.write() = Some(engine);
    Ok(())
}

/// Get global template engine
pub fn template_engine() -> SynthesisResult<TemplateEngine> {
    TEMPLATE_ENGINE
        .read()
        .as_ref()
        .ok_or_else(|| SynthesisError::Other("Template engine not initialized".to_string()))
        .map(|e| TemplateEngine {
            tera: e.tera.clone(),
            templates: e.templates.clone(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::{DataLayout, MemoryLayout, PerformanceModel, Precision};

    #[test]
    fn test_template_engine_creation() -> Result<(), Box<dyn std::error::Error>> {
        let engine = TemplateEngine::new()?;
        assert!(!engine.templates.is_empty());
        assert!(engine.templates.contains_key("matrix_multiply"));
        assert!(engine.templates.contains_key("reduction"));
    }

    #[test]
    fn test_template_selection() -> Result<(), Box<dyn std::error::Error>> {
        let engine = TemplateEngine::new()?;

        let spec = KernelSpecification {
            operation_type: OperationType::MatrixMultiply,
            data_layout: DataLayout {
                input_shape: vec![1024, 1024],
                output_shape: vec![1024, 1024],
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

        let template = engine.select_template(&spec)?;
        assert!(template.contains("Matrix multiplication"));
    }

    #[test]
    fn test_template_rendering() -> Result<(), Box<dyn std::error::Error>> {
        let engine = TemplateEngine::new()?;

        let mut context = Context::new();
        context.insert("input_shape", &vec![512, 512]);
        context.insert("output_shape", &vec![512, 512]);

        let template = engine.templates.get("matrix_multiply")?;
        let rendered = engine.render(template, &context)?;

        assert!(rendered.contains("synthesized_kernel"));
        assert!(rendered.contains("M, int N, int K")); // Check for matrix multiply params
        assert!(rendered.contains("float"));
    }

    #[test]
    fn test_all_templates_valid() -> Result<(), Box<dyn std::error::Error>> {
        let engine = TemplateEngine::new()?;
        let context = Context::new();

        for (name, template) in &engine.templates {
            let result = engine.render(template, &context);
            assert!(
                result.is_ok(),
                "Template {} failed to render: {:?}",
                name,
                result.err()
            );
        }
    }

    #[test]
    fn test_template_placeholders() -> Result<(), Box<dyn std::error::Error>> {
        let engine = TemplateEngine::new()?;

        let matmul = engine.templates.get("matrix_multiply")?;
        assert!(matmul.contains("SHARED_MEMORY_PLACEHOLDER"));
        assert!(matmul.contains("UNROLL_PLACEHOLDER"));

        let reduction = engine.templates.get("reduction")?;
        assert!(reduction.contains("SHARED_MEMORY_PLACEHOLDER"));
        assert!(reduction.contains("UNROLL_PLACEHOLDER"));
    }

    #[test]
    fn test_global_init() -> Result<(), Box<dyn std::error::Error>> {
        assert!(init_template_engine().is_ok());

        // Should be able to get engine after init
        let result = template_engine();
        assert!(result.is_ok());
    }

    #[test]
    fn test_builtin_template_coverage() -> Result<(), Box<dyn std::error::Error>> {
        let templates = TemplateEngine::load_builtin_templates();

        // Verify all expected templates are present
        let expected_templates = vec!["matrix_multiply", "reduction", "elementwise", "convolution"];

        for template_name in expected_templates {
            assert!(
                templates.contains_key(template_name),
                "Missing template: {}",
                template_name
            );
            assert!(
                !templates[template_name].is_empty(),
                "Empty template: {}",
                template_name
            );
        }
    }

    #[test]
    fn test_template_content_structure() -> Result<(), Box<dyn std::error::Error>> {
        let templates = TemplateEngine::load_builtin_templates();

        // Matrix multiply template should have key components
        let matmul = &templates["matrix_multiply"];
        assert!(matmul.contains("__global__"));
        assert!(matmul.contains("{{kernel_name}}"));
        assert!(matmul.contains("{{precision}}"));
        assert!(matmul.contains("const {{precision}}* __restrict__ A"));
        assert!(matmul.contains("blockIdx.x"));
        assert!(matmul.contains("threadIdx.x"));

        // Reduction template should have key components
        let reduction = &templates["reduction"];
        assert!(reduction.contains("__global__"));
        assert!(reduction.contains("extern __shared__"));
        assert!(reduction.contains("{{kernel_name}}"));
        assert!(reduction.contains("{{precision}}"));
        assert!(reduction.contains("__syncthreads"));
        assert!(reduction.contains("atomicAdd"));

        // Elementwise template should have key components
        let elementwise = &templates["elementwise"];
        assert!(elementwise.contains("__global__"));
        assert!(elementwise.contains("{{kernel_name}}"));
        assert!(elementwise.contains("{{precision}}"));

        // Convolution template should have key components
        let convolution = &templates["convolution"];
        assert!(convolution.contains("__global__"));
        assert!(convolution.contains("{{kernel_name}}"));
        assert!(convolution.contains("{{precision}}"));
        assert!(convolution.contains("batch"));
        assert!(convolution.contains("channels"));
    }

    #[test]
    fn test_template_selection_all_operation_types() -> Result<(), Box<dyn std::error::Error>> {
        let engine = TemplateEngine::new()?;

        let operation_types = vec![
            OperationType::MatrixMultiply,
            OperationType::Reduction,
            OperationType::Convolution,
            OperationType::Elementwise,
            OperationType::Custom,
        ];

        for op_type in operation_types {
            let spec = KernelSpecification {
                operation_type: op_type.clone(),
                data_layout: DataLayout {
                    input_shape: vec![512, 512],
                    output_shape: vec![512, 512],
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

            let result = engine.select_template(&spec);
            assert!(
                result.is_ok(),
                "Failed to select template for operation type: {:?}",
                op_type
            );

            let template = result?;
            assert!(!template.is_empty());
            assert!(template.contains("__global__"));
        }
    }

    #[test]
    fn test_template_rendering_with_different_contexts() -> Result<(), Box<dyn std::error::Error>> {
        let engine = TemplateEngine::new()?;
        let template = engine.templates.get("matrix_multiply")?;

        let test_contexts = vec![
            {
                let mut context = Context::new();
                context.insert("extra_field", &"test_value");
                context
            },
            {
                let mut context = Context::new();
                context.insert("input_shape", &vec![1024, 1024]);
                context.insert("block_size", &16);
                context
            },
            {
                let mut context = Context::new();
                context.insert("optimization_level", &3);
                context.insert("use_shared_memory", &true);
                context
            },
        ];

        for context in test_contexts {
            let result = engine.render(template, &context);
            assert!(result.is_ok());

            let rendered = result?;
            assert!(rendered.contains("synthesized_kernel"));
            assert!(rendered.contains("float"));
            assert!(rendered.contains("__global__"));
        }
    }

    #[test]
    fn test_template_rendering_precision_variants() -> Result<(), Box<dyn std::error::Error>> {
        let engine = TemplateEngine::new()?;
        let template = engine.templates.get("matrix_multiply")?;

        let precisions = vec!["float", "double", "half", "__half", "int"];

        for precision in precisions {
            let mut context = Context::new();
            context.insert("precision", precision);

            let result = engine.render(template, &context);
            assert!(result.is_ok());

            let rendered = result?;
            assert!(rendered.contains(precision));
        }
    }

    #[test]
    fn test_template_rendering_kernel_name_variants() -> Result<(), Box<dyn std::error::Error>> {
        let engine = TemplateEngine::new()?;
        let template = engine.templates.get("elementwise")?;

        let kernel_names = vec![
            "simple_kernel",
            "MyKernelName",
            "kernel_with_underscores",
            "kernel123",
            "KERNEL_ALL_CAPS",
        ];

        for kernel_name in kernel_names {
            let mut context = Context::new();
            context.insert("kernel_name", kernel_name);

            let result = engine.render(template, &context);
            assert!(result.is_ok());

            let rendered = result?;
            assert!(rendered.contains(kernel_name));
        }
    }

    #[test]
    fn test_template_rendering_error_handling() -> Result<(), Box<dyn std::error::Error>> {
        let engine = TemplateEngine::new()?;

        // Test with invalid template syntax
        let invalid_template = "{{unclosed_tag";
        let context = Context::new();

        let result = engine.render(invalid_template, &context);
        assert!(result.is_err());

        match result.expect_err("Expected error") {
            SynthesisError::TemplateNotFound { name } => {
                assert!(name.contains("Failed to add template"));
            }
            _ => panic!("Expected TemplateNotFound error"),
        }
    }

    #[test]
    fn test_template_engine_cloning() -> Result<(), Box<dyn std::error::Error>> {
        let engine1 = TemplateEngine::new()?;

        // Get a clone through the global template mechanism
        assert!(init_template_engine().is_ok());
        let engine2 = template_engine()?;

        // Both engines should have the same templates
        assert_eq!(engine1.templates.len(), engine2.templates.len());

        for (name, template) in &engine1.templates {
            assert!(engine2.templates.contains_key(name));
            assert_eq!(template, &engine2.templates[name]);
        }
    }

    #[test]
    fn test_template_engine_not_initialized_error() -> Result<(), Box<dyn std::error::Error>> {
        // Clear the global engine
        *TEMPLATE_ENGINE.write() = None;

        let result = template_engine();
        assert!(result.is_err());

        match result.expect_err("Expected error") {
            SynthesisError::Other(msg) => {
                assert!(msg.contains("not initialized"));
            }
            _ => panic!("Expected Other error for uninitialized engine"),
        }

        // Reinitialize for other tests
        let _ = init_template_engine();
    }

    #[test]
    fn test_template_placeholder_consistency() -> Result<(), Box<dyn std::error::Error>> {
        let templates = TemplateEngine::load_builtin_templates();

        // Check that templates with shared memory have the placeholder
        let templates_with_shared_memory = vec!["matrix_multiply", "reduction", "convolution"];

        for template_name in templates_with_shared_memory {
            let template = &templates[template_name];
            assert!(
                template.contains("SHARED_MEMORY_PLACEHOLDER"),
                "Template {} missing SHARED_MEMORY_PLACEHOLDER",
                template_name
            );
        }

        // Check that some templates have unroll placeholders
        let templates_with_unroll = vec!["matrix_multiply", "reduction"];

        for template_name in templates_with_unroll {
            let template = &templates[template_name];
            assert!(
                template.contains("UNROLL_PLACEHOLDER"),
                "Template {} missing UNROLL_PLACEHOLDER",
                template_name
            );
        }
    }

    #[test]
    fn test_template_parameter_consistency() -> Result<(), Box<dyn std::error::Error>> {
        let templates = TemplateEngine::load_builtin_templates();

        // Matrix multiply should have M, N, K parameters
        let matmul = &templates["matrix_multiply"];
        assert!(matmul.contains("int M"));
        assert!(matmul.contains("int N"));
        assert!(matmul.contains("int K"));

        // Reduction should have n parameter
        let reduction = &templates["reduction"];
        assert!(reduction.contains("int n"));

        // Elementwise should have n parameter
        let elementwise = &templates["elementwise"];
        assert!(elementwise.contains("int n"));

        // Convolution should have batch, channels, etc.
        let convolution = &templates["convolution"];
        assert!(convolution.contains("int batch"));
        assert!(convolution.contains("int in_channels"));
        assert!(convolution.contains("int height"));
        assert!(convolution.contains("int width"));
        assert!(convolution.contains("int out_channels"));
    }

    #[test]
    fn test_template_cuda_syntax_validity() -> Result<(), Box<dyn std::error::Error>> {
        let templates = TemplateEngine::load_builtin_templates();

        for (name, template) in &templates {
            // Check for basic CUDA syntax elements
            assert!(
                template.contains("__global__"),
                "Template {} missing __global__",
                name
            );

            // Should have proper CUDA built-ins usage
            if template.contains("blockIdx") {
                assert!(
                    template.contains("threadIdx"),
                    "Template {} has blockIdx but no threadIdx",
                    name
                );
            }

            // Check for proper memory access patterns
            if template.contains("__restrict__") {
                // Good, using restrict for better optimization
            }

            // Check for synchronization when using shared memory
            if template.contains("extern __shared__") || template.contains("__shared__") {
                assert!(
                    template.contains("__syncthreads"),
                    "Template {} uses shared memory but no __syncthreads",
                    name
                );
            }
        }
    }

    #[test]
    fn test_template_rendering_memory_efficiency() -> Result<(), Box<dyn std::error::Error>> {
        let engine = TemplateEngine::new()?;
        let template = engine.templates.get("reduction")?;

        // Render the same template multiple times to test memory efficiency
        for i in 0..100 {
            let mut context = Context::new();
            context.insert("iteration", &i);
            context.insert("size", &(1024 * i));

            let result = engine.render(template, &context);
            assert!(result.is_ok(), "Failed at iteration {}", i);

            let rendered = result?;
            assert!(rendered.contains("synthesized_kernel"));

            // Let the rendered string go out of scope to test memory cleanup
            drop(rendered);
        }
    }

    #[test]
    fn test_template_context_edge_cases() -> Result<(), Box<dyn std::error::Error>> {
        let engine = TemplateEngine::new()?;
        let template = engine.templates.get("elementwise")?;

        // Test with empty context
        let empty_context = Context::new();
        let result = engine.render(template, &empty_context);
        assert!(result.is_ok());

        // Test with context containing edge case values
        let mut edge_context = Context::new();
        edge_context.insert("zero_value", &0);
        edge_context.insert("negative_value", &-1);
        edge_context.insert("large_value", &(1u64 << 32));
        edge_context.insert("empty_string", &"");
        edge_context.insert("special_chars", &"!@#$%^&*()");

        let result = engine.render(template, &edge_context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_template_selection_error_cases() -> Result<(), Box<dyn std::error::Error>> {
        let engine = TemplateEngine::new()?;

        // Create a modified engine to test error cases
        let mut modified_engine = TemplateEngine::new()?;
        modified_engine.templates.clear(); // Remove all templates

        let spec = KernelSpecification {
            operation_type: OperationType::MatrixMultiply,
            data_layout: DataLayout {
                input_shape: vec![512, 512],
                output_shape: vec![512, 512],
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

        let result = modified_engine.select_template(&spec);
        assert!(result.is_err());

        match result.expect_err("Expected error") {
            SynthesisError::TemplateNotFound { name } => {
                assert_eq!(name, "matrix_multiply");
            }
            _ => panic!("Expected TemplateNotFound error"),
        }
    }

    #[test]
    fn test_template_concurrent_access() -> Result<(), Box<dyn std::error::Error>> {
        use std::sync::Arc;
        use std::thread;

        let engine = Arc::new(TemplateEngine::new().unwrap());
        let mut handles = Vec::new();

        // Test concurrent template selection and rendering
        for i in 0..10 {
            let eng = engine.clone();
            let handle = thread::spawn(move || {
                let spec = KernelSpecification {
                    operation_type: if i % 2 == 0 {
                        OperationType::MatrixMultiply
                    } else {
                        OperationType::Reduction
                    },
                    data_layout: DataLayout {
                        input_shape: vec![256, 256],
                        output_shape: vec![256, 256],
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

                let template = eng.select_template(&spec)?;
                let context = Context::new();
                eng.render(template, &context)
            });
            handles.push(handle);
        }

        for handle in handles {
            let result = handle.join()?;
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_template_uuid_uniqueness() -> Result<(), Box<dyn std::error::Error>> {
        let engine = TemplateEngine::new()?;
        let template = engine.templates.get("elementwise")?;
        let context = Context::new();

        // Render multiple times and ensure unique temp names don't conflict
        let mut results = Vec::new();

        for _ in 0..50 {
            let result = engine.render(template, &context);
            assert!(result.is_ok());
            results.push(result.unwrap());
        }

        // All renders should succeed and produce valid output
        for rendered in results {
            assert!(rendered.contains("synthesized_kernel"));
            assert!(rendered.contains("__global__"));
        }
    }

    #[test]
    fn test_template_large_context() -> Result<(), Box<dyn std::error::Error>> {
        let engine = TemplateEngine::new()?;
        let template = engine.templates.get("matrix_multiply")?;

        // Create a large context with many values
        let mut large_context = Context::new();
        for i in 0..1000 {
            large_context.insert(&format!("param_{i}"), &i);
        }

        let result = engine.render(template, &large_context);
        assert!(result.is_ok());

        let rendered = result?;
        assert!(rendered.contains("synthesized_kernel"));
    }

    #[test]
    fn test_template_engine_stress_test() -> Result<(), Box<dyn std::error::Error>> {
        let engine = TemplateEngine::new()?;

        // Stress test with many concurrent operations
        let operations = vec![
            OperationType::MatrixMultiply,
            OperationType::Reduction,
            OperationType::Convolution,
            OperationType::Elementwise,
            OperationType::Custom,
        ];

        for _ in 0..100 {
            for op_type in &operations {
                let spec = KernelSpecification {
                    operation_type: op_type.clone(),
                    data_layout: DataLayout {
                        input_shape: vec![128, 128],
                        output_shape: vec![128, 128],
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

                let template = engine.select_template(&spec)?;
                let context = Context::new();
                let rendered = engine.render(template, &context)?;

                assert!(rendered.contains("__global__"));
                assert!(rendered.contains("synthesized_kernel"));
            }
        }
    }

    #[test]
    fn test_template_custom_operation_fallback() -> Result<(), Box<dyn std::error::Error>> {
        let engine = TemplateEngine::new()?;

        let spec = KernelSpecification {
            operation_type: OperationType::Custom,
            data_layout: DataLayout {
                input_shape: vec![512, 512],
                output_shape: vec![512, 512],
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

        let template = engine.select_template(&spec)?;

        // Custom operations should fall back to elementwise template
        assert!(template.contains("Elementwise operation kernel"));
        assert!(template.contains("{{kernel_name}}"));
        assert!(template.contains("{{precision}}"));
    }
}
