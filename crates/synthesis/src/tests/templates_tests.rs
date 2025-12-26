//! Tests for synthesis templates module

use crate::templates::*;
use crate::error::SynthesisResult;

#[test]
    fn test_template_registry_creation() -> Result<(), Box<dyn std::error::Error>> {
    let registry = TemplateRegistry::new().unwrap();
    assert!(registry.is_initialized());
}

#[test]
    fn test_template_registration() -> Result<(), Box<dyn std::error::Error>> {
    let mut registry = TemplateRegistry::new().unwrap();
    
    let template = KernelTemplate {
        name: "vector_add".to_string(),
        description: "Element-wise vector addition".to_string(),
        category: TemplateCategory::ElementWise,
        algorithm_type: AlgorithmType::ElementWise,
        source_code: r#"
            template<typename T>
            __global__ void vector_add(T* a, T* b, T* c, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    c[idx] = a[idx] + b[idx];
                }
            }
        "#.to_string(),
        parameters: vec![
            TemplateParameter {
                name: "T".to_string(),
                param_type: ParameterType::DataType,
                default_value: Some("float".to_string()),
                constraints: vec!["numeric".to_string()],
                description: "Data type for vector elements".to_string(),
            },
            TemplateParameter {
                name: "BLOCK_SIZE".to_string(),
                param_type: ParameterType::Integer,
                default_value: Some("256".to_string()),
                constraints: vec!["power_of_2".to_string(), "32..1024".to_string()],
                description: "Thread block size".to_string(),
            },
        ],
        launch_configuration: LaunchConfigTemplate {
            grid_size_formula: "ceil(n / BLOCK_SIZE)".to_string(),
            block_size_formula: "BLOCK_SIZE".to_string(),
            shared_memory_formula: "0".to_string(),
        },
        performance_characteristics: PerformanceCharacteristics {
            time_complexity: "O(n)".to_string(),
            space_complexity: "O(1)".to_string(),
            memory_access_pattern: MemoryAccessPattern::Sequential,
            compute_intensity: ComputeIntensity::Low,
            scalability: ScalabilityPattern::Linear,
        },
        supported_architectures: vec!["sm_35".to_string(), "sm_70".to_string(), "sm_80".to_string()],
        optimization_hints: vec![
            OptimizationHint::VectorizedAccess,
            OptimizationHint::MemoryCoalescing,
        ],
    };
    
    registry.register_template(template.clone()).unwrap();
    
    let retrieved = registry.get_template("vector_add").unwrap();
    assert_eq!(retrieved.name, template.name);
    assert_eq!(retrieved.category, template.category);
}

#[test]
    fn test_template_search() -> Result<(), Box<dyn std::error::Error>> {
    let mut registry = TemplateRegistry::new().unwrap();
    
    // Register multiple templates
    let templates = vec![
        ("vector_add", TemplateCategory::ElementWise, AlgorithmType::ElementWise),
        ("matrix_mul", TemplateCategory::LinearAlgebra, AlgorithmType::MatrixOperation),
        ("reduce_sum", TemplateCategory::Reduction, AlgorithmType::Reduction),
        ("conv2d", TemplateCategory::Convolution, AlgorithmType::Convolution),
    ];
    
    for (name, category, algorithm) in templates {
        let template = KernelTemplate {
            name: name.to_string(),
            description: format!("{} template", name),
            category,
            algorithm_type: algorithm,
            source_code: format!("template code for {}", name),
            parameters: vec![],
            launch_configuration: LaunchConfigTemplate::default(),
            performance_characteristics: PerformanceCharacteristics::default(),
            supported_architectures: vec!["sm_70".to_string()],
            optimization_hints: vec![],
        };
        registry.register_template(template).unwrap();
    }
    
    // Search by category
    let elementwise_templates = registry.search_by_category(TemplateCategory::ElementWise).unwrap();
    assert_eq!(elementwise_templates.len(), 1);
    assert_eq!(elementwise_templates.get(0).ok_or("Index out of bounds")?.name, "vector_add");
    
    // Search by algorithm type
    let matrix_templates = registry.search_by_algorithm(AlgorithmType::MatrixOperation).unwrap();
    assert_eq!(matrix_templates.len(), 1);
    assert_eq!(matrix_templates.get(0).ok_or("Index out of bounds")?.name, "matrix_mul");
}

#[test]
    fn test_template_instantiation() -> Result<(), Box<dyn std::error::Error>> {
    let registry = TemplateRegistry::new().unwrap();
    
    let template = KernelTemplate {
        name: "templated_kernel".to_string(),
        description: "Test template instantiation".to_string(),
        category: TemplateCategory::ElementWise,
        algorithm_type: AlgorithmType::ElementWise,
        source_code: r#"
            template<typename {{T}}, int {{BLOCK_SIZE}}>
            __global__ void templated_kernel({{T}}* data, int n) {
                int idx = blockIdx.x * {{BLOCK_SIZE}} + threadIdx.x;
                if (idx < n) {
                    data[idx] = data[idx] * {{SCALE_FACTOR}};
                }
            }
        "#.to_string(),
        parameters: vec![
            TemplateParameter {
                name: "T".to_string(),
                param_type: ParameterType::DataType,
                default_value: Some("float".to_string()),
                constraints: vec![],
                description: "Data type".to_string(),
            },
            TemplateParameter {
                name: "BLOCK_SIZE".to_string(),
                param_type: ParameterType::Integer,
                default_value: Some("256".to_string()),
                constraints: vec![],
                description: "Block size".to_string(),
            },
            TemplateParameter {
                name: "SCALE_FACTOR".to_string(),
                param_type: ParameterType::Float,
                default_value: Some("2.0f".to_string()),
                constraints: vec![],
                description: "Scaling factor".to_string(),
            },
        ],
        launch_configuration: LaunchConfigTemplate::default(),
        performance_characteristics: PerformanceCharacteristics::default(),
        supported_architectures: vec!["sm_70".to_string()],
        optimization_hints: vec![],
    };
    
    let instantiation_params = vec![
        ("T".to_string(), "double".to_string()),
        ("BLOCK_SIZE".to_string(), "512".to_string()),
        ("SCALE_FACTOR".to_string(), "3.14".to_string()),
    ];
    
    let instantiated = registry.instantiate_template(&template, &instantiation_params).unwrap();
    
    assert!(instantiated.contains("double"));
    assert!(instantiated.contains("512"));
    assert!(instantiated.contains("3.14"));
    assert!(!instantiated.contains("{{T}}"));
    assert!(!instantiated.contains("{{BLOCK_SIZE}}"));
    assert!(!instantiated.contains("{{SCALE_FACTOR}}"));
}

#[test]
    fn test_matrix_multiplication_template() -> Result<(), Box<dyn std::error::Error>> {
    let registry = TemplateRegistry::new().unwrap();
    
    let matmul_template = KernelTemplate {
        name: "matrix_multiply_blocked".to_string(),
        description: "Blocked matrix multiplication with shared memory".to_string(),
        category: TemplateCategory::LinearAlgebra,
        algorithm_type: AlgorithmType::MatrixOperation,
        source_code: r#"
            template<typename {{T}}, int {{TILE_SIZE}}>
            __global__ void matrix_multiply_blocked({{T}}* A, {{T}}* B, {{T}}* C, 
                                                   int M, int N, int K) {
                __shared__ {{T}} As[{{TILE_SIZE}}][{{TILE_SIZE}}];
                __shared__ {{T}} Bs[{{TILE_SIZE}}][{{TILE_SIZE}}];
                
                int bx = blockIdx.x, by = blockIdx.y;
                int tx = threadIdx.x, ty = threadIdx.y;
                
                int Row = by * {{TILE_SIZE}} + ty;
                int Col = bx * {{TILE_SIZE}} + tx;
                
                {{T}} sum = 0.0;
                
                for (int t = 0; t < (K + {{TILE_SIZE}} - 1) / {{TILE_SIZE}}; t++) {
                    if (Row < M && t * {{TILE_SIZE}} + tx < K)
                        As[ty][tx] = A[Row * K + t * {{TILE_SIZE}} + tx];
                    else
                        As[ty][tx] = 0.0;
                        
                    if (Col < N && t * {{TILE_SIZE}} + ty < K)
                        Bs[ty][tx] = B[(t * {{TILE_SIZE}} + ty) * N + Col];
                    else
                        Bs[ty][tx] = 0.0;
                        
                    __syncthreads();
                    
                    for (int k = 0; k < {{TILE_SIZE}}; k++) {
                        sum += As[ty][k] * Bs[k][tx];
                    }
                    
                    __syncthreads();
                }
                
                if (Row < M && Col < N) {
                    C[Row * N + Col] = sum;
                }
            }
        "#.to_string(),
        parameters: vec![
            TemplateParameter {
                name: "T".to_string(),
                param_type: ParameterType::DataType,
                default_value: Some("float".to_string()),
                constraints: vec!["numeric".to_string()],
                description: "Matrix element data type".to_string(),
            },
            TemplateParameter {
                name: "TILE_SIZE".to_string(),
                param_type: ParameterType::Integer,
                default_value: Some("16".to_string()),
                constraints: vec!["power_of_2".to_string(), "8..32".to_string()],
                description: "Tile size for blocking".to_string(),
            },
        ],
        launch_configuration: LaunchConfigTemplate {
            grid_size_formula: "(ceil(N/TILE_SIZE), ceil(M/TILE_SIZE))".to_string(),
            block_size_formula: "(TILE_SIZE, TILE_SIZE)".to_string(),
            shared_memory_formula: "2 * TILE_SIZE * TILE_SIZE * sizeof(T)".to_string(),
        },
        performance_characteristics: PerformanceCharacteristics {
            time_complexity: "O(M*N*K)".to_string(),
            space_complexity: "O(TILE_SIZE^2)".to_string(),
            memory_access_pattern: MemoryAccessPattern::Blocked,
            compute_intensity: ComputeIntensity::High,
            scalability: ScalabilityPattern::Cubic,
        },
        supported_architectures: vec!["sm_35".to_string(), "sm_70".to_string(), "sm_80".to_string()],
        optimization_hints: vec![
            OptimizationHint::SharedMemoryBlocking,
            OptimizationHint::MemoryCoalescing,
            OptimizationHint::LoopUnrolling,
        ],
    };
    
    let instantiation_params = vec![
        ("T".to_string(), "float".to_string()),
        ("TILE_SIZE".to_string(), "16".to_string()),
    ];
    
    let instantiated = registry.instantiate_template(&matmul_template, &instantiation_params).unwrap();
    
    assert!(instantiated.contains("__shared__ float As[16][16]"));
    assert!(instantiated.contains("__shared__ float Bs[16][16]"));
    assert!(instantiated.contains("blockIdx.x"));
    assert!(instantiated.contains("__syncthreads"));
}

#[test]
    fn test_reduction_template() -> Result<(), Box<dyn std::error::Error>> {
    let registry = TemplateRegistry::new().unwrap();
    
    let reduction_template = KernelTemplate {
        name: "reduction_sum".to_string(),
        description: "Parallel reduction using shared memory".to_string(),
        category: TemplateCategory::Reduction,
        algorithm_type: AlgorithmType::Reduction,
        source_code: r#"
            template<typename {{T}}, int {{BLOCK_SIZE}}>
            __global__ void reduction_sum({{T}}* input, {{T}}* output, int n) {
                extern __shared__ {{T}} sdata[];
                
                int tid = threadIdx.x;
                int idx = blockIdx.x * {{BLOCK_SIZE}} + threadIdx.x;
                
                // Load data into shared memory
                sdata[tid] = (idx < n) ? input[idx] : 0;
                __syncthreads();
                
                // Perform reduction in shared memory
                for (int s = {{BLOCK_SIZE}} / 2; s > 0; s >>= 1) {
                    if (tid < s) {
                        sdata[tid] += sdata[tid + s];
                    }
                    __syncthreads();
                }
                
                // Write result for this block to global memory
                if (tid == 0) {
                    output[blockIdx.x] = sdata[0];
                }
            }
        "#.to_string(),
        parameters: vec![
            TemplateParameter {
                name: "T".to_string(),
                param_type: ParameterType::DataType,
                default_value: Some("float".to_string()),
                constraints: vec!["numeric".to_string()],
                description: "Data type for reduction".to_string(),
            },
            TemplateParameter {
                name: "BLOCK_SIZE".to_string(),
                param_type: ParameterType::Integer,
                default_value: Some("256".to_string()),
                constraints: vec!["power_of_2".to_string(), "32..1024".to_string()],
                description: "Thread block size".to_string(),
            },
        ],
        launch_configuration: LaunchConfigTemplate {
            grid_size_formula: "ceil(n / BLOCK_SIZE)".to_string(),
            block_size_formula: "BLOCK_SIZE".to_string(),
            shared_memory_formula: "BLOCK_SIZE * sizeof(T)".to_string(),
        },
        performance_characteristics: PerformanceCharacteristics {
            time_complexity: "O(log n)".to_string(),
            space_complexity: "O(BLOCK_SIZE)".to_string(),
            memory_access_pattern: MemoryAccessPattern::Random,
            compute_intensity: ComputeIntensity::Medium,
            scalability: ScalabilityPattern::Logarithmic,
        },
        supported_architectures: vec!["sm_35".to_string(), "sm_70".to_string(), "sm_80".to_string()],
        optimization_hints: vec![
            OptimizationHint::SharedMemoryOptimization,
            OptimizationHint::WarpShuffle,
            OptimizationHint::BankConflictAvoidance,
        ],
    };
    
    let instantiation_params = vec![
        ("T".to_string(), "double".to_string()),
        ("BLOCK_SIZE".to_string(), "512".to_string()),
    ];
    
    let instantiated = registry.instantiate_template(&reduction_template, &instantiation_params).unwrap();
    
    assert!(instantiated.contains("extern __shared__ double sdata[]"));
    assert!(instantiated.contains("blockIdx.x * 512"));
    assert!(instantiated.contains("for (int s = 512 / 2"));
}

#[test]
    fn test_convolution_template() -> Result<(), Box<dyn std::error::Error>> {
    let registry = TemplateRegistry::new().unwrap();
    
    let conv_template = KernelTemplate {
        name: "convolution_2d".to_string(),
        description: "2D convolution with constant memory filter".to_string(),
        category: TemplateCategory::Convolution,
        algorithm_type: AlgorithmType::Convolution,
        source_code: r#"
            #define FILTER_SIZE {{FILTER_SIZE}}
            __constant__ float filter[FILTER_SIZE][FILTER_SIZE];
            
            template<int {{TILE_SIZE}}>
            __global__ void convolution_2d(float* input, float* output, 
                                         int width, int height) {
                __shared__ float tile[{{TILE_SIZE}} + FILTER_SIZE - 1][{{TILE_SIZE}} + FILTER_SIZE - 1];
                
                int tx = threadIdx.x;
                int ty = threadIdx.y;
                int bx = blockIdx.x * {{TILE_SIZE}};
                int by = blockIdx.y * {{TILE_SIZE}};
                
                // Load tile with halo
                int input_x = bx + tx - FILTER_SIZE/2;
                int input_y = by + ty - FILTER_SIZE/2;
                
                if (input_x >= 0 && input_x < width && input_y >= 0 && input_y < height) {
                    tile[ty][tx] = input[input_y * width + input_x];
                } else {
                    tile[ty][tx] = 0.0f;
                }
                
                __syncthreads();
                
                // Compute convolution
                if (tx < {{TILE_SIZE}} && ty < {{TILE_SIZE}}) {
                    int out_x = bx + tx;
                    int out_y = by + ty;
                    
                    if (out_x < width - FILTER_SIZE + 1 && out_y < height - FILTER_SIZE + 1) {
                        float sum = 0.0f;
                        for (int fy = 0; fy < FILTER_SIZE; fy++) {
                            for (int fx = 0; fx < FILTER_SIZE; fx++) {
                                sum += tile[ty + fy][tx + fx] * filter[fy][fx];
                            }
                        }
                        output[out_y * (width - FILTER_SIZE + 1) + out_x] = sum;
                    }
                }
            }
        "#.to_string(),
        parameters: vec![
            TemplateParameter {
                name: "FILTER_SIZE".to_string(),
                param_type: ParameterType::Integer,
                default_value: Some("3".to_string()),
                constraints: vec!["odd".to_string(), "3..15".to_string()],
                description: "Size of convolution filter".to_string(),
            },
            TemplateParameter {
                name: "TILE_SIZE".to_string(),
                param_type: ParameterType::Integer,
                default_value: Some("16".to_string()),
                constraints: vec!["4..32".to_string()],
                description: "Tile size for shared memory".to_string(),
            },
        ],
        launch_configuration: LaunchConfigTemplate {
            grid_size_formula: "(ceil(width/TILE_SIZE), ceil(height/TILE_SIZE))".to_string(),
            block_size_formula: "(TILE_SIZE + FILTER_SIZE - 1, TILE_SIZE + FILTER_SIZE - 1)".to_string(),
            shared_memory_formula: "(TILE_SIZE + FILTER_SIZE - 1)^2 * sizeof(float)".to_string(),
        },
        performance_characteristics: PerformanceCharacteristics {
            time_complexity: "O(W*H*F^2)".to_string(),
            space_complexity: "O((TILE_SIZE + F)^2)".to_string(),
            memory_access_pattern: MemoryAccessPattern::Stencil,
            compute_intensity: ComputeIntensity::High,
            scalability: ScalabilityPattern::Quadratic,
        },
        supported_architectures: vec!["sm_35".to_string(), "sm_70".to_string(), "sm_80".to_string()],
        optimization_hints: vec![
            OptimizationHint::ConstantMemory,
            OptimizationHint::SharedMemoryTiling,
            OptimizationHint::HaloLoading,
        ],
    };
    
    let instantiation_params = vec![
        ("FILTER_SIZE".to_string(), "5".to_string()),
        ("TILE_SIZE".to_string(), "16".to_string()),
    ];
    
    let instantiated = registry.instantiate_template(&conv_template, &instantiation_params).unwrap();
    
    assert!(instantiated.contains("#define FILTER_SIZE 5"));
    assert!(instantiated.contains("float filter[5][5]"));
    assert!(instantiated.contains("__shared__ float tile[20][20]")); // 16 + 5 - 1 = 20
    assert!(instantiated.contains("for (int fy = 0; fy < 5"));
}

#[test]
    fn test_template_validation() -> Result<(), Box<dyn std::error::Error>> {
    let registry = TemplateRegistry::new().unwrap();
    
    let invalid_template = KernelTemplate {
        name: "".to_string(), // Invalid empty name
        description: "Invalid template".to_string(),
        category: TemplateCategory::ElementWise,
        algorithm_type: AlgorithmType::ElementWise,
        source_code: "invalid code {{UNDEFINED_PARAM}}".to_string(), // Undefined parameter
        parameters: vec![
            TemplateParameter {
                name: "BLOCK_SIZE".to_string(),
                param_type: ParameterType::Integer,
                default_value: Some("-1".to_string()), // Invalid default
                constraints: vec!["positive".to_string()],
                description: "Block size".to_string(),
            },
        ],
        launch_configuration: LaunchConfigTemplate::default(),
        performance_characteristics: PerformanceCharacteristics::default(),
        supported_architectures: vec![],
        optimization_hints: vec![],
    };
    
    let validation_result = registry.validate_template(&invalid_template);
    assert!(validation_result.is_err());
}

#[test]
    fn test_template_performance_estimation() -> Result<(), Box<dyn std::error::Error>> {
    let registry = TemplateRegistry::new().unwrap();
    
    let template = KernelTemplate {
        name: "performance_test".to_string(),
        description: "Template for performance estimation".to_string(),
        category: TemplateCategory::ElementWise,
        algorithm_type: AlgorithmType::ElementWise,
        source_code: "simple template".to_string(),
        parameters: vec![],
        launch_configuration: LaunchConfigTemplate {
            grid_size_formula: "ceil(n / 256)".to_string(),
            block_size_formula: "256".to_string(),
            shared_memory_formula: "0".to_string(),
        },
        performance_characteristics: PerformanceCharacteristics {
            time_complexity: "O(n)".to_string(),
            space_complexity: "O(1)".to_string(),
            memory_access_pattern: MemoryAccessPattern::Sequential,
            compute_intensity: ComputeIntensity::Low,
            scalability: ScalabilityPattern::Linear,
        },
        supported_architectures: vec!["sm_70".to_string()],
        optimization_hints: vec![],
    };
    
    let device_info = DeviceInfo {
        compute_capability: (7, 0),
        max_threads_per_block: 1024,
        shared_memory_per_block: 48 * 1024,
        max_registers_per_block: 65536,
        memory_bandwidth_gbps: 900.0,
        peak_flops: 7000.0e9,
        warp_size: 32,
    };
    
    let problem_size = ProblemSize {
        input_elements: 1000000,
        output_elements: 1000000,
        work_items: 1000000,
    };
    
    let performance_estimate = registry.estimate_performance(&template, &device_info, &problem_size).unwrap();
    
    assert!(performance_estimate.estimated_execution_time_ms > 0.0);
    assert!(performance_estimate.estimated_memory_bandwidth_gbps > 0.0);
    assert!(performance_estimate.estimated_occupancy >= 0.0 && performance_estimate.estimated_occupancy <= 1.0);
    assert!(performance_estimate.confidence_score >= 0.0 && performance_estimate.confidence_score <= 1.0);
}

#[test]
    fn test_template_compatibility() -> Result<(), Box<dyn std::error::Error>> {
    let registry = TemplateRegistry::new().unwrap();
    
    let template = KernelTemplate {
        name: "compatibility_test".to_string(),
        description: "Test template compatibility".to_string(),
        category: TemplateCategory::ElementWise,
        algorithm_type: AlgorithmType::ElementWise,
        source_code: "template code".to_string(),
        parameters: vec![],
        launch_configuration: LaunchConfigTemplate::default(),
        performance_characteristics: PerformanceCharacteristics::default(),
        supported_architectures: vec!["sm_70".to_string(), "sm_80".to_string()],
        optimization_hints: vec![],
    };
    
    let requirements = CompatibilityRequirements {
        target_architecture: "sm_80".to_string(),
        max_shared_memory_bytes: 48 * 1024,
        max_registers_per_thread: 64,
        required_features: vec!["tensor_cores".to_string()],
    };
    
    let compatibility = registry.check_compatibility(&template, &requirements).unwrap();
    
    assert!(compatibility.is_compatible);
    assert!(compatibility.architecture_supported);
}

#[test]
    fn test_template_optimization() -> Result<(), Box<dyn std::error::Error>> {
    let mut registry = TemplateRegistry::new().unwrap();
    
    let base_template = KernelTemplate {
        name: "base_template".to_string(),
        description: "Base template for optimization".to_string(),
        category: TemplateCategory::ElementWise,
        algorithm_type: AlgorithmType::ElementWise,
        source_code: r#"
            __global__ void base_kernel(float* data, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    data[idx] = data[idx] * 2.0f;
                }
            }
        "#.to_string(),
        parameters: vec![],
        launch_configuration: LaunchConfigTemplate::default(),
        performance_characteristics: PerformanceCharacteristics::default(),
        supported_architectures: vec!["sm_70".to_string()],
        optimization_hints: vec![OptimizationHint::VectorizedAccess],
    };
    
    let optimization_request = TemplateOptimizationRequest {
        target_performance: PerformanceTarget {
            min_throughput_gflops: 100.0,
            max_execution_time_ms: 5.0,
            target_occupancy: 0.80,
        },
        device_constraints: DeviceConstraints {
            max_shared_memory_bytes: 48 * 1024,
            max_registers_per_thread: 64,
            compute_capability: (7, 0),
        },
        optimization_preferences: OptimizationPreferences {
            prioritize_performance: true,
            allow_precision_loss: false,
            enable_aggressive_optimizations: true,
        },
    };
    
    let optimized_template = registry.optimize_template(&base_template, &optimization_request).unwrap();
    
    assert_ne!(optimized_template.source_code, base_template.source_code);
    assert!(!optimized_template.optimization_hints.is_empty());
}

#[test]
    fn test_template_library_management() -> Result<(), Box<dyn std::error::Error>> {
    let mut registry = TemplateRegistry::new().unwrap();
    
    // Create a template library
    let library = TemplateLibrary {
        name: "standard_library".to_string(),
        version: "1.0.0".to_string(),
        description: "Standard kernel templates".to_string(),
        author: "ExoRust Team".to_string(),
        templates: vec![
            "vector_add".to_string(),
            "matrix_mul".to_string(),
            "reduction_sum".to_string(),
        ],
        dependencies: vec![],
        metadata: vec![
            ("license".to_string(), "MIT".to_string()),
            ("category".to_string(), "general".to_string()),
        ],
    };
    
    registry.install_library(&library).unwrap();
    
    let installed_libraries = registry.get_installed_libraries().unwrap();
    assert!(!installed_libraries.is_empty());
    
    let found_library = installed_libraries.iter()
        .find(|lib| lib.name == "standard_library")
        .unwrap();
    assert_eq!(found_library.version, "1.0.0");
}

#[test]
    fn test_template_code_generation() -> Result<(), Box<dyn std::error::Error>> {
    let registry = TemplateRegistry::new().unwrap();
    
    let generation_request = CodeGenerationRequest {
        algorithm_type: AlgorithmType::MatrixOperation,
        input_specification: vec![
            DataSpecification {
                data_type: DataType::Float32,
                dimensions: vec![1024, 1024],
                layout: DataLayout::RowMajor,
            },
            DataSpecification {
                data_type: DataType::Float32,
                dimensions: vec![1024, 1024],
                layout: DataLayout::RowMajor,
            },
        ],
        output_specification: DataSpecification {
            data_type: DataType::Float32,
            dimensions: vec![1024, 1024],
            layout: DataLayout::RowMajor,
        },
        performance_requirements: PerformanceRequirements {
            max_execution_time_ms: 50.0,
            min_throughput_gflops: 500.0,
            max_memory_usage_mb: 256.0,
            target_occupancy: 0.80,
        },
        optimization_preferences: OptimizationPreferences {
            prioritize_performance: true,
            allow_precision_loss: false,
            enable_aggressive_optimizations: false,
        },
    };
    
    let generated_code = registry.generate_code(&generation_request).unwrap();
    
    assert!(!generated_code.kernel_code.is_empty());
    assert!(!generated_code.host_code.is_empty());
    assert!(!generated_code.template_name.is_empty());
    assert!(!generated_code.instantiation_parameters.is_empty());
}

#[test]
    fn test_template_benchmarking() -> Result<(), Box<dyn std::error::Error>> {
    let registry = TemplateRegistry::new().unwrap();
    
    let template = KernelTemplate {
        name: "benchmark_test".to_string(),
        description: "Template for benchmarking".to_string(),
        category: TemplateCategory::ElementWise,
        algorithm_type: AlgorithmType::ElementWise,
        source_code: "benchmark template".to_string(),
        parameters: vec![],
        launch_configuration: LaunchConfigTemplate::default(),
        performance_characteristics: PerformanceCharacteristics::default(),
        supported_architectures: vec!["sm_70".to_string()],
        optimization_hints: vec![],
    };
    
    let benchmark_config = BenchmarkConfig {
        input_sizes: vec![1000, 10000, 100000],
        iterations_per_size: 10,
        warmup_iterations: 3,
        measure_memory_transfers: true,
        measure_kernel_execution: true,
    };
    
    let benchmark_results = registry.benchmark_template(&template, &benchmark_config).unwrap();
    
    assert_eq!(benchmark_results.results.len(), 3); // One for each input size
    for result in benchmark_results.results {
        assert!(result.average_execution_time_ms > 0.0);
        assert!(result.throughput_gflops >= 0.0);
        assert!(result.input_size > 0);
    }
}

#[test]
    fn test_template_auto_tuning() -> Result<(), Box<dyn std::error::Error>> {
    let registry = TemplateRegistry::new().unwrap();
    
    let template = KernelTemplate {
        name: "auto_tune_test".to_string(),
        description: "Template for auto-tuning".to_string(),
        category: TemplateCategory::ElementWise,
        algorithm_type: AlgorithmType::ElementWise,
        source_code: r#"
            template<int {{BLOCK_SIZE}}>
            __global__ void auto_tune_kernel(float* data, int n) {
                int idx = blockIdx.x * {{BLOCK_SIZE}} + threadIdx.x;
                if (idx < n) {
                    data[idx] = data[idx] * 2.0f;
                }
            }
        "#.to_string(),
        parameters: vec![
            TemplateParameter {
                name: "BLOCK_SIZE".to_string(),
                param_type: ParameterType::Integer,
                default_value: Some("256".to_string()),
                constraints: vec!["32..1024".to_string(), "power_of_2".to_string()],
                description: "Block size for tuning".to_string(),
            },
        ],
        launch_configuration: LaunchConfigTemplate::default(),
        performance_characteristics: PerformanceCharacteristics::default(),
        supported_architectures: vec!["sm_70".to_string()],
        optimization_hints: vec![],
    };
    
    let tuning_space = AutoTuningSpace {
        parameter_ranges: vec![
            ("BLOCK_SIZE".to_string(), vec!["32".to_string(), "64".to_string(), "128".to_string(), "256".to_string(), "512".to_string()]),
        ],
        search_strategy: SearchStrategy::GridSearch,
        max_evaluations: 10,
        evaluation_metric: EvaluationMetric::ExecutionTime,
    };
    
    let problem_spec = ProblemSpecification {
        input_size: 100000,
        data_type: DataType::Float32,
        target_device: "Tesla V100".to_string(),
    };
    
    let tuning_results = registry.auto_tune_template(&template, &tuning_space, &problem_spec).unwrap();
    
    assert!(!tuning_results.optimal_parameters.is_empty());
    assert!(tuning_results.best_performance.execution_time_ms > 0.0);
    assert!(!tuning_results.evaluation_history.is_empty());
    assert!(tuning_results.convergence_iteration > 0);
}