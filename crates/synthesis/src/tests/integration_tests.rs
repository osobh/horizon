//! Integration tests for synthesis crate

use crate::*;
use std::time::Duration;

#[test]
    fn test_end_to_end_vector_addition() -> Result<(), Box<dyn std::error::Error>> {
    // Test complete synthesis pipeline for vector addition
    let synthesizer = synthesizer::KernelSynthesizer::new().unwrap();
    
    let goal = "Add two vectors of 10000 elements each";
    let spec = synthesizer::KernelSynthesisSpec {
        name: "e2e_vector_add".to_string(),
        goal: goal.to_string(),
        input_types: vec![DataType::Float32, DataType::Float32],
        output_types: vec![DataType::Float32],
        input_shapes: vec![vec![10000], vec![10000]],
        output_shapes: vec![vec![10000]],
        algorithm_hints: vec![AlgorithmHint::ElementWise],
        performance_requirements: PerformanceRequirements {
            max_execution_time_ms: 10.0,
            min_throughput_gflops: 50.0,
            max_memory_usage_mb: 128.0,
            target_occupancy: 0.75,
        },
        constraints: KernelConstraints::default(),
    };
    
    let synthesis_result = synthesizer.synthesize_kernel(&spec).unwrap();
    assert!(synthesis_result.success);
    assert!(!synthesis_result.kernel_code.is_empty());
    
    // Test goal interpretation
    let interpreter = interpreter::GoalInterpreter::new();
    let interpreted_spec = interpreter.interpret_goal(goal).unwrap();
    assert_eq!(interpreted_spec.algorithm_type, AlgorithmType::ElementWise);
    assert_eq!(interpreted_spec.input_types.len(), 2);
    
    // Test compilation
    let compiler = compiler::CudaCompiler::new().unwrap();
    let compilation_result = compiler.compile_source(&synthesis_result.kernel_code, &compiler::CompilerOptions::default());
    assert!(compilation_result.is_ok());
}

#[test]
    fn test_end_to_end_matrix_multiplication() -> Result<(), Box<dyn std::error::Error>> {
    // Test complete synthesis pipeline for matrix multiplication
    let synthesizer = synthesizer::KernelSynthesizer::new().unwrap();
    
    let goal = "Multiply two 512x512 matrices using shared memory optimization";
    let spec = synthesizer::KernelSynthesisSpec {
        name: "e2e_matrix_mul".to_string(),
        goal: goal.to_string(),
        input_types: vec![DataType::Float32, DataType::Float32],
        output_types: vec![DataType::Float32],
        input_shapes: vec![vec![512, 512], vec![512, 512]],
        output_shapes: vec![vec![512, 512]],
        algorithm_hints: vec![AlgorithmHint::MatrixOperation, AlgorithmHint::UseSharedMemory],
        performance_requirements: PerformanceRequirements {
            max_execution_time_ms: 100.0,
            min_throughput_gflops: 500.0,
            max_memory_usage_mb: 512.0,
            target_occupancy: 0.80,
        },
        constraints: KernelConstraints {
            max_shared_memory_bytes: 48 * 1024,
            max_registers_per_thread: 64,
            ..KernelConstraints::default()
        },
    };
    
    let synthesis_result = synthesizer.synthesize_kernel(&spec).unwrap();
    assert!(synthesis_result.success);
    assert!(synthesis_result.kernel_code.contains("__shared__"));
    
    // Test template matching
    let template_registry = templates::TemplateRegistry::new().unwrap();
    let matching_templates = template_registry.search_by_algorithm(AlgorithmType::MatrixOperation).unwrap();
    assert!(!matching_templates.is_empty());
    
    // Test improvement suggestions
    let improvement_engine = improvement::ImprovementEngine::new().unwrap();
    let performance_info = improvement::PerformanceInfo {
        execution_time_ms: 80.0,
        memory_bandwidth_gbps: 600.0,
        compute_utilization: 0.75,
        memory_utilization: 0.85,
        occupancy: 0.70,
        register_usage: 48,
        shared_memory_usage: 32 * 1024,
        warp_efficiency: 0.90,
    };
    
    let analysis = improvement_engine.analyze_performance(&synthesis_result.kernel_code, &performance_info).unwrap();
    assert!(!analysis.improvement_opportunities.is_empty());
}

#[test]
    fn test_pipeline_integration() -> Result<(), Box<dyn std::error::Error>> {
    // Test complete synthesis pipeline integration
    let pipeline = pipeline::SynthesisPipeline::new().unwrap();
    
    let synthesis_request = pipeline::SynthesisRequest {
        id: "integration_test".to_string(),
        goal: "Apply 3x3 convolution filter to 256x256 image".to_string(),
        input_specification: pipeline::InputSpecification {
            input_types: vec!["float*".to_string(), "float*".to_string()],
            input_sizes: vec![256 * 256, 3 * 3],
            data_layout: pipeline::DataLayout::RowMajor,
        },
        output_specification: pipeline::OutputSpecification {
            output_types: vec!["float*".to_string()],
            output_sizes: vec![254 * 254],
            data_layout: pipeline::DataLayout::RowMajor,
        },
        constraints: pipeline::SynthesisConstraints {
            max_execution_time_ms: 50.0,
            max_memory_usage_mb: 256,
            target_accuracy: 1e-6,
            preserve_precision: true,
        },
        preferences: pipeline::SynthesisPreferences {
            optimize_for: pipeline::OptimizationTarget::Performance,
            allow_approximations: false,
            debug_mode: false,
        },
    };
    
    let result = pipeline.synthesize(&synthesis_request).unwrap();
    assert!(result.success);
    assert!(!result.generated_code.is_empty());
    assert!(!result.stage_results.is_empty());
    
    // Verify all expected pipeline stages executed
    let stage_types: Vec<_> = result.stage_results.iter()
        .map(|r| r.stage_type)
        .collect();
    
    assert!(stage_types.contains(&pipeline::PipelineStage::GoalInterpretation));
    assert!(stage_types.contains(&pipeline::PipelineStage::CodeGeneration));
    assert!(stage_types.contains(&pipeline::PipelineStage::Compilation));
}

#[test]
    fn test_executor_pipeline_integration() -> Result<(), Box<dyn std::error::Error>> {
    // Test executor integration with synthesis pipeline
    let executor = executor::SynthesisExecutor::new().unwrap();
    
    let task = executor::SynthesisTask {
        id: "executor_integration_test".to_string(),
        goal: "Process array of 50000 elements with reduction operation".to_string(),
        priority: executor::TaskPriority::High,
        input_data: (0..50000).map(|i| i as f32).collect(),
        expected_output_size: 1,
        timeout: Some(Duration::from_secs(30)),
    };
    
    let task_id = executor.submit_task(task).unwrap();
    let result = executor.wait_for_completion(&task_id).unwrap();
    
    assert!(result.success);
    assert_eq!(result.output_data.len(), 1);
    assert!(result.execution_time.as_millis() > 0);
    
    // Test profiling data is available
    if let Some(profiling) = result.profiling_data {
        assert!(profiling.compilation_time.as_millis() > 0);
        assert!(profiling.execution_time.as_millis() > 0);
    }
}

#[test]
    fn test_error_propagation() -> Result<(), Box<dyn std::error::Error>> {
    // Test error handling across different components
    let synthesizer = synthesizer::KernelSynthesizer::new().unwrap();
    
    // Invalid synthesis specification
    let invalid_spec = synthesizer::KernelSynthesisSpec {
        name: "".to_string(), // Invalid empty name
        goal: "".to_string(), // Invalid empty goal
        input_types: vec![],  // Invalid empty inputs
        output_types: vec![], // Invalid empty outputs
        input_shapes: vec![],
        output_shapes: vec![],
        algorithm_hints: vec![],
        performance_requirements: PerformanceRequirements::default(),
        constraints: KernelConstraints::default(),
    };
    
    let result = synthesizer.synthesize_kernel(&invalid_spec);
    assert!(result.is_err());
    
    // Error should propagate through pipeline
    let pipeline = pipeline::SynthesisPipeline::new().unwrap();
    let invalid_request = pipeline::SynthesisRequest {
        id: "error_test".to_string(),
        goal: "".to_string(), // Invalid empty goal
        input_specification: pipeline::InputSpecification::default(),
        output_specification: pipeline::OutputSpecification::default(),
        constraints: pipeline::SynthesisConstraints::default(),
        preferences: pipeline::SynthesisPreferences::default(),
    };
    
    let pipeline_result = pipeline.synthesize(&invalid_request);
    assert!(pipeline_result.is_err());
}

#[test]
    fn test_cross_component_caching() -> Result<(), Box<dyn std::error::Error>> {
    // Test caching works across different components
    let synthesizer = synthesizer::KernelSynthesizer::with_config(
        synthesizer::SynthesizerConfig {
            cache_compiled_kernels: true,
            ..Default::default()
        }
    )?;
    
    let pipeline = pipeline::SynthesisPipeline::with_config(
        pipeline::PipelineConfig {
            enable_caching: true,
            cache_size_mb: 128,
            ..Default::default()
        }
    ).unwrap();
    
    let goal = "Add two vectors of 5000 elements";
    
    // First synthesis
    let spec1 = synthesizer::KernelSynthesisSpec {
        name: "cache_test_1".to_string(),
        goal: goal.to_string(),
        input_types: vec![DataType::Float32, DataType::Float32],
        output_types: vec![DataType::Float32],
        input_shapes: vec![vec![5000], vec![5000]],
        output_shapes: vec![vec![5000]],
        algorithm_hints: vec![AlgorithmHint::ElementWise],
        performance_requirements: PerformanceRequirements::default(),
        constraints: KernelConstraints::default(),
    };
    
    let start1 = std::time::Instant::now();
    let result1 = synthesizer.synthesize_kernel(&spec1).unwrap();
    let duration1 = start1.elapsed();
    
    // Second synthesis with same goal (should use cache)
    let spec2 = synthesizer::KernelSynthesisSpec {
        name: "cache_test_2".to_string(),
        goal: goal.to_string(),
        input_types: vec![DataType::Float32, DataType::Float32],
        output_types: vec![DataType::Float32],
        input_shapes: vec![vec![5000], vec![5000]],
        output_shapes: vec![vec![5000]],
        algorithm_hints: vec![AlgorithmHint::ElementWise],
        performance_requirements: PerformanceRequirements::default(),
        constraints: KernelConstraints::default(),
    };
    
    let start2 = std::time::Instant::now();
    let result2 = synthesizer.synthesize_kernel(&spec2).unwrap();
    let duration2 = start2.elapsed();
    
    assert!(result1.success && result2.success);
    assert_eq!(result1.kernel_code, result2.kernel_code);
    assert!(duration2 <= duration1); // Second should be faster due to caching
}

#[test]
    fn test_template_synthesis_integration() -> Result<(), Box<dyn std::error::Error>> {
    // Test template-based synthesis integration
    let mut template_registry = templates::TemplateRegistry::new().unwrap();
    
    // Register a custom template
    let custom_template = templates::KernelTemplate {
        name: "custom_elementwise".to_string(),
        description: "Custom element-wise operation template".to_string(),
        category: templates::TemplateCategory::ElementWise,
        algorithm_type: AlgorithmType::ElementWise,
        source_code: r#"
            template<typename {{T}}, int {{BLOCK_SIZE}}>
            __global__ void custom_elementwise({{T}}* input, {{T}}* output, int n) {
                int idx = blockIdx.x * {{BLOCK_SIZE}} + threadIdx.x;
                if (idx < n) {
                    output[idx] = input[idx] * {{SCALE}} + {{OFFSET}};
                }
            }
        "#.to_string(),
        parameters: vec![
            templates::TemplateParameter {
                name: "T".to_string(),
                param_type: templates::ParameterType::DataType,
                default_value: Some("float".to_string()),
                constraints: vec!["numeric".to_string()],
                description: "Data type".to_string(),
            },
            templates::TemplateParameter {
                name: "BLOCK_SIZE".to_string(),
                param_type: templates::ParameterType::Integer,
                default_value: Some("256".to_string()),
                constraints: vec!["power_of_2".to_string()],
                description: "Block size".to_string(),
            },
            templates::TemplateParameter {
                name: "SCALE".to_string(),
                param_type: templates::ParameterType::Float,
                default_value: Some("2.0f".to_string()),
                constraints: vec![],
                description: "Scale factor".to_string(),
            },
            templates::TemplateParameter {
                name: "OFFSET".to_string(),
                param_type: templates::ParameterType::Float,
                default_value: Some("1.0f".to_string()),
                constraints: vec![],
                description: "Offset value".to_string(),
            },
        ],
        launch_configuration: templates::LaunchConfigTemplate {
            grid_size_formula: "ceil(n / BLOCK_SIZE)".to_string(),
            block_size_formula: "BLOCK_SIZE".to_string(),
            shared_memory_formula: "0".to_string(),
        },
        performance_characteristics: templates::PerformanceCharacteristics::default(),
        supported_architectures: vec!["sm_70".to_string()],
        optimization_hints: vec![],
    };
    
    template_registry.register_template(custom_template).unwrap();
    
    // Use template in synthesis
    let synthesizer = synthesizer::KernelSynthesizer::new().unwrap();
    let spec = synthesizer::KernelSynthesisSpec {
        name: "template_integration_test".to_string(),
        goal: "Scale and offset array elements".to_string(),
        input_types: vec![DataType::Float32],
        output_types: vec![DataType::Float32],
        input_shapes: vec![vec![10000]],
        output_shapes: vec![vec![10000]],
        algorithm_hints: vec![
            AlgorithmHint::ElementWise,
            AlgorithmHint::UseTemplate("custom_elementwise".to_string())
        ],
        performance_requirements: PerformanceRequirements::default(),
        constraints: KernelConstraints::default(),
    };
    
    let result = synthesizer.synthesize_kernel(&spec).unwrap();
    assert!(result.success);
    assert!(result.template_info.is_some());
    assert_eq!(result.template_info.unwrap().template_name, "custom_elementwise");
}

#[test]
    fn test_improvement_synthesis_cycle() -> Result<(), Box<dyn std::error::Error>> {
    // Test improvement and synthesis feedback cycle
    let synthesizer = synthesizer::KernelSynthesizer::new().unwrap();
    let improvement_engine = improvement::ImprovementEngine::new().unwrap();
    
    // Initial synthesis
    let spec = synthesizer::KernelSynthesisSpec {
        name: "improvement_cycle_test".to_string(),
        goal: "Matrix transpose operation".to_string(),
        input_types: vec![DataType::Float32],
        output_types: vec![DataType::Float32],
        input_shapes: vec![vec![1024, 1024]],
        output_shapes: vec![vec![1024, 1024]],
        algorithm_hints: vec![AlgorithmHint::MatrixOperation],
        performance_requirements: PerformanceRequirements {
            max_execution_time_ms: 20.0,
            min_throughput_gflops: 200.0,
            max_memory_usage_mb: 256.0,
            target_occupancy: 0.80,
        },
        constraints: KernelConstraints::default(),
    };
    
    let initial_result = synthesizer.synthesize_kernel(&spec).unwrap();
    assert!(initial_result.success);
    
    // Analyze performance and suggest improvements
    let performance_info = improvement::PerformanceInfo {
        execution_time_ms: 25.0, // Slower than target
        memory_bandwidth_gbps: 400.0,
        compute_utilization: 0.60,
        memory_utilization: 0.90,
        occupancy: 0.65, // Lower than target
        register_usage: 32,
        shared_memory_usage: 0, // No shared memory used
        warp_efficiency: 0.80,
    };
    
    let analysis = improvement_engine.analyze_performance(&initial_result.kernel_code, &performance_info).unwrap();
    assert!(!analysis.bottlenecks.is_empty());
    assert!(!analysis.improvement_opportunities.is_empty());
    
    // Apply improvements
    let improvement_request = improvement::ImprovementRequest {
        original_code: initial_result.kernel_code,
        target_metrics: improvement::TargetMetrics {
            min_memory_bandwidth: 500.0,
            min_compute_utilization: 0.75,
            max_execution_time_ms: 20.0,
            min_occupancy: 0.80,
        },
        constraints: improvement::ImprovementConstraints::default(),
    };
    
    let improvement_result = improvement_engine.improve_kernel(&improvement_request).unwrap();
    if improvement_result.success {
        assert!(!improvement_result.improved_code.is_empty());
        assert!(!improvement_result.applied_optimizations.is_empty());
        assert!(improvement_result.performance_gain > 0.0);
    }
}

#[test]
    fn test_concurrent_synthesis_operations() -> Result<(), Box<dyn std::error::Error>> {
    // Test concurrent synthesis operations
    use std::sync::Arc;
    use std::thread;
    
    let synthesizer = Arc::new(synthesizer::KernelSynthesizer::new()?);
    let executor = Arc::new(executor::SynthesisExecutor::new()?);
    
    let mut handles = Vec::new();
    
    // Launch multiple concurrent synthesis operations
    for i in 0..4 {
        let synthesizer_clone = synthesizer.clone();
        let executor_clone = executor.clone();
        
        let handle = thread::spawn(move || {
            // Synthesizer test
            let spec = synthesizer::KernelSynthesisSpec {
                name: format!("concurrent_synth_{}", i),
                goal: format!("Process array {} with element-wise operations", i),
                input_types: vec![DataType::Float32],
                output_types: vec![DataType::Float32],
                input_shapes: vec![vec![1000 * (i + 1)]],
                output_shapes: vec![vec![1000 * (i + 1)]],
                algorithm_hints: vec![AlgorithmHint::ElementWise],
                performance_requirements: PerformanceRequirements::default(),
                constraints: KernelConstraints::default(),
            };
            
            let synth_result = synthesizer_clone.synthesize_kernel(&spec).unwrap();
            
            // Executor test
            let task = executor::SynthesisTask {
                id: format!("concurrent_exec_{}", i),
                goal: format!("Execute concurrent task {}", i),
                priority: executor::TaskPriority::Normal,
                input_data: vec![0.0; 100],
                expected_output_size: 100,
                timeout: Some(Duration::from_secs(10)),
            };
            
            let task_id = executor_clone.submit_task(task).unwrap();
            let exec_result = executor_clone.wait_for_completion(&task_id).unwrap();
            
            (synth_result.success, exec_result.success)
        });
        
        handles.push(handle);
    }
    
    // Wait for all operations to complete
    for handle in handles {
        let (synth_success, exec_success) = handle.join().map_err(|_| "Thread join error")?;
        assert!(synth_success);
        assert!(exec_success);
    }
}

#[test]
    fn test_memory_management_integration() -> Result<(), Box<dyn std::error::Error>> {
    // Test memory management across components
    let config = executor::ExecutorConfig {
        memory_limit_mb: 256, // Set memory limit
        ..Default::default()
    };
    let executor = executor::SynthesisExecutor::with_config(config)?;
    
    // Submit multiple memory-intensive tasks
    let mut task_ids = Vec::new();
    for i in 0..3 {
        let task = executor::SynthesisTask {
            id: format!("memory_test_{}", i),
            goal: "Large matrix operation".to_string(),
            priority: executor::TaskPriority::Normal,
            input_data: vec![0.0; 100000], // Large input data
            expected_output_size: 100000,
            timeout: Some(Duration::from_secs(30)),
        };
        
        let task_id = executor.submit_task(task).unwrap();
        task_ids.push(task_id);
    }
    
    // Monitor memory usage
    let initial_metrics = executor.get_resource_metrics().unwrap();
    
    // Wait for completion
    let results = executor.wait_for_all(&task_ids).unwrap();
    
    // All tasks should complete successfully or fail gracefully due to memory limits
    for result in results {
        // Either success or memory-related failure is acceptable
        if !result.success {
            // Check if failure was due to memory constraints
            assert!(result.error_message.is_some());
        }
    }
    
    let final_metrics = executor.get_resource_metrics().unwrap();
    assert!(final_metrics.memory_usage_mb <= 256.0 * 1.2); // Allow some overhead
}

#[test]
    fn test_performance_monitoring_integration() -> Result<(), Box<dyn std::error::Error>> {
    // Test performance monitoring across all components
    let config = synthesizer::SynthesizerConfig {
        profiling_enabled: true,
        ..Default::default()
    };
    let synthesizer = synthesizer::KernelSynthesizer::with_config(config)?;
    
    let pipeline_config = pipeline::PipelineConfig {
        enable_profiling: true,
        ..Default::default()
    };
    let pipeline = pipeline::SynthesisPipeline::with_config(pipeline_config).unwrap();
    
    let executor_config = executor::ExecutorConfig {
        enable_profiling: true,
        ..Default::default()
    };
    let executor = executor::SynthesisExecutor::with_config(executor_config).unwrap();
    
    // Test synthesizer profiling
    let spec = synthesizer::KernelSynthesisSpec {
        name: "profiling_test".to_string(),
        goal: "Test performance monitoring".to_string(),
        input_types: vec![DataType::Float32],
        output_types: vec![DataType::Float32],
        input_shapes: vec![vec![10000]],
        output_shapes: vec![vec![10000]],
        algorithm_hints: vec![AlgorithmHint::ElementWise],
        performance_requirements: PerformanceRequirements::default(),
        constraints: KernelConstraints::default(),
    };
    
    let synth_result = synthesizer.synthesize_kernel(&spec).unwrap();
    assert!(synth_result.profiling_data.is_some());
    
    // Test pipeline profiling
    let synthesis_request = pipeline::SynthesisRequest {
        id: "pipeline_profiling_test".to_string(),
        goal: "Test pipeline profiling".to_string(),
        input_specification: pipeline::InputSpecification::default(),
        output_specification: pipeline::OutputSpecification::default(),
        constraints: pipeline::SynthesisConstraints::default(),
        preferences: pipeline::SynthesisPreferences::default(),
    };
    
    let pipeline_result = pipeline.synthesize(&synthesis_request).unwrap();
    assert!(pipeline_result.profiling_data.is_some());
    
    // Test executor profiling
    let task = executor::SynthesisTask {
        id: "executor_profiling_test".to_string(),
        goal: "Test executor profiling".to_string(),
        priority: executor::TaskPriority::Normal,
        input_data: vec![0.0; 1000],
        expected_output_size: 1000,
        timeout: Some(Duration::from_secs(10)),
    };
    
    let task_id = executor.submit_task(task).unwrap();
    let exec_result = executor.wait_for_completion(&task_id).unwrap();
    assert!(exec_result.profiling_data.is_some());
    
    // Collect overall statistics
    let synth_stats = synthesizer.get_synthesis_statistics().unwrap();
    let pipeline_metrics = pipeline.get_pipeline_metrics().unwrap();
    let exec_stats = executor.get_statistics().unwrap();
    
    assert!(synth_stats.total_syntheses > 0);
    assert!(pipeline_metrics.total_requests > 0);
    assert!(exec_stats.total_tasks_executed > 0);
}