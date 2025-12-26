//! Tests for synthesis pipeline module

use crate::pipeline::*;
use crate::error::SynthesisResult;

#[test]
    fn test_synthesis_pipeline_creation() -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = SynthesisPipeline::new().unwrap();
    assert!(pipeline.is_initialized());
    assert_eq!(pipeline.get_status(), PipelineStatus::Ready);
}

#[test]
    fn test_pipeline_configuration() -> Result<(), Box<dyn std::error::Error>> {
    let config = PipelineConfig {
        enable_caching: true,
        cache_size_mb: 256,
        max_concurrent_stages: 4,
        timeout_seconds: 600,
        enable_profiling: true,
        optimization_level: OptimizationLevel::Aggressive,
        target_architecture: "sm_80".to_string(),
        fallback_on_failure: true,
    };
    
    let pipeline = SynthesisPipeline::with_config(config.clone()).unwrap();
    let retrieved_config = pipeline.get_config();
    
    assert_eq!(retrieved_config.enable_caching, config.enable_caching);
    assert_eq!(retrieved_config.cache_size_mb, config.cache_size_mb);
    assert_eq!(retrieved_config.max_concurrent_stages, config.max_concurrent_stages);
    assert_eq!(retrieved_config.optimization_level, config.optimization_level);
}

#[test]
    fn test_stage_registration() -> Result<(), Box<dyn std::error::Error>> {
    let mut pipeline = SynthesisPipeline::new().unwrap();
    
    let stages = vec![
        PipelineStage::GoalInterpretation,
        PipelineStage::TemplateSelection,
        PipelineStage::CodeGeneration,
        PipelineStage::Compilation,
        PipelineStage::Optimization,
        PipelineStage::Validation,
        PipelineStage::Deployment,
    ];
    
    for stage in stages {
        pipeline.register_stage(stage).unwrap();
    }
    
    let registered_stages = pipeline.get_registered_stages();
    assert_eq!(registered_stages.len(), 7);
}

#[test]
    fn test_simple_synthesis_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = SynthesisPipeline::new().unwrap();
    
    let synthesis_request = SynthesisRequest {
        id: "simple_pipeline_test".to_string(),
        goal: "Add two vectors element-wise".to_string(),
        input_specification: InputSpecification {
            input_types: vec!["float*".to_string(), "float*".to_string()],
            input_sizes: vec![1000, 1000],
            data_layout: DataLayout::Contiguous,
        },
        output_specification: OutputSpecification {
            output_types: vec!["float*".to_string()],
            output_sizes: vec![1000],
            data_layout: DataLayout::Contiguous,
        },
        constraints: SynthesisConstraints {
            max_execution_time_ms: 100.0,
            max_memory_usage_mb: 128,
            target_accuracy: 1e-6,
            preserve_precision: true,
        },
        preferences: SynthesisPreferences {
            optimize_for: OptimizationTarget::Performance,
            allow_approximations: false,
            debug_mode: false,
        },
    };
    
    let result = pipeline.synthesize(&synthesis_request).unwrap();
    
    assert!(result.success);
    assert!(!result.generated_code.is_empty());
    assert!(!result.execution_plan.is_empty());
    assert!(result.estimated_performance.execution_time_ms > 0.0);
}

#[test]
    fn test_matrix_multiplication_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = SynthesisPipeline::new().unwrap();
    
    let synthesis_request = SynthesisRequest {
        id: "matrix_mult_pipeline".to_string(),
        goal: "Multiply two 1024x1024 matrices".to_string(),
        input_specification: InputSpecification {
            input_types: vec!["float*".to_string(), "float*".to_string()],
            input_sizes: vec![1024 * 1024, 1024 * 1024],
            data_layout: DataLayout::RowMajor,
        },
        output_specification: OutputSpecification {
            output_types: vec!["float*".to_string()],
            output_sizes: vec![1024 * 1024],
            data_layout: DataLayout::RowMajor,
        },
        constraints: SynthesisConstraints {
            max_execution_time_ms: 50.0,
            max_memory_usage_mb: 512,
            target_accuracy: 1e-5,
            preserve_precision: true,
        },
        preferences: SynthesisPreferences {
            optimize_for: OptimizationTarget::Performance,
            allow_approximations: false,
            debug_mode: false,
        },
    };
    
    let result = pipeline.synthesize(&synthesis_request).unwrap();
    
    assert!(result.success);
    assert!(result.generated_code.contains("matrix") || result.generated_code.contains("mul"));
    assert!(!result.stage_results.is_empty());
    
    // Should have template selection and optimization stages
    let has_template_stage = result.stage_results.iter()
        .any(|stage| stage.stage_type == PipelineStage::TemplateSelection);
    let has_optimization_stage = result.stage_results.iter()
        .any(|stage| stage.stage_type == PipelineStage::Optimization);
    
    assert!(has_template_stage);
    assert!(has_optimization_stage);
}

#[test]
    fn test_reduction_operation_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = SynthesisPipeline::new().unwrap();
    
    let synthesis_request = SynthesisRequest {
        id: "reduction_pipeline".to_string(),
        goal: "Sum all elements in an array of 1 million floats".to_string(),
        input_specification: InputSpecification {
            input_types: vec!["float*".to_string()],
            input_sizes: vec![1000000],
            data_layout: DataLayout::Contiguous,
        },
        output_specification: OutputSpecification {
            output_types: vec!["float".to_string()],
            output_sizes: vec![1],
            data_layout: DataLayout::Scalar,
        },
        constraints: SynthesisConstraints {
            max_execution_time_ms: 10.0,
            max_memory_usage_mb: 64,
            target_accuracy: 1e-4,
            preserve_precision: false, // Allow some precision loss for performance
        },
        preferences: SynthesisPreferences {
            optimize_for: OptimizationTarget::Performance,
            allow_approximations: true,
            debug_mode: false,
        },
    };
    
    let result = pipeline.synthesize(&synthesis_request).unwrap();
    
    assert!(result.success);
    assert!(result.generated_code.contains("sum") || result.generated_code.contains("reduce"));
    
    // Should use shared memory for reduction
    assert!(result.generated_code.contains("__shared__"));
}

#[test]
    fn test_convolution_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = SynthesisPipeline::new().unwrap();
    
    let synthesis_request = SynthesisRequest {
        id: "convolution_pipeline".to_string(),
        goal: "Apply 3x3 convolution filter to 512x512 image".to_string(),
        input_specification: InputSpecification {
            input_types: vec!["float*".to_string(), "float*".to_string()], // image and filter
            input_sizes: vec![512 * 512, 3 * 3],
            data_layout: DataLayout::RowMajor,
        },
        output_specification: OutputSpecification {
            output_types: vec!["float*".to_string()],
            output_sizes: vec![510 * 510], // Output size after convolution
            data_layout: DataLayout::RowMajor,
        },
        constraints: SynthesisConstraints {
            max_execution_time_ms: 25.0,
            max_memory_usage_mb: 256,
            target_accuracy: 1e-6,
            preserve_precision: true,
        },
        preferences: SynthesisPreferences {
            optimize_for: OptimizationTarget::Balanced,
            allow_approximations: false,
            debug_mode: false,
        },
    };
    
    let result = pipeline.synthesize(&synthesis_request).unwrap();
    
    assert!(result.success);
    assert!(result.generated_code.contains("conv") || result.generated_code.contains("filter"));
    
    // Should use shared memory for tile-based processing
    assert!(result.generated_code.contains("__shared__"));
}

#[test]
    fn test_pipeline_stage_dependencies() -> Result<(), Box<dyn std::error::Error>> {
    let mut pipeline = SynthesisPipeline::new().unwrap();
    
    // Create custom stage dependency chain
    let stage_dependencies = vec![
        (PipelineStage::GoalInterpretation, vec![]),
        (PipelineStage::TemplateSelection, vec![PipelineStage::GoalInterpretation]),
        (PipelineStage::CodeGeneration, vec![PipelineStage::TemplateSelection]),
        (PipelineStage::Compilation, vec![PipelineStage::CodeGeneration]),
        (PipelineStage::Optimization, vec![PipelineStage::Compilation]),
        (PipelineStage::Validation, vec![PipelineStage::Optimization]),
    ];
    
    for (stage, deps) in stage_dependencies {
        pipeline.set_stage_dependencies(stage, deps).unwrap();
    }
    
    let synthesis_request = SynthesisRequest {
        id: "dependency_test".to_string(),
        goal: "Test stage dependencies".to_string(),
        input_specification: InputSpecification::default(),
        output_specification: OutputSpecification::default(),
        constraints: SynthesisConstraints::default(),
        preferences: SynthesisPreferences::default(),
    };
    
    let result = pipeline.synthesize(&synthesis_request).unwrap();
    
    // Stages should execute in dependency order
    let stage_execution_order: Vec<_> = result.stage_results.iter()
        .map(|r| r.stage_type)
        .collect();
    
    let goal_index = stage_execution_order.iter().position(|&s| s == PipelineStage::GoalInterpretation);
    let template_index = stage_execution_order.iter().position(|&s| s == PipelineStage::TemplateSelection);
    let code_index = stage_execution_order.iter().position(|&s| s == PipelineStage::CodeGeneration);
    
    assert!(goal_index.is_ok() < template_index.unwrap());
    assert!(template_index.is_ok() < code_index.unwrap());
}

#[test]
    fn test_parallel_stage_execution() -> Result<(), Box<dyn std::error::Error>> {
    let config = PipelineConfig {
        max_concurrent_stages: 3,
        ..PipelineConfig::default()
    };
    let pipeline = SynthesisPipeline::with_config(config)?;
    
    let synthesis_request = SynthesisRequest {
        id: "parallel_execution_test".to_string(),
        goal: "Test parallel stage execution".to_string(),
        input_specification: InputSpecification::default(),
        output_specification: OutputSpecification::default(),
        constraints: SynthesisConstraints::default(),
        preferences: SynthesisPreferences::default(),
    };
    
    let start_time = std::time::Instant::now();
    let result = pipeline.synthesize(&synthesis_request).unwrap();
    let total_time = start_time.elapsed();
    
    assert!(result.success);
    
    // With parallel execution, total time should be less than sum of individual stage times
    let sum_of_stage_times: std::time::Duration = result.stage_results.iter()
        .map(|r| r.execution_time)
        .sum();
    
    assert!(total_time <= sum_of_stage_times);
}

#[test]
    fn test_pipeline_caching() -> Result<(), Box<dyn std::error::Error>> {
    let config = PipelineConfig {
        enable_caching: true,
        cache_size_mb: 128,
        ..PipelineConfig::default()
    };
    let pipeline = SynthesisPipeline::with_config(config)?;
    
    let synthesis_request = SynthesisRequest {
        id: "cache_test_1".to_string(),
        goal: "Identical operation for caching test".to_string(),
        input_specification: InputSpecification::default(),
        output_specification: OutputSpecification::default(),
        constraints: SynthesisConstraints::default(),
        preferences: SynthesisPreferences::default(),
    };
    
    // First execution
    let start_1 = std::time::Instant::now();
    let result_1 = pipeline.synthesize(&synthesis_request).unwrap();
    let duration_1 = start_1.elapsed();
    
    // Second execution with same request (should use cache)
    let mut request_2 = synthesis_request;
    request_2.id = "cache_test_2".to_string();
    
    let start_2 = std::time::Instant::now();
    let result_2 = pipeline.synthesize(&request_2).unwrap();
    let duration_2 = start_2.elapsed();
    
    assert!(result_1.success && result_2.success);
    assert_eq!(result_1.generated_code, result_2.generated_code);
    
    // Second execution should be faster due to caching
    assert!(duration_2 <= duration_1);
}

#[test]
    fn test_pipeline_error_handling() -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = SynthesisPipeline::new().unwrap();
    
    let invalid_request = SynthesisRequest {
        id: "error_test".to_string(),
        goal: "".to_string(), // Empty goal should cause error
        input_specification: InputSpecification::default(),
        output_specification: OutputSpecification::default(),
        constraints: SynthesisConstraints::default(),
        preferences: SynthesisPreferences::default(),
    };
    
    let result = pipeline.synthesize(&invalid_request);
    
    // Should fail gracefully
    assert!(result.is_err());
}

#[test]
    fn test_pipeline_fallback_mechanism() -> Result<(), Box<dyn std::error::Error>> {
    let config = PipelineConfig {
        fallback_on_failure: true,
        ..PipelineConfig::default()
    };
    let pipeline = SynthesisPipeline::with_config(config)?;
    
    let challenging_request = SynthesisRequest {
        id: "fallback_test".to_string(),
        goal: "Extremely complex operation that might fail".to_string(),
        input_specification: InputSpecification::default(),
        output_specification: OutputSpecification::default(),
        constraints: SynthesisConstraints {
            max_execution_time_ms: 0.001, // Unrealistic constraint
            max_memory_usage_mb: 1, // Very limited memory
            target_accuracy: 1e-15, // Extremely high precision
            preserve_precision: true,
        },
        preferences: SynthesisPreferences::default(),
    };
    
    let result = pipeline.synthesize(&challenging_request).unwrap();
    
    // Should either succeed with fallback or provide meaningful partial result
    if result.success {
        assert!(!result.generated_code.is_empty());
    } else {
        assert!(!result.fallback_results.is_empty());
    }
}

#[test]
    fn test_pipeline_profiling() -> Result<(), Box<dyn std::error::Error>> {
    let config = PipelineConfig {
        enable_profiling: true,
        ..PipelineConfig::default()
    };
    let pipeline = SynthesisPipeline::with_config(config)?;
    
    let synthesis_request = SynthesisRequest {
        id: "profiling_test".to_string(),
        goal: "Profile pipeline execution".to_string(),
        input_specification: InputSpecification::default(),
        output_specification: OutputSpecification::default(),
        constraints: SynthesisConstraints::default(),
        preferences: SynthesisPreferences::default(),
    };
    
    let result = pipeline.synthesize(&synthesis_request).unwrap();
    
    assert!(result.success);
    assert!(result.profiling_data.is_some());
    
    let profiling = result.profiling_data.unwrap();
    assert!(profiling.total_execution_time.as_millis() > 0);
    assert!(!profiling.stage_timings.is_empty());
    assert!(profiling.memory_peak_usage_mb > 0.0);
    assert!(profiling.cache_hit_rate >= 0.0);
}

#[test]
    fn test_batch_synthesis() -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = SynthesisPipeline::new().unwrap();
    
    let batch_requests = (0..5).map(|i| SynthesisRequest {
        id: format!("batch_request_{}", i),
        goal: format!("Process array {} with element-wise operations", i),
        input_specification: InputSpecification {
            input_types: vec!["float*".to_string()],
            input_sizes: vec![1000 * (i + 1)],
            data_layout: DataLayout::Contiguous,
        },
        output_specification: OutputSpecification {
            output_types: vec!["float*".to_string()],
            output_sizes: vec![1000 * (i + 1)],
            data_layout: DataLayout::Contiguous,
        },
        constraints: SynthesisConstraints::default(),
        preferences: SynthesisPreferences::default(),
    }).collect();
    
    let results = pipeline.synthesize_batch(batch_requests).unwrap();
    
    assert_eq!(results.len(), 5);
    for result in results {
        assert!(result.success);
        assert!(!result.generated_code.is_empty());
    }
}

#[test]
    fn test_pipeline_state_management() -> Result<(), Box<dyn std::error::Error>> {
    let mut pipeline = SynthesisPipeline::new().unwrap();
    
    assert_eq!(pipeline.get_status(), PipelineStatus::Ready);
    
    // Pause pipeline
    pipeline.pause()?;
    assert_eq!(pipeline.get_status(), PipelineStatus::Paused);
    
    // Resume pipeline
    pipeline.resume()?;
    assert_eq!(pipeline.get_status(), PipelineStatus::Ready);
    
    // Reset pipeline state
    pipeline.reset().unwrap();
    assert_eq!(pipeline.get_status(), PipelineStatus::Ready);
}

#[test]
    fn test_pipeline_resource_monitoring() -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = SynthesisPipeline::new().unwrap();
    
    let initial_metrics = pipeline.get_resource_metrics().unwrap();
    assert_eq!(initial_metrics.active_syntheses, 0);
    
    // Start a synthesis in background
    let synthesis_request = SynthesisRequest {
        id: "resource_monitor_test".to_string(),
        goal: "Monitor resource usage during synthesis".to_string(),
        input_specification: InputSpecification::default(),
        output_specification: OutputSpecification::default(),
        constraints: SynthesisConstraints::default(),
        preferences: SynthesisPreferences::default(),
    };
    
    let _result = pipeline.synthesize(&synthesis_request).unwrap();
    
    let final_metrics = pipeline.get_resource_metrics().unwrap();
    assert!(final_metrics.total_syntheses_completed >= 1);
}

#[test]
    fn test_custom_stage_implementation() -> Result<(), Box<dyn std::error::Error>> {
    let mut pipeline = SynthesisPipeline::new().unwrap();
    
    let custom_stage = CustomPipelineStage {
        name: "CustomOptimization".to_string(),
        description: "Custom optimization stage".to_string(),
        implementation: Box::new(|input| {
            Ok(StageResult {
                stage_type: PipelineStage::Custom("CustomOptimization".to_string()),
                success: true,
                output_data: input.code.replace("float", "double"), // Simple transformation
                execution_time: std::time::Duration::from_millis(10),
                memory_usage_mb: 5.0,
                metadata: vec![("transformation".to_string(), "float_to_double".to_string())],
            })
        }),
        dependencies: vec![PipelineStage::CodeGeneration],
    };
    
    pipeline.register_custom_stage(custom_stage).unwrap();
    
    let synthesis_request = SynthesisRequest {
        id: "custom_stage_test".to_string(),
        goal: "Test custom stage implementation".to_string(),
        input_specification: InputSpecification::default(),
        output_specification: OutputSpecification::default(),
        constraints: SynthesisConstraints::default(),
        preferences: SynthesisPreferences::default(),
    };
    
    let result = pipeline.synthesize(&synthesis_request).unwrap();
    
    assert!(result.success);
    
    // Should have executed custom stage
    let has_custom_stage = result.stage_results.iter()
        .any(|r| matches!(r.stage_type, PipelineStage::Custom(ref name) if name == "CustomOptimization"));
    assert!(has_custom_stage);
}

#[test]
    fn test_pipeline_metrics_collection() -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = SynthesisPipeline::new().unwrap();
    
    // Execute multiple synthesis requests
    for i in 0..3 {
        let synthesis_request = SynthesisRequest {
            id: format!("metrics_test_{}", i),
            goal: "Collect pipeline metrics".to_string(),
            input_specification: InputSpecification::default(),
            output_specification: OutputSpecification::default(),
            constraints: SynthesisConstraints::default(),
            preferences: SynthesisPreferences::default(),
        };
        
        let _result = pipeline.synthesize(&synthesis_request);
    }
    
    let metrics = pipeline.get_pipeline_metrics().unwrap();
    
    assert!(metrics.total_requests >= 3);
    assert!(metrics.successful_requests <= metrics.total_requests);
    assert!(metrics.average_execution_time.as_millis() > 0);
    assert!(metrics.stage_success_rates.len() > 0);
    
    for (stage, success_rate) in metrics.stage_success_rates {
        assert!(success_rate >= 0.0 && success_rate <= 1.0);
    }
}

#[test]
    fn test_pipeline_configuration_validation() -> Result<(), Box<dyn std::error::Error>> {
    // Test invalid configuration
    let invalid_config = PipelineConfig {
        max_concurrent_stages: 0, // Invalid: must be > 0
        cache_size_mb: -1, // Invalid: must be >= 0
        timeout_seconds: 0, // Invalid: must be > 0
        ..PipelineConfig::default()
    };
    
    let result = SynthesisPipeline::with_config(invalid_config);
    assert!(result.is_err());
    
    // Test valid configuration
    let valid_config = PipelineConfig {
        max_concurrent_stages: 2,
        cache_size_mb: 64,
        timeout_seconds: 300,
        ..PipelineConfig::default()
    };
    
    let result = SynthesisPipeline::with_config(valid_config);
    assert!(result.is_ok());
}