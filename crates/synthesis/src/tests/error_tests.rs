//! Tests for synthesis error handling

use crate::error::*;

#[test]
    fn test_synthesis_error_creation() -> Result<(), Box<dyn std::error::Error>> {
    let error = SynthesisError::InvalidGoal("Test error".to_string());
    assert_eq!(error.to_string(), "Invalid goal: Test error");
}

#[test]
    fn test_synthesis_error_types() -> Result<(), Box<dyn std::error::Error>> {
    let errors = vec![
        SynthesisError::InvalidGoal("Empty goal".to_string()),
        SynthesisError::CompilationError("NVCC failed".to_string()),
        SynthesisError::OptimizationError("No viable optimization".to_string()),
        SynthesisError::TemplateError("Template not found".to_string()),
        SynthesisError::ResourceError("Insufficient memory".to_string()),
        SynthesisError::RuntimeError("Kernel launch failed".to_string()),
        SynthesisError::ValidationError("Invalid parameters".to_string()),
        SynthesisError::ConfigurationError("Invalid configuration".to_string()),
    ];
    
    for error in errors {
        assert!(!error.to_string().is_empty());
        assert!(error.to_string().len() > 5);
    }
}

#[test]
    fn test_synthesis_result_ok() -> Result<(), Box<dyn std::error::Error>> {
    let result: SynthesisResult<i32> = Ok(42);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 42);
}

#[test]
    fn test_synthesis_result_err() -> Result<(), Box<dyn std::error::Error>> {
    let result: SynthesisResult<i32> = Err(SynthesisError::InvalidGoal("Test".to_string()));
    assert!(result.is_err());
    
    match result {
        Err(SynthesisError::InvalidGoal(msg)) => assert_eq!(msg, "Test"),
        _ => panic!("Unexpected error type"),
    }
}

#[test]
    fn test_error_chaining() -> Result<(), Box<dyn std::error::Error>> {
    let root_cause = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
    let error = SynthesisError::CompilationError(format!("Failed to read file: {}", root_cause));
    
    assert!(error.to_string().contains("File not found"));
}

#[test]
    fn test_error_from_conversions() -> Result<(), Box<dyn std::error::Error>> {
    // Test conversion from std::io::Error
    let io_error = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "Access denied");
    let synthesis_error: SynthesisError = io_error.into();
    
    match synthesis_error {
        SynthesisError::RuntimeError(msg) => assert!(msg.contains("Access denied")),
        _ => panic!("Unexpected error type"),
    }
}

#[test]
    fn test_error_debug_formatting() -> Result<(), Box<dyn std::error::Error>> {
    let error = SynthesisError::OptimizationError("Optimization failed".to_string());
    let debug_str = format!("{:?}", error);
    
    assert!(debug_str.contains("OptimizationError"));
    assert!(debug_str.contains("Optimization failed"));
}

#[test]
    fn test_error_display_formatting() -> Result<(), Box<dyn std::error::Error>> {
    let error = SynthesisError::ValidationError("Parameter out of range".to_string());
    let display_str = format!("{}", error);
    
    assert!(display_str.contains("Validation error"));
    assert!(display_str.contains("Parameter out of range"));
}

#[test]
    fn test_compilation_error_details() -> Result<(), Box<dyn std::error::Error>> {
    let compilation_details = CompilationErrorDetails {
        source_file: Some("kernel.cu".to_string()),
        line_number: Some(42),
        column_number: Some(15),
        error_message: "Expected ';' before '}'".to_string(),
        error_code: Some("C2143".to_string()),
        suggestions: vec!["Add semicolon".to_string()],
    };
    
    let error = SynthesisError::CompilationErrorDetailed(compilation_details);
    let error_str = error.to_string();
    
    assert!(error_str.contains("kernel.cu"));
    assert!(error_str.contains("42"));
    assert!(error_str.contains("15"));
    assert!(error_str.contains("Expected ';'"));
}

#[test]
    fn test_optimization_error_context() -> Result<(), Box<dyn std::error::Error>> {
    let optimization_context = OptimizationErrorContext {
        algorithm_type: AlgorithmType::MatrixOperation,
        target_optimization: OptimizationLevel::Aggressive,
        attempted_techniques: vec![
            "Loop unrolling".to_string(),
            "Memory coalescing".to_string(),
        ],
        resource_constraints: vec![
            "Shared memory limit exceeded".to_string(),
        ],
        suggestions: vec![
            "Reduce block size".to_string(),
            "Use register blocking".to_string(),
        ],
    };
    
    let error = SynthesisError::OptimizationErrorDetailed(optimization_context);
    let error_str = error.to_string();
    
    assert!(error_str.contains("MatrixOperation"));
    assert!(error_str.contains("Aggressive"));
    assert!(error_str.contains("Loop unrolling"));
    assert!(error_str.contains("Shared memory"));
}

#[test]
    fn test_template_error_details() -> Result<(), Box<dyn std::error::Error>> {
    let template_error = TemplateErrorDetails {
        template_name: "matrix_multiply_blocked".to_string(),
        parameter_errors: vec![
            ("BLOCK_SIZE".to_string(), "Must be power of 2".to_string()),
            ("DATA_TYPE".to_string(), "Unsupported type".to_string()),
        ],
        missing_specializations: vec![
            "float".to_string(),
            "double".to_string(),
        ],
        conflicting_constraints: vec![
            "Block size conflicts with shared memory limit".to_string(),
        ],
    };
    
    let error = SynthesisError::TemplateErrorDetailed(template_error);
    let error_str = error.to_string();
    
    assert!(error_str.contains("matrix_multiply_blocked"));
    assert!(error_str.contains("BLOCK_SIZE"));
    assert!(error_str.contains("power of 2"));
    assert!(error_str.contains("float"));
}

#[test]
    fn test_resource_error_breakdown() -> Result<(), Box<dyn std::error::Error>> {
    let resource_error = ResourceErrorDetails {
        resource_type: ResourceType::Memory,
        requested_amount: 1024 * 1024 * 1024, // 1GB
        available_amount: 512 * 1024 * 1024,  // 512MB
        peak_usage: 768 * 1024 * 1024,        // 768MB
        allocation_strategy: "Best fit".to_string(),
        recommendations: vec![
            "Reduce batch size".to_string(),
            "Use memory streaming".to_string(),
        ],
    };
    
    let error = SynthesisError::ResourceErrorDetailed(resource_error);
    let error_str = error.to_string();
    
    assert!(error_str.contains("Memory"));
    assert!(error_str.contains("1073741824")); // 1GB in bytes
    assert!(error_str.contains("536870912"));  // 512MB in bytes
    assert!(error_str.contains("Best fit"));
}

#[test]
    fn test_runtime_error_context() -> Result<(), Box<dyn std::error::Error>> {
    let runtime_context = RuntimeErrorContext {
        kernel_name: "vector_add".to_string(),
        launch_config: LaunchConfigInfo {
            grid_dim: (256, 1, 1),
            block_dim: (128, 1, 1),
            shared_memory: 1024,
            stream_id: Some(42),
        },
        cuda_error_code: Some(700), // CUDA_ERROR_ILLEGAL_ADDRESS
        host_error_info: Some("Segmentation fault".to_string()),
        device_error_info: Some("Invalid memory access".to_string()),
        stack_trace: vec![
            "kernel.cu:15 in vector_add".to_string(),
            "main.cu:42 in main".to_string(),
        ],
    };
    
    let error = SynthesisError::RuntimeErrorDetailed(runtime_context);
    let error_str = error.to_string();
    
    assert!(error_str.contains("vector_add"));
    assert!(error_str.contains("256"));
    assert!(error_str.contains("128"));
    assert!(error_str.contains("700"));
    assert!(error_str.contains("Invalid memory"));
}

#[test]
    fn test_validation_error_details() -> Result<(), Box<dyn std::error::Error>> {
    let validation_error = ValidationErrorDetails {
        parameter_name: "grid_dimensions".to_string(),
        provided_value: "(-1, 0, 65536)".to_string(),
        expected_range: "1 <= x <= 65535 for each dimension".to_string(),
        validation_rules: vec![
            "All dimensions must be positive".to_string(),
            "Maximum dimension is 65535".to_string(),
        ],
        related_parameters: vec![
            "block_dimensions".to_string(),
            "total_threads".to_string(),
        ],
    };
    
    let error = SynthesisError::ValidationErrorDetailed(validation_error);
    let error_str = error.to_string();
    
    assert!(error_str.contains("grid_dimensions"));
    assert!(error_str.contains("-1"));
    assert!(error_str.contains("65536"));
    assert!(error_str.contains("positive"));
}

#[test]
    fn test_configuration_error_context() -> Result<(), Box<dyn std::error::Error>> {
    let config_error = ConfigurationErrorContext {
        setting_name: "optimization_level".to_string(),
        current_value: "Ultra".to_string(),
        valid_values: vec![
            "Debug".to_string(),
            "Fast".to_string(),
            "Balanced".to_string(),
            "Aggressive".to_string(),
            "Size".to_string(),
        ],
        dependencies: vec![
            ("target_arch".to_string(), "sm_80".to_string()),
        ],
        conflicts: vec![
            "Cannot use Ultra with debug symbols enabled".to_string(),
        ],
    };
    
    let error = SynthesisError::ConfigurationErrorDetailed(config_error);
    let error_str = error.to_string();
    
    assert!(error_str.contains("optimization_level"));
    assert!(error_str.contains("Ultra"));
    assert!(error_str.contains("Debug"));
    assert!(error_str.contains("sm_80"));
}

#[test]
    fn test_error_severity_levels() -> Result<(), Box<dyn std::error::Error>> {
    let errors = vec![
        (SynthesisError::ValidationError("Minor issue".to_string()), ErrorSeverity::Warning),
        (SynthesisError::CompilationError("Syntax error".to_string()), ErrorSeverity::Error),
        (SynthesisError::RuntimeError("Kernel crash".to_string()), ErrorSeverity::Critical),
        (SynthesisError::ResourceError("Out of memory".to_string()), ErrorSeverity::Critical),
    ];
    
    for (error, expected_severity) in errors {
        assert_eq!(error.severity(), expected_severity);
    }
}

#[test]
    fn test_error_categorization() -> Result<(), Box<dyn std::error::Error>> {
    let errors = vec![
        (SynthesisError::InvalidGoal("Empty".to_string()), ErrorCategory::Input),
        (SynthesisError::CompilationError("NVCC".to_string()), ErrorCategory::Compilation),
        (SynthesisError::OptimizationError("Failed".to_string()), ErrorCategory::Optimization),
        (SynthesisError::RuntimeError("Crash".to_string()), ErrorCategory::Runtime),
        (SynthesisError::ResourceError("Memory".to_string()), ErrorCategory::Resource),
    ];
    
    for (error, expected_category) in errors {
        assert_eq!(error.category(), expected_category);
    }
}

#[test]
    fn test_error_recovery_suggestions() -> Result<(), Box<dyn std::error::Error>> {
    let error = SynthesisError::CompilationError("Unknown identifier 'invalid_func'".to_string());
    let suggestions = error.recovery_suggestions();
    
    assert!(!suggestions.is_empty());
    assert!(suggestions.iter().any(|s| s.contains("Check spelling")));
}

#[test]
    fn test_error_serialization() -> Result<(), Box<dyn std::error::Error>> {
    let error = SynthesisError::OptimizationError("Test optimization error".to_string());
    
    let serialized = serde_json::to_string(&error).unwrap();
    assert!(serialized.contains("OptimizationError"));
    assert!(serialized.contains("Test optimization error"));
    
    let deserialized: SynthesisError = serde_json::from_str(&serialized)?;
    match deserialized {
        SynthesisError::OptimizationError(msg) => assert_eq!(msg, "Test optimization error"),
        _ => panic!("Deserialization failed"),
    }
}

#[test]
    fn test_error_aggregation() -> Result<(), Box<dyn std::error::Error>> {
    let errors = vec![
        SynthesisError::ValidationError("Invalid parameter 1".to_string()),
        SynthesisError::ValidationError("Invalid parameter 2".to_string()),
        SynthesisError::CompilationError("Syntax error".to_string()),
    ];
    
    let aggregated = SynthesisError::Multiple(errors);
    let error_str = aggregated.to_string();
    
    assert!(error_str.contains("Multiple errors"));
    assert!(error_str.contains("parameter 1"));
    assert!(error_str.contains("parameter 2"));
    assert!(error_str.contains("Syntax error"));
}

#[test]
    fn test_error_location_tracking() -> Result<(), Box<dyn std::error::Error>> {
    let location = ErrorLocation {
        file: "synthesis.rs".to_string(),
        line: 42,
        column: 15,
        function: Some("interpret_goal".to_string()),
    };
    
    let error = SynthesisError::InvalidGoal("Test error".to_string())
        .with_location(location);
    
    let error_str = error.to_string();
    assert!(error_str.contains("synthesis.rs"));
    assert!(error_str.contains("42"));
    assert!(error_str.contains("interpret_goal"));
}

#[test]
    fn test_error_timing_information() -> Result<(), Box<dyn std::error::Error>> {
    let error = SynthesisError::CompilationError("Timeout".to_string())
        .with_timestamp(std::time::SystemTime::now())
        .with_duration(std::time::Duration::from_secs(30));
    
    let error_str = error.to_string();
    assert!(error_str.contains("30"));
    assert!(error_str.contains("seconds") || error_str.contains("s"));
}

#[test]
    fn test_custom_error_extensions() -> Result<(), Box<dyn std::error::Error>> {
    let mut error = SynthesisError::RuntimeError("Base error".to_string());
    
    error.add_context("device_id", "0");
    error.add_context("memory_usage", "85%");
    error.add_context("temperature", "78C");
    
    let error_str = error.to_string();
    assert!(error_str.contains("device_id"));
    assert!(error_str.contains("85%"));
    assert!(error_str.contains("78C"));
}

#[test]
    fn test_error_filtering() -> Result<(), Box<dyn std::error::Error>> {
    let errors = vec![
        SynthesisError::ValidationError("Warning level".to_string()),
        SynthesisError::CompilationError("Error level".to_string()),
        SynthesisError::RuntimeError("Critical level".to_string()),
    ];
    
    let critical_errors: Vec<_> = errors.into_iter()
        .filter(|e| e.severity() == ErrorSeverity::Critical)
        .collect();
    
    assert_eq!(critical_errors.len(), 1);
    match &critical_errors[0] {
        SynthesisError::RuntimeError(msg) => assert_eq!(msg, "Critical level"),
        _ => panic!("Unexpected error type"),
    }
}

#[test]
    fn test_error_metrics() -> Result<(), Box<dyn std::error::Error>> {
    let errors = vec![
        SynthesisError::ValidationError("Error 1".to_string()),
        SynthesisError::ValidationError("Error 2".to_string()),
        SynthesisError::CompilationError("Error 3".to_string()),
        SynthesisError::RuntimeError("Error 4".to_string()),
    ];
    
    let metrics = ErrorMetrics::from_errors(&errors);
    
    assert_eq!(metrics.total_count, 4);
    assert_eq!(metrics.validation_errors, 2);
    assert_eq!(metrics.compilation_errors, 1);
    assert_eq!(metrics.runtime_errors, 1);
    assert_eq!(metrics.most_common_category, ErrorCategory::Input);
}