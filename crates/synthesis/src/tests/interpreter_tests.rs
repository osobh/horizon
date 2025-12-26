//! Tests for goal interpretation module

use crate::interpreter::*;
use crate::error::SynthesisResult;

#[test]
    fn test_goal_interpreter_creation() -> Result<(), Box<dyn std::error::Error>> {
    let interpreter = GoalInterpreter::new();
    assert!(interpreter.is_initialized());
}

#[test]
    fn test_kernel_specification_creation() -> Result<(), Box<dyn std::error::Error>> {
    let spec = KernelSpecification {
        name: "vector_add".to_string(),
        input_types: vec!["float*".to_string(), "float*".to_string()],
        output_types: vec!["float*".to_string()],
        grid_dimensions: (256, 1, 1),
        block_dimensions: (128, 1, 1),
        shared_memory_bytes: 0,
        registers_per_thread: None,
        compute_capability: (8, 0),
        optimization_level: OptimizationLevel::Balanced,
        memory_pattern: MemoryPattern::Streaming,
        algorithm_type: AlgorithmType::ElementWise,
    };
    
    assert_eq!(spec.name, "vector_add");
    assert_eq!(spec.input_types.len(), 2);
    assert_eq!(spec.output_types.len(), 1);
    assert_eq!(spec.grid_dimensions, (256, 1, 1));
}

#[test]
    fn test_goal_interpretation_basic() -> Result<(), Box<dyn std::error::Error>> {
    let interpreter = GoalInterpreter::new();
    
    let goal = "Add two vectors of size 1000 element-wise";
    let spec = interpreter.interpret_goal(goal).unwrap();
    
    assert!(spec.name.contains("vector") || spec.name.contains("add"));
    assert_eq!(spec.algorithm_type, AlgorithmType::ElementWise);
    assert!(spec.input_types.len() >= 2);
    assert_eq!(spec.output_types.len(), 1);
}

#[test]
    fn test_goal_interpretation_matrix_operations() -> Result<(), Box<dyn std::error::Error>> {
    let interpreter = GoalInterpreter::new();
    
    let goals = vec![
        "Multiply two 512x512 matrices",
        "Compute matrix transpose of 1024x1024 matrix",
        "Perform matrix-vector multiplication",
    ];
    
    for goal in goals {
        let spec = interpreter.interpret_goal(goal)?;
        assert_eq!(spec.algorithm_type, AlgorithmType::MatrixOperation);
        assert!(spec.name.contains("matrix"));
    }
}

#[test]
    fn test_goal_interpretation_reduction_operations() -> Result<(), Box<dyn std::error::Error>> {
    let interpreter = GoalInterpreter::new();
    
    let goals = vec![
        "Sum all elements in an array of 10000 floats",
        "Find maximum value in vector",
        "Compute dot product of two vectors",
        "Calculate mean of array elements",
    ];
    
    for goal in goals {
        let spec = interpreter.interpret_goal(goal).unwrap();
        assert_eq!(spec.algorithm_type, AlgorithmType::Reduction);
        assert!(spec.memory_pattern == MemoryPattern::Random || 
                spec.memory_pattern == MemoryPattern::Streaming);
    }
}

#[test]
    fn test_goal_interpretation_convolution() -> Result<(), Box<dyn std::error::Error>> {
    let interpreter = GoalInterpreter::new();
    
    let goals = vec![
        "Apply 3x3 convolution filter to image",
        "Perform 1D convolution on signal",
        "Compute 2D correlation between images",
    ];
    
    for goal in goals {
        let spec = interpreter.interpret_goal(goal)?;
        assert_eq!(spec.algorithm_type, AlgorithmType::Convolution);
        assert!(spec.shared_memory_bytes > 0); // Convolution typically uses shared memory
    }
}

#[test]
    fn test_goal_interpretation_sorting() -> Result<(), Box<dyn std::error::Error>> {
    let interpreter = GoalInterpreter::new();
    
    let goals = vec![
        "Sort array of 100000 integers",
        "Perform radix sort on unsigned integers",
        "Merge sort two sorted arrays",
    ];
    
    for goal in goals {
        let spec = interpreter.interpret_goal(goal)?;
        assert_eq!(spec.algorithm_type, AlgorithmType::Sorting);
        assert!(spec.grid_dimensions.0 > 1); // Sorting typically needs multiple blocks
    }
}

#[test]
    fn test_optimization_level_interpretation() -> Result<(), Box<dyn std::error::Error>> {
    let interpreter = GoalInterpreter::new();
    
    let test_cases = vec![
        ("Quick prototype matrix multiplication", OptimizationLevel::Fast),
        ("High-performance matrix multiplication for production", OptimizationLevel::Aggressive),
        ("Balance speed and memory for vector addition", OptimizationLevel::Balanced),
        ("Memory-efficient sorting algorithm", OptimizationLevel::Size),
    ];
    
    for (goal, expected_opt) in test_cases {
        let spec = interpreter.interpret_goal(goal).unwrap();
        assert_eq!(spec.optimization_level, expected_opt);
    }
}

#[test]
    fn test_compute_capability_detection() -> Result<(), Box<dyn std::error::Error>> {
    let interpreter = GoalInterpreter::new();
    
    let goals_with_features = vec![
        ("Use tensor cores for matrix multiplication", (8, 0)), // Tensor cores need 8.0+
        ("Perform half-precision floating point operations", (5, 3)), // FP16 needs 5.3+
        ("Use cooperative groups for reduction", (6, 0)), // Cooperative groups need 6.0+
        ("Basic vector addition", (3, 5)), // Basic operations work on older hardware
    ];
    
    for (goal, min_capability) in goals_with_features {
        let spec = interpreter.interpret_goal(goal).unwrap();
        assert!(spec.compute_capability.0 >= min_capability.0);
        if spec.compute_capability.0 == min_capability.0 {
            assert!(spec.compute_capability.1 >= min_capability.1);
        }
    }
}

#[test]
    fn test_memory_pattern_detection() -> Result<(), Box<dyn std::error::Error>> {
    let interpreter = GoalInterpreter::new();
    
    let test_cases = vec![
        ("Stream through large array sequentially", MemoryPattern::Streaming),
        ("Random access to sparse matrix elements", MemoryPattern::Random),
        ("Transpose matrix with coalesced access", MemoryPattern::Coalesced),
        ("Stencil computation on 2D grid", MemoryPattern::Stencil),
    ];
    
    for (goal, expected_pattern) in test_cases {
        let spec = interpreter.interpret_goal(goal).unwrap();
        assert_eq!(spec.memory_pattern, expected_pattern);
    }
}

#[test]
    fn test_grid_block_dimension_calculation() -> Result<(), Box<dyn std::error::Error>> {
    let interpreter = GoalInterpreter::new();
    
    let goals = vec![
        "Process array of 1000 elements",
        "Process array of 100000 elements", 
        "Process array of 10000000 elements",
    ];
    
    for goal in goals {
        let spec = interpreter.interpret_goal(goal)?;
        
        // Verify dimensions are reasonable
        assert!(spec.grid_dimensions.0 > 0);
        assert!(spec.block_dimensions.0 > 0);
        assert!(spec.block_dimensions.0 <= 1024); // CUDA limit
        
        // Total threads should be reasonable for problem size
        let total_threads = spec.grid_dimensions.0 * spec.block_dimensions.0;
        assert!(total_threads > 0);
    }
}

#[test]
    fn test_register_usage_estimation() -> Result<(), Box<dyn std::error::Error>> {
    let interpreter = GoalInterpreter::new();
    
    let goals = vec![
        ("Simple element-wise addition", Some(8)),
        ("Complex mathematical function with many temporaries", Some(32)),
        ("Matrix multiplication with blocking", Some(64)),
    ];
    
    for (goal, expected_range) in goals {
        let spec = interpreter.interpret_goal(goal)?;
        
        if let Some(registers) = spec.registers_per_thread {
            assert!(registers > 0);
            assert!(registers <= 255); // CUDA limit
            
            if let Some(expected) = expected_range {
                // Should be in reasonable range of expected
                assert!(registers <= expected * 2);
            }
        }
    }
}

#[test]
    fn test_shared_memory_calculation() -> Result<(), Box<dyn std::error::Error>> {
    let interpreter = GoalInterpreter::new();
    
    let goals = vec![
        ("Matrix multiplication with shared memory blocking", true),
        ("Convolution with shared memory tiles", true),
        ("Simple element-wise operations", false),
        ("Reduction with shared memory optimization", true),
    ];
    
    for (goal, should_use_shared) in goals {
        let spec = interpreter.interpret_goal(goal).unwrap();
        
        if should_use_shared {
            assert!(spec.shared_memory_bytes > 0);
            assert!(spec.shared_memory_bytes <= 48 * 1024); // Typical limit
        } else {
            assert_eq!(spec.shared_memory_bytes, 0);
        }
    }
}

#[test]
    fn test_input_output_type_inference() -> Result<(), Box<dyn std::error::Error>> {
    let interpreter = GoalInterpreter::new();
    
    let test_cases = vec![
        ("Add two float arrays", vec!["float*", "float*"], vec!["float*"]),
        ("Convert int array to float", vec!["int*"], vec!["float*"]),
        ("Matrix multiply float matrices", vec!["float*", "float*"], vec!["float*"]),
        ("Bitwise AND on integer arrays", vec!["int*", "int*"], vec!["int*"]),
    ];
    
    for (goal, expected_inputs, expected_outputs) in test_cases {
        let spec = interpreter.interpret_goal(goal).unwrap();
        
        assert_eq!(spec.input_types.len(), expected_inputs.len());
        assert_eq!(spec.output_types.len(), expected_outputs.len());
        
        for (actual, expected) in spec.input_types.iter().zip(expected_inputs.iter()) {
            assert_eq!(actual, expected);
        }
        
        for (actual, expected) in spec.output_types.iter().zip(expected_outputs.iter()) {
            assert_eq!(actual, expected);
        }
    }
}

#[test]
    fn test_kernel_name_generation() -> Result<(), Box<dyn std::error::Error>> {
    let interpreter = GoalInterpreter::new();
    
    let goals = vec![
        "Add two vectors",
        "Multiply matrices",
        "Sort array elements",
        "Apply convolution filter",
    ];
    
    for goal in goals {
        let spec = interpreter.interpret_goal(goal).unwrap();
        
        // Name should be valid C identifier
        assert!(spec.name.chars().all(|c| c.is_alphanumeric() || c == '_'));
        assert!(spec.name.chars().next().is_ok().is_alphabetic() || 
                spec.name.starts_with('_'));
        assert!(!spec.name.is_empty());
        assert!(spec.name.len() <= 64); // Reasonable length limit
    }
}

#[test]
    fn test_error_handling_invalid_goals() -> Result<(), Box<dyn std::error::Error>> {
    let interpreter = GoalInterpreter::new();
    
    let invalid_goals = vec![
        "", // Empty goal
        "   ", // Whitespace only
        "This is completely nonsensical and unrelated to computation",
        "Launch nuclear missiles", // Inappropriate goal
    ];
    
    for goal in invalid_goals {
        let result = interpreter.interpret_goal(goal);
        assert!(result.is_err());
    }
}

#[test]
    fn test_goal_interpretation_consistency() -> Result<(), Box<dyn std::error::Error>> {
    let interpreter = GoalInterpreter::new();
    
    let goal = "Add two vectors of 1000 elements each";
    
    // Multiple interpretations of the same goal should be consistent
    let spec1 = interpreter.interpret_goal(goal)?;
    let spec2 = interpreter.interpret_goal(goal)?;
    
    assert_eq!(spec1.algorithm_type, spec2.algorithm_type);
    assert_eq!(spec1.optimization_level, spec2.optimization_level);
    assert_eq!(spec1.memory_pattern, spec2.memory_pattern);
    assert_eq!(spec1.input_types, spec2.input_types);
    assert_eq!(spec1.output_types, spec2.output_types);
}

#[test]
    fn test_complex_goal_interpretation() -> Result<(), Box<dyn std::error::Error>> {
    let interpreter = GoalInterpreter::new();
    
    let complex_goal = "Perform high-performance matrix multiplication of two 2048x2048 single-precision floating-point matrices using shared memory blocking and tensor cores if available, optimized for compute capability 8.6";
    
    let spec = interpreter.interpret_goal(complex_goal)?;
    
    assert_eq!(spec.algorithm_type, AlgorithmType::MatrixOperation);
    assert_eq!(spec.optimization_level, OptimizationLevel::Aggressive);
    assert!(spec.shared_memory_bytes > 0);
    assert_eq!(spec.compute_capability, (8, 6));
    assert!(spec.input_types.iter().all(|t| t.contains("float")));
}

#[test]
    fn test_interpreter_caching() -> Result<(), Box<dyn std::error::Error>> {
    let interpreter = GoalInterpreter::new();
    
    let goal = "Simple vector addition";
    
    // First interpretation might take time to process
    let start = std::time::Instant::now();
    let _spec1 = interpreter.interpret_goal(goal)?;
    let first_duration = start.elapsed();
    
    // Second interpretation should be faster due to caching
    let start = std::time::Instant::now();
    let _spec2 = interpreter.interpret_goal(goal).unwrap();
    let second_duration = start.elapsed();
    
    // Second should be faster (or at least not significantly slower)
    assert!(second_duration <= first_duration * 2);
}

#[test]
    fn test_interpreter_context_awareness() -> Result<(), Box<dyn std::error::Error>> {
    let mut interpreter = GoalInterpreter::new();
    
    // Set context for GPU with specific capabilities
    interpreter.set_target_device_info(DeviceInfo {
        compute_capability: (7, 5),
        max_threads_per_block: 1024,
        shared_memory_per_block: 48 * 1024,
        max_registers_per_block: 65536,
        warp_size: 32,
    });
    
    let goal = "Optimize for current GPU device";
    let spec = interpreter.interpret_goal(goal).unwrap();
    
    assert_eq!(spec.compute_capability, (7, 5));
    assert!(spec.block_dimensions.0 <= 1024);
}

#[test]
    fn test_batch_goal_interpretation() -> Result<(), Box<dyn std::error::Error>> {
    let interpreter = GoalInterpreter::new();
    
    let goals = vec![
        "Add vectors A and B",
        "Subtract vector B from A", 
        "Multiply vectors element-wise",
        "Divide vector A by B",
    ];
    
    let specs = interpreter.interpret_goals_batch(&goals)?;
    
    assert_eq!(specs.len(), goals.len());
    for spec in specs {
        assert_eq!(spec.algorithm_type, AlgorithmType::ElementWise);
        assert_eq!(spec.input_types.len(), 2);
        assert_eq!(spec.output_types.len(), 1);
    }
}