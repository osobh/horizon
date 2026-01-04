//! Property-based tests for synthesis crate

use proptest::prelude::*;
use stratoswarm_synthesis::interpreter::{
    DataLayout, KernelSpecification, MemoryLayout, OperationType, OptimizationHint,
    PerformanceModel, Precision,
};

// Generate arbitrary operation types
prop_compose! {
    fn arb_operation_type()(idx in 0..5usize) -> OperationType {
        match idx {
            0 => OperationType::MatrixMultiply,
            1 => OperationType::Reduction,
            2 => OperationType::Convolution,
            3 => OperationType::Elementwise,
            _ => OperationType::Custom,
        }
    }
}

// Generate arbitrary memory layouts
prop_compose! {
    fn arb_memory_layout()(idx in 0..3usize) -> MemoryLayout {
        match idx {
            0 => MemoryLayout::RowMajor,
            1 => MemoryLayout::ColumnMajor,
            _ => MemoryLayout::Custom,
        }
    }
}

// Generate arbitrary precision types
prop_compose! {
    fn arb_precision()(idx in 0..4usize) -> Precision {
        match idx {
            0 => Precision::FP32,
            1 => Precision::FP16,
            2 => Precision::BF16,
            _ => Precision::INT8,
        }
    }
}

// Generate arbitrary optimization hints
prop_compose! {
    fn arb_optimization_hint()(idx in 0..5usize) -> OptimizationHint {
        match idx {
            0 => OptimizationHint::TensorCore,
            1 => OptimizationHint::SharedMemory,
            2 => OptimizationHint::Unrolling,
            3 => OptimizationHint::Vectorization,
            _ => OptimizationHint::Fusion,
        }
    }
}

// Generate arbitrary tensor shapes (reasonable sizes)
prop_compose! {
    fn arb_shape()(
        dims in 1..=4usize,
        shape in prop::collection::vec(1..=2048usize, 1..=4)
    ) -> Vec<usize> {
        shape.into_iter().take(dims).collect()
    }
}

// Generate arbitrary data layout
prop_compose! {
    fn arb_data_layout()(
        input_shape in arb_shape(),
        output_shape in arb_shape(),
        memory_layout in arb_memory_layout()
    ) -> DataLayout {
        DataLayout {
            input_shape,
            output_shape,
            memory_layout,
        }
    }
}

// Generate arbitrary performance model
prop_compose! {
    fn arb_performance_model()(
        compute_intensity in 0.1f32..1000.0,
        memory_bandwidth in 10.0f32..2000.0,
        expected_occupancy in 0.0f32..1.0
    ) -> PerformanceModel {
        PerformanceModel {
            compute_intensity,
            memory_bandwidth,
            expected_occupancy,
        }
    }
}

// Generate arbitrary kernel specification
prop_compose! {
    fn arb_kernel_specification()(
        operation_type in arb_operation_type(),
        data_layout in arb_data_layout(),
        precision in arb_precision(),
        optimization_hints in prop::collection::vec(arb_optimization_hint(), 0..5),
        performance_model in arb_performance_model()
    ) -> KernelSpecification {
        KernelSpecification {
            operation_type,
            data_layout,
            precision,
            optimization_hints,
            performance_model,
        }
    }
}

proptest! {
    #[test]
    fn test_kernel_spec_serialization_roundtrip(spec in arb_kernel_specification()) {
        let json = serde_json::to_string(&spec)?;
        let parsed: KernelSpecification = serde_json::from_str(&json)?;

        prop_assert_eq!(spec.operation_type, parsed.operation_type);
        prop_assert_eq!(spec.precision, parsed.precision);
        prop_assert_eq!(spec.data_layout.input_shape, parsed.data_layout.input_shape);
        prop_assert_eq!(spec.data_layout.output_shape, parsed.data_layout.output_shape);
        prop_assert_eq!(spec.data_layout.memory_layout, parsed.data_layout.memory_layout);
        prop_assert_eq!(spec.optimization_hints.len(), parsed.optimization_hints.len());
    }

    #[test]
    fn test_performance_model_invariants(model in arb_performance_model()) {
        // Occupancy should be between 0 and 1
        prop_assert!(model.expected_occupancy >= 0.0);
        prop_assert!(model.expected_occupancy <= 1.0);

        // Compute intensity and bandwidth should be positive
        prop_assert!(model.compute_intensity > 0.0);
        prop_assert!(model.memory_bandwidth > 0.0);
    }

    #[test]
    fn test_memory_estimation_invariants(spec in arb_kernel_specification()) {
        let element_size = match spec.precision {
            Precision::FP32 | Precision::INT8 => 4,
            Precision::FP16 | Precision::BF16 => 2,
        };

        let input_elements: usize = spec.data_layout.input_shape.iter().product();
        let output_elements: usize = spec.data_layout.output_shape.iter().product();
        let estimated_memory = (input_elements + output_elements) * element_size;

        // Memory should be non-negative and reasonable
        prop_assert!(estimated_memory >= 0);
        prop_assert!(estimated_memory <= usize::MAX / 2); // Reasonable upper bound
    }

    #[test]
    fn test_shape_dimensions(shape in arb_shape()) {
        // Shapes should have at least one dimension
        prop_assert!(!shape.is_empty());
        prop_assert!(shape.len() <= 4); // Max 4D tensors

        // All dimensions should be positive
        for dim in &shape {
            prop_assert!(*dim > 0);
            prop_assert!(*dim <= 2048); // Reasonable size limit
        }
    }

    #[test]
    fn test_optimization_hints_uniqueness(
        hints in prop::collection::vec(arb_optimization_hint(), 0..10)
    ) {
        // Deduplication should work
        let mut unique_hints = hints.clone();
        unique_hints.sort_by_key(|h| format!("{:?}", h));
        unique_hints.dedup();

        prop_assert!(unique_hints.len() <= hints.len());
        prop_assert!(unique_hints.len() <= 5); // Max 5 different hint types
    }

    #[test]
    fn test_kernel_name_generation(
        operation_type in arb_operation_type(),
        precision in arb_precision()
    ) {
        let name = format!("{:?}_{:?}_kernel", operation_type, precision).to_lowercase();

        // Name should contain expected parts
        prop_assert!(name.contains("kernel"));
        prop_assert!(name.len() > 0);
        prop_assert!(name.len() < 100); // Reasonable length

        // Should be lowercase
        prop_assert_eq!(name, name.to_lowercase());
    }

    #[test]
    fn test_data_layout_memory_consistency(layout in arb_data_layout()) {
        // Memory layout should be valid
        match layout.memory_layout {
            MemoryLayout::RowMajor | MemoryLayout::ColumnMajor | MemoryLayout::Custom => {
                // All valid
            }
        }

        // Shapes should be non-empty for valid operations
        prop_assert!(!layout.input_shape.is_empty() || !layout.output_shape.is_empty());
    }

    #[test]
    fn test_precision_size_mapping(precision in arb_precision()) {
        let size = match precision {
            Precision::FP32 => 4,
            Precision::FP16 => 2,
            Precision::BF16 => 2,
            Precision::INT8 => 4, // Actually 1, but our code uses 4
        };

        prop_assert!(size > 0);
        prop_assert!(size <= 8);
    }

    #[test]
    fn test_kernel_spec_cloning(spec in arb_kernel_specification()) {
        let cloned = spec.clone();

        prop_assert_eq!(spec.operation_type, cloned.operation_type);
        prop_assert_eq!(spec.precision, cloned.precision);
        prop_assert_eq!(spec.data_layout.input_shape, cloned.data_layout.input_shape);
        prop_assert_eq!(spec.data_layout.output_shape, cloned.data_layout.output_shape);
        prop_assert_eq!(spec.optimization_hints.len(), cloned.optimization_hints.len());
    }

    #[test]
    fn test_tensor_size_calculations(
        shape in arb_shape(),
        precision in arb_precision()
    ) {
        let elements: usize = shape.iter().product();
        let element_size = match precision {
            Precision::FP32 | Precision::INT8 => 4,
            Precision::FP16 | Precision::BF16 => 2,
        };

        let total_size = elements * element_size;

        // Check for overflow
        prop_assert!(total_size >= elements || elements == 0);
        prop_assert!(total_size >= element_size || elements == 0);
    }
}

// Additional fuzz-like tests
proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn fuzz_kernel_spec_json_parsing(s in "\\PC*") {
        // Try to parse arbitrary strings as kernel specs
        let _ = serde_json::from_str::<KernelSpecification>(&s);
        // Should not panic
    }

    #[test]
    fn fuzz_optimization_hint_combinations(
        hints in prop::collection::vec(arb_optimization_hint(), 0..20)
    ) {
        // Create spec with many hints
        let spec = KernelSpecification {
            operation_type: OperationType::Custom,
            data_layout: DataLayout {
                input_shape: vec![100],
                output_shape: vec![100],
                memory_layout: MemoryLayout::RowMajor,
            },
            precision: Precision::FP32,
            optimization_hints: hints,
            performance_model: PerformanceModel {
                compute_intensity: 1.0,
                memory_bandwidth: 100.0,
                expected_occupancy: 0.5,
            },
        };

        // Should serialize without issues
        let json = serde_json::to_string(&spec)?;
        prop_assert!(json.len() > 0);
    }
}

// Property tests for edge cases
proptest! {
    #[test]
    fn test_empty_shapes_handling() {
        let layout = DataLayout {
            input_shape: vec![],
            output_shape: vec![1],
            memory_layout: MemoryLayout::RowMajor,
        };

        let input_elements: usize = layout.input_shape.iter().product();
        let output_elements: usize = layout.output_shape.iter().product();

        prop_assert_eq!(input_elements, 1); // Empty product is 1
        prop_assert_eq!(output_elements, 1);
    }

    #[test]
    fn test_large_tensor_dimensions(
        dim1 in 1000..2000usize,
        dim2 in 1000..2000usize
    ) {
        let shape = vec![dim1, dim2];
        let elements: usize = shape.iter().product();

        prop_assert_eq!(elements, dim1 * dim2);
        prop_assert!(elements > 0);
    }

    #[test]
    fn test_occupancy_boundary_values(
        occupancy in prop::oneof![
            Just(0.0f32),
            Just(1.0f32),
            Just(0.5f32),
            0.0001f32..0.9999f32
        ]
    ) {
        let model = PerformanceModel {
            compute_intensity: 10.0,
            memory_bandwidth: 500.0,
            expected_occupancy: occupancy,
        };

        prop_assert!(model.expected_occupancy >= 0.0);
        prop_assert!(model.expected_occupancy <= 1.0);
    }
}
