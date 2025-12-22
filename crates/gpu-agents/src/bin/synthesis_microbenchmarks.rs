//! Synthesis Micro-benchmarks
//!
//! Comprehensive benchmarks to measure and improve synthesis performance
//! Target: 2.6B operations/second

use anyhow::Result;
use cudarc::driver::{CudaDevice, DevicePtr};
use gpu_agents::synthesis::{
    ast::GpuAstTransformer, pattern::GpuPatternMatcher, pattern_dynamic::DynamicGpuPatternMatcher,
    pattern_simple::SimpleGpuPatternMatcher, template::GpuTemplateExpander, AstNode, NodeType,
    Pattern, Template, Token, TransformRule,
};
use std::sync::Arc;
use std::time::{Duration, Instant};

const TARGET_OPS_PER_SEC: f64 = 2.6e9; // 2.6 billion ops/sec

#[derive(Default)]
struct BenchmarkResults {
    pattern_matching_basic: f64,
    pattern_matching_simple: f64,
    pattern_matching_dynamic: f64,
    template_expansion: f64,
    ast_transformation: f64,
    kernel_launch_overhead: f64,
    memory_transfer_bandwidth: f64,
}

fn main() -> Result<()> {
    env_logger::init();

    println!("🚀 Synthesis Micro-benchmarks");
    println!("============================");
    println!("Target: {:.1} billion ops/sec\n", TARGET_OPS_PER_SEC / 1e9);

    let device = CudaDevice::new(0)?;

    let mut results = BenchmarkResults::default();

    // Phase 1: Kernel Launch Overhead
    println!("📊 Phase 1: Measuring Kernel Launch Overhead");
    println!("-------------------------------------------");
    results.kernel_launch_overhead = measure_kernel_launch_overhead(&device)?;

    // Phase 2: Memory Transfer Bandwidth
    println!("\n📊 Phase 2: Memory Transfer Bandwidth");
    println!("------------------------------------");
    results.memory_transfer_bandwidth = measure_memory_bandwidth(&device)?;

    // Phase 3: Pattern Matching Benchmarks
    println!("\n📊 Phase 3: Pattern Matching Performance");
    println!("---------------------------------------");

    // Basic pattern matcher
    println!("\n  Testing basic pattern matcher...");
    results.pattern_matching_basic = benchmark_basic_pattern_matcher(device.clone())?;

    // Simple pattern matcher
    println!("\n  Testing simple pattern matcher...");
    results.pattern_matching_simple = benchmark_simple_pattern_matcher(device.clone())?;

    // Dynamic pattern matcher
    println!("\n  Testing dynamic pattern matcher...");
    results.pattern_matching_dynamic = benchmark_dynamic_pattern_matcher(device.clone())?;

    // Phase 4: Template Expansion
    println!("\n📊 Phase 4: Template Expansion Performance");
    println!("----------------------------------------");
    results.template_expansion = benchmark_template_expansion(device.clone())?;

    // Phase 5: AST Transformation
    println!("\n📊 Phase 5: AST Transformation Performance");
    println!("-----------------------------------------");
    results.ast_transformation = benchmark_ast_transformation(device.clone())?;

    // Phase 6: Results Summary
    print_results_summary(&results);

    // Phase 7: Optimization Recommendations
    print_optimization_recommendations(&results);

    Ok(())
}

/// Measure kernel launch overhead
fn measure_kernel_launch_overhead(device: &Arc<CudaDevice>) -> Result<f64> {
    const NUM_LAUNCHES: u32 = 10000;

    // Allocate minimal buffers
    let buffer = device.alloc_zeros::<u8>(64)?;

    let start = Instant::now();

    for _ in 0..NUM_LAUNCHES {
        unsafe {
            gpu_agents::synthesis::launch_match_patterns_fast(
                *buffer.device_ptr() as *const u8,
                *buffer.device_ptr() as *const u8,
                *buffer.device_ptr() as *mut u32,
                1,
                1,
            );
        }
    }

    device.synchronize()?;

    let elapsed = start.elapsed();
    let overhead_us = elapsed.as_micros() as f64 / NUM_LAUNCHES as f64;

    println!("  Kernel launch overhead: {:.2} μs", overhead_us);
    println!("  Max kernel launches/sec: {:.0}", 1e6 / overhead_us);

    Ok(overhead_us)
}

/// Measure memory transfer bandwidth
fn measure_memory_bandwidth(device: &Arc<CudaDevice>) -> Result<f64> {
    const BUFFER_SIZE: usize = 256 * 1024 * 1024; // 256 MB
    const NUM_TRANSFERS: u32 = 100;

    let host_buffer = vec![0u8; BUFFER_SIZE];
    let device_buffer = unsafe { device.alloc::<u8>(BUFFER_SIZE)? };

    // Host to Device
    let start = Instant::now();
    for _ in 0..NUM_TRANSFERS {
        device.htod_copy_into(host_buffer.clone(), &mut device_buffer.clone())?;
    }
    device.synchronize()?;
    let htod_time = start.elapsed();

    let htod_bandwidth =
        (BUFFER_SIZE as f64 * NUM_TRANSFERS as f64) / htod_time.as_secs_f64() / 1e9;

    // Device to Host
    let mut host_result = vec![0u8; BUFFER_SIZE];
    let start = Instant::now();
    for _ in 0..NUM_TRANSFERS {
        device.dtoh_sync_copy_into(&device_buffer, &mut host_result)?;
    }
    let dtoh_time = start.elapsed();

    let dtoh_bandwidth =
        (BUFFER_SIZE as f64 * NUM_TRANSFERS as f64) / dtoh_time.as_secs_f64() / 1e9;

    println!("  Host→Device bandwidth: {:.1} GB/s", htod_bandwidth);
    println!("  Device→Host bandwidth: {:.1} GB/s", dtoh_bandwidth);

    Ok((htod_bandwidth + dtoh_bandwidth) / 2.0)
}

/// Benchmark basic pattern matcher
fn benchmark_basic_pattern_matcher(device: Arc<CudaDevice>) -> Result<f64> {
    let matcher = GpuPatternMatcher::new(device, 10000)?;

    // Create test patterns and ASTs
    let patterns = create_test_patterns(64);
    let asts = create_test_asts(10000);

    let start = Instant::now();
    let mut operations = 0u64;

    while start.elapsed() < Duration::from_secs(5) {
        let pattern = &patterns[operations as usize % patterns.len()];
        let ast = &asts[operations as usize % asts.len()];

        let _matches = matcher.match_pattern(pattern, ast)?;
        operations += 1;
    }

    let ops_per_sec = operations as f64 / start.elapsed().as_secs_f64();
    println!(
        "  Basic matcher: {:.2} million ops/sec ({:.1}% of target)",
        ops_per_sec / 1e6,
        (ops_per_sec / TARGET_OPS_PER_SEC) * 100.0
    );

    Ok(ops_per_sec)
}

/// Benchmark simple pattern matcher
fn benchmark_simple_pattern_matcher(device: Arc<CudaDevice>) -> Result<f64> {
    let matcher = SimpleGpuPatternMatcher::new(device)?;

    let patterns = create_test_patterns(32);
    let asts = create_test_asts(5000);

    let start = Instant::now();
    let mut operations = 0u64;

    while start.elapsed() < Duration::from_secs(5) {
        // Process each pattern against each AST
        for pattern in &patterns {
            for ast in &asts {
                let _matches = matcher.match_pattern(pattern, ast)?;
                operations += 1;
            }
        }
    }

    let ops_per_sec = operations as f64 / start.elapsed().as_secs_f64();
    println!(
        "  Simple matcher: {:.2} million ops/sec ({:.1}% of target)",
        ops_per_sec / 1e6,
        (ops_per_sec / TARGET_OPS_PER_SEC) * 100.0
    );

    Ok(ops_per_sec)
}

/// Benchmark dynamic pattern matcher
fn benchmark_dynamic_pattern_matcher(device: Arc<CudaDevice>) -> Result<f64> {
    let matcher = DynamicGpuPatternMatcher::new(device)?;

    let patterns = create_test_patterns(32);
    let asts = create_test_asts(5000);

    let start = Instant::now();
    let mut operations = 0u64;

    while start.elapsed() < Duration::from_secs(5) {
        // Process all patterns against all ASTs in batch
        let _matches = matcher.match_batch(&patterns, &asts)?;
        operations += (patterns.len() * asts.len()) as u64;
    }

    let ops_per_sec = operations as f64 / start.elapsed().as_secs_f64();
    println!(
        "  Dynamic matcher: {:.2} million ops/sec ({:.1}% of target)",
        ops_per_sec / 1e6,
        (ops_per_sec / TARGET_OPS_PER_SEC) * 100.0
    );

    Ok(ops_per_sec)
}

/// Benchmark template expansion
fn benchmark_template_expansion(device: Arc<CudaDevice>) -> Result<f64> {
    let expander = GpuTemplateExpander::new(device, 10000)?;

    let templates = create_test_templates(100);
    let bindings = std::collections::HashMap::from([
        ("var1".to_string(), "value1".to_string()),
        ("var2".to_string(), "value2".to_string()),
        ("var3".to_string(), "value3".to_string()),
    ]);

    let start = Instant::now();
    let mut operations = 0u64;

    while start.elapsed() < Duration::from_secs(5) {
        let template = &templates[operations as usize % templates.len()];
        let _result = expander.expand_template(template, &bindings)?;
        operations += 1;
    }

    let ops_per_sec = operations as f64 / start.elapsed().as_secs_f64();
    println!(
        "  Template expansion: {:.2} million ops/sec ({:.1}% of target)",
        ops_per_sec / 1e6,
        (ops_per_sec / TARGET_OPS_PER_SEC) * 100.0
    );

    Ok(ops_per_sec)
}

/// Benchmark AST transformation
fn benchmark_ast_transformation(device: Arc<CudaDevice>) -> Result<f64> {
    let transformer = GpuAstTransformer::new(device, 10000)?;

    let asts = create_test_asts(1000);
    let rules = create_test_transform_rules(10);

    let start = Instant::now();
    let mut operations = 0u64;

    while start.elapsed() < Duration::from_secs(5) {
        let ast = &asts[operations as usize % asts.len()];
        let rule = &rules[operations as usize % rules.len()];
        let _result = transformer.transform_ast(ast, rule)?;
        operations += count_ast_nodes(ast) as u64;
    }

    let ops_per_sec = operations as f64 / start.elapsed().as_secs_f64();
    println!(
        "  AST transformation: {:.2} million ops/sec ({:.1}% of target)",
        ops_per_sec / 1e6,
        (ops_per_sec / TARGET_OPS_PER_SEC) * 100.0
    );

    Ok(ops_per_sec)
}

/// Create test patterns
fn create_test_patterns(count: usize) -> Vec<Pattern> {
    (0..count)
        .map(|i| Pattern {
            node_type: match i % 4 {
                0 => NodeType::Function,
                1 => NodeType::Variable,
                2 => NodeType::BinaryOp,
                _ => NodeType::Literal,
            },
            children: vec![],
            value: Some(format!("pattern_{}", i)),
        })
        .collect()
}

/// Create test AST nodes
fn create_test_asts(count: usize) -> Vec<AstNode> {
    (0..count)
        .map(|i| {
            let mut node = AstNode {
                node_type: match i % 5 {
                    0 => NodeType::Function,
                    1 => NodeType::Variable,
                    2 => NodeType::BinaryOp,
                    3 => NodeType::Literal,
                    _ => NodeType::Call,
                },
                children: vec![],
                value: Some(format!("node_{}", i)),
            };

            // Add some children for complexity
            if i % 3 == 0 {
                node.children.push(AstNode {
                    node_type: NodeType::Variable,
                    children: vec![],
                    value: Some(format!("child_{}", i)),
                });
            }

            node
        })
        .collect()
}

/// Create test templates
fn create_test_templates(count: usize) -> Vec<Template> {
    (0..count)
        .map(|i| Template {
            tokens: vec![
                Token::Literal(format!("template_{}_", i)),
                Token::Variable("var1".to_string()),
                Token::Literal(" = ".to_string()),
                Token::Variable("var2".to_string()),
                Token::Literal(format!(" // {}", i)),
            ],
        })
        .collect()
}

/// Create test transform rules
fn create_test_transform_rules(count: usize) -> Vec<TransformRule> {
    (0..count)
        .map(|i| TransformRule {
            pattern: Pattern {
                node_type: NodeType::BinaryOp,
                children: vec![],
                value: Some("+".to_string()),
            },
            replacement: AstNode {
                node_type: NodeType::Call,
                children: vec![],
                value: Some(format!("optimized_add_{}", i)),
            },
        })
        .collect()
}

/// Count nodes in AST
fn count_ast_nodes(ast: &AstNode) -> usize {
    1 + ast.children.iter().map(count_ast_nodes).sum::<usize>()
}

/// Print results summary
fn print_results_summary(results: &BenchmarkResults) {
    println!("\n📈 Results Summary");
    println!("=================");

    let best_ops = results.pattern_matching_basic.max(
        results
            .pattern_matching_simple
            .max(results.pattern_matching_dynamic),
    );

    println!("\n  Pattern Matching Performance:");
    println!(
        "  - Basic:     {:.2} million ops/sec",
        results.pattern_matching_basic / 1e6
    );
    println!(
        "  - Simple:    {:.2} million ops/sec",
        results.pattern_matching_simple / 1e6
    );
    println!(
        "  - Dynamic:   {:.2} million ops/sec",
        results.pattern_matching_dynamic / 1e6
    );

    println!("\n  Other Operations:");
    println!(
        "  - Template Expansion:    {:.2} million ops/sec",
        results.template_expansion / 1e6
    );
    println!(
        "  - AST Transformation:    {:.2} million ops/sec",
        results.ast_transformation / 1e6
    );

    println!("\n  System Characteristics:");
    println!(
        "  - Kernel Launch Overhead: {:.2} μs",
        results.kernel_launch_overhead
    );
    println!(
        "  - Memory Bandwidth:       {:.1} GB/s",
        results.memory_transfer_bandwidth
    );

    let percentage = (best_ops / TARGET_OPS_PER_SEC) * 100.0;
    println!(
        "\n  Best Performance: {:.2} million ops/sec ({:.1}% of target)",
        best_ops / 1e6,
        percentage
    );

    if percentage >= 100.0 {
        println!("\n🎉 SUCCESS: Target performance achieved!");
    } else {
        println!(
            "\n⚠️  Currently at {:.1}% of target performance",
            percentage
        );
    }
}

/// Print optimization recommendations
fn print_optimization_recommendations(results: &BenchmarkResults) {
    println!("\n💡 Optimization Recommendations");
    println!("==============================");

    let best_ops = results.pattern_matching_basic.max(
        results
            .pattern_matching_simple
            .max(results.pattern_matching_dynamic),
    );
    let gap = TARGET_OPS_PER_SEC - best_ops;

    if gap > 0.0 {
        println!("\n  To reach the 2.6B ops/sec target:");

        // Calculate required improvements
        let speedup_needed = TARGET_OPS_PER_SEC / best_ops;
        println!("  - Need {:.1}x speedup from current best", speedup_needed);

        // Specific recommendations based on bottlenecks
        if results.kernel_launch_overhead > 1.0 {
            println!("\n  1. Reduce Kernel Launch Overhead:");
            println!("     - Batch more operations per kernel");
            println!("     - Use persistent kernels");
            println!("     - Implement kernel fusion");
        }

        if results.memory_transfer_bandwidth < 500.0 {
            println!("\n  2. Optimize Memory Access:");
            println!("     - Use texture memory for patterns");
            println!("     - Implement memory prefetching");
            println!("     - Optimize data layout for coalescing");
        }

        println!("\n  3. Algorithm Optimizations:");
        println!("     - Use warp-level primitives");
        println!("     - Implement tile-based processing");
        println!("     - Add early termination logic");
        println!("     - Use tensor cores if available");

        println!("\n  4. System-Level Optimizations:");
        println!("     - Enable GPU boost clocks");
        println!("     - Use multiple GPU streams");
        println!("     - Overlap computation and transfers");
        println!("     - Profile with Nsight Compute");
    } else {
        println!("\n  ✅ Target performance achieved!");
        println!("  Consider:");
        println!("  - Maintaining performance under load");
        println!("  - Scaling to multiple GPUs");
        println!("  - Reducing power consumption");
    }
}
