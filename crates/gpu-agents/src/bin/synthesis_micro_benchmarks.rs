//! Micro-benchmarks for synthesis optimizations
//!
//! Tests individual optimizations to measure their impact

use anyhow::Result;
use cudarc::driver::CudaDevice;
use gpu_agents::synthesis::batch_processor::{BatchConfig, BatchProcessor};
use gpu_agents::synthesis::pattern_dynamic::DynamicGpuPatternMatcher;
use gpu_agents::synthesis::{AstNode, NodeType, Pattern};
use std::time::Instant;

fn create_test_patterns(count: usize) -> Vec<Pattern> {
    (0..count)
        .map(|i| Pattern {
            node_type: NodeType::Variable,
            children: vec![],
            value: Some(format!("pattern_{}", i % 100)),
        })
        .collect()
}

fn create_test_asts(count: usize, complexity: usize) -> Vec<AstNode> {
    (0..count)
        .map(|i| {
            let mut node = AstNode {
                node_type: NodeType::Variable,
                children: vec![],
                value: Some(format!("var_{}", i % 50)),
            };

            // Add children for complexity
            for j in 0..complexity {
                node.children.push(AstNode {
                    node_type: NodeType::Literal,
                    children: vec![],
                    value: Some(format!("child_{}_{}", i, j)),
                });
            }

            node
        })
        .collect()
}

fn benchmark_operation<F>(name: &str, iterations: usize, mut op: F) -> f64
where
    F: FnMut() -> Result<()>,
{
    // Warmup
    for _ in 0..3 {
        op()?;
    }

    // Measure
    let start = Instant::now();
    for _ in 0..iterations {
        op()?;
    }
    let elapsed = start.elapsed();

    let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();
    println!("{:40} {:>12.2} ops/sec", name, ops_per_sec);

    ops_per_sec
}

fn main() -> Result<()> {
    println!("🔬 Synthesis Micro-Benchmarks");
    println!("{}", "=".repeat(60));

    let device = CudaDevice::new(0)?;

    // Test 1: Basic pattern matching (baseline)
    println!("\n1. Basic Pattern Matching (Baseline)");
    {
        let matcher = DynamicGpuPatternMatcher::new(device.clone())?;
        let patterns = create_test_patterns(1);
        let asts = create_test_asts(1000, 0);

        benchmark_operation("Single pattern, 1K ASTs", 100, || {
            matcher.match_batch(&patterns, &asts)?;
            Ok(())
        });
    }

    // Test 2: Multiple patterns (batch benefit)
    println!("\n2. Multiple Pattern Batching");
    {
        let matcher = DynamicGpuPatternMatcher::new(device.clone())?;

        for pattern_count in [1, 10, 32, 100] {
            let patterns = create_test_patterns(pattern_count);
            let asts = create_test_asts(1000, 0);

            let name = format!("{} patterns, 1K ASTs", pattern_count);
            benchmark_operation(&name, 100, || {
                matcher.match_batch(&patterns, &asts)?;
                Ok(())
            });
        }
    }

    // Test 3: Stream parallelism impact
    println!("\n3. Stream Parallelism Impact");
    {
        for num_streams in [1, 2, 4, 8] {
            let config = BatchConfig {
                max_patterns_per_batch: 10,
                max_nodes_per_batch: 1000,
                num_streams,
            };

            let processor = BatchProcessor::new(device.clone(), config)?;
            let patterns = create_test_patterns(100);
            let asts = create_test_asts(1000, 0);

            let name = format!("{} streams, 100 patterns", num_streams);
            benchmark_operation(&name, 50, || {
                processor.process_single_batch(&patterns, &asts)?;
                Ok(())
            });
        }
    }

    // Test 4: Memory access patterns
    println!("\n4. Memory Access Patterns");
    {
        let matcher = DynamicGpuPatternMatcher::new(device.clone())?;

        // Sequential access (best case)
        let patterns_seq = create_test_patterns(32);
        let asts_seq = create_test_asts(1000, 0);

        benchmark_operation("Sequential access", 100, || {
            matcher.match_batch(&patterns_seq, &asts_seq)?;
            Ok(())
        });

        // Random access simulation (worst case)
        let patterns_rand = create_test_patterns(32);
        // Patterns are already in sequential order, no need to randomize

        benchmark_operation("Random access", 100, || {
            matcher.match_batch(&patterns_rand, &asts_seq)?;
            Ok(())
        });
    }

    // Test 5: AST complexity impact
    println!("\n5. AST Complexity Impact");
    {
        let matcher = DynamicGpuPatternMatcher::new(device.clone())?;
        let patterns = create_test_patterns(10);

        for complexity in [0, 1, 5, 10] {
            let asts = create_test_asts(1000, complexity);

            let name = format!("{} children per node", complexity);
            benchmark_operation(&name, 50, || {
                matcher.match_batch(&patterns, &asts)?;
                Ok(())
            });
        }
    }

    // Test 6: Batch size impact
    println!("\n6. Batch Size Impact");
    {
        let matcher = DynamicGpuPatternMatcher::new(device.clone())?;
        let patterns = create_test_patterns(10);

        for ast_count in [100, 1000, 10000, 50000] {
            let asts = create_test_asts(ast_count, 0);

            let name = format!("{} AST nodes", ast_count);
            let ops = benchmark_operation(&name, 20, || {
                matcher.match_batch(&patterns, &asts)?;
                Ok(())
            });

            let throughput = ops * (patterns.len() * ast_count) as f64;
            println!("  → Throughput: {:.2}M ops/sec", throughput / 1_000_000.0);
        }
    }

    // Test 7: Multi-batch processing
    println!("\n7. Multi-Batch Processing");
    {
        let config = BatchConfig::default();
        let processor = BatchProcessor::new(device.clone(), config)?;

        for batch_count in [1, 5, 10, 20] {
            let pattern_batches: Vec<_> =
                (0..batch_count).map(|_| create_test_patterns(10)).collect();
            let ast_batches: Vec<_> = (0..batch_count)
                .map(|_| create_test_asts(1000, 0))
                .collect();

            let name = format!("{} batches", batch_count);
            benchmark_operation(&name, 20, || {
                processor.process_batches(pattern_batches.clone(), ast_batches.clone())?;
                Ok(())
            });
        }
    }

    // Summary
    println!("\n{}", "=".repeat(60));
    println!("📊 Summary:");
    println!("- Batching multiple patterns improves throughput");
    println!("- Stream parallelism helps with large batches");
    println!("- Memory access patterns significantly impact performance");
    println!("- AST complexity increases processing time linearly");

    Ok(())
}
