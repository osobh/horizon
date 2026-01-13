//! Integration Performance Test
//!
//! Tests the performance of the consensus-synthesis integration engine
//! that we have implemented, focusing on the core functionality that
//! is already working and demonstrating real performance metrics.

use anyhow::{Context, Result};
use cudarc::driver::CudaContext;
use gpu_agents::consensus_synthesis::integration::{
    ConsensusSynthesisEngine, IntegrationConfig, WorkflowResult,
};
use gpu_agents::synthesis::{NodeType, Pattern, SynthesisTask, Template, Token};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug)]
struct PerformanceMetrics {
    pub total_tasks: usize,
    pub successful_tasks: usize,
    pub throughput_tasks_per_sec: f64,
    pub average_latency_ms: f64,
    pub consensus_success_rate: f64,
    pub max_latency_ms: f64,
    pub min_latency_ms: f64,
}

/// Test configuration
#[derive(Debug)]
struct TestConfig {
    pub num_tasks: usize,
    pub num_nodes: usize,
    pub consensus_threshold: f32,
    pub timeout_seconds: u64,
    pub concurrent_batches: bool,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            num_tasks: 100,
            num_nodes: 8,
            consensus_threshold: 0.7,
            timeout_seconds: 30,
            concurrent_batches: true,
        }
    }
}

async fn run_performance_test() -> Result<PerformanceMetrics> {
    println!("🚀 ExoRust Integration Performance Test");
    println!("=======================================");

    let test_config = TestConfig::default();
    let ctx = CudaContext::new(0).context("Failed to initialize CUDA device")?;

    // Initialize the integration engine
    let integration_config = IntegrationConfig {
        max_concurrent_tasks: 200,
        voting_timeout: Duration::from_secs(test_config.timeout_seconds),
        min_voters: 3,
        retry_attempts: 2,
        conflict_resolution_strategy:
            gpu_agents::consensus_synthesis::integration::ConflictStrategy::HighestVoteWins,
    };

    let engine = ConsensusSynthesisEngine::new(ctx, integration_config)
        .context("Failed to create consensus synthesis engine")?;

    println!("✅ Integration engine initialized");
    println!("   Max concurrent tasks: {}", 200);
    println!(
        "   Consensus threshold: {}%",
        test_config.consensus_threshold * 100.0
    );
    println!("   Test nodes: {}", test_config.num_nodes);

    // Create test synthesis tasks
    let synthesis_goals = vec![
        "Create matrix multiplication kernel",
        "Implement parallel reduction algorithm",
        "Generate convolution filter kernel",
        "Build sorting network implementation",
        "Create parallel prefix sum kernel",
        "Implement FFT butterfly operations",
        "Generate histogram computation kernel",
        "Build nearest neighbor search",
        "Create GEMM optimization kernel",
        "Implement parallel scan algorithm",
    ];

    let tasks: Vec<SynthesisTask> = (0..test_config.num_tasks)
        .map(|i| {
            let goal = &synthesis_goals[i % synthesis_goals.len()];
            SynthesisTask {
                pattern: Pattern {
                    node_type: NodeType::Function,
                    children: vec![],
                    value: Some(format!("{}_{}", goal, i)),
                },
                template: Template {
                    tokens: vec![
                        Token::Literal("// Synthesis Goal: ".to_string()),
                        Token::Literal(goal.to_string()),
                        Token::Literal("\n__global__ void ".to_string()),
                        Token::Variable("kernel_name".to_string()),
                        Token::Literal("(float* input, float* output, int n) {\n".to_string()),
                        Token::Literal(
                            "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n".to_string(),
                        ),
                        Token::Literal("    if (idx < n) {\n".to_string()),
                        Token::Literal("        // Generated synthesis code\n".to_string()),
                        Token::Literal("        output[idx] = input[idx] * 2.0f;\n".to_string()),
                        Token::Literal("    }\n}\n".to_string()),
                    ],
                },
            }
        })
        .collect();

    // Node configuration for consensus voting
    let node_ids: Vec<u32> = (1..=test_config.num_nodes as u32).collect();

    println!("\n📊 Running performance test...");
    println!("   Tasks to process: {}", test_config.num_tasks);
    println!("   Voting nodes: {:?}", node_ids);

    let mut successful_tasks = 0;
    let mut consensus_successes = 0;
    let mut latencies = Vec::new();

    let test_start = Instant::now();

    if test_config.concurrent_batches {
        // Test parallel batch processing
        println!("   Mode: Parallel batch processing");

        let batch_size = 25;
        let batches: Vec<Vec<SynthesisTask>> = tasks
            .chunks(batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        for (batch_idx, batch) in batches.into_iter().enumerate() {
            let batch_start = Instant::now();

            match engine.process_tasks_parallel(
                batch.clone(),
                &node_ids,
                test_config.consensus_threshold,
            ) {
                Ok(results) => {
                    let batch_latency = batch_start.elapsed();
                    latencies.push(batch_latency.as_millis() as f64 / batch.len() as f64);

                    successful_tasks += results.len();
                    for result in results {
                        if result.consensus_achieved {
                            consensus_successes += 1;
                        }
                    }

                    println!(
                        "   ✅ Batch {} completed: {} tasks in {:.2}ms",
                        batch_idx + 1,
                        batch.len(),
                        batch_latency.as_millis()
                    );
                }
                Err(e) => {
                    println!("   ❌ Batch {} failed: {}", batch_idx + 1, e);
                }
            }
        }
    } else {
        // Test individual task processing
        println!("   Mode: Sequential task processing");

        for (task_idx, task) in tasks.into_iter().enumerate() {
            let task_start = Instant::now();

            match engine.run_workflow(
                task,
                &node_ids,
                test_config.consensus_threshold,
                Duration::from_secs(test_config.timeout_seconds),
            ) {
                Ok(result) => {
                    let task_latency = task_start.elapsed().as_millis() as f64;
                    latencies.push(task_latency);

                    successful_tasks += 1;
                    if result.consensus_achieved {
                        consensus_successes += 1;
                    }

                    if task_idx % 10 == 0 {
                        println!(
                            "   ✅ Task {} completed in {:.2}ms (consensus: {})",
                            task_idx + 1,
                            task_latency,
                            result.consensus_achieved
                        );
                    }
                }
                Err(e) => {
                    println!("   ❌ Task {} failed: {}", task_idx + 1, e);
                }
            }
        }
    }

    let total_duration = test_start.elapsed();

    // Calculate performance metrics
    let throughput = successful_tasks as f64 / total_duration.as_secs_f64();
    let average_latency = if !latencies.is_empty() {
        latencies.iter().sum::<f64>() / latencies.len() as f64
    } else {
        0.0
    };
    let consensus_success_rate = consensus_successes as f64 / test_config.num_tasks as f64;
    let max_latency = latencies.iter().fold(0.0f64, |a, &b| a.max(b));
    let min_latency = latencies.iter().fold(f64::INFINITY, |a, &b| a.min(b));

    println!("\n🎯 Performance Test Results");
    println!("============================");
    println!("Total Duration: {:.2}s", total_duration.as_secs_f64());
    println!(
        "Tasks Processed: {}/{}",
        successful_tasks, test_config.num_tasks
    );
    println!("Throughput: {:.2} tasks/second", throughput);
    println!("Average Latency: {:.2} ms", average_latency);
    println!("Min Latency: {:.2} ms", min_latency);
    println!("Max Latency: {:.2} ms", max_latency);
    println!(
        "Consensus Success Rate: {:.2}%",
        consensus_success_rate * 100.0
    );

    // Performance analysis
    println!("\n📈 Performance Analysis");
    println!("========================");

    if throughput >= 20.0 {
        println!("✅ EXCELLENT: Throughput exceeds 20 tasks/second");
    } else if throughput >= 10.0 {
        println!("✅ GOOD: Throughput above 10 tasks/second");
    } else if throughput >= 5.0 {
        println!("⚠️  ACCEPTABLE: Throughput at {} tasks/second", throughput);
    } else {
        println!("❌ NEEDS OPTIMIZATION: Throughput below 5 tasks/second");
    }

    if average_latency <= 50.0 {
        println!("✅ EXCELLENT: Average latency under 50ms");
    } else if average_latency <= 100.0 {
        println!("✅ GOOD: Average latency under 100ms");
    } else if average_latency <= 500.0 {
        println!(
            "⚠️  ACCEPTABLE: Average latency at {:.2}ms",
            average_latency
        );
    } else {
        println!("❌ NEEDS OPTIMIZATION: Average latency over 500ms");
    }

    if consensus_success_rate >= 0.8 {
        println!("✅ EXCELLENT: Consensus success rate above 80%");
    } else if consensus_success_rate >= 0.6 {
        println!("✅ GOOD: Consensus success rate above 60%");
    } else {
        println!(
            "❌ NEEDS TUNING: Consensus success rate at {:.2}%",
            consensus_success_rate * 100.0
        );
    }

    // Overall verdict
    let overall_score = ((throughput / 20.0).min(1.0)
        + (100.0 / average_latency.max(1.0)).min(1.0)
        + consensus_success_rate)
        / 3.0;

    println!("\n🏆 Overall Performance Score: {:.2}/1.0", overall_score);

    if overall_score >= 0.8 {
        println!("🌟 PRODUCTION READY: System meets enterprise performance requirements");
    } else if overall_score >= 0.6 {
        println!("✅ DEPLOYMENT READY: System meets basic performance requirements");
    } else {
        println!("🔧 OPTIMIZATION NEEDED: System requires performance improvements");
    }

    Ok(PerformanceMetrics {
        total_tasks: test_config.num_tasks,
        successful_tasks,
        throughput_tasks_per_sec: throughput,
        average_latency_ms: average_latency,
        consensus_success_rate,
        max_latency_ms: max_latency,
        min_latency_ms: min_latency,
    })
}

/// Test stress scenario with high concurrent load
async fn run_stress_test() -> Result<()> {
    println!("\n🔥 Stress Test: High Concurrent Load");
    println!("=====================================");

    let ctx = CudaContext::new(0).context("Failed to initialize CUDA device")?;

    let stress_config = IntegrationConfig {
        max_concurrent_tasks: 500,
        voting_timeout: Duration::from_secs(60),
        min_voters: 5,
        retry_attempts: 3,
        conflict_resolution_strategy:
            gpu_agents::consensus_synthesis::integration::ConflictStrategy::HighestVoteWins,
    };

    let engine = ConsensusSynthesisEngine::new(ctx, stress_config)
        .context("Failed to create stress test engine")?;

    // Create 200 concurrent tasks
    let stress_tasks: Vec<SynthesisTask> = (0..200)
        .map(|i| SynthesisTask {
            pattern: Pattern {
                node_type: NodeType::Function,
                children: vec![],
                value: Some(format!("stress_test_kernel_{}", i)),
            },
            template: Template {
                tokens: vec![
                    Token::Literal("__global__ void stress_kernel_".to_string()),
                    Token::Variable("id".to_string()),
                    Token::Literal("() { /* stress test */ }".to_string()),
                ],
            },
        })
        .collect();

    let node_ids: Vec<u32> = (1..=12).collect(); // 12 nodes for stress

    println!(
        "Submitting {} concurrent tasks with {} voting nodes...",
        stress_tasks.len(),
        node_ids.len()
    );

    let stress_start = Instant::now();
    match engine.process_tasks_parallel(stress_tasks, &node_ids, 0.7) {
        Ok(results) => {
            let stress_duration = stress_start.elapsed();
            let successful = results.iter().filter(|r| r.consensus_achieved).count();

            println!(
                "✅ Stress test completed in {:.2}s",
                stress_duration.as_secs_f64()
            );
            println!("   Tasks processed: {}/{}", results.len(), 200);
            println!("   Consensus achieved: {}/{}", successful, results.len());
            println!(
                "   Stress throughput: {:.2} tasks/second",
                results.len() as f64 / stress_duration.as_secs_f64()
            );

            if results.len() >= 190 && successful >= 150 {
                println!("🌟 STRESS TEST PASSED: System handles high concurrent load");
            } else {
                println!("⚠️  STRESS TEST WARNING: Some degradation under load");
            }
        }
        Err(e) => {
            println!("❌ Stress test failed: {}", e);
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    println!("🔬 ExoRust Integration Performance Test Suite");
    println!("==============================================");
    println!("Testing consensus-synthesis integration engine");
    println!("Target: >10 tasks/sec, <100ms latency, >70% consensus\n");

    // Run main performance test
    match run_performance_test().await {
        Ok(metrics) => {
            // Run additional stress test
            if let Err(e) = run_stress_test().await {
                println!("⚠️  Stress test encountered issues: {}", e);
            }

            // Final summary
            println!("\n📋 Final Test Summary");
            println!("=====================");
            println!("Performance Test: ✅ COMPLETED");
            println!(
                "Throughput: {:.2} tasks/second",
                metrics.throughput_tasks_per_sec
            );
            println!("Latency: {:.2} ms average", metrics.average_latency_ms);
            println!(
                "Consensus: {:.2}% success rate",
                metrics.consensus_success_rate * 100.0
            );

            // Check if we meet performance targets
            let meets_targets = metrics.throughput_tasks_per_sec >= 10.0
                && metrics.average_latency_ms <= 100.0
                && metrics.consensus_success_rate >= 0.7;

            if meets_targets {
                println!("\n🎉 ALL PERFORMANCE TARGETS ACHIEVED!");
                println!("Integration is ready for production deployment.");
            } else {
                println!("\n💡 Performance targets not fully met - consider optimization:");
                if metrics.throughput_tasks_per_sec < 10.0 {
                    println!(
                        "   - Throughput: {:.2}/10.0 tasks/sec",
                        metrics.throughput_tasks_per_sec
                    );
                }
                if metrics.average_latency_ms > 100.0 {
                    println!("   - Latency: {:.2}/100.0 ms", metrics.average_latency_ms);
                }
                if metrics.consensus_success_rate < 0.7 {
                    println!(
                        "   - Consensus: {:.2}/70% success",
                        metrics.consensus_success_rate * 100.0
                    );
                }
            }
        }
        Err(e) => {
            eprintln!("❌ Performance test failed: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}
