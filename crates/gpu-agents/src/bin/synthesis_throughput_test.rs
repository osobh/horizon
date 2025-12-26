//! Synthesis Throughput Test - Measures pattern matching ops/sec

use clap::Parser;
use cudarc::driver::CudaDevice;
use gpu_agents::synthesis::pattern_dynamic::DynamicGpuPatternMatcher;
use gpu_agents::synthesis::{AstNode, NodeType, Pattern};
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of patterns to test
    #[arg(short, long, default_value = "32")]
    patterns: usize,

    /// Number of AST nodes
    #[arg(short, long, default_value = "10000")]
    nodes: usize,

    /// Test duration in seconds
    #[arg(short, long, default_value = "10")]
    duration: u64,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("🚀 GPU Synthesis Throughput Test");
    println!("================================");
    println!("Patterns: {}", args.patterns);
    println!("AST Nodes: {}", args.nodes);
    println!("Duration: {}s", args.duration);
    println!();

    // Initialize CUDA
    let device = CudaDevice::new(0)?;
    println!("✅ CUDA device initialized");

    // Get GPU info
    println!("GPU: NVIDIA GPU");
    println!();

    // Create dynamic pattern matcher
    let matcher = DynamicGpuPatternMatcher::new(device)?;
    println!("✅ Dynamic pattern matcher created");

    // Generate test patterns
    let patterns: Vec<Pattern> = (0..args.patterns)
        .map(|i| Pattern {
            node_type: match i % 4 {
                0 => NodeType::Variable,
                1 => NodeType::Function,
                2 => NodeType::Literal,
                _ => NodeType::BinaryOp,
            },
            children: if i % 3 == 0 {
                vec![Pattern {
                    node_type: NodeType::Variable,
                    children: vec![],
                    value: Some(format!("child{}", i)),
                }]
            } else {
                vec![]
            },
            value: Some(format!("pattern{}", i)),
        })
        .collect();

    // Generate test AST nodes
    let asts: Vec<AstNode> = (0..args.nodes)
        .map(|i| AstNode {
            node_type: match i % 4 {
                0 => NodeType::Variable,
                1 => NodeType::Function,
                2 => NodeType::Literal,
                _ => NodeType::BinaryOp,
            },
            children: if i % 3 == 0 {
                vec![AstNode {
                    node_type: NodeType::Variable,
                    children: vec![],
                    value: Some(format!("child{}", i % args.patterns)),
                }]
            } else {
                vec![]
            },
            value: Some(format!("pattern{}", i % args.patterns)),
        })
        .collect();

    println!("📊 Test data generated");
    println!(
        "   - Pattern complexity: {} with children",
        patterns.iter().filter(|p| !p.children.is_empty()).count()
    );
    println!(
        "   - AST complexity: {} with children",
        asts.iter().filter(|a| !a.children.is_empty()).count()
    );
    println!();

    // Warmup
    println!("🔥 Warming up GPU...");
    for _ in 0..10 {
        matcher.match_batch(&patterns[..1], &asts[..100])?;
    }

    // Benchmark
    println!("⏱️  Starting benchmark...");
    let start = Instant::now();
    let mut operations = 0u64;
    let mut iterations = 0u64;

    while start.elapsed() < Duration::from_secs(args.duration) {
        let batch_start = Instant::now();

        let results = matcher.match_batch(&patterns, &asts)?;

        operations += (patterns.len() * asts.len()) as u64;
        iterations += 1;

        if args.verbose && iterations % 100 == 0 {
            let batch_time = batch_start.elapsed();
            let batch_ops = (patterns.len() * asts.len()) as f64;
            let batch_throughput = batch_ops / batch_time.as_secs_f64();
            println!(
                "   Iteration {}: {:.2}M ops/sec",
                iterations,
                batch_throughput / 1_000_000.0
            );
        }
    }

    let elapsed = start.elapsed();
    let total_seconds = elapsed.as_secs_f64();
    let ops_per_second = operations as f64 / total_seconds;
    let billion_ops = ops_per_second / 1_000_000_000.0;

    println!();
    println!("📊 Benchmark Results");
    println!("====================");
    println!("Total operations: {}", operations);
    println!("Total iterations: {}", iterations);
    println!("Total time: {:.2}s", total_seconds);
    println!("Throughput: {:.2}M ops/sec", ops_per_second / 1_000_000.0);
    println!("Throughput: {:.3}B ops/sec", billion_ops);
    println!();

    // Performance analysis
    let ops_per_iteration = (patterns.len() * asts.len()) as f64;
    let time_per_iteration = total_seconds / iterations as f64;
    let iterations_per_second = 1.0 / time_per_iteration;

    println!("📈 Performance Analysis");
    println!("======================");
    println!("Operations per iteration: {}", ops_per_iteration);
    println!("Time per iteration: {:.3}ms", time_per_iteration * 1000.0);
    println!("Iterations per second: {:.1}", iterations_per_second);
    println!();

    // Target comparison
    let target_ops = 2.6e9;
    let achievement = (ops_per_second / target_ops) * 100.0;

    println!("🎯 Target Comparison");
    println!("===================");
    println!("Target: 2.6B ops/sec");
    println!("Achieved: {:.3}B ops/sec", billion_ops);
    println!("Achievement: {:.1}%", achievement);

    if achievement >= 100.0 {
        println!("✅ TARGET ACHIEVED!");
    } else {
        println!("⚠️  Target not yet reached");
        let needed_improvement = target_ops / ops_per_second;
        println!("   Need {:.1}x improvement", needed_improvement);
    }

    Ok(())
}
