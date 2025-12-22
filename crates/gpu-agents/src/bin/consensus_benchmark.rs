//! GPU Consensus Performance Benchmark
//!
//! Validates the <100μs consensus latency claim with various agent counts
//! and Byzantine fault scenarios.

use clap::Parser;
use cudarc::driver::CudaDevice;
use gpu_agents::consensus::{GpuConsensusModule, Proposal, Vote};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of agents in the consensus
    #[arg(short, long, default_value = "1000")]
    agents: usize,

    /// Number of iterations to run
    #[arg(short, long, default_value = "1000")]
    iterations: usize,

    /// Percentage of Byzantine (faulty) nodes (0-33)
    #[arg(short, long, default_value = "0")]
    byzantine_percent: u8,

    /// Output file for results
    #[arg(short, long, default_value = "consensus_results.json")]
    output: String,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Warmup iterations
    #[arg(short, long, default_value = "100")]
    warmup: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkResults {
    agent_count: usize,
    iterations: usize,
    byzantine_percent: u8,
    latencies_us: Vec<f64>,
    statistics: LatencyStats,
    gpu_info: GpuInfo,
    timestamp: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct LatencyStats {
    min_us: f64,
    max_us: f64,
    mean_us: f64,
    median_us: f64,
    p95_us: f64,
    p99_us: f64,
    std_dev_us: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct GpuInfo {
    name: String,
    compute_capability: String,
    memory_mb: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("🚀 GPU Consensus Performance Benchmark");
    println!("=====================================");
    println!("Agents: {}", args.agents);
    println!("Iterations: {}", args.iterations);
    println!("Byzantine: {}%", args.byzantine_percent);
    println!();

    // Initialize CUDA
    let device = CudaDevice::new(0)?;
    let gpu_info = get_gpu_info(&device)?;

    println!("GPU: {} ({}MB)", gpu_info.name, gpu_info.memory_mb);
    println!();

    // Create consensus module
    let mut consensus = GpuConsensusModule::new(device.clone(), args.agents)?;

    // Generate test votes
    let votes = generate_votes(args.agents, args.byzantine_percent);
    let proposal = Proposal {
        id: 1,
        proposer_id: 0,
        value: 42,
        round: 1,
    };

    // Warmup
    if args.verbose {
        println!("Running {} warmup iterations...", args.warmup);
    }
    for _ in 0..args.warmup {
        let _ = consensus.run_consensus_round(proposal, &votes)?;
    }

    // Synchronize after warmup
    device.synchronize()?;

    // Benchmark
    println!("Running {} benchmark iterations...", args.iterations);
    let mut latencies_us = Vec::with_capacity(args.iterations);

    let total_start = Instant::now();

    for i in 0..args.iterations {
        // Create new proposal for each round
        let proposal = Proposal {
            id: (i + 1) as u32,
            proposer_id: (i % args.agents) as u32,
            value: 42 + i as u32,
            round: (i + 1) as u32,
        };

        // Time the consensus operation
        let start = Instant::now();
        let _result = consensus.run_consensus_round(proposal, &votes)?;
        device.synchronize()?; // Ensure GPU operations complete
        let duration = start.elapsed();

        let latency_us = duration.as_secs_f64() * 1_000_000.0;
        latencies_us.push(latency_us);

        if args.verbose && i % 100 == 0 {
            println!("  Iteration {}: {:.2} μs", i, latency_us);
        }
    }

    let total_duration = total_start.elapsed();

    // Calculate statistics
    let statistics = calculate_statistics(&latencies_us);

    // Print results
    println!("\n📊 Results:");
    println!("===========");
    println!("Total time: {:.2} seconds", total_duration.as_secs_f64());
    println!(
        "Throughput: {:.0} consensus/sec",
        args.iterations as f64 / total_duration.as_secs_f64()
    );
    println!();
    println!("Latency Statistics:");
    println!("  Min:    {:.2} μs", statistics.min_us);
    println!("  Mean:   {:.2} μs", statistics.mean_us);
    println!("  Median: {:.2} μs", statistics.median_us);
    println!("  P95:    {:.2} μs", statistics.p95_us);
    println!("  P99:    {:.2} μs", statistics.p99_us);
    println!("  Max:    {:.2} μs", statistics.max_us);
    println!("  StdDev: {:.2} μs", statistics.std_dev_us);

    // Check against target
    let success = statistics.p95_us < 100.0;
    println!();
    if success {
        println!(
            "✅ SUCCESS: P95 latency {:.2} μs < 100 μs target",
            statistics.p95_us
        );
    } else {
        println!(
            "❌ FAILURE: P95 latency {:.2} μs > 100 μs target",
            statistics.p95_us
        );
    }

    // Save results
    let results = BenchmarkResults {
        agent_count: args.agents,
        iterations: args.iterations,
        byzantine_percent: args.byzantine_percent,
        latencies_us,
        statistics,
        gpu_info,
        timestamp: chrono::Local::now().to_rfc3339(),
    };

    let json = serde_json::to_string_pretty(&results)?;
    let mut file = File::create(&args.output)?;
    file.write_all(json.as_bytes())?;

    println!("\nResults saved to: {}", args.output);

    Ok(())
}

fn generate_votes(num_agents: usize, byzantine_percent: u8) -> Vec<Vote> {
    let num_byzantine = (num_agents * byzantine_percent as usize) / 100;
    let mut votes = Vec::with_capacity(num_agents);

    for i in 0..num_agents {
        let value = if i < num_byzantine {
            // Byzantine nodes vote randomly
            (i % 3) as u32
        } else {
            // Honest nodes vote for value 1
            1
        };

        votes.push(Vote {
            agent_id: i as u32,
            proposal_id: 1,
            value,
            timestamp: 1000 + i as u64,
        });
    }

    votes
}

fn calculate_statistics(latencies: &[f64]) -> LatencyStats {
    let mut sorted = latencies.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b)?);

    let n = sorted.len();
    let min = sorted[0];
    let max = sorted[n - 1];
    let median = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };

    let mean = latencies.iter().sum::<f64>() / n as f64;

    let p95_idx = (n as f64 * 0.95) as usize;
    let p99_idx = (n as f64 * 0.99) as usize;
    let p95 = sorted[p95_idx.min(n - 1)];
    let p99 = sorted[p99_idx.min(n - 1)];

    let variance = latencies.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();

    LatencyStats {
        min_us: min,
        max_us: max,
        mean_us: mean,
        median_us: median,
        p95_us: p95,
        p99_us: p99,
        std_dev_us: std_dev,
    }
}

fn get_gpu_info(_device: &Arc<CudaDevice>) -> anyhow::Result<GpuInfo> {
    // Get device properties
    // Note: This is simplified - in production you'd use cuDeviceGetAttribute
    Ok(GpuInfo {
        name: "NVIDIA GPU".to_string(), // Would get actual name via CUDA API
        compute_capability: "8.9".to_string(), // Would get via cuDeviceGetAttribute
        memory_mb: 24576,               // Would get actual memory via cuMemGetInfo
    })
}
