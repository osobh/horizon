//! Storage benchmark CLI
//!
//! Run storage benchmarks for GPU agent swarms

use anyhow::Result;
use clap::{Parser, ValueEnum};
use gpu_agents::benchmarks::storage_benchmark::StorageBenchmarkReport;
use gpu_agents::benchmarks::{
    StorageBenchmark, StorageBenchmarkConfig, StorageBenchmarkResults, StorageBenchmarkScenario,
};
use std::path::PathBuf;

#[derive(Debug, Clone, ValueEnum)]
enum Scenario {
    BurstWrite,
    RandomAccess,
    HotCold,
    SwarmCheckpoint,
    KnowledgeGraph,
    Concurrent,
    MemoryPressure,
    Evolution,
    All,
}

impl From<Scenario> for StorageBenchmarkScenario {
    fn from(s: Scenario) -> Self {
        match s {
            Scenario::BurstWrite => StorageBenchmarkScenario::BurstWrite,
            Scenario::RandomAccess => StorageBenchmarkScenario::RandomAccess,
            Scenario::HotCold => StorageBenchmarkScenario::HotColdPattern,
            Scenario::SwarmCheckpoint => StorageBenchmarkScenario::SwarmCheckpoint,
            Scenario::KnowledgeGraph => StorageBenchmarkScenario::KnowledgeGraphUpdate,
            Scenario::Concurrent => StorageBenchmarkScenario::ConcurrentSwarmAccess,
            Scenario::MemoryPressure => StorageBenchmarkScenario::MemoryPressure,
            Scenario::Evolution => StorageBenchmarkScenario::EvolutionPersistence,
            Scenario::All => unreachable!(),
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about = "GPU Agent Storage Benchmark", long_about = None)]
struct Args {
    /// Benchmark scenario to run
    #[arg(short, long, value_enum, default_value = "all")]
    scenario: Scenario,

    /// Number of agents to test
    #[arg(short, long, default_value = "10000")]
    agents: usize,

    /// Number of iterations
    #[arg(short, long, default_value = "100")]
    iterations: usize,

    /// Enable GPU cache
    #[arg(long, default_value = "true")]
    gpu_cache: bool,

    /// GPU cache size in MB
    #[arg(long, default_value = "1024")]
    cache_mb: usize,

    /// Number of concurrent tasks
    #[arg(long, default_value = "4")]
    concurrent: usize,

    /// Agent state vector size
    #[arg(long, default_value = "256")]
    state_size: usize,

    /// Enable compression
    #[arg(long)]
    compression: bool,

    /// Storage path (default: temp directory)
    #[arg(long)]
    storage_path: Option<PathBuf>,

    /// Output results as JSON
    #[arg(long)]
    json: bool,

    /// Save results to file
    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("🚀 GPU Agent Storage Benchmark");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Run scenarios
    let scenarios = if matches!(args.scenario, Scenario::All) {
        vec![
            Scenario::BurstWrite,
            Scenario::RandomAccess,
            Scenario::HotCold,
            Scenario::SwarmCheckpoint,
            Scenario::KnowledgeGraph,
            Scenario::Concurrent,
            Scenario::MemoryPressure,
            Scenario::Evolution,
        ]
    } else {
        vec![args.scenario.clone()]
    };

    let mut all_results = Vec::new();

    for scenario in scenarios {
        println!("\n▶️  Running {} scenario...", format!("{:?}", scenario));

        let config = StorageBenchmarkConfig {
            scenario: scenario.clone().into(),
            agent_count: args.agents,
            iterations: args.iterations,
            enable_gpu_cache: args.gpu_cache,
            cache_size_mb: args.cache_mb,
            concurrent_tasks: args.concurrent,
            agent_state_size: args.state_size,
            use_compression: args.compression,
            storage_path: args
                .storage_path
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            ..Default::default()
        };

        let benchmark = StorageBenchmark::new(config)?;
        let results = benchmark.run().await?;

        if !args.json {
            // Print human-readable results
            println!("\n📊 Results for {:?}:", scenario);
            println!("   Duration: {:.2}s", results.duration.as_secs_f64());
            println!(
                "   Store throughput: {:.0} agents/sec",
                results.metrics.store_throughput_agents_per_sec
            );
            println!(
                "   Retrieve throughput: {:.0} agents/sec",
                results.metrics.retrieve_throughput_agents_per_sec
            );

            if results.metrics.cache_hit_rate > 0.0 {
                println!(
                    "   Cache hit rate: {:.1}%",
                    results.metrics.cache_hit_rate * 100.0
                );
            }

            if results.metrics.concurrent_throughput > 0.0 {
                println!(
                    "   Concurrent throughput: {:.0} agents/sec",
                    results.metrics.concurrent_throughput
                );
            }

            println!(
                "   Avg store latency: {:.2}ms (P99: {:.2}ms)",
                results.metrics.avg_store_latency_ms, results.metrics.p99_store_latency_ms
            );
            println!(
                "   Avg retrieve latency: {:.2}ms (P99: {:.2}ms)",
                results.metrics.avg_retrieve_latency_ms, results.metrics.p99_retrieve_latency_ms
            );

            // Show warnings
            let report = StorageBenchmarkReport::from_metrics(results.metrics.clone());
            for warning in &report.warnings {
                println!("   {}", warning);
            }
        }

        all_results.push(results);
    }

    // Save results if requested
    if let Some(output_path) = args.output {
        let json = serde_json::to_string_pretty(&all_results)?;
        tokio::fs::write(&output_path, json).await?;
        println!("\n✅ Results saved to: {}", output_path.display());
    }

    // Print JSON if requested
    if args.json {
        println!("{}", serde_json::to_string_pretty(&all_results)?);
    }

    // Summary comparison if multiple scenarios
    if all_results.len() > 1 && !args.json {
        println!("\n📈 Scenario Comparison:");
        println!("━━━━━━━━━━━━━━━━━━━━━━");

        let comparison = gpu_agents::benchmarks::storage_benchmark::compare_scenarios(all_results);

        println!(
            "Best write performance: {:?} ({:.0} agents/sec)",
            comparison.best_write_scenario, comparison.best_write_throughput
        );
        println!(
            "Best read performance: {:?} ({:.0} agents/sec)",
            comparison.best_read_scenario, comparison.best_read_throughput
        );
    }

    println!("\n✅ Storage benchmark complete!");

    Ok(())
}
