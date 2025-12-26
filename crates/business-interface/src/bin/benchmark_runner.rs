//! LLM Benchmark Runner
//!
//! Usage: cargo run --bin benchmark_runner

use stratoswarm_business_interface::benchmarks::{ComprehensiveBenchmarkResults, LlmBenchmark};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting LLM Benchmark Suite for ExoRust");
    println!("{}", "=".repeat(50));

    // Initialize benchmark
    let mut benchmark = LlmBenchmark::new();

    // Check if Ollama is available
    match benchmark.initialize().await {
        Ok(_) => {
            println!("✅ Ollama connection established");
        }
        Err(e) => {
            println!("❌ Failed to connect to Ollama: {}", e);
            println!("💡 Make sure Ollama is running and models are available");
            println!("   Run: ollama list");
            return Ok(());
        }
    }

    // Run comprehensive benchmarks
    println!("\n🔍 Running comprehensive benchmarks...");
    println!("This may take several minutes depending on the number of models\n");

    let start_time = std::time::Instant::now();

    match benchmark.run_comprehensive_benchmark().await {
        Ok(results) => {
            println!("✅ Benchmarks completed in {:?}", start_time.elapsed());
            print_results(&results);
        }
        Err(e) => {
            println!("❌ Benchmark failed: {}", e);
        }
    }

    Ok(())
}

fn print_results(results: &ComprehensiveBenchmarkResults) {
    println!("\n📊 BENCHMARK RESULTS");
    println!("{}", "=".repeat(80));

    // Print overall statistics
    println!("Total Duration: {:?}", results.total_duration);
    println!(
        "Total Tests: {}",
        results.results.iter().map(|r| r.test_cases).sum::<usize>()
    );
    println!(
        "Models Tested: {}",
        results
            .results
            .iter()
            .map(|r| &r.model_name)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    // Print recommendations
    println!("\n🏆 RECOMMENDED MODELS BY TASK");
    println!("{}", "-".repeat(80));
    for rec in &results.recommendations {
        println!("Task: {:?}", rec.task_type);
        println!("  Recommended Model: {}", rec.recommended_model);
        println!("  Score: {:.3}", rec.score);
        println!("  Reason: {}", rec.reason);
        println!();
    }

    // Print detailed results
    println!("📈 DETAILED RESULTS");
    println!("{}", "-".repeat(80));

    for result in &results.results {
        println!(
            "Model: {} | Task: {:?}",
            result.model_name, result.task_type
        );
        println!(
            "  ⏱️  Avg Response Time: {:.0}ms",
            result.avg_response_time_ms
        );
        println!("  ✅ Success Rate: {:.1}%", result.success_rate * 100.0);
        println!("  ⭐ Quality Score: {:.3}", result.quality_score);
        println!("  📝 Test Cases: {}", result.test_cases);

        if !result.errors.is_empty() {
            println!("  ❌ Errors: {}", result.errors.len());
            for error in result.errors.iter().take(3) {
                println!("     - {}", error);
            }
            if result.errors.len() > 3 {
                println!("     ... and {} more", result.errors.len() - 3);
            }
        }
        println!();
    }

    // Performance summary
    println!("📊 PERFORMANCE SUMMARY");
    println!("{}", "-".repeat(80));

    let task_types = [
        ("Goal Parsing", "TaskType::GoalParsing"),
        ("Safety Validation", "TaskType::SafetyValidation"),
        ("Code Generation", "TaskType::CodeGeneration"),
        ("Reasoning", "TaskType::Reasoning"),
        ("Explanation", "TaskType::Explanation"),
    ];

    for (task_name, _) in task_types {
        let task_results: Vec<_> = results
            .results
            .iter()
            .filter(|r| format!("{:?}", r.task_type).contains(&task_name.replace(" ", "")))
            .collect();

        if !task_results.is_empty() {
            let avg_time: f64 = task_results
                .iter()
                .map(|r| r.avg_response_time_ms)
                .sum::<f64>()
                / task_results.len() as f64;
            let avg_quality: f64 = task_results.iter().map(|r| r.quality_score).sum::<f64>()
                / task_results.len() as f64;
            let avg_success: f64 = task_results.iter().map(|r| r.success_rate).sum::<f64>()
                / task_results.len() as f64;

            println!(
                "{}: Avg {:.0}ms | Quality {:.3} | Success {:.1}%",
                task_name,
                avg_time,
                avg_quality,
                avg_success * 100.0
            );
        }
    }

    println!("\n💡 Use these results to optimize model selection for ExoRust agents!");
}
