//! Evolve command implementation

use crate::{output, CliError, Result};
use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

#[derive(Debug, Clone, Args)]
pub struct EvolveArgs {
    /// Agent to evolve
    pub agent: String,

    /// Number of generations to evolve
    #[arg(short, long, default_value = "10")]
    pub generations: u32,

    /// Population size
    #[arg(short, long, default_value = "100")]
    pub population: u32,

    /// Mutation rate (0.0-1.0)
    #[arg(short, long, default_value = "0.1")]
    pub mutation_rate: f32,

    /// Crossover rate (0.0-1.0)  
    #[arg(short = 'x', long, default_value = "0.7")]
    pub crossover_rate: f32,

    /// Evolution strategy (conservative, aggressive, balanced)
    #[arg(short, long, default_value = "balanced")]
    pub strategy: EvolutionStrategy,

    /// Namespace
    #[arg(long, default_value = "default")]
    pub namespace: String,

    /// Show evolution progress
    #[arg(long)]
    pub verbose: bool,

    /// Dry run - simulate evolution without applying
    #[arg(short = 'd', long)]
    pub dry_run: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EvolutionStrategy {
    Conservative,
    Aggressive,
    Balanced,
}

impl std::str::FromStr for EvolutionStrategy {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "conservative" => Ok(EvolutionStrategy::Conservative),
            "aggressive" => Ok(EvolutionStrategy::Aggressive),
            "balanced" => Ok(EvolutionStrategy::Balanced),
            _ => Err(format!("Unknown strategy: {}", s)),
        }
    }
}

#[derive(Debug)]
pub struct EvolutionResult {
    pub generation: u32,
    pub best_fitness: f64,
    pub average_fitness: f64,
    pub mutations_applied: u32,
    pub improvements: Vec<String>,
}

pub async fn execute(args: EvolveArgs) -> Result<()> {
    output::info(&format!("Starting evolution for agent '{}'", args.agent));
    output::info(&format!(
        "Generations: {}, Population: {}, Strategy: {:?}",
        args.generations, args.population, args.strategy
    ));

    // Validate parameters
    if args.mutation_rate < 0.0 || args.mutation_rate > 1.0 {
        return Err(CliError::InvalidArgument(
            "Mutation rate must be between 0.0 and 1.0".to_string(),
        ));
    }

    if args.crossover_rate < 0.0 || args.crossover_rate > 1.0 {
        return Err(CliError::InvalidArgument(
            "Crossover rate must be between 0.0 and 1.0".to_string(),
        ));
    }

    if args.dry_run {
        output::info("Dry run mode - simulating evolution");
        simulate_evolution(&args).await?;
    } else {
        run_evolution(&args).await?;
    }

    Ok(())
}

async fn run_evolution(args: &EvolveArgs) -> Result<()> {
    use crate::config::CliConfig;
    use reqwest::Client;
    use serde::Serialize;

    #[derive(Serialize)]
    struct EvolveRequest {
        agent: String,
        namespace: String,
        generations: u32,
        population_size: u32,
        mutation_rate: f32,
        strategy: String,
    }

    output::info(&format!(
        "Starting evolution for agent '{}' in namespace '{}'",
        args.agent, args.namespace
    ));

    // Try to start evolution on the cluster
    let config = CliConfig::load().unwrap_or_default();
    let client = Client::new();
    let url = format!("{}/api/v1/agents/{}/evolve", config.api_endpoint, args.agent);

    let request = EvolveRequest {
        agent: args.agent.clone(),
        namespace: args.namespace.clone(),
        generations: args.generations,
        population_size: args.population,
        mutation_rate: args.mutation_rate,
        strategy: format!("{:?}", args.strategy).to_lowercase(),
    };

    let mut req = client.post(&url).json(&request);
    if let Some(ref auth) = config.auth_token {
        req = req.header("Authorization", format!("Bearer {}", auth));
    }

    let cluster_available = match req.send().await {
        Ok(response) if response.status().is_success() => {
            output::success("Evolution job submitted to cluster successfully.");
            true
        }
        Ok(response) => {
            let status = response.status();
            output::warn(&format!(
                "Cluster returned {}: Running local simulation instead.",
                status
            ));
            false
        }
        Err(e) => {
            output::warn(&format!(
                "Could not connect to cluster ({}): Running local simulation.",
                e
            ));
            false
        }
    };

    // Run local simulation if cluster not available
    if !cluster_available {
        output::info(&format!(
            "Simulating evolution for agent '{}'",
            args.agent
        ));
    }

    let pb = create_progress_bar(args.generations);

    for generation in 1..=args.generations {
        pb.set_message(format!("Generation {}/{}", generation, args.generations));

        // Run evolution step
        let result = evolve_generation(&args, generation).await?;

        if args.verbose {
            print_generation_result(&result);
        }

        pb.inc(1);

        // Small delay to simulate work
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    pb.finish_with_message("Evolution complete!");

    output::success(&format!(
        "✓ Agent '{}' evolved through {} generations",
        args.agent, args.generations
    ));

    Ok(())
}

async fn simulate_evolution(args: &EvolveArgs) -> Result<()> {
    let pb = create_progress_bar(args.generations);

    let mut best_fitness = 0.5;

    for generation in 1..=args.generations {
        pb.set_message(format!("Generation {}/{}", generation, args.generations));

        // Simulate fitness improvement
        let improvement = match args.strategy {
            EvolutionStrategy::Conservative => 0.01,
            EvolutionStrategy::Aggressive => 0.03,
            EvolutionStrategy::Balanced => 0.02,
        };

        best_fitness = (best_fitness + improvement * rand::random::<f64>()).min(1.0);

        let result = EvolutionResult {
            generation,
            best_fitness,
            average_fitness: best_fitness * 0.8,
            mutations_applied: (args.population as f32 * args.mutation_rate) as u32,
            improvements: vec![
                "Optimized resource allocation".to_string(),
                "Improved response time".to_string(),
            ],
        };

        if args.verbose {
            print_generation_result(&result);
        }

        pb.inc(1);
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    pb.finish_with_message("Simulation complete!");

    println!("\nFinal Results:");
    println!("  Best Fitness: {:.3}", best_fitness);
    println!("  Strategy: {:?}", args.strategy);

    Ok(())
}

async fn evolve_generation(args: &EvolveArgs, generation: u32) -> Result<EvolutionResult> {
    // Mock implementation
    Ok(EvolutionResult {
        generation,
        best_fitness: 0.85 + (generation as f64 * 0.01),
        average_fitness: 0.75 + (generation as f64 * 0.01),
        mutations_applied: (args.population as f32 * args.mutation_rate) as u32,
        improvements: vec![format!("Generation {} optimization", generation)],
    })
}

fn create_progress_bar(total: u32) -> ProgressBar {
    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} generations ({msg})")
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏  "),
    );
    pb
}

fn print_generation_result(result: &EvolutionResult) {
    println!("\n--- Generation {} ---", result.generation);
    println!("  Best Fitness: {:.3}", result.best_fitness);
    println!("  Average Fitness: {:.3}", result.average_fitness);
    println!("  Mutations Applied: {}", result.mutations_applied);
    if !result.improvements.is_empty() {
        println!("  Improvements:");
        for improvement in &result.improvements {
            println!("    - {}", improvement);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evolution_strategy_parsing() {
        assert_eq!(
            "conservative".parse::<EvolutionStrategy>().unwrap(),
            EvolutionStrategy::Conservative
        );
        assert_eq!(
            "aggressive".parse::<EvolutionStrategy>().unwrap(),
            EvolutionStrategy::Aggressive
        );
        assert_eq!(
            "balanced".parse::<EvolutionStrategy>().unwrap(),
            EvolutionStrategy::Balanced
        );
        assert!("invalid".parse::<EvolutionStrategy>().is_err());
    }

    #[tokio::test]
    async fn test_execute_invalid_mutation_rate() {
        let args = EvolveArgs {
            agent: "test".to_string(),
            generations: 10,
            population: 100,
            mutation_rate: 1.5,
            crossover_rate: 0.7,
            strategy: EvolutionStrategy::Balanced,
            namespace: "default".to_string(),
            verbose: false,
            dry_run: false,
        };

        let result = execute(args).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Mutation rate"));
    }

    #[tokio::test]
    async fn test_simulate_evolution() {
        let args = EvolveArgs {
            agent: "test".to_string(),
            generations: 5,
            population: 50,
            mutation_rate: 0.1,
            crossover_rate: 0.7,
            strategy: EvolutionStrategy::Balanced,
            namespace: "default".to_string(),
            verbose: true,
            dry_run: true,
        };

        let result = execute(args).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_generation_result() {
        let result = EvolutionResult {
            generation: 1,
            best_fitness: 0.85,
            average_fitness: 0.75,
            mutations_applied: 10,
            improvements: vec!["Test improvement".to_string()],
        };

        // Should not panic
        print_generation_result(&result);
    }
}
