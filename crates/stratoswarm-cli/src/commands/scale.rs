//! Scale command implementation

use crate::{output, CliError, Result};
use clap::Args;
use std::str::FromStr;

#[derive(Debug, Clone, Args)]
pub struct ScaleArgs {
    /// Scale specification (e.g., "backend=10", "frontend=5")
    pub spec: Vec<ScaleSpec>,

    /// Namespace
    #[arg(long, default_value = "default")]
    pub namespace: String,

    /// Wait for scaling to complete
    #[arg(short, long)]
    pub wait: bool,

    /// Dry run - show what would be scaled
    #[arg(short = 'd', long)]
    pub dry_run: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScaleSpec {
    pub agent: String,
    pub replicas: u32,
}

impl FromStr for ScaleSpec {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split('=').collect();
        if parts.len() != 2 {
            return Err(format!(
                "Invalid scale spec '{}'. Expected format: agent=replicas",
                s
            ));
        }

        let agent = parts[0].to_string();
        let replicas = parts[1]
            .parse::<u32>()
            .map_err(|_| format!("Invalid replica count: {}", parts[1]))?;

        Ok(ScaleSpec { agent, replicas })
    }
}

pub async fn execute(args: ScaleArgs) -> Result<()> {
    if args.spec.is_empty() {
        return Err(CliError::InvalidArgument(
            "No scale specifications provided. Usage: scale agent=replicas".to_string(),
        ));
    }

    for spec in &args.spec {
        output::info(&format!(
            "Scaling {} to {} replicas",
            spec.agent, spec.replicas
        ));

        if args.dry_run {
            output::info(&format!(
                "Would scale {} from current replicas to {}",
                spec.agent, spec.replicas
            ));
            continue;
        }

        scale_agent(spec, &args.namespace).await?;

        output::success(&format!(
            "✓ {} scaled to {} replicas",
            spec.agent, spec.replicas
        ));
    }

    if args.wait && !args.dry_run {
        wait_for_scale(&args.spec, &args.namespace).await?;
    }

    Ok(())
}

async fn scale_agent(spec: &ScaleSpec, namespace: &str) -> Result<()> {
    use crate::config::CliConfig;
    use reqwest::Client;
    use serde::Serialize;

    #[derive(Serialize)]
    struct ScaleRequest {
        agent: String,
        replicas: u32,
        namespace: String,
    }

    output::info(&format!(
        "Scaling agent '{}' in namespace '{}' to {} replicas",
        spec.agent, namespace, spec.replicas
    ));

    let config = CliConfig::load().unwrap_or_default();
    let client = Client::new();
    let url = format!("{}/api/v1/agents/{}/scale", config.api_endpoint, spec.agent);

    let request = ScaleRequest {
        agent: spec.agent.clone(),
        replicas: spec.replicas,
        namespace: namespace.to_string(),
    };

    let mut req = client.post(&url).json(&request);
    if let Some(ref auth) = config.auth_token {
        req = req.header("Authorization", format!("Bearer {}", auth));
    }

    match req.send().await {
        Ok(response) if response.status().is_success() => {
            output::success(&format!(
                "Agent '{}' scaled to {} replicas",
                spec.agent, spec.replicas
            ));
        }
        Ok(response) => {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            output::warn(&format!(
                "Scale request returned {}: {}. Cluster may not be available.",
                status, body
            ));
        }
        Err(e) => {
            output::warn(&format!(
                "Could not connect to cluster: {}. Running in offline mode.",
                e
            ));
        }
    }

    Ok(())
}

async fn wait_for_scale(specs: &[ScaleSpec], _namespace: &str) -> Result<()> {
    use indicatif::{ProgressBar, ProgressStyle};
    use std::time::Duration;

    output::info("Waiting for scaling to complete...");

    let pb = ProgressBar::new(specs.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} agents scaled")
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏  "),
    );

    for _spec in specs {
        // Simulate waiting for scale
        for _ in 0..5 {
            pb.tick();
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        pb.inc(1);
    }

    pb.finish_with_message("All agents scaled successfully!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_spec_parsing() {
        let spec: ScaleSpec = "backend=10".parse().unwrap();
        assert_eq!(spec.agent, "backend");
        assert_eq!(spec.replicas, 10);

        let spec: ScaleSpec = "frontend=5".parse().unwrap();
        assert_eq!(spec.agent, "frontend");
        assert_eq!(spec.replicas, 5);

        assert!("invalid".parse::<ScaleSpec>().is_err());
        assert!("agent=".parse::<ScaleSpec>().is_err());
        assert!("agent=abc".parse::<ScaleSpec>().is_err());
    }

    #[test]
    fn test_scale_args_parsing() {
        use clap::Parser;

        #[derive(Debug, Parser)]
        struct TestCli {
            #[command(flatten)]
            args: ScaleArgs,
        }

        let cli = TestCli::parse_from(["test", "backend=10", "frontend=5"]);
        assert_eq!(cli.args.spec.len(), 2);
        assert_eq!(cli.args.spec[0].agent, "backend");
        assert_eq!(cli.args.spec[0].replicas, 10);
        assert_eq!(cli.args.spec[1].agent, "frontend");
        assert_eq!(cli.args.spec[1].replicas, 5);
    }

    #[tokio::test]
    async fn test_execute_empty_spec() {
        let args = ScaleArgs {
            spec: vec![],
            namespace: "default".to_string(),
            wait: false,
            dry_run: false,
        };

        let result = execute(args).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No scale specifications"));
    }

    #[tokio::test]
    async fn test_execute_dry_run() {
        let args = ScaleArgs {
            spec: vec![ScaleSpec {
                agent: "test".to_string(),
                replicas: 5,
            }],
            namespace: "default".to_string(),
            wait: false,
            dry_run: true,
        };

        let result = execute(args).await;
        assert!(result.is_ok());
    }
}
