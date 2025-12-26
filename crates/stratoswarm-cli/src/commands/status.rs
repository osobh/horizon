//! Status command implementation

use crate::{output, Result};
use clap::Args;
use comfy_table::{Cell, Color, Table};

#[derive(Debug, Clone, Args)]
pub struct StatusArgs {
    /// Filter by namespace
    #[arg(short, long)]
    pub namespace: Option<String>,

    /// Filter by agent name
    #[arg(short, long)]
    pub agent: Option<String>,

    /// Show detailed status
    #[arg(short = 'd', long)]
    pub detailed: bool,

    /// Output format (table, json, yaml)
    #[arg(short, long, default_value = "table")]
    pub format: OutputFormat,

    /// Watch for changes
    #[arg(short, long)]
    pub watch: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OutputFormat {
    Table,
    Json,
    Yaml,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "table" => Ok(OutputFormat::Table),
            "json" => Ok(OutputFormat::Json),
            "yaml" => Ok(OutputFormat::Yaml),
            _ => Err(format!("Unknown format: {}", s)),
        }
    }
}

pub async fn execute(args: StatusArgs) -> Result<()> {
    let status = fetch_status(&args).await?;

    match args.format {
        OutputFormat::Table => print_table(&status, args.detailed),
        OutputFormat::Json => print_json(&status)?,
        OutputFormat::Yaml => print_yaml(&status)?,
    }

    if args.watch {
        watch_status(args).await?;
    }

    Ok(())
}

#[derive(Debug, serde::Serialize)]
pub struct AgentStatus {
    pub name: String,
    pub namespace: String,
    pub agent_type: String,
    pub status: String,
    pub replicas: String,
    pub cpu: f64,
    pub memory: u64,
    pub gpu: Option<f64>,
    pub uptime: String,
    pub generation: u32,
    pub fitness: f64,
}

async fn fetch_status(args: &StatusArgs) -> Result<Vec<AgentStatus>> {
    // Mock implementation for now
    let mut status = vec![
        AgentStatus {
            name: "frontend".to_string(),
            namespace: "default".to_string(),
            agent_type: "WebAgent".to_string(),
            status: "Running".to_string(),
            replicas: "3/3".to_string(),
            cpu: 0.5,
            memory: 1024 * 1024 * 512, // 512MB
            gpu: None,
            uptime: "2d 5h".to_string(),
            generation: 42,
            fitness: 0.95,
        },
        AgentStatus {
            name: "backend".to_string(),
            namespace: "default".to_string(),
            agent_type: "ComputeAgent".to_string(),
            status: "Running".to_string(),
            replicas: "5/5".to_string(),
            cpu: 2.0,
            memory: 1024 * 1024 * 1024 * 4, // 4GB
            gpu: Some(0.5),
            uptime: "2d 5h".to_string(),
            generation: 38,
            fitness: 0.88,
        },
    ];

    // Apply filters
    if let Some(ns) = &args.namespace {
        status.retain(|s| &s.namespace == ns);
    }

    if let Some(agent) = &args.agent {
        status.retain(|s| s.name.contains(agent));
    }

    Ok(status)
}

fn print_table(status: &[AgentStatus], detailed: bool) {
    let mut table = Table::new();

    if detailed {
        table.set_header(vec![
            "Name",
            "Namespace",
            "Type",
            "Status",
            "Replicas",
            "CPU",
            "Memory",
            "GPU",
            "Uptime",
            "Generation",
            "Fitness",
        ]);

        for s in status {
            let status_cell = if s.status == "Running" {
                Cell::new(&s.status).fg(Color::Green)
            } else {
                Cell::new(&s.status).fg(Color::Red)
            };

            table.add_row(vec![
                Cell::new(&s.name),
                Cell::new(&s.namespace),
                Cell::new(&s.agent_type),
                status_cell,
                Cell::new(&s.replicas),
                Cell::new(format!("{:.2}", s.cpu)),
                Cell::new(humansize::format_size(s.memory, humansize::BINARY)),
                Cell::new(
                    s.gpu
                        .map(|g| format!("{:.2}", g))
                        .unwrap_or_else(|| "-".to_string()),
                ),
                Cell::new(&s.uptime),
                Cell::new(s.generation.to_string()),
                Cell::new(format!("{:.2}", s.fitness)),
            ]);
        }
    } else {
        table.set_header(vec!["Name", "Type", "Status", "Replicas", "CPU", "Memory"]);

        for s in status {
            let status_cell = if s.status == "Running" {
                Cell::new(&s.status).fg(Color::Green)
            } else {
                Cell::new(&s.status).fg(Color::Red)
            };

            table.add_row(vec![
                Cell::new(&s.name),
                Cell::new(&s.agent_type),
                status_cell,
                Cell::new(&s.replicas),
                Cell::new(format!("{:.2}", s.cpu)),
                Cell::new(humansize::format_size(s.memory, humansize::BINARY)),
            ]);
        }
    }

    println!("{}", table);
}

fn print_json(status: &[AgentStatus]) -> Result<()> {
    let json = serde_json::to_string_pretty(status)?;
    println!("{}", json);
    Ok(())
}

fn print_yaml(status: &[AgentStatus]) -> Result<()> {
    let yaml = serde_yaml::to_string(status)
        .map_err(|e| crate::CliError::Command(format!("Failed to serialize to YAML: {}", e)))?;
    println!("{}", yaml);
    Ok(())
}

async fn watch_status(args: StatusArgs) -> Result<()> {
    use std::time::Duration;

    output::info("Watching for status changes... (press Ctrl+C to stop)");

    loop {
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Clear screen
        print!("\x1B[2J\x1B[1;1H");

        let status = fetch_status(&args).await?;
        print_table(&status, args.detailed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_format_parsing() {
        assert_eq!(
            "table".parse::<OutputFormat>().unwrap(),
            OutputFormat::Table
        );
        assert_eq!("json".parse::<OutputFormat>().unwrap(), OutputFormat::Json);
        assert_eq!("yaml".parse::<OutputFormat>().unwrap(), OutputFormat::Yaml);
        assert!("invalid".parse::<OutputFormat>().is_err());
    }

    #[tokio::test]
    async fn test_fetch_status_with_filters() {
        let args = StatusArgs {
            namespace: Some("default".to_string()),
            agent: Some("front".to_string()),
            detailed: false,
            format: OutputFormat::Table,
            watch: false,
        };

        let status = fetch_status(&args).await.unwrap();
        assert_eq!(status.len(), 1);
        assert_eq!(status[0].name, "frontend");
    }

    #[tokio::test]
    async fn test_execute_json_format() {
        let args = StatusArgs {
            namespace: None,
            agent: None,
            detailed: false,
            format: OutputFormat::Json,
            watch: false,
        };

        let result = execute(args).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_table() {
        let status = vec![AgentStatus {
            name: "test".to_string(),
            namespace: "default".to_string(),
            agent_type: "WebAgent".to_string(),
            status: "Running".to_string(),
            replicas: "1/1".to_string(),
            cpu: 0.5,
            memory: 1024 * 1024 * 256,
            gpu: None,
            uptime: "1h".to_string(),
            generation: 1,
            fitness: 0.9,
        }];

        // Should not panic
        print_table(&status, false);
        print_table(&status, true);
    }
}
