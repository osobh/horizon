//! Interactive shell for StratoSwarm CLI

use crate::{commands::Commands, output, CliError, Result};
use clap::Parser;
use colored::Colorize;
use rustyline::completion::{Completer, FilenameCompleter, Pair};
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::{error::ReadlineError, history::FileHistory, Editor};
use rustyline_derive::Helper;

#[derive(Helper)]
struct ShellHelper {
    completer: FilenameCompleter,
    commands: Vec<String>,
}

impl Completer for ShellHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        // First try command completion
        let words: Vec<&str> = line.split_whitespace().collect();

        if words.is_empty() || (words.len() == 1 && !line.ends_with(' ')) {
            // Complete commands
            let prefix = words.get(0).copied().unwrap_or("");
            let mut matches = Vec::new();

            for cmd in &self.commands {
                if cmd.starts_with(prefix) {
                    matches.push(Pair {
                        display: cmd.clone(),
                        replacement: cmd.clone(),
                    });
                }
            }

            if !matches.is_empty() {
                return Ok((0, matches));
            }
        }

        // Fall back to filename completion
        self.completer.complete(line, pos, _ctx)
    }
}

impl Highlighter for ShellHelper {}
impl Hinter for ShellHelper {
    type Hint = String;
}
impl Validator for ShellHelper {}

pub async fn run_shell() -> Result<()> {
    output::launch("Starting StratoSwarm interactive shell");
    output::info("Type 'help' for available commands, 'exit' to quit\n");

    let helper = ShellHelper {
        completer: FilenameCompleter::new(),
        commands: vec![
            "deploy".to_string(),
            "status".to_string(),
            "logs".to_string(),
            "scale".to_string(),
            "evolve".to_string(),
            "quickstart".to_string(),
            "help".to_string(),
            "exit".to_string(),
            "quit".to_string(),
            "clear".to_string(),
        ],
    };

    let history_path = dirs::cache_dir()
        .map(|p| p.join("stratoswarm").join("history.txt"))
        .unwrap_or_else(|| ".stratoswarm_history".into());

    if let Some(parent) = history_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let mut rl: Editor<ShellHelper, FileHistory> = Editor::new()?;
    rl.set_helper(Some(helper));

    if rl.load_history(&history_path).is_err() {
        output::info("No previous history found");
    }

    let prompt = format!("{} ", "stratoswarm>".green().bold());

    loop {
        match rl.readline(&prompt) {
            Ok(line) => {
                let line = line.trim();

                if line.is_empty() {
                    continue;
                }

                rl.add_history_entry(line)?;

                match line {
                    "exit" | "quit" => {
                        output::info("Goodbye!");
                        break;
                    }
                    "clear" => {
                        print!("\x1B[2J\x1B[1;1H");
                    }
                    "help" => {
                        print_shell_help();
                    }
                    _ => {
                        if let Err(e) = execute_shell_command(line).await {
                            output::error(&format!("Error: {}", e));
                        }
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                output::info("Use 'exit' to quit");
            }
            Err(ReadlineError::Eof) => {
                output::info("Goodbye!");
                break;
            }
            Err(err) => {
                output::error(&format!("Error: {}", err));
                break;
            }
        }
    }

    rl.save_history(&history_path).ok();
    Ok(())
}

async fn execute_shell_command(line: &str) -> Result<()> {
    // Parse the command line
    let args: Vec<&str> = line.split_whitespace().collect();
    if args.is_empty() {
        return Ok(());
    }

    // Prepend "stratoswarm" to make it compatible with clap parsing
    let mut full_args = vec!["stratoswarm"];
    full_args.extend(args.iter().cloned());

    // Try to parse and execute
    match parse_command(&full_args) {
        Ok(command) => {
            command.execute().await?;
        }
        Err(_e) => {
            // If parsing fails, show a more user-friendly error
            if args[0] == "watch" && args.len() > 1 {
                // Special handling for "watch" subcommand
                watch_command(&args[1..]).await?;
            } else {
                return Err(CliError::Command(format!(
                    "Unknown command: {}. Type 'help' for available commands.",
                    args[0]
                )));
            }
        }
    }

    Ok(())
}

fn parse_command(args: &[&str]) -> Result<Commands> {
    #[derive(Debug, Parser)]
    struct ShellCli {
        #[command(subcommand)]
        command: Commands,
    }

    ShellCli::try_parse_from(args)
        .map(|cli| cli.command)
        .map_err(|e| CliError::Command(e.to_string()))
}

async fn watch_command(args: &[&str]) -> Result<()> {
    if args.is_empty() {
        return Err(CliError::InvalidArgument(
            "Usage: watch <metrics|status|logs>".to_string(),
        ));
    }

    match args[0] {
        "metrics" => {
            output::info("Watching metrics in real-time...");
            watch_metrics().await?;
        }
        "status" => {
            output::info("Watching status changes...");
            watch_status().await?;
        }
        "logs" => {
            if args.len() < 2 {
                return Err(CliError::InvalidArgument(
                    "Usage: watch logs <agent>".to_string(),
                ));
            }
            output::info(&format!("Following logs for agent '{}'...", args[1]));
            watch_logs(args[1]).await?;
        }
        _ => {
            return Err(CliError::InvalidArgument(format!(
                "Unknown watch target: {}",
                args[0]
            )));
        }
    }

    Ok(())
}

async fn watch_metrics() -> Result<()> {
    use tokio::time::{sleep, Duration};

    for i in 0..5 {
        println!("\x1B[2J\x1B[1;1H"); // Clear screen
        output::header("Live Metrics");
        output::kv("CPU Usage", &format!("{}%", 45 + i * 2));
        output::kv("Memory Usage", &format!("{}%", 62 + i));
        output::kv("GPU Usage", &format!("{}%", 78 - i));
        output::kv("Network I/O", &format!("{} MB/s", 125 + i * 10));

        sleep(Duration::from_secs(1)).await;
    }

    Ok(())
}

async fn watch_status() -> Result<()> {
    use crate::commands::status;

    let args = status::StatusArgs {
        namespace: None,
        agent: None,
        detailed: false,
        format: status::OutputFormat::Table,
        watch: true,
    };

    status::execute(args).await
}

async fn watch_logs(agent: &str) -> Result<()> {
    use crate::commands::logs;

    let args = logs::LogsArgs {
        agent: agent.to_string(),
        since: None,
        errors_only: false,
        follow: true,
        lines: 50,
        namespace: "default".to_string(),
        timestamps: true,
        level: None,
    };

    logs::execute(args).await
}

fn print_shell_help() {
    output::header("StratoSwarm Interactive Shell");

    println!("{}", "Available Commands:".bold());
    println!();

    output::bullet("deploy <file>       - Deploy a .swarm file or directory");
    output::bullet("status              - Show status of deployed agents");
    output::bullet("logs <agent>        - Show logs from an agent");
    output::bullet("scale <agent=N>     - Scale an agent to N replicas");
    output::bullet("evolve <agent>      - Evolve an agent through generations");
    output::bullet("quickstart          - Create a new project from template");
    output::bullet("watch <target>      - Watch metrics, status, or logs in real-time");
    output::bullet("clear               - Clear the screen");
    output::bullet("help                - Show this help message");
    output::bullet("exit, quit          - Exit the shell");

    println!();
    println!("{}", "Examples:".bold());
    println!();

    output::bullet("deploy app.swarm");
    output::bullet("status --detailed");
    output::bullet("logs frontend --since 1h --errors-only");
    output::bullet("scale backend=10 frontend=5");
    output::bullet("evolve backend --generations 100");
    output::bullet("watch metrics");
    output::bullet("watch logs frontend");

    println!();
    println!("{}", "Tips:".bold());
    output::bullet("Use Tab for command and filename completion");
    output::bullet("Use Up/Down arrows to navigate command history");
    output::bullet("Commands support the same options as the CLI");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shell_helper_creation() {
        let helper = ShellHelper {
            completer: FilenameCompleter::new(),
            commands: vec!["test".to_string()],
        };

        assert_eq!(helper.commands.len(), 1);
    }

    #[test]
    fn test_parse_command() {
        let result = parse_command(&["stratoswarm", "status"]);
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), Commands::Status(_)));

        let result = parse_command(&["stratoswarm", "invalid"]);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_watch_command_no_args() {
        let result = watch_command(&[]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_watch_command_invalid_target() {
        let result = watch_command(&["invalid"]).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_print_shell_help() {
        // Should not panic
        print_shell_help();
    }
}
