//! Logs command implementation

use crate::{output, Result};
use chrono::{DateTime, Duration, Utc};
use clap::Args;
use colored::Colorize;

#[derive(Debug, Clone, Args)]
pub struct LogsArgs {
    /// Agent name to show logs for
    pub agent: String,

    /// Show logs since duration (e.g., 1h, 30m, 5s)
    #[arg(short, long)]
    pub since: Option<humantime::Duration>,

    /// Show only error logs
    #[arg(long)]
    pub errors_only: bool,

    /// Follow log output
    #[arg(short, long)]
    pub follow: bool,

    /// Number of lines to show
    #[arg(short = 'n', long, default_value = "100")]
    pub lines: usize,

    /// Namespace
    #[arg(long, default_value = "default")]
    pub namespace: String,

    /// Show timestamps
    #[arg(short = 't', long)]
    pub timestamps: bool,

    /// Filter by log level (debug, info, warn, error)
    #[arg(long)]
    pub level: Option<LogLevel>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

impl std::str::FromStr for LogLevel {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "debug" => Ok(LogLevel::Debug),
            "info" => Ok(LogLevel::Info),
            "warn" | "warning" => Ok(LogLevel::Warn),
            "error" => Ok(LogLevel::Error),
            _ => Err(format!("Unknown log level: {}", s)),
        }
    }
}

#[derive(Debug)]
pub struct LogEntry {
    pub timestamp: DateTime<Utc>,
    pub level: LogLevel,
    pub agent: String,
    pub message: String,
}

pub async fn execute(args: LogsArgs) -> Result<()> {
    output::info(&format!("Fetching logs for agent '{}'", args.agent));

    let logs = if args.follow {
        follow_logs(&args).await?
    } else {
        fetch_logs(&args).await?
    };

    print_logs(&logs, &args);

    Ok(())
}

async fn fetch_logs(args: &LogsArgs) -> Result<Vec<LogEntry>> {
    // Mock implementation
    let mut logs = vec![
        LogEntry {
            timestamp: Utc::now() - Duration::minutes(5),
            level: LogLevel::Info,
            agent: args.agent.clone(),
            message: "Agent started successfully".to_string(),
        },
        LogEntry {
            timestamp: Utc::now() - Duration::minutes(4),
            level: LogLevel::Debug,
            agent: args.agent.clone(),
            message: "Connected to cluster mesh".to_string(),
        },
        LogEntry {
            timestamp: Utc::now() - Duration::minutes(3),
            level: LogLevel::Info,
            agent: args.agent.clone(),
            message: "Processing request from client".to_string(),
        },
        LogEntry {
            timestamp: Utc::now() - Duration::minutes(2),
            level: LogLevel::Warn,
            agent: args.agent.clone(),
            message: "High memory usage detected (85%)".to_string(),
        },
        LogEntry {
            timestamp: Utc::now() - Duration::minutes(1),
            level: LogLevel::Error,
            agent: args.agent.clone(),
            message: "Failed to connect to database: connection timeout".to_string(),
        },
    ];

    // Apply filters
    if args.errors_only {
        logs.retain(|log| log.level == LogLevel::Error);
    }

    if let Some(level) = &args.level {
        logs.retain(|log| match level {
            LogLevel::Debug => true,
            LogLevel::Info => !matches!(log.level, LogLevel::Debug),
            LogLevel::Warn => matches!(log.level, LogLevel::Warn | LogLevel::Error),
            LogLevel::Error => log.level == LogLevel::Error,
        });
    }

    if let Some(since) = &args.since {
        let std_duration = (*since).into();
        let cutoff = Utc::now() - Duration::from_std(std_duration).unwrap();
        logs.retain(|log| log.timestamp > cutoff);
    }

    // Limit number of lines
    let total_logs = logs.len();
    if total_logs > args.lines {
        logs = logs.into_iter().skip(total_logs - args.lines).collect();
    }

    Ok(logs)
}

async fn follow_logs(args: &LogsArgs) -> Result<Vec<LogEntry>> {
    use tokio::time::{sleep, Duration};

    output::info("Following logs... (press Ctrl+C to stop)");

    let mut all_logs = fetch_logs(args).await?;
    print_logs(&all_logs, args);

    // Simulate following logs
    for i in 0..5 {
        sleep(Duration::from_secs(2)).await;

        let new_log = LogEntry {
            timestamp: Utc::now(),
            level: match i % 4 {
                0 => LogLevel::Info,
                1 => LogLevel::Debug,
                2 => LogLevel::Warn,
                _ => LogLevel::Error,
            },
            agent: args.agent.clone(),
            message: format!("Live log entry #{}", i + 1),
        };

        print_log_entry(&new_log, args);
        all_logs.push(new_log);
    }

    Ok(all_logs)
}

fn print_logs(logs: &[LogEntry], args: &LogsArgs) {
    for log in logs {
        print_log_entry(log, args);
    }
}

fn print_log_entry(log: &LogEntry, args: &LogsArgs) {
    let level_str = match log.level {
        LogLevel::Debug => "DEBUG".bright_black(),
        LogLevel::Info => "INFO ".green(),
        LogLevel::Warn => "WARN ".yellow(),
        LogLevel::Error => "ERROR".red(),
    };

    if args.timestamps {
        let timestamp = log.timestamp.format("%Y-%m-%d %H:%M:%S%.3f");
        println!(
            "[{}] {} [{}] {}",
            timestamp, level_str, log.agent, log.message
        );
    } else {
        println!("{} [{}] {}", level_str, log.agent, log.message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_parsing() {
        assert_eq!("debug".parse::<LogLevel>().unwrap(), LogLevel::Debug);
        assert_eq!("info".parse::<LogLevel>().unwrap(), LogLevel::Info);
        assert_eq!("warn".parse::<LogLevel>().unwrap(), LogLevel::Warn);
        assert_eq!("warning".parse::<LogLevel>().unwrap(), LogLevel::Warn);
        assert_eq!("error".parse::<LogLevel>().unwrap(), LogLevel::Error);
        assert!("invalid".parse::<LogLevel>().is_err());
    }

    #[tokio::test]
    async fn test_fetch_logs_with_filters() {
        let args = LogsArgs {
            agent: "test".to_string(),
            since: Some(humantime::Duration::from(std::time::Duration::from_secs(
                180,
            ))),
            errors_only: true,
            follow: false,
            lines: 10,
            namespace: "default".to_string(),
            timestamps: false,
            level: None,
        };

        let logs = fetch_logs(&args).await.unwrap();
        assert!(logs.iter().all(|log| log.level == LogLevel::Error));
    }

    #[tokio::test]
    async fn test_fetch_logs_with_level_filter() {
        let args = LogsArgs {
            agent: "test".to_string(),
            since: None,
            errors_only: false,
            follow: false,
            lines: 100,
            namespace: "default".to_string(),
            timestamps: false,
            level: Some(LogLevel::Warn),
        };

        let logs = fetch_logs(&args).await.unwrap();
        assert!(logs
            .iter()
            .all(|log| matches!(log.level, LogLevel::Warn | LogLevel::Error)));
    }

    #[test]
    fn test_print_log_entry() {
        let log = LogEntry {
            timestamp: Utc::now(),
            level: LogLevel::Info,
            agent: "test".to_string(),
            message: "Test message".to_string(),
        };

        let args = LogsArgs {
            agent: "test".to_string(),
            since: None,
            errors_only: false,
            follow: false,
            lines: 10,
            namespace: "default".to_string(),
            timestamps: true,
            level: None,
        };

        // Should not panic
        print_log_entry(&log, &args);
    }
}
