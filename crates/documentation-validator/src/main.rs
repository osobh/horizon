//! Documentation Reality Alignment CLI
//!
//! Command-line interface for the documentation validator tool.
//! Provides comprehensive documentation audit and alignment capabilities.

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use stratoswarm_documentation_validator::{
    alignment::DocumentationAligner, audit::DocumentationAuditor,
    status_indicators::StatusIndicators, validation::ClaimValidator, ValidationConfig,
};

#[derive(Parser)]
#[command(name = "doc-validator")]
#[command(about = "StratoSwarm Documentation Reality Alignment Tool")]
#[command(long_about = "Audit and align documentation claims with validated performance metrics")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Audit documentation for performance claims
    Audit {
        /// Base directory to search for documentation
        #[arg(short, long, default_value = ".")]
        base_dir: PathBuf,
        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: String,
    },
    /// Align documentation with validated metrics
    Align {
        /// Base directory to search for documentation
        #[arg(short, long, default_value = ".")]
        base_dir: PathBuf,
        /// Dry run - show changes without applying them
        #[arg(long)]
        dry_run: bool,
    },
    /// Generate performance documentation
    Generate {
        /// Base directory to search for documentation
        #[arg(short, long, default_value = ".")]
        base_dir: PathBuf,
        /// Output file for generated documentation
        #[arg(short, long, default_value = "PERFORMANCE.md")]
        output: PathBuf,
    },
    /// Generate validation report
    Report {
        /// Base directory to search for documentation
        #[arg(short, long, default_value = ".")]
        base_dir: PathBuf,
        /// Output file for the report
        #[arg(short, long, default_value = "validation_report.md")]
        output: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Audit { base_dir, format } => audit_documentation(&base_dir, &format).await,
        Commands::Align { base_dir, dry_run } => align_documentation(&base_dir, dry_run).await,
        Commands::Generate { base_dir, output } => {
            generate_performance_docs(&base_dir, &output).await
        }
        Commands::Report { base_dir, output } => {
            generate_validation_report(&base_dir, &output).await
        }
    }
}

async fn audit_documentation(base_dir: &Path, format: &str) -> Result<()> {
    println!("🔍 Auditing documentation in: {}", base_dir.display());

    let config = ValidationConfig {
        base_dir: base_dir.to_path_buf(),
        ..Default::default()
    };

    let auditor = DocumentationAuditor::new(config);
    let results = auditor.audit_documentation()?;

    match format {
        "json" => {
            let json = serde_json::to_string_pretty(&results.claims)?;
            println!("{json}");
        }
        _ => {
            println!("{}", results.summary());

            if !results.claims.is_empty() {
                println!("\n📊 Claims by Status:");
                for claim in &results.claims {
                    println!(
                        "  {} {} ({}:{})",
                        claim.status.indicator(),
                        claim.text,
                        claim.file_path.display(),
                        claim.line_number
                    );
                }
            }
        }
    }

    Ok(())
}

async fn align_documentation(base_dir: &Path, dry_run: bool) -> Result<()> {
    println!("🔧 Aligning documentation in: {}", base_dir.display());

    let config = ValidationConfig {
        base_dir: base_dir.to_path_buf(),
        ..Default::default()
    };

    // First, audit to find claims
    let auditor = DocumentationAuditor::new(config.clone());
    let audit_results = auditor.audit_documentation()?;

    // Validate claims against known metrics
    let validator = ClaimValidator::default();
    let mut claims = audit_results.claims;
    validator.validate_claims(&mut claims)?;

    // Align documentation
    let mut aligner = DocumentationAligner::new();
    aligner.align_claims(&claims)?;

    if dry_run {
        println!("🔍 Preview of changes:");
        println!("{}", aligner.preview_changes()?);
    } else {
        let updated_files = aligner.apply_replacements()?;

        println!("✅ Updated {} files:", updated_files.len());
        for file in updated_files {
            println!("  📝 {}", file.display());
        }

        println!("\n📊 Final Status Summary:");
        println!("{}", StatusIndicators::generate_status_summary(&claims));
    }

    Ok(())
}

async fn generate_performance_docs(base_dir: &Path, output: &PathBuf) -> Result<()> {
    println!(
        "📋 Generating performance documentation for: {}",
        base_dir.display()
    );

    let config = ValidationConfig {
        base_dir: base_dir.to_path_buf(),
        ..Default::default()
    };

    // Audit and validate
    let auditor = DocumentationAuditor::new(config);
    let audit_results = auditor.audit_documentation()?;

    let validator = ClaimValidator::default();
    let mut claims = audit_results.claims;
    validator.validate_claims(&mut claims)?;

    // Generate documentation
    let aligner = DocumentationAligner::new();
    let performance_doc = aligner.create_performance_documentation(&claims)?;

    std::fs::write(output, performance_doc)?;
    println!(
        "✅ Performance documentation written to: {}",
        output.display()
    );

    Ok(())
}

async fn generate_validation_report(base_dir: &Path, output: &PathBuf) -> Result<()> {
    println!(
        "📊 Generating validation report for: {}",
        base_dir.display()
    );

    let config = ValidationConfig {
        base_dir: base_dir.to_path_buf(),
        ..Default::default()
    };

    // Audit and validate
    let auditor = DocumentationAuditor::new(config);
    let audit_results = auditor.audit_documentation()?;

    let validator = ClaimValidator::default();
    let mut claims = audit_results.claims;
    validator.validate_claims(&mut claims)?;

    // Generate report
    let report = StatusIndicators::generate_validation_report(&claims);

    std::fs::write(output, report)?;
    println!("✅ Validation report written to: {}", output.display());

    Ok(())
}
