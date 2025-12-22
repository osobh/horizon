//! Documentation validation binary
//! Validates all crates in the workspace for documentation accuracy and file size compliance

use std::path::PathBuf;
use stratoswarm_zero_config::crate_validator::WorkspaceValidator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get workspace root (assuming we're in crates/zero-config)
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .ok_or("Failed to find workspace root")?
        .to_path_buf();

    println!("Validating StratoSwarm workspace at: {:?}", workspace_root);
    println!("This may take a moment...\n");

    let validator = WorkspaceValidator::new(workspace_root.clone());
    let summary = validator.validate_all_crates()?;

    // Print summary
    println!("=== Validation Summary ===");
    println!("Total Crates: {}", summary.total_crates);
    println!("Crates with Issues: {}", summary.crates_with_issues);
    println!("Files Over 750 Lines: {}", summary.total_files_over_limit);
    println!(
        "Average Accuracy Score: {:.1}%",
        summary.average_accuracy_score
    );

    // Print critical issues (files over 750 lines)
    if !summary.critical_issues.is_empty() {
        println!("\n=== CRITICAL: Files Over 750 Lines ===");
        println!("These files must be split into smaller modules:\n");

        for (i, issue) in summary.critical_issues.iter().take(50).enumerate() {
            println!(
                "{}. {} - {} ({} lines)",
                i + 1,
                issue.crate_name,
                issue
                    .file_path
                    .strip_prefix(&workspace_root)
                    .unwrap_or(&issue.file_path)
                    .display(),
                issue.line_count.unwrap_or(0)
            );
        }

        if summary.critical_issues.len() > 50 {
            println!(
                "\n... and {} more files",
                summary.critical_issues.len() - 50
            );
        }
    }

    // Generate markdown report
    let report = validator.generate_markdown_report(&summary);
    let report_path = workspace_root.join("documentation-validation-report.md");
    std::fs::write(&report_path, report)?;
    println!("\n✓ Full report written to: {}", report_path.display());

    // Exit with error if there are critical issues
    if summary.total_files_over_limit > 0 {
        eprintln!(
            "\n❌ Validation failed: {} files exceed 750 line limit",
            summary.total_files_over_limit
        );
        std::process::exit(1);
    }

    println!("\n✓ All files are within size limits!");
    Ok(())
}
