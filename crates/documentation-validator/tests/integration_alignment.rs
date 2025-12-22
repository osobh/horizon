//! Integration tests for documentation alignment functionality
//!
//! These tests verify that the documentation validator can properly
//! align performance claims with validated metrics and update files
//! with appropriate status indicators.

use anyhow::Result;
use std::fs;
use stratoswarm_documentation_validator::{
    alignment::DocumentationAligner, audit::DocumentationAuditor,
    status_indicators::StatusIndicators, validation::ClaimValidator, ValidationConfig,
};
use tempfile::TempDir;

/// Create a test markdown file with various performance claims
fn create_test_markdown() -> &'static str {
    r#"# StratoSwarm Performance Documentation

## Scale Capabilities

- **Consensus Latency**: 73.89μs achieved in production
- Target latency: <100μs for global agreement
- Current GPU utilization: 73.7% average
- Goal: Achieve 85%+ GPU utilization  
- Scale: 1000+ nodes supported
- Agent capacity: 10M+ agents capability

## Memory Performance

- Memory migration: <50ms achieved
- Target: <10ms migration latency
- Current throughput: 79,998 messages/sec

## Future Goals

- Planned feature: Real-time ML inference at edge
- Target: 1μs ultra-low latency consensus
"#
}

#[tokio::test]
async fn test_documentation_alignment_workflow() -> Result<()> {
    // Arrange
    let temp_dir = TempDir::new()?;
    let test_file = temp_dir.path().join("performance.md");
    fs::write(&test_file, create_test_markdown())?;

    let config = ValidationConfig {
        base_dir: temp_dir.path().to_path_buf(),
        include_patterns: vec!["*.md".to_string()],
        exclude_patterns: vec![],
        validated_metrics: Default::default(),
    };

    // Act - Audit
    let auditor = DocumentationAuditor::new(config.clone());
    let audit_results = auditor.audit_documentation()?;

    assert!(!audit_results.claims.is_empty());
    assert_eq!(audit_results.files_scanned, 1);

    // Act - Validate
    let validator = ClaimValidator::default();
    let mut claims = audit_results.claims;
    validator.validate_claims(&mut claims)?;

    // Check validation results
    let validated_count = claims
        .iter()
        .filter(|c| {
            matches!(
                c.status,
                stratoswarm_documentation_validator::ClaimStatus::Validated
            )
        })
        .count();
    let planned_count = claims
        .iter()
        .filter(|c| {
            matches!(
                c.status,
                stratoswarm_documentation_validator::ClaimStatus::Planned
            )
        })
        .count();

    assert!(validated_count > 0, "Should have validated claims");
    assert!(planned_count > 0, "Should have planned claims");

    // Act - Align
    let mut aligner = DocumentationAligner::new();
    aligner.align_claims(&claims)?;

    // Preview changes before applying
    let preview = aligner.preview_changes()?;
    assert!(preview.contains("✅"));
    assert!(preview.contains("📋"));

    // Apply changes
    let updated_files = aligner.apply_replacements()?;
    assert_eq!(updated_files.len(), 1);

    // Assert - Verify file was updated
    let updated_content = fs::read_to_string(&test_file)?;

    // Check that status indicators were added
    assert!(
        updated_content.contains("✅"),
        "Should contain validated indicator"
    );
    assert!(
        updated_content.contains("📋"),
        "Should contain planned indicator"
    );
    assert!(updated_content.contains("(validated: scale_test_results.json)"));

    // Verify specific claims were updated correctly
    assert!(updated_content.contains("✅ - **Consensus Latency**: 73.89μs"));
    assert!(updated_content.contains("✅ - Scale: 1000+ nodes"));
    assert!(updated_content.contains("✅ - Agent capacity: 10M+ agents"));
    assert!(updated_content.contains("📋 - Target: 1μs ultra-low latency"));

    Ok(())
}

#[tokio::test]
async fn test_validation_report_generation() -> Result<()> {
    // Arrange
    let temp_dir = TempDir::new()?;
    let test_file = temp_dir.path().join("test.md");
    fs::write(&test_file, create_test_markdown())?;

    let config = ValidationConfig {
        base_dir: temp_dir.path().to_path_buf(),
        include_patterns: vec!["*.md".to_string()],
        exclude_patterns: vec![],
        validated_metrics: Default::default(),
    };

    // Act
    let auditor = DocumentationAuditor::new(config);
    let audit_results = auditor.audit_documentation()?;

    let validator = ClaimValidator::default();
    let mut claims = audit_results.claims;
    validator.validate_claims(&mut claims)?;

    let report = StatusIndicators::generate_validation_report(&claims);

    // Assert
    assert!(report.contains("# Documentation Validation Report"));
    assert!(report.contains("## Status Legend"));
    assert!(report.contains("✅ Validated"));
    assert!(report.contains("📋 Planned"));

    // Check that specific validated claims appear in report
    assert!(report.contains("73.89μs"));
    assert!(report.contains("Evidence: scale_test_results.json"));

    Ok(())
}

#[tokio::test]
async fn test_performance_documentation_generation() -> Result<()> {
    // Arrange
    let temp_dir = TempDir::new()?;
    let test_file = temp_dir.path().join("metrics.md");
    fs::write(&test_file, create_test_markdown())?;

    let config = ValidationConfig {
        base_dir: temp_dir.path().to_path_buf(),
        include_patterns: vec!["*.md".to_string()],
        exclude_patterns: vec![],
        validated_metrics: Default::default(),
    };

    // Act
    let auditor = DocumentationAuditor::new(config);
    let audit_results = auditor.audit_documentation()?;

    let validator = ClaimValidator::default();
    let mut claims = audit_results.claims;
    validator.validate_claims(&mut claims)?;

    let aligner = DocumentationAligner::new();
    let perf_doc = aligner.create_performance_documentation(&claims)?;

    // Assert
    assert!(perf_doc.contains("# StratoSwarm Performance Documentation"));
    assert!(perf_doc.contains("## Validated Performance Metrics"));
    assert!(perf_doc.contains("### ✅ Validated Performance"));
    assert!(perf_doc.contains("### 📋 Planned Features"));

    // Check specific metrics appear
    assert!(perf_doc.contains("Consensus Latency: 73.89μs"));
    assert!(perf_doc.contains("Evidence: scale_test_results.json"));
    assert!(perf_doc.contains("1000+ nodes"));
    assert!(perf_doc.contains("10M+ agents"));

    Ok(())
}

#[tokio::test]
async fn test_multi_file_alignment() -> Result<()> {
    // Arrange - Create multiple markdown files
    let temp_dir = TempDir::new()?;

    let files = vec![
        (
            "readme.md",
            "# Project\n\nPerformance: 73.89μs consensus achieved",
        ),
        (
            "docs/perf.md",
            "## Metrics\n\n- GPU: 73.7% utilization\n- Target: 85%+ GPU usage",
        ),
        (
            "next_steps.md",
            "# Next Steps\n\nGoal: Achieve 1μs latency in future",
        ),
    ];

    for (name, content) in &files {
        let path = temp_dir.path().join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, content)?;
    }

    let config = ValidationConfig {
        base_dir: temp_dir.path().to_path_buf(),
        include_patterns: vec!["*.md".to_string(), "docs/*.md".to_string()],
        exclude_patterns: vec![],
        validated_metrics: Default::default(),
    };

    // Act
    let auditor = DocumentationAuditor::new(config);
    let audit_results = auditor.audit_documentation()?;

    assert_eq!(audit_results.files_scanned, 3);

    let validator = ClaimValidator::default();
    let mut claims = audit_results.claims;
    validator.validate_claims(&mut claims)?;

    let mut aligner = DocumentationAligner::new();
    aligner.align_claims(&claims)?;
    let updated_files = aligner.apply_replacements()?;

    // Assert
    assert!(updated_files.len() >= 2, "Should update multiple files");

    // Check each file was updated correctly
    let readme_content = fs::read_to_string(temp_dir.path().join("readme.md"))?;
    assert!(readme_content.contains("✅"));

    let perf_content = fs::read_to_string(temp_dir.path().join("docs/perf.md"))?;
    assert!(
        perf_content.contains("📋"),
        "Target claims should be marked as planned"
    );

    let next_steps_content = fs::read_to_string(temp_dir.path().join("next_steps.md"))?;
    assert!(next_steps_content.contains("📋"));

    Ok(())
}
