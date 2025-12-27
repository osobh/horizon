//! Tests for code modification engine

use super::types::*;
use super::*;
use std::collections::HashMap;

type TestResult = Result<(), Box<dyn std::error::Error>>;

// Helper function to create test code analysis
fn create_test_analysis() -> CodeAnalysis {
    let mut existing_tools = HashMap::new();
    existing_tools.insert(
        "bash".to_string(),
        ToolInfo {
            name: "bash".to_string(),
            description: "Execute bash commands".to_string(),
            usage_frequency: 0.8,
            success_rate: 0.7,
        },
    );
    existing_tools.insert(
        "edit".to_string(),
        ToolInfo {
            name: "edit".to_string(),
            description: "Edit files".to_string(),
            usage_frequency: 0.9,
            success_rate: 0.6,
        },
    );

    CodeAnalysis {
        opportunities: vec![
            ImprovementOpportunity {
                improvement_type: ModificationType::ToolEnhancement,
                location: CodeLocation {
                    file: "tools/edit.py".to_string(),
                    start_line: 45,
                    end_line: 67,
                },
                confidence: 0.8,
                description: "Edit tool could support line-based editing".to_string(),
            },
            ImprovementOpportunity {
                improvement_type: ModificationType::WorkflowChange,
                location: CodeLocation {
                    file: "agent/workflow.py".to_string(),
                    start_line: 100,
                    end_line: 150,
                },
                confidence: 0.7,
                description: "Add retry logic for failed edits".to_string(),
            },
        ],
        existing_tools,
        workflow_patterns: vec![WorkflowPattern {
            name: "edit-test-cycle".to_string(),
            steps: vec!["edit".to_string(), "bash".to_string(), "edit".to_string()],
            frequency: 0.6,
        }],
        bottlenecks: vec![Bottleneck {
            bottleneck_type: "file_operations".to_string(),
            location: CodeLocation {
                file: "tools/edit.py".to_string(),
                start_line: 200,
                end_line: 250,
            },
            impact: 0.7,
        }],
    }
}

// Helper function to create test performance feedback
fn create_test_feedback() -> PerformanceFeedback {
    let mut error_patterns = HashMap::new();
    error_patterns.insert("file_not_found".to_string(), 15);
    error_patterns.insert("syntax_error".to_string(), 8);

    let mut tool_usage = HashMap::new();
    tool_usage.insert(
        "bash".to_string(),
        ToolUsageStats {
            usage_count: 100,
            success_rate: 0.7,
            avg_execution_time: 2.5,
        },
    );
    tool_usage.insert(
        "edit".to_string(),
        ToolUsageStats {
            usage_count: 150,
            success_rate: 0.6,
            avg_execution_time: 1.2,
        },
    );

    PerformanceFeedback {
        success_rate: 0.65,
        avg_completion_time: 45.3,
        error_patterns,
        tool_usage,
    }
}

#[test]
fn test_code_analyzer_analyze_codebase() -> TestResult {
    let analyzer = CodeAnalyzer::new();
    let code_path = "/test/agent/code";
    let feedback = create_test_feedback();

    let analysis = analyzer.analyze_codebase(code_path, &feedback)?;

    assert!(!analysis.opportunities.is_empty());
    assert!(!analysis.existing_tools.is_empty());
    assert!(!analysis.workflow_patterns.is_empty());
    Ok(())
}

#[test]
fn test_code_analyzer_identify_bottlenecks() {
    let analyzer = CodeAnalyzer::new();
    let feedback = create_test_feedback();

    let bottlenecks = analyzer.identify_bottlenecks(&feedback);

    assert!(!bottlenecks.is_empty());
    assert!(bottlenecks
        .iter()
        .any(|b| b.bottleneck_type == "file_operations"));
}

#[test]
fn test_code_modifier_propose_modifications() -> TestResult {
    let modifier = CodeModifier::new();
    let analysis = create_test_analysis();
    let feedback = create_test_feedback();

    let proposals = modifier.propose_modifications(&analysis, &feedback)?;

    assert!(!proposals.is_empty());
    assert!(proposals
        .iter()
        .any(|p| matches!(p.modification_type, ModificationType::ToolEnhancement)));
    assert!(proposals
        .iter()
        .all(|p| p.priority >= 0.0 && p.priority <= 1.0));
    Ok(())
}

#[test]
fn test_code_modifier_apply_modification() {
    let modifier = CodeModifier::new();
    let proposal = ModificationProposal {
        id: "mod_001".to_string(),
        modification_type: ModificationType::ToolEnhancement,
        rationale: "Improve edit tool with line-based editing".to_string(),
        expected_impact: "Reduce file operation errors by 30%".to_string(),
        priority: 0.8,
        target_metrics: vec!["file_operation_success".to_string()],
    };

    let code_content = r#"
class EditTool:
    def edit_file(self, path, content):
        with open(path, 'w') as f:
            f.write(content)
"#;

    let result = modifier
        .apply_modification(&proposal, code_content)
        .unwrap();

    assert!(result.success);
    assert!(result.modified_code.is_some());
    assert!(result.modified_code.unwrap().contains("edit_lines"));
}

#[test]
fn test_code_modifier_generate_tool_enhancement() {
    let modifier = CodeModifier::new();
    let tool_info = ToolInfo {
        name: "edit".to_string(),
        description: "Edit files".to_string(),
        usage_frequency: 0.9,
        success_rate: 0.6,
    };

    let enhancement = modifier
        .generate_tool_enhancement(&tool_info, &["line_editing"])
        .unwrap();

    assert!(enhancement.contains("def"));
    assert!(enhancement.contains("lines"));
}

#[test]
fn test_modification_validator_validate_syntax() {
    let validator = ModificationValidator::new();

    let valid_code = r#"
def test_function():
    return True
"#;

    let invalid_code = r#"
def test_function()
    return True
"#;

    assert!(validator.validate_syntax(valid_code, "python").unwrap());
    assert!(!validator.validate_syntax(invalid_code, "python").unwrap());
}

#[test]
fn test_modification_validator_validate_safety() {
    let validator = ModificationValidator::new();

    let safe_code = r#"
def process_data(data):
    return data.upper()
"#;

    let unsafe_code = r#"
import os
os.system("rm -rf /")
"#;

    assert!(validator.validate_safety(safe_code).unwrap());
    assert!(!validator.validate_safety(unsafe_code).unwrap());
}

#[test]
fn test_modification_validator_check_compatibility() {
    let validator = ModificationValidator::new();
    let modification = CodeModification {
        modification_type: ModificationType::ToolEnhancement,
        target_file: "tools/edit.py".to_string(),
        description: "Add line editing".to_string(),
        diff: "+ def edit_lines(...)".to_string(),
        affected_lines: vec![100, 101, 102],
        dependencies: vec!["re".to_string()],
    };

    let existing_code = r#"
import os
import json

class EditTool:
    pass
"#;

    assert!(validator
        .check_compatibility(&modification, existing_code)
        .unwrap());
}

#[test]
fn test_code_analyzer_detect_workflow_patterns() {
    let analyzer = CodeAnalyzer::new();
    let execution_logs = vec![
        vec!["edit", "bash", "edit"],
        vec!["edit", "bash", "bash", "edit"],
        vec!["edit", "bash", "edit"],
        vec!["bash", "edit"],
    ];

    let patterns = analyzer.detect_workflow_patterns(&execution_logs);

    assert!(!patterns.is_empty());
    assert!(patterns
        .iter()
        .any(|p| p.name.contains("edit") && p.name.contains("bash")));
}

#[test]
fn test_code_modifier_prioritize_modifications() {
    let modifier = CodeModifier::new();
    let proposals = vec![
        ModificationProposal {
            id: "1".to_string(),
            modification_type: ModificationType::ToolEnhancement,
            rationale: "Low impact".to_string(),
            expected_impact: "Minor improvement".to_string(),
            priority: 0.3,
            target_metrics: vec![],
        },
        ModificationProposal {
            id: "2".to_string(),
            modification_type: ModificationType::PerformanceOptimization,
            rationale: "High impact".to_string(),
            expected_impact: "Major improvement".to_string(),
            priority: 0.9,
            target_metrics: vec![],
        },
        ModificationProposal {
            id: "3".to_string(),
            modification_type: ModificationType::WorkflowChange,
            rationale: "Medium impact".to_string(),
            expected_impact: "Moderate improvement".to_string(),
            priority: 0.6,
            target_metrics: vec![],
        },
    ];

    let prioritized = modifier.prioritize_modifications(proposals);

    assert_eq!(prioritized[0].id, "2");
    assert_eq!(prioritized[1].id, "3");
    assert_eq!(prioritized[2].id, "1");
}

#[test]
fn test_end_to_end_modification_workflow() -> TestResult {
    // Test complete workflow from analysis to modification
    let analyzer = CodeAnalyzer::new();
    let modifier = CodeModifier::new();
    let validator = ModificationValidator::new();

    // Analyze
    let feedback = create_test_feedback();
    let analysis = analyzer.analyze_codebase("/test/code", &feedback)?;

    // Propose
    let proposals = modifier.propose_modifications(&analysis, &feedback)?;
    assert!(!proposals.is_empty());

    // Select highest priority
    let selected = proposals
        .into_iter()
        .max_by(|a, b| a.priority.partial_cmp(&b.priority).unwrap())
        .unwrap();

    // Apply
    let test_code = r#"
class EditTool:
    def edit_file(self, path, content):
        with open(path, 'w') as f:
            f.write(content)
"#;

    let result = modifier.apply_modification(&selected, test_code)?;
    assert!(result.success);

    // Validate
    let modified_code = result.modified_code.unwrap();
    assert!(validator.validate_syntax(&modified_code, "python")?);
    assert!(validator.validate_safety(&modified_code)?);
    Ok(())
}

#[test]
fn test_modification_types_coverage() {
    // Ensure all modification types can be handled
    let modifier = CodeModifier::new();
    let types = vec![
        ModificationType::ToolAddition,
        ModificationType::ToolEnhancement,
        ModificationType::WorkflowChange,
        ModificationType::ErrorHandling,
        ModificationType::PerformanceOptimization,
        ModificationType::CapabilityExtension,
        ModificationType::Refactoring,
    ];

    for mod_type in types {
        let proposal = ModificationProposal {
            id: format!("{:?}", mod_type),
            modification_type: mod_type.clone(),
            rationale: "Test".to_string(),
            expected_impact: "Test".to_string(),
            priority: 0.5,
            target_metrics: vec![],
        };

        let result = modifier
            .apply_modification(&proposal, "# Test code")
            .unwrap();
        assert!(result.modified_code.is_some());
    }
}
