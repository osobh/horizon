//! Code analysis functionality for identifying improvement opportunities

use super::types::*;
use crate::error::{EvolutionEngineError, EvolutionEngineResult};
use std::collections::HashMap;

/// Analyzes code to identify improvement opportunities
pub struct CodeAnalyzer {
    // Placeholder for future fields
}

impl CodeAnalyzer {
    /// Create new code analyzer
    pub fn new() -> Self {
        Self {}
    }

    /// Analyze codebase to identify improvement opportunities
    pub fn analyze_codebase(
        &self,
        code_path: &str,
        feedback: &PerformanceFeedback,
    ) -> EvolutionEngineResult<CodeAnalysis> {
        // Extract existing tools from simulated codebase
        let mut existing_tools = HashMap::new();

        // Simulate finding bash and edit tools
        existing_tools.insert(
            "bash".to_string(),
            ToolInfo {
                name: "bash".to_string(),
                description: "Execute bash commands".to_string(),
                usage_frequency: feedback
                    .tool_usage
                    .get("bash")
                    .map(|s| s.usage_count as f64 / 100.0)
                    .unwrap_or(0.0),
                success_rate: feedback
                    .tool_usage
                    .get("bash")
                    .map(|s| s.success_rate)
                    .unwrap_or(0.0),
            },
        );

        existing_tools.insert(
            "edit".to_string(),
            ToolInfo {
                name: "edit".to_string(),
                description: "Edit files".to_string(),
                usage_frequency: feedback
                    .tool_usage
                    .get("edit")
                    .map(|s| s.usage_count as f64 / 100.0)
                    .unwrap_or(0.0),
                success_rate: feedback
                    .tool_usage
                    .get("edit")
                    .map(|s| s.success_rate)
                    .unwrap_or(0.0),
            },
        );

        // Identify improvement opportunities based on feedback
        let mut opportunities = Vec::new();

        // If edit tool has low success rate, suggest enhancement
        if let Some(edit_stats) = feedback.tool_usage.get("edit") {
            if edit_stats.success_rate < 0.7 {
                opportunities.push(ImprovementOpportunity {
                    improvement_type: ModificationType::ToolEnhancement,
                    location: CodeLocation {
                        file: format!("{}/tools/edit.py", code_path),
                        start_line: 45,
                        end_line: 67,
                    },
                    confidence: 0.8,
                    description: "Edit tool could support line-based editing".to_string(),
                });
            }
        }

        // If error rate is high, suggest workflow improvements
        if feedback.error_patterns.values().sum::<usize>() > 10 {
            opportunities.push(ImprovementOpportunity {
                improvement_type: ModificationType::WorkflowChange,
                location: CodeLocation {
                    file: format!("{}/agent/workflow.py", code_path),
                    start_line: 100,
                    end_line: 150,
                },
                confidence: 0.7,
                description: "Add retry logic for failed operations".to_string(),
            });
        }

        // Detect workflow patterns
        let workflow_patterns = vec![WorkflowPattern {
            name: "edit-test-cycle".to_string(),
            steps: vec!["edit".to_string(), "bash".to_string(), "edit".to_string()],
            frequency: 0.6,
        }];

        // Identify bottlenecks
        let bottlenecks = self.identify_bottlenecks(feedback);

        Ok(CodeAnalysis {
            opportunities,
            existing_tools,
            workflow_patterns,
            bottlenecks,
        })
    }

    /// Identify performance bottlenecks from feedback
    pub fn identify_bottlenecks(&self, feedback: &PerformanceFeedback) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();

        // Check for file operation bottlenecks
        if let Some(file_errors) = feedback.error_patterns.get("file_not_found") {
            if *file_errors > 10 {
                bottlenecks.push(Bottleneck {
                    bottleneck_type: "file_operations".to_string(),
                    location: CodeLocation {
                        file: "tools/edit.py".to_string(),
                        start_line: 200,
                        end_line: 250,
                    },
                    impact: 0.7,
                });
            }
        }

        // Check for slow tools
        for (tool_name, stats) in &feedback.tool_usage {
            if stats.avg_execution_time > 5.0 {
                bottlenecks.push(Bottleneck {
                    bottleneck_type: "slow_tool".to_string(),
                    location: CodeLocation {
                        file: format!("tools/{}.py", tool_name),
                        start_line: 1,
                        end_line: 100,
                    },
                    impact: stats.avg_execution_time / 10.0,
                });
            }
        }

        bottlenecks
    }

    /// Detect workflow patterns from execution logs
    pub fn detect_workflow_patterns(&self, execution_logs: &[Vec<&str>]) -> Vec<WorkflowPattern> {
        let mut pattern_counts: HashMap<String, usize> = HashMap::new();

        // Count patterns of length 2-4
        for log in execution_logs {
            for window_size in 2..=4.min(log.len()) {
                for window in log.windows(window_size) {
                    let pattern_key = window.join("-");
                    *pattern_counts.entry(pattern_key).or_insert(0) += 1;
                }
            }
        }

        // Convert to WorkflowPattern
        let mut patterns: Vec<WorkflowPattern> = pattern_counts
            .into_iter()
            .filter(|(_, count)| *count >= 2)
            .map(|(pattern, count)| {
                let steps: Vec<String> = pattern.split('-').map(|s| s.to_string()).collect();
                WorkflowPattern {
                    name: pattern.clone(),
                    steps,
                    frequency: count as f64 / execution_logs.len() as f64,
                }
            })
            .collect();

        patterns.sort_by(|a, b| b.frequency.partial_cmp(&a.frequency).unwrap());
        patterns.truncate(5); // Keep top 5 patterns

        patterns
    }
}
