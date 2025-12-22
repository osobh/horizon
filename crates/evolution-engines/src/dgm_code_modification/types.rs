//! Type definitions for code modification system

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Types of code modifications that can be applied
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModificationType {
    /// Add new tool functionality
    ToolAddition,
    /// Enhance existing tool
    ToolEnhancement,
    /// Modify agent workflow
    WorkflowChange,
    /// Add error handling
    ErrorHandling,
    /// Performance optimization
    PerformanceOptimization,
    /// Add new capability
    CapabilityExtension,
    /// Refactor existing code
    Refactoring,
}

/// Represents a code modification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeModification {
    /// Type of modification
    pub modification_type: ModificationType,
    /// Target file path
    pub target_file: String,
    /// Description of the change
    pub description: String,
    /// The actual code diff
    pub diff: String,
    /// Line numbers affected
    pub affected_lines: Vec<usize>,
    /// Dependencies this modification requires
    pub dependencies: Vec<String>,
}

/// Proposal for a code modification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModificationProposal {
    /// Unique ID for the proposal
    pub id: String,
    /// Type of modification proposed
    pub modification_type: ModificationType,
    /// Rationale for the modification
    pub rationale: String,
    /// Expected impact
    pub expected_impact: String,
    /// Priority score (0.0 to 1.0)
    pub priority: f64,
    /// Performance metrics this targets
    pub target_metrics: Vec<String>,
}

/// Result of applying a modification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModificationResult {
    /// Whether the modification was successful
    pub success: bool,
    /// Modified code content
    pub modified_code: Option<String>,
    /// Error message if failed
    pub error: Option<String>,
    /// Files that were changed
    pub changed_files: Vec<String>,
    /// Performance delta after modification
    pub performance_delta: Option<f64>,
}

/// Analysis of existing code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeAnalysis {
    /// Identified improvement opportunities
    pub opportunities: Vec<ImprovementOpportunity>,
    /// Current tool inventory
    pub existing_tools: HashMap<String, ToolInfo>,
    /// Workflow patterns detected
    pub workflow_patterns: Vec<WorkflowPattern>,
    /// Performance bottlenecks
    pub bottlenecks: Vec<Bottleneck>,
}

/// An opportunity for code improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementOpportunity {
    /// Type of improvement
    pub improvement_type: ModificationType,
    /// Location in code
    pub location: CodeLocation,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Detailed description
    pub description: String,
}

/// Information about a tool in the codebase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInfo {
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// Usage frequency in benchmarks
    pub usage_frequency: f64,
    /// Success rate when used
    pub success_rate: f64,
}

/// A detected workflow pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowPattern {
    /// Pattern name
    pub name: String,
    /// Steps in the workflow
    pub steps: Vec<String>,
    /// Frequency of occurrence
    pub frequency: f64,
}

/// A performance bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    /// Type of bottleneck
    pub bottleneck_type: String,
    /// Location in code
    pub location: CodeLocation,
    /// Impact score (0.0 to 1.0)
    pub impact: f64,
}

/// Location in code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeLocation {
    /// File path
    pub file: String,
    /// Start line
    pub start_line: usize,
    /// End line
    pub end_line: usize,
}

/// Performance feedback from benchmark evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceFeedback {
    /// Task success rate
    pub success_rate: f64,
    /// Average task completion time
    pub avg_completion_time: f64,
    /// Error patterns encountered
    pub error_patterns: HashMap<String, usize>,
    /// Tool usage statistics
    pub tool_usage: HashMap<String, ToolUsageStats>,
}

/// Statistics about tool usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUsageStats {
    /// Number of times used
    pub usage_count: usize,
    /// Success rate when used
    pub success_rate: f64,
    /// Average execution time
    pub avg_execution_time: f64,
}
