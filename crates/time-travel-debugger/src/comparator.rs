use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

use crate::error::{Result, TimeDebuggerError};
use crate::snapshot::{ChangeType, StateChange, StateSnapshot};

/// Comparison options for state analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonOptions {
    /// Include only specific field paths
    pub include_paths: Option<HashSet<String>>,
    /// Exclude specific field paths
    pub exclude_paths: Option<HashSet<String>>,
    /// Maximum depth for nested comparisons
    pub max_depth: Option<usize>,
    /// Ignore changes smaller than this threshold
    pub ignore_threshold: Option<f64>,
    /// Compare only specific data types
    pub data_type_filter: Option<HashSet<String>>,
}

impl Default for ComparisonOptions {
    fn default() -> Self {
        Self {
            include_paths: None,
            exclude_paths: None,
            max_depth: None,
            ignore_threshold: None,
            data_type_filter: None,
        }
    }
}

/// Result of state comparison with detailed analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub from_snapshot_id: Uuid,
    pub to_snapshot_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub changes: Vec<StateChange>,
    pub summary: ComparisonSummary,
    pub metadata: HashMap<String, String>,
}

/// Summary statistics of the comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonSummary {
    pub total_changes: usize,
    pub additions: usize,
    pub modifications: usize,
    pub removals: usize,
    pub similarity_score: f64, // 0.0 to 1.0
    pub change_categories: HashMap<String, usize>,
}

/// Categories of changes for analysis
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ChangeCategory {
    Structural, // Object/array structure changes
    Value,      // Primitive value changes
    Type,       // Data type changes
    Size,       // Array/object size changes
    Metadata,   // Metadata-related changes
    Custom(String),
}

impl ChangeCategory {
    fn as_string(&self) -> String {
        match self {
            ChangeCategory::Structural => "structural".to_string(),
            ChangeCategory::Value => "value".to_string(),
            ChangeCategory::Type => "type".to_string(),
            ChangeCategory::Size => "size".to_string(),
            ChangeCategory::Metadata => "metadata".to_string(),
            ChangeCategory::Custom(name) => name.clone(),
        }
    }
}

/// Advanced state comparison and analysis
pub struct StateComparator {
    default_options: ComparisonOptions,
    change_categorizers: Vec<Box<dyn ChangeCategorizer + Send + Sync>>,
}

/// Trait for custom change categorization logic
pub trait ChangeCategorizer {
    fn categorize(&self, change: &StateChange) -> Vec<ChangeCategory>;
}

/// Default categorizer implementation
struct DefaultCategorizer;

impl ChangeCategorizer for DefaultCategorizer {
    fn categorize(&self, change: &StateChange) -> Vec<ChangeCategory> {
        let mut categories = Vec::new();

        // Type-based categorization
        if let (Some(old_val), Some(new_val)) = (&change.old_value, &change.new_value) {
            if std::mem::discriminant(old_val) != std::mem::discriminant(new_val) {
                categories.push(ChangeCategory::Type);
            }
        }

        // Structure-based categorization
        match &change.change_type {
            ChangeType::Added | ChangeType::Removed => {
                if change.path.contains('.') || change.path.contains('[') {
                    categories.push(ChangeCategory::Structural);
                } else {
                    categories.push(ChangeCategory::Value);
                }
            }
            ChangeType::Modified => {
                if let (Some(old_val), Some(new_val)) = (&change.old_value, &change.new_value) {
                    match (old_val, new_val) {
                        (serde_json::Value::Array(old_arr), serde_json::Value::Array(new_arr)) => {
                            if old_arr.len() != new_arr.len() {
                                categories.push(ChangeCategory::Size);
                            }
                            categories.push(ChangeCategory::Structural);
                        }
                        (
                            serde_json::Value::Object(old_obj),
                            serde_json::Value::Object(new_obj),
                        ) => {
                            if old_obj.len() != new_obj.len() {
                                categories.push(ChangeCategory::Size);
                            }
                            categories.push(ChangeCategory::Structural);
                        }
                        _ => {
                            categories.push(ChangeCategory::Value);
                        }
                    }
                }
            }
        }

        // Metadata categorization
        if change.path.starts_with("metadata") || change.path.starts_with("_") {
            categories.push(ChangeCategory::Metadata);
        }

        // Default to value if no other categories
        if categories.is_empty() {
            categories.push(ChangeCategory::Value);
        }

        categories
    }
}

impl StateComparator {
    pub fn new(default_options: ComparisonOptions) -> Self {
        let mut comparator = Self {
            default_options,
            change_categorizers: Vec::new(),
        };

        // Add default categorizer
        comparator.add_categorizer(Box::new(DefaultCategorizer));
        comparator
    }

    /// Add a custom change categorizer
    pub fn add_categorizer(&mut self, categorizer: Box<dyn ChangeCategorizer + Send + Sync>) {
        self.change_categorizers.push(categorizer);
    }

    /// Compare two state snapshots
    pub async fn compare_snapshots(
        &self,
        from_snapshot: &StateSnapshot,
        to_snapshot: &StateSnapshot,
        options: Option<ComparisonOptions>,
    ) -> Result<ComparisonResult> {
        let options = options.unwrap_or_else(|| self.default_options.clone());

        // Validate that snapshots are from the same agent
        if from_snapshot.agent_id != to_snapshot.agent_id {
            return Err(TimeDebuggerError::ComparisonFailed {
                reason: "Snapshots are from different agents".to_string(),
            });
        }

        let changes = self.compute_changes(
            &from_snapshot.state_data,
            &to_snapshot.state_data,
            &options,
            "",
            0,
        )?;

        let summary = self.generate_summary(&changes);

        Ok(ComparisonResult {
            from_snapshot_id: from_snapshot.id,
            to_snapshot_id: to_snapshot.id,
            timestamp: Utc::now(),
            changes,
            summary,
            metadata: HashMap::new(),
        })
    }

    /// Compare states across multiple snapshots (timeline analysis)
    pub async fn compare_timeline(
        &self,
        snapshots: &[StateSnapshot],
        options: Option<ComparisonOptions>,
    ) -> Result<Vec<ComparisonResult>> {
        if snapshots.len() < 2 {
            return Err(TimeDebuggerError::ComparisonFailed {
                reason: "Need at least 2 snapshots for timeline comparison".to_string(),
            });
        }

        let mut results = Vec::new();

        for i in 1..snapshots.len() {
            let comparison = self
                .compare_snapshots(&snapshots[i - 1], &snapshots[i], options.clone())
                .await?;
            results.push(comparison);
        }

        Ok(results)
    }

    /// Find divergence point between two snapshot sequences
    pub async fn find_divergence(
        &self,
        sequence_a: &[StateSnapshot],
        sequence_b: &[StateSnapshot],
        options: Option<ComparisonOptions>,
    ) -> Result<Option<(usize, usize)>> {
        let min_len = std::cmp::min(sequence_a.len(), sequence_b.len());

        for i in 0..min_len {
            let comparison = self
                .compare_snapshots(&sequence_a[i], &sequence_b[i], options.clone())
                .await?;

            if !comparison.changes.is_empty() {
                return Ok(Some((i, i)));
            }
        }

        // If sequences have different lengths, divergence is at the shorter length
        if sequence_a.len() != sequence_b.len() {
            Ok(Some((min_len, min_len)))
        } else {
            Ok(None) // No divergence found
        }
    }

    /// Analyze change patterns across multiple comparisons
    pub async fn analyze_patterns(
        &self,
        comparisons: &[ComparisonResult],
    ) -> Result<PatternAnalysis> {
        if comparisons.is_empty() {
            return Err(TimeDebuggerError::ComparisonFailed {
                reason: "No comparisons provided for pattern analysis".to_string(),
            });
        }

        let mut field_change_counts: HashMap<String, usize> = HashMap::new();
        let mut category_trends: HashMap<String, Vec<usize>> = HashMap::new();
        let mut similarity_trend: Vec<f64> = Vec::new();

        for comparison in comparisons {
            similarity_trend.push(comparison.summary.similarity_score);

            // Count field changes
            for change in &comparison.changes {
                *field_change_counts.entry(change.path.clone()).or_insert(0) += 1;
            }

            // Track category trends
            for (category, count) in &comparison.summary.change_categories {
                category_trends
                    .entry(category.clone())
                    .or_insert_with(Vec::new)
                    .push(*count);
            }
        }

        // Find most volatile fields
        let mut volatile_fields: Vec<(String, usize)> = field_change_counts.into_iter().collect();
        volatile_fields.sort_by(|a, b| b.1.cmp(&a.1));
        volatile_fields.truncate(10); // Top 10

        // Calculate trend directions
        let mut category_trend_directions = HashMap::new();
        for (category, counts) in category_trends {
            let trend = if counts.len() >= 2 {
                let first_half = counts.iter().take(counts.len() / 2).sum::<usize>() as f64;
                let second_half = counts.iter().skip(counts.len() / 2).sum::<usize>() as f64;

                let first_avg = first_half / (counts.len() / 2) as f64;
                let second_avg = second_half / (counts.len() - counts.len() / 2) as f64;

                if second_avg > first_avg * 1.1 {
                    TrendDirection::Increasing
                } else if second_avg < first_avg * 0.9 {
                    TrendDirection::Decreasing
                } else {
                    TrendDirection::Stable
                }
            } else {
                TrendDirection::Unknown
            };

            category_trend_directions.insert(category, trend);
        }

        Ok(PatternAnalysis {
            volatile_fields,
            category_trends: category_trend_directions,
            similarity_trend,
            change_frequency: comparisons.len() as f64
                / (comparisons.last().unwrap().timestamp - comparisons.first().unwrap().timestamp)
                    .num_minutes() as f64,
        })
    }

    /// Generate a visual diff representation
    pub async fn generate_visual_diff(
        &self,
        comparison: &ComparisonResult,
        format: DiffFormat,
    ) -> Result<String> {
        match format {
            DiffFormat::Unified => self.generate_unified_diff(comparison),
            DiffFormat::SideBySide => self.generate_side_by_side_diff(comparison),
            DiffFormat::Json => self.generate_json_diff(comparison),
            DiffFormat::Html => self.generate_html_diff(comparison),
        }
    }

    // Private helper methods

    fn compute_changes(
        &self,
        from_value: &serde_json::Value,
        to_value: &serde_json::Value,
        options: &ComparisonOptions,
        path: &str,
        depth: usize,
    ) -> Result<Vec<StateChange>> {
        let mut changes = Vec::new();
        self.compute_changes_recursive(from_value, to_value, path, &mut changes, options, depth)?;
        Ok(changes)
    }

    fn compute_changes_recursive(
        &self,
        from_value: &serde_json::Value,
        to_value: &serde_json::Value,
        path: &str,
        changes: &mut Vec<StateChange>,
        options: &ComparisonOptions,
        depth: usize,
    ) -> Result<()> {
        use serde_json::Value;

        // Check depth limit
        if let Some(max_depth) = options.max_depth {
            if depth > max_depth {
                return Ok(());
            }
        }

        // Helper function to check if a path should be excluded
        let should_exclude_path = |check_path: &str| -> bool {
            if let Some(exclude_paths) = &options.exclude_paths {
                exclude_paths.iter().any(|p| check_path.starts_with(p))
            } else {
                false
            }
        };

        // Helper function to check if a path should be included
        let should_include_path = |check_path: &str| -> bool {
            if let Some(include_paths) = &options.include_paths {
                include_paths.iter().any(|p| check_path.starts_with(p))
            } else {
                true // Include by default if no include filter
            }
        };

        match (from_value, to_value) {
            (Value::Object(from_obj), Value::Object(to_obj)) => {
                // Check for removed or modified keys
                for (key, from_val) in from_obj {
                    let new_path = if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{}.{}", path, key)
                    };

                    // Skip if path should be excluded or not included
                    if should_exclude_path(&new_path) || !should_include_path(&new_path) {
                        continue;
                    }

                    match to_obj.get(key) {
                        Some(to_val) => {
                            self.compute_changes_recursive(
                                from_val,
                                to_val,
                                &new_path,
                                changes,
                                options,
                                depth + 1,
                            )?;
                        }
                        None => {
                            changes.push(StateChange {
                                path: new_path,
                                change_type: ChangeType::Removed,
                                old_value: Some(from_val.clone()),
                                new_value: None,
                            });
                        }
                    }
                }

                // Check for added keys
                for (key, to_val) in to_obj {
                    if !from_obj.contains_key(key) {
                        let new_path = if path.is_empty() {
                            key.clone()
                        } else {
                            format!("{}.{}", path, key)
                        };

                        // Skip if path should be excluded or not included
                        if should_exclude_path(&new_path) || !should_include_path(&new_path) {
                            continue;
                        }

                        changes.push(StateChange {
                            path: new_path,
                            change_type: ChangeType::Added,
                            old_value: None,
                            new_value: Some(to_val.clone()),
                        });
                    }
                }
            }
            (Value::Array(from_arr), Value::Array(to_arr)) => {
                let max_len = std::cmp::max(from_arr.len(), to_arr.len());

                for i in 0..max_len {
                    let new_path = if path.is_empty() {
                        format!("[{}]", i)
                    } else {
                        format!("{}[{}]", path, i)
                    };

                    // Skip if path should be excluded or not included
                    if should_exclude_path(&new_path) || !should_include_path(&new_path) {
                        continue;
                    }

                    match (from_arr.get(i), to_arr.get(i)) {
                        (Some(from_val), Some(to_val)) => {
                            self.compute_changes_recursive(
                                from_val,
                                to_val,
                                &new_path,
                                changes,
                                options,
                                depth + 1,
                            )?;
                        }
                        (Some(from_val), None) => {
                            changes.push(StateChange {
                                path: new_path,
                                change_type: ChangeType::Removed,
                                old_value: Some(from_val.clone()),
                                new_value: None,
                            });
                        }
                        (None, Some(to_val)) => {
                            changes.push(StateChange {
                                path: new_path,
                                change_type: ChangeType::Added,
                                old_value: None,
                                new_value: Some(to_val.clone()),
                            });
                        }
                        (None, None) => unreachable!(),
                    }
                }
            }
            _ => {
                if from_value != to_value {
                    // Skip if path should be excluded or not included
                    if should_exclude_path(path) || !should_include_path(path) {
                        return Ok(());
                    }

                    // Check ignore threshold for numeric values
                    if let Some(threshold) = options.ignore_threshold {
                        if let (Value::Number(from_num), Value::Number(to_num)) =
                            (from_value, to_value)
                        {
                            if let (Some(from_f64), Some(to_f64)) =
                                (from_num.as_f64(), to_num.as_f64())
                            {
                                let diff = (to_f64 - from_f64).abs();
                                if diff < threshold {
                                    return Ok(());
                                }
                            }
                        }
                    }

                    changes.push(StateChange {
                        path: path.to_string(),
                        change_type: ChangeType::Modified,
                        old_value: Some(from_value.clone()),
                        new_value: Some(to_value.clone()),
                    });
                }
            }
        }

        Ok(())
    }

    fn generate_summary(&self, changes: &[StateChange]) -> ComparisonSummary {
        let mut additions = 0;
        let mut modifications = 0;
        let mut removals = 0;
        let mut change_categories: HashMap<String, usize> = HashMap::new();

        for change in changes {
            match change.change_type {
                ChangeType::Added => additions += 1,
                ChangeType::Modified => modifications += 1,
                ChangeType::Removed => removals += 1,
            }

            // Categorize changes
            for categorizer in &self.change_categorizers {
                let categories = categorizer.categorize(change);
                for category in categories {
                    *change_categories.entry(category.as_string()).or_insert(0) += 1;
                }
            }
        }

        // Calculate similarity score (simple heuristic)
        let total_changes = changes.len() as f64;
        let similarity_score = if total_changes == 0.0 {
            1.0
        } else {
            // More sophisticated similarity calculation could be implemented
            (1.0 - (total_changes / 100.0)).max(0.0)
        };

        ComparisonSummary {
            total_changes: changes.len(),
            additions,
            modifications,
            removals,
            similarity_score,
            change_categories,
        }
    }

    fn generate_unified_diff(&self, comparison: &ComparisonResult) -> Result<String> {
        let mut diff = String::new();

        diff.push_str(&format!(
            "--- Snapshot {} ({})\n",
            comparison.from_snapshot_id, "timestamp_from"
        )); // In real implementation, include actual timestamps
        diff.push_str(&format!(
            "+++ Snapshot {} ({})\n",
            comparison.to_snapshot_id, "timestamp_to"
        ));

        for change in &comparison.changes {
            match change.change_type {
                ChangeType::Added => {
                    diff.push_str(&format!(
                        "+{}: {}\n",
                        change.path,
                        change.new_value.as_ref().unwrap()
                    ));
                }
                ChangeType::Removed => {
                    diff.push_str(&format!(
                        "-{}: {}\n",
                        change.path,
                        change.old_value.as_ref().unwrap()
                    ));
                }
                ChangeType::Modified => {
                    diff.push_str(&format!(
                        "-{}: {}\n",
                        change.path,
                        change.old_value.as_ref().unwrap()
                    ));
                    diff.push_str(&format!(
                        "+{}: {}\n",
                        change.path,
                        change.new_value.as_ref().unwrap()
                    ));
                }
            }
        }

        Ok(diff)
    }

    fn generate_side_by_side_diff(&self, comparison: &ComparisonResult) -> Result<String> {
        let mut diff = String::new();

        diff.push_str(&format!(
            "{:<50} | {:<50}\n",
            format!("Snapshot {}", comparison.from_snapshot_id),
            format!("Snapshot {}", comparison.to_snapshot_id)
        ));
        diff.push_str(&format!("{:-<50} | {:-<50}\n", "", ""));

        for change in &comparison.changes {
            let left = match &change.old_value {
                Some(val) => format!("{}: {}", change.path, val),
                None => format!("{}: <not present>", change.path),
            };

            let right = match &change.new_value {
                Some(val) => format!("{}: {}", change.path, val),
                None => format!("{}: <removed>", change.path),
            };

            diff.push_str(&format!("{:<50} | {:<50}\n", left, right));
        }

        Ok(diff)
    }

    fn generate_json_diff(&self, comparison: &ComparisonResult) -> Result<String> {
        serde_json::to_string_pretty(comparison).map_err(|e| e.into())
    }

    fn generate_html_diff(&self, comparison: &ComparisonResult) -> Result<String> {
        let mut html = String::new();

        html.push_str("<div class=\"diff-container\">\n");
        html.push_str(&format!(
            "<h3>Comparison: {} → {}</h3>\n",
            comparison.from_snapshot_id, comparison.to_snapshot_id
        ));

        html.push_str("<table class=\"diff-table\">\n");
        html.push_str(
            "<tr><th>Path</th><th>Change Type</th><th>Old Value</th><th>New Value</th></tr>\n",
        );

        for change in &comparison.changes {
            let change_class = match change.change_type {
                ChangeType::Added => "added",
                ChangeType::Modified => "modified",
                ChangeType::Removed => "removed",
            };

            let old_value = change
                .old_value
                .as_ref()
                .map(|v| v.to_string())
                .unwrap_or_else(|| "<not present>".to_string());

            let new_value = change
                .new_value
                .as_ref()
                .map(|v| v.to_string())
                .unwrap_or_else(|| "<removed>".to_string());

            html.push_str(&format!(
                "<tr class=\"{}\">\n  <td>{}</td>\n  <td>{:?}</td>\n  <td>{}</td>\n  <td>{}</td>\n</tr>\n",
                change_class, change.path, change.change_type, old_value, new_value
            ));
        }

        html.push_str("</table>\n");
        html.push_str("</div>\n");

        Ok(html)
    }
}

/// Pattern analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternAnalysis {
    pub volatile_fields: Vec<(String, usize)>, // (field_path, change_count)
    pub category_trends: HashMap<String, TrendDirection>,
    pub similarity_trend: Vec<f64>,
    pub change_frequency: f64, // changes per minute
}

/// Trend direction for pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Unknown,
}

/// Supported diff output formats
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiffFormat {
    Unified,
    SideBySide,
    Json,
    Html,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn create_test_snapshot(
        id: Uuid,
        agent_id: Uuid,
        state_data: serde_json::Value,
    ) -> StateSnapshot {
        StateSnapshot {
            id,
            agent_id,
            timestamp: Utc::now(),
            state_data,
            memory_usage: 1024,
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_basic_comparison() {
        let comparator = StateComparator::new(ComparisonOptions::default());
        let agent_id = Uuid::new_v4();

        let snapshot1 = create_test_snapshot(
            Uuid::new_v4(),
            agent_id,
            json!({"health": 100, "position": {"x": 0, "y": 0}}),
        );

        let snapshot2 = create_test_snapshot(
            Uuid::new_v4(),
            agent_id,
            json!({"health": 80, "position": {"x": 5, "y": 0}, "inventory": ["sword"]}),
        );

        let result = comparator
            .compare_snapshots(&snapshot1, &snapshot2, None)
            .await
            .unwrap();

        assert_eq!(result.summary.total_changes, 3);
        assert_eq!(result.summary.modifications, 2); // health and position.x
        assert_eq!(result.summary.additions, 1); // inventory

        // Find specific changes
        let health_change = result.changes.iter().find(|c| c.path == "health")?;
        assert_eq!(health_change.old_value, Some(json!(100)));
        assert_eq!(health_change.new_value, Some(json!(80)));

        let inventory_change = result
            .changes
            .iter()
            .find(|c| c.path == "inventory")
            .unwrap();
        assert!(matches!(inventory_change.change_type, ChangeType::Added));
    }

    #[tokio::test]
    async fn test_comparison_with_filters() {
        let comparator = StateComparator::new(ComparisonOptions::default());
        let agent_id = Uuid::new_v4();

        let snapshot1 = create_test_snapshot(
            Uuid::new_v4(),
            agent_id,
            json!({"health": 100, "position": {"x": 0, "y": 0}, "metadata": {"version": 1}}),
        );

        let snapshot2 = create_test_snapshot(
            Uuid::new_v4(),
            agent_id,
            json!({"health": 80, "position": {"x": 5, "y": 0}, "metadata": {"version": 2}}),
        );

        // Exclude metadata from comparison
        let options = ComparisonOptions {
            exclude_paths: Some(["metadata".to_string()].into_iter().collect()),
            ..Default::default()
        };

        let result = comparator
            .compare_snapshots(&snapshot1, &snapshot2, Some(options))
            .await
            .unwrap();

        // Should not include metadata changes
        assert!(!result
            .changes
            .iter()
            .any(|c| c.path.starts_with("metadata")));
        assert_eq!(result.summary.total_changes, 2); // Only health and position.x
    }

    #[tokio::test]
    async fn test_array_comparison() {
        let comparator = StateComparator::new(ComparisonOptions::default());
        let agent_id = Uuid::new_v4();

        let snapshot1 = create_test_snapshot(
            Uuid::new_v4(),
            agent_id,
            json!({"inventory": ["sword", "shield"]}),
        );

        let snapshot2 = create_test_snapshot(
            Uuid::new_v4(),
            agent_id,
            json!({"inventory": ["sword", "potion", "shield"]}),
        );

        let result = comparator
            .compare_snapshots(&snapshot1, &snapshot2, None)
            .await
            .unwrap();

        // Should detect array changes
        assert!(result.changes.iter().any(|c| c.path.contains("inventory")));

        // The algorithm detects this as: inventory[1] modified from "shield" to "potion"
        // and inventory[2] added as "shield"
        let potion_change = result
            .changes
            .iter()
            .find(|c| c.path == "inventory[1]")
            .unwrap();
        assert!(matches!(potion_change.change_type, ChangeType::Modified));
        assert_eq!(potion_change.new_value, Some(json!("potion")));

        // Shield was moved to position 2
        let shield_change = result
            .changes
            .iter()
            .find(|c| c.path == "inventory[2]")
            .unwrap();
        assert!(matches!(shield_change.change_type, ChangeType::Added));
        assert_eq!(shield_change.new_value, Some(json!("shield")));
    }

    #[tokio::test]
    async fn test_ignore_threshold() {
        let comparator = StateComparator::new(ComparisonOptions::default());
        let agent_id = Uuid::new_v4();

        let snapshot1 = create_test_snapshot(
            Uuid::new_v4(),
            agent_id,
            json!({"temperature": 25.1, "pressure": 100.0}),
        );

        let snapshot2 = create_test_snapshot(
            Uuid::new_v4(),
            agent_id,
            json!({"temperature": 25.11, "pressure": 105.0}),
        );

        // Ignore changes smaller than 0.1
        let options = ComparisonOptions {
            ignore_threshold: Some(0.1),
            ..Default::default()
        };

        let result = comparator
            .compare_snapshots(&snapshot1, &snapshot2, Some(options))
            .await
            .unwrap();

        // Temperature change should be ignored, pressure should be detected
        assert!(!result.changes.iter().any(|c| c.path == "temperature"));
        assert!(result.changes.iter().any(|c| c.path == "pressure"));
    }

    #[tokio::test]
    async fn test_depth_limit() {
        let comparator = StateComparator::new(ComparisonOptions::default());
        let agent_id = Uuid::new_v4();

        let snapshot1 = create_test_snapshot(
            Uuid::new_v4(),
            agent_id,
            json!({
                "level1": {
                    "level2": {
                        "level3": {
                            "value": "old"
                        }
                    }
                }
            }),
        );

        let snapshot2 = create_test_snapshot(
            Uuid::new_v4(),
            agent_id,
            json!({
                "level1": {
                    "level2": {
                        "level3": {
                            "value": "new"
                        }
                    }
                }
            }),
        );

        // Limit depth to 2 levels
        let options = ComparisonOptions {
            max_depth: Some(2),
            ..Default::default()
        };

        let result = comparator
            .compare_snapshots(&snapshot1, &snapshot2, Some(options))
            .await
            .unwrap();

        // Should not detect changes at level 3
        assert!(!result.changes.iter().any(|c| c.path.contains("level3")));
    }

    #[tokio::test]
    async fn test_timeline_comparison() {
        let comparator = StateComparator::new(ComparisonOptions::default());
        let agent_id = Uuid::new_v4();

        let snapshots = vec![
            create_test_snapshot(Uuid::new_v4(), agent_id, json!({"step": 1, "value": 10})),
            create_test_snapshot(Uuid::new_v4(), agent_id, json!({"step": 2, "value": 20})),
            create_test_snapshot(Uuid::new_v4(), agent_id, json!({"step": 3, "value": 30})),
        ];

        let results = comparator.compare_timeline(&snapshots, None).await?;

        assert_eq!(results.len(), 2); // 3 snapshots = 2 comparisons

        // Check first comparison
        assert_eq!(results[0].from_snapshot_id, snapshots[0].id);
        assert_eq!(results[0].to_snapshot_id, snapshots[1].id);

        // Each comparison should have 2 changes (step and value)
        assert_eq!(results[0].summary.total_changes, 2);
        assert_eq!(results[1].summary.total_changes, 2);
    }

    #[tokio::test]
    async fn test_find_divergence() {
        let comparator = StateComparator::new(ComparisonOptions::default());
        let agent_id = Uuid::new_v4();

        let sequence_a = vec![
            create_test_snapshot(Uuid::new_v4(), agent_id, json!({"value": 1})),
            create_test_snapshot(Uuid::new_v4(), agent_id, json!({"value": 2})),
            create_test_snapshot(Uuid::new_v4(), agent_id, json!({"value": 3})),
        ];

        let sequence_b = vec![
            create_test_snapshot(Uuid::new_v4(), agent_id, json!({"value": 1})),
            create_test_snapshot(Uuid::new_v4(), agent_id, json!({"value": 5})), // Divergence here
            create_test_snapshot(Uuid::new_v4(), agent_id, json!({"value": 6})),
        ];

        let divergence = comparator
            .find_divergence(&sequence_a, &sequence_b, None)
            .await
            .unwrap();

        assert_eq!(divergence, Some((1, 1))); // Divergence at index 1
    }

    #[tokio::test]
    async fn test_pattern_analysis() {
        let comparator = StateComparator::new(ComparisonOptions::default());

        // Create mock comparison results
        let mut comparisons = Vec::new();
        for i in 0..5 {
            let changes = vec![
                StateChange {
                    path: "health".to_string(),
                    change_type: ChangeType::Modified,
                    old_value: Some(json!(100 - i * 10)),
                    new_value: Some(json!(100 - (i + 1) * 10)),
                },
                StateChange {
                    path: format!("items[{}]", i),
                    change_type: ChangeType::Added,
                    old_value: None,
                    new_value: Some(json!(format!("item_{}", i))),
                },
            ];

            let summary = ComparisonSummary {
                total_changes: changes.len(),
                additions: 1,
                modifications: 1,
                removals: 0,
                similarity_score: 0.8 - (i as f64 * 0.1),
                change_categories: [("value".to_string(), 2)].into_iter().collect(),
            };

            comparisons.push(ComparisonResult {
                from_snapshot_id: Uuid::new_v4(),
                to_snapshot_id: Uuid::new_v4(),
                timestamp: Utc::now(),
                changes,
                summary,
                metadata: HashMap::new(),
            });
        }

        let analysis = comparator.analyze_patterns(&comparisons).await?;

        // Health should be the most volatile field
        assert_eq!(analysis.volatile_fields[0].0, "health");
        assert_eq!(analysis.volatile_fields[0].1, 5); // Changed in all 5 comparisons

        // Similarity should be trending downward
        assert_eq!(analysis.similarity_trend.len(), 5);
        assert!(analysis.similarity_trend[0] > analysis.similarity_trend[4]);
    }

    #[tokio::test]
    async fn test_visual_diff_generation() {
        let comparator = StateComparator::new(ComparisonOptions::default());

        let comparison = ComparisonResult {
            from_snapshot_id: Uuid::new_v4(),
            to_snapshot_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            changes: vec![
                StateChange {
                    path: "health".to_string(),
                    change_type: ChangeType::Modified,
                    old_value: Some(json!(100)),
                    new_value: Some(json!(80)),
                },
                StateChange {
                    path: "weapon".to_string(),
                    change_type: ChangeType::Added,
                    old_value: None,
                    new_value: Some(json!("sword")),
                },
            ],
            summary: ComparisonSummary {
                total_changes: 2,
                additions: 1,
                modifications: 1,
                removals: 0,
                similarity_score: 0.9,
                change_categories: HashMap::new(),
            },
            metadata: HashMap::new(),
        };

        // Test unified diff
        let unified = comparator
            .generate_visual_diff(&comparison, DiffFormat::Unified)
            .await
            .unwrap();
        assert!(unified.contains("-health: 100"));
        assert!(unified.contains("+health: 80"));
        assert!(unified.contains("+weapon: \"sword\""));

        // Test JSON diff
        let json_diff = comparator
            .generate_visual_diff(&comparison, DiffFormat::Json)
            .await
            .unwrap();
        assert!(serde_json::from_str::<serde_json::Value>(&json_diff).is_ok());

        // Test HTML diff
        let html_diff = comparator
            .generate_visual_diff(&comparison, DiffFormat::Html)
            .await
            .unwrap();
        assert!(html_diff.contains("<div class=\"diff-container\">"));
        assert!(html_diff.contains("<table class=\"diff-table\">"));
    }

    #[tokio::test]
    async fn test_change_categorization() {
        let categorizer = DefaultCategorizer;

        // Test type change
        let type_change = StateChange {
            path: "value".to_string(),
            change_type: ChangeType::Modified,
            old_value: Some(json!("string")),
            new_value: Some(json!(42)),
        };
        let categories = categorizer.categorize(&type_change);
        assert!(categories.contains(&ChangeCategory::Type));

        // Test metadata change
        let metadata_change = StateChange {
            path: "metadata.version".to_string(),
            change_type: ChangeType::Modified,
            old_value: Some(json!(1)),
            new_value: Some(json!(2)),
        };
        let categories = categorizer.categorize(&metadata_change);
        assert!(categories.contains(&ChangeCategory::Metadata));

        // Test structural change
        let structural_change = StateChange {
            path: "config.settings.option".to_string(),
            change_type: ChangeType::Added,
            old_value: None,
            new_value: Some(json!(true)),
        };
        let categories = categorizer.categorize(&structural_change);
        assert!(categories.contains(&ChangeCategory::Structural));
    }

    #[tokio::test]
    async fn test_comparison_error_handling() {
        let comparator = StateComparator::new(ComparisonOptions::default());
        let agent1_id = Uuid::new_v4();
        let agent2_id = Uuid::new_v4();

        let snapshot1 = create_test_snapshot(Uuid::new_v4(), agent1_id, json!({}));
        let snapshot2 = create_test_snapshot(Uuid::new_v4(), agent2_id, json!({}));

        // Should fail when comparing snapshots from different agents
        let result = comparator
            .compare_snapshots(&snapshot1, &snapshot2, None)
            .await;
        assert!(matches!(
            result,
            Err(TimeDebuggerError::ComparisonFailed { .. })
        ));

        // Should fail timeline comparison with too few snapshots
        let result = comparator.compare_timeline(&[snapshot1], None).await;
        assert!(matches!(
            result,
            Err(TimeDebuggerError::ComparisonFailed { .. })
        ));

        // Should fail pattern analysis with empty comparisons
        let result = comparator.analyze_patterns(&[]).await;
        assert!(matches!(
            result,
            Err(TimeDebuggerError::ComparisonFailed { .. })
        ));
    }
}
