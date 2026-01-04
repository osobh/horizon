//! Goal representation and management

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Unique goal identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GoalId(pub Uuid);

impl GoalId {
    /// Create a new goal ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for GoalId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for GoalId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Goal priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub enum GoalPriority {
    /// Lowest priority
    Low = 0,
    /// Normal priority
    #[default]
    Normal = 1,
    /// High priority
    High = 2,
    /// Critical priority
    Critical = 3,
}

/// Goal state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalState {
    /// Goal is pending
    Pending,
    /// Goal is being analyzed
    Analyzing,
    /// Goal is being planned
    Planning,
    /// Goal is being executed
    Executing,
    /// Goal completed successfully
    Completed,
    /// Goal failed
    Failed,
    /// Goal was cancelled
    Cancelled,
}

/// Goal constraints
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GoalConstraints {
    /// Memory limit in bytes
    pub memory_limit: Option<usize>,
    /// GPU memory limit in bytes
    pub gpu_memory_limit: Option<usize>,
    /// Time limit
    pub time_limit: Option<std::time::Duration>,
    /// Accuracy target (0.0 - 1.0)
    pub accuracy_target: Option<f32>,
    /// Throughput target (operations per second)
    pub throughput_target: Option<f64>,
    /// Power budget in watts
    pub power_budget: Option<f32>,
    /// Custom constraints
    pub custom: HashMap<String, serde_json::Value>,
}

/// A goal represents a high-level objective for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    /// Unique goal ID
    pub id: GoalId,
    /// Human-readable description
    pub description: String,
    /// Goal priority
    pub priority: GoalPriority,
    /// Goal state
    pub state: GoalState,
    /// Goal constraints
    pub constraints: GoalConstraints,
    /// Parent goal ID (for sub-goals)
    pub parent_id: Option<GoalId>,
    /// Metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Creation time
    pub created_at: DateTime<Utc>,
    /// Last update time
    pub updated_at: DateTime<Utc>,
    /// Completion time
    pub completed_at: Option<DateTime<Utc>>,
}

impl Goal {
    /// Create a new goal
    pub fn new(description: String, priority: GoalPriority) -> Self {
        let now = Utc::now();
        Self {
            id: GoalId::new(),
            description,
            priority,
            state: GoalState::Pending,
            constraints: GoalConstraints::default(),
            parent_id: None,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
            completed_at: None,
        }
    }

    /// Create a new goal with constraints
    pub fn with_constraints(
        description: String,
        priority: GoalPriority,
        constraints: GoalConstraints,
    ) -> Self {
        let mut goal = Self::new(description, priority);
        goal.constraints = constraints;
        goal
    }

    /// Create a sub-goal
    pub fn create_subgoal(&self, description: String) -> Self {
        let mut subgoal = Self::new(description, self.priority);
        subgoal.parent_id = Some(self.id);
        subgoal
    }

    /// Update goal state
    pub fn update_state(&mut self, new_state: GoalState) {
        self.state = new_state;
        self.updated_at = Utc::now();

        if matches!(
            new_state,
            GoalState::Completed | GoalState::Failed | GoalState::Cancelled
        ) {
            self.completed_at = Some(Utc::now());
        }
    }

    /// Check if goal is terminal (completed, failed, or cancelled)
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            GoalState::Completed | GoalState::Failed | GoalState::Cancelled
        )
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: serde_json::Value) {
        self.metadata.insert(key, value);
        self.updated_at = Utc::now();
    }

    /// Get execution time
    pub fn execution_time(&self) -> Option<std::time::Duration> {
        self.completed_at.map(|completed| {
            let duration = completed - self.created_at;
            std::time::Duration::from_secs(duration.num_seconds() as u64)
        })
    }
}

/// Goal template for common goal patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalTemplate {
    /// Template name
    pub name: String,
    /// Template description
    pub description: String,
    /// Default priority
    pub default_priority: GoalPriority,
    /// Default constraints
    pub default_constraints: GoalConstraints,
    /// Required parameters
    pub required_params: Vec<String>,
    /// Optional parameters
    pub optional_params: Vec<String>,
}

impl GoalTemplate {
    /// Create a goal from template
    pub fn instantiate(&self, params: HashMap<String, serde_json::Value>) -> Result<Goal, String> {
        // Check required parameters
        for param in &self.required_params {
            if !params.contains_key(param) {
                return Err(format!("Missing required parameter: {param}"));
            }
        }

        // Build description with parameters
        let mut description = self.description.clone();
        for (key, value) in &params {
            let placeholder = format!("{{{}}}", key);
            let replacement = match value {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            description = description.replace(&placeholder, &replacement);
        }

        let mut goal = Goal::new(description, self.default_priority);
        goal.constraints = self.default_constraints.clone();
        goal.metadata = params;

        Ok(goal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goal_id() {
        let id1 = GoalId::new();
        let id2 = GoalId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_goal_priority_ordering() {
        assert!(GoalPriority::Critical > GoalPriority::High);
        assert!(GoalPriority::High > GoalPriority::Normal);
        assert!(GoalPriority::Normal > GoalPriority::Low);
    }

    #[test]
    fn test_goal_creation() {
        let goal = Goal::new(
            "Optimize matrix multiplication".to_string(),
            GoalPriority::High,
        );

        assert_eq!(goal.description, "Optimize matrix multiplication");
        assert_eq!(goal.priority, GoalPriority::High);
        assert_eq!(goal.state, GoalState::Pending);
        assert!(goal.parent_id.is_none());
    }

    #[test]
    fn test_goal_with_constraints() {
        let constraints = GoalConstraints {
            memory_limit: Some(1024 * 1024 * 1024), // 1GB
            accuracy_target: Some(0.99),
            ..Default::default()
        };

        let goal =
            Goal::with_constraints("Train model".to_string(), GoalPriority::Normal, constraints);

        assert_eq!(goal.constraints.memory_limit, Some(1024 * 1024 * 1024));
        assert_eq!(goal.constraints.accuracy_target, Some(0.99));
    }

    #[test]
    fn test_subgoal_creation() {
        let parent = Goal::new("Parent goal".to_string(), GoalPriority::High);

        let subgoal = parent.create_subgoal("Sub goal".to_string());

        assert_eq!(subgoal.parent_id, Some(parent.id));
        assert_eq!(subgoal.priority, parent.priority);
    }

    #[test]
    fn test_goal_state_update() {
        let mut goal = Goal::new("Test goal".to_string(), GoalPriority::Normal);

        assert!(goal.completed_at.is_none());

        goal.update_state(GoalState::Executing);
        assert_eq!(goal.state, GoalState::Executing);
        assert!(goal.completed_at.is_none());

        goal.update_state(GoalState::Completed);
        assert_eq!(goal.state, GoalState::Completed);
        assert!(goal.completed_at.is_some());
        assert!(goal.is_terminal());
    }

    #[test]
    fn test_goal_metadata() {
        let mut goal = Goal::new("Test goal".to_string(), GoalPriority::Normal);

        goal.add_metadata(
            "model_type".to_string(),
            serde_json::Value::String("transformer".to_string()),
        );

        assert_eq!(goal.metadata.len(), 1);
        assert_eq!(
            goal.metadata.get("model_type"),
            Some(&serde_json::Value::String("transformer".to_string()))
        );
    }

    #[test]
    fn test_goal_template() {
        let template = GoalTemplate {
            name: "optimize_kernel".to_string(),
            description: "Optimize {kernel_type} kernel for {device}".to_string(),
            default_priority: GoalPriority::High,
            default_constraints: GoalConstraints {
                throughput_target: Some(1000.0),
                ..Default::default()
            },
            required_params: vec!["kernel_type".to_string(), "device".to_string()],
            optional_params: vec!["batch_size".to_string()],
        };

        let mut params = HashMap::new();
        params.insert(
            "kernel_type".to_string(),
            serde_json::Value::String("matmul".to_string()),
        );
        params.insert(
            "device".to_string(),
            serde_json::Value::String("RTX_4090".to_string()),
        );

        let goal = template.instantiate(params).unwrap();
        assert_eq!(goal.description, "Optimize matmul kernel for RTX_4090");
        assert_eq!(goal.priority, GoalPriority::High);
        assert_eq!(goal.constraints.throughput_target, Some(1000.0));
    }

    #[test]
    fn test_goal_template_missing_param() {
        let template = GoalTemplate {
            name: "test".to_string(),
            description: "Test {param}".to_string(),
            default_priority: GoalPriority::Normal,
            default_constraints: GoalConstraints::default(),
            required_params: vec!["param".to_string()],
            optional_params: vec![],
        };

        let params = HashMap::new();
        let result = template.instantiate(params);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing required parameter"));
    }

    #[test]
    fn test_goal_id_display() {
        let id = GoalId::new();
        let display = format!("{id}");
        assert!(!display.is_empty());
        assert_eq!(display.len(), 36); // UUID string length
    }

    #[test]
    fn test_goal_priority_default() {
        let priority = GoalPriority::default();
        assert_eq!(priority, GoalPriority::Normal);
    }

    #[test]
    fn test_goal_state_terminal_checks() {
        let states = vec![
            (GoalState::Pending, false),
            (GoalState::Executing, false),
            (GoalState::Suspended, false),
            (GoalState::Completed, true),
            (GoalState::Failed, true),
            (GoalState::Cancelled, true),
        ];

        for (state, expected_terminal) in states {
            let mut goal = Goal::new("test".to_string(), GoalPriority::Normal);
            goal.state = state;
            assert_eq!(goal.is_terminal(), expected_terminal);
        }
    }

    #[test]
    fn test_goal_with_all_constraints() {
        let constraints = GoalConstraints {
            max_time: Some(chrono::Duration::minutes(30)),
            memory_limit: Some(2_147_483_648), // 2GB
            throughput_target: Some(500.0),
            accuracy_target: Some(0.95),
            energy_limit: Some(100.0),
            cost_limit: Some(1000.0),
        };

        let goal = Goal::with_constraints(
            "Complex goal".to_string(),
            GoalPriority::Critical,
            constraints.clone(),
        );

        assert_eq!(
            goal.constraints.max_time,
            Some(chrono::Duration::minutes(30))
        );
        assert_eq!(goal.constraints.memory_limit, Some(2_147_483_648));
        assert_eq!(goal.constraints.throughput_target, Some(500.0));
        assert_eq!(goal.constraints.accuracy_target, Some(0.95));
        assert_eq!(goal.constraints.energy_limit, Some(100.0));
        assert_eq!(goal.constraints.cost_limit, Some(1000.0));
    }

    #[test]
    fn test_goal_metadata_types() {
        let mut goal = Goal::new("test".to_string(), GoalPriority::Normal);

        // Test different value types
        goal.add_metadata("string_val".to_string(), serde_json::json!("text"));
        goal.add_metadata("number_val".to_string(), serde_json::json!(42));
        goal.add_metadata("float_val".to_string(), serde_json::json!(3.14));
        goal.add_metadata("bool_val".to_string(), serde_json::json!(true));
        goal.add_metadata("array_val".to_string(), serde_json::json!([1, 2, 3]));
        goal.add_metadata(
            "object_val".to_string(),
            serde_json::json!({"key": "value"}),
        );

        assert_eq!(goal.metadata.len(), 6);
        assert_eq!(
            goal.metadata.get("string_val"),
            Some(&serde_json::json!("text"))
        );
        assert_eq!(
            goal.metadata.get("number_val"),
            Some(&serde_json::json!(42))
        );
        assert_eq!(
            goal.metadata.get("float_val"),
            Some(&serde_json::json!(3.14))
        );
        assert_eq!(
            goal.metadata.get("bool_val"),
            Some(&serde_json::json!(true))
        );
    }

    #[test]
    fn test_goal_serialization() {
        let mut goal = Goal::new("Test goal".to_string(), GoalPriority::High);
        goal.add_metadata("test_key".to_string(), serde_json::json!("test_value"));

        let json = serde_json::to_string(&goal).unwrap();
        let deserialized: Goal = serde_json::from_str(&json).unwrap();

        assert_eq!(goal.id, deserialized.id);
        assert_eq!(goal.description, deserialized.description);
        assert_eq!(goal.priority, deserialized.priority);
        assert_eq!(goal.state, deserialized.state);
        assert_eq!(goal.metadata, deserialized.metadata);
    }

    #[test]
    fn test_goal_subgoal_hierarchy() {
        let parent = Goal::new("Parent".to_string(), GoalPriority::High);
        let child1 = parent.create_subgoal("Child 1".to_string());
        let child2 = parent.create_subgoal("Child 2".to_string());
        let grandchild = child1.create_subgoal("Grandchild".to_string());

        assert_eq!(child1.parent_id, Some(parent.id));
        assert_eq!(child2.parent_id, Some(parent.id));
        assert_eq!(grandchild.parent_id, Some(child1.id));

        // Priority inheritance
        assert_eq!(child1.priority, parent.priority);
        assert_eq!(grandchild.priority, child1.priority);
    }

    #[test]
    fn test_goal_state_transitions() {
        let mut goal = Goal::new("test".to_string(), GoalPriority::Normal);

        // Valid transitions
        goal.update_state(GoalState::Executing);
        assert_eq!(goal.state, GoalState::Executing);

        goal.update_state(GoalState::Suspended);
        assert_eq!(goal.state, GoalState::Suspended);

        goal.update_state(GoalState::Executing);
        goal.update_state(GoalState::Completed);
        assert_eq!(goal.state, GoalState::Completed);
        assert!(goal.completed_at.is_some());
    }

    #[test]
    fn test_goal_constraints_default() {
        let constraints = GoalConstraints::default();
        assert!(constraints.max_time.is_none());
        assert!(constraints.memory_limit.is_none());
        assert!(constraints.throughput_target.is_none());
        assert!(constraints.accuracy_target.is_none());
        assert!(constraints.energy_limit.is_none());
        assert!(constraints.cost_limit.is_none());
    }

    #[test]
    fn test_goal_template_complex_params() {
        let template = GoalTemplate {
            name: "complex_template".to_string(),
            description: "Process {count} items of type {type} with {algorithm} algorithm"
                .to_string(),
            default_priority: GoalPriority::Normal,
            default_constraints: GoalConstraints::default(),
            required_params: vec![
                "count".to_string(),
                "type".to_string(),
                "algorithm".to_string(),
            ],
            optional_params: vec!["batch_size".to_string()],
        };

        let mut params = HashMap::new();
        params.insert("count".to_string(), serde_json::json!(1000));
        params.insert("type".to_string(), serde_json::json!("images"));
        params.insert("algorithm".to_string(), serde_json::json!("CNN"));
        params.insert("batch_size".to_string(), serde_json::json!(32));

        let goal = template.instantiate(params.clone()).unwrap();
        assert_eq!(
            goal.description,
            "Process 1000 items of type images with CNN algorithm"
        );
        assert_eq!(goal.metadata.len(), 4);
    }

    #[test]
    fn test_goal_template_optional_params() {
        let template = GoalTemplate {
            name: "optional_template".to_string(),
            description: "Execute task {task}".to_string(),
            default_priority: GoalPriority::Low,
            default_constraints: GoalConstraints::default(),
            required_params: vec!["task".to_string()],
            optional_params: vec!["timeout".to_string(), "retries".to_string()],
        };

        // Only required params
        let mut params = HashMap::new();
        params.insert("task".to_string(), serde_json::json!("optimization"));

        let goal = template.instantiate(params).unwrap();
        assert_eq!(goal.description, "Execute task optimization");

        // With optional params
        let mut params = HashMap::new();
        params.insert("task".to_string(), serde_json::json!("optimization"));
        params.insert("timeout".to_string(), serde_json::json!(30));
        params.insert("retries".to_string(), serde_json::json!(3));

        let goal = template.instantiate(params).unwrap();
        assert_eq!(goal.metadata.len(), 3);
    }

    #[test]
    fn test_goal_creation_timestamp() {
        let before = chrono::Utc::now();
        let goal = Goal::new("test".to_string(), GoalPriority::Normal);
        let after = chrono::Utc::now();

        assert!(goal.created_at >= before);
        assert!(goal.created_at <= after);
        assert!(goal.updated_at >= before);
        assert!(goal.updated_at <= after);
    }

    #[test]
    fn test_goal_update_timestamp() {
        let mut goal = Goal::new("test".to_string(), GoalPriority::Normal);
        let initial_updated = goal.updated_at;

        std::thread::sleep(std::time::Duration::from_millis(10));
        goal.update_state(GoalState::Executing);

        assert!(goal.updated_at > initial_updated);
    }

    #[test]
    fn test_goal_priority_comparison() {
        let critical = Goal::new("critical".to_string(), GoalPriority::Critical);
        let high = Goal::new("high".to_string(), GoalPriority::High);
        let normal = Goal::new("normal".to_string(), GoalPriority::Normal);
        let low = Goal::new("low".to_string(), GoalPriority::Low);

        // Create a vector and sort by priority
        let mut goals = vec![normal.clone(), critical.clone(), low.clone(), high.clone()];
        goals.sort_by_key(|g| std::cmp::Reverse(g.priority));

        assert_eq!(goals[0].priority, GoalPriority::Critical);
        assert_eq!(goals[1].priority, GoalPriority::High);
        assert_eq!(goals[2].priority, GoalPriority::Normal);
        assert_eq!(goals[3].priority, GoalPriority::Low);
    }

    #[test]
    fn test_goal_constraints_serialization() {
        let constraints = GoalConstraints {
            max_time: Some(chrono::Duration::hours(2)),
            memory_limit: Some(4_294_967_296), // 4GB
            throughput_target: Some(1000.0),
            accuracy_target: Some(0.99),
            energy_limit: Some(500.0),
            cost_limit: Some(10000.0),
        };

        let json = serde_json::to_string(&constraints).unwrap();
        let deserialized: GoalConstraints = serde_json::from_str(&json).unwrap();

        assert_eq!(constraints.max_time, deserialized.max_time);
        assert_eq!(constraints.memory_limit, deserialized.memory_limit);
        assert_eq!(
            constraints.throughput_target,
            deserialized.throughput_target
        );
        assert_eq!(constraints.accuracy_target, deserialized.accuracy_target);
        assert_eq!(constraints.energy_limit, deserialized.energy_limit);
        assert_eq!(constraints.cost_limit, deserialized.cost_limit);
    }

    #[test]
    fn test_goal_state_edge_cases() {
        let mut goal = Goal::new("test".to_string(), GoalPriority::Normal);

        // Multiple updates to completed_at
        goal.update_state(GoalState::Completed);
        let first_completed = goal.completed_at;

        std::thread::sleep(std::time::Duration::from_millis(10));
        goal.update_state(GoalState::Failed);

        // completed_at should not change when transitioning between terminal states
        assert_eq!(goal.completed_at, first_completed);
    }

    #[test]
    fn test_goal_metadata_overwrite() {
        let mut goal = Goal::new("test".to_string(), GoalPriority::Normal);

        goal.add_metadata("key".to_string(), serde_json::json!("value1"));
        assert_eq!(goal.metadata.get("key"), Some(&serde_json::json!("value1")));

        goal.add_metadata("key".to_string(), serde_json::json!("value2"));
        assert_eq!(goal.metadata.get("key"), Some(&serde_json::json!("value2")));
        assert_eq!(goal.metadata.len(), 1);
    }

    #[test]
    fn test_goal_template_error_messages() {
        let template = GoalTemplate {
            name: "test".to_string(),
            description: "Test {a} and {b}".to_string(),
            default_priority: GoalPriority::Normal,
            default_constraints: GoalConstraints::default(),
            required_params: vec!["a".to_string(), "b".to_string()],
            optional_params: vec![],
        };

        let mut params = HashMap::new();
        params.insert("a".to_string(), serde_json::json!("value"));
        // Missing 'b'

        let result = template.instantiate(params);
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.contains("Missing required parameter: b"));
    }

    #[test]
    fn test_goal_id_uniqueness() {
        use std::collections::HashSet;
        let mut ids = HashSet::new();

        // Generate many IDs and ensure uniqueness
        for _ in 0..1000 {
            let id = GoalId::new();
            assert!(ids.insert(id));
        }
    }

    #[test]
    fn test_goal_memory_efficiency() {
        use std::mem::size_of;

        // Ensure Goal struct is reasonably sized
        assert!(size_of::<Goal>() < 512); // Should be compact
        assert!(size_of::<GoalId>() == 16); // UUID size
        assert!(size_of::<GoalPriority>() == 1); // Enum with few variants
        assert!(size_of::<GoalState>() == 1); // Enum with few variants
    }
}
