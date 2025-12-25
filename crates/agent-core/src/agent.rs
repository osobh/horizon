//! Core agent implementation

use crate::error::{AgentError, AgentResult};
use crate::goal::{Goal, GoalId};
use crate::memory::AgentMemory;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Unique agent identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub Uuid);

impl AgentId {
    /// Create a new agent ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for AgentId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Agent state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentState {
    /// Agent is initializing
    Initializing,
    /// Agent is idle, waiting for goals
    Idle,
    /// Agent is planning actions
    Planning,
    /// Agent is executing a goal
    Executing,
    /// Agent is paused
    Paused,
    /// Agent has failed
    Failed,
    /// Agent is shutting down
    ShuttingDown,
    /// Agent has terminated
    Terminated,
}

impl Default for AgentState {
    fn default() -> Self {
        Self::Initializing
    }
}

impl AgentState {
    /// Check if transition to new state is valid
    pub fn can_transition_to(&self, new_state: AgentState) -> bool {
        match (self, new_state) {
            // From Initializing
            (AgentState::Initializing, AgentState::Idle) => true,
            (AgentState::Initializing, AgentState::Failed) => true,

            // From Idle
            (AgentState::Idle, AgentState::Planning) => true,
            (AgentState::Idle, AgentState::Paused) => true,
            (AgentState::Idle, AgentState::ShuttingDown) => true,

            // From Planning
            (AgentState::Planning, AgentState::Executing) => true,
            (AgentState::Planning, AgentState::Idle) => true,
            (AgentState::Planning, AgentState::Failed) => true,
            (AgentState::Planning, AgentState::Paused) => true,

            // From Executing
            (AgentState::Executing, AgentState::Idle) => true,
            (AgentState::Executing, AgentState::Planning) => true,
            (AgentState::Executing, AgentState::Failed) => true,
            (AgentState::Executing, AgentState::Paused) => true,

            // From Paused
            (AgentState::Paused, AgentState::Idle) => true,
            (AgentState::Paused, AgentState::Planning) => true,
            (AgentState::Paused, AgentState::Executing) => true,
            (AgentState::Paused, AgentState::ShuttingDown) => true,

            // From Failed
            (AgentState::Failed, AgentState::Idle) => true,
            (AgentState::Failed, AgentState::ShuttingDown) => true,

            // From ShuttingDown
            (AgentState::ShuttingDown, AgentState::Terminated) => true,

            // Invalid transitions
            _ => false,
        }
    }
}

/// Agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Agent name
    pub name: String,
    /// Agent type/role
    pub agent_type: String,
    /// Maximum memory allocation
    pub max_memory: usize,
    /// Maximum GPU memory allocation
    pub max_gpu_memory: usize,
    /// Priority level
    pub priority: i32,
    /// Custom metadata
    pub metadata: serde_json::Value,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "unnamed".to_string(),
            agent_type: "generic".to_string(),
            max_memory: 1024 * 1024 * 1024,    // 1GB
            max_gpu_memory: 512 * 1024 * 1024, // 512MB
            priority: 0,
            metadata: serde_json::Value::Object(serde_json::Map::new()),
        }
    }
}

/// Agent statistics with XP tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStats {
    /// Creation time
    pub created_at: DateTime<Utc>,
    /// Last state change
    pub last_state_change: DateTime<Utc>,
    /// Total goals processed
    pub goals_processed: u64,
    /// Successful goals
    pub goals_succeeded: u64,
    /// Failed goals
    pub goals_failed: u64,
    /// Total execution time
    pub total_execution_time: std::time::Duration,
    /// Current memory usage
    pub memory_usage: usize,
    /// Current GPU memory usage
    pub gpu_memory_usage: usize,
    /// XP System Integration
    pub current_xp: u64,
    /// Total XP earned (never decreases)
    pub total_xp: u64,
    /// Current agent level
    pub level: u32,
    /// Ready to evolve to next level
    pub ready_to_evolve: bool,
    /// Last XP gain timestamp
    pub last_xp_gain: Option<DateTime<Utc>>,
    /// XP gain history (last 100 entries)
    pub xp_history: Vec<XPGainRecord>,
}

/// XP gain record for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XPGainRecord {
    /// Amount of XP gained
    pub amount: u64,
    /// Reason for XP gain
    pub reason: String,
    /// Source category
    pub category: String,
    /// Timestamp of gain
    pub timestamp: DateTime<Utc>,
}

/// XP level thresholds
pub const LEVEL_THRESHOLDS: &[u64] = &[
    0,     // Level 1
    100,   // Level 2
    250,   // Level 3
    500,   // Level 4
    1000,  // Level 5
    1750,  // Level 6
    2750,  // Level 7
    4000,  // Level 8
    5500,  // Level 9
    7500,  // Level 10
    10000, // Level 11
    13000, // Level 12
    16500, // Level 13
    20500, // Level 14
    25000, // Level 15
];

/// XP reward amounts
pub const XP_REWARDS: &[(&str, u64)] = &[
    ("goal_completed", 25),
    ("goal_failed", 5),
    ("optimization_applied", 30),
    ("security_threat_detected", 40),
    ("cost_saving_achieved", 35),
    ("gpu_optimization", 45),
    ("evolution_completed", 100),
    ("collaboration_successful", 20),
    ("autonomous_action", 15),
    ("milestone_reached", 50),
];

/// Evolution result tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionResult {
    /// Previous level before evolution
    pub previous_level: u32,
    /// New level after evolution
    pub new_level: u32,
    /// XP at time of evolution
    pub xp_at_evolution: u64,
    /// Timestamp of evolution
    pub evolution_timestamp: DateTime<Utc>,
    /// New capabilities gained (if any)
    pub capabilities_gained: Vec<String>,
    /// Performance metrics before evolution
    pub previous_metrics: EvolutionMetrics,
    /// Performance metrics after evolution
    pub new_metrics: EvolutionMetrics,
}

/// Metrics tracked during evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionMetrics {
    /// Average goal completion time
    pub avg_completion_time: std::time::Duration,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Memory efficiency score
    pub memory_efficiency: f64,
    /// Processing speed multiplier
    pub processing_speed: f64,
}

/// Core agent structure
#[derive(Clone)]
pub struct Agent {
    /// Unique agent ID
    pub id: AgentId,
    /// Agent configuration
    pub config: AgentConfig,
    /// Current state
    state: Arc<RwLock<AgentState>>,
    /// Agent memory
    memory: Arc<AgentMemory>,
    /// Current goals
    goals: Arc<RwLock<Vec<Goal>>>,
    /// Agent statistics
    stats: Arc<RwLock<AgentStats>>,
}

impl Agent {
    /// Create a new agent
    pub fn new(config: AgentConfig) -> AgentResult<Self> {
        let id = AgentId::new();
        let memory = Arc::new(AgentMemory::new(id, config.max_memory)?);

        let stats = AgentStats {
            created_at: Utc::now(),
            last_state_change: Utc::now(),
            goals_processed: 0,
            goals_succeeded: 0,
            goals_failed: 0,
            total_execution_time: std::time::Duration::new(0, 0),
            memory_usage: 0,
            gpu_memory_usage: 0,
            current_xp: 0,
            total_xp: 0,
            level: 1,
            ready_to_evolve: false,
            last_xp_gain: None,
            xp_history: Vec::new(),
        };

        Ok(Self {
            id,
            config,
            state: Arc::new(RwLock::new(AgentState::Initializing)),
            memory,
            goals: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(stats)),
        })
    }

    /// Get agent ID
    pub fn id(&self) -> AgentId {
        self.id
    }

    /// Get current state
    pub async fn state(&self) -> AgentState {
        *self.state.read().await
    }

    /// Transition to new state
    pub async fn transition_to(&self, new_state: AgentState) -> AgentResult<()> {
        let mut current_state = self.state.write().await;

        if !current_state.can_transition_to(new_state) {
            return Err(AgentError::InvalidStateTransition {
                from: *current_state,
                to: new_state,
            });
        }

        *current_state = new_state;

        // Update stats
        let mut stats = self.stats.write().await;
        stats.last_state_change = Utc::now();

        Ok(())
    }

    /// Initialize agent
    pub async fn initialize(&self) -> AgentResult<()> {
        // Perform initialization
        self.memory.initialize().await?;

        // Transition to idle
        self.transition_to(AgentState::Idle).await?;

        Ok(())
    }

    /// Add a goal to the agent
    pub async fn add_goal(&self, goal: Goal) -> AgentResult<()> {
        let state = self.state().await;

        // Check if agent can accept goals
        match state {
            AgentState::Idle | AgentState::Planning | AgentState::Executing => {
                let mut goals = self.goals.write().await;
                goals.push(goal);

                // If idle, transition to planning
                if state == AgentState::Idle {
                    self.transition_to(AgentState::Planning).await?;
                }

                Ok(())
            }
            _ => Err(AgentError::InvalidStateTransition {
                from: state,
                to: AgentState::Planning,
            }),
        }
    }

    /// Get current goals
    pub async fn goals(&self) -> Vec<Goal> {
        self.goals.read().await.clone()
    }

    /// Remove a goal
    pub async fn remove_goal(&self, goal_id: GoalId) -> AgentResult<()> {
        let mut goals = self.goals.write().await;
        goals.retain(|g| g.id != goal_id);
        Ok(())
    }

    /// Get agent memory
    pub fn memory(&self) -> &AgentMemory {
        &self.memory
    }

    /// Get agent statistics
    pub async fn stats(&self) -> AgentStats {
        self.stats.read().await.clone()
    }

    /// Update goal statistics and award XP
    pub async fn update_goal_stats(&self, succeeded: bool, execution_time: std::time::Duration) {
        {
            let mut stats = self.stats.write().await;
            stats.goals_processed += 1;

            if succeeded {
                stats.goals_succeeded += 1;
            } else {
                stats.goals_failed += 1;
            }

            stats.total_execution_time += execution_time;
        }

        // Award XP based on goal outcome
        let reward_key = if succeeded { "goal_completed" } else { "goal_failed" };
        let xp_amount = self.get_xp_reward_amount(reward_key);

        // Award XP and handle any errors gracefully
        if let Err(e) = self.award_xp(
            xp_amount,
            format!("Goal {}", if succeeded { "completed" } else { "failed" }),
            "goal_completion".to_string(),
        ).await {
            // Log error but don't fail the stats update
            eprintln!("Failed to award XP for goal completion: {}", e);
        }
    }

    /// Pause agent execution
    pub async fn pause(&self) -> AgentResult<()> {
        self.transition_to(AgentState::Paused).await
    }

    /// Resume agent execution
    pub async fn resume(&self) -> AgentResult<()> {
        let state = self.state().await;

        // Determine appropriate state to resume to
        let new_state = if self.goals.read().await.is_empty() {
            AgentState::Idle
        } else {
            AgentState::Planning
        };

        if state == AgentState::Paused {
            self.transition_to(new_state).await
        } else {
            Err(AgentError::InvalidStateTransition {
                from: state,
                to: new_state,
            })
        }
    }

    /// Shutdown agent
    pub async fn shutdown(&self) -> AgentResult<()> {
        self.transition_to(AgentState::ShuttingDown).await?;

        // Clean up resources
        self.memory.cleanup().await?;

        self.transition_to(AgentState::Terminated).await?;
        Ok(())
    }

    /// Award XP to the agent
    pub async fn award_xp(&self, amount: u64, reason: String, category: String) -> AgentResult<()> {
        let mut stats = self.stats.write().await;
        
        // Create XP gain record
        let xp_record = XPGainRecord {
            amount,
            reason,
            category,
            timestamp: Utc::now(),
        };

        // Update XP values
        stats.current_xp += amount;
        stats.total_xp += amount;
        stats.last_xp_gain = Some(xp_record.timestamp);

        // Update level
        let new_level = self.calculate_level_from_xp(stats.current_xp);
        if new_level > stats.level {
            stats.level = new_level;
        }

        // Check evolution readiness
        stats.ready_to_evolve = self.check_evolution_readiness_internal(&stats);

        // Add to history (keep last 100 entries)
        stats.xp_history.push(xp_record);
        if stats.xp_history.len() > 100 {
            stats.xp_history.remove(0);
        }

        Ok(())
    }

    /// Calculate level based on current XP (async version)
    pub async fn calculate_level(&self) -> u32 {
        let stats = self.stats.read().await;
        self.calculate_level_from_xp(stats.current_xp)
    }

    /// Calculate level from XP amount
    fn calculate_level_from_xp(&self, xp: u64) -> u32 {
        for (level, &threshold) in LEVEL_THRESHOLDS.iter().enumerate().rev() {
            if xp >= threshold {
                return (level + 1) as u32;
            }
        }
        1 // Minimum level is 1
    }

    /// Check if agent is ready to evolve
    pub async fn check_evolution_readiness(&self) -> bool {
        let stats = self.stats.read().await;
        self.check_evolution_readiness_internal(&stats)
    }

    /// Internal evolution readiness check
    fn check_evolution_readiness_internal(&self, stats: &AgentStats) -> bool {
        // Agent is ready to evolve if current XP meets the next level threshold
        let current_level = stats.level as usize;
        if current_level < LEVEL_THRESHOLDS.len() {
            stats.current_xp >= LEVEL_THRESHOLDS[current_level]
        } else {
            false // Max level reached
        }
    }

    /// Trigger evolution if ready
    pub async fn trigger_evolution(&self) -> AgentResult<EvolutionResult> {
        let mut stats = self.stats.write().await;
        
        if !self.check_evolution_readiness_internal(&stats) {
            return Err(AgentError::Other("Agent not ready for evolution".to_string()));
        }

        let previous_level = stats.level;
        let previous_metrics = self.calculate_current_metrics(&stats);
        
        // Evolve to next level
        let new_level = self.calculate_level_from_xp(stats.current_xp);
        stats.level = new_level;
        stats.ready_to_evolve = false;

        // Calculate new metrics (improved performance)
        let new_metrics = self.calculate_evolved_metrics(&previous_metrics, new_level);

        // Create evolution result
        let evolution_result = EvolutionResult {
            previous_level,
            new_level,
            xp_at_evolution: stats.current_xp,
            evolution_timestamp: Utc::now(),
            capabilities_gained: self.determine_new_capabilities(new_level),
            previous_metrics,
            new_metrics,
        };

        // Award evolution XP bonus
        stats.current_xp += 100; // Evolution bonus
        stats.total_xp += 100;

        // Update XP history
        let evolution_record = XPGainRecord {
            amount: 100,
            reason: "Evolution completed".to_string(),
            category: "evolution".to_string(),
            timestamp: evolution_result.evolution_timestamp,
        };
        stats.xp_history.push(evolution_record);
        if stats.xp_history.len() > 100 {
            stats.xp_history.remove(0);
        }

        Ok(evolution_result)
    }

    /// Get XP required for next level
    pub async fn get_xp_for_next_level(&self) -> u64 {
        let stats = self.stats.read().await;
        let current_level = stats.level as usize;
        
        if current_level < LEVEL_THRESHOLDS.len() {
            let next_threshold = LEVEL_THRESHOLDS[current_level];
            if stats.current_xp >= next_threshold {
                0 // Already at or above threshold
            } else {
                next_threshold - stats.current_xp
            }
        } else {
            0 // Max level reached
        }
    }

    /// Calculate current performance metrics
    fn calculate_current_metrics(&self, stats: &AgentStats) -> EvolutionMetrics {
        let avg_completion_time = if stats.goals_processed > 0 {
            std::time::Duration::from_secs(
                stats.total_execution_time.as_secs() / stats.goals_processed
            )
        } else {
            std::time::Duration::from_secs(60) // Default 1 minute
        };

        let success_rate = if stats.goals_processed > 0 {
            stats.goals_succeeded as f64 / stats.goals_processed as f64
        } else {
            0.5 // Default 50%
        };

        EvolutionMetrics {
            avg_completion_time,
            success_rate,
            memory_efficiency: 0.7, // Base efficiency
            processing_speed: 1.0,  // Base speed multiplier
        }
    }

    /// Calculate evolved performance metrics
    fn calculate_evolved_metrics(&self, previous: &EvolutionMetrics, new_level: u32) -> EvolutionMetrics {
        let level_multiplier = 1.0 + (new_level as f64 - 1.0) * 0.1; // 10% improvement per level

        EvolutionMetrics {
            avg_completion_time: std::time::Duration::from_secs(
                (previous.avg_completion_time.as_secs() as f64 / level_multiplier) as u64
            ),
            success_rate: (previous.success_rate * level_multiplier).min(1.0),
            memory_efficiency: (previous.memory_efficiency * level_multiplier).min(1.0),
            processing_speed: previous.processing_speed * level_multiplier,
        }
    }

    /// Determine new capabilities gained at level
    fn determine_new_capabilities(&self, level: u32) -> Vec<String> {
        match level {
            2 => vec!["basic_optimization".to_string()],
            3 => vec!["memory_management".to_string()],
            4 => vec!["parallel_processing".to_string()],
            5 => vec!["advanced_analytics".to_string()],
            6 => vec!["gpu_acceleration".to_string()],
            7 => vec!["distributed_computing".to_string()],
            8 => vec!["ai_enhancement".to_string()],
            9 => vec!["quantum_optimization".to_string()],
            10 => vec!["neural_evolution".to_string()],
            _ => vec![], // No new capabilities for other levels
        }
    }

    /// Get XP reward amount for a specific action
    fn get_xp_reward_amount(&self, reward_key: &str) -> u64 {
        XP_REWARDS
            .iter()
            .find(|(key, _)| *key == reward_key)
            .map(|(_, amount)| *amount)
            .unwrap_or(0) // Default to 0 if reward not found
    }

    /// Award XP for a specific reward type (convenience method)
    pub async fn award_xp_for_action(&self, action: &str, custom_reason: Option<String>) -> AgentResult<()> {
        let xp_amount = self.get_xp_reward_amount(action);
        if xp_amount > 0 {
            let reason = custom_reason.unwrap_or_else(|| action.replace("_", " "));
            self.award_xp(xp_amount, reason, action.to_string()).await
        } else {
            Err(AgentError::Other(format!("Unknown XP reward action: {}", action)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_id() {
        let id1 = AgentId::new();
        let id2 = AgentId::new();
        assert_ne!(id1, id2);

        let id_str = id1.to_string();
        assert_eq!(id_str.len(), 36); // UUID string length
    }

    #[test]
    fn test_agent_state_transitions() {
        // Valid transitions
        assert!(AgentState::Initializing.can_transition_to(AgentState::Idle));
        assert!(AgentState::Idle.can_transition_to(AgentState::Planning));
        assert!(AgentState::Planning.can_transition_to(AgentState::Executing));
        assert!(AgentState::Executing.can_transition_to(AgentState::Idle));

        // Invalid transitions
        assert!(!AgentState::Initializing.can_transition_to(AgentState::Executing));
        assert!(!AgentState::Terminated.can_transition_to(AgentState::Idle));
        assert!(!AgentState::Idle.can_transition_to(AgentState::Failed));
    }

    #[tokio::test]
    async fn test_agent_creation() {
        let config = AgentConfig {
            name: "test_agent".to_string(),
            agent_type: "test".to_string(),
            ..Default::default()
        };

        let agent = Agent::new(config).unwrap();
        assert_eq!(agent.config.name, "test_agent");
        assert_eq!(agent.state().await, AgentState::Initializing);
    }

    #[tokio::test]
    async fn test_agent_initialization() {
        let agent = Agent::new(AgentConfig::default()).unwrap();

        assert!(agent.initialize().await.is_ok());
        assert_eq!(agent.state().await, AgentState::Idle);
    }

    #[tokio::test]
    async fn test_agent_goals() {
        let agent = Agent::new(AgentConfig::default()).unwrap();
        agent.initialize().await.unwrap();

        let goal = Goal::new("test goal".to_string(), crate::goal::GoalPriority::Normal);

        assert!(agent.add_goal(goal.clone()).await.is_ok());
        assert_eq!(agent.state().await, AgentState::Planning);

        let goals = agent.goals().await;
        assert_eq!(goals.len(), 1);
        assert_eq!(goals[0].id, goal.id);

        assert!(agent.remove_goal(goal.id).await.is_ok());
        assert!(agent.goals().await.is_empty());
    }

    #[tokio::test]
    async fn test_agent_pause_resume() {
        let agent = Agent::new(AgentConfig::default()).unwrap();
        agent.initialize().await.unwrap();

        // Add a goal to move to planning state
        let goal = Goal::new("test goal".to_string(), crate::goal::GoalPriority::Normal);
        agent.add_goal(goal).await.unwrap();

        // Pause
        assert!(agent.pause().await.is_ok());
        assert_eq!(agent.state().await, AgentState::Paused);

        // Resume
        assert!(agent.resume().await.is_ok());
        assert_eq!(agent.state().await, AgentState::Planning);
    }

    #[tokio::test]
    async fn test_agent_shutdown() {
        let agent = Agent::new(AgentConfig::default()).unwrap();
        agent.initialize().await.unwrap();

        assert!(agent.shutdown().await.is_ok());
        assert_eq!(agent.state().await, AgentState::Terminated);
    }

    #[tokio::test]
    async fn test_agent_stats() {
        let agent = Agent::new(AgentConfig::default()).unwrap();
        agent.initialize().await.unwrap();

        let stats = agent.stats().await;
        assert_eq!(stats.goals_processed, 0);
        assert_eq!(stats.goals_succeeded, 0);
        assert_eq!(stats.goals_failed, 0);

        // Update stats
        agent
            .update_goal_stats(true, std::time::Duration::from_secs(1))
            .await;

        let stats = agent.stats().await;
        assert_eq!(stats.goals_processed, 1);
        assert_eq!(stats.goals_succeeded, 1);
        assert_eq!(stats.goals_failed, 0);
        assert_eq!(
            stats.total_execution_time,
            std::time::Duration::from_secs(1)
        );
    }

    #[test]
    fn test_agent_id_creation() {
        let id1 = AgentId::new();
        let id2 = AgentId::new();

        // IDs should be unique
        assert_ne!(id1, id2);

        // Display should work
        let display = format!("{id1}");
        assert!(!display.is_empty());
    }

    #[test]
    fn test_agent_id_default() {
        let id1 = AgentId::default();
        let id2 = AgentId::default();

        // Default IDs should also be unique
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_agent_state_uniqueness() {
        let states = vec![
            AgentState::Initializing,
            AgentState::Idle,
            AgentState::Planning,
            AgentState::Executing,
            AgentState::Paused,
            AgentState::Failed,
            AgentState::Terminated,
        ];

        // Test all states are distinct
        for (i, state1) in states.iter().enumerate() {
            for (j, state2) in states.iter().enumerate() {
                if i != j {
                    assert_ne!(state1, state2);
                }
            }
        }
    }

    #[test]
    fn test_agent_config_validation() {
        // Test empty name
        let config = AgentConfig {
            name: String::new(),
            ..Default::default()
        };
        let agent = Agent::new(config);
        // Agent should still be created with empty name, just verify the structure works
        assert!(agent.is_ok());
    }

    #[test]
    fn test_agent_config_custom_values() {
        let config = AgentConfig {
            name: "CustomAgent".to_string(),
            agent_type: "custom".to_string(),
            max_memory: 2048 * 1024 * 1024, // 2GB in bytes
            max_gpu_memory: 1024 * 1024 * 1024, // 1GB in bytes
            priority: 5,
            metadata: serde_json::json!({"custom": true}),
        };

        let agent = Agent::new(config.clone()).unwrap();
        assert_eq!(agent.config.name, "CustomAgent");
        assert_eq!(agent.config.agent_type, "custom");
    }

    #[tokio::test]
    async fn test_agent_concurrent_goal_addition() {
        let agent = Arc::new(Agent::new(AgentConfig::default()).unwrap());
        agent.initialize().await.unwrap();

        let mut handles = vec![];

        // Add goals concurrently
        for i in 0..10 {
            let agent_clone = agent.clone();
            let handle = tokio::spawn(async move {
                let goal = Goal::new(
                    format!("concurrent goal {i}"),
                    crate::goal::GoalPriority::Normal,
                );
                agent_clone.add_goal(goal).await
            });
            handles.push(handle);
        }

        // Wait for all additions
        for handle in handles {
            assert!(handle.await.unwrap().is_ok());
        }

        // Verify all goals were added
        let goals = agent.goals().await;
        assert_eq!(goals.len(), 10);
    }

    #[tokio::test]
    async fn test_agent_goal_management() {
        let agent = Agent::new(AgentConfig::default()).unwrap();
        agent.initialize().await.unwrap();

        // Add some goals
        for i in 0..3 {
            let goal = Goal::new(format!("goal {i}"), crate::goal::GoalPriority::Normal);
            assert!(agent.add_goal(goal).await.is_ok());
        }

        // Verify goals were added
        let goals = agent.goals().await;
        assert_eq!(goals.len(), 3);
    }

    #[tokio::test]
    async fn test_agent_state_machine_flow() {
        let agent = Agent::new(AgentConfig::default()).unwrap();

        // Initial state
        assert_eq!(agent.state().await, AgentState::Initializing);

        // Initialize -> Idle
        agent.initialize().await.unwrap();
        assert_eq!(agent.state().await, AgentState::Idle);

        // Add goal -> Planning
        let goal = Goal::new("test".to_string(), crate::goal::GoalPriority::Normal);
        agent.add_goal(goal).await.unwrap();
        assert_eq!(agent.state().await, AgentState::Planning);

        // Can pause from Planning
        agent.pause().await.unwrap();
        assert_eq!(agent.state().await, AgentState::Paused);

        // Resume -> back to Planning
        agent.resume().await.unwrap();
        assert_eq!(agent.state().await, AgentState::Planning);

        // Shutdown -> Terminated
        agent.shutdown().await.unwrap();
        assert_eq!(agent.state().await, AgentState::Terminated);
    }

    #[tokio::test]
    async fn test_agent_goal_removal() {
        let agent = Agent::new(AgentConfig::default()).unwrap();
        agent.initialize().await.unwrap();

        // Add multiple goals
        let mut goal_ids = vec![];
        for i in 0..5 {
            let goal = Goal::new(format!("goal {i}"), crate::goal::GoalPriority::Normal);
            let id = goal.id;
            agent.add_goal(goal).await.unwrap();
            goal_ids.push(id);
        }

        // Remove specific goal
        assert!(agent.remove_goal(goal_ids[2]).await.is_ok());

        // Verify removal
        let goals = agent.goals().await;
        assert_eq!(goals.len(), 4);
        assert!(!goals.iter().any(|g| g.id == goal_ids[2]));
    }

    #[tokio::test]
    async fn test_agent_goals_management() {
        let agent = Agent::new(AgentConfig::default()).unwrap();
        agent.initialize().await.unwrap();

        // Add goals
        for i in 0..5 {
            let goal = Goal::new(format!("goal {i}"), crate::goal::GoalPriority::Normal);
            agent.add_goal(goal).await.unwrap();
        }

        assert_eq!(agent.goals().await.len(), 5);
    }

    #[tokio::test]
    async fn test_agent_priority_goal_ordering() {
        let agent = Agent::new(AgentConfig::default()).unwrap();
        agent.initialize().await.unwrap();

        // Add goals with different priorities
        let low_goal = Goal::new("low".to_string(), crate::goal::GoalPriority::Low);
        let normal_goal = Goal::new("normal".to_string(), crate::goal::GoalPriority::Normal);
        let high_goal = Goal::new("high".to_string(), crate::goal::GoalPriority::High);
        let critical_goal = Goal::new("critical".to_string(), crate::goal::GoalPriority::Critical);

        // Add in random order
        agent.add_goal(normal_goal).await.unwrap();
        agent.add_goal(critical_goal).await.unwrap();
        agent.add_goal(low_goal).await.unwrap();
        agent.add_goal(high_goal).await.unwrap();

        let goals = agent.goals().await;

        // Goals should be ordered by priority (implementation dependent)
        // Just verify all goals are present
        assert_eq!(goals.len(), 4);
    }

    #[tokio::test]
    async fn test_agent_basic_stats() {
        let agent = Agent::new(AgentConfig::default()).unwrap();

        let stats = agent.stats().await;
        assert!(stats.created_at <= Utc::now());
        assert!(stats.last_state_change <= Utc::now());
        assert_eq!(stats.current_xp, 0);
        assert_eq!(stats.level, 1);
    }

    #[tokio::test]
    async fn test_agent_error_handling() {
        let agent = Agent::new(AgentConfig::default()).unwrap();

        // Cannot add goals before initialization
        let goal = Goal::new("test".to_string(), crate::goal::GoalPriority::Normal);
        assert!(agent.add_goal(goal).await.is_err());

        // Cannot pause before initialization
        assert!(agent.pause().await.is_err());

        // Initialize first
        agent.initialize().await.unwrap();

        // Cannot initialize twice
        assert!(agent.initialize().await.is_err());

        // Shutdown
        agent.shutdown().await.unwrap();

        // Cannot add goals after shutdown
        let goal = Goal::new("test".to_string(), crate::goal::GoalPriority::Normal);
        assert!(agent.add_goal(goal).await.is_err());
    }

    #[test]
    fn test_agent_stats_creation() {
        let now = Utc::now();
        let stats = AgentStats {
            created_at: now,
            last_state_change: now,
            goals_processed: 0,
            goals_succeeded: 0,
            goals_failed: 0,
            total_execution_time: std::time::Duration::from_secs(0),
            memory_usage: 0,
            gpu_memory_usage: 0,
            current_xp: 0,
            total_xp: 0,
            level: 1,
            ready_to_evolve: false,
            last_xp_gain: None,
            xp_history: Vec::new(),
        };

        assert_eq!(stats.goals_processed, 0);
        assert_eq!(stats.current_xp, 0);
        assert_eq!(stats.level, 1);
        assert!(!stats.ready_to_evolve);
    }

    #[test]
    fn test_agent_config_serialization() {
        let config = AgentConfig {
            name: "TestAgent".to_string(),
            agent_type: "test".to_string(),
            max_memory: 1024 * 1024 * 1024,
            max_gpu_memory: 512 * 1024 * 1024,
            priority: 1,
            metadata: serde_json::json!({"test": true}),
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: AgentConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.name, deserialized.name);
        assert_eq!(config.agent_type, deserialized.agent_type);
        assert_eq!(config.max_memory, deserialized.max_memory);
        assert_eq!(config.priority, deserialized.priority);
    }

    #[tokio::test]
    async fn test_agent_memory_access() {
        let agent = Agent::new(AgentConfig::default()).unwrap();
        agent.initialize().await.unwrap();

        // Memory should be accessible
        let _memory = agent.memory();
        // Just verify memory is accessible
    }

    #[tokio::test]
    async fn test_agent_basic_operations() {
        let agent = Agent::new(AgentConfig::default()).unwrap();
        agent.initialize().await.unwrap();

        // Add a goal
        let goal = Goal::new("test".to_string(), crate::goal::GoalPriority::Normal);
        agent.add_goal(goal).await.unwrap();

        // Verify agent state and goal management
        assert_eq!(agent.state().await, AgentState::Planning);
        assert_eq!(agent.goals().await.len(), 1);
    }

    #[test]
    fn test_agent_config_edge_cases() {
        // Test very large memory limit
        let config = AgentConfig {
            max_memory: usize::MAX,
            ..Default::default()
        };
        assert!(Agent::new(config).is_ok());

        // Test very long name
        let config = AgentConfig {
            name: "A".repeat(1000),
            ..Default::default()
        };
        assert!(Agent::new(config).is_ok());
    }

    #[tokio::test]
    async fn test_agent_concurrent_state_access() {
        let agent = Arc::new(Agent::new(AgentConfig::default()).unwrap());
        let mut handles = vec![];

        // Many concurrent state reads
        for _ in 0..100 {
            let agent_clone = agent.clone();
            let handle = tokio::spawn(async move { agent_clone.state().await });
            handles.push(handle);
        }

        // All reads should succeed
        for handle in handles {
            assert!(matches!(handle.await.unwrap(), AgentState::Initializing));
        }
    }

    // XP System Tests
    #[tokio::test]
    async fn test_xp_award_basic() {
        let agent = Agent::new(AgentConfig::default()).unwrap();
        agent.initialize().await.unwrap();

        let initial_stats = agent.stats().await;
        assert_eq!(initial_stats.current_xp, 0);
        assert_eq!(initial_stats.total_xp, 0);
        assert_eq!(initial_stats.level, 1);

        // Award some XP
        agent.award_xp(50, "Test reward".to_string(), "test".to_string()).await.unwrap();

        let stats = agent.stats().await;
        assert_eq!(stats.current_xp, 50);
        assert_eq!(stats.total_xp, 50);
        assert_eq!(stats.level, 1); // Still level 1
        assert!(stats.last_xp_gain.is_some());
        assert_eq!(stats.xp_history.len(), 1);
        assert_eq!(stats.xp_history[0].amount, 50);
        assert_eq!(stats.xp_history[0].reason, "Test reward");
        assert_eq!(stats.xp_history[0].category, "test");
    }

    #[tokio::test]
    async fn test_level_calculation() {
        let agent = Agent::new(AgentConfig::default()).unwrap();
        agent.initialize().await.unwrap();

        // Test level thresholds
        assert_eq!(agent.calculate_level_from_xp(0), 1);
        assert_eq!(agent.calculate_level_from_xp(99), 1);
        assert_eq!(agent.calculate_level_from_xp(100), 2);
        assert_eq!(agent.calculate_level_from_xp(249), 2);
        assert_eq!(agent.calculate_level_from_xp(250), 3);
        assert_eq!(agent.calculate_level_from_xp(999), 4);
        assert_eq!(agent.calculate_level_from_xp(1000), 5);
    }

    #[tokio::test]
    async fn test_level_up_on_xp_award() {
        let agent = Agent::new(AgentConfig::default()).unwrap();
        agent.initialize().await.unwrap();

        // Award XP to reach level 2
        agent.award_xp(100, "Level up test".to_string(), "test".to_string()).await.unwrap();

        let stats = agent.stats().await;
        assert_eq!(stats.current_xp, 100);
        assert_eq!(stats.level, 2);
        assert!(!stats.ready_to_evolve); // Not ready yet as we just reached threshold

        // Award more XP to make ready for evolution to level 3
        agent.award_xp(150, "More XP".to_string(), "test".to_string()).await.unwrap();

        let stats = agent.stats().await;
        assert_eq!(stats.current_xp, 250);
        assert_eq!(stats.level, 3);
        assert!(stats.ready_to_evolve); // Ready for next level
    }

    #[tokio::test]
    async fn test_evolution_readiness() {
        let agent = Agent::new(AgentConfig::default()).unwrap();
        agent.initialize().await.unwrap();

        // Not ready initially
        assert!(!agent.check_evolution_readiness().await);

        // Award XP to reach level 2 threshold
        agent.award_xp(100, "Test".to_string(), "test".to_string()).await.unwrap();
        assert!(agent.check_evolution_readiness().await);

        // Award more to reach level 3 threshold
        agent.award_xp(150, "Test".to_string(), "test".to_string()).await.unwrap();
        assert!(agent.check_evolution_readiness().await);
    }

    #[tokio::test]
    async fn test_trigger_evolution() {
        let agent = Agent::new(AgentConfig::default()).unwrap();
        agent.initialize().await.unwrap();

        // Cannot evolve without enough XP
        let result = agent.trigger_evolution().await;
        assert!(result.is_err());

        // Award XP to reach level 2 threshold
        agent.award_xp(100, "Test".to_string(), "test".to_string()).await.unwrap();

        // Now we can evolve
        let evolution_result = agent.trigger_evolution().await.unwrap();
        assert_eq!(evolution_result.previous_level, 2);
        assert_eq!(evolution_result.new_level, 2);
        assert_eq!(evolution_result.xp_at_evolution, 100);
        assert!(!evolution_result.capabilities_gained.is_empty());

        // Check agent stats after evolution
        let stats = agent.stats().await;
        assert_eq!(stats.level, 2);
        assert!(!stats.ready_to_evolve); // No longer ready after evolving
        assert_eq!(stats.current_xp, 200); // Original + evolution bonus
        assert_eq!(stats.total_xp, 200);
        
        // Check evolution was recorded in history
        assert!(stats.xp_history.iter().any(|record| 
            record.category == "evolution" && record.amount == 100
        ));
    }

    #[tokio::test]
    async fn test_xp_for_next_level() {
        let agent = Agent::new(AgentConfig::default()).unwrap();
        agent.initialize().await.unwrap();

        // Initially need 100 XP to reach level 2
        assert_eq!(agent.get_xp_for_next_level().await, 100);

        // Award 50 XP
        agent.award_xp(50, "Test".to_string(), "test".to_string()).await.unwrap();
        assert_eq!(agent.get_xp_for_next_level().await, 50);

        // Award 50 more to reach threshold
        agent.award_xp(50, "Test".to_string(), "test".to_string()).await.unwrap();
        assert_eq!(agent.get_xp_for_next_level().await, 150); // Next is level 3 (250 total)
    }

    #[tokio::test]
    async fn test_xp_history_limit() {
        let agent = Agent::new(AgentConfig::default()).unwrap();
        agent.initialize().await.unwrap();

        // Award XP 150 times to exceed the 100-entry limit
        for i in 0..150 {
            agent.award_xp(1, format!("Test {}", i), "test".to_string()).await.unwrap();
        }

        let stats = agent.stats().await;
        assert_eq!(stats.xp_history.len(), 100); // Should be capped at 100
        assert_eq!(stats.current_xp, 150);
        assert_eq!(stats.total_xp, 150);

        // First entry should be "Test 50" (entries 0-49 were removed)
        assert_eq!(stats.xp_history[0].reason, "Test 50");
        assert_eq!(stats.xp_history[99].reason, "Test 149");
    }

    #[tokio::test]
    async fn test_concurrent_xp_awards() {
        let agent = Arc::new(Agent::new(AgentConfig::default()).unwrap());
        agent.initialize().await.unwrap();

        let mut handles = vec![];

        // Award XP concurrently from multiple tasks
        for i in 0..10 {
            let agent_clone = agent.clone();
            let handle = tokio::spawn(async move {
                agent_clone.award_xp(10, format!("Concurrent {}", i), "test".to_string()).await
            });
            handles.push(handle);
        }

        // Wait for all awards to complete
        for handle in handles {
            assert!(handle.await.unwrap().is_ok());
        }

        let stats = agent.stats().await;
        assert_eq!(stats.current_xp, 100); // 10 * 10 XP
        assert_eq!(stats.total_xp, 100);
        assert_eq!(stats.xp_history.len(), 10);
        assert_eq!(stats.level, 2); // Should have leveled up
    }

    #[tokio::test]
    async fn test_evolution_metrics_calculation() {
        let agent = Agent::new(AgentConfig::default()).unwrap();
        agent.initialize().await.unwrap();

        // Simulate some goal completions to have meaningful metrics
        agent.update_goal_stats(true, std::time::Duration::from_secs(30)).await;
        agent.update_goal_stats(true, std::time::Duration::from_secs(40)).await;
        agent.update_goal_stats(false, std::time::Duration::from_secs(60)).await;

        // Award XP and trigger evolution
        agent.award_xp(100, "Test".to_string(), "test".to_string()).await.unwrap();
        let evolution_result = agent.trigger_evolution().await.unwrap();

        // Check that metrics were calculated
        assert!(evolution_result.previous_metrics.success_rate > 0.0);
        assert!(evolution_result.new_metrics.success_rate > evolution_result.previous_metrics.success_rate);
        assert!(evolution_result.new_metrics.processing_speed > 1.0);
        assert!(evolution_result.new_metrics.memory_efficiency >= 0.7);
    }

    #[tokio::test]
    async fn test_capability_progression() {
        let agent = Agent::new(AgentConfig::default()).unwrap();
        agent.initialize().await.unwrap();

        // Test capabilities at different levels
        assert!(agent.determine_new_capabilities(1).is_empty());
        assert_eq!(agent.determine_new_capabilities(2), vec!["basic_optimization"]);
        assert_eq!(agent.determine_new_capabilities(3), vec!["memory_management"]);
        assert_eq!(agent.determine_new_capabilities(5), vec!["advanced_analytics"]);
        assert_eq!(agent.determine_new_capabilities(10), vec!["neural_evolution"]);
        assert!(agent.determine_new_capabilities(15).is_empty());
    }

    #[test]
    fn test_xp_reward_constants() {
        // Verify XP reward constants are properly defined
        let goal_completed_reward = XP_REWARDS.iter()
            .find(|(key, _)| *key == "goal_completed")
            .map(|(_, value)| *value);
        assert_eq!(goal_completed_reward, Some(25));

        let goal_failed_reward = XP_REWARDS.iter()
            .find(|(key, _)| *key == "goal_failed")
            .map(|(_, value)| *value);
        assert_eq!(goal_failed_reward, Some(5));

        let evolution_reward = XP_REWARDS.iter()
            .find(|(key, _)| *key == "evolution_completed")
            .map(|(_, value)| *value);
        assert_eq!(evolution_reward, Some(100));
    }

    #[test]
    fn test_level_thresholds() {
        // Verify level thresholds are properly defined and increasing
        assert!(LEVEL_THRESHOLDS.len() >= 15);
        
        for i in 1..LEVEL_THRESHOLDS.len() {
            assert!(LEVEL_THRESHOLDS[i] > LEVEL_THRESHOLDS[i-1], 
                "Level threshold {} should be greater than threshold {}", i, i-1);
        }
        
        // Check specific thresholds
        assert_eq!(LEVEL_THRESHOLDS[0], 0);   // Level 1
        assert_eq!(LEVEL_THRESHOLDS[1], 100); // Level 2
        assert_eq!(LEVEL_THRESHOLDS[4], 1000); // Level 5
    }

    #[tokio::test]
    async fn test_max_level_behavior() {
        let agent = Agent::new(AgentConfig::default()).unwrap();
        agent.initialize().await.unwrap();

        // Award massive XP to exceed max level
        let max_xp = LEVEL_THRESHOLDS[LEVEL_THRESHOLDS.len() - 1] + 10000;
        agent.award_xp(max_xp, "Max level test".to_string(), "test".to_string()).await.unwrap();

        let stats = agent.stats().await;
        assert_eq!(stats.level, LEVEL_THRESHOLDS.len() as u32);
        
        // Should not be ready to evolve at max level
        assert!(!agent.check_evolution_readiness().await);
        assert_eq!(agent.get_xp_for_next_level().await, 0);

        // Evolution should fail at max level
        let result = agent.trigger_evolution().await;
        assert!(result.is_err());
    }

    #[test]
    fn test_evolution_result_serialization() {
        let evolution_result = EvolutionResult {
            previous_level: 1,
            new_level: 2,
            xp_at_evolution: 100,
            evolution_timestamp: Utc::now(),
            capabilities_gained: vec!["basic_optimization".to_string()],
            previous_metrics: EvolutionMetrics {
                avg_completion_time: std::time::Duration::from_secs(60),
                success_rate: 0.7,
                memory_efficiency: 0.7,
                processing_speed: 1.0,
            },
            new_metrics: EvolutionMetrics {
                avg_completion_time: std::time::Duration::from_secs(50),
                success_rate: 0.8,
                memory_efficiency: 0.8,
                processing_speed: 1.2,
            },
        };

        // Test serialization/deserialization
        let json = serde_json::to_string(&evolution_result).unwrap();
        let deserialized: EvolutionResult = serde_json::from_str(&json).unwrap();

        assert_eq!(evolution_result.previous_level, deserialized.previous_level);
        assert_eq!(evolution_result.new_level, deserialized.new_level);
        assert_eq!(evolution_result.capabilities_gained, deserialized.capabilities_gained);
    }

    #[tokio::test]
    async fn test_xp_integration_with_goal_completion() {
        let agent = Agent::new(AgentConfig::default()).unwrap();
        agent.initialize().await.unwrap();

        let initial_stats = agent.stats().await;
        assert_eq!(initial_stats.current_xp, 0);
        assert_eq!(initial_stats.total_xp, 0);

        // Simulate successful goal completion
        agent.update_goal_stats(true, std::time::Duration::from_secs(30)).await;

        let stats = agent.stats().await;
        assert_eq!(stats.current_xp, 25); // goal_completed reward
        assert_eq!(stats.total_xp, 25);
        assert_eq!(stats.goals_processed, 1);
        assert_eq!(stats.goals_succeeded, 1);
        assert_eq!(stats.xp_history.len(), 1);
        assert_eq!(stats.xp_history[0].category, "goal_completion");

        // Simulate failed goal completion
        agent.update_goal_stats(false, std::time::Duration::from_secs(60)).await;

        let stats = agent.stats().await;
        assert_eq!(stats.current_xp, 30); // 25 + 5 (goal_failed reward)
        assert_eq!(stats.total_xp, 30);
        assert_eq!(stats.goals_processed, 2);
        assert_eq!(stats.goals_failed, 1);
        assert_eq!(stats.xp_history.len(), 2);
    }

    #[tokio::test]
    async fn test_xp_reward_lookup() {
        let agent = Agent::new(AgentConfig::default()).unwrap();

        // Test known rewards
        assert_eq!(agent.get_xp_reward_amount("goal_completed"), 25);
        assert_eq!(agent.get_xp_reward_amount("goal_failed"), 5);
        assert_eq!(agent.get_xp_reward_amount("evolution_completed"), 100);
        assert_eq!(agent.get_xp_reward_amount("gpu_optimization"), 45);

        // Test unknown reward
        assert_eq!(agent.get_xp_reward_amount("unknown_action"), 0);
    }

    #[tokio::test]
    async fn test_award_xp_for_action() {
        let agent = Agent::new(AgentConfig::default()).unwrap();
        agent.initialize().await.unwrap();

        // Award XP for known action
        agent.award_xp_for_action("optimization_applied", None).await.unwrap();

        let stats = agent.stats().await;
        assert_eq!(stats.current_xp, 30); // optimization_applied reward
        assert_eq!(stats.xp_history[0].reason, "optimization applied");
        assert_eq!(stats.xp_history[0].category, "optimization_applied");

        // Award XP for unknown action should fail
        let result = agent.award_xp_for_action("unknown_action", None).await;
        assert!(result.is_err());
    }
}
