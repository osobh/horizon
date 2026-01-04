//! Eventual consistency management

use crate::error::{GlobalKnowledgeGraphError, GlobalKnowledgeGraphResult};
use async_trait::async_trait;
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::interval;
use uuid::Uuid;

/// Consistency configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyConfig {
    /// Consistency level
    pub consistency_level: ConsistencyLevel,
    /// Conflict detection window in milliseconds
    pub conflict_window_ms: u64,
    /// Convergence check interval
    pub convergence_interval: Duration,
    /// Maximum divergence time allowed
    pub max_divergence_ms: u64,
    /// Enable auto-resolution
    pub enable_auto_resolution: bool,
    /// Vector clock pruning interval
    pub pruning_interval: Duration,
}

impl Default for ConsistencyConfig {
    fn default() -> Self {
        Self {
            consistency_level: ConsistencyLevel::Eventual,
            conflict_window_ms: 5000,
            convergence_interval: Duration::from_secs(10),
            max_divergence_ms: 30000,
            enable_auto_resolution: true,
            pruning_interval: Duration::from_secs(300),
        }
    }
}

/// Consistency level
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConsistencyLevel {
    /// Strong consistency (linearizable)
    Strong,
    /// Bounded staleness
    Bounded { max_staleness_ms: u64 },
    /// Session consistency
    Session,
    /// Eventual consistency
    Eventual,
}

/// Vector clock for distributed ordering
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VectorClock {
    /// Clock values per region
    clocks: HashMap<String, u64>,
}

impl VectorClock {
    /// Create new vector clock
    pub fn new() -> Self {
        Self {
            clocks: HashMap::new(),
        }
    }

    /// Increment clock for region
    pub fn increment(&mut self, region: &str) -> u64 {
        let counter = self.clocks.entry(region.to_string()).or_insert(0);
        *counter += 1;
        *counter
    }

    /// Get clock value for region
    pub fn get(&self, region: &str) -> u64 {
        self.clocks.get(region).copied().unwrap_or(0)
    }

    /// Merge with another vector clock
    pub fn merge(&mut self, other: &VectorClock) {
        for (region, &clock) in &other.clocks {
            let current = self.clocks.entry(region.clone()).or_insert(0);
            *current = (*current).max(clock);
        }
    }

    /// Check if this clock is concurrent with another
    pub fn is_concurrent(&self, other: &VectorClock) -> bool {
        !self.happens_before(other) && !other.happens_before(self)
    }

    /// Check if this clock happens before another
    pub fn happens_before(&self, other: &VectorClock) -> bool {
        let mut at_least_one_less = false;

        for (region, &clock) in &self.clocks {
            let other_clock = other.get(region);
            if clock > other_clock {
                return false;
            }
            if clock < other_clock {
                at_least_one_less = true;
            }
        }

        // Check regions in other but not in self
        for region in other.clocks.keys() {
            if !self.clocks.contains_key(region) && other.get(region) > 0 {
                at_least_one_less = true;
            }
        }

        at_least_one_less
    }
}

/// Conflict information
#[derive(Debug, Clone)]
pub struct Conflict {
    /// Conflict ID
    pub id: String,
    /// Resource ID
    pub resource_id: String,
    /// Resource type
    pub resource_type: ResourceType,
    /// Conflicting versions
    pub versions: Vec<ConflictVersion>,
    /// Detection time
    pub detected_at: Instant,
    /// Resolution status
    pub status: ConflictStatus,
}

/// Resource type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ResourceType {
    /// Node resource
    Node,
    /// Edge resource
    Edge,
}

/// Conflicting version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictVersion {
    /// Region
    pub region: String,
    /// Version number
    pub version: u64,
    /// Vector clock
    pub vector_clock: VectorClock,
    /// Update timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Conflict status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConflictStatus {
    /// Detected but not resolved
    Detected,
    /// Being resolved
    Resolving,
    /// Resolved
    Resolved,
    /// Failed to resolve
    Failed,
}

/// Conflict resolution strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ResolutionStrategy {
    /// Last write wins
    LastWriteWins,
    /// Higher version wins
    HigherVersionWins,
    /// Custom resolver
    Custom,
    /// Manual resolution required
    Manual,
}

/// Convergence state
#[derive(Debug, Clone)]
pub struct ConvergenceState {
    /// Region
    pub region: String,
    /// Last convergence check
    pub last_check: Instant,
    /// Is converged
    pub is_converged: bool,
    /// Divergence time in ms
    pub divergence_ms: u64,
    /// Pending operations
    pub pending_ops: usize,
}

/// Consistency checker trait
#[async_trait]
pub trait ConsistencyChecker: Send + Sync {
    /// Check consistency for resource
    async fn check_consistency(
        &self,
        resource_id: &str,
        resource_type: ResourceType,
    ) -> GlobalKnowledgeGraphResult<bool>;

    /// Get vector clock for resource
    async fn get_vector_clock(&self, resource_id: &str) -> GlobalKnowledgeGraphResult<VectorClock>;

    /// Update vector clock
    async fn update_vector_clock(
        &self,
        resource_id: &str,
        clock: VectorClock,
    ) -> GlobalKnowledgeGraphResult<()>;
}

/// Consistency manager with conflict detection
pub struct ConsistencyManager {
    config: Arc<ConsistencyConfig>,
    vector_clocks: Arc<DashMap<String, VectorClock>>,
    conflicts: Arc<DashMap<String, Conflict>>,
    convergence_states: Arc<DashMap<String, ConvergenceState>>,
    resolution_strategies: Arc<DashMap<String, ResolutionStrategy>>,
    shutdown_tx: mpsc::Sender<()>,
    shutdown_rx: Arc<RwLock<Option<mpsc::Receiver<()>>>>,
}

impl ConsistencyManager {
    /// Create new consistency manager
    pub fn new(config: ConsistencyConfig) -> Self {
        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);

        Self {
            config: Arc::new(config),
            vector_clocks: Arc::new(DashMap::new()),
            conflicts: Arc::new(DashMap::new()),
            convergence_states: Arc::new(DashMap::new()),
            resolution_strategies: Arc::new(DashMap::new()),
            shutdown_tx,
            shutdown_rx: Arc::new(RwLock::new(Some(shutdown_rx))),
        }
    }

    /// Start consistency manager
    pub async fn start(&self) {
        let config = self.config.clone();
        let convergence_states = self.convergence_states.clone();
        let vector_clocks = self.vector_clocks.clone();

        // Take ownership of the receiver
        let mut shutdown_rx = self
            .shutdown_rx
            .write()
            .take()
            .expect("start called multiple times");

        tokio::spawn(async move {
            let mut convergence_interval = interval(config.convergence_interval);
            let mut pruning_interval = interval(config.pruning_interval);

            loop {
                tokio::select! {
                    _ = convergence_interval.tick() => {
                        Self::check_convergence(&convergence_states, &config).await;
                    }
                    _ = pruning_interval.tick() => {
                        Self::prune_vector_clocks(&vector_clocks).await;
                    }
                    _ = shutdown_rx.recv() => {
                        break;
                    }
                }
            }
        });
    }

    /// Stop consistency manager
    pub async fn stop(&self) {
        let _ = self.shutdown_tx.send(()).await;
    }

    /// Update resource with consistency tracking
    pub async fn update_resource(
        &self,
        resource_id: &str,
        resource_type: ResourceType,
        region: &str,
        version: u64,
    ) -> GlobalKnowledgeGraphResult<()> {
        // Update vector clock
        let mut clock = self
            .vector_clocks
            .entry(resource_id.to_string())
            .or_insert_with(VectorClock::new);

        clock.increment(region);

        // Check for conflicts
        if self.config.consistency_level != ConsistencyLevel::Eventual {
            self.detect_conflicts(resource_id, resource_type, region, version, &clock)
                .await?;
        }

        Ok(())
    }

    /// Detect conflicts
    async fn detect_conflicts(
        &self,
        resource_id: &str,
        resource_type: ResourceType,
        region: &str,
        version: u64,
        clock: &VectorClock,
    ) -> GlobalKnowledgeGraphResult<()> {
        // Check if there are concurrent updates
        let conflict_key = format!("{}:{:?}", resource_id, resource_type);

        if let Some(mut conflict) = self.conflicts.get_mut(&conflict_key) {
            // Add new version to existing conflict
            conflict.versions.push(ConflictVersion {
                region: region.to_string(),
                version,
                vector_clock: clock.clone(),
                updated_at: chrono::Utc::now(),
            });

            if self.config.enable_auto_resolution {
                self.auto_resolve_conflict(&mut conflict).await?;
            }
        } else {
            // Check if this update conflicts with any recent updates
            let _now = Instant::now();
            let _window = Duration::from_millis(self.config.conflict_window_ms);

            // In production, would check against recent updates from other regions
            // For now, we'll just track the update
        }

        Ok(())
    }

    /// Auto-resolve conflict
    async fn auto_resolve_conflict(
        &self,
        conflict: &mut Conflict,
    ) -> GlobalKnowledgeGraphResult<()> {
        let strategy = self
            .resolution_strategies
            .get(&conflict.resource_id)
            .map(|s| s.clone())
            .unwrap_or(ResolutionStrategy::LastWriteWins);

        match strategy {
            ResolutionStrategy::LastWriteWins => {
                // Sort by timestamp and pick the latest
                conflict.versions.sort_by_key(|v| v.updated_at);
                if let Some(winner) = conflict.versions.last() {
                    conflict.status = ConflictStatus::Resolved;
                    tracing::info!(
                        "Resolved conflict {} using LastWriteWins: region {} wins",
                        conflict.id,
                        winner.region
                    );
                }
            }
            ResolutionStrategy::HigherVersionWins => {
                // Pick the highest version
                if let Some(winner) = conflict.versions.iter().max_by_key(|v| v.version) {
                    conflict.status = ConflictStatus::Resolved;
                    tracing::info!(
                        "Resolved conflict {} using HigherVersionWins: region {} wins",
                        conflict.id,
                        winner.region
                    );
                }
            }
            _ => {
                conflict.status = ConflictStatus::Detected;
            }
        }

        Ok(())
    }

    /// Register conflict manually
    pub async fn register_conflict(
        &self,
        resource_id: String,
        resource_type: ResourceType,
        versions: Vec<ConflictVersion>,
    ) -> GlobalKnowledgeGraphResult<String> {
        let conflict_id = Uuid::new_v4().to_string();
        let conflict = Conflict {
            id: conflict_id.clone(),
            resource_id: resource_id.clone(),
            resource_type,
            versions,
            detected_at: Instant::now(),
            status: ConflictStatus::Detected,
        };

        let conflict_key = format!("{}:{:?}", resource_id, conflict.resource_type);
        self.conflicts.insert(conflict_key, conflict);

        Ok(conflict_id)
    }

    /// Resolve conflict manually
    pub async fn resolve_conflict(
        &self,
        conflict_id: &str,
        winning_region: &str,
    ) -> GlobalKnowledgeGraphResult<()> {
        let mut found = false;

        for mut conflict_ref in self.conflicts.iter_mut() {
            if conflict_ref.id == conflict_id {
                if conflict_ref
                    .versions
                    .iter()
                    .any(|v| v.region == winning_region)
                {
                    conflict_ref.status = ConflictStatus::Resolved;
                    found = true;
                    break;
                }
            }
        }

        if !found {
            return Err(GlobalKnowledgeGraphError::ConsistencyConflict {
                region1: "unknown".to_string(),
                region2: "unknown".to_string(),
                conflict_type: "Conflict not found".to_string(),
            });
        }

        Ok(())
    }

    /// Get conflicts
    pub fn get_conflicts(&self, status_filter: Option<ConflictStatus>) -> Vec<Conflict> {
        self.conflicts
            .iter()
            .filter(|entry| {
                status_filter
                    .as_ref()
                    .map_or(true, |status| &entry.value().status == status)
            })
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Set resolution strategy
    pub fn set_resolution_strategy(&self, resource_id: String, strategy: ResolutionStrategy) {
        self.resolution_strategies.insert(resource_id, strategy);
    }

    /// Get vector clock for resource
    pub fn get_vector_clock(&self, resource_id: &str) -> Option<VectorClock> {
        self.vector_clocks
            .get(resource_id)
            .map(|clock| clock.clone())
    }

    /// Check convergence status
    pub fn check_convergence_status(&self, region: &str) -> Option<ConvergenceState> {
        self.convergence_states
            .get(region)
            .map(|state| state.clone())
    }

    /// Update convergence state
    pub fn update_convergence_state(&self, region: String, is_converged: bool, divergence_ms: u64) {
        self.convergence_states.alter(&region, |_, mut state| {
            state.last_check = Instant::now();
            state.is_converged = is_converged;
            state.divergence_ms = divergence_ms;
            state
        });
    }

    /// Check convergence across regions
    async fn check_convergence(
        states: &Arc<DashMap<String, ConvergenceState>>,
        config: &ConsistencyConfig,
    ) {
        for mut state_ref in states.iter_mut() {
            let state = state_ref.value_mut();
            state.last_check = Instant::now();

            // In production, would check actual divergence
            // For now, simulate convergence check
            if state.divergence_ms > config.max_divergence_ms {
                state.is_converged = false;
                tracing::warn!(
                    "Region {} has diverged beyond threshold: {}ms",
                    state.region,
                    state.divergence_ms
                );
            } else {
                state.is_converged = true;
            }
        }
    }

    /// Prune old vector clocks
    async fn prune_vector_clocks(clocks: &Arc<DashMap<String, VectorClock>>) {
        // In production, would prune clocks for deleted resources
        // For now, just log the action
        tracing::debug!("Pruning vector clocks, current count: {}", clocks.len());
    }

    /// Get consistency level
    pub fn get_consistency_level(&self) -> &ConsistencyLevel {
        &self.config.consistency_level
    }

    /// Monitor divergence
    pub async fn monitor_divergence(
        &self,
        region1: &str,
        region2: &str,
    ) -> GlobalKnowledgeGraphResult<u64> {
        // In production, would calculate actual divergence between regions
        // For now, return a simulated value
        let state1 = self.convergence_states.get(region1);
        let state2 = self.convergence_states.get(region2);

        match (state1, state2) {
            (Some(s1), Some(s2)) => Ok((s1.divergence_ms + s2.divergence_ms) / 2),
            _ => Ok(0),
        }
    }

    /// Force convergence
    pub async fn force_convergence(&self, region: &str) -> GlobalKnowledgeGraphResult<()> {
        self.convergence_states.alter(region, |_, mut state| {
            state.is_converged = true;
            state.divergence_ms = 0;
            state.last_check = Instant::now();
            state
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_clock_new() {
        let clock = VectorClock::new();
        assert_eq!(clock.clocks.len(), 0);
    }

    #[test]
    fn test_vector_clock_increment() {
        let mut clock = VectorClock::new();

        let value1 = clock.increment("us-east-1");
        assert_eq!(value1, 1);

        let value2 = clock.increment("us-east-1");
        assert_eq!(value2, 2);

        let value3 = clock.increment("eu-west-1");
        assert_eq!(value3, 1);
    }

    #[test]
    fn test_vector_clock_get() {
        let mut clock = VectorClock::new();
        clock.increment("us-east-1");

        assert_eq!(clock.get("us-east-1"), 1);
        assert_eq!(clock.get("eu-west-1"), 0);
    }

    #[test]
    fn test_vector_clock_merge() {
        let mut clock1 = VectorClock::new();
        clock1.increment("us-east-1");
        clock1.increment("us-east-1");

        let mut clock2 = VectorClock::new();
        clock2.increment("us-east-1");
        clock2.increment("eu-west-1");

        clock1.merge(&clock2);

        assert_eq!(clock1.get("us-east-1"), 2); // Max(2, 1) = 2
        assert_eq!(clock1.get("eu-west-1"), 1); // Max(0, 1) = 1
    }

    #[test]
    fn test_vector_clock_happens_before() {
        let mut clock1 = VectorClock::new();
        clock1.increment("us-east-1");

        let mut clock2 = VectorClock::new();
        clock2.increment("us-east-1");
        clock2.increment("us-east-1");

        assert!(clock1.happens_before(&clock2));
        assert!(!clock2.happens_before(&clock1));
    }

    #[test]
    fn test_vector_clock_concurrent() {
        let mut clock1 = VectorClock::new();
        clock1.increment("us-east-1");

        let mut clock2 = VectorClock::new();
        clock2.increment("eu-west-1");

        assert!(clock1.is_concurrent(&clock2));
        assert!(clock2.is_concurrent(&clock1));
    }

    #[tokio::test]
    async fn test_consistency_manager_creation() {
        let config = ConsistencyConfig::default();
        let manager = ConsistencyManager::new(config);
        assert_eq!(manager.conflicts.len(), 0);
    }

    #[tokio::test]
    async fn test_update_resource() {
        let manager = ConsistencyManager::new(ConsistencyConfig::default());

        let result = manager
            .update_resource("node-123", ResourceType::Node, "us-east-1", 1)
            .await;

        assert!(result.is_ok());

        let clock = manager.get_vector_clock("node-123");
        assert!(clock.is_some());
        assert_eq!(clock.unwrap().get("us-east-1"), 1);
    }

    #[tokio::test]
    async fn test_register_conflict() {
        let manager = ConsistencyManager::new(ConsistencyConfig::default());

        let versions = vec![
            ConflictVersion {
                region: "us-east-1".to_string(),
                version: 1,
                vector_clock: VectorClock::new(),
                updated_at: chrono::Utc::now(),
            },
            ConflictVersion {
                region: "eu-west-1".to_string(),
                version: 2,
                vector_clock: VectorClock::new(),
                updated_at: chrono::Utc::now(),
            },
        ];

        let conflict_id = manager
            .register_conflict("node-123".to_string(), ResourceType::Node, versions)
            .await
            .unwrap();

        assert!(!conflict_id.is_empty());
        assert_eq!(manager.conflicts.len(), 1);
    }

    #[tokio::test]
    async fn test_resolve_conflict() {
        let manager = ConsistencyManager::new(ConsistencyConfig::default());

        let versions = vec![
            ConflictVersion {
                region: "us-east-1".to_string(),
                version: 1,
                vector_clock: VectorClock::new(),
                updated_at: chrono::Utc::now(),
            },
            ConflictVersion {
                region: "eu-west-1".to_string(),
                version: 2,
                vector_clock: VectorClock::new(),
                updated_at: chrono::Utc::now(),
            },
        ];

        let conflict_id = manager
            .register_conflict("node-123".to_string(), ResourceType::Node, versions)
            .await
            .unwrap();

        let result = manager.resolve_conflict(&conflict_id, "us-east-1").await;
        assert!(result.is_ok());

        let conflicts = manager.get_conflicts(Some(ConflictStatus::Resolved));
        assert_eq!(conflicts.len(), 1);
    }

    #[tokio::test]
    async fn test_get_conflicts_by_status() {
        let manager = ConsistencyManager::new(ConsistencyConfig::default());

        // Register multiple conflicts
        for i in 0..3 {
            let versions = vec![ConflictVersion {
                region: "us-east-1".to_string(),
                version: i,
                vector_clock: VectorClock::new(),
                updated_at: chrono::Utc::now(),
            }];

            manager
                .register_conflict(format!("node-{}", i), ResourceType::Node, versions)
                .await
                .unwrap();
        }

        let detected = manager.get_conflicts(Some(ConflictStatus::Detected));
        assert_eq!(detected.len(), 3);

        let resolved = manager.get_conflicts(Some(ConflictStatus::Resolved));
        assert_eq!(resolved.len(), 0);
    }

    #[tokio::test]
    async fn test_set_resolution_strategy() {
        let manager = ConsistencyManager::new(ConsistencyConfig::default());

        manager.set_resolution_strategy(
            "node-123".to_string(),
            ResolutionStrategy::HigherVersionWins,
        );

        assert_eq!(manager.resolution_strategies.len(), 1);
    }

    #[tokio::test]
    async fn test_auto_resolution_last_write_wins() {
        let mut config = ConsistencyConfig::default();
        config.enable_auto_resolution = true;

        let manager = ConsistencyManager::new(config);

        let mut conflict = Conflict {
            id: "c1".to_string(),
            resource_id: "node-123".to_string(),
            resource_type: ResourceType::Node,
            versions: vec![
                ConflictVersion {
                    region: "us-east-1".to_string(),
                    version: 1,
                    vector_clock: VectorClock::new(),
                    updated_at: chrono::Utc::now() - chrono::Duration::seconds(10),
                },
                ConflictVersion {
                    region: "eu-west-1".to_string(),
                    version: 2,
                    vector_clock: VectorClock::new(),
                    updated_at: chrono::Utc::now(),
                },
            ],
            detected_at: Instant::now(),
            status: ConflictStatus::Detected,
        };

        manager.auto_resolve_conflict(&mut conflict).await.unwrap();
        assert_eq!(conflict.status, ConflictStatus::Resolved);
    }

    #[tokio::test]
    async fn test_auto_resolution_higher_version_wins() {
        let mut config = ConsistencyConfig::default();
        config.enable_auto_resolution = true;

        let manager = ConsistencyManager::new(config);

        manager.set_resolution_strategy(
            "node-123".to_string(),
            ResolutionStrategy::HigherVersionWins,
        );

        let mut conflict = Conflict {
            id: "c1".to_string(),
            resource_id: "node-123".to_string(),
            resource_type: ResourceType::Node,
            versions: vec![
                ConflictVersion {
                    region: "us-east-1".to_string(),
                    version: 5,
                    vector_clock: VectorClock::new(),
                    updated_at: chrono::Utc::now() - chrono::Duration::seconds(10),
                },
                ConflictVersion {
                    region: "eu-west-1".to_string(),
                    version: 3,
                    vector_clock: VectorClock::new(),
                    updated_at: chrono::Utc::now(),
                },
            ],
            detected_at: Instant::now(),
            status: ConflictStatus::Detected,
        };

        manager.auto_resolve_conflict(&mut conflict).await.unwrap();
        assert_eq!(conflict.status, ConflictStatus::Resolved);
    }

    #[tokio::test]
    async fn test_convergence_state() {
        let manager = ConsistencyManager::new(ConsistencyConfig::default());

        let state = ConvergenceState {
            region: "us-east-1".to_string(),
            last_check: Instant::now(),
            is_converged: true,
            divergence_ms: 0,
            pending_ops: 0,
        };

        manager
            .convergence_states
            .insert("us-east-1".to_string(), state);

        let retrieved = manager.check_convergence_status("us-east-1");
        assert!(retrieved.is_some());
        assert!(retrieved.unwrap().is_converged);
    }

    #[tokio::test]
    async fn test_update_convergence_state() {
        let manager = ConsistencyManager::new(ConsistencyConfig::default());

        let state = ConvergenceState {
            region: "us-east-1".to_string(),
            last_check: Instant::now(),
            is_converged: false,
            divergence_ms: 100,
            pending_ops: 5,
        };

        manager
            .convergence_states
            .insert("us-east-1".to_string(), state);
        manager.update_convergence_state("us-east-1".to_string(), true, 0);

        let updated = manager.check_convergence_status("us-east-1").unwrap();
        assert!(updated.is_converged);
        assert_eq!(updated.divergence_ms, 0);
    }

    #[tokio::test]
    async fn test_monitor_divergence() {
        let manager = ConsistencyManager::new(ConsistencyConfig::default());

        let state1 = ConvergenceState {
            region: "us-east-1".to_string(),
            last_check: Instant::now(),
            is_converged: true,
            divergence_ms: 50,
            pending_ops: 0,
        };

        let state2 = ConvergenceState {
            region: "eu-west-1".to_string(),
            last_check: Instant::now(),
            is_converged: true,
            divergence_ms: 30,
            pending_ops: 0,
        };

        manager
            .convergence_states
            .insert("us-east-1".to_string(), state1);
        manager
            .convergence_states
            .insert("eu-west-1".to_string(), state2);

        let divergence = manager
            .monitor_divergence("us-east-1", "eu-west-1")
            .await
            .unwrap();
        assert_eq!(divergence, 40); // (50 + 30) / 2
    }

    #[tokio::test]
    async fn test_force_convergence() {
        let manager = ConsistencyManager::new(ConsistencyConfig::default());

        let state = ConvergenceState {
            region: "us-east-1".to_string(),
            last_check: Instant::now(),
            is_converged: false,
            divergence_ms: 1000,
            pending_ops: 10,
        };

        manager
            .convergence_states
            .insert("us-east-1".to_string(), state);

        manager.force_convergence("us-east-1").await.unwrap();

        let updated = manager.check_convergence_status("us-east-1").unwrap();
        assert!(updated.is_converged);
        assert_eq!(updated.divergence_ms, 0);
    }

    #[tokio::test]
    async fn test_consistency_levels() {
        let config = ConsistencyConfig {
            consistency_level: ConsistencyLevel::Strong,
            ..Default::default()
        };

        let manager = ConsistencyManager::new(config);

        match manager.get_consistency_level() {
            ConsistencyLevel::Strong => assert!(true),
            _ => panic!("Wrong consistency level"),
        }
    }

    #[tokio::test]
    async fn test_bounded_consistency() {
        let config = ConsistencyConfig {
            consistency_level: ConsistencyLevel::Bounded {
                max_staleness_ms: 5000,
            },
            ..Default::default()
        };

        let manager = ConsistencyManager::new(config);

        match manager.get_consistency_level() {
            ConsistencyLevel::Bounded { max_staleness_ms } => {
                assert_eq!(*max_staleness_ms, 5000);
            }
            _ => panic!("Wrong consistency level"),
        }
    }

    #[tokio::test]
    async fn test_start_stop_manager() {
        let manager = ConsistencyManager::new(ConsistencyConfig::default());

        // Start the manager
        manager.start().await;

        // Give it a moment to initialize
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Stop the manager
        manager.stop().await;

        // Give it a moment to shut down
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    #[tokio::test]
    async fn test_multiple_updates_same_resource() {
        let manager = ConsistencyManager::new(ConsistencyConfig::default());

        // Multiple updates from different regions
        manager
            .update_resource("node-123", ResourceType::Node, "us-east-1", 1)
            .await?;
        manager
            .update_resource("node-123", ResourceType::Node, "eu-west-1", 2)
            .await
            .unwrap();
        manager
            .update_resource("node-123", ResourceType::Node, "us-east-1", 3)
            .await
            .unwrap();

        let clock = manager.get_vector_clock("node-123").unwrap();
        assert_eq!(clock.get("us-east-1"), 2);
        assert_eq!(clock.get("eu-west-1"), 1);
    }
}
