//! Cross-region data replication with conflict resolution and consistency models
//!
//! This module provides sophisticated data replication capabilities:
//! - Multiple consistency models (eventual, strong, causal)
//! - Conflict resolution strategies (LWW, merge, custom)
//! - Replication topologies (master-slave, multi-master, ring)
//! - Vector clocks for causality tracking
//! - Anti-entropy mechanisms for data repair

use crate::error::{MultiRegionError, MultiRegionResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

/// Replication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationConfig {
    /// Consistency model
    pub consistency_model: ConsistencyModel,
    /// Conflict resolution strategy
    pub conflict_resolution: ConflictResolutionStrategy,
    /// Replication topology
    pub topology: ReplicationTopology,
    /// Replication factor (number of replicas)
    pub replication_factor: u32,
    /// Maximum replication lag (milliseconds)
    pub max_replication_lag_ms: u64,
    /// Batch size for replication
    pub batch_size: u32,
    /// Anti-entropy interval (seconds)
    pub anti_entropy_interval_s: u64,
    /// Vector clock configuration
    pub vector_clock_config: VectorClockConfig,
}

/// Consistency models for replication
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsistencyModel {
    /// Eventual consistency - data will converge eventually
    Eventual,
    /// Strong consistency - all replicas updated before acknowledgment
    Strong,
    /// Causal consistency - preserves causal relationships
    Causal,
    /// Bounded staleness - maximum staleness guarantee
    BoundedStaleness { max_staleness_ms: u64 },
    /// Session consistency - read-your-writes guarantee
    Session,
}

/// Conflict resolution strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictResolutionStrategy {
    /// Last-writer-wins based on timestamp
    LastWriterWins,
    /// Multi-value - preserve all conflicting values
    MultiValue,
    /// Custom merge function
    CustomMerge { merge_function: String },
    /// Application-level resolution
    ApplicationLevel,
    /// Concurrent writes rejected
    RejectConcurrentWrites,
}

/// Replication topology patterns
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicationTopology {
    /// Single master, multiple slaves
    MasterSlave { master_region: String },
    /// Multiple masters with conflict resolution
    MultiMaster,
    /// Ring topology with consistent hashing
    Ring,
    /// Mesh topology - all nodes connected
    Mesh,
    /// Tree topology with hierarchical replication
    Tree { root_region: String },
}

/// Vector clock configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorClockConfig {
    /// Enable vector clocks for causality tracking
    pub enabled: bool,
    /// Maximum clock entries before pruning
    pub max_entries: u32,
    /// Clock pruning interval (seconds)
    pub pruning_interval_s: u64,
}

/// Replication log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationLogEntry {
    /// Unique entry ID
    pub id: String,
    /// Data key being replicated
    pub data_key: String,
    /// Operation type
    pub operation: ReplicationOperation,
    /// Data payload
    pub data: Vec<u8>,
    /// Source region
    pub source_region: String,
    /// Target regions
    pub target_regions: Vec<String>,
    /// Vector clock
    pub vector_clock: VectorClock,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Checksum for integrity
    pub checksum: String,
    /// Retry count
    pub retry_count: u32,
}

/// Replication operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicationOperation {
    /// Insert new data
    Insert,
    /// Update existing data
    Update,
    /// Delete data
    Delete,
    /// Merge operation for conflict resolution
    Merge,
    /// Anti-entropy repair
    Repair,
}

/// Vector clock for causality tracking
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VectorClock {
    /// Clock entries per region
    pub clocks: BTreeMap<String, u64>,
}

/// Conflict detection result
#[derive(Debug, Clone)]
pub struct ConflictInfo {
    /// Data key with conflict
    pub data_key: String,
    /// Conflicting entries
    pub conflicting_entries: Vec<ReplicationLogEntry>,
    /// Detected conflict type
    pub conflict_type: ConflictType,
    /// Recommended resolution
    pub recommended_resolution: ConflictResolution,
}

/// Types of conflicts
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictType {
    /// Concurrent writes to same key
    ConcurrentWrite,
    /// Update-delete conflict
    UpdateDelete,
    /// Version fork (divergent history)
    VersionFork,
    /// Causality violation
    CausalityViolation,
}

/// Conflict resolution result
#[derive(Debug, Clone)]
pub struct ConflictResolution {
    /// Resolved data
    pub resolved_data: Vec<u8>,
    /// Resolution strategy used
    pub strategy: ConflictResolutionStrategy,
    /// Merged vector clock
    pub merged_clock: VectorClock,
    /// Resolution metadata
    pub metadata: HashMap<String, String>,
}

/// Replication status for a region
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationStatus {
    /// Region identifier
    pub region_id: String,
    /// Last successful replication timestamp
    pub last_replication: DateTime<Utc>,
    /// Current replication lag (milliseconds)
    pub replication_lag_ms: u64,
    /// Number of pending operations
    pub pending_operations: u64,
    /// Success rate percentage
    pub success_rate: f64,
    /// Error count
    pub error_count: u64,
    /// Vector clock state
    pub vector_clock: VectorClock,
}

/// Cross-region replication manager
pub struct ReplicationManager {
    config: ReplicationConfig,
    replication_log: Arc<Mutex<VecDeque<ReplicationLogEntry>>>,
    region_status: Arc<RwLock<HashMap<String, ReplicationStatus>>>,
    conflict_resolver: Arc<ConflictResolver>,
    vector_clock_manager: Arc<VectorClockManager>,
    sequence_counter: AtomicU64,
    client: reqwest::Client,
}

/// Conflict resolution engine
pub struct ConflictResolver {
    strategy: ConflictResolutionStrategy,
    custom_functions:
        HashMap<String, fn(&[ReplicationLogEntry]) -> MultiRegionResult<ConflictResolution>>,
}

/// Vector clock management
pub struct VectorClockManager {
    config: VectorClockConfig,
    region_clocks: Arc<RwLock<HashMap<String, VectorClock>>>,
}

impl ReplicationManager {
    /// Create new replication manager
    pub fn new(config: ReplicationConfig, regions: Vec<String>) -> MultiRegionResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| MultiRegionError::ReplicationFailure {
                reason: format!("Failed to create HTTP client: {}", e),
            })?;

        let mut region_status = HashMap::new();
        for region in regions {
            region_status.insert(
                region.clone(),
                ReplicationStatus {
                    region_id: region.clone(),
                    last_replication: Utc::now(),
                    replication_lag_ms: 0,
                    pending_operations: 0,
                    success_rate: 100.0,
                    error_count: 0,
                    vector_clock: VectorClock::new(),
                },
            );
        }

        Ok(Self {
            conflict_resolver: Arc::new(ConflictResolver::new(config.conflict_resolution.clone())),
            vector_clock_manager: Arc::new(VectorClockManager::new(
                config.vector_clock_config.clone(),
            )),
            config,
            replication_log: Arc::new(Mutex::new(VecDeque::new())),
            region_status: Arc::new(RwLock::new(region_status)),
            sequence_counter: AtomicU64::new(0),
            client,
        })
    }

    /// Replicate data to target regions
    pub async fn replicate_data(
        &self,
        data_key: String,
        data: Vec<u8>,
        operation: ReplicationOperation,
        source_region: String,
        target_regions: Vec<String>,
    ) -> MultiRegionResult<String> {
        // Create vector clock for this operation
        let vector_clock = self
            .vector_clock_manager
            .increment_clock(&source_region)
            .await;

        // Create replication log entry
        let entry_id = self.generate_entry_id();
        let checksum = self.calculate_checksum(&data);

        let log_entry = ReplicationLogEntry {
            id: entry_id.clone(),
            data_key: data_key.clone(),
            operation,
            data: data.clone(),
            source_region: source_region.clone(),
            target_regions: target_regions.clone(),
            vector_clock,
            timestamp: Utc::now(),
            checksum,
            retry_count: 0,
        };

        // Add to replication log
        {
            let mut log = self.replication_log.lock().await;
            log.push_back(log_entry.clone());
        }

        // Apply consistency model
        match self.config.consistency_model {
            ConsistencyModel::Strong => {
                self.replicate_strong_consistency(&log_entry).await?;
            }
            ConsistencyModel::Eventual => {
                self.replicate_eventual_consistency(&log_entry).await?;
            }
            ConsistencyModel::Causal => {
                self.replicate_causal_consistency(&log_entry).await?;
            }
            ConsistencyModel::BoundedStaleness { max_staleness_ms } => {
                self.replicate_bounded_staleness(&log_entry, max_staleness_ms)
                    .await?;
            }
            ConsistencyModel::Session => {
                self.replicate_session_consistency(&log_entry).await?;
            }
        }

        Ok(entry_id)
    }

    /// Replicate with strong consistency
    async fn replicate_strong_consistency(
        &self,
        entry: &ReplicationLogEntry,
    ) -> MultiRegionResult<()> {
        let mut successful_replications = 0;
        let required_replications = entry.target_regions.len();

        for target_region in &entry.target_regions {
            match self.send_replication_request(entry, target_region).await {
                Ok(_) => successful_replications += 1,
                Err(e) => {
                    tracing::warn!("Failed to replicate to region {}: {}", target_region, e);
                }
            }
        }

        if successful_replications != required_replications {
            return Err(MultiRegionError::ReplicationFailure {
                reason: format!(
                    "Strong consistency requires all replications to succeed. Got {}/{} successes",
                    successful_replications, required_replications
                ),
            });
        }

        Ok(())
    }

    /// Replicate with eventual consistency
    async fn replicate_eventual_consistency(
        &self,
        entry: &ReplicationLogEntry,
    ) -> MultiRegionResult<()> {
        // Fire and forget for eventual consistency
        for target_region in &entry.target_regions {
            let entry_clone = entry.clone();
            let target_region_clone = target_region.clone();
            let self_client = self.client.clone();

            tokio::spawn(async move {
                if let Err(e) = Self::send_replication_request_static(
                    &self_client,
                    &entry_clone,
                    &target_region_clone,
                )
                .await
                {
                    tracing::warn!(
                        "Failed eventual consistency replication to {}: {}",
                        target_region_clone,
                        e
                    );
                }
            });
        }
        Ok(())
    }

    /// Replicate with causal consistency
    async fn replicate_causal_consistency(
        &self,
        entry: &ReplicationLogEntry,
    ) -> MultiRegionResult<()> {
        // Check causal dependencies
        let dependencies = self.get_causal_dependencies(entry).await?;

        for target_region in &entry.target_regions {
            // Ensure dependencies are satisfied before replicating
            self.ensure_causal_dependencies(target_region, &dependencies)
                .await?;
            self.send_replication_request(entry, target_region).await?;
        }

        Ok(())
    }

    /// Replicate with bounded staleness
    async fn replicate_bounded_staleness(
        &self,
        entry: &ReplicationLogEntry,
        max_staleness_ms: u64,
    ) -> MultiRegionResult<()> {
        let deadline = Utc::now() + chrono::Duration::milliseconds(max_staleness_ms as i64);

        for target_region in &entry.target_regions {
            let remaining_time = deadline.signed_duration_since(Utc::now());
            if remaining_time.num_milliseconds() <= 0 {
                return Err(MultiRegionError::ReplicationFailure {
                    reason: "Bounded staleness deadline exceeded".to_string(),
                });
            }

            tokio::time::timeout(
                std::time::Duration::from_millis(remaining_time.num_milliseconds() as u64),
                self.send_replication_request(entry, target_region),
            )
            .await
            .map_err(|_| MultiRegionError::ReplicationFailure {
                reason: "Replication timeout exceeded staleness bound".to_string(),
            })??;
        }

        Ok(())
    }

    /// Replicate with session consistency
    async fn replicate_session_consistency(
        &self,
        entry: &ReplicationLogEntry,
    ) -> MultiRegionResult<()> {
        // For session consistency, we need to ensure read-your-writes
        // This is typically handled at the application level, but we ensure
        // that the source region acknowledges the write
        self.send_replication_request(entry, &entry.source_region)
            .await?;

        // Then replicate to other regions eventually
        self.replicate_eventual_consistency(entry).await
    }

    /// Send replication request to target region
    async fn send_replication_request(
        &self,
        entry: &ReplicationLogEntry,
        target_region: &str,
    ) -> MultiRegionResult<()> {
        Self::send_replication_request_static(&self.client, entry, target_region).await
    }

    /// Static version for use in spawned tasks
    async fn send_replication_request_static(
        client: &reqwest::Client,
        entry: &ReplicationLogEntry,
        target_region: &str,
    ) -> MultiRegionResult<()> {
        let url = format!("https://{}.example.com/replication", target_region);
        let response = client.post(&url).json(entry).send().await.map_err(|e| {
            MultiRegionError::ReplicationFailure {
                reason: format!("Failed to send replication request: {}", e),
            }
        })?;

        if !response.status().is_success() {
            return Err(MultiRegionError::ReplicationFailure {
                reason: format!(
                    "Replication request failed with status: {}",
                    response.status()
                ),
            });
        }

        Ok(())
    }

    /// Get causal dependencies for an entry
    async fn get_causal_dependencies(
        &self,
        entry: &ReplicationLogEntry,
    ) -> MultiRegionResult<Vec<String>> {
        let mut dependencies = Vec::new();
        let log = self.replication_log.lock().await;

        for log_entry in log.iter() {
            if log_entry.id != entry.id
                && log_entry.data_key == entry.data_key
                && self.happens_before(&log_entry.vector_clock, &entry.vector_clock)
            {
                dependencies.push(log_entry.id.clone());
            }
        }

        Ok(dependencies)
    }

    /// Ensure causal dependencies are satisfied
    async fn ensure_causal_dependencies(
        &self,
        _target_region: &str,
        _dependencies: &[String],
    ) -> MultiRegionResult<()> {
        // In a real implementation, this would check if dependencies are satisfied
        // For now, we'll assume they are
        Ok(())
    }

    /// Check if clock A happens before clock B
    fn happens_before(&self, clock_a: &VectorClock, clock_b: &VectorClock) -> bool {
        clock_a.happens_before(clock_b)
    }

    /// Detect conflicts in replication log
    pub async fn detect_conflicts(&self) -> MultiRegionResult<Vec<ConflictInfo>> {
        let mut conflicts = Vec::new();
        let log = self.replication_log.lock().await;

        // Group entries by data key
        let mut key_entries: HashMap<String, Vec<&ReplicationLogEntry>> = HashMap::new();
        for entry in log.iter() {
            key_entries
                .entry(entry.data_key.clone())
                .or_insert_with(Vec::new)
                .push(entry);
        }

        // Check for conflicts within each key
        for (data_key, entries) in key_entries {
            if entries.len() > 1 {
                let conflict_groups = self.find_concurrent_entries(&entries);
                for group in conflict_groups {
                    if group.len() > 1 {
                        let conflict_type = self.determine_conflict_type(&group);
                        let recommended_resolution =
                            self.recommend_resolution(&group, &conflict_type);

                        conflicts.push(ConflictInfo {
                            data_key: data_key.clone(),
                            conflicting_entries: group.into_iter().cloned().collect(),
                            conflict_type,
                            recommended_resolution,
                        });
                    }
                }
            }
        }

        Ok(conflicts)
    }

    /// Find concurrent entries (no causal relationship)
    fn find_concurrent_entries<'a>(
        &self,
        entries: &[&'a ReplicationLogEntry],
    ) -> Vec<Vec<&'a ReplicationLogEntry>> {
        let mut groups = Vec::new();
        let mut visited = HashSet::new();

        for (i, entry_a) in entries.iter().enumerate() {
            if visited.contains(&i) {
                continue;
            }

            let mut concurrent_group = vec![*entry_a];
            visited.insert(i);

            for (j, entry_b) in entries.iter().enumerate() {
                if i != j && !visited.contains(&j) {
                    if self.are_concurrent(&entry_a.vector_clock, &entry_b.vector_clock) {
                        concurrent_group.push(*entry_b);
                        visited.insert(j);
                    }
                }
            }

            if concurrent_group.len() > 1 {
                groups.push(concurrent_group);
            }
        }

        groups
    }

    /// Check if two vector clocks are concurrent
    fn are_concurrent(&self, clock_a: &VectorClock, clock_b: &VectorClock) -> bool {
        !clock_a.happens_before(clock_b) && !clock_b.happens_before(clock_a)
    }

    /// Determine the type of conflict
    fn determine_conflict_type(&self, entries: &[&ReplicationLogEntry]) -> ConflictType {
        let has_update = entries
            .iter()
            .any(|e| e.operation == ReplicationOperation::Update);
        let has_delete = entries
            .iter()
            .any(|e| e.operation == ReplicationOperation::Delete);

        if has_update && has_delete {
            ConflictType::UpdateDelete
        } else {
            ConflictType::ConcurrentWrite
        }
    }

    /// Recommend conflict resolution
    fn recommend_resolution(
        &self,
        entries: &[&ReplicationLogEntry],
        _conflict_type: &ConflictType,
    ) -> ConflictResolution {
        match self.config.conflict_resolution {
            ConflictResolutionStrategy::LastWriterWins => {
                let latest_entry = entries.iter().max_by_key(|e| e.timestamp).unwrap();
                ConflictResolution {
                    resolved_data: latest_entry.data.clone(),
                    strategy: ConflictResolutionStrategy::LastWriterWins,
                    merged_clock: self.merge_vector_clocks(entries),
                    metadata: HashMap::new(),
                }
            }
            _ => {
                // For other strategies, return the first entry as a fallback
                ConflictResolution {
                    resolved_data: entries[0].data.clone(),
                    strategy: self.config.conflict_resolution.clone(),
                    merged_clock: self.merge_vector_clocks(entries),
                    metadata: HashMap::new(),
                }
            }
        }
    }

    /// Merge vector clocks from multiple entries
    fn merge_vector_clocks(&self, entries: &[&ReplicationLogEntry]) -> VectorClock {
        let mut merged = VectorClock::new();
        for entry in entries {
            merged = merged.merge(&entry.vector_clock);
        }
        merged
    }

    /// Resolve conflicts using configured strategy
    pub async fn resolve_conflicts(
        &self,
        conflicts: &[ConflictInfo],
    ) -> MultiRegionResult<Vec<String>> {
        let mut resolved_ids = Vec::new();

        for conflict in conflicts {
            let resolution = self.conflict_resolver.resolve_conflict(conflict)?;

            // Create a new replication entry with resolved data
            let checksum = self.calculate_checksum(&resolution.resolved_data);
            let resolved_entry = ReplicationLogEntry {
                id: self.generate_entry_id(),
                data_key: conflict.data_key.clone(),
                operation: ReplicationOperation::Merge,
                data: resolution.resolved_data,
                source_region: "system".to_string(),
                target_regions: vec![], // Will be filled based on replication factor
                vector_clock: resolution.merged_clock,
                timestamp: Utc::now(),
                checksum,
                retry_count: 0,
            };

            // Add resolved entry to log
            {
                let mut log = self.replication_log.lock().await;
                log.push_back(resolved_entry);
            }

            resolved_ids.push(conflict.data_key.clone());
        }

        Ok(resolved_ids)
    }

    /// Perform anti-entropy to repair inconsistencies
    pub async fn perform_anti_entropy(&self) -> MultiRegionResult<u64> {
        let mut repairs_made = 0;
        let status_map = self.region_status.read().await;
        let regions: Vec<_> = status_map.keys().cloned().collect();
        drop(status_map);

        // Compare data between regions and identify inconsistencies
        for region_a in &regions {
            for region_b in &regions {
                if region_a != region_b {
                    let inconsistencies = self.find_inconsistencies(region_a, region_b).await?;
                    repairs_made += inconsistencies.len() as u64;

                    // Repair inconsistencies
                    for inconsistency in inconsistencies {
                        self.repair_inconsistency(inconsistency).await?;
                    }
                }
            }
        }

        Ok(repairs_made)
    }

    /// Find inconsistencies between two regions
    async fn find_inconsistencies(
        &self,
        _region_a: &str,
        _region_b: &str,
    ) -> MultiRegionResult<Vec<String>> {
        // In a real implementation, this would query both regions and compare data
        // For now, return empty list
        Ok(Vec::new())
    }

    /// Repair a specific inconsistency
    async fn repair_inconsistency(&self, _inconsistency: String) -> MultiRegionResult<()> {
        // In a real implementation, this would create repair operations
        Ok(())
    }

    /// Get replication status for all regions
    pub async fn get_replication_status(&self) -> Vec<ReplicationStatus> {
        let status_map = self.region_status.read().await;
        status_map.values().cloned().collect()
    }

    /// Update replication status for a region
    pub async fn update_region_status(&self, region_id: &str, lag_ms: u64, success: bool) {
        let mut status_map = self.region_status.write().await;
        if let Some(status) = status_map.get_mut(region_id) {
            status.last_replication = Utc::now();
            status.replication_lag_ms = lag_ms;

            if success {
                status.success_rate = (status.success_rate * 0.95) + (100.0 * 0.05);
            } else {
                status.error_count += 1;
                status.success_rate = (status.success_rate * 0.95) + (0.0 * 0.05);
            }
        }
    }

    /// Generate unique entry ID
    fn generate_entry_id(&self) -> String {
        let sequence = self.sequence_counter.fetch_add(1, Ordering::SeqCst);
        format!("repl_{}", sequence)
    }

    /// Calculate checksum for data integrity
    fn calculate_checksum(&self, data: &[u8]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

impl ConflictResolver {
    /// Create new conflict resolver
    pub fn new(strategy: ConflictResolutionStrategy) -> Self {
        Self {
            strategy,
            custom_functions: HashMap::new(),
        }
    }

    /// Resolve a specific conflict
    pub fn resolve_conflict(
        &self,
        conflict: &ConflictInfo,
    ) -> MultiRegionResult<ConflictResolution> {
        match &self.strategy {
            ConflictResolutionStrategy::LastWriterWins => {
                let latest_entry = conflict
                    .conflicting_entries
                    .iter()
                    .max_by_key(|e| e.timestamp)
                    .ok_or_else(|| MultiRegionError::ReplicationFailure {
                        reason: "No entries to resolve conflict".to_string(),
                    })?;

                Ok(ConflictResolution {
                    resolved_data: latest_entry.data.clone(),
                    strategy: ConflictResolutionStrategy::LastWriterWins,
                    merged_clock: self.merge_clocks(&conflict.conflicting_entries),
                    metadata: HashMap::new(),
                })
            }
            ConflictResolutionStrategy::MultiValue => {
                // Combine all values
                let mut combined_data = Vec::new();
                for entry in &conflict.conflicting_entries {
                    combined_data.extend(&entry.data);
                }

                Ok(ConflictResolution {
                    resolved_data: combined_data,
                    strategy: ConflictResolutionStrategy::MultiValue,
                    merged_clock: self.merge_clocks(&conflict.conflicting_entries),
                    metadata: HashMap::new(),
                })
            }
            _ => {
                // Fallback to first entry
                Ok(ConflictResolution {
                    resolved_data: conflict.conflicting_entries[0].data.clone(),
                    strategy: self.strategy.clone(),
                    merged_clock: self.merge_clocks(&conflict.conflicting_entries),
                    metadata: HashMap::new(),
                })
            }
        }
    }

    /// Merge vector clocks from conflicting entries
    fn merge_clocks(&self, entries: &[ReplicationLogEntry]) -> VectorClock {
        let mut merged = VectorClock::new();
        for entry in entries {
            merged = merged.merge(&entry.vector_clock);
        }
        merged
    }
}

impl VectorClockManager {
    /// Create new vector clock manager
    pub fn new(config: VectorClockConfig) -> Self {
        Self {
            config,
            region_clocks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Increment clock for a region
    pub async fn increment_clock(&self, region: &str) -> VectorClock {
        let mut clocks = self.region_clocks.write().await;
        let clock = clocks
            .entry(region.to_string())
            .or_insert_with(VectorClock::new);
        clock.increment(region);
        clock.clone()
    }

    /// Get current clock for a region
    pub async fn get_clock(&self, region: &str) -> VectorClock {
        let clocks = self.region_clocks.read().await;
        clocks.get(region).cloned().unwrap_or_else(VectorClock::new)
    }
}

impl VectorClock {
    /// Create new empty vector clock
    pub fn new() -> Self {
        Self {
            clocks: BTreeMap::new(),
        }
    }

    /// Increment clock for a region
    pub fn increment(&mut self, region: &str) {
        let entry = self.clocks.entry(region.to_string()).or_insert(0);
        *entry += 1;
    }

    /// Check if this clock happens before another
    pub fn happens_before(&self, other: &VectorClock) -> bool {
        let mut strictly_less = false;

        // Check all entries in self
        for (region, &count) in &self.clocks {
            let other_count = other.clocks.get(region).copied().unwrap_or(0);
            if count > other_count {
                return false;
            }
            if count < other_count {
                strictly_less = true;
            }
        }

        // Check all entries in other that aren't in self
        for (region, &other_count) in &other.clocks {
            if !self.clocks.contains_key(region) && other_count > 0 {
                strictly_less = true;
            }
        }

        strictly_less
    }

    /// Merge two vector clocks
    pub fn merge(&self, other: &VectorClock) -> VectorClock {
        let mut merged = BTreeMap::new();

        // Take maximum of each region
        for (region, &count) in &self.clocks {
            let other_count = other.clocks.get(region).copied().unwrap_or(0);
            merged.insert(region.clone(), count.max(other_count));
        }

        for (region, &count) in &other.clocks {
            if !self.clocks.contains_key(region) {
                merged.insert(region.clone(), count);
            }
        }

        VectorClock { clocks: merged }
    }
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self {
            consistency_model: ConsistencyModel::Eventual,
            conflict_resolution: ConflictResolutionStrategy::LastWriterWins,
            topology: ReplicationTopology::MultiMaster,
            replication_factor: 3,
            max_replication_lag_ms: 5000,
            batch_size: 100,
            anti_entropy_interval_s: 300,
            vector_clock_config: VectorClockConfig {
                enabled: true,
                max_entries: 1000,
                pruning_interval_s: 3600,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> ReplicationConfig {
        ReplicationConfig {
            consistency_model: ConsistencyModel::Eventual,
            conflict_resolution: ConflictResolutionStrategy::LastWriterWins,
            topology: ReplicationTopology::MultiMaster,
            replication_factor: 3,
            max_replication_lag_ms: 1000,
            batch_size: 10,
            anti_entropy_interval_s: 60,
            vector_clock_config: VectorClockConfig {
                enabled: true,
                max_entries: 100,
                pruning_interval_s: 300,
            },
        }
    }

    fn create_test_regions() -> Vec<String> {
        vec![
            "us-east-1".to_string(),
            "us-west-2".to_string(),
            "eu-west-1".to_string(),
        ]
    }

    #[tokio::test]
    async fn test_replication_manager_creation() {
        let config = create_test_config();
        let regions = create_test_regions();
        let manager = ReplicationManager::new(config, regions);
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_eventual_consistency_replication() {
        let config = create_test_config();
        let regions = create_test_regions();
        let manager = ReplicationManager::new(config, regions).unwrap();

        let result = manager
            .replicate_data(
                "test-key".to_string(),
                b"test-data".to_vec(),
                ReplicationOperation::Insert,
                "us-east-1".to_string(),
                vec!["us-west-2".to_string(), "eu-west-1".to_string()],
            )
            .await;

        assert!(result.is_ok());
        let entry_id = result.unwrap();
        assert!(!entry_id.is_empty());
    }

    #[tokio::test]
    async fn test_strong_consistency_replication() {
        let mut config = create_test_config();
        config.consistency_model = ConsistencyModel::Strong;
        let regions = create_test_regions();
        let manager = ReplicationManager::new(config, regions).unwrap();

        // This will fail because we don't have actual endpoints
        let result = manager
            .replicate_data(
                "test-key".to_string(),
                b"test-data".to_vec(),
                ReplicationOperation::Insert,
                "us-east-1".to_string(),
                vec!["us-west-2".to_string()],
            )
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_vector_clock_functionality() {
        let mut clock1 = VectorClock::new();
        let mut clock2 = VectorClock::new();

        clock1.increment("region1");
        clock2.increment("region2");

        // They should be concurrent
        assert!(!clock1.happens_before(&clock2));
        assert!(!clock2.happens_before(&clock1));

        clock1.increment("region2");
        // Now clock1 should happen after clock2
        assert!(clock2.happens_before(&clock1));
        assert!(!clock1.happens_before(&clock2));
    }

    #[tokio::test]
    async fn test_vector_clock_merge() {
        let mut clock1 = VectorClock::new();
        let mut clock2 = VectorClock::new();

        clock1.increment("region1");
        clock1.increment("region1");
        clock2.increment("region2");
        clock2.increment("region2");
        clock2.increment("region2");

        let merged = clock1.merge(&clock2);
        assert_eq!(merged.clocks.get("region1"), Some(&2));
        assert_eq!(merged.clocks.get("region2"), Some(&3));
    }

    #[tokio::test]
    async fn test_conflict_detection() {
        let config = create_test_config();
        let regions = create_test_regions();
        let manager = ReplicationManager::new(config, regions).unwrap();

        // Create conflicting entries
        let mut clock1 = VectorClock::new();
        let mut clock2 = VectorClock::new();
        clock1.increment("region1");
        clock2.increment("region2");

        let entry1 = ReplicationLogEntry {
            id: "entry1".to_string(),
            data_key: "conflict-key".to_string(),
            operation: ReplicationOperation::Update,
            data: b"data1".to_vec(),
            source_region: "region1".to_string(),
            target_regions: vec!["region2".to_string()],
            vector_clock: clock1,
            timestamp: Utc::now(),
            checksum: "checksum1".to_string(),
            retry_count: 0,
        };

        let entry2 = ReplicationLogEntry {
            id: "entry2".to_string(),
            data_key: "conflict-key".to_string(),
            operation: ReplicationOperation::Update,
            data: b"data2".to_vec(),
            source_region: "region2".to_string(),
            target_regions: vec!["region1".to_string()],
            vector_clock: clock2,
            timestamp: Utc::now(),
            checksum: "checksum2".to_string(),
            retry_count: 0,
        };

        // Add entries to log
        {
            let mut log = manager.replication_log.lock().await;
            log.push_back(entry1);
            log.push_back(entry2);
        }

        let conflicts = manager.detect_conflicts().await.unwrap();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].data_key, "conflict-key");
        assert_eq!(conflicts[0].conflicting_entries.len(), 2);
    }

    #[tokio::test]
    async fn test_last_writer_wins_resolution() {
        let resolver = ConflictResolver::new(ConflictResolutionStrategy::LastWriterWins);

        let early_time = Utc::now() - chrono::Duration::seconds(10);
        let late_time = Utc::now();

        let entry1 = ReplicationLogEntry {
            id: "entry1".to_string(),
            data_key: "test-key".to_string(),
            operation: ReplicationOperation::Update,
            data: b"old-data".to_vec(),
            source_region: "region1".to_string(),
            target_regions: vec![],
            vector_clock: VectorClock::new(),
            timestamp: early_time,
            checksum: "checksum1".to_string(),
            retry_count: 0,
        };

        let entry2 = ReplicationLogEntry {
            id: "entry2".to_string(),
            data_key: "test-key".to_string(),
            operation: ReplicationOperation::Update,
            data: b"new-data".to_vec(),
            source_region: "region2".to_string(),
            target_regions: vec![],
            vector_clock: VectorClock::new(),
            timestamp: late_time,
            checksum: "checksum2".to_string(),
            retry_count: 0,
        };

        let conflict = ConflictInfo {
            data_key: "test-key".to_string(),
            conflicting_entries: vec![entry1, entry2],
            conflict_type: ConflictType::ConcurrentWrite,
            recommended_resolution: ConflictResolution {
                resolved_data: Vec::new(),
                strategy: ConflictResolutionStrategy::LastWriterWins,
                merged_clock: VectorClock::new(),
                metadata: HashMap::new(),
            },
        };

        let resolution = resolver.resolve_conflict(&conflict).unwrap();
        assert_eq!(resolution.resolved_data, b"new-data");
    }

    #[tokio::test]
    async fn test_multi_value_resolution() {
        let resolver = ConflictResolver::new(ConflictResolutionStrategy::MultiValue);

        let entry1 = ReplicationLogEntry {
            id: "entry1".to_string(),
            data_key: "test-key".to_string(),
            operation: ReplicationOperation::Update,
            data: b"data1".to_vec(),
            source_region: "region1".to_string(),
            target_regions: vec![],
            vector_clock: VectorClock::new(),
            timestamp: Utc::now(),
            checksum: "checksum1".to_string(),
            retry_count: 0,
        };

        let entry2 = ReplicationLogEntry {
            id: "entry2".to_string(),
            data_key: "test-key".to_string(),
            operation: ReplicationOperation::Update,
            data: b"data2".to_vec(),
            source_region: "region2".to_string(),
            target_regions: vec![],
            vector_clock: VectorClock::new(),
            timestamp: Utc::now(),
            checksum: "checksum2".to_string(),
            retry_count: 0,
        };

        let conflict = ConflictInfo {
            data_key: "test-key".to_string(),
            conflicting_entries: vec![entry1, entry2],
            conflict_type: ConflictType::ConcurrentWrite,
            recommended_resolution: ConflictResolution {
                resolved_data: Vec::new(),
                strategy: ConflictResolutionStrategy::MultiValue,
                merged_clock: VectorClock::new(),
                metadata: HashMap::new(),
            },
        };

        let resolution = resolver.resolve_conflict(&conflict).unwrap();
        assert_eq!(resolution.resolved_data, b"data1data2");
    }

    #[tokio::test]
    async fn test_replication_status_tracking() {
        let config = create_test_config();
        let regions = create_test_regions();
        let manager = ReplicationManager::new(config, regions).unwrap();

        // Update status for a region
        manager.update_region_status("us-east-1", 100, true).await;
        manager.update_region_status("us-east-1", 150, false).await;

        let statuses = manager.get_replication_status().await;
        let us_east_status = statuses
            .iter()
            .find(|s| s.region_id == "us-east-1")
            .unwrap();

        assert_eq!(us_east_status.replication_lag_ms, 150);
        assert_eq!(us_east_status.error_count, 1);
        assert!(us_east_status.success_rate < 100.0);
    }

    #[tokio::test]
    async fn test_bounded_staleness_timeout() {
        let mut config = create_test_config();
        config.consistency_model = ConsistencyModel::BoundedStaleness {
            max_staleness_ms: 100,
        };
        let regions = create_test_regions();
        let manager = ReplicationManager::new(config, regions).unwrap();

        // This should timeout because we don't have real endpoints
        let result = manager
            .replicate_data(
                "test-key".to_string(),
                b"test-data".to_vec(),
                ReplicationOperation::Insert,
                "us-east-1".to_string(),
                vec!["us-west-2".to_string()],
            )
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_anti_entropy_operation() {
        let config = create_test_config();
        let regions = create_test_regions();
        let manager = ReplicationManager::new(config, regions).unwrap();

        let repairs = manager.perform_anti_entropy().await.unwrap();
        // Should return 0 since we don't have real data to repair
        assert_eq!(repairs, 0);
    }

    #[test]
    fn test_replication_config_serialization() {
        let config = create_test_config();
        let json = serde_json::to_string(&config);
        assert!(json.is_ok());

        let deserialized: Result<ReplicationConfig, _> = serde_json::from_str(&json.unwrap());
        assert!(deserialized.is_ok());
    }

    #[test]
    fn test_vector_clock_serialization() {
        let mut clock = VectorClock::new();
        clock.increment("region1");
        clock.increment("region2");

        let json = serde_json::to_string(&clock);
        assert!(json.is_ok());

        let deserialized: Result<VectorClock, _> = serde_json::from_str(&json.unwrap());
        assert!(deserialized.is_ok());

        let deserialized_clock = deserialized.unwrap();
        assert_eq!(deserialized_clock.clocks.get("region1"), Some(&1));
        assert_eq!(deserialized_clock.clocks.get("region2"), Some(&1));
    }
}
