//! GPU Container implementation

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use uuid::Uuid;

use crate::personality::{
    AgentPersonality, Decision, Outcome, PersonalityInfluence, PersonalityType,
};
use crate::{ContainerState, RuntimeError};

/// Hardware affinity configuration for agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareAffinity {
    pub preferred_gpu_devices: Vec<u32>,
    pub numa_node: Option<u32>,
    pub cpu_core_affinity: Vec<u32>,
    pub memory_tier_preference: Vec<MemoryTier>,
    pub thermal_threshold: Option<f32>,
    pub power_budget: Option<f32>,
}

impl Default for HardwareAffinity {
    fn default() -> Self {
        Self {
            preferred_gpu_devices: vec![0], // Default to first GPU
            numa_node: None,
            cpu_core_affinity: Vec::new(),
            memory_tier_preference: vec![MemoryTier::GPU, MemoryTier::CPU, MemoryTier::NVMe],
            thermal_threshold: Some(80.0), // 80°C default
            power_budget: None,
        }
    }
}

/// Memory tier preferences for agent workloads
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryTier {
    GPU,
    CPU,
    NVMe,
    SSD,
    HDD,
}

/// GPU container for isolated agent execution with personality
#[derive(Debug, Clone)]
pub struct GpuContainer {
    pub id: String,
    pub config: ContainerConfig,
    pub state: Arc<Mutex<ContainerState>>,
    pub created_at: Instant,
    pub memory_handle: Option<stratoswarm_memory::GpuMemoryHandle>,
    pub personality: Arc<Mutex<AgentPersonality>>,
    pub evolution: Arc<Mutex<EvolutionState>>,
    /// Counter for kernel executions
    pub kernel_executions: Arc<AtomicU64>,
    /// Timestamp of last activity
    pub last_activity: Arc<Mutex<Option<Instant>>>,
}

/// Evolution state for containers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionState {
    pub generation: u64,
    pub fitness_history: Vec<f32>,
    pub mutation_count: u64,
    pub crossover_count: u64,
    pub last_evolution: Option<std::time::SystemTime>,
    pub evolution_enabled: bool,
}

impl Default for EvolutionState {
    fn default() -> Self {
        Self {
            generation: 0,
            fitness_history: Vec::new(),
            mutation_count: 0,
            crossover_count: 0,
            last_evolution: None,
            evolution_enabled: true,
        }
    }
}

/// Container configuration with agent personality support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerConfig {
    pub memory_limit_bytes: usize,
    pub gpu_compute_units: usize,
    pub timeout_seconds: Option<u64>,
    pub agent_type: String,
    pub environment: std::collections::HashMap<String, String>,
    pub personality_type: Option<PersonalityType>,
    pub personality: Option<AgentPersonality>,
    pub evolution_enabled: bool,
    pub hardware_affinity: HardwareAffinity,
}

impl Default for ContainerConfig {
    fn default() -> Self {
        Self {
            memory_limit_bytes: 1024 * 1024 * 100, // 100MB default
            gpu_compute_units: 1,
            timeout_seconds: Some(300), // 5 minutes default
            agent_type: "default".to_string(),
            environment: std::collections::HashMap::new(),
            personality_type: Some(PersonalityType::Balanced),
            personality: None, // Will be auto-generated from type
            evolution_enabled: true,
            hardware_affinity: HardwareAffinity::default(),
        }
    }
}

/// Container runtime statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerStats {
    pub container_id: String,
    pub state: String,
    pub uptime_seconds: u64,
    pub memory_usage_bytes: usize,
    pub gpu_utilization_percent: f32,
    pub kernel_executions: u64,
    pub last_activity_seconds: Option<u64>, // Changed from Instant to u64 for serialization
}

impl GpuContainer {
    /// Create a new container with personality and evolution support
    pub fn new(config: ContainerConfig) -> Self {
        let id = Uuid::new_v4().to_string();

        // Initialize personality from config
        let personality = match &config.personality {
            Some(p) => p.clone(),
            None => match config.personality_type {
                Some(ptype) => AgentPersonality::from_type(ptype),
                None => AgentPersonality::default(),
            },
        };

        let evolution_state = EvolutionState {
            evolution_enabled: config.evolution_enabled,
            ..Default::default()
        };

        Self {
            id,
            config,
            state: Arc::new(Mutex::new(ContainerState::Created)),
            created_at: Instant::now(),
            memory_handle: None,
            personality: Arc::new(Mutex::new(personality)),
            evolution: Arc::new(Mutex::new(evolution_state)),
            kernel_executions: Arc::new(AtomicU64::new(0)),
            last_activity: Arc::new(Mutex::new(None)),
        }
    }

    /// Record a kernel execution
    pub fn record_kernel_execution(&self) {
        self.kernel_executions.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut last) = self.last_activity.lock() {
            *last = Some(Instant::now());
        }
    }

    /// Record activity without kernel execution
    pub fn record_activity(&self) {
        if let Ok(mut last) = self.last_activity.lock() {
            *last = Some(Instant::now());
        }
    }

    /// Get container ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get current container state
    pub fn current_state(&self) -> Result<ContainerState, RuntimeError> {
        self.state
            .lock()
            .map(|state| *state)
            .map_err(|e| RuntimeError::StartupFailed {
                reason: format!("Failed to acquire state lock: {e}"),
            })
    }

    /// Set container state
    pub fn set_state(&self, new_state: ContainerState) -> Result<(), RuntimeError> {
        let mut state = self.state.lock().map_err(|e| RuntimeError::StartupFailed {
            reason: format!("Failed to acquire state lock: {e}"),
        })?;

        // Validate state transitions
        match (*state, new_state) {
            (ContainerState::Created, ContainerState::Starting) => {}
            (ContainerState::Starting, ContainerState::Running) => {}
            (ContainerState::Running, ContainerState::Stopping) => {}
            (ContainerState::Stopping, ContainerState::Stopped) => {}
            (ContainerState::Stopped, ContainerState::Starting) => {} // restart
            (from, to) => {
                return Err(RuntimeError::InvalidStateTransition {
                    from: format!("{:?}", from),
                    to: format!("{:?}", to),
                });
            }
        }

        *state = new_state;
        Ok(())
    }

    /// Get container statistics
    pub fn stats(&self) -> Result<ContainerStats, RuntimeError> {
        let state = self.current_state()?;
        let uptime = self.created_at.elapsed();

        // Calculate last activity time
        let last_activity_seconds = self
            .last_activity
            .lock()
            .ok()
            .and_then(|guard| guard.map(|instant| instant.elapsed().as_secs()));

        Ok(ContainerStats {
            container_id: self.id.clone(),
            state: format!("{:?}", state),
            uptime_seconds: uptime.as_secs(),
            memory_usage_bytes: self.memory_handle.as_ref().map(|h| h.size()).unwrap_or(0),
            gpu_utilization_percent: 0.0, // GPU utilization requires driver integration (CUDA/Metal/etc.)
            kernel_executions: self.kernel_executions.load(Ordering::Relaxed),
            last_activity_seconds,
        })
    }

    /// Validate container configuration
    pub fn validate_config(&self) -> Result<(), RuntimeError> {
        if self.config.memory_limit_bytes == 0 {
            return Err(RuntimeError::InvalidConfig {
                reason: "Memory limit cannot be zero".to_string(),
            });
        }

        if self.config.gpu_compute_units == 0 {
            return Err(RuntimeError::InvalidConfig {
                reason: "GPU compute units cannot be zero".to_string(),
            });
        }

        if let Some(timeout) = self.config.timeout_seconds {
            if timeout == 0 {
                return Err(RuntimeError::InvalidConfig {
                    reason: "Timeout cannot be zero if specified".to_string(),
                });
            }
        }

        Ok(())
    }

    /// Get agent personality
    pub fn get_personality(&self) -> Result<AgentPersonality, RuntimeError> {
        self.personality
            .lock()
            .map(|p| p.clone())
            .map_err(|e| RuntimeError::StartupFailed {
                reason: format!("Failed to acquire personality lock: {e}"),
            })
    }

    /// Update agent personality
    pub fn update_personality(&self, personality: AgentPersonality) -> Result<(), RuntimeError> {
        self.personality
            .lock()
            .map(|mut p| *p = personality)
            .map_err(|e| RuntimeError::StartupFailed {
                reason: format!("Failed to acquire personality lock: {e}"),
            })
    }

    /// Make a decision influenced by personality
    pub fn make_decision(&self, mut decision: Decision) -> Result<Decision, RuntimeError> {
        let personality = self.get_personality()?;
        personality.influence_decision(&mut decision);
        Ok(decision)
    }

    /// Update personality from outcome and potentially evolve
    pub fn learn_from_outcome(&self, outcome: &Outcome) -> Result<(), RuntimeError> {
        let mut personality = self
            .personality
            .lock()
            .map_err(|e| RuntimeError::StartupFailed {
                reason: format!("Failed to acquire personality lock: {e}"),
            })?;

        let mut evolution_state =
            self.evolution
                .lock()
                .map_err(|e| RuntimeError::StartupFailed {
                    reason: format!("Failed to acquire evolution lock: {e}"),
                })?;

        // Update personality fitness
        personality.update_from_outcome(outcome);

        // Record fitness in history
        evolution_state.fitness_history.push(personality.fitness);

        // Trigger evolution if conditions are met
        if evolution_state.evolution_enabled
            && self.should_evolve(&evolution_state, &personality)?
        {
            personality
                .mutate()
                .map_err(|e| RuntimeError::StartupFailed {
                    reason: format!("Evolution failed: {e}"),
                })?;

            evolution_state.mutation_count += 1;
            evolution_state.generation = personality.generation;
            evolution_state.last_evolution = Some(std::time::SystemTime::now());
        }

        Ok(())
    }

    /// Check if agent should evolve based on conditions
    fn should_evolve(
        &self,
        evolution_state: &EvolutionState,
        _personality: &AgentPersonality,
    ) -> Result<bool, RuntimeError> {
        // Don't evolve too frequently (at least 10 outcomes between evolutions)
        if evolution_state.fitness_history.len() < 10 {
            return Ok(false);
        }

        // Check if fitness has been stagnant
        let recent_fitness: Vec<f32> = evolution_state
            .fitness_history
            .iter()
            .rev()
            .take(10)
            .cloned()
            .collect();

        let avg_recent = recent_fitness.iter().sum::<f32>() / recent_fitness.len() as f32;
        let fitness_variance = recent_fitness
            .iter()
            .map(|&x| (x - avg_recent).powi(2))
            .sum::<f32>()
            / recent_fitness.len() as f32;

        // Evolve if fitness is stagnant (low variance) or very low
        Ok(fitness_variance < 0.01 || avg_recent < 0.3)
    }

    /// Create offspring by crossing over with another container
    pub fn crossover(&self, other: &GpuContainer) -> Result<AgentPersonality, RuntimeError> {
        let self_personality = self.get_personality()?;
        let other_personality = other.get_personality()?;

        self_personality
            .crossover(&other_personality)
            .map_err(|e| RuntimeError::StartupFailed {
                reason: format!("Crossover failed: {e}"),
            })
    }

    /// Get evolution statistics
    pub fn get_evolution_stats(&self) -> Result<EvolutionState, RuntimeError> {
        self.evolution
            .lock()
            .map(|e| e.clone())
            .map_err(|e| RuntimeError::StartupFailed {
                reason: format!("Failed to acquire evolution lock: {e}"),
            })
    }

    /// Apply hardware-aware resource allocation based on personality
    pub fn optimize_resources(&self) -> Result<(), RuntimeError> {
        let personality = self.get_personality()?;

        // Personality-driven resource optimization would be implemented here
        // This is a placeholder for hardware-aware scheduling integration

        // Example: Aggressive personalities might request more GPU resources
        if personality.risk_tolerance > 0.7 {
            // Request additional GPU compute units if available
            // This would integrate with the kernel modules
        }

        // Example: Conservative personalities prefer stable memory tiers
        if personality.stability_preference > 0.8 {
            // Prefer CPU/NVMe over GPU memory for stability
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AlgorithmChoice, CommunicationPattern, OptimizationTarget, ResourceAllocation};

    #[test]
    fn test_container_creation() {
        let config = ContainerConfig::default();
        let container = GpuContainer::new(config.clone());

        assert!(!container.id().is_empty());
        assert_eq!(
            container.config.memory_limit_bytes,
            config.memory_limit_bytes
        );
        assert!(matches!(
            container.current_state(),
            Ok(ContainerState::Created)
        ));
    }

    #[test]
    fn test_container_state_transitions() {
        let container = GpuContainer::new(ContainerConfig::default());

        // Valid transition: Created -> Starting
        assert!(container.set_state(ContainerState::Starting).is_ok());
        assert!(matches!(
            container.current_state(),
            Ok(ContainerState::Starting)
        ));

        // Valid transition: Starting -> Running
        assert!(container.set_state(ContainerState::Running).is_ok());
        assert!(matches!(
            container.current_state(),
            Ok(ContainerState::Running)
        ));

        // Invalid transition: Running -> Created
        assert!(container.set_state(ContainerState::Created).is_err());
    }

    #[test]
    fn test_config_validation() {
        let mut config = ContainerConfig::default();
        let container = GpuContainer::new(config.clone());
        assert!(container.validate_config().is_ok());

        // Test invalid memory limit
        config.memory_limit_bytes = 0;
        let invalid_container = GpuContainer::new(config);
        let result = invalid_container.validate_config();
        assert!(matches!(result, Err(RuntimeError::InvalidConfig { .. })));
    }

    #[test]
    fn test_config_validation_gpu_compute_units() {
        let mut config = ContainerConfig::default();
        config.gpu_compute_units = 0;
        let container = GpuContainer::new(config);
        let result = container.validate_config();
        assert!(matches!(result, Err(RuntimeError::InvalidConfig { .. })));
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("GPU compute units cannot be zero"));
    }

    #[test]
    fn test_config_validation_timeout() {
        let mut config = ContainerConfig::default();
        config.timeout_seconds = Some(0);
        let container = GpuContainer::new(config);
        let result = container.validate_config();
        assert!(matches!(result, Err(RuntimeError::InvalidConfig { .. })));
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Timeout cannot be zero"));
    }

    #[test]
    fn test_config_validation_none_timeout() {
        let mut config = ContainerConfig::default();
        config.timeout_seconds = None;
        let container = GpuContainer::new(config);
        assert!(container.validate_config().is_ok());
    }

    #[test]
    fn test_invalid_state_transitions() {
        let container = GpuContainer::new(ContainerConfig::default());

        // Try invalid transition: Created -> Running (should go through Starting)
        let result = container.set_state(ContainerState::Running);
        assert!(matches!(
            result,
            Err(RuntimeError::InvalidStateTransition { .. })
        ));

        // Try invalid transition: Created -> Stopping
        let result = container.set_state(ContainerState::Stopping);
        assert!(matches!(
            result,
            Err(RuntimeError::InvalidStateTransition { .. })
        ));

        // Try invalid transition: Created -> Stopped
        let result = container.set_state(ContainerState::Stopped);
        assert!(matches!(
            result,
            Err(RuntimeError::InvalidStateTransition { .. })
        ));
    }

    #[test]
    fn test_state_transitions_from_starting() {
        let container = GpuContainer::new(ContainerConfig::default());

        // Valid: Created -> Starting
        assert!(container.set_state(ContainerState::Starting).is_ok());

        // Invalid from Starting: Starting -> Created
        let result = container.set_state(ContainerState::Created);
        assert!(matches!(
            result,
            Err(RuntimeError::InvalidStateTransition { .. })
        ));

        // Invalid from Starting: Starting -> Stopping
        let result = container.set_state(ContainerState::Stopping);
        assert!(matches!(
            result,
            Err(RuntimeError::InvalidStateTransition { .. })
        ));

        // Invalid from Starting: Starting -> Stopped
        let result = container.set_state(ContainerState::Stopped);
        assert!(matches!(
            result,
            Err(RuntimeError::InvalidStateTransition { .. })
        ));
    }

    #[test]
    fn test_state_transitions_from_running() {
        let container = GpuContainer::new(ContainerConfig::default());

        // Get to Running state
        assert!(container.set_state(ContainerState::Starting).is_ok());
        assert!(container.set_state(ContainerState::Running).is_ok());

        // Invalid from Running: Running -> Created
        let result = container.set_state(ContainerState::Created);
        assert!(matches!(
            result,
            Err(RuntimeError::InvalidStateTransition { .. })
        ));

        // Invalid from Running: Running -> Starting
        let result = container.set_state(ContainerState::Starting);
        assert!(matches!(
            result,
            Err(RuntimeError::InvalidStateTransition { .. })
        ));

        // Invalid from Running: Running -> Stopped (should go through Stopping)
        let result = container.set_state(ContainerState::Stopped);
        assert!(matches!(
            result,
            Err(RuntimeError::InvalidStateTransition { .. })
        ));
    }

    #[test]
    fn test_restart_flow() {
        let container = GpuContainer::new(ContainerConfig::default());

        // Full lifecycle: Created -> Starting -> Running -> Stopping -> Stopped
        assert!(container.set_state(ContainerState::Starting).is_ok());
        assert!(container.set_state(ContainerState::Running).is_ok());
        assert!(container.set_state(ContainerState::Stopping).is_ok());
        assert!(container.set_state(ContainerState::Stopped).is_ok());

        // Test restart: Stopped -> Starting
        assert!(container.set_state(ContainerState::Starting).is_ok());
        assert!(matches!(
            container.current_state(),
            Ok(ContainerState::Starting)
        ));
    }

    #[test]
    fn test_container_with_memory_handle() {
        let mut container = GpuContainer::new(ContainerConfig::default());

        // SAFETY: Creating a null DevicePointer for testing purposes only.
        // This pointer is never dereferenced - it's used purely for stats testing.
        #[cfg(feature = "cuda")]
        let handle = unsafe {
            stratoswarm_memory::GpuMemoryHandle::new_unchecked(
                cust::memory::DevicePointer::from_raw(0u64),
                1024,
                uuid::Uuid::new_v4(),
            )
        };
        #[cfg(not(feature = "cuda"))]
        let handle = unsafe {
            stratoswarm_memory::GpuMemoryHandle::new_unchecked(0usize, 1024, uuid::Uuid::new_v4())
        };

        container.memory_handle = Some(handle);

        let stats = container.stats().expect("Should get stats");
        assert_eq!(stats.memory_usage_bytes, 1024);
    }

    #[test]
    fn test_container_stats_fields() {
        let container = GpuContainer::new(ContainerConfig::default());
        let stats = container.stats().expect("Should get stats");

        assert_eq!(stats.container_id, container.id());
        assert_eq!(stats.state, "Created");
        assert!(stats.uptime_seconds >= 0);
        assert_eq!(stats.memory_usage_bytes, 0);
        assert_eq!(stats.gpu_utilization_percent, 0.0);
        assert_eq!(stats.kernel_executions, 0);
        assert!(stats.last_activity_seconds.is_none());
    }

    #[test]
    fn test_container_stats() {
        let container = GpuContainer::new(ContainerConfig::default());
        let stats = container.stats().expect("Stats should be available");

        assert_eq!(stats.container_id, container.id());
        assert_eq!(stats.state, "Created");
        assert_eq!(stats.memory_usage_bytes, 0);
    }

    #[test]
    fn test_default_config() {
        let config = ContainerConfig::default();
        assert_eq!(config.memory_limit_bytes, 1024 * 1024 * 100);
        assert_eq!(config.gpu_compute_units, 1);
        assert_eq!(config.timeout_seconds, Some(300));
        assert_eq!(config.agent_type, "default");
    }

    #[test]
    fn test_mutex_poisoning_current_state() {
        use crate::test_helpers::tests::create_poisoned_mutex;

        let container = GpuContainer {
            id: "test-container".to_string(),
            config: ContainerConfig::default(),
            state: create_poisoned_mutex(),
            created_at: Instant::now(),
            memory_handle: None,
            personality: Arc::new(Mutex::new(AgentPersonality::default())),
            evolution: Arc::new(Mutex::new(EvolutionState::default())),
            kernel_executions: Arc::new(AtomicU64::new(0)),
            last_activity: Arc::new(Mutex::new(None)),
        };

        let result = container.current_state();
        assert!(result.is_err());

        match result {
            Err(RuntimeError::StartupFailed { reason }) => {
                assert!(reason.contains("Failed to acquire state lock"));
            }
            _ => panic!("Expected StartupFailed error with lock failure"),
        }
    }

    #[test]
    fn test_mutex_poisoning_set_state() {
        use crate::test_helpers::tests::create_poisoned_mutex;

        let container = GpuContainer {
            id: "test-container".to_string(),
            config: ContainerConfig::default(),
            state: create_poisoned_mutex(),
            created_at: Instant::now(),
            memory_handle: None,
            personality: Arc::new(Mutex::new(AgentPersonality::default())),
            evolution: Arc::new(Mutex::new(EvolutionState::default())),
            kernel_executions: Arc::new(AtomicU64::new(0)),
            last_activity: Arc::new(Mutex::new(None)),
        };

        let result = container.set_state(ContainerState::Starting);
        assert!(result.is_err());

        match result {
            Err(RuntimeError::StartupFailed { reason }) => {
                assert!(reason.contains("Failed to acquire state lock"));
            }
            _ => panic!("Expected StartupFailed error with lock failure"),
        }
    }

    #[test]
    fn test_container_config_with_environment_vars() {
        let mut config = ContainerConfig::default();
        config
            .environment
            .insert("GPU_DEVICE".to_string(), "0".to_string());
        config
            .environment
            .insert("AGENT_NAME".to_string(), "test-agent".to_string());
        config
            .environment
            .insert("DEBUG".to_string(), "true".to_string());

        assert_eq!(config.environment.len(), 3);
        assert_eq!(config.environment.get("GPU_DEVICE"), Some(&"0".to_string()));
        assert_eq!(
            config.environment.get("AGENT_NAME"),
            Some(&"test-agent".to_string())
        );
        assert_eq!(config.environment.get("DEBUG"), Some(&"true".to_string()));
    }

    #[test]
    fn test_container_config_edge_values() {
        let config = ContainerConfig {
            memory_limit_bytes: usize::MAX,
            gpu_compute_units: usize::MAX,
            timeout_seconds: Some(u64::MAX),
            agent_type: String::new(),
            environment: std::collections::HashMap::new(),
            personality_type: None,
            personality: None,
            evolution_enabled: false,
            hardware_affinity: HardwareAffinity::default(),
        };

        assert_eq!(config.memory_limit_bytes, usize::MAX);
        assert_eq!(config.gpu_compute_units, usize::MAX);
        assert_eq!(config.timeout_seconds, Some(u64::MAX));
        assert!(config.agent_type.is_empty());
    }

    #[test]
    fn test_container_config_serialization() {
        let mut config = ContainerConfig {
            memory_limit_bytes: 2_147_483_648, // 2GB
            gpu_compute_units: 4,
            timeout_seconds: Some(600),
            agent_type: "ml-inference".to_string(),
            environment: std::collections::HashMap::new(),
            personality_type: None,
            personality: None,
            evolution_enabled: false,
            hardware_affinity: HardwareAffinity::default(),
        };
        config
            .environment
            .insert("MODEL".to_string(), "gpt".to_string());

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ContainerConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.memory_limit_bytes, deserialized.memory_limit_bytes);
        assert_eq!(config.gpu_compute_units, deserialized.gpu_compute_units);
        assert_eq!(config.timeout_seconds, deserialized.timeout_seconds);
        assert_eq!(config.agent_type, deserialized.agent_type);
        assert_eq!(config.environment, deserialized.environment);
    }

    #[test]
    fn test_container_stats_serialization() {
        let stats = ContainerStats {
            container_id: "test-123".to_string(),
            state: "Running".to_string(),
            uptime_seconds: 3600,
            memory_usage_bytes: 1_000_000,
            gpu_utilization_percent: 85.5,
            kernel_executions: 1000,
            last_activity_seconds: Some(10),
        };

        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: ContainerStats = serde_json::from_str(&json).unwrap();

        assert_eq!(stats.container_id, deserialized.container_id);
        assert_eq!(stats.state, deserialized.state);
        assert_eq!(stats.uptime_seconds, deserialized.uptime_seconds);
        assert_eq!(stats.memory_usage_bytes, deserialized.memory_usage_bytes);
        assert_eq!(
            stats.gpu_utilization_percent,
            deserialized.gpu_utilization_percent
        );
        assert_eq!(stats.kernel_executions, deserialized.kernel_executions);
        assert_eq!(
            stats.last_activity_seconds,
            deserialized.last_activity_seconds
        );
    }

    #[test]
    fn test_container_id_generation() {
        let container1 = GpuContainer::new(ContainerConfig::default());
        let container2 = GpuContainer::new(ContainerConfig::default());

        // IDs should be unique
        assert_ne!(container1.id(), container2.id());

        // IDs should be valid UUIDs
        assert!(uuid::Uuid::parse_str(&container1.id()).is_ok());
        assert!(uuid::Uuid::parse_str(&container2.id()).is_ok());
    }

    #[test]
    fn test_container_uptime_calculation() {
        let container = GpuContainer::new(ContainerConfig::default());

        // Sleep for at least 1 second to ensure uptime_seconds > 0
        std::thread::sleep(std::time::Duration::from_millis(1100));

        let stats = container.stats().unwrap();
        assert!(stats.uptime_seconds > 0);
    }

    #[test]
    fn test_container_config_unicode_strings() {
        let mut config = ContainerConfig::default();
        config.agent_type = "代理-エージェント-🚀".to_string();
        config
            .environment
            .insert("中文_KEY".to_string(), "значение".to_string());
        config
            .environment
            .insert("🔑".to_string(), "🎯".to_string());

        assert_eq!(config.agent_type, "代理-エージェント-🚀");
        assert_eq!(
            config.environment.get("中文_KEY"),
            Some(&"значение".to_string())
        );
        assert_eq!(config.environment.get("🔑"), Some(&"🎯".to_string()));
    }

    #[test]
    fn test_container_stats_edge_values() {
        let stats = ContainerStats {
            container_id: String::new(),
            state: String::new(),
            uptime_seconds: u64::MAX,
            memory_usage_bytes: usize::MAX,
            gpu_utilization_percent: f32::INFINITY,
            kernel_executions: u64::MAX,
            last_activity_seconds: None,
        };

        assert!(stats.container_id.is_empty());
        assert!(stats.state.is_empty());
        assert_eq!(stats.uptime_seconds, u64::MAX);
        assert_eq!(stats.memory_usage_bytes, usize::MAX);
        assert!(stats.gpu_utilization_percent.is_infinite());
        assert_eq!(stats.kernel_executions, u64::MAX);
        assert!(stats.last_activity_seconds.is_none());
    }

    #[test]
    fn test_container_clone() {
        let config = ContainerConfig {
            memory_limit_bytes: 1024,
            gpu_compute_units: 2,
            timeout_seconds: Some(60),
            agent_type: "test".to_string(),
            environment: std::collections::HashMap::new(),
            personality_type: None,
            personality: None,
            evolution_enabled: false,
            hardware_affinity: HardwareAffinity::default(),
        };

        let container = GpuContainer::new(config.clone());
        let cloned = container.clone();

        assert_eq!(container.id(), cloned.id());
        assert_eq!(
            container.config.memory_limit_bytes,
            cloned.config.memory_limit_bytes
        );
        assert_eq!(
            container.config.gpu_compute_units,
            cloned.config.gpu_compute_units
        );
        assert_eq!(
            container.config.timeout_seconds,
            cloned.config.timeout_seconds
        );
        assert_eq!(container.config.agent_type, cloned.config.agent_type);
    }

    #[test]
    fn test_container_debug_formatting() {
        let container = GpuContainer::new(ContainerConfig::default());
        let debug_str = format!("{:?}", container);

        assert!(debug_str.contains("GpuContainer"));
        assert!(debug_str.contains(&container.id()));
        assert!(debug_str.contains("ContainerConfig"));
    }

    #[test]
    fn test_container_stats_debug_formatting() {
        let stats = ContainerStats {
            container_id: "debug-test".to_string(),
            state: "Running".to_string(),
            uptime_seconds: 100,
            memory_usage_bytes: 1000,
            gpu_utilization_percent: 75.0,
            kernel_executions: 50,
            last_activity_seconds: Some(5),
        };

        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("ContainerStats"));
        assert!(debug_str.contains("debug-test"));
        assert!(debug_str.contains("Running"));
    }

    #[test]
    fn test_container_concurrent_state_changes() {
        use std::sync::Arc;
        use std::thread;

        let container = Arc::new(GpuContainer::new(ContainerConfig::default()));

        // First set to Starting state
        container.set_state(ContainerState::Starting).unwrap();

        let mut handles = vec![];

        // Try concurrent state changes
        for _ in 0..5 {
            let container_clone = container.clone();
            let handle = thread::spawn(move || container_clone.set_state(ContainerState::Running));
            handles.push(handle);
        }

        // Wait for all threads
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // Only one should succeed since Starting -> Running is valid only once
        // After that, the state is Running and Running -> Running is not allowed
        let success_count = results.iter().filter(|r| r.is_ok()).count();
        assert_eq!(
            success_count, 1,
            "Only one concurrent state change should succeed"
        );

        // Final state should be Running
        assert_eq!(container.current_state().unwrap(), ContainerState::Running);
    }

    // New tests for personality and evolution functionality

    #[test]
    fn test_container_personality_initialization() {
        // Test default personality
        let config = ContainerConfig::default();
        let container = GpuContainer::new(config);

        let personality = container.get_personality().unwrap();
        assert!(personality.risk_tolerance >= 0.0 && personality.risk_tolerance <= 1.0);
        assert!(personality.cooperation >= 0.0 && personality.cooperation <= 1.0);
        assert_eq!(personality.generation, 0);
        assert_eq!(personality.fitness, 0.0);
    }

    #[test]
    fn test_container_custom_personality() {
        let mut config = ContainerConfig::default();
        config.personality_type = Some(PersonalityType::Aggressive);

        let container = GpuContainer::new(config);
        let personality = container.get_personality().unwrap();

        // Aggressive personality should have high risk tolerance and exploration
        assert!(personality.risk_tolerance > 0.8);
        assert!(personality.exploration > 0.7);
    }

    #[test]
    fn test_container_personality_update() {
        let container = GpuContainer::new(ContainerConfig::default());

        let new_personality = AgentPersonality::from_type(PersonalityType::Conservative);
        container
            .update_personality(new_personality.clone())
            .unwrap();

        let retrieved = container.get_personality().unwrap();
        assert_eq!(retrieved.risk_tolerance, new_personality.risk_tolerance);
        assert_eq!(retrieved.cooperation, new_personality.cooperation);
    }

    #[test]
    fn test_container_decision_making() {
        let mut config = ContainerConfig::default();
        config.personality_type = Some(PersonalityType::Aggressive);
        let container = GpuContainer::new(config);

        let decision = Decision {
            resource_allocation: ResourceAllocation {
                memory_request: 1024 * 1024, // 1MB
                cpu_cores: 1,
                gpu_compute_units: 1,
                priority: 5,
            },
            communication_pattern: CommunicationPattern::Broadcast,
            algorithm_choice: AlgorithmChoice::Conservative,
            optimization_target: OptimizationTarget::Balanced,
        };

        let influenced_decision = container.make_decision(decision).unwrap();

        // Aggressive personality should request more memory and prefer experimental algorithms
        assert!(influenced_decision.resource_allocation.memory_request > 1024 * 1024);
        assert_eq!(
            influenced_decision.algorithm_choice,
            AlgorithmChoice::Experimental
        );
        assert_eq!(
            influenced_decision.optimization_target,
            OptimizationTarget::Speed
        );
    }

    #[test]
    fn test_container_learning_from_success() {
        let container = GpuContainer::new(ContainerConfig::default());

        let initial_fitness = container.get_personality().unwrap().fitness;

        // Learn from success
        container
            .learn_from_outcome(&Outcome::Success(0.8))
            .unwrap();

        let updated_fitness = container.get_personality().unwrap().fitness;
        assert!(updated_fitness > initial_fitness);

        let evolution_stats = container.get_evolution_stats().unwrap();
        assert_eq!(evolution_stats.fitness_history.len(), 1);
        assert_eq!(evolution_stats.fitness_history[0], updated_fitness);
    }

    #[test]
    fn test_container_learning_from_failure() {
        let container = GpuContainer::new(ContainerConfig::default());

        // Set initial fitness
        container
            .learn_from_outcome(&Outcome::Success(0.5))
            .unwrap();
        let initial_fitness = container.get_personality().unwrap().fitness;

        // Learn from failure
        container
            .learn_from_outcome(&Outcome::Failure(0.3))
            .unwrap();

        let updated_fitness = container.get_personality().unwrap().fitness;
        assert!(updated_fitness < initial_fitness);
    }

    #[test]
    fn test_container_evolution_trigger() {
        let container = GpuContainer::new(ContainerConfig::default());
        let initial_generation = container.get_personality().unwrap().generation;

        // Generate enough stagnant fitness to trigger evolution
        for _ in 0..15 {
            container
                .learn_from_outcome(&Outcome::Success(0.2))
                .unwrap(); // Low fitness
        }

        let evolution_stats = container.get_evolution_stats().unwrap();
        let final_generation = container.get_personality().unwrap().generation;

        // Should have evolved at least once due to low fitness
        assert!(final_generation > initial_generation);
        assert!(evolution_stats.mutation_count > 0);
        assert!(evolution_stats.last_evolution.is_some());
    }

    #[test]
    fn test_container_crossover() {
        let config1 = ContainerConfig {
            personality_type: Some(PersonalityType::Conservative),
            ..Default::default()
        };
        let config2 = ContainerConfig {
            personality_type: Some(PersonalityType::Aggressive),
            ..Default::default()
        };

        let container1 = GpuContainer::new(config1);
        let container2 = GpuContainer::new(config2);

        let offspring_personality = container1.crossover(&container2).unwrap();

        // Offspring should have valid traits and new generation
        assert!(
            offspring_personality.risk_tolerance >= 0.0
                && offspring_personality.risk_tolerance <= 1.0
        );
        assert!(
            offspring_personality.cooperation >= 0.0 && offspring_personality.cooperation <= 1.0
        );
        assert_eq!(offspring_personality.generation, 1);
        assert_eq!(offspring_personality.fitness, 0.0);
    }

    #[test]
    fn test_container_hardware_affinity() {
        let config = ContainerConfig {
            hardware_affinity: HardwareAffinity {
                preferred_gpu_devices: vec![1, 2],
                numa_node: Some(0),
                cpu_core_affinity: vec![0, 1, 2, 3],
                memory_tier_preference: vec![MemoryTier::GPU, MemoryTier::NVMe],
                thermal_threshold: Some(70.0),
                power_budget: Some(200.0),
            },
            ..Default::default()
        };

        let container = GpuContainer::new(config.clone());

        assert_eq!(
            container.config.hardware_affinity.preferred_gpu_devices,
            vec![1, 2]
        );
        assert_eq!(container.config.hardware_affinity.numa_node, Some(0));
        assert_eq!(
            container.config.hardware_affinity.cpu_core_affinity,
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            container.config.hardware_affinity.memory_tier_preference,
            vec![MemoryTier::GPU, MemoryTier::NVMe]
        );
        assert_eq!(
            container.config.hardware_affinity.thermal_threshold,
            Some(70.0)
        );
        assert_eq!(container.config.hardware_affinity.power_budget, Some(200.0));
    }

    #[test]
    fn test_container_evolution_disabled() {
        let config = ContainerConfig {
            evolution_enabled: false,
            ..Default::default()
        };
        let container = GpuContainer::new(config);

        let initial_generation = container.get_personality().unwrap().generation;

        // Even with multiple learning events, evolution should not occur
        for _ in 0..20 {
            container
                .learn_from_outcome(&Outcome::Success(0.1))
                .unwrap();
        }

        let final_generation = container.get_personality().unwrap().generation;
        let evolution_stats = container.get_evolution_stats().unwrap();

        assert_eq!(final_generation, initial_generation);
        assert_eq!(evolution_stats.mutation_count, 0);
        assert!(evolution_stats.last_evolution.is_none());
        assert!(!evolution_stats.evolution_enabled);
    }

    #[test]
    fn test_container_memory_tier_enum() {
        let tiers = vec![
            MemoryTier::GPU,
            MemoryTier::CPU,
            MemoryTier::NVMe,
            MemoryTier::SSD,
            MemoryTier::HDD,
        ];

        // Test serialization
        for tier in &tiers {
            let json = serde_json::to_string(tier).unwrap();
            let deserialized: MemoryTier = serde_json::from_str(&json).unwrap();
            assert_eq!(*tier, deserialized);
        }

        // Test equality
        assert_eq!(MemoryTier::GPU, MemoryTier::GPU);
        assert_ne!(MemoryTier::GPU, MemoryTier::CPU);
    }

    #[test]
    fn test_container_evolution_state_serialization() {
        let evolution_state = EvolutionState {
            generation: 42,
            fitness_history: vec![0.1, 0.2, 0.3, 0.8],
            mutation_count: 5,
            crossover_count: 2,
            last_evolution: Some(std::time::SystemTime::now()),
            evolution_enabled: true,
        };

        let json = serde_json::to_string(&evolution_state).unwrap();
        let deserialized: EvolutionState = serde_json::from_str(&json).unwrap();

        assert_eq!(evolution_state.generation, deserialized.generation);
        assert_eq!(
            evolution_state.fitness_history,
            deserialized.fitness_history
        );
        assert_eq!(evolution_state.mutation_count, deserialized.mutation_count);
        assert_eq!(
            evolution_state.crossover_count,
            deserialized.crossover_count
        );
        assert_eq!(
            evolution_state.evolution_enabled,
            deserialized.evolution_enabled
        );
    }

    #[test]
    fn test_container_resource_optimization() {
        let container = GpuContainer::new(ContainerConfig::default());

        // This should not fail even though it's a placeholder
        container.optimize_resources().unwrap();

        // Test with different personality types
        let aggressive_config = ContainerConfig {
            personality_type: Some(PersonalityType::Aggressive),
            ..Default::default()
        };
        let aggressive_container = GpuContainer::new(aggressive_config);
        aggressive_container.optimize_resources().unwrap();

        let conservative_config = ContainerConfig {
            personality_type: Some(PersonalityType::Conservative),
            ..Default::default()
        };
        let conservative_container = GpuContainer::new(conservative_config);
        conservative_container.optimize_resources().unwrap();
    }

    #[test]
    fn test_container_cooperative_personality_communication() {
        let config = ContainerConfig {
            personality_type: Some(PersonalityType::Cooperative),
            ..Default::default()
        };
        let container = GpuContainer::new(config);

        let decision = Decision {
            resource_allocation: ResourceAllocation {
                memory_request: 1024,
                cpu_cores: 1,
                gpu_compute_units: 1,
                priority: 5,
            },
            communication_pattern: CommunicationPattern::Minimal,
            algorithm_choice: AlgorithmChoice::Conservative,
            optimization_target: OptimizationTarget::Balanced,
        };

        let influenced_decision = container.make_decision(decision).unwrap();

        // Cooperative personality should prefer cooperative communication
        assert_eq!(
            influenced_decision.communication_pattern,
            CommunicationPattern::Cooperative
        );
    }

    #[test]
    fn test_container_explorer_personality_algorithms() {
        let config = ContainerConfig {
            personality_type: Some(PersonalityType::Explorer),
            ..Default::default()
        };
        let container = GpuContainer::new(config);

        let decision = Decision {
            resource_allocation: ResourceAllocation {
                memory_request: 1024,
                cpu_cores: 1,
                gpu_compute_units: 1,
                priority: 5,
            },
            communication_pattern: CommunicationPattern::Broadcast,
            algorithm_choice: AlgorithmChoice::Conservative,
            optimization_target: OptimizationTarget::Balanced,
        };

        let influenced_decision = container.make_decision(decision).unwrap();

        // Explorer personality should prefer experimental algorithms
        assert_eq!(
            influenced_decision.algorithm_choice,
            AlgorithmChoice::Experimental
        );
    }

    #[test]
    fn test_container_fitness_history_tracking() {
        let container = GpuContainer::new(ContainerConfig::default());

        let outcomes = vec![
            Outcome::Success(0.3),
            Outcome::Success(0.7),
            Outcome::Failure(0.2),
            Outcome::Neutral,
            Outcome::Success(0.9),
        ];

        for outcome in &outcomes {
            container.learn_from_outcome(outcome).unwrap();
        }

        let evolution_stats = container.get_evolution_stats().unwrap();
        assert_eq!(evolution_stats.fitness_history.len(), 5);

        // Each learning event should have updated fitness
        for &fitness in &evolution_stats.fitness_history {
            assert!(fitness >= 0.0 && fitness <= 1.0);
        }
    }

    #[test]
    fn test_container_configuration_serialization() {
        let config = ContainerConfig {
            memory_limit_bytes: 1024 * 1024 * 256, // 256MB
            gpu_compute_units: 4,
            timeout_seconds: Some(600),
            agent_type: "test-agent".to_string(),
            environment: {
                let mut env = std::collections::HashMap::new();
                env.insert("TEST_VAR".to_string(), "test_value".to_string());
                env
            },
            personality_type: Some(PersonalityType::Explorer),
            personality: Some(AgentPersonality::from_type(PersonalityType::Aggressive)),
            evolution_enabled: false,
            hardware_affinity: HardwareAffinity {
                preferred_gpu_devices: vec![0, 1],
                numa_node: Some(1),
                cpu_core_affinity: vec![4, 5, 6, 7],
                memory_tier_preference: vec![MemoryTier::GPU, MemoryTier::CPU],
                thermal_threshold: Some(75.0),
                power_budget: Some(150.0),
            },
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ContainerConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.memory_limit_bytes, deserialized.memory_limit_bytes);
        assert_eq!(config.gpu_compute_units, deserialized.gpu_compute_units);
        assert_eq!(config.timeout_seconds, deserialized.timeout_seconds);
        assert_eq!(config.agent_type, deserialized.agent_type);
        assert_eq!(config.environment, deserialized.environment);
        assert_eq!(config.personality_type, deserialized.personality_type);
        assert_eq!(config.evolution_enabled, deserialized.evolution_enabled);
    }
}
