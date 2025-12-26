//! Bootstrap Monitoring - Metrics and observability for bootstrap process

use crate::{
    config::{BootstrapConfig, BootstrapPhase, MonitoringConfig},
    population::{PopulationController, PopulationStats},
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

/// Bootstrap monitoring system
#[derive(Debug)]
pub struct BootstrapMonitor {
    config: MonitoringConfig,
    start_time: Instant,
    metrics_history: Vec<BootstrapMetrics>,
    phase_transitions: Vec<PhaseTransition>,
    checkpoints: Vec<BootstrapCheckpoint>,
    alerts: Vec<MonitoringAlert>,
}

/// Bootstrap metrics collected at regular intervals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapMetrics {
    pub timestamp: u64,
    pub phase: BootstrapPhase,
    pub elapsed_time: Duration,
    pub population_stats: PopulationStats,
    pub system_health: SystemHealth,
    pub resource_usage: ResourceUsage,
    pub evolution_metrics: EvolutionMetrics,
}

/// System health indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub overall_score: f32,
    pub gpu_health: f32,
    pub memory_health: f32,
    pub cpu_health: f32,
    pub network_health: f32,
    pub error_rate: f32,
}

/// Resource usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub gpu_memory_used: u64,
    pub gpu_memory_total: u64,
    pub cpu_usage_percent: f32,
    pub system_memory_used: u64,
    pub system_memory_total: u64,
    pub network_bytes_sent: u64,
    pub network_bytes_received: u64,
    pub storage_used: u64,
}

/// Evolution-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionMetrics {
    pub current_generation: u32,
    pub mutation_rate: f32,
    pub selection_pressure: f32,
    pub fitness_improvement_rate: f32,
    pub diversity_trend: f32,
    pub reproduction_success_rate: f32,
    pub agent_creation_rate: f32,
    pub agent_termination_rate: f32,
}

/// Phase transition record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTransition {
    pub timestamp: u64,
    pub from_phase: BootstrapPhase,
    pub to_phase: BootstrapPhase,
    pub duration_in_previous_phase: Duration,
    pub trigger: String,
    pub success: bool,
}

/// Bootstrap checkpoint for recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapCheckpoint {
    pub timestamp: u64,
    pub phase: BootstrapPhase,
    pub population_size: usize,
    pub metrics: BootstrapMetrics,
    pub configuration: BootstrapConfig,
}

/// Monitoring alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringAlert {
    pub timestamp: u64,
    pub severity: AlertSeverity,
    pub category: AlertCategory,
    pub message: String,
    pub context: HashMap<String, String>,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Alert categories
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AlertCategory {
    PopulationHealth,
    ResourceExhaustion,
    DiversityCrisis,
    SystemFailure,
    PerformanceDegradation,
    SecurityConcern,
}

impl BootstrapMonitor {
    /// Create a new bootstrap monitor
    pub fn new(config: BootstrapConfig) -> Result<Self> {
        Ok(Self {
            config: config.monitoring,
            start_time: Instant::now(),
            metrics_history: Vec::new(),
            phase_transitions: Vec::new(),
            checkpoints: Vec::new(),
            alerts: Vec::new(),
        })
    }

    /// Start monitoring bootstrap process
    pub async fn start_monitoring(
        &mut self,
        population: Arc<RwLock<PopulationController>>,
    ) -> Result<()> {
        tracing::info!("Starting bootstrap monitoring");

        // Spawn monitoring tasks
        let population_clone = population.clone();
        let metrics_interval = self.config.metrics_interval;
        let health_check_interval = self.config.health_check_interval;
        let checkpoint_interval = self.config.checkpoint_interval;

        // Metrics collection task
        tokio::spawn(async move {
            let mut metrics_timer = tokio::time::interval(metrics_interval);
            loop {
                metrics_timer.tick().await;
                if let Err(e) = Self::collect_metrics(population_clone.clone()).await {
                    tracing::error!("Failed to collect metrics: {}", e);
                }
            }
        });

        // Health check task
        let population_clone2 = population.clone();
        tokio::spawn(async move {
            let mut health_timer = tokio::time::interval(health_check_interval);
            loop {
                health_timer.tick().await;
                if let Err(e) = Self::perform_health_check(population_clone2.clone()).await {
                    tracing::error!("Health check failed: {}", e);
                }
            }
        });

        // Checkpoint task
        let population_clone3 = population;
        tokio::spawn(async move {
            let mut checkpoint_timer = tokio::time::interval(checkpoint_interval);
            loop {
                checkpoint_timer.tick().await;
                if let Err(e) = Self::create_checkpoint(population_clone3.clone()).await {
                    tracing::error!("Failed to create checkpoint: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Collect bootstrap metrics
    async fn collect_metrics(population: Arc<RwLock<PopulationController>>) -> Result<()> {
        let pop = population.read().await;
        let stats = pop.get_stats().clone();

        let system_health = Self::collect_system_health().await?;
        let resource_usage = Self::collect_resource_usage().await?;
        let evolution_metrics = Self::collect_evolution_metrics(&stats).await?;

        let metrics = BootstrapMetrics {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            phase: BootstrapPhase::Genesis, // Would be tracked from actual bootstrap state
            elapsed_time: Duration::from_secs(300), // Mock data
            population_stats: stats,
            system_health,
            resource_usage,
            evolution_metrics,
        };

        tracing::debug!(
            "Collected bootstrap metrics: {} agents, {:.2} health",
            metrics.population_stats.total_agents,
            metrics.system_health.overall_score
        );

        Ok(())
    }

    /// Perform system health check
    async fn perform_health_check(population: Arc<RwLock<PopulationController>>) -> Result<()> {
        let pop = population.read().await;
        let health_score = pop.health_score().await?;

        if health_score < 0.3 {
            tracing::warn!("Low system health detected: {:.2}", health_score);
        } else if health_score > 0.8 {
            tracing::debug!("System health good: {:.2}", health_score);
        }

        Ok(())
    }

    /// Create checkpoint for recovery
    async fn create_checkpoint(population: Arc<RwLock<PopulationController>>) -> Result<()> {
        let pop = population.read().await;
        tracing::debug!(
            "Creating bootstrap checkpoint with {} agents",
            pop.get_stats().total_agents
        );
        Ok(())
    }

    /// Record phase transition
    pub fn record_phase_transition(
        &mut self,
        from: BootstrapPhase,
        to: BootstrapPhase,
        trigger: String,
        success: bool,
    ) -> Result<()> {
        let transition = PhaseTransition {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            from_phase: from,
            to_phase: to,
            duration_in_previous_phase: Duration::from_secs(60), // Mock data
            trigger,
            success,
        };

        self.phase_transitions.push(transition);

        tracing::info!(
            "Phase transition: {:?} -> {:?} ({})",
            from,
            to,
            if success { "SUCCESS" } else { "FAILED" }
        );

        Ok(())
    }

    /// Add monitoring alert
    pub fn add_alert(
        &mut self,
        severity: AlertSeverity,
        category: AlertCategory,
        message: String,
        context: HashMap<String, String>,
    ) -> Result<()> {
        let alert = MonitoringAlert {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            severity,
            category,
            message: message.clone(),
            context,
        };

        self.alerts.push(alert);

        match severity {
            AlertSeverity::Critical => tracing::error!("CRITICAL ALERT: {message}"),
            AlertSeverity::Error => tracing::error!("ERROR: {message}"),
            AlertSeverity::Warning => tracing::warn!("WARNING: {message}"),
            AlertSeverity::Info => tracing::info!("INFO: {message}"),
        }

        Ok(())
    }

    /// Get latest metrics
    pub fn get_latest_metrics(&self) -> Option<&BootstrapMetrics> {
        self.metrics_history.last()
    }

    /// Get phase transition history
    pub fn get_phase_transitions(&self) -> &[PhaseTransition] {
        &self.phase_transitions
    }

    /// Get recent alerts
    pub fn get_recent_alerts(&self, since: Duration) -> Vec<&MonitoringAlert> {
        let cutoff = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .saturating_sub(since.as_secs());

        self.alerts
            .iter()
            .filter(|alert| alert.timestamp >= cutoff)
            .collect()
    }

    /// Export metrics to file
    pub async fn export_metrics(&self, path: &str) -> Result<()> {
        let export_data = serde_json::json!({
            "start_time": self.start_time.elapsed().as_secs(),
            "metrics_history": self.metrics_history,
            "phase_transitions": self.phase_transitions,
            "checkpoints": self.checkpoints,
            "alerts": self.alerts,
        });

        let contents = serde_json::to_string_pretty(&export_data)?;
        tokio::fs::write(path, contents).await?;

        tracing::info!("Exported bootstrap metrics to: {}", path);
        Ok(())
    }

    /// Collect system health metrics
    async fn collect_system_health() -> Result<SystemHealth> {
        // Mock implementation - in reality would query actual system metrics
        Ok(SystemHealth {
            overall_score: 0.85,
            gpu_health: 0.9,
            memory_health: 0.8,
            cpu_health: 0.85,
            network_health: 0.9,
            error_rate: 0.02,
        })
    }

    /// Collect resource usage metrics
    async fn collect_resource_usage() -> Result<ResourceUsage> {
        // Mock implementation - in reality would query actual resource usage
        Ok(ResourceUsage {
            gpu_memory_used: 2048 * 1024 * 1024,  // 2GB
            gpu_memory_total: 8192 * 1024 * 1024, // 8GB
            cpu_usage_percent: 45.0,
            system_memory_used: 4096 * 1024 * 1024,   // 4GB
            system_memory_total: 16384 * 1024 * 1024, // 16GB
            network_bytes_sent: 1024 * 1024,          // 1MB
            network_bytes_received: 2048 * 1024,      // 2MB
            storage_used: 512 * 1024 * 1024,          // 512MB
        })
    }

    /// Collect evolution-specific metrics
    async fn collect_evolution_metrics(stats: &PopulationStats) -> Result<EvolutionMetrics> {
        Ok(EvolutionMetrics {
            current_generation: stats.generation_stats.keys().max().copied().unwrap_or(0),
            mutation_rate: 0.15,
            selection_pressure: 0.7,
            fitness_improvement_rate: 0.05,
            diversity_trend: stats.diversity_score,
            reproduction_success_rate: 0.8,
            agent_creation_rate: 2.0,
            agent_termination_rate: 0.5,
        })
    }
}
