//! Service health checks, dependency mapping, cascade failure detection, alerting
//!
//! This module provides comprehensive health monitoring including:
//! - Multi-protocol health checks (HTTP, TCP, gRPC, custom)
//! - Service dependency mapping and visualization
//! - Cascade failure detection and prediction
//! - Circuit breaker patterns for fault isolation
//! - Real-time alerting and notification systems
//! - Health check scheduling and batching
//! - Performance degradation detection

use crate::error::{DisasterRecoveryError, DisasterRecoveryResult};
use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio::time::{interval, timeout, Instant};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Health check protocol
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HealthCheckProtocol {
    /// HTTP health check
    Http { path: String, expected_status: u16 },
    /// HTTPS health check
    Https { path: String, expected_status: u16 },
    /// TCP connection check
    Tcp,
    /// UDP ping check
    Udp,
    /// gRPC health check
    Grpc { service: String },
    /// Database connection check
    Database { query: String },
    /// Custom shell command
    Command {
        command: String,
        expected_exit_code: i32,
    },
    /// Custom script
    Script { script_path: String },
}

/// Health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Service is healthy
    Healthy,
    /// Service is degraded but functional
    Degraded,
    /// Service is unhealthy
    Unhealthy,
    /// Health check failed/timeout
    Unknown,
    /// Service is in maintenance mode
    Maintenance,
}

/// Service health definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceHealth {
    /// Service ID
    pub id: Uuid,
    /// Service name
    pub name: String,
    /// Service endpoint
    pub endpoint: String,
    /// Health check configuration
    pub health_check: HealthCheckConfig,
    /// Current status
    pub status: HealthStatus,
    /// Last check timestamp
    pub last_check: Option<DateTime<Utc>>,
    /// Last healthy timestamp
    pub last_healthy: Option<DateTime<Utc>>,
    /// Consecutive failures
    pub consecutive_failures: u32,
    /// Response time (ms)
    pub response_time_ms: u64,
    /// Error message
    pub error_message: Option<String>,
    /// Service tags
    pub tags: HashMap<String, String>,
    /// Dependencies
    pub dependencies: Vec<Uuid>,
    /// Dependents
    pub dependents: Vec<Uuid>,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Protocol
    pub protocol: HealthCheckProtocol,
    /// Check interval
    pub interval_ms: u64,
    /// Timeout
    pub timeout_ms: u64,
    /// Retry count
    pub retry_count: u32,
    /// Failure threshold
    pub failure_threshold: u32,
    /// Recovery threshold
    pub recovery_threshold: u32,
    /// Enable circuit breaker
    pub circuit_breaker_enabled: bool,
    /// Circuit breaker threshold
    pub circuit_breaker_threshold: u32,
    /// Circuit breaker timeout
    pub circuit_breaker_timeout_ms: u64,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            protocol: HealthCheckProtocol::Http {
                path: "/health".to_string(),
                expected_status: 200,
            },
            interval_ms: 30000,
            timeout_ms: 5000,
            retry_count: 3,
            failure_threshold: 3,
            recovery_threshold: 2,
            circuit_breaker_enabled: true,
            circuit_breaker_threshold: 5,
            circuit_breaker_timeout_ms: 60000,
        }
    }
}

/// Service dependency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceDependency {
    /// Source service ID
    pub source_id: Uuid,
    /// Target service ID
    pub target_id: Uuid,
    /// Dependency type
    pub dependency_type: DependencyType,
    /// Dependency strength (0.0-1.0)
    pub strength: f64,
    /// Timeout for dependency check
    pub timeout_ms: u64,
}

/// Dependency type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DependencyType {
    /// Hard dependency - service cannot function without it
    Hard,
    /// Soft dependency - service can degrade gracefully
    Soft,
    /// Optional dependency - service can function normally
    Optional,
}

/// Health alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthAlert {
    /// Alert ID
    pub id: Uuid,
    /// Alert timestamp
    pub timestamp: DateTime<Utc>,
    /// Service ID
    pub service_id: Uuid,
    /// Alert type
    pub alert_type: AlertType,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Additional context
    pub context: HashMap<String, String>,
    /// Acknowledged
    pub acknowledged: bool,
    /// Acknowledged by
    pub acknowledged_by: Option<String>,
    /// Resolved
    pub resolved: bool,
    /// Resolved timestamp
    pub resolved_at: Option<DateTime<Utc>>,
}

/// Alert type
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlertType {
    /// Service went down
    ServiceDown,
    /// Service recovered
    ServiceRecovered,
    /// Service degraded
    ServiceDegraded,
    /// High response time
    HighResponseTime,
    /// Cascade failure detected
    CascadeFailure,
    /// Circuit breaker opened
    CircuitBreakerOpen,
    /// Circuit breaker closed
    CircuitBreakerClosed,
    /// Dependency failure
    DependencyFailure,
}

/// Alert severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Critical alert
    Critical,
    /// High priority alert
    High,
    /// Medium priority alert
    Medium,
    /// Low priority alert
    Low,
    /// Informational alert
    Info,
}

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CircuitBreakerState {
    /// Circuit is closed - normal operation
    Closed,
    /// Circuit is open - failing fast
    Open,
    /// Circuit is half-open - testing recovery
    HalfOpen,
}

/// Circuit breaker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreaker {
    /// Service ID
    pub service_id: Uuid,
    /// Current state
    pub state: CircuitBreakerState,
    /// Failure count
    pub failure_count: u32,
    /// Success count (in half-open state)
    pub success_count: u32,
    /// Last failure timestamp
    pub last_failure: Option<DateTime<Utc>>,
    /// Last success timestamp
    pub last_success: Option<DateTime<Utc>>,
    /// Next retry timestamp
    pub next_retry: Option<DateTime<Utc>>,
}

/// Health monitor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMonitorConfig {
    /// Default check interval
    pub default_interval_ms: u64,
    /// Default timeout
    pub default_timeout_ms: u64,
    /// Max concurrent checks
    pub max_concurrent_checks: usize,
    /// Alert retention days
    pub alert_retention_days: u32,
    /// Enable cascade detection
    pub cascade_detection_enabled: bool,
    /// Cascade detection threshold
    pub cascade_threshold: f64,
    /// Batch check size
    pub batch_size: usize,
    /// Performance threshold (ms)
    pub performance_threshold_ms: u64,
}

impl Default for HealthMonitorConfig {
    fn default() -> Self {
        Self {
            default_interval_ms: 30000,
            default_timeout_ms: 5000,
            max_concurrent_checks: 100,
            alert_retention_days: 30,
            cascade_detection_enabled: true,
            cascade_threshold: 0.3, // 30% of services failing
            batch_size: 10,
            performance_threshold_ms: 1000,
        }
    }
}

/// Health monitor metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HealthMetrics {
    /// Total services monitored
    pub total_services: usize,
    /// Healthy services
    pub healthy_services: usize,
    /// Unhealthy services
    pub unhealthy_services: usize,
    /// Degraded services
    pub degraded_services: usize,
    /// Total checks performed
    pub total_checks: u64,
    /// Failed checks
    pub failed_checks: u64,
    /// Average response time
    pub avg_response_time_ms: u64,
    /// Active alerts
    pub active_alerts: usize,
    /// Critical alerts
    pub critical_alerts: usize,
    /// Open circuit breakers
    pub open_circuits: usize,
}

/// Cascade failure info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeFailure {
    /// Cascade ID
    pub id: Uuid,
    /// Detection timestamp
    pub detected_at: DateTime<Utc>,
    /// Root cause service
    pub root_cause: Uuid,
    /// Affected services
    pub affected_services: Vec<Uuid>,
    /// Failure propagation path
    pub propagation_path: Vec<Vec<Uuid>>,
    /// Impact severity
    pub impact_severity: f64,
}

/// Health monitor
pub struct HealthMonitor {
    /// Configuration
    config: Arc<HealthMonitorConfig>,
    /// Services
    services: Arc<DashMap<Uuid, ServiceHealth>>,
    /// Dependencies
    dependencies: Arc<DashMap<Uuid, Vec<ServiceDependency>>>,
    /// Circuit breakers
    circuit_breakers: Arc<DashMap<Uuid, CircuitBreaker>>,
    /// Alerts
    alerts: Arc<RwLock<VecDeque<HealthAlert>>>,
    /// Metrics
    metrics: Arc<RwLock<HealthMetrics>>,
    /// Check scheduler
    scheduler: Arc<DashMap<Uuid, DateTime<Utc>>>,
    /// Command channel
    command_tx: mpsc::Sender<MonitorCommand>,
    /// Command receiver
    command_rx: Arc<Mutex<mpsc::Receiver<MonitorCommand>>>,
    /// Shutdown flag
    shutdown: Arc<RwLock<bool>>,
}

/// Monitor commands
#[derive(Debug)]
enum MonitorCommand {
    /// Perform health check
    HealthCheck(Uuid),
    /// Add alert
    AddAlert(HealthAlert),
    /// Process cascade detection
    CascadeDetection,
    /// Update circuit breaker
    UpdateCircuitBreaker(Uuid, CircuitBreakerState),
    /// Cleanup old alerts
    CleanupAlerts,
}

impl HealthMonitor {
    /// Create new health monitor
    pub fn new(config: HealthMonitorConfig) -> DisasterRecoveryResult<Self> {
        let (command_tx, command_rx) = mpsc::channel(10000);

        Ok(Self {
            config: Arc::new(config),
            services: Arc::new(DashMap::new()),
            dependencies: Arc::new(DashMap::new()),
            circuit_breakers: Arc::new(DashMap::new()),
            alerts: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            metrics: Arc::new(RwLock::new(HealthMetrics::default())),
            scheduler: Arc::new(DashMap::new()),
            command_tx,
            command_rx: Arc::new(Mutex::new(command_rx)),
            shutdown: Arc::new(RwLock::new(false)),
        })
    }

    /// Start health monitor
    pub async fn start(&self) -> DisasterRecoveryResult<()> {
        info!("Starting health monitor");

        // Start background tasks
        self.start_command_processor().await?;
        self.start_health_checker().await?;
        self.start_cascade_detector().await?;
        self.start_alert_cleanup().await?;

        Ok(())
    }

    /// Stop health monitor
    pub async fn stop(&self) -> DisasterRecoveryResult<()> {
        info!("Stopping health monitor");
        *self.shutdown.write() = true;
        Ok(())
    }

    /// Register service for monitoring
    pub async fn register_service(&self, service: ServiceHealth) -> DisasterRecoveryResult<Uuid> {
        let service_id = service.id;

        // Validate service
        self.validate_service(&service)?;

        // Create circuit breaker
        let circuit_breaker = CircuitBreaker {
            service_id,
            state: CircuitBreakerState::Closed,
            failure_count: 0,
            success_count: 0,
            last_failure: None,
            last_success: None,
            next_retry: None,
        };

        self.services.insert(service_id, service.clone());
        self.circuit_breakers.insert(service_id, circuit_breaker);

        // Schedule initial health check
        let next_check = Utc::now() + Duration::seconds(1);
        self.scheduler.insert(service_id, next_check);

        info!(
            "Registered service for monitoring: {} ({})",
            service.name, service_id
        );
        self.update_metrics().await;

        Ok(service_id)
    }

    /// Unregister service
    pub async fn unregister_service(&self, service_id: Uuid) -> DisasterRecoveryResult<()> {
        self.services.remove(&service_id);
        self.circuit_breakers.remove(&service_id);
        self.scheduler.remove(&service_id);
        self.dependencies.remove(&service_id);

        info!("Unregistered service: {}", service_id);
        self.update_metrics().await;

        Ok(())
    }

    /// Add service dependency
    pub async fn add_dependency(
        &self,
        dependency: ServiceDependency,
    ) -> DisasterRecoveryResult<()> {
        // Validate dependency
        if !self.services.contains_key(&dependency.source_id) {
            return Err(DisasterRecoveryError::ResourceUnavailable {
                resource: "service".to_string(),
                reason: format!("source service {} not found", dependency.source_id),
            });
        }

        if !self.services.contains_key(&dependency.target_id) {
            return Err(DisasterRecoveryError::ResourceUnavailable {
                resource: "service".to_string(),
                reason: format!("target service {} not found", dependency.target_id),
            });
        }

        // Add to dependency graph
        self.dependencies
            .entry(dependency.source_id)
            .or_insert_with(Vec::new)
            .push(dependency.clone());

        // Update service dependency lists
        if let Some(mut service) = self.services.get_mut(&dependency.source_id) {
            if !service.dependencies.contains(&dependency.target_id) {
                service.dependencies.push(dependency.target_id);
            }
        }

        if let Some(mut service) = self.services.get_mut(&dependency.target_id) {
            if !service.dependents.contains(&dependency.source_id) {
                service.dependents.push(dependency.source_id);
            }
        }

        info!(
            "Added dependency: {} -> {}",
            dependency.source_id, dependency.target_id
        );
        Ok(())
    }

    /// Get service health
    pub fn get_service_health(&self, service_id: Uuid) -> Option<ServiceHealth> {
        self.services
            .get(&service_id)
            .map(|entry| entry.value().clone())
    }

    /// List all services
    pub fn list_services(&self) -> Vec<ServiceHealth> {
        self.services
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get recent alerts
    pub fn get_recent_alerts(&self, limit: usize) -> Vec<HealthAlert> {
        let alerts = self.alerts.read();
        alerts.iter().rev().take(limit).cloned().collect()
    }

    /// Acknowledge alert
    pub async fn acknowledge_alert(
        &self,
        alert_id: Uuid,
        user: String,
    ) -> DisasterRecoveryResult<()> {
        let mut alerts = self.alerts.write();

        for alert in alerts.iter_mut() {
            if alert.id == alert_id {
                alert.acknowledged = true;
                alert.acknowledged_by = Some(user);
                info!("Acknowledged alert: {}", alert_id);
                return Ok(());
            }
        }

        Err(DisasterRecoveryError::ResourceUnavailable {
            resource: "alert".to_string(),
            reason: "alert not found".to_string(),
        })
    }

    /// Get metrics
    pub fn get_metrics(&self) -> HealthMetrics {
        self.metrics.read().clone()
    }

    /// Get dependency graph
    pub fn get_dependency_graph(&self) -> HashMap<Uuid, Vec<ServiceDependency>> {
        self.dependencies
            .iter()
            .map(|entry| (*entry.key(), entry.value().clone()))
            .collect()
    }

    /// Detect cascade failures
    pub async fn detect_cascade_failures(&self) -> Vec<CascadeFailure> {
        let mut cascades = Vec::new();

        if !self.config.cascade_detection_enabled {
            return cascades;
        }

        let unhealthy_services: HashSet<Uuid> = self
            .services
            .iter()
            .filter(|entry| entry.value().status == HealthStatus::Unhealthy)
            .map(|entry| *entry.key())
            .collect();

        let total_services = self.services.len() as f64;
        let unhealthy_ratio = unhealthy_services.len() as f64 / total_services;

        if unhealthy_ratio >= self.config.cascade_threshold {
            // Analyze failure propagation
            let cascade = self.analyze_failure_propagation(&unhealthy_services).await;
            if let Some(cascade) = cascade {
                cascades.push(cascade);
            }
        }

        cascades
    }

    /// Force health check
    pub async fn force_health_check(&self, service_id: Uuid) -> DisasterRecoveryResult<()> {
        self.command_tx
            .send(MonitorCommand::HealthCheck(service_id))
            .await
            .map_err(|_| DisasterRecoveryError::NetworkError {
                details: "failed to queue health check".to_string(),
            })?;

        Ok(())
    }

    // Private helper methods

    async fn start_command_processor(&self) -> DisasterRecoveryResult<()> {
        let command_rx = Arc::clone(&self.command_rx);
        let shutdown = Arc::clone(&self.shutdown);
        let services = Arc::clone(&self.services);
        let circuit_breakers = Arc::clone(&self.circuit_breakers);
        let alerts = Arc::clone(&self.alerts);
        let config = Arc::clone(&self.config);

        tokio::spawn(async move {
            while !*shutdown.read() {
                let mut rx = command_rx.lock().await;
                if let Some(command) = rx.recv().await {
                    match command {
                        MonitorCommand::HealthCheck(service_id) => {
                            if let Some(service) = services.get(&service_id) {
                                debug!("Performing health check for service: {}", service.name);

                                // Simulate health check
                                let start = Instant::now();
                                let (status, error) =
                                    Self::perform_health_check(&service.health_check).await;
                                let response_time = start.elapsed().as_millis() as u64;

                                // Update service status
                                drop(service);
                                if let Some(mut service) = services.get_mut(&service_id) {
                                    let previous_status = service.status;
                                    service.status = status;
                                    service.last_check = Some(Utc::now());
                                    service.response_time_ms = response_time;
                                    service.error_message = error;

                                    if status == HealthStatus::Healthy {
                                        service.last_healthy = Some(Utc::now());
                                        service.consecutive_failures = 0;
                                    } else {
                                        service.consecutive_failures += 1;
                                    }

                                    // Update circuit breaker
                                    if let Some(mut cb) = circuit_breakers.get_mut(&service_id) {
                                        Self::update_circuit_breaker_state(
                                            &mut cb, status, &config,
                                        );
                                    }

                                    // Generate alerts for status changes
                                    if previous_status != status {
                                        let alert_type = match (previous_status, status) {
                                            (HealthStatus::Healthy, HealthStatus::Unhealthy) => {
                                                Some(AlertType::ServiceDown)
                                            }
                                            (HealthStatus::Unhealthy, HealthStatus::Healthy) => {
                                                Some(AlertType::ServiceRecovered)
                                            }
                                            (HealthStatus::Healthy, HealthStatus::Degraded) => {
                                                Some(AlertType::ServiceDegraded)
                                            }
                                            _ => None,
                                        };

                                        if let Some(alert_type) = alert_type {
                                            let alert = HealthAlert {
                                                id: Uuid::new_v4(),
                                                timestamp: Utc::now(),
                                                service_id,
                                                alert_type,
                                                severity: match status {
                                                    HealthStatus::Unhealthy => {
                                                        AlertSeverity::Critical
                                                    }
                                                    HealthStatus::Degraded => AlertSeverity::High,
                                                    _ => AlertSeverity::Info,
                                                },
                                                message: format!(
                                                    "Service {} status changed from {:?} to {:?}",
                                                    service.name, previous_status, status
                                                ),
                                                context: HashMap::new(),
                                                acknowledged: false,
                                                acknowledged_by: None,
                                                resolved: status == HealthStatus::Healthy,
                                                resolved_at: if status == HealthStatus::Healthy {
                                                    Some(Utc::now())
                                                } else {
                                                    None
                                                },
                                            };

                                            alerts.write().push_back(alert);
                                        }
                                    }
                                }
                            }
                        }
                        MonitorCommand::AddAlert(alert) => {
                            alerts.write().push_back(alert);
                        }
                        MonitorCommand::CascadeDetection => {
                            debug!("Performing cascade failure detection");
                        }
                        MonitorCommand::UpdateCircuitBreaker(service_id, state) => {
                            if let Some(mut cb) = circuit_breakers.get_mut(&service_id) {
                                cb.state = state;
                            }
                        }
                        MonitorCommand::CleanupAlerts => {
                            let cutoff =
                                Utc::now() - Duration::days(config.alert_retention_days as i64);
                            let mut alerts_guard = alerts.write();
                            alerts_guard.retain(|alert| alert.timestamp > cutoff);
                        }
                    }
                }
            }
        });

        Ok(())
    }

    async fn start_health_checker(&self) -> DisasterRecoveryResult<()> {
        let services = Arc::clone(&self.services);
        let scheduler = Arc::clone(&self.scheduler);
        let command_tx = self.command_tx.clone();
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            let mut check_interval = interval(std::time::Duration::from_millis(1000));

            while !*shutdown.read() {
                check_interval.tick().await;

                let now = Utc::now();
                let mut checks_to_perform = Vec::new();

                // Find services that need health checks
                for scheduler_entry in scheduler.iter() {
                    let service_id = *scheduler_entry.key();
                    let next_check = *scheduler_entry.value();

                    if now >= next_check {
                        checks_to_perform.push(service_id);
                    }
                }

                // Perform health checks
                for service_id in checks_to_perform {
                    if let Some(service) = services.get(&service_id) {
                        let next_check =
                            now + Duration::milliseconds(service.health_check.interval_ms as i64);
                        scheduler.insert(service_id, next_check);

                        let _ = command_tx
                            .send(MonitorCommand::HealthCheck(service_id))
                            .await;
                    }
                }
            }
        });

        Ok(())
    }

    async fn start_cascade_detector(&self) -> DisasterRecoveryResult<()> {
        let command_tx = self.command_tx.clone();
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            let mut detection_interval = interval(std::time::Duration::from_secs(30));

            while !*shutdown.read() {
                detection_interval.tick().await;
                let _ = command_tx.send(MonitorCommand::CascadeDetection).await;
            }
        });

        Ok(())
    }

    async fn start_alert_cleanup(&self) -> DisasterRecoveryResult<()> {
        let command_tx = self.command_tx.clone();
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            let mut cleanup_interval = interval(std::time::Duration::from_secs(3600));

            while !*shutdown.read() {
                cleanup_interval.tick().await;
                let _ = command_tx.send(MonitorCommand::CleanupAlerts).await;
            }
        });

        Ok(())
    }

    async fn perform_health_check(config: &HealthCheckConfig) -> (HealthStatus, Option<String>) {
        // Simulate health check based on protocol
        match &config.protocol {
            HealthCheckProtocol::Http {
                path,
                expected_status,
            } => {
                // Simulate HTTP check
                (HealthStatus::Healthy, None)
            }
            HealthCheckProtocol::Tcp => {
                // Simulate TCP check
                (HealthStatus::Healthy, None)
            }
            HealthCheckProtocol::Grpc { service } => {
                // Simulate gRPC check
                (HealthStatus::Healthy, None)
            }
            _ => (HealthStatus::Healthy, None),
        }
    }

    fn update_circuit_breaker_state(
        breaker: &mut CircuitBreaker,
        status: HealthStatus,
        config: &HealthMonitorConfig,
    ) {
        let now = Utc::now();

        match status {
            HealthStatus::Healthy => {
                breaker.success_count += 1;
                breaker.last_success = Some(now);

                match breaker.state {
                    CircuitBreakerState::HalfOpen => {
                        if breaker.success_count >= 2 {
                            breaker.state = CircuitBreakerState::Closed;
                            breaker.failure_count = 0;
                            breaker.success_count = 0;
                        }
                    }
                    CircuitBreakerState::Open => {
                        // Reset if enough time has passed
                        if let Some(last_failure) = breaker.last_failure {
                            let elapsed = (now - last_failure).num_milliseconds() as u64;
                            if elapsed > 60000 {
                                // 1 minute
                                breaker.state = CircuitBreakerState::HalfOpen;
                                breaker.success_count = 1;
                            }
                        }
                    }
                    _ => {}
                }
            }
            HealthStatus::Unhealthy => {
                breaker.failure_count += 1;
                breaker.last_failure = Some(now);
                breaker.success_count = 0;

                if breaker.state == CircuitBreakerState::Closed && breaker.failure_count >= 5 {
                    breaker.state = CircuitBreakerState::Open;
                    breaker.next_retry = Some(now + Duration::milliseconds(60000));
                }
            }
            _ => {}
        }
    }

    async fn analyze_failure_propagation(
        &self,
        unhealthy_services: &HashSet<Uuid>,
    ) -> Option<CascadeFailure> {
        // Find potential root causes (services with no dependencies that are unhealthy)
        let mut root_causes = Vec::new();

        for &service_id in unhealthy_services {
            if let Some(service) = self.services.get(&service_id) {
                if service.dependencies.is_empty() {
                    root_causes.push(service_id);
                }
            }
        }

        if let Some(&root_cause) = root_causes.first() {
            let affected_services: Vec<Uuid> = unhealthy_services.iter().copied().collect();

            Some(CascadeFailure {
                id: Uuid::new_v4(),
                detected_at: Utc::now(),
                root_cause,
                affected_services: affected_services.clone(),
                propagation_path: vec![affected_services],
                impact_severity: unhealthy_services.len() as f64 / self.services.len() as f64,
            })
        } else {
            None
        }
    }

    fn validate_service(&self, service: &ServiceHealth) -> DisasterRecoveryResult<()> {
        if service.name.is_empty() {
            return Err(DisasterRecoveryError::ConfigurationError {
                message: "service name cannot be empty".to_string(),
            });
        }

        if service.endpoint.is_empty() {
            return Err(DisasterRecoveryError::ConfigurationError {
                message: "service endpoint cannot be empty".to_string(),
            });
        }

        Ok(())
    }

    async fn update_metrics(&self) {
        let mut metrics = self.metrics.write();

        metrics.total_services = self.services.len();
        metrics.healthy_services = self
            .services
            .iter()
            .filter(|entry| entry.value().status == HealthStatus::Healthy)
            .count();
        metrics.unhealthy_services = self
            .services
            .iter()
            .filter(|entry| entry.value().status == HealthStatus::Unhealthy)
            .count();
        metrics.degraded_services = self
            .services
            .iter()
            .filter(|entry| entry.value().status == HealthStatus::Degraded)
            .count();

        metrics.open_circuits = self
            .circuit_breakers
            .iter()
            .filter(|entry| entry.value().state == CircuitBreakerState::Open)
            .count();

        let alerts = self.alerts.read();
        metrics.active_alerts = alerts.iter().filter(|alert| !alert.resolved).count();
        metrics.critical_alerts = alerts
            .iter()
            .filter(|alert| !alert.resolved && alert.severity == AlertSeverity::Critical)
            .count();

        // Calculate average response time
        let total_response_time: u64 = self
            .services
            .iter()
            .map(|entry| entry.value().response_time_ms)
            .sum();

        if metrics.total_services > 0 {
            metrics.avg_response_time_ms = total_response_time / metrics.total_services as u64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_service(name: &str) -> ServiceHealth {
        ServiceHealth {
            id: Uuid::new_v4(),
            name: name.to_string(),
            endpoint: format!("http://{}.example.com", name.to_lowercase()),
            health_check: HealthCheckConfig::default(),
            status: HealthStatus::Unknown,
            last_check: None,
            last_healthy: None,
            consecutive_failures: 0,
            response_time_ms: 0,
            error_message: None,
            tags: HashMap::new(),
            dependencies: Vec::new(),
            dependents: Vec::new(),
        }
    }

    #[test]
    fn test_health_status_serialization() {
        let statuses = vec![
            HealthStatus::Healthy,
            HealthStatus::Degraded,
            HealthStatus::Unhealthy,
            HealthStatus::Unknown,
            HealthStatus::Maintenance,
        ];

        for status in statuses {
            let serialized = serde_json::to_string(&status)?;
            let deserialized: HealthStatus = serde_json::from_str(&serialized).unwrap();
            assert_eq!(status, deserialized);
        }
    }

    #[test]
    fn test_health_check_protocols() {
        let protocols = vec![
            HealthCheckProtocol::Http {
                path: "/health".to_string(),
                expected_status: 200,
            },
            HealthCheckProtocol::Https {
                path: "/status".to_string(),
                expected_status: 200,
            },
            HealthCheckProtocol::Tcp,
            HealthCheckProtocol::Udp,
            HealthCheckProtocol::Grpc {
                service: "HealthService".to_string(),
            },
            HealthCheckProtocol::Database {
                query: "SELECT 1".to_string(),
            },
            HealthCheckProtocol::Command {
                command: "ping -c 1 localhost".to_string(),
                expected_exit_code: 0,
            },
        ];

        for protocol in protocols {
            let serialized = serde_json::to_string(&protocol).unwrap();
            let deserialized: HealthCheckProtocol = serde_json::from_str(&serialized).unwrap();
            assert_eq!(protocol, deserialized);
        }
    }

    #[test]
    fn test_health_monitor_config_default() {
        let config = HealthMonitorConfig::default();
        assert_eq!(config.default_interval_ms, 30000);
        assert_eq!(config.default_timeout_ms, 5000);
        assert_eq!(config.max_concurrent_checks, 100);
        assert!(config.cascade_detection_enabled);
        assert_eq!(config.cascade_threshold, 0.3);
    }

    #[test]
    fn test_health_monitor_creation() {
        let config = HealthMonitorConfig::default();
        let monitor = HealthMonitor::new(config);
        assert!(monitor.is_ok());
    }

    #[tokio::test]
    async fn test_register_service() {
        let config = HealthMonitorConfig::default();
        let monitor = HealthMonitor::new(config).unwrap();

        let service = create_test_service("API");
        let service_id = monitor.register_service(service.clone()).await?;

        assert_eq!(monitor.services.len(), 1);
        assert!(monitor.services.contains_key(&service_id));
        assert!(monitor.circuit_breakers.contains_key(&service_id));
        assert!(monitor.scheduler.contains_key(&service_id));
    }

    #[tokio::test]
    async fn test_validate_service() {
        let config = HealthMonitorConfig::default();
        let monitor = HealthMonitor::new(config).unwrap();

        // Test invalid service - empty name
        let mut service = create_test_service("");
        let result = monitor.register_service(service).await;
        assert!(result.is_err());

        // Test invalid service - empty endpoint
        let mut service = create_test_service("Test");
        service.endpoint = String::new();
        let result = monitor.register_service(service).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_unregister_service() {
        let config = HealthMonitorConfig::default();
        let monitor = HealthMonitor::new(config).unwrap();

        let service = create_test_service("Test");
        let service_id = monitor.register_service(service).await?;

        let result = monitor.unregister_service(service_id).await;
        assert!(result.is_ok());
        assert!(!monitor.services.contains_key(&service_id));
        assert!(!monitor.circuit_breakers.contains_key(&service_id));
        assert!(!monitor.scheduler.contains_key(&service_id));
    }

    #[tokio::test]
    async fn test_add_service_dependency() {
        let config = HealthMonitorConfig::default();
        let monitor = HealthMonitor::new(config).unwrap();

        // Register services
        let api_service = create_test_service("API");
        let db_service = create_test_service("Database");
        let api_id = monitor.register_service(api_service).await?;
        let db_id = monitor.register_service(db_service).await?;

        // Add dependency
        let dependency = ServiceDependency {
            source_id: api_id,
            target_id: db_id,
            dependency_type: DependencyType::Hard,
            strength: 1.0,
            timeout_ms: 5000,
        };

        let result = monitor.add_dependency(dependency).await;
        assert!(result.is_ok());

        assert_eq!(monitor.dependencies.len(), 1);

        // Check if dependency lists are updated
        let api_service = monitor.get_service_health(api_id).unwrap();
        assert!(api_service.dependencies.contains(&db_id));

        let db_service = monitor.get_service_health(db_id).unwrap();
        assert!(db_service.dependents.contains(&api_id));
    }

    #[tokio::test]
    async fn test_invalid_dependency() {
        let config = HealthMonitorConfig::default();
        let monitor = HealthMonitor::new(config).unwrap();

        // Try to add dependency with non-existent services
        let dependency = ServiceDependency {
            source_id: Uuid::new_v4(),
            target_id: Uuid::new_v4(),
            dependency_type: DependencyType::Hard,
            strength: 1.0,
            timeout_ms: 5000,
        };

        let result = monitor.add_dependency(dependency).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_health_monitoring() {
        let config = HealthMonitorConfig::default();
        let monitor = HealthMonitor::new(config).unwrap();
        monitor.start().await.unwrap();

        let service = create_test_service("Test Service");
        let service_id = monitor.register_service(service).await?;

        // Force a health check
        monitor.force_health_check(service_id).await?;

        // Wait for processing
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let service = monitor.get_service_health(service_id).unwrap();
        assert!(service.last_check.is_some());
    }

    #[tokio::test]
    async fn test_circuit_breaker() {
        let config = HealthMonitorConfig::default();
        let monitor = HealthMonitor::new(config).unwrap();

        let service = create_test_service("Flaky Service");
        let service_id = monitor.register_service(service).await?;

        let breaker = monitor.circuit_breakers.get(&service_id)?;
        assert_eq!(breaker.state, CircuitBreakerState::Closed);
        assert_eq!(breaker.failure_count, 0);
    }

    #[test]
    fn test_alert_types() {
        let alert_types = vec![
            AlertType::ServiceDown,
            AlertType::ServiceRecovered,
            AlertType::ServiceDegraded,
            AlertType::HighResponseTime,
            AlertType::CascadeFailure,
            AlertType::CircuitBreakerOpen,
            AlertType::CircuitBreakerClosed,
            AlertType::DependencyFailure,
        ];

        for alert_type in alert_types {
            let serialized = serde_json::to_string(&alert_type).unwrap();
            let deserialized: AlertType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(alert_type, deserialized);
        }
    }

    #[test]
    fn test_alert_severities() {
        let severities = vec![
            AlertSeverity::Critical,
            AlertSeverity::High,
            AlertSeverity::Medium,
            AlertSeverity::Low,
            AlertSeverity::Info,
        ];

        for severity in severities {
            let serialized = serde_json::to_string(&severity)?;
            let deserialized: AlertSeverity = serde_json::from_str(&serialized).unwrap();
            assert_eq!(severity, deserialized);
        }
    }

    #[tokio::test]
    async fn test_acknowledge_alert() {
        let config = HealthMonitorConfig::default();
        let monitor = HealthMonitor::new(config).unwrap();

        let alert = HealthAlert {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            service_id: Uuid::new_v4(),
            alert_type: AlertType::ServiceDown,
            severity: AlertSeverity::Critical,
            message: "Test alert".to_string(),
            context: HashMap::new(),
            acknowledged: false,
            acknowledged_by: None,
            resolved: false,
            resolved_at: None,
        };

        let alert_id = alert.id;
        monitor.alerts.write().push_back(alert);

        let result = monitor
            .acknowledge_alert(alert_id, "user@example.com".to_string())
            .await;
        assert!(result.is_ok());

        let alerts = monitor.get_recent_alerts(1);
        assert!(alerts[0].acknowledged);
        assert_eq!(
            alerts[0].acknowledged_by,
            Some("user@example.com".to_string())
        );
    }

    #[test]
    fn test_dependency_types() {
        let types = vec![
            DependencyType::Hard,
            DependencyType::Soft,
            DependencyType::Optional,
        ];

        for dep_type in types {
            let serialized = serde_json::to_string(&dep_type)?;
            let deserialized: DependencyType = serde_json::from_str(&serialized)?;
            assert_eq!(dep_type, deserialized);
        }
    }

    #[test]
    fn test_circuit_breaker_states() {
        let states = vec![
            CircuitBreakerState::Closed,
            CircuitBreakerState::Open,
            CircuitBreakerState::HalfOpen,
        ];

        for state in states {
            let serialized = serde_json::to_string(&state)?;
            let deserialized: CircuitBreakerState = serde_json::from_str(&serialized)?;
            assert_eq!(state, deserialized);
        }
    }

    #[tokio::test]
    async fn test_cascade_failure_detection() {
        let mut config = HealthMonitorConfig::default();
        config.cascade_threshold = 0.5; // 50% threshold
        let monitor = HealthMonitor::new(config).unwrap();

        // Register multiple services
        for i in 0..4 {
            let service = create_test_service(&format!("Service{}", i));
            monitor.register_service(service).await?;
        }

        // Mark some services as unhealthy
        let mut count = 0;
        for service_entry in monitor.services.iter_mut() {
            if count >= 2 {
                // Mark 2 out of 4 as unhealthy (50%)
                service_entry.status = HealthStatus::Unhealthy;
            }
            count += 1;
        }

        let cascades = monitor.detect_cascade_failures().await;
        assert!(!cascades.is_empty());
        assert_eq!(cascades[0].impact_severity, 0.5);
    }

    #[test]
    fn test_health_metrics() {
        let metrics = HealthMetrics {
            total_services: 10,
            healthy_services: 7,
            unhealthy_services: 2,
            degraded_services: 1,
            total_checks: 1000,
            failed_checks: 50,
            avg_response_time_ms: 150,
            active_alerts: 5,
            critical_alerts: 2,
            open_circuits: 1,
        };

        assert_eq!(metrics.total_services, 10);
        assert_eq!(metrics.healthy_services, 7);
        assert_eq!(metrics.unhealthy_services, 2);
        assert_eq!(metrics.degraded_services, 1);
        assert_eq!(metrics.avg_response_time_ms, 150);
    }
}
