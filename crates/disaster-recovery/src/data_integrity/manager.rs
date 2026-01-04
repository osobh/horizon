//! Main data integrity manager

use crate::error::{DisasterRecoveryError, DisasterRecoveryResult};
use chrono::{Duration, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::path::Path;
use std::sync::Arc;
use tokio::fs;
use tokio::sync::{mpsc, Mutex, Semaphore};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use super::{
    algorithms::*, audit::*, config::*, corruption::*, metrics::*, repair::*, types::*,
    verification::*,
};

/// Data integrity manager
pub struct DataIntegrityManager {
    /// Configuration
    config: Arc<DataIntegrityConfig>,
    /// Monitored objects
    objects: Arc<DashMap<Uuid, DataObject>>,
    /// Verification tasks
    tasks: Arc<DashMap<Uuid, VerificationTask>>,
    /// Corruption detections
    detections: Arc<DashMap<Uuid, CorruptionDetection>>,
    /// Repair records
    repairs: Arc<DashMap<Uuid, RepairRecord>>,
    /// Audit trail
    audit_trail: Arc<RwLock<AuditTrail>>,
    /// Metrics
    metrics: Arc<RwLock<IntegrityMetrics>>,
    /// Verification semaphore
    verification_semaphore: Arc<Semaphore>,
    /// Repair semaphore
    repair_semaphore: Arc<Semaphore>,
    /// Command channel sender
    command_tx: mpsc::Sender<IntegrityCommand>,
    /// Command channel receiver
    command_rx: Arc<Mutex<mpsc::Receiver<IntegrityCommand>>>,
    /// Shutdown flag
    shutdown: Arc<RwLock<bool>>,
}

/// Integrity manager commands
#[derive(Debug)]
enum IntegrityCommand {
    /// Verify object
    Verify(Uuid),
    /// Repair corruption
    Repair(Uuid),
    /// Schedule verification
    Schedule(Uuid),
    /// Stop manager
    Shutdown,
}

impl DataIntegrityManager {
    /// Create new data integrity manager
    pub fn new(config: DataIntegrityConfig) -> DisasterRecoveryResult<Self> {
        config
            .validate()
            .map_err(|e| DisasterRecoveryError::ConfigurationError { message: e })?;

        let (command_tx, command_rx) = mpsc::channel(1000);
        let verification_semaphore = Arc::new(Semaphore::new(config.max_parallel_verifications));
        let repair_semaphore = Arc::new(Semaphore::new(config.repair_config.max_parallel_repairs));

        Ok(Self {
            config: Arc::new(config),
            objects: Arc::new(DashMap::new()),
            tasks: Arc::new(DashMap::new()),
            detections: Arc::new(DashMap::new()),
            repairs: Arc::new(DashMap::new()),
            audit_trail: Arc::new(RwLock::new(AuditTrail::new(10000))),
            metrics: Arc::new(RwLock::new(IntegrityMetrics::default())),
            verification_semaphore,
            repair_semaphore,
            command_tx,
            command_rx: Arc::new(Mutex::new(command_rx)),
            shutdown: Arc::new(RwLock::new(false)),
        })
    }

    /// Start the manager
    pub async fn start(&self) -> DisasterRecoveryResult<()> {
        info!("Starting data integrity manager");

        // Start command processor
        self.start_command_processor().await?;

        // Start verification scheduler
        self.start_verification_scheduler().await?;

        // Start corruption monitor
        if self.config.realtime_detection {
            self.start_corruption_monitor().await?;
        }

        // Add audit entry
        self.audit_trail.write().add_entry(AuditEntry::new(
            AuditEventType::ConfigurationChanged,
            "system".to_string(),
            "Data integrity manager started".to_string(),
            AuditResult::Success,
        ));

        Ok(())
    }

    /// Stop the manager
    pub async fn stop(&self) -> DisasterRecoveryResult<()> {
        info!("Stopping data integrity manager");
        *self.shutdown.write() = true;

        // Send shutdown command
        let _ = self.command_tx.send(IntegrityCommand::Shutdown).await;

        // Add audit entry
        self.audit_trail.write().add_entry(AuditEntry::new(
            AuditEventType::ConfigurationChanged,
            "system".to_string(),
            "Data integrity manager stopped".to_string(),
            AuditResult::Success,
        ));

        Ok(())
    }

    /// Register object for monitoring
    pub async fn register_object(&self, mut object: DataObject) -> DisasterRecoveryResult<Uuid> {
        let object_id = object.id;

        // Calculate initial checksum
        if let Ok(data) = fs::read(&object.path).await {
            let checksum = calculate_checksum(&data, self.config.default_algorithm)?;
            let checksum_info = ChecksumInfo {
                algorithm: self.config.default_algorithm,
                value: checksum,
                calculated_at: Utc::now(),
                block_checksums: None,
            };
            object
                .checksums
                .insert(self.config.default_algorithm, checksum_info);
            object.integrity_status = IntegrityStatus::Verified;
        }

        self.objects.insert(object_id, object.clone());

        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.total_objects += 1;
            metrics.total_bytes += object.size_bytes;
        }

        // Add audit entry
        self.audit_trail.write().add_entry(
            AuditEntry::new(
                AuditEventType::ChecksumCalculated,
                "system".to_string(),
                format!("Registered object: {}", object.path),
                AuditResult::Success,
            )
            .with_object_id(object_id),
        );

        info!("Registered object {} for integrity monitoring", object_id);

        Ok(object_id)
    }

    /// Create verification task
    pub async fn create_verification_task(
        &self,
        object_id: Uuid,
        check_type: IntegrityCheckType,
        schedule: VerificationSchedule,
    ) -> DisasterRecoveryResult<Uuid> {
        let object = self
            .objects
            .get(&object_id)
            .ok_or_else(|| DisasterRecoveryError::Other(format!("Object {} not found", object_id)))?
            .clone();

        let mut task = VerificationTask::new(
            object.clone(),
            check_type,
            self.config.default_algorithm,
            schedule,
        );

        task.calculate_next_run();
        let task_id = task.id;

        self.tasks.insert(task_id, task);

        info!(
            "Created verification task {} for object {}",
            task_id, object_id
        );

        Ok(task_id)
    }

    /// Manually verify object integrity
    pub async fn verify_object(&self, object_id: Uuid) -> DisasterRecoveryResult<bool> {
        let _permit = self.verification_semaphore.acquire().await.map_err(|e| {
            DisasterRecoveryError::Other(format!("Failed to acquire permit: {}", e))
        })?;

        let object = self
            .objects
            .get(&object_id)
            .ok_or_else(|| DisasterRecoveryError::Other(format!("Object {} not found", object_id)))?
            .clone();

        let start = Utc::now();

        // Read current data
        let data = fs::read(&object.path).await?;

        // Calculate checksum
        let algorithm = self.config.default_algorithm;
        let calculated = calculate_checksum(&data, algorithm)?;

        // Get expected checksum
        let expected = object
            .checksums
            .get(&algorithm)
            .map(|info| info.value.clone())
            .ok_or_else(|| {
                DisasterRecoveryError::Other("No baseline checksum found".to_string())
            })?;

        let valid = verify_checksum(&calculated, &expected);

        let duration = Utc::now() - start;

        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.record_verification(valid, duration);
            metrics.record_algorithm_usage(algorithm);
        }

        // Add audit entry
        self.audit_trail.write().add_entry(
            AuditEntry::new(
                AuditEventType::IntegrityCheck,
                "system".to_string(),
                format!(
                    "Verified object {}: {}",
                    object_id,
                    if valid { "valid" } else { "corrupted" }
                ),
                if valid {
                    AuditResult::Success
                } else {
                    AuditResult::Failure
                },
            )
            .with_object_id(object_id),
        );

        if !valid {
            warn!("Corruption detected in object {}", object_id);
            self.handle_corruption_detection(object_id, calculated, expected)
                .await?;
        } else {
            debug!("Object {} integrity verified", object_id);
        }

        Ok(valid)
    }

    /// Handle corruption detection
    async fn handle_corruption_detection(
        &self,
        object_id: Uuid,
        actual: Vec<u8>,
        expected: Vec<u8>,
    ) -> DisasterRecoveryResult<()> {
        let mut detection = CorruptionDetection::new(
            object_id,
            CorruptionType::ChecksumMismatch,
            CorruptionSeverity::High,
        );

        detection.set_checksums(expected, actual);
        detection.detection_method =
            DetectionMethod::ChecksumVerification(self.config.default_algorithm);

        // Suggest repair strategies
        detection.add_repair_strategy(RepairStrategy::RestoreFromBackup {
            backup_id: Uuid::new_v4(), // Would need actual backup ID
        });

        let detection_id = detection.id;
        self.detections.insert(detection_id, detection);

        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.record_corruption(CorruptionSeverity::High);
        }

        // Add audit entry
        self.audit_trail.write().add_entry(
            AuditEntry::new(
                AuditEventType::CorruptionDetected,
                "system".to_string(),
                format!("Corruption detected in object {}", object_id),
                AuditResult::Failure,
            )
            .with_object_id(object_id),
        );

        // Attempt auto-repair if enabled
        if self.config.auto_repair {
            self.command_tx
                .send(IntegrityCommand::Repair(detection_id))
                .await
                .map_err(|e| {
                    DisasterRecoveryError::Other(format!("Failed to send repair command: {}", e))
                })?;
        }

        Ok(())
    }

    /// Get metrics
    pub fn get_metrics(&self) -> IntegrityMetrics {
        self.metrics.read().clone()
    }

    /// Get audit trail
    pub fn get_audit_trail(&self) -> Vec<AuditEntry> {
        let trail = self.audit_trail.read();
        trail.get_recent_entries(100).into_iter().cloned().collect()
    }

    // Private helper methods

    async fn start_command_processor(&self) -> DisasterRecoveryResult<()> {
        let command_rx = self.command_rx.clone();
        let shutdown = self.shutdown.clone();

        tokio::spawn(async move {
            let mut receiver = command_rx.lock().await;

            while !*shutdown.read() {
                if let Some(command) = receiver.recv().await {
                    match command {
                        IntegrityCommand::Verify(object_id) => {
                            debug!("Processing verify command for {}", object_id);
                        }
                        IntegrityCommand::Repair(detection_id) => {
                            debug!("Processing repair command for {}", detection_id);
                        }
                        IntegrityCommand::Schedule(task_id) => {
                            debug!("Processing schedule command for {}", task_id);
                        }
                        IntegrityCommand::Shutdown => {
                            info!("Received shutdown command");
                            break;
                        }
                    }
                }
            }
        });

        Ok(())
    }

    async fn start_verification_scheduler(&self) -> DisasterRecoveryResult<()> {
        let tasks = self.tasks.clone();
        let command_tx = self.command_tx.clone();
        let shutdown = self.shutdown.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));

            while !*shutdown.read() {
                interval.tick().await;

                for entry in tasks.iter() {
                    let task = entry.value();
                    if task.should_run_now() {
                        let _ = command_tx
                            .send(IntegrityCommand::Verify(task.object.id))
                            .await;
                    }
                }
            }
        });

        Ok(())
    }

    async fn start_corruption_monitor(&self) -> DisasterRecoveryResult<()> {
        info!("Starting real-time corruption monitor");
        // Real-time monitoring implementation would go here
        Ok(())
    }
}
