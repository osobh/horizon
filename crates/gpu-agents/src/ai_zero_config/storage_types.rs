//! Storage-related types for AI Assistant Zero-Config Integration
//! Split from types.rs to keep files under 750 lines

use std::collections::HashMap;
use std::time::Duration;

/// Storage configuration
#[derive(Debug, Clone)]
pub struct StorageConfiguration {
    pub persistent_volumes: Vec<PersistentVolumeConfiguration>,
    pub volume_mounts: Vec<VolumeMountConfiguration>,
    pub backup_configuration: BackupConfiguration,
    pub data_lifecycle: DataLifecycleConfiguration,
}

/// Persistent volume configuration
#[derive(Debug, Clone)]
pub struct PersistentVolumeConfiguration {
    pub name: String,
    pub size: String,
    pub storage_class: String,
    pub access_modes: Vec<AccessMode>,
    pub volume_mode: VolumeMode,
    pub reclaim_policy: ReclaimPolicy,
}

/// Volume access modes
#[derive(Debug, Clone, PartialEq)]
pub enum AccessMode {
    ReadWriteOnce,
    ReadOnlyMany,
    ReadWriteMany,
}

/// Volume modes
#[derive(Debug, Clone, PartialEq)]
pub enum VolumeMode {
    Filesystem,
    Block,
}

/// Reclaim policies
#[derive(Debug, Clone, PartialEq)]
pub enum ReclaimPolicy {
    Retain,
    Delete,
    Recycle,
}

/// Volume mount configuration
#[derive(Debug, Clone)]
pub struct VolumeMountConfiguration {
    pub name: String,
    pub mount_path: String,
    pub sub_path: Option<String>,
    pub read_only: bool,
    pub mount_propagation: Option<MountPropagation>,
}

/// Mount propagation modes
#[derive(Debug, Clone, PartialEq)]
pub enum MountPropagation {
    None,
    HostToContainer,
    Bidirectional,
}

/// Backup configuration
#[derive(Debug, Clone)]
pub struct BackupConfiguration {
    pub enabled: bool,
    pub backup_schedule: String,
    pub retention_policy: BackupRetentionPolicy,
    pub backup_storage: BackupStorageConfiguration,
    pub encryption_enabled: bool,
}

/// Backup retention policies
#[derive(Debug, Clone)]
pub struct BackupRetentionPolicy {
    pub daily_backups: u32,
    pub weekly_backups: u32,
    pub monthly_backups: u32,
    pub yearly_backups: u32,
}

/// Backup storage configuration
#[derive(Debug, Clone)]
pub struct BackupStorageConfiguration {
    pub storage_type: BackupStorageType,
    pub storage_location: String,
    pub cross_region_replication: bool,
}

/// Backup storage types
#[derive(Debug, Clone, PartialEq)]
pub enum BackupStorageType {
    S3,
    GCS,
    AzureBlob,
    Local,
    NFS,
}

/// Data lifecycle configuration
#[derive(Debug, Clone)]
pub struct DataLifecycleConfiguration {
    pub lifecycle_policies: Vec<DataLifecyclePolicy>,
    pub data_classification: HashMap<String, super::DataClassification>,
    pub retention_schedules: Vec<RetentionSchedule>,
}

/// Data lifecycle policies
#[derive(Debug, Clone)]
pub struct DataLifecyclePolicy {
    pub name: String,
    pub data_types: Vec<String>,
    pub lifecycle_stages: Vec<LifecycleStage>,
}

/// Lifecycle stages
#[derive(Debug, Clone)]
pub struct LifecycleStage {
    pub stage_name: String,
    pub trigger_condition: TriggerCondition,
    pub actions: Vec<LifecycleAction>,
}

/// Trigger conditions
#[derive(Debug, Clone)]
pub struct TriggerCondition {
    pub condition_type: TriggerConditionType,
    pub threshold: String,
}

/// Trigger condition types
#[derive(Debug, Clone, PartialEq)]
pub enum TriggerConditionType {
    Age,
    Size,
    AccessFrequency,
    LastModified,
    DataClassification,
}

/// Lifecycle actions
#[derive(Debug, Clone, PartialEq)]
pub enum LifecycleAction {
    Archive,
    Delete,
    Compress,
    Encrypt,
    Migrate,
    Notify,
}

/// Retention schedules
#[derive(Debug, Clone)]
pub struct RetentionSchedule {
    pub data_type: String,
    pub retention_period: Duration,
    pub deletion_method: DeletionMethod,
    pub verification_required: bool,
}

/// Deletion methods
#[derive(Debug, Clone, PartialEq)]
pub enum DeletionMethod {
    SoftDelete,
    HardDelete,
    Anonymize,
    Encrypt,
}