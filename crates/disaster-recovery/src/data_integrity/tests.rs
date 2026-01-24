//! Tests for data integrity module

#[cfg(test)]
mod tests {
    use super::super::*;
    use chrono::Utc;
    use std::collections::HashMap;
    use std::io::Write;
    use tempfile::NamedTempFile;
    use uuid::Uuid;

    fn create_test_object(path: &str) -> DataObject {
        DataObject {
            id: Uuid::new_v4(),
            path: path.to_string(),
            object_type: ObjectType::Other,
            size_bytes: 1024,
            created_at: Utc::now(),
            modified_at: Utc::now(),
            checksums: HashMap::new(),
            integrity_status: IntegrityStatus::Unknown,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_checksum_algorithm_properties() -> anyhow::Result<()> {
        assert_eq!(ChecksumAlgorithm::SHA256.output_length(), 32);
        assert_eq!(ChecksumAlgorithm::SHA512.output_length(), 64);
        assert_eq!(ChecksumAlgorithm::CRC32.output_length(), 4);
        assert_eq!(ChecksumAlgorithm::MD5.output_length(), 16);
        assert_eq!(ChecksumAlgorithm::Blake2b.output_length(), 64);
        assert_eq!(ChecksumAlgorithm::XXHash.output_length(), 8);
        Ok(())
    }

    #[test]
    fn test_integrity_status_serialization() -> anyhow::Result<()> {
        let statuses = vec![
            IntegrityStatus::Verified,
            IntegrityStatus::Corrupted,
            IntegrityStatus::Repairing,
            IntegrityStatus::Repaired,
            IntegrityStatus::Pending,
            IntegrityStatus::Verifying,
            IntegrityStatus::Unknown,
        ];

        for status in statuses {
            let serialized = serde_json::to_string(&status).unwrap();
            let deserialized: IntegrityStatus = serde_json::from_str(&serialized).unwrap();
            assert_eq!(status, deserialized);
        }
        Ok(())
    }

    #[test]
    fn test_corruption_detection_creation() -> anyhow::Result<()> {
        let object_id = Uuid::new_v4();
        let mut detection = CorruptionDetection::new(
            object_id,
            CorruptionType::ChecksumMismatch,
            CorruptionSeverity::High,
        );

        assert_eq!(detection.object_id, object_id);
        assert_eq!(detection.corruption_type, CorruptionType::ChecksumMismatch);
        assert_eq!(detection.severity, CorruptionSeverity::High);
        assert!(detection.affected_ranges.is_empty());

        detection.add_affected_range(100, 200);
        assert_eq!(detection.affected_ranges.len(), 1);
        assert_eq!(detection.total_corrupted_bytes(), 100);
        Ok(())
    }

    #[test]
    fn test_repair_record_lifecycle() -> anyhow::Result<()> {
        let detection_id = Uuid::new_v4();
        let object_id = Uuid::new_v4();
        let mut repair = RepairRecord::new(
            detection_id,
            object_id,
            RepairStrategy::RestoreFromBackup {
                backup_id: Uuid::new_v4(),
            },
        );

        assert_eq!(repair.status, RepairStatus::Pending);

        repair.start();
        assert_eq!(repair.status, RepairStatus::InProgress);

        repair.complete(true, 1024);
        assert_eq!(repair.status, RepairStatus::Completed);
        assert!(repair.success);
        assert_eq!(repair.bytes_repaired, 1024);
        assert!(repair.duration.is_some());
        Ok(())
    }

    #[test]
    fn test_audit_trail() -> anyhow::Result<()> {
        let mut trail = AuditTrail::new(100);
        assert!(trail.is_empty());

        let entry = AuditEntry::new(
            AuditEventType::IntegrityCheck,
            "test_user".to_string(),
            "Test integrity check".to_string(),
            AuditResult::Success,
        );

        trail.add_entry(entry.clone());
        assert_eq!(trail.len(), 1);

        let recent = trail.get_recent_entries(10);
        assert_eq!(recent.len(), 1);
        Ok(())
    }

    #[test]
    fn test_metrics_recording() -> anyhow::Result<()> {
        use chrono::Duration;
        let mut metrics = IntegrityMetrics::default();

        metrics.record_verification(true, Duration::seconds(5));
        assert_eq!(metrics.total_verifications, 1);
        assert_eq!(metrics.successful_verifications, 1);

        metrics.record_corruption(CorruptionSeverity::High);
        assert_eq!(metrics.corruptions_detected, 1);

        metrics.record_repair(true, 1024, Duration::minutes(2));
        assert_eq!(metrics.successful_repairs, 1);
        assert_eq!(metrics.bytes_repaired, 1024);

        assert_eq!(metrics.verification_success_rate(), 1.0);
        assert_eq!(metrics.repair_success_rate(), 1.0);
        Ok(())
    }

    #[test]
    fn test_verification_task_scheduling() -> anyhow::Result<()> {
        let object = create_test_object("/test/file.dat");
        let mut task = VerificationTask::new(
            object,
            IntegrityCheckType::Full,
            ChecksumAlgorithm::SHA256,
            VerificationSchedule::Hourly,
        );

        assert!(task.should_run_now());

        task.calculate_next_run();
        assert!(task.next_run.is_some());

        task.record_execution(true);
        assert_eq!(task.run_count, 1);
        assert_eq!(task.success_count, 1);
        assert_eq!(task.success_rate(), 1.0);
        Ok(())
    }

    #[test]
    fn test_config_validation() -> anyhow::Result<()> {
        let mut config = DataIntegrityConfig::default();
        assert!(config.validate().is_ok());

        config.enabled_algorithms.clear();
        assert!(config.validate().is_err());

        config.enabled_algorithms.push(ChecksumAlgorithm::SHA256);
        config.default_algorithm = ChecksumAlgorithm::SHA512;
        assert!(config.validate().is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_checksum_calculation() -> anyhow::Result<()> {
        let data = b"Hello, World!";
        let checksum = calculate_checksum(data, ChecksumAlgorithm::SHA256).unwrap();
        assert_eq!(checksum.len(), 32);

        let checksum2 = calculate_checksum(data, ChecksumAlgorithm::SHA256).unwrap();
        assert!(verify_checksum(&checksum, &checksum2));
        Ok(())
    }

    #[tokio::test]
    async fn test_manager_creation() -> anyhow::Result<()> {
        let config = DataIntegrityConfig::default();
        let manager = DataIntegrityManager::new(config);
        assert!(manager.is_ok());
        Ok(())
    }
}
