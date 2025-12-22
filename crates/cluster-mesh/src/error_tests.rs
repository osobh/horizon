//! Comprehensive tests for error handling

#[cfg(test)]
mod error_comprehensive_tests {
    use super::super::*;
    use crate::error::{ClusterMeshError, Result};
    use std::io;
    use uuid::Uuid;

    #[test]
    fn test_all_error_variants_display() {
        // Test all error variants have proper display formatting
        let node_id = Uuid::new_v4();

        let errors = vec![
            ClusterMeshError::Network("network failure".to_string()),
            ClusterMeshError::NodeNotFound(node_id),
            ClusterMeshError::NodeAlreadyExists(node_id),
            ClusterMeshError::HardwareDetection("CPU detection failed".to_string()),
            ClusterMeshError::GpuDetection("No GPUs found".to_string()),
            ClusterMeshError::InvalidNodeClass("Unknown class".to_string()),
            ClusterMeshError::SchedulingFailed("No resources".to_string()),
            ClusterMeshError::NoSuitableNode,
            ClusterMeshError::NodeOffline(node_id),
            ClusterMeshError::InsufficientResources("Not enough memory".to_string()),
            ClusterMeshError::MeshFormationFailed("Connection failed".to_string()),
            ClusterMeshError::NatTraversalFailed("STUN failed".to_string()),
            ClusterMeshError::Certificate("Invalid cert".to_string()),
            ClusterMeshError::Authentication("Auth failed".to_string()),
            ClusterMeshError::BatteryLow(10),
            ClusterMeshError::ThermalLimit(95.0),
            ClusterMeshError::MigrationFailed("Migration error".to_string()),
            ClusterMeshError::HeartbeatTimeout(node_id),
            ClusterMeshError::Configuration("Bad config".to_string()),
            ClusterMeshError::SystemInfo("System error".to_string()),
            ClusterMeshError::Other("Unknown error".to_string()),
        ];

        for error in errors {
            // Ensure all errors have non-empty display strings
            let display = format!("{}", error);
            assert!(!display.is_empty());

            // Ensure debug representation works
            let debug = format!("{:?}", error);
            assert!(!debug.is_empty());
        }
    }

    #[test]
    fn test_error_result_type() {
        // Test that Result type alias works correctly
        fn returns_ok() -> Result<i32> {
            Ok(42)
        }

        fn returns_err() -> Result<i32> {
            Err(ClusterMeshError::Other("test error".to_string()))
        }

        assert!(returns_ok().is_ok());
        assert!(returns_err().is_err());
    }

    #[test]
    fn test_io_error_conversion() {
        // Test various IO error kinds
        let io_errors = vec![
            io::ErrorKind::NotFound,
            io::ErrorKind::PermissionDenied,
            io::ErrorKind::ConnectionRefused,
            io::ErrorKind::TimedOut,
            io::ErrorKind::UnexpectedEof,
        ];

        for kind in io_errors {
            let io_err = io::Error::new(kind, "test io error");
            let mesh_err: ClusterMeshError = io_err.into();

            match mesh_err {
                ClusterMeshError::Io(e) => {
                    assert_eq!(e.kind(), kind);
                }
                _ => panic!("Expected Io error variant"),
            }
        }
    }

    #[test]
    fn test_anyhow_conversion() {
        use anyhow::anyhow;

        let anyhow_err = anyhow!("test anyhow error");
        let mesh_err: ClusterMeshError = anyhow_err.into();

        assert!(matches!(mesh_err, ClusterMeshError::Anyhow(_)));
    }

    #[test]
    fn test_error_chaining() {
        // Test that errors can be chained properly
        fn inner_function() -> Result<()> {
            Err(ClusterMeshError::NodeNotFound(Uuid::new_v4()))
        }

        fn outer_function() -> Result<()> {
            inner_function().map_err(|e| ClusterMeshError::Other(format!("Wrapped: {}", e)))?;
            Ok(())
        }

        let result = outer_function();
        assert!(result.is_err());

        if let Err(e) = result {
            let msg = format!("{}", e);
            assert!(msg.contains("Wrapped"));
            assert!(msg.contains("Node"));
        }
    }

    #[test]
    fn test_specific_error_messages() {
        // Test specific error message formats
        let node_id = Uuid::nil();

        let battery_err = ClusterMeshError::BatteryLow(5);
        assert_eq!(format!("{}", battery_err), "Battery level too low: 5%");

        let thermal_err = ClusterMeshError::ThermalLimit(100.5);
        assert_eq!(
            format!("{}", thermal_err),
            "Thermal limit exceeded: 100.5°C"
        );

        let heartbeat_err = ClusterMeshError::HeartbeatTimeout(node_id);
        assert_eq!(
            format!("{}", heartbeat_err),
            format!("Heartbeat timeout for node {}", node_id)
        );

        let offline_err = ClusterMeshError::NodeOffline(node_id);
        assert_eq!(
            format!("{}", offline_err),
            format!("Node {} is offline", node_id)
        );
    }

    #[test]
    fn test_json_serialization_error() {
        use serde::Deserialize;

        #[derive(Deserialize)]
        struct TestStruct {
            _field: String,
        }

        let invalid_json = "{ invalid json }";
        let result: std::result::Result<TestStruct, serde_json::Error> =
            serde_json::from_str(invalid_json);

        assert!(result.is_err());

        let json_err = result.unwrap_err();
        let mesh_err: ClusterMeshError = json_err.into();

        assert!(matches!(mesh_err, ClusterMeshError::Serialization(_)));
    }

    #[test]
    fn test_error_source_chain() {
        // Test that error sources are properly preserved
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let mesh_err = ClusterMeshError::from(io_err);

        // Convert to Box<dyn Error> to test error trait implementation
        let boxed: Box<dyn std::error::Error> = Box::new(mesh_err);
        assert!(boxed.source().is_some());
    }
}
