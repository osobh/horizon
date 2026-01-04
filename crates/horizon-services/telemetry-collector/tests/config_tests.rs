use std::io::Write;
use telemetry_collector::config::CollectorConfig;
use tempfile::NamedTempFile;

#[test]
fn test_config_parse_valid_yaml() {
    let yaml_content = r#"
server:
  listen_addr: "0.0.0.0:5001"
  max_connections: 1000
  connection_timeout_secs: 300

security:
  tls_cert_path: "/etc/horizon/certs/server.pem"
  tls_key_path: "/etc/horizon/certs/server-key.pem"
  tls_ca_path: "/etc/horizon/certs/ca.pem"

influxdb:
  url: "http://localhost:8086"
  org: "horizon"
  bucket: "telemetry"
  token: "test-token"

parquet:
  output_dir: "/var/lib/horizon/telemetry"
  rotation_interval_secs: 3600
  compression: "snappy"

limits:
  max_cardinality: 100000
  max_batch_size: 1000
  backpressure_threshold: 5000

observability:
  metrics_port: 9091
  log_level: "info"
"#;

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(yaml_content.as_bytes()).unwrap();
    temp_file.flush().unwrap();

    let config = CollectorConfig::from_file(temp_file.path()).unwrap();

    assert_eq!(config.server.listen_addr, "0.0.0.0:5001");
    assert_eq!(config.server.max_connections, 1000);
    assert_eq!(config.server.connection_timeout_secs, 300);
    assert_eq!(config.influxdb.url, "http://localhost:8086");
    assert_eq!(config.parquet.compression, "snappy");
    assert_eq!(config.limits.max_cardinality, 100000);
}

#[test]
fn test_config_validation_missing_required_fields() {
    let yaml_content = r#"
server:
  listen_addr: "0.0.0.0:5001"
"#;

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(yaml_content.as_bytes()).unwrap();
    temp_file.flush().unwrap();

    let result = CollectorConfig::from_file(temp_file.path());
    assert!(result.is_err());
}

#[test]
fn test_config_validation_invalid_port() {
    let yaml_content = r#"
server:
  listen_addr: "0.0.0.0:99999"
  max_connections: 1000
  connection_timeout_secs: 300

security:
  tls_cert_path: "/etc/horizon/certs/server.pem"
  tls_key_path: "/etc/horizon/certs/server-key.pem"
  tls_ca_path: "/etc/horizon/certs/ca.pem"

influxdb:
  url: "http://localhost:8086"
  org: "horizon"
  bucket: "telemetry"
  token: "test-token"

parquet:
  output_dir: "/var/lib/horizon/telemetry"
  rotation_interval_secs: 3600
  compression: "snappy"

limits:
  max_cardinality: 100000
  max_batch_size: 1000
  backpressure_threshold: 5000

observability:
  metrics_port: 9091
  log_level: "info"
"#;

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(yaml_content.as_bytes()).unwrap();
    temp_file.flush().unwrap();

    let result = CollectorConfig::from_file(temp_file.path());
    assert!(result.is_err());
}

#[test]
fn test_config_validation_invalid_compression() {
    let yaml_content = r#"
server:
  listen_addr: "0.0.0.0:5001"
  max_connections: 1000
  connection_timeout_secs: 300

security:
  tls_cert_path: "/etc/horizon/certs/server.pem"
  tls_key_path: "/etc/horizon/certs/server-key.pem"
  tls_ca_path: "/etc/horizon/certs/ca.pem"

influxdb:
  url: "http://localhost:8086"
  org: "horizon"
  bucket: "telemetry"
  token: "test-token"

parquet:
  output_dir: "/var/lib/horizon/telemetry"
  rotation_interval_secs: 3600
  compression: "invalid"

limits:
  max_cardinality: 100000
  max_batch_size: 1000
  backpressure_threshold: 5000

observability:
  metrics_port: 9091
  log_level: "info"
"#;

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(yaml_content.as_bytes()).unwrap();
    temp_file.flush().unwrap();

    let result = CollectorConfig::from_file(temp_file.path());
    assert!(result.is_err());
}

#[test]
fn test_config_default_values() {
    let config = CollectorConfig::default();

    assert_eq!(config.server.listen_addr, "0.0.0.0:5001");
    assert_eq!(config.server.max_connections, 1000);
    assert_eq!(config.server.connection_timeout_secs, 300);
    assert_eq!(config.limits.max_cardinality, 100000);
    assert_eq!(config.limits.max_batch_size, 1000);
    assert_eq!(config.limits.backpressure_threshold, 5000);
    assert_eq!(config.observability.metrics_port, 9091);
    assert_eq!(config.observability.log_level, "info");
}

#[test]
fn test_config_env_var_substitution() {
    std::env::set_var("TEST_INFLUX_TOKEN", "secret-token");

    let yaml_content = r#"
server:
  listen_addr: "0.0.0.0:5001"
  max_connections: 1000
  connection_timeout_secs: 300

security:
  tls_cert_path: "/etc/horizon/certs/server.pem"
  tls_key_path: "/etc/horizon/certs/server-key.pem"
  tls_ca_path: "/etc/horizon/certs/ca.pem"

influxdb:
  url: "http://localhost:8086"
  org: "horizon"
  bucket: "telemetry"
  token: "${TEST_INFLUX_TOKEN}"

parquet:
  output_dir: "/var/lib/horizon/telemetry"
  rotation_interval_secs: 3600
  compression: "snappy"

limits:
  max_cardinality: 100000
  max_batch_size: 1000
  backpressure_threshold: 5000

observability:
  metrics_port: 9091
  log_level: "info"
"#;

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(yaml_content.as_bytes()).unwrap();
    temp_file.flush().unwrap();

    let config = CollectorConfig::from_file(temp_file.path()).unwrap();
    assert_eq!(config.influxdb.token, "secret-token");

    std::env::remove_var("TEST_INFLUX_TOKEN");
}

#[test]
fn test_config_validation_negative_values() {
    let yaml_content = r#"
server:
  listen_addr: "0.0.0.0:5001"
  max_connections: -1
  connection_timeout_secs: 300

security:
  tls_cert_path: "/etc/horizon/certs/server.pem"
  tls_key_path: "/etc/horizon/certs/server-key.pem"
  tls_ca_path: "/etc/horizon/certs/ca.pem"

influxdb:
  url: "http://localhost:8086"
  org: "horizon"
  bucket: "telemetry"
  token: "test-token"

parquet:
  output_dir: "/var/lib/horizon/telemetry"
  rotation_interval_secs: 3600
  compression: "snappy"

limits:
  max_cardinality: 100000
  max_batch_size: 1000
  backpressure_threshold: 5000

observability:
  metrics_port: 9091
  log_level: "info"
"#;

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(yaml_content.as_bytes()).unwrap();
    temp_file.flush().unwrap();

    let result = CollectorConfig::from_file(temp_file.path());
    assert!(result.is_err());
}

#[test]
fn test_config_validate_method() {
    let mut config = CollectorConfig::default();
    config.influxdb.token = "test-token".to_string(); // Need token for validation
    assert!(config.validate().is_ok());

    // Test invalid listen address
    config.server.listen_addr = "invalid".to_string();
    assert!(config.validate().is_err());

    config.server.listen_addr = "0.0.0.0:5001".to_string();
    assert!(config.validate().is_ok());

    // Test zero max_connections
    config.server.max_connections = 0;
    assert!(config.validate().is_err());

    config.server.max_connections = 1000;
    assert!(config.validate().is_ok());

    // Test invalid compression
    config.parquet.compression = "invalid".to_string();
    assert!(config.validate().is_err());
}
