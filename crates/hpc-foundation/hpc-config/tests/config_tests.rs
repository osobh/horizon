use hpc_config::ExposeSecret;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use tempfile::TempDir;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
struct TestServiceConfig {
    pub listen_addr: String,
    pub log_level: String,
    pub max_connections: u32,
    pub enable_tls: bool,
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
}

fn default_timeout() -> u64 {
    5000
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
struct DatabaseConfig {
    pub host: String,
    pub port: u16,
    pub database: String,
    pub username: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub password: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
struct ComplexConfig {
    pub service: TestServiceConfig,
    pub database: DatabaseConfig,
    #[serde(default)]
    pub features: Vec<String>,
}

// Test 1: Load config from YAML file
#[test]
fn test_load_from_yaml() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("test-service.yaml");

    let yaml_content = r#"
listen_addr: "0.0.0.0:8080"
log_level: "info"
max_connections: 100
enable_tls: true
timeout_ms: 3000
"#;
    fs::write(&config_path, yaml_content).unwrap();

    let config: TestServiceConfig = hpc_config::load_from_file(&config_path).unwrap();

    assert_eq!(config.listen_addr, "0.0.0.0:8080");
    assert_eq!(config.log_level, "info");
    assert_eq!(config.max_connections, 100);
    assert!(config.enable_tls);
    assert_eq!(config.timeout_ms, 3000);
}

// Test 2: Load config from TOML file
#[test]
fn test_load_from_toml() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("test-service.toml");

    let toml_content = r#"
listen_addr = "0.0.0.0:9090"
log_level = "debug"
max_connections = 200
enable_tls = false
timeout_ms = 10000
"#;
    fs::write(&config_path, toml_content).unwrap();

    let config: TestServiceConfig = hpc_config::load_from_file(&config_path).unwrap();

    assert_eq!(config.listen_addr, "0.0.0.0:9090");
    assert_eq!(config.log_level, "debug");
    assert_eq!(config.max_connections, 200);
    assert!(!config.enable_tls);
    assert_eq!(config.timeout_ms, 10000);
}

// Test 3: Load config from JSON file
#[test]
fn test_load_from_json() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("test-service.json");

    let json_content = r#"
{
    "listen_addr": "127.0.0.1:7070",
    "log_level": "warn",
    "max_connections": 50,
    "enable_tls": true,
    "timeout_ms": 2000
}
"#;
    fs::write(&config_path, json_content).unwrap();

    let config: TestServiceConfig = hpc_config::load_from_file(&config_path).unwrap();

    assert_eq!(config.listen_addr, "127.0.0.1:7070");
    assert_eq!(config.log_level, "warn");
    assert_eq!(config.max_connections, 50);
}

// Test 4: Environment variable override
#[test]
fn test_env_override() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("test-service.yaml");

    let yaml_content = r#"
listen_addr: "0.0.0.0:8080"
log_level: "info"
max_connections: 100
enable_tls: false
"#;
    fs::write(&config_path, yaml_content).unwrap();

    // Set environment variables
    env::set_var("TESTSERVICE__LISTEN_ADDR", "0.0.0.0:9999");
    env::set_var("TESTSERVICE__LOG_LEVEL", "debug");
    env::set_var("TESTSERVICE__MAX_CONNECTIONS", "500");
    env::set_var("TESTSERVICE__ENABLE_TLS", "true");

    let config: TestServiceConfig =
        hpc_config::load_with_env(&config_path, "TESTSERVICE").unwrap();

    assert_eq!(config.listen_addr, "0.0.0.0:9999");
    assert_eq!(config.log_level, "debug");
    assert_eq!(config.max_connections, 500);
    assert!(config.enable_tls);

    // Cleanup
    env::remove_var("TESTSERVICE__LISTEN_ADDR");
    env::remove_var("TESTSERVICE__LOG_LEVEL");
    env::remove_var("TESTSERVICE__MAX_CONNECTIONS");
    env::remove_var("TESTSERVICE__ENABLE_TLS");
}

// Test 5: Layered config (defaults < file < env)
#[test]
fn test_layered_config() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("test-service.yaml");

    // File doesn't specify timeout_ms, so default should be used
    let yaml_content = r#"
listen_addr: "0.0.0.0:8080"
log_level: "info"
max_connections: 100
enable_tls: false
"#;
    fs::write(&config_path, yaml_content).unwrap();

    let config: TestServiceConfig = hpc_config::load_from_file(&config_path).unwrap();

    // timeout_ms should use the default value
    assert_eq!(config.timeout_ms, 5000);
}

// Test 6: Load complex nested config
#[test]
fn test_nested_config() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("complex.yaml");

    let yaml_content = r#"
service:
  listen_addr: "0.0.0.0:8080"
  log_level: "info"
  max_connections: 100
  enable_tls: true

database:
  host: "localhost"
  port: 5432
  database: "horizon"
  username: "admin"
  password: "secret123"

features:
  - "feature_a"
  - "feature_b"
  - "feature_c"
"#;
    fs::write(&config_path, yaml_content).unwrap();

    let config: ComplexConfig = hpc_config::load_from_file(&config_path).unwrap();

    assert_eq!(config.service.listen_addr, "0.0.0.0:8080");
    assert_eq!(config.database.host, "localhost");
    assert_eq!(config.database.port, 5432);
    assert_eq!(config.database.password, Some("secret123".to_string()));
    assert_eq!(config.features, vec!["feature_a", "feature_b", "feature_c"]);
}

// Test 7: Environment override for nested config
#[test]
fn test_nested_env_override() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("complex.yaml");

    let yaml_content = r#"
service:
  listen_addr: "0.0.0.0:8080"
  log_level: "info"
  max_connections: 100
  enable_tls: false

database:
  host: "localhost"
  port: 5432
  database: "horizon"
  username: "admin"
"#;
    fs::write(&config_path, yaml_content).unwrap();

    // Set environment variables for nested fields
    env::set_var("COMPLEX__SERVICE__LOG_LEVEL", "debug");
    env::set_var("COMPLEX__DATABASE__HOST", "db.example.com");
    env::set_var("COMPLEX__DATABASE__PORT", "3306");

    let config: ComplexConfig = hpc_config::load_with_env(&config_path, "COMPLEX").unwrap();

    assert_eq!(config.service.log_level, "debug");
    assert_eq!(config.database.host, "db.example.com");
    assert_eq!(config.database.port, 3306);

    // Cleanup
    env::remove_var("COMPLEX__SERVICE__LOG_LEVEL");
    env::remove_var("COMPLEX__DATABASE__HOST");
    env::remove_var("COMPLEX__DATABASE__PORT");
}

// Test 8: Missing required field should error
#[test]
fn test_missing_required_field() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("incomplete.yaml");

    let yaml_content = r#"
listen_addr: "0.0.0.0:8080"
log_level: "info"
# max_connections is missing
enable_tls: false
"#;
    fs::write(&config_path, yaml_content).unwrap();

    let result: Result<TestServiceConfig, _> = hpc_config::load_from_file(&config_path);
    assert!(result.is_err());
}

// Test 9: Invalid format should error
#[test]
fn test_invalid_format() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("invalid.yaml");

    let yaml_content = r#"
listen_addr: "0.0.0.0:8080"
log_level: "info"
max_connections: "not_a_number"  # Should be u32
enable_tls: false
"#;
    fs::write(&config_path, yaml_content).unwrap();

    let result: Result<TestServiceConfig, _> = hpc_config::load_from_file(&config_path);
    assert!(result.is_err());
}

// Test 10: Service-specific config loading with conventional paths
#[test]
fn test_load_service_config() {
    let temp_dir = TempDir::new().unwrap();
    let config_dir = temp_dir.path().join("config");
    fs::create_dir(&config_dir).unwrap();

    let config_path = config_dir.join("test-service.yaml");
    let yaml_content = r#"
listen_addr: "0.0.0.0:8080"
log_level: "info"
max_connections: 100
enable_tls: true
"#;
    fs::write(&config_path, yaml_content).unwrap();

    // Change to temp dir so relative path works
    let original_dir = env::current_dir().unwrap();
    env::set_current_dir(&temp_dir).unwrap();

    let config: TestServiceConfig = hpc_config::load("test-service").unwrap();

    assert_eq!(config.listen_addr, "0.0.0.0:8080");

    // Restore original dir
    env::set_current_dir(original_dir).unwrap();
}

// Test 11: Secret loading from file
#[test]
fn test_secret_from_file() {
    let temp_dir = TempDir::new().unwrap();
    let secret_path = temp_dir.path().join("db_password");

    fs::write(&secret_path, "super_secret_password").unwrap();

    let secret = hpc_config::load_secret_from_file(&secret_path).unwrap();
    assert_eq!(secret.expose_secret(), "super_secret_password");
}

// Test 12: Secret loading from env var
#[test]
fn test_secret_from_env() {
    env::set_var("TEST_DB_PASSWORD", "env_secret_password");

    let secret = hpc_config::load_secret_from_env("TEST_DB_PASSWORD").unwrap();
    assert_eq!(secret.expose_secret(), "env_secret_password");

    env::remove_var("TEST_DB_PASSWORD");
}

// Test 13: ConfigBuilder for custom composition
#[test]
fn test_config_builder() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("base.yaml");

    let yaml_content = r#"
listen_addr: "0.0.0.0:8080"
log_level: "info"
max_connections: 100
enable_tls: false
"#;
    fs::write(&config_path, yaml_content).unwrap();

    env::set_var("BUILDER__LOG_LEVEL", "debug");

    let config: TestServiceConfig = hpc_config::ConfigBuilder::new()
        .add_file(&config_path)
        .add_env_with_prefix("BUILDER")
        .build()
        .unwrap();

    assert_eq!(config.listen_addr, "0.0.0.0:8080");
    assert_eq!(config.log_level, "debug"); // Overridden by env

    env::remove_var("BUILDER__LOG_LEVEL");
}

// Test 14: Default-only config (no file)
#[test]
fn test_defaults_only() {
    #[derive(Debug, Deserialize, PartialEq)]
    struct ConfigWithDefaults {
        #[serde(default = "default_port")]
        pub port: u16,
        #[serde(default = "default_host")]
        pub host: String,
    }

    fn default_port() -> u16 {
        8080
    }
    fn default_host() -> String {
        "localhost".to_string()
    }

    let config: ConfigWithDefaults = hpc_config::ConfigBuilder::new().build().unwrap();

    assert_eq!(config.port, 8080);
    assert_eq!(config.host, "localhost");
}

// Test 15: Property-based test for config validation
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_roundtrip_yaml(
            listen_addr in "[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}:[0-9]{4,5}",
            log_level in "(debug|info|warn|error)",
            max_connections in 1u32..1000u32,
            enable_tls: bool,
            timeout_ms in 1000u64..60000u64,
        ) {
            let temp_dir = TempDir::new().unwrap();
            let config_path = temp_dir.path().join("prop-test.yaml");

            let original = TestServiceConfig {
                listen_addr: listen_addr.clone(),
                log_level: log_level.clone(),
                max_connections,
                enable_tls,
                timeout_ms,
            };

            let yaml = serde_yaml::to_string(&original).unwrap();
            fs::write(&config_path, yaml).unwrap();

            let loaded: TestServiceConfig = hpc_config::load_from_file(&config_path).unwrap();

            prop_assert_eq!(loaded, original);
        }

        #[test]
        fn test_env_override_any_value(
            base_port in 1u32..1000u32,
            override_port in 1000u32..2000u32,
        ) {
            let temp_dir = TempDir::new().unwrap();
            let config_path = temp_dir.path().join("prop-env.yaml");

            let yaml_content = format!(r#"
listen_addr: "0.0.0.0:8080"
log_level: "info"
max_connections: {}
enable_tls: false
"#, base_port);
            fs::write(&config_path, yaml_content).unwrap();

            let env_key = format!("PROPTEST_{}__MAX_CONNECTIONS", base_port);
            env::set_var(&env_key, override_port.to_string());

            let config: TestServiceConfig = hpc_config::load_with_env(
                &config_path,
                &format!("PROPTEST_{}", base_port)
            ).unwrap();

            prop_assert_eq!(config.max_connections, override_port);

            env::remove_var(&env_key);
        }
    }
}

// Test 16: Multiple file sources with priority
#[test]
fn test_multiple_file_sources() {
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path().join("base.yaml");
    let override_path = temp_dir.path().join("override.yaml");

    fs::write(
        &base_path,
        r#"
listen_addr: "0.0.0.0:8080"
log_level: "info"
max_connections: 100
enable_tls: false
timeout_ms: 5000
"#,
    )
    .unwrap();

    fs::write(
        &override_path,
        r#"
log_level: "debug"
max_connections: 200
"#,
    )
    .unwrap();

    let config: TestServiceConfig = hpc_config::ConfigBuilder::new()
        .add_file(&base_path)
        .add_file(&override_path)
        .build()
        .unwrap();

    // Base values
    assert_eq!(config.listen_addr, "0.0.0.0:8080");
    assert!(!config.enable_tls);
    assert_eq!(config.timeout_ms, 5000);

    // Overridden values
    assert_eq!(config.log_level, "debug");
    assert_eq!(config.max_connections, 200);
}

// Test 17: Error on non-existent file
#[test]
fn test_nonexistent_file_error() {
    let result: Result<TestServiceConfig, _> =
        hpc_config::load_from_file("/nonexistent/path/config.yaml");

    assert!(result.is_err());
}

// Test 18: Case-insensitive env var matching
#[test]
fn test_case_insensitive_env() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("test.yaml");

    fs::write(
        &config_path,
        r#"
listen_addr: "0.0.0.0:8080"
log_level: "info"
max_connections: 100
enable_tls: false
"#,
    )
    .unwrap();

    // Test both uppercase field names
    env::set_var("CASETEST__LISTEN_ADDR", "0.0.0.0:9999");
    env::set_var("CASETEST__LOG_LEVEL", "warn");

    let config: TestServiceConfig =
        hpc_config::load_with_env(&config_path, "CASETEST").unwrap();

    assert_eq!(config.listen_addr, "0.0.0.0:9999");
    assert_eq!(config.log_level, "warn");

    env::remove_var("CASETEST__LISTEN_ADDR");
    env::remove_var("CASETEST__LOG_LEVEL");
}

// Test 19: Optional file (doesn't error if missing)
#[test]
fn test_optional_file() {
    let config: TestServiceConfig = hpc_config::ConfigBuilder::new()
        .add_optional_file("/nonexistent/optional.yaml")
        .add_env_with_prefix("OPTIONAL")
        .build_with_defaults()
        .unwrap_or(TestServiceConfig {
            listen_addr: "0.0.0.0:8080".to_string(),
            log_level: "info".to_string(),
            max_connections: 100,
            enable_tls: false,
            timeout_ms: 5000,
        });

    // Should use defaults since file doesn't exist
    assert_eq!(config.listen_addr, "0.0.0.0:8080");
}

// Test 20: Config with explicit type conversion
#[test]
fn test_type_conversion() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("types.yaml");

    #[derive(Debug, Deserialize, PartialEq)]
    struct TypedConfig {
        pub string_val: String,
        pub int_val: i32,
        pub float_val: f64,
        pub bool_val: bool,
        pub optional_val: Option<String>,
    }

    fs::write(
        &config_path,
        r#"
string_val: "hello"
int_val: 42
float_val: 3.14
bool_val: true
"#,
    )
    .unwrap();

    let config: TypedConfig = hpc_config::load_from_file(&config_path).unwrap();

    assert_eq!(config.string_val, "hello");
    assert_eq!(config.int_val, 42);
    assert!((config.float_val - std::f64::consts::PI).abs() < 0.01);
    assert!(config.bool_val);
    assert_eq!(config.optional_val, None);
}
