# horizon-configx

A flexible, layered configuration management system for Horizon services.

## Features

- **Multiple Format Support**: YAML, TOML, and JSON configuration files
- **Environment Variable Overrides**: Hierarchical env var support with custom prefixes
- **Type Safety**: Full serde deserialization with compile-time type checking
- **Secret Management**: Load secrets from files or environment variables with redaction
- **Builder Pattern**: Flexible configuration composition from multiple sources
- **Layered Configuration**: Clear precedence: Environment > Files > Defaults

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
horizon-configx = { path = "../configx" }
serde = { version = "1.0", features = ["derive"] }
```

## Quick Start

### Basic Usage

```rust
use serde::Deserialize;
use horizon_configx;

#[derive(Deserialize)]
struct MyConfig {
    listen_addr: String,
    log_level: String,
    #[serde(default = "default_timeout")]
    timeout_ms: u64,
}

fn default_timeout() -> u64 { 5000 }

// Load from conventional path: config/{service-name}.yaml
let config: MyConfig = horizon_configx::load("my-service")?;

// Or load from specific file
let config: MyConfig = horizon_configx::load_from_file("config.yaml")?;
```

### Environment Variable Overrides

Environment variables follow the pattern: `{PREFIX}__{FIELD_NAME}`

```bash
# Override specific fields
export MYSERVICE__LISTEN_ADDR="0.0.0.0:8080"
export MYSERVICE__LOG_LEVEL="debug"
export MYSERVICE__TIMEOUT_MS="10000"

# Nested fields use double underscores
export MYSERVICE__DATABASE__HOST="db.example.com"
export MYSERVICE__DATABASE__PORT="5432"
```

```rust
let config: MyConfig = horizon_configx::load_with_env(
    "config.yaml",
    "MYSERVICE"
)?;
```

### Configuration Builder

For advanced scenarios with multiple sources:

```rust
use horizon_configx::ConfigBuilder;

let config: MyConfig = ConfigBuilder::new()
    .add_file("base.yaml")              // Base configuration
    .add_file("production.yaml")        // Environment-specific overrides
    .add_env_with_prefix("MYSERVICE")   // Environment variable overrides
    .build()?;
```

### Secret Management

Secrets are automatically redacted when debug-printed:

```rust
use horizon_configx::{load_secret_from_file, ExposeSecret};

// Load from file (e.g., Kubernetes secret mount)
let db_password = load_secret_from_file("/run/secrets/db_password")?;

// Load from environment variable
let api_key = load_secret_from_env("API_KEY")?;

// Access the actual value only when needed
let password: &str = db_password.expose_secret();

// Debug printing is safe - secrets are redacted
println!("{:?}", db_password); // Output: Secret([REDACTED String])
```

## Configuration Precedence

Configuration values are merged in the following order (later sources override earlier ones):

1. **Default values** (defined via `#[serde(default)]`)
2. **Configuration files** (added in order via `ConfigBuilder`)
3. **Environment variables** (highest priority)

## Supported File Formats

### YAML

```yaml
# config/my-service.yaml
listen_addr: "0.0.0.0:8080"
log_level: "info"
database:
  host: "localhost"
  port: 5432
features:
  - "metrics"
  - "tracing"
```

### TOML

```toml
# config/my-service.toml
listen_addr = "0.0.0.0:8080"
log_level = "info"

[database]
host = "localhost"
port = 5432

features = ["metrics", "tracing"]
```

### JSON

```json
{
  "listen_addr": "0.0.0.0:8080",
  "log_level": "info",
  "database": {
    "host": "localhost",
    "port": 5432
  },
  "features": ["metrics", "tracing"]
}
```

## Advanced Examples

### Complex Nested Configuration

```rust
#[derive(Deserialize)]
struct ServiceConfig {
    pub name: String,
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    #[serde(default)]
    pub features: Vec<String>,
}

#[derive(Deserialize)]
struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
}

#[derive(Deserialize)]
struct DatabaseConfig {
    pub host: String,
    pub port: u16,
    pub database: String,
}

fn default_host() -> String { "0.0.0.0".to_string() }
fn default_port() -> u16 { 8080 }

// Environment overrides for nested fields
// MYSERVICE__SERVER__PORT=9999
// MYSERVICE__DATABASE__HOST=db.prod.com

let config: ServiceConfig = horizon_configx::load_with_env(
    "config/service.yaml",
    "MYSERVICE"
)?;
```

### Optional Configuration Files

```rust
let config: MyConfig = ConfigBuilder::new()
    .add_optional_file("config/base.yaml")        // Won't error if missing
    .add_optional_file("config/local.yaml")       // Local overrides
    .add_env_with_prefix("MYSERVICE")
    .build()?;
```

### Default-Only Configuration

```rust
#[derive(Deserialize)]
struct ConfigWithDefaults {
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_host")]
    pub host: String,
}

fn default_port() -> u16 { 8080 }
fn default_host() -> String { "localhost".to_string() }

// Create config using only defaults
let config: ConfigWithDefaults = ConfigBuilder::new().build()?;

assert_eq!(config.port, 8080);
assert_eq!(config.host, "localhost");
```

## Error Handling

All functions return `Result<T, horizon_error::HorizonError>`:

```rust
use horizon_error::HorizonError;

match horizon_configx::load::<MyConfig>("my-service") {
    Ok(config) => {
        // Use config
    }
    Err(HorizonError::Config(msg)) => {
        eprintln!("Configuration error: {}", msg);
    }
    Err(e) => {
        eprintln!("Unexpected error: {}", e);
    }
}
```

## Testing

The crate includes comprehensive tests:

```bash
# Run all tests
cargo test -p horizon-configx

# Run with output
cargo test -p horizon-configx -- --nocapture

# Run property-based tests
cargo test -p horizon-configx property_tests

# Run benchmarks
cargo bench -p horizon-configx
```

## Benchmarks

Performance characteristics (on typical hardware):

| Operation | Time (µs) | Notes |
|-----------|-----------|-------|
| Load YAML | ~5.6 | Simple config |
| Load TOML | ~5.9 | Simple config |
| Load JSON | ~4.1 | Simple config (fastest) |
| With Env Override | ~11.6 | Includes env var parsing |
| Nested Config | ~10.8 | Complex hierarchical |
| Builder (2 sources) | ~8.8 | Multiple file merge |

Run benchmarks:

```bash
cargo bench -p horizon-configx
```

## Examples

See the `examples/` directory:

```bash
# Run the comprehensive example
cargo run --example service_config -p horizon-configx

# With environment overrides
export EXAMPLE__SERVER__PORT=9999
cargo run --example service_config -p horizon-configx
```

## Design Principles

1. **Type Safety First**: Leverage Rust's type system and serde for compile-time guarantees
2. **Fail Fast**: Configuration errors are detected at startup, not at runtime
3. **Explicit Precedence**: Clear, documented override behavior
4. **Zero Magic**: No hidden configuration sources or implicit behavior
5. **Testability**: All features work with temporary files for easy testing

## Integration with Horizon

This crate is designed to work seamlessly with other Horizon crates:

- **horizon-error**: All errors use the common `HorizonError` type
- **horizon-tracingx**: Configuration for tracing/logging setup
- **horizon-authx**: Load TLS certificates and keys
- Services: Each service loads its configuration using conventional paths
