use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use serde::{Deserialize, Serialize};
use std::fs;
use tempfile::TempDir;

#[derive(Debug, Clone, Deserialize, Serialize)]
struct SimpleConfig {
    pub listen_addr: String,
    pub log_level: String,
    pub max_connections: u32,
    pub enable_tls: bool,
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct NestedConfig {
    pub server: SimpleConfig,
    pub database: DatabaseConfig,
    pub features: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct DatabaseConfig {
    pub host: String,
    pub port: u16,
    pub database: String,
}

fn bench_load_yaml(c: &mut Criterion) {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("bench.yaml");

    let yaml_content = r#"
listen_addr: "0.0.0.0:8080"
log_level: "info"
max_connections: 100
enable_tls: true
timeout_ms: 5000
"#;
    fs::write(&config_path, yaml_content).unwrap();

    c.bench_function("load_yaml_simple", |b| {
        b.iter(|| {
            let config: SimpleConfig = hpc_config::load_from_file(&config_path).unwrap();
            black_box(config);
        });
    });
}

fn bench_load_toml(c: &mut Criterion) {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("bench.toml");

    let toml_content = r#"
listen_addr = "0.0.0.0:8080"
log_level = "info"
max_connections = 100
enable_tls = true
timeout_ms = 5000
"#;
    fs::write(&config_path, toml_content).unwrap();

    c.bench_function("load_toml_simple", |b| {
        b.iter(|| {
            let config: SimpleConfig = hpc_config::load_from_file(&config_path).unwrap();
            black_box(config);
        });
    });
}

fn bench_load_json(c: &mut Criterion) {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("bench.json");

    let json_content = r#"
{
    "listen_addr": "0.0.0.0:8080",
    "log_level": "info",
    "max_connections": 100,
    "enable_tls": true,
    "timeout_ms": 5000
}
"#;
    fs::write(&config_path, json_content).unwrap();

    c.bench_function("load_json_simple", |b| {
        b.iter(|| {
            let config: SimpleConfig = hpc_config::load_from_file(&config_path).unwrap();
            black_box(config);
        });
    });
}

fn bench_load_with_env(c: &mut Criterion) {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("bench.yaml");

    let yaml_content = r#"
listen_addr: "0.0.0.0:8080"
log_level: "info"
max_connections: 100
enable_tls: true
timeout_ms: 5000
"#;
    fs::write(&config_path, yaml_content).unwrap();

    std::env::set_var("BENCH__LOG_LEVEL", "debug");
    std::env::set_var("BENCH__MAX_CONNECTIONS", "500");

    c.bench_function("load_with_env_override", |b| {
        b.iter(|| {
            let config: SimpleConfig =
                hpc_config::load_with_env(&config_path, "BENCH").unwrap();
            black_box(config);
        });
    });

    std::env::remove_var("BENCH__LOG_LEVEL");
    std::env::remove_var("BENCH__MAX_CONNECTIONS");
}

fn bench_nested_config(c: &mut Criterion) {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("nested.yaml");

    let yaml_content = r#"
server:
  listen_addr: "0.0.0.0:8080"
  log_level: "info"
  max_connections: 100
  enable_tls: true
  timeout_ms: 5000

database:
  host: "localhost"
  port: 5432
  database: "horizon"

features:
  - "feature_a"
  - "feature_b"
  - "feature_c"
"#;
    fs::write(&config_path, yaml_content).unwrap();

    c.bench_function("load_nested_config", |b| {
        b.iter(|| {
            let config: NestedConfig = hpc_config::load_from_file(&config_path).unwrap();
            black_box(config);
        });
    });
}

fn bench_builder_pattern(c: &mut Criterion) {
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

    c.bench_function("builder_multiple_sources", |b| {
        b.iter(|| {
            let config: SimpleConfig = hpc_config::ConfigBuilder::new()
                .add_file(&base_path)
                .add_file(&override_path)
                .build()
                .unwrap();
            black_box(config);
        });
    });
}

fn bench_format_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("format_comparison");

    let temp_dir = TempDir::new().unwrap();

    // YAML
    let yaml_path = temp_dir.path().join("config.yaml");
    fs::write(
        &yaml_path,
        r#"
listen_addr: "0.0.0.0:8080"
log_level: "info"
max_connections: 100
enable_tls: true
timeout_ms: 5000
"#,
    )
    .unwrap();

    group.bench_with_input(BenchmarkId::new("load", "yaml"), &yaml_path, |b, path| {
        b.iter(|| {
            let config: SimpleConfig = hpc_config::load_from_file(path).unwrap();
            black_box(config);
        });
    });

    // TOML
    let toml_path = temp_dir.path().join("config.toml");
    fs::write(
        &toml_path,
        r#"
listen_addr = "0.0.0.0:8080"
log_level = "info"
max_connections = 100
enable_tls = true
timeout_ms = 5000
"#,
    )
    .unwrap();

    group.bench_with_input(BenchmarkId::new("load", "toml"), &toml_path, |b, path| {
        b.iter(|| {
            let config: SimpleConfig = hpc_config::load_from_file(path).unwrap();
            black_box(config);
        });
    });

    // JSON
    let json_path = temp_dir.path().join("config.json");
    fs::write(
        &json_path,
        r#"{
    "listen_addr": "0.0.0.0:8080",
    "log_level": "info",
    "max_connections": 100,
    "enable_tls": true,
    "timeout_ms": 5000
}"#,
    )
    .unwrap();

    group.bench_with_input(BenchmarkId::new("load", "json"), &json_path, |b, path| {
        b.iter(|| {
            let config: SimpleConfig = hpc_config::load_from_file(path).unwrap();
            black_box(config);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_load_yaml,
    bench_load_toml,
    bench_load_json,
    bench_load_with_env,
    bench_nested_config,
    bench_builder_pattern,
    bench_format_comparison
);
criterion_main!(benches);
