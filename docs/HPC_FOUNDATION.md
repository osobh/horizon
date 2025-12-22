# HPC Foundation - Enterprise Infrastructure Layer

The HPC Foundation provides enterprise-grade infrastructure primitives for StratoSwarm, merged from the Horizon project. These 13 crates add robust security, observability, cloud abstraction, and communication layers.

## Overview

| Category | Crates | Tests | Purpose |
|----------|--------|-------|---------|
| Security & Identity | 4 | 257 | mTLS, encryption, policy engine |
| Communication | 3 | 65 | gRPC, QUIC, MCP protocol |
| Configuration & Observability | 3 | 63 | Config, tracing, error handling |
| Cloud & Resources | 3 | 135 | Provider abstraction, resource model |
| **Total** | **13** | **520+** | Enterprise infrastructure |

## Crate Reference

### Security & Identity

#### hpc-auth (111 tests)
Zero-trust mTLS authentication and service identity.

```rust
use hpc_auth::cert::{generate_ca_cert, generate_signed_cert, ServiceIdentity};

// Generate CA and service certificates
let ca = generate_ca_cert("StratoSwarm CA")?;
let service = ServiceIdentity::new("my-service");
let cert = generate_signed_cert(&service, &ca)?;

// Create mTLS server config
let server_config = create_server_config_with_client_auth(&cert, &ca)?;
```

**Features:**
- Certificate generation (CA and service certs)
- rustls integration for TLS 1.3
- mTLS with client certificate verification
- Service identity extraction from certificates
- PEM format support

#### hpc-crypto (71 tests)
Cryptographic primitives for authentication and integrity.

```rust
use hpc_crypto::signing::KeyPair;
use hpc_crypto::merkle::MerkleTree;

// Digital signatures
let keypair = KeyPair::generate();
let signature = keypair.sign(b"message");
assert!(keypair.verify(b"message", &signature));

// Merkle trees for audit trails
let tree = MerkleTree::from_leaves(&[b"tx1", b"tx2", b"tx3"]);
let proof = tree.prove(1)?;
assert!(tree.verify(&proof, b"tx2"));
```

**Features:**
- Ed25519 digital signatures
- Blake3 cryptographic hashing
- Merkle trees with inclusion proofs
- Automatic key zeroization

#### hpc-vault (54 tests)
Secure credential encryption and storage.

```rust
use hpc_vault::{MasterKey, VaultEncryption};

// Derive master key from passphrase
let master_key = MasterKey::from_passphrase("passphrase", b"salt")?;
let vault = VaultEncryption::new(master_key);

// Encrypt/decrypt credentials
let encrypted = vault.encrypt(b"aws-secret-key")?;
let decrypted = vault.decrypt(&encrypted)?;
```

**Features:**
- AES-256-GCM encryption
- Argon2 key derivation
- Secure memory zeroing (zeroize)
- Key rotation support

#### hpc-policy (21 tests)
Declarative policy engine for authorization.

```rust
use hpc_policy::{parse_policy, evaluate, EvaluationContext};

let policy = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: gpu-access
spec:
  principals:
    - type: role
      value: gpu-user
  resources:
    - type: job
      pattern: "jobs/*"
  actions: ["read", "execute"]
  effect: allow
"#;

let policy = parse_policy(policy)?;
let ctx = EvaluationContext::new(principal, resource, "execute");
let allowed = evaluate(&policy, &ctx)?;
```

**Features:**
- YAML-based policy DSL
- RBAC and ABAC support
- Glob pattern matching
- Rich condition operators

### Communication & Protocols

#### hpc-rpc (23 tests)
High-performance RPC with gRPC and QUIC.

```rust
use hpc_rpc::grpc::{GrpcServerBuilder, GrpcClientBuilder};

// Server with mTLS
let server = GrpcServerBuilder::new(addr)
    .with_tls(server_cert)?
    .with_client_auth(ca)?
    .add_service(MyService::new())
    .build()?;

// Client with connection pooling
let channel = GrpcClientBuilder::new("https://server:50051")
    .with_server_ca(ca)
    .with_client_cert(client_cert)
    .with_pool_size(5)
    .connect()
    .await?;
```

**Features:**
- gRPC server/client with mTLS
- QUIC endpoints for low-latency streaming
- Connection pooling
- Backpressure handling

#### hpc-mcp (28 tests)
Model Context Protocol for AI agent integration.

```rust
use hpc_mcp::{McpServer, Tool, ToolHandler};

struct EchoHandler;

#[async_trait]
impl ToolHandler for EchoHandler {
    async fn handle(&self, call: ToolCall) -> Result<ToolResult> {
        Ok(ToolResult::success(vec![Content::text(&call.arguments)]))
    }
}

let server = McpServer::builder()
    .register_tool(Tool::new("echo", "Echo a message"), EchoHandler)
    .build();
```

**Features:**
- JSON-RPC 2.0 protocol
- Tool registry with type-safe schemas
- Async server and client
- AI agent tool access

#### hpc-types (14 tests)
Protobuf schema definitions and builders.

```rust
use hpc_types::telemetry_helpers::GpuMetricBuilder;

let metric = GpuMetricBuilder::new("host-001", "gpu-0")
    .utilization(87.5)
    .memory(64.5, 80.0)
    .temperature(68.5)
    .power(350.0)
    .build();
```

**Features:**
- Common protobuf messages
- Telemetry message builders
- Timestamp helpers
- Validation utilities

### Configuration & Observability

#### hpc-config (29 tests)
Layered configuration with secret management.

```rust
use hpc_config::ConfigBuilder;

#[derive(Deserialize)]
struct AppConfig {
    listen_addr: String,
    log_level: String,
    #[serde(default)]
    database_url: Option<String>,
}

let config: AppConfig = ConfigBuilder::new()
    .add_file("config.yaml")?
    .add_env_prefix("APP")
    .build()?;
```

**Features:**
- YAML, TOML, JSON support
- Environment variable overrides
- Secret file loading
- Nested configuration

#### hpc-tracing (30 tests)
Unified tracing and metrics.

```rust
use hpc_tracing::{TracingConfig, init, init_metrics};

// Initialize metrics endpoint
init_metrics("0.0.0.0:9090".parse()?)?;

// Initialize tracing with OTLP export
let config = TracingConfig {
    service_name: "my-service".to_string(),
    log_level: "info".to_string(),
    otlp_endpoint: Some("http://collector:4317".to_string()),
};
let _guard = init(config)?;
```

**Features:**
- OpenTelemetry export
- Prometheus metrics endpoint
- Structured logging
- Guard pattern for cleanup

#### hpc-error (6 tests)
Unified error handling.

```rust
use hpc_error::{HpcError, Result};

fn operation() -> Result<String> {
    Err(HpcError::Database("connection failed".to_string()))
}

// Converts to tonic::Status for gRPC
let status: tonic::Status = error.into();

// Check if retriable
if error.is_retriable() {
    // Retry logic
}
```

**Features:**
- Comprehensive error variants
- gRPC status conversion
- Error categorization
- anyhow integration

### Cloud & Resources

#### hpc-provider (46 tests)
Cloud provider abstraction.

```rust
use hpc_provider::{CapacityProvider, QuoteRequest, ProvisionSpec};

#[async_trait]
impl CapacityProvider for AwsProvider {
    async fn get_quote(&self, request: &QuoteRequest) -> Result<Quote> {
        // Get pricing for instances
    }

    async fn provision(&self, spec: &ProvisionSpec) -> Result<ProvisionResult> {
        // Provision cloud instances
    }

    async fn check_spot_prices(&self, instance_type: &str, region: &str) -> Result<SpotPrices> {
        // Check spot market
    }
}
```

**Features:**
- Unified provider trait
- Quote and provisioning APIs
- Spot price monitoring
- Service quota tracking

#### hpc-resources (46 tests)
Universal resource abstraction.

```rust
use hpc_resources::{ResourceRequest, ResourceSpec, GpuVendor};

let request = ResourceRequest::new("gpu-job")
    .add_gpu_nvidia_h100(4)
    .add_memory(128, ResourceUnit::GB)
    .add_storage(1, ResourceUnit::TB)
    .with_priority(RequestPriority::High)
    .build()?;
```

**Features:**
- Multi-resource requests (GPU, TPU, CPU, memory, storage, network)
- Vendor-specific constraints (Nvidia, AMD, Intel, Apple)
- Allocation tracking
- Audit trails

#### hpc-tsdb (23 tests)
Time-series database abstractions.

```rust
use hpc_tsdb::{InfluxDbClient, QueryBuilder, TimeRange};

let client = InfluxDbClient::new("http://influxdb:8086", "token")?;

let query = QueryBuilder::new("gpu_metrics")
    .filter("host", "server-01")
    .range(TimeRange::last_hour())
    .aggregate(Aggregation::Mean, "1m")
    .build();

let series = client.query(query).await?;
```

**Features:**
- InfluxDB client
- Query builder
- Time range utilities
- Aggregation support

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      StratoSwarm Application                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   hpc-mcp   │  │   hpc-rpc   │  │ hpc-tracing │             │
│  │ (AI Tools)  │  │ (gRPC/QUIC) │  │(Observability)│            │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                 │                     │
│  ┌──────┴────────────────┴─────────────────┴──────┐             │
│  │                   hpc-auth                      │             │
│  │            (mTLS, Service Identity)             │             │
│  └──────────────────────┬──────────────────────────┘             │
│                         │                                        │
│  ┌──────────────────────┼──────────────────────┐                │
│  │                      │                       │                │
│  │  ┌─────────────┐  ┌──┴──────────┐  ┌───────┴─────┐          │
│  │  │ hpc-crypto  │  │  hpc-vault  │  │  hpc-policy │          │
│  │  │(Signatures) │  │ (Encryption)│  │   (RBAC)    │          │
│  │  └─────────────┘  └─────────────┘  └─────────────┘          │
│  │                                                              │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  │hpc-provider │  │hpc-resources│  │  hpc-tsdb   │          │
│  │  │   (Cloud)   │  │ (Resources) │  │(Time Series)│          │
│  │  └─────────────┘  └─────────────┘  └─────────────┘          │
│  │                                                              │
│  └──────────────────────────────────────────────────────────────┘
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │        hpc-config           │         hpc-error             ││
│  │    (Configuration)          │      (Error Handling)         ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Integration with StratoSwarm

The HPC Foundation integrates seamlessly with StratoSwarm's existing components:

| StratoSwarm Component | HPC Integration |
|----------------------|-----------------|
| GPU Agents | hpc-auth for identity, hpc-rpc for communication |
| Consensus Layer | hpc-crypto for signatures, hpc-tracing for metrics |
| Knowledge Graph | hpc-mcp for AI agent tool access |
| Evolution Engine | hpc-policy for safety constraints |
| Multi-Region | hpc-provider for cloud abstraction |
| Monitoring | hpc-tracing + hpc-tsdb for observability |

## Testing

All HPC foundation crates are fully tested:

```bash
# Run all HPC foundation tests
cargo test -p hpc-error -p hpc-crypto -p hpc-config -p hpc-tracing \
    -p hpc-auth -p hpc-policy -p hpc-vault -p hpc-mcp \
    -p hpc-rpc -p hpc-resources -p hpc-types -p hpc-provider -p hpc-tsdb

# 520+ tests passing
```

## Future Work

- **Horizon Services**: Enable the full cost intelligence and enterprise services stack
- **Horizon Agents**: Enable the autonomous operations agents
- **Integration Tests**: Cross-crate integration testing
- **Benchmarks**: Performance benchmarks for RPC and crypto operations
