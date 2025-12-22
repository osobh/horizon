# Getting Started with StratoSwarm

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Basic Concepts](#basic-concepts)
5. [Your First Service](#your-first-service)
6. [Security Setup](#security-setup)
7. [Observability](#observability)
8. [Running a Swarm](#running-a-swarm)
9. [Next Steps](#next-steps)

## Prerequisites

### Hardware Requirements

**Linux (NVIDIA GPU)**
- **GPU**: NVIDIA GPU with Compute Capability 7.0+ (RTX 20 series or newer)
- **Memory**:
  - Minimum: 32GB RAM, 8GB VRAM
  - Recommended: 96GB RAM, 32GB VRAM (RTX 5090)
- **Storage**: 100GB+ NVMe SSD

**macOS (Apple Silicon)**
- **GPU**: Apple M1, M2, M3, or M4 (any variant)
- **Memory**: 16GB+ unified memory
- **macOS**: 13.0+ (Ventura)
- **Metal**: 3.0+

### Software Requirements

- **Rust**: 1.70 or newer
- **CUDA**: 11.8+ (Linux with NVIDIA GPU)
- **Xcode Command Line Tools**: Latest (macOS)
- **Docker**: Optional but recommended

## Installation

### 1. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustc --version
```

### 2. Platform-Specific Setup

**Linux with NVIDIA GPU:**
```bash
# Verify CUDA installation
nvcc --version
nvidia-smi

# If CUDA not installed:
# https://developer.nvidia.com/cuda-downloads
```

**macOS with Apple Silicon:**
```bash
# Verify Metal support
system_profiler SPDisplaysDataType | grep Metal

# Install Xcode command line tools
xcode-select --install
```

### 3. Clone and Build StratoSwarm

```bash
git clone https://github.com/stratoswarm/stratoswarm.git
cd stratoswarm

# Build all components
cargo build --release

# Run tests (2500+ tests across 70 crates)
cargo test --release
```

## Quick Start

### One-Command Launch

```bash
# Intelligent platform launch
./quickstart-stratoswarm.sh

# This automatically:
# 1. Detects your hardware (GPU type, memory, CPU)
# 2. Generates mTLS certificates via hpc-auth
# 3. Initializes observability (hpc-tracing)
# 4. Starts appropriate components
# 5. Shows access URLs
```

### Manual Launch

```bash
# Coordinator node
./launch-stratoswarm-platform.sh

# Worker nodes (on other machines)
./attach-swarmlet.sh docker <coordinator-ip>:7946 <cluster-token>
```

## Basic Concepts

### Architecture Layers

StratoSwarm consists of four integrated layers:

| Layer | Components | Purpose |
|-------|------------|---------|
| **Core Platform** | GPU Consensus, Memory Tiers, Evolution | GPU-native distributed primitives |
| **Enterprise (HPC)** | Auth, Crypto, Policy, RPC, Config, Tracing | Production infrastructure |
| **Intelligence** | Knowledge Graph, Neural Router, MCP | ML-powered optimization |
| **Agents** | Horizon Agents (9 specialized) | Autonomous operations |

### Memory Tiers

StratoSwarm uses GPU-primary memory:

| Tier | Type | Capacity | Use Case |
|------|------|----------|----------|
| 1 | GPU | 32GB | Active computation |
| 2 | CPU | 96GB | Warm cache |
| 3 | NVMe | 3.2TB | Recent data |

### Security Model

All communication uses mTLS with certificates from `hpc-auth`:
- Automatic CA and service certificate generation
- Ed25519 signatures for integrity
- Policy-based access control

## Your First Service

### Creating a Secure gRPC Service

```rust
use hpc_auth::cert::{generate_ca_cert, generate_signed_cert, ServiceIdentity};
use hpc_rpc::grpc::GrpcServerBuilder;
use hpc_tracing::{TracingConfig, init, init_metrics};
use hpc_error::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Initialize observability
    init_metrics("0.0.0.0:9090".parse()?)?;
    let config = TracingConfig {
        service_name: "my-service".to_string(),
        log_level: "info".to_string(),
        otlp_endpoint: None,
    };
    let _guard = init(config)?;

    // 2. Generate certificates
    let ca = generate_ca_cert("StratoSwarm CA")?;
    let identity = ServiceIdentity::new("my-service");
    let cert = generate_signed_cert(&identity, &ca)?;

    // 3. Create mTLS server
    let server = GrpcServerBuilder::new("0.0.0.0:50051".parse()?)
        .with_tls(cert)?
        .with_client_auth(ca)?
        .add_service(MyService::new())
        .build()?;

    tracing::info!("Server starting on :50051");
    server.serve().await?;

    Ok(())
}
```

### Creating a Client

```rust
use hpc_auth::cert::{generate_signed_cert, ServiceIdentity};
use hpc_rpc::grpc::GrpcClientBuilder;

async fn connect_to_service(
    ca: &CertificateWithKey,
    server_url: &str,
) -> Result<MyServiceClient<Channel>> {
    // Generate client certificate
    let identity = ServiceIdentity::new("my-client");
    let cert = generate_signed_cert(&identity, ca)?;

    // Connect with mTLS
    let channel = GrpcClientBuilder::new(server_url)
        .with_server_ca(ca.clone())
        .with_client_cert(cert)
        .with_pool_size(5)
        .connect()
        .await?;

    Ok(MyServiceClient::new(channel))
}
```

## Security Setup

### Certificate Management

```rust
use hpc_auth::cert::{generate_ca_cert, generate_signed_cert, ServiceIdentity};

// Generate CA (once per cluster)
let ca = generate_ca_cert("StratoSwarm Cluster CA")?;

// Generate service certificates
let services = ["consensus", "scheduler", "coordinator"];
for service in services {
    let identity = ServiceIdentity::new(service);
    let cert = generate_signed_cert(&identity, &ca)?;
    // Store cert for the service
}
```

### Secret Encryption

```rust
use hpc_vault::{MasterKey, VaultEncryption};

// Derive master key from passphrase
let master_key = MasterKey::from_passphrase(
    "cluster-passphrase",
    b"unique-cluster-salt",
)?;

let vault = VaultEncryption::new(master_key);

// Encrypt secrets
let encrypted = vault.encrypt(b"api-key-12345")?;

// Decrypt when needed
let decrypted = vault.decrypt(&encrypted)?;
```

### Policy-Based Authorization

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
      value: ml-engineer
  resources:
    - type: gpu-job
      pattern: "jobs/*"
  actions: ["create", "read"]
  effect: allow
"#;

let policy = parse_policy(policy)?;
let allowed = evaluate(&policy, &context)?;
```

## Observability

### Tracing and Metrics

```rust
use hpc_tracing::{TracingConfig, init, init_metrics};
use tracing::{info, instrument, span, Level};

// Initialize metrics endpoint (Prometheus)
init_metrics("0.0.0.0:9090".parse()?)?;

// Initialize tracing (with optional OTLP export)
let config = TracingConfig {
    service_name: "gpu-consensus".to_string(),
    log_level: "info".to_string(),
    otlp_endpoint: Some("http://collector:4317".to_string()),
};
let _guard = init(config)?;

// Use structured logging
#[instrument]
async fn process_round(round_id: u64) -> Result<()> {
    info!(round = round_id, "Starting consensus round");

    let span = span!(Level::DEBUG, "validation");
    let _enter = span.enter();

    // Processing with automatic span tracking
    validate_proposals().await?;

    info!("Round complete");
    Ok(())
}
```

### Error Handling

```rust
use hpc_error::{HpcError, Result};

fn database_operation() -> Result<String> {
    // Unified error type
    Err(HpcError::Database("connection failed".to_string()))
}

fn handle_errors() {
    match database_operation() {
        Ok(value) => println!("Success: {}", value),
        Err(e) => {
            // Check if retriable
            if e.is_retriable() {
                // Retry logic
            }

            // Convert to gRPC status if needed
            let status: tonic::Status = e.into();
        }
    }
}
```

## Running a Swarm

### GPU-Native Consensus

```rust
use stratoswarm::{
    consensus::MillionNodeConsensus,
    memory::GpuMemoryTier,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize GPU memory tier
    let memory = GpuMemoryTier::new(MemoryConfig {
        gpu_memory_gb: 32,
        cpu_memory_gb: 128,
        prefer_gpu: true,
        overflow_threshold: 0.85,
    })?;

    // Start consensus
    let consensus = MillionNodeConsensus::new(ConsensusConfig {
        target_latency_ns: 500,  // Sub-microsecond
        byzantine_tolerance: 0.33,
        gpu_acceleration: true,
    })?;

    // Process consensus rounds
    loop {
        let result = consensus.process_round().await?;
        tracing::info!(
            latency_ns = result.latency_ns,
            agreement = result.agreement_percentage,
            "Consensus round complete"
        );
    }
}
```

### Evolution Engine

```rust
use stratoswarm::evolution::{KernelHotSwap, EvolutionConfig};

let evolution = KernelHotSwap::new(EvolutionConfig {
    enable_zero_downtime: true,
    performance_monitoring: true,
    cross_cluster_sharing: true,
})?;

// Evolve with performance feedback
for gen in 0..100 {
    let metrics = evolution.evolve_generation().await?;
    tracing::info!(
        generation = gen,
        best_fitness = metrics.best_fitness,
        "Evolution complete"
    );
}
```

## Next Steps

### 1. Explore Documentation

| Document | Description |
|----------|-------------|
| [HPC Foundation](../HPC_FOUNDATION.md) | Enterprise infrastructure reference |
| [Performance Tuning](./performance-tuning.md) | Optimization guide |
| [Swarmlet Guide](../swarmlet.md) | Node deployment |
| [API Reference](../api/) | GraphQL and REST APIs |

### 2. Run Examples

```bash
# List examples
ls examples/

# Run specific examples
cargo run --example evolution_demo --release
cargo run --example knowledge_graph --release
```

### 3. Join the Community

- **GitHub**: https://github.com/stratoswarm/stratoswarm
- **Discord**: https://discord.gg/stratoswarm
- **Documentation**: https://docs.stratoswarm.io

### 4. Deploy to Production

```bash
# Launch coordinator
./launch-stratoswarm-platform.sh

# Add worker nodes
./attach-swarmlet.sh docker <coordinator>:7946 <token>

# Deploy applications
./target/release/stratoswarm deploy app.swarm
```

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
nvidia-smi  # Check memory usage
# Reduce agent count or batch size
```

**Metal Shader Compilation Slow**
```bash
# First run compiles shaders - subsequent runs use cache
# Warm compilation: ~23μs, Cold: ~100ms
```

**mTLS Connection Failed**
```rust
// Ensure CA matches between server and client
// Check certificate expiry
// Verify service identity matches
```

**Low GPU Utilization**
```rust
// Enable aggressive optimization
let config = ControllerConfig {
    target_utilization: 0.95,
    aggressive_mode: true,
    ..Default::default()
};
```

For more help, see [CONTRIBUTING.md](../../CONTRIBUTING.md) or open a GitHub issue.
