# StratoSwarm Swarmlet: Easy Node Joining System

## Overview

The StratoSwarm Swarmlet is a lightweight Docker container that enables any device to easily join an existing StratoSwarm cluster without requiring repository clones, complex builds, or manual configuration. It's designed to work universally across architectures (x86_64, ARM64) and device types (servers, workstations, laptops, Raspberry Pi).

## Architecture

### Core Concept

```
┌─────────────────────────────────────────────────────────────────┐
│                    StratoSwarm Cluster                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ DataCenter  │  │ Workstation │  │   Laptop    │            │
│  │ (Full Node) │  │ (Full Node) │  │ (Full Node) │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│         │                 │                 │                 │
│    ┌────┴─────────────────┴─────────────────┴────┐           │
│    │          Cluster Mesh Network                │           │
│    └────┬─────────────────┬─────────────────┬────┘           │
│         │                 │                 │                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Swarmlet   │  │  Swarmlet   │  │  Swarmlet   │            │
│  │ (RaspPi)    │  │  (Edge)     │  │ (IoT Device)│            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### Swarmlet vs Full Node

| Component | Full StratoSwarm Node | Swarmlet |
|-----------|----------------------|----------|
| **Size** | ~500MB+ | <20MB |
| **Capabilities** | Full orchestration | Join + execute work |
| **Dependencies** | Full Rust toolchain | Docker only |
| **Setup Time** | 10-30 minutes | <1 minute |
| **Use Cases** | Control plane, heavy compute | Edge, IoT, lightweight compute |

## Docker Container Design

### Multi-Architecture Support

The swarmlet container is built for multiple architectures:

- **x86_64**: Servers, workstations, most laptops
- **ARM64**: Raspberry Pi 4+, Apple Silicon, ARM servers
- **ARMv7**: Older Raspberry Pi models (optional)

### Container Specifications

```dockerfile
# Multi-stage build for minimal size
FROM --platform=$BUILDPLATFORM rust:1.79-alpine AS builder

# Install cross-compilation tools
RUN apk add --no-cache musl-dev

# Set target architecture
ARG TARGETPLATFORM
RUN case "$TARGETPLATFORM" in \
      "linux/amd64") echo x86_64-unknown-linux-musl > /.target ;; \
      "linux/arm64") echo aarch64-unknown-linux-musl > /.target ;; \
      "linux/arm/v7") echo armv7-unknown-linux-musleabihf > /.target ;; \
    esac

# Build swarmlet binary
WORKDIR /build
COPY crates/swarmlet/ .
RUN rustup target add $(cat /.target)
RUN cargo build --release --target $(cat /.target)
RUN cp target/$(cat /.target)/release/swarmlet /swarmlet

# Final minimal image
FROM scratch
COPY --from=builder /swarmlet /swarmlet
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
ENTRYPOINT ["/swarmlet"]
```

### Expected Container Size

- **Compressed**: ~8-12MB
- **Uncompressed**: ~15-20MB
- **Memory Usage**: ~10-50MB runtime (depends on workload)

## Node Joining Process

### 1. Cluster Preparation

On an existing StratoSwarm cluster node:

```bash
# Generate a join token (valid for 24 hours)
stratoswarm cluster create-join-token --expires 24h

# Output example:
# Join Token: swarm_join_abc123def456...
# Cluster IP: 192.168.1.100:7946
# Expires: 2024-08-01 15:30:00 UTC
```

### 2. Node Discovery & Joining

On the new device (Raspberry Pi, edge device, etc.):

```bash
# Method 1: Direct join with token
docker run --rm -it \
  --network host \
  --name swarmlet \
  stratoswarm/swarmlet:latest \
  join --token swarm_join_abc123def456... \
       --cluster 192.168.1.100:7946

# Method 2: Network discovery (if enabled)
docker run --rm -it \
  --network host \
  --name swarmlet \
  stratoswarm/swarmlet:latest \
  discover-and-join
```

### 3. Automatic Hardware Profiling

The swarmlet automatically profiles the device:

```json
{
  "node_id": "swarmlet-rpi4-001",
  "hardware": {
    "cpu_model": "ARM Cortex-A72",
    "cpu_cores": 4,
    "memory_gb": 8.0,
    "storage_gb": 64.0,
    "architecture": "arm64",
    "device_type": "raspberry_pi_4"
  },
  "capabilities": {
    "gpu_count": 0,
    "network_bandwidth_mbps": 100.0,
    "battery_powered": false,
    "thermal_constraints": {
      "max_temp_celsius": 85.0,
      "current_temp_celsius": 45.0
    }
  },
  "classification": {
    "node_class": "Edge",
    "suitability": ["lightweight_compute", "sensors", "data_collection"]
  }
}
```

### 4. Cluster Integration

Once joined, the swarmlet:

1. **Registers** with cluster mesh
2. **Receives** node classification (Edge, IoT, etc.)
3. **Starts** accepting work assignments
4. **Reports** health and metrics
5. **Participates** in cluster consensus (voting only)

## Configuration Options

### Environment Variables

```bash
# Basic configuration
SWARM_NODE_NAME=my-raspberry-pi-01      # Override default hostname
SWARM_LOG_LEVEL=info                    # debug, info, warn, error
SWARM_DATA_DIR=/data                    # Persistent data directory

# Network configuration
SWARM_DISCOVERY_PORT=7946               # Cluster discovery port
SWARM_API_PORT=8080                     # Local API port
SWARM_BIND_INTERFACE=eth0               # Network interface to bind

# Resource limits
SWARM_MAX_MEMORY_MB=512                 # Memory limit for workloads
SWARM_MAX_CPU_CORES=2                   # CPU cores available to cluster
SWARM_MAX_STORAGE_GB=10                 # Storage available to cluster

# Security
SWARM_TLS_ENABLED=true                  # Enable TLS for cluster communication
SWARM_AUTH_TOKEN_FILE=/secrets/token    # Path to join token file
```

### Volume Mounts

```bash
# Persistent data storage
-v /opt/swarmlet/data:/data

# Configuration files
-v /opt/swarmlet/config:/config

# Device access (for sensors, GPIO)
-v /dev:/dev --privileged

# Docker socket (for container workloads)
-v /var/run/docker.sock:/var/run/docker.sock
```

## Networking Requirements

### Ports

| Port | Protocol | Purpose | Required |
|------|----------|---------|----------|
| 7946 | TCP/UDP | Cluster discovery & gossip | Yes |
| 8080 | TCP | Local API & health checks | Yes |
| 9090 | TCP | Metrics & monitoring | Optional |
| 443 | TCP | Secure cluster communication | If TLS enabled |

### Network Discovery

The swarmlet supports multiple discovery methods:

1. **Direct Connection**: Using IP address and join token
2. **mDNS**: Automatic discovery on local network
3. **Broadcast**: UDP broadcast for cluster discovery
4. **DNS**: Service discovery via DNS TXT records

## Security Model

### Authentication & Authorization

```
┌─────────────────────────────────────────────────────────────────┐
│                     Security Flow                                │
│                                                                 │
│  1. Generate Join Token                                         │
│     ┌─────────────────┐                                        │
│     │ Full Node       │ generates token with:                   │
│     │ (Cluster Admin) │ - Expiration time                      │
│     └─────────────────┘ - Node capabilities allowed            │
│              │          - Network constraints                   │
│              ▼                                                  │
│  2. Token Exchange                                              │
│     ┌─────────────────┐ ◄──── Token + Node Profile ────────┐  │
│     │ Cluster         │                                    │   │
│     │ Coordinator     │ ────── Node Certificate ──────────► │  │
│     └─────────────────┘                                    │   │
│              │                                             │   │
│              ▼                                             │   │
│  3. Ongoing Authentication                               ┌─────┴─┐
│     ┌─────────────────┐ ◄──── Heartbeat + Metrics ─────│Swarmlet│
│     │ Cluster Mesh    │                                 └───────┘
│     │ Network         │ ────── Work Assignments ──────────────► │
│     └─────────────────┘                                         │
└─────────────────────────────────────────────────────────────────┘
```

### Cryptographic Security

- **Join Tokens**: Ed25519 signatures with time-based expiration
- **Node Certificates**: x.509 certificates for ongoing authentication  
- **Communication**: TLS 1.3 for all cluster communication
- **Work Isolation**: Containers run with limited privileges

## Use Cases & Examples

### 1. Raspberry Pi Edge Computing

```bash
# On cluster coordinator
stratoswarm cluster create-join-token --node-type edge --expires 1h

# On Raspberry Pi
docker run -d \
  --restart unless-stopped \
  --network host \
  --name swarmlet \
  -v /opt/swarmlet:/data \
  -e SWARM_NODE_NAME=garden-sensor-01 \
  -e SWARM_MAX_MEMORY_MB=256 \
  stratoswarm/swarmlet:latest \
  join --token <TOKEN> --cluster 192.168.1.100:7946
```

**Workloads**: Sensor data collection, image processing, IoT gateway

### 2. Laptop Development Node

```bash
# Join laptop to development cluster
docker run -d \
  --network host \
  --name swarmlet-dev \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e SWARM_NODE_NAME=$(hostname)-dev \
  -e SWARM_MAX_CPU_CORES=4 \
  stratoswarm/swarmlet:latest \
  join --token <TOKEN> --cluster dev-cluster.local:7946
```

**Workloads**: Development testing, CI/CD runners, load testing

### 3. Multi-Cloud Edge

```bash
# Join cloud VM to hybrid cluster
docker run -d \
  --network host \
  --name swarmlet-cloud \
  -e SWARM_NODE_NAME=aws-us-west-2-edge \
  -e SWARM_MAX_MEMORY_MB=2048 \
  -e SWARM_TLS_ENABLED=true \
  stratoswarm/swarmlet:latest \
  join --token <TOKEN> --cluster main-cluster.company.com:7946
```

**Workloads**: Data processing, geographic distribution, backup services

## Monitoring & Observability

### Health Checks

The swarmlet exposes health endpoints:

```bash
# Container health
curl http://localhost:8080/health

# Cluster connectivity  
curl http://localhost:8080/cluster/status

# Resource usage
curl http://localhost:8080/metrics
```

### Metrics Integration

Prometheus metrics are available at `/metrics`:

- `swarmlet_uptime_seconds`
- `swarmlet_cpu_usage_percent`
- `swarmlet_memory_usage_bytes`
- `swarmlet_network_bytes_total`
- `swarmlet_workloads_active`
- `swarmlet_cluster_heartbeat_success`

### Logging

```bash
# View swarmlet logs
docker logs swarmlet

# Follow logs in real-time
docker logs -f swarmlet

# Export logs to cluster
docker run ... -e SWARM_LOG_EXPORT=cluster swarmlet
```

## Troubleshooting

### Common Issues

**Connection Failed**
```bash
# Check network connectivity
docker run --rm --network host stratoswarm/swarmlet:latest \
  test-connection --cluster 192.168.1.100:7946

# Check firewall rules
sudo ufw status
sudo iptables -L
```

**Token Expired**
```bash
# Generate new token from cluster
stratoswarm cluster create-join-token --expires 1h

# Or use discovery if enabled
docker run --rm --network host stratoswarm/swarmlet:latest discover
```

**Resource Constraints**
```bash
# Check available resources
docker run --rm stratoswarm/swarmlet:latest profile-hardware

# Adjust limits
docker run ... -e SWARM_MAX_MEMORY_MB=128 swarmlet
```

### Debug Mode

```bash
# Run with debug logging
docker run --rm -it \
  --network host \
  -e SWARM_LOG_LEVEL=debug \
  stratoswarm/swarmlet:latest \
  join --token <TOKEN> --cluster <CLUSTER>
```

## Roadmap

### Phase 1: Core Implementation (Current Focus)
- [x] Basic swarmlet crate structure
- [ ] Docker multi-architecture build
- [ ] Join protocol implementation
- [ ] Hardware profiling
- [ ] Basic workload execution

### Phase 2: Enhanced Features
- [ ] Network auto-discovery (mDNS)
- [ ] Advanced security (mTLS, RBAC)
- [ ] GPU passthrough support
- [ ] Device-specific optimizations

### Phase 3: Ecosystem Integration
- [ ] Kubernetes compatibility layer
- [ ] Cloud provider integrations
- [ ] IoT device templates
- [ ] Enterprise management features

## Development & Contributing

### Local Development

```bash
# Clone repository
git clone https://github.com/stratoswarm/stratoswarm
cd stratoswarm

# Build swarmlet locally
cd crates/swarmlet
cargo build --release

# Run tests
cargo test

# Build Docker container
docker build -t swarmlet-dev .
```

### Cross-Compilation Testing

```bash
# Test ARM64 build
docker buildx build --platform linux/arm64 -t swarmlet-arm64 .

# Test on Raspberry Pi
scp target/aarch64-unknown-linux-musl/release/swarmlet pi@raspberrypi.local:~/
ssh pi@raspberrypi.local './swarmlet --help'
```

This swarmlet system transforms the node joining experience from complex (clone repo, build, configure) to simple (docker run with token), making StratoSwarm accessible to any device type while maintaining security and performance.