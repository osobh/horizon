# Horizon

**Unified HPC-AI Platform** - A Tauri-based desktop application providing a single pane of glass to the entire HPC-AI stack.

## Overview

Horizon is the unified interface for the HPC-AI ecosystem, bringing together:
- **Notebook IDE** (ML Researchers) - Interactive Rust notebooks with GPU acceleration
- **Cluster Management** (DevOps) - StratoSwarm cluster topology and .swarm DSL editor
- **Executive Dashboard** - Cost attribution, real-time metrics, and evolution insights

```
┌─────────────────────────────────────────────────────────────────┐
│                      HORIZON (Tauri App)                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    React/Monaco Frontend                  │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │   │
│  │  │ Notebook │  │ Cluster  │  │ .swarm   │  │Executive │  │   │
│  │  │ IDE      │  │ Topology │  │ Editor   │  │Dashboard │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                     Rust Backend (Tauri)                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐  │
│  │ notebook   │  │ cluster    │  │ training   │  │ storage  │  │
│  │ -kernel    │  │ -mesh      │  │ (rtx)      │  │ (warp)   │  │
│  └────────────┘  └────────────┘  └────────────┘  └──────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                  Real-Time Event System                         │
│       metrics:update │ training:progress │ cluster:status       │
└─────────────────────────────────────────────────────────────────┘
```

## Advanced Synergies

Horizon integrates unique capabilities from across the HPC-AI stack that no other platform offers:

### Synergy 1: GPU-Compiled Notebooks
**Projects**: 01-rust + 08-rustybooks + 07-rustytorch
```
User types Rust code → GPU compiler (10x faster) → RTX tensor ops → Output
                         ↓
                  Sub-100ms feedback loop
```
- **Component**: `GpuCompilerPanel.tsx`
- **Bridge**: `gpu_compiler_bridge.rs`
- **Feature Flag**: `gpu-compiler`
- **Unique**: No notebook system compiles on GPU. We achieve instant compilation.

### Synergy 2: Evolution Dashboard
**Projects**: 05-stratoswarm (ADAS + DGM + SwarmAgentic)
```
ADAS explores design space → DGM evolves agent code → SwarmAgentic optimizes
        ↓                           ↓                        ↓
   New agent designs        Self-modifying code       Population search
```
- **Component**: `EvolutionPanel.tsx`
- **Bridge**: `evolution_bridge.rs`
- **Unique**: Three evolution engines working in concert with real-time visualization.

### Synergy 3: RDMA + ZK Visualization
**Projects**: 10-nebula (RDMA + ZK Proofs + Mesh)
```
GPU A (Machine 1) ←→ 400Gbps RDMA ←→ GPU B (Machine 2)
                          ↓
            Zero-knowledge proof of correct training
```
- **Component**: `NetworkTopology.tsx`
- **Bridge**: `nebula_bridge.rs`
- **Feature Flag**: `embedded-nebula`
- **Unique**: GPU-direct transfers with cryptographic verification.

### Synergy 4: GPU Data Pipeline
**Projects**: 04-warp + 08-rustybooks + 10-nebula
```
Data arrives → WARP GPU encryption (20GB/s) → GPUDirect to GPU memory
                        ↓                              ↓
              ChaCha20-Poly1305              Training starts without CPU touch
```
- **Component**: `DataPipeline.tsx`
- **Bridge**: `data_pipeline_bridge.rs`
- **Unique**: End-to-end GPU data path from network to training with 20+ GB/s encryption.

### Synergy 5: Intelligent Edge Proxy
**Projects**: 09-vortex + 03-SLAI + 05-stratoswarm
```
HTTP request → Vortex proxy → SLAI Brain routes → Knowledge graph decides
                    ↓                 ↓                    ↓
         Protocol transmutation   GPU health ML      Best node selection
```
- **Component**: `EdgeProxy.tsx`
- **Bridge**: `edge_proxy_bridge.rs`
- **Unique**: AI-driven request routing with GPU failure prediction and preemptive migration.

### Synergy 6: Tensor Mesh
**Projects**: 10-nebula RDMA + 02-RMPI + 07-rustytorch
```
Distributed GPUs ←→ Type-safe RMPI collectives ←→ RDMA transport
                          ↓
            All-reduce, broadcast, scatter at 400+ Gbps
```
- **Component**: `TensorMesh.tsx`
- **Bridge**: `tensor_mesh_bridge.rs`
- **Unique**: GPU-direct distributed training with compile-time safety.

## Features

### Notebook View (ML Researchers)
- Monaco editor with Rust syntax highlighting
- Local kernel execution with GPU acceleration (via `embedded-kernel` feature)
- Variable inspector with tensor previews
- Cell-based execution with real-time output

### Cluster View (DevOps)
- **Nodes Tab**: Visual node topology with GPU servers, workstations, edge devices
- **Config Tab**: Full .swarm DSL editor with:
  - Monarch tokenizer for syntax highlighting
  - Auto-completion with agent type snippets
  - Live validation (brace matching, required blocks)
  - Deploy button for cluster configuration
- Real-time health monitoring
- Connect/disconnect from StratoSwarm clusters

### Dashboard View (Executives)
- **Real-Time Metrics**: Live CPU, memory, GPU usage with 60-second history charts
- **System Info Panel**: Local GPU detection (Apple Silicon/NVIDIA)
- GPU utilization across cluster nodes
- Cost tracking and trends
- Evolution insights from StratoSwarm's self-improving engines
- Incident timeline

## Tech Stack

### Backend (Rust)
- **Tauri 2.x** - Cross-platform desktop framework
- **Tokio** - Async runtime
- **Serde** - Serialization
- **Tracing** - Structured logging
- **Sysinfo** - System/GPU detection

### Frontend (TypeScript)
- **React 19** - UI framework
- **Zustand** - State management
- **Monaco Editor** - Code editing with custom .swarm language
- **Recharts** - Data visualization
- **Tailwind CSS** - Styling
- **Vite 6** - Build tooling
- **Tauri API** - IPC and event subscriptions

## Development

### Prerequisites
- Rust 1.83+
- Node.js 20+
- Tauri CLI: `cargo install tauri-cli`

### Setup

```bash
# Install frontend dependencies
cd web && npm install && cd ..

# Run in development mode
cargo tauri dev
```

### Build

```bash
# Build for production (optimized ~5MB binary)
cargo tauri build

# Output: src-tauri/target/release/bundle/
```

### Feature Flags

The backend supports optional feature flags for embedding additional functionality:

| Feature | Description | Platform |
|---------|-------------|----------|
| `embedded-kernel` | Real notebook execution via notebook-kernel | All |
| `embedded-cluster` | Real cluster management via stratoswarm | Linux only |
| `embedded-training` | Distributed training via rtx-distributed | CUDA required |
| `embedded-storage` | File transfers via warp-core | All |
| `gpu-compiler` | GPU-accelerated Rust compilation (Synergy 1) | Metal/CUDA |
| `embedded-nebula` | RDMA + ZK proofs + mesh (Synergies 3 & 6) | Linux + InfiniBand |

```bash
# Build with all embedded features (Linux with CUDA)
cargo tauri build --features full

# Build minimal (mock data only)
cargo tauri build --features minimal
```

## Project Structure

```
11-horizon/
├── Cargo.toml                    # Workspace root
├── README.md                     # This file
├── src-tauri/                    # Rust backend
│   ├── Cargo.toml                # Dependencies + feature flags
│   ├── tauri.conf.json           # Tauri config (updater, window settings)
│   └── src/
│       ├── main.rs               # Entry point, metrics collector init
│       ├── state.rs              # AppState (bridges + notebook state)
│       ├── events.rs             # Real-time event emission
│       ├── cluster_bridge.rs     # Cluster management (mock/real)
│       ├── kernel_bridge.rs      # Notebook execution (mock/real)
│       ├── training_bridge.rs    # ML training jobs (mock/real)
│       ├── storage_bridge.rs     # File transfers (mock/real)
│       ├── gpu_compiler_bridge.rs # GPU compilation (Synergy 1)
│       ├── evolution_bridge.rs   # Evolution engines (Synergy 2)
│       ├── nebula_bridge.rs      # RDMA + ZK + mesh (Synergy 3)
│       ├── data_pipeline_bridge.rs # GPU encryption pipeline (Synergy 4)
│       ├── edge_proxy_bridge.rs  # Vortex + SLAI routing (Synergy 5)
│       ├── tensor_mesh_bridge.rs # Tensor mesh ops (Synergy 6)
│       └── commands/
│           ├── mod.rs
│           ├── cluster.rs        # Cluster IPC commands
│           ├── notebook.rs       # Notebook IPC commands
│           ├── training.rs       # Training IPC commands
│           ├── storage.rs        # Storage IPC commands
│           ├── system.rs         # GPU detection, system info
│           ├── gpu_compiler.rs   # GPU compiler commands
│           ├── evolution.rs      # Evolution engine commands
│           ├── nebula.rs         # RDMA/ZK/mesh commands
│           ├── data_pipeline.rs  # GPU pipeline commands (Synergy 4)
│           ├── edge_proxy.rs     # Edge proxy commands (Synergy 5)
│           └── tensor_mesh.rs    # Tensor mesh commands
└── web/                          # React frontend
    ├── package.json
    ├── vite.config.ts
    ├── tailwind.config.js
    └── src/
        ├── main.tsx              # Entry + Monaco language registration
        ├── App.tsx               # Router + sidebar navigation
        ├── index.css             # Global styles
        ├── languages/
        │   └── swarm.ts          # .swarm DSL Monaco configuration
        ├── hooks/
        │   └── useMetrics.ts     # Real-time event subscriptions
        ├── stores/
        │   ├── clusterStore.ts   # Cluster state management
        │   ├── notebookStore.ts  # Notebook cells/variables
        │   └── trainingStore.ts  # Training jobs
        ├── components/
        │   ├── SwarmConfigEditor.tsx   # .swarm DSL editor
        │   ├── SystemInfoPanel.tsx     # GPU/system info display
        │   ├── RealTimeMetrics.tsx     # Live metrics charts
        │   ├── GpuCompilerPanel.tsx    # GPU compiler status (Synergy 1)
        │   ├── EvolutionPanel.tsx      # Evolution metrics (Synergy 2)
        │   ├── NetworkTopology.tsx     # RDMA mesh visualization (Synergy 3)
        │   ├── DataPipeline.tsx        # GPU encryption pipeline (Synergy 4)
        │   ├── EdgeProxy.tsx           # Vortex + SLAI routing (Synergy 5)
        │   └── TensorMesh.tsx          # Tensor mesh ops (Synergy 6)
        └── views/
            ├── NotebookView/     # Monaco cells + variable panel
            ├── ClusterView/      # Nodes grid + .swarm editor tabs
            └── DashboardView/    # Charts + metrics + evolution
```

## Integration Points

| Feature | Source Project | Integration Method |
|---------|---------------|-------------------|
| Notebook execution | 08-rustybooks | `notebook-kernel` crate (embedded) |
| Cluster management | 05-stratoswarm | `cluster-mesh` crate (Linux only) |
| ML training | 07-rustytorch | `rtx-distributed` crate (CUDA) |
| Data transfer | 04-warp | `warp-core` crate |
| IPC channels | 00-hpc-channels | Unified message types |

## Implementation Status

### Phase 1: Foundation ✓
- [x] Tauri 2.x project structure
- [x] React 19 frontend with Vite 6
- [x] IPC commands scaffold (cluster, notebook, training, storage)
- [x] Three persona views (Notebook, Cluster, Dashboard)
- [x] Zustand state management

### Phase 2: Notebook Integration ✓
- [x] `embedded-kernel` feature flag
- [x] KernelBridge with conditional compilation
- [x] Real execution when feature enabled
- [ ] GPU-direct integration for tensor ops (future)

### Phase 3: Cluster Integration ✓
- [x] `embedded-cluster` feature flag (Linux only)
- [x] ClusterBridge with mock/real implementations
- [x] Node topology visualization
- [x] Connect/disconnect functionality

### Phase 4: Training Integration ✓
- [x] `embedded-training` feature flag (CUDA)
- [x] TrainingBridge with job management
- [x] Progress tracking via events

### Phase 5: Storage Integration ✓
- [x] `embedded-storage` feature flag
- [x] StorageBridge with TransferEngine
- [x] Upload/download with progress

### Phase 6: Enhanced UI ✓
- [x] .swarm DSL Monaco language (syntax highlighting, completion)
- [x] SwarmConfigEditor component with validation
- [x] GPU detection (Apple Silicon, NVIDIA)
- [x] SystemInfoPanel component
- [x] Real-time metrics via Tauri events
- [x] RealTimeMetrics component with charts

### Phase 7: Production ✓
- [x] Tauri updater configuration
- [x] Optimized release build (~5MB)
- [x] Cross-platform support (macOS, Linux, Windows)

### Phase 8: Advanced Synergies ✓
- [x] **Synergy 1**: GPU-compiled notebooks (`gpu_compiler_bridge.rs`, `GpuCompilerPanel.tsx`)
- [x] **Synergy 2**: Evolution dashboard (`evolution_bridge.rs`, `EvolutionPanel.tsx`)
- [x] **Synergy 3**: RDMA + ZK visualization (`nebula_bridge.rs`, `NetworkTopology.tsx`)
- [x] **Synergy 4**: GPU data pipeline (`data_pipeline_bridge.rs`, `DataPipeline.tsx`)
- [x] **Synergy 5**: Intelligent edge proxy (`edge_proxy_bridge.rs`, `EdgeProxy.tsx`)
- [x] **Synergy 6**: Tensor mesh operations (`tensor_mesh_bridge.rs`, `TensorMesh.tsx`)

## GPU Support

| Platform | GPU Type | Detection Method |
|----------|----------|-----------------|
| macOS (Apple Silicon) | Metal | `sysctl` + unified memory |
| Linux/Windows | NVIDIA CUDA | `nvidia-smi` query |
| Linux | AMD ROCm | Future |

## Event System

The backend emits real-time events that the frontend subscribes to:

| Channel | Payload | Frequency |
|---------|---------|-----------|
| `metrics:update` | CPU, memory, GPU usage | Every 2s |
| `training:progress` | Epoch, loss, ETA | On update |
| `cluster:status` | Connected, node count | On change |
| `system:alert` | Message, severity | On alert |

## License

MIT OR Apache-2.0
