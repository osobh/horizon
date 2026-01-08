# HPC-AI CLI

The most powerful unified command-line tool for HPC-AI operators to query, deploy, administrate, and interact with all platform components.

## Features

- **Unified CLI** - Single binary (`hpc`) to manage all 15 HPC-AI projects
- **Interactive TUI** - Full terminal dashboard with GPU metrics, cluster health, and job monitoring
- **Smart Deployment** - Automatic dependency resolution with multiple deployment modes
- **Stack Management** - Full lifecycle management (init, up, down, promote)
- **Profile System** - Preset deployment profiles for common use cases
- **Interactive Picker** - Terminal-based project selection with dependency visualization

## Installation

### From Source

```bash
cd hpc-cli
cargo build --release

# Optional: Add to PATH
cp target/release/hpc ~/.local/bin/
```

### Requirements

- Rust 1.75+
- For TUI: Terminal with 256-color support

## Quick Start

```bash
# Initialize configuration
hpc init

# Show system information
hpc info

# Deploy with interactive picker
hpc deploy interactive

# Launch TUI dashboard
hpc --tui
```

---

## Commands Overview

### Global Options

```
-v, --verbose...         Increase verbosity (-v, -vv, -vvv)
-f, --format <FORMAT>    Output format [table, json, plain]
-c, --config <CONFIG>    Configuration file path
-p, --profile <PROFILE>  Active profile (dev, staging, prod)
    --tui                Launch Terminal UI mode
```

### Project Commands

Each HPC-AI project has its own subcommand:

| Command | Project | Description |
|---------|---------|-------------|
| `hpc rustg` | rustg | GPU-accelerated Rust compiler |
| `hpc channels` | hpc-channels | IPC and message passing utilities |
| `hpc parcode` | parcode | Lazy-loading object storage |
| `hpc rmpi` | rmpi | Rust MPI utilities |
| `hpc rnccl` | rnccl | GPU collective communication |
| `hpc slai` | slai | GPU detection and cluster management |
| `hpc warp` | warp | GPU-accelerated bulk data transfer |
| `hpc swarm` | stratoswarm | Unified orchestration platform |
| `hpc spark` | rustyspark | Distributed data processing |
| `hpc torch` | rustytorch | GPU-accelerated ML training |
| `hpc vortex` | vortex | Intelligent edge proxy |
| `hpc nebula` | nebula | Real-time communication |
| `hpc argus` | argus | Observability platform |

### Meta Commands

```bash
hpc info          # Show system information
hpc version       # Show version information
hpc init          # Initialize configuration directories
hpc completions   # Generate shell completions (bash, zsh, fish)
```

---

## Deployment

### Deploy Specific Projects

```bash
# Deploy single project
hpc deploy projects torch

# Deploy multiple projects (comma-separated)
hpc deploy projects torch,spark,argus

# Deploy to specific target and environment
hpc deploy projects rnccl,slai --target cluster --env staging

# Dry run to see what would be deployed
hpc deploy projects torch --dry-run
```

Dependencies are automatically resolved:
```
$ hpc deploy projects torch --dry-run

Project Deployment
========================================

Projects:    torch
Target:      local
Environment: dev

Resolving dependencies...
  + Adding dependencies: channels, rnccl, slai, rmpi

[DRY RUN] Would deploy:

  [1/5] would deploy HPC Channels
  [2/5] would deploy RNCCL
  [3/5] would deploy RMPI
  [4/5] would deploy SLAI (port 9100)
  [5/5] would deploy RustyTorch
```

### Deploy Using Profiles

Pre-configured deployment profiles for common use cases:

```bash
hpc deploy profile ml-training      # GPU ML training stack
hpc deploy profile data-processing  # Distributed data processing
hpc deploy profile full-stack       # Complete platform
hpc deploy profile monitoring       # Observability only
hpc deploy profile minimal          # Minimal stack (stratoswarm, argus)
```

Profile contents:

| Profile | Projects |
|---------|----------|
| ml-training | rnccl, slai, torch, argus |
| data-processing | rmpi, spark, warp, argus |
| full-stack | rnccl, slai, torch, spark, warp, stratoswarm, nebula, vortex, argus |
| monitoring | argus, slai |
| minimal | stratoswarm, argus |

### Interactive Deploy

Launch the interactive project picker:

```bash
hpc deploy interactive
```

Features:
- **Profile selection** - Choose from preset profiles
- **Category selection** - Pick by project category with refinement
- **Individual selection** - Select specific projects with dependency info
- Automatic dependency resolution with visualization
- Dry run confirmation before deployment

### Local Development

```bash
# Deploy default local stack (stratoswarm, argus)
hpc deploy local

# Deploy specific projects locally
hpc deploy local --projects rnccl,slai,torch
```

### Cluster Deployment

```bash
hpc deploy cluster my-cluster --namespace hpc-ai
hpc deploy cluster prod-cluster --projects torch,spark
```

---

## Stack Management

Stacks provide full lifecycle management for deployment configurations.

### Initialize a Stack

```bash
# Create from template
hpc stack init my-stack --template ml-training

# Create empty stack
hpc stack init my-stack

# Specify output directory
hpc stack init my-stack --output ./stacks/
```

### Deploy a Stack

```bash
# Deploy to local
hpc stack up --target local

# Deploy specific stack to staging
hpc stack up my-stack --target cluster --env staging

# Dry run
hpc stack up --dry-run

# Watch deployment progress
hpc stack up --watch
```

### Manage Stack Lifecycle

```bash
# Check status
hpc stack status
hpc stack status my-stack --detailed
hpc stack status --watch

# Teardown
hpc stack down my-stack
hpc stack down my-stack --volumes  # Also remove data
hpc stack down my-stack --force    # Skip confirmation

# Promote between environments
hpc stack promote my-stack --from dev --to staging
hpc stack promote my-stack --from staging --to prod --yes

# List all stacks
hpc stack list
hpc stack list --all  # Include stopped

# Validate configuration
hpc stack validate ./my-stack.toml
```

### Stack Configuration Format

Stacks are defined in TOML format (`~/.hpc/stacks/<name>.toml`):

```toml
[stack]
name = "ml-training"
description = "Full ML training stack with GPU support"

[environments.dev]
replicas = 1
resources.gpu = 0

[environments.staging]
replicas = 2
resources.gpu = 1

[environments.prod]
replicas = 4
resources.gpu = 4

[projects.rnccl]
enabled = true
config.algorithm = "ring"

[projects.torch]
enabled = true
depends_on = ["rnccl", "slai"]

[targets.local]
type = "local"
docker_compose = true

[targets.cluster]
type = "stratoswarm"
endpoint = "stratoswarm://cluster.hpc.local:8080"
```

---

## TUI Dashboard

Launch the interactive terminal dashboard:

```bash
hpc --tui
# or
hpc tui
```

### Layout

```
┌─────────────────────────────┬─────────────────────────────┐
│      DASHBOARD              │      COMMAND MENU           │
│                             │  ┌─────────────────────────┐│
│  GPU Usage                  │  │ [Commands] Deploy  Proj ││
│  ████████████░░ 78%         │  ├─────────────────────────┤│
│                             │  │ > slai detect           ││
│  Cluster Health             │  │   swarm status          ││
│  Nodes: 4 | Running: 12     │  │   argus status          ││
│                             │  │   torch models          ││
│  Jobs                       │  │   rnccl info            ││
│  ✓ training-001  running    │  │                         ││
│  ○ inference-002 pending    │  └─────────────────────────┘│
└─────────────────────────────┴─────────────────────────────┘
│                        LOGS                                │
│ [12:34:56] INFO  Deploying rnccl to local...              │
│ [12:34:58] INFO  GPU 0: NVIDIA A100 detected              │
│ [12:35:01] INFO  Stack ml-training is now running         │
└────────────────────────────────────────────────────────────┘
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Tab` / `Shift+Tab` | Switch panel focus |
| `↑` `↓` / `j` `k` | Navigate up/down |
| `←` `→` / `h` `l` | Navigate left/right (tabs) |
| `1` `2` `3` | Switch command menu tabs |
| `Enter` / `Space` | Select/activate |
| `r` | Refresh data |
| `?` | Toggle help overlay |
| `q` / `Esc` | Quit |

### Panels

**Dashboard (Left)**
- GPU usage gauges with temperature and memory
- Cluster health (nodes, running services)
- Active jobs with status and progress

**Command Menu (Right)**
- **Commands tab**: Quick access to common commands
- **Deploy tab**: Interactive project selector with checkboxes
- **Projects tab**: Project status overview

**Log Viewer (Bottom)**
- Real-time log streaming
- Color-coded log levels (ERROR, WARN, INFO, DEBUG)
- Scroll history with auto-scroll toggle
- Filter by log level

---

## Configuration

Configuration is stored in `~/.hpc/`:

```
~/.hpc/
├── config.toml      # Main configuration
├── stacks/          # Stack definitions
│   ├── ml-training.toml
│   └── my-stack.toml
└── state/           # Runtime state
```

### config.toml

```toml
# Default profile to use
default_profile = "default"

# Default environment (dev, staging, prod)
default_environment = "dev"

# TUI Settings
[tui]
tick_rate_ms = 250
max_log_entries = 1000
theme = "dark"
mouse_support = true
show_gpu_metrics = true

# Deployment Settings
[deploy]
local_defaults = ["stratoswarm", "argus"]
default_namespace = "hpc-ai"
timeout_seconds = 300

# Environment Profiles
[profiles.dev]
replicas = 1
gpu_resources = 0

[profiles.staging]
replicas = 2
gpu_resources = 1

[profiles.prod]
replicas = 4
gpu_resources = 4
```

---

## Project Dependencies

The CLI automatically resolves project dependencies during deployment:

```
hpc-channels (foundation)
    │
    ├── parcode, rmpi, rnccl
    │       │
    │       └── slai ──┬── torch
    │                  │
    │       └── rmpi ──┼── spark
    │                  │
    ├── warp ──────────┤
    │                  │
    └── stratoswarm ───┴── nebula, vortex
                              │
                           argus
```

| Project | Dependencies |
|---------|--------------|
| parcode | channels |
| rmpi | channels |
| rnccl | channels |
| slai | rnccl |
| warp | channels |
| stratoswarm | channels, warp |
| spark | rmpi, warp |
| torch | rnccl, slai, rmpi |
| vortex | slai |
| nebula | stratoswarm, rmpi |
| argus | channels |

---

## Command Examples

### RNCCL - GPU Collective Communication

```bash
# Show RNCCL configuration and version
hpc rnccl info

# Display GPU topology
hpc rnccl topology --detailed

# Run collective benchmarks
hpc rnccl bench allreduce --sizes "1K,1M,1G" --iterations 100

# Initialize a communicator
hpc rnccl init --nranks 4 --rank 0 --algo ring

# Show active communicators
hpc rnccl status
```

### Parcode - Lazy-Loading Storage

```bash
# Load an object lazily
hpc parcode load ./model.bin --page-size 4096

# Show object information
hpc parcode info ./model.bin --pages

# Manage cache
hpc parcode cache stats
hpc parcode cache list
hpc parcode cache clear --stale

# Prefetch objects
hpc parcode prefetch ./model1.bin ./model2.bin --priority high

# Verify object integrity
hpc parcode verify ./model.bin --method checksum
```

### Argus - Observability Platform

```bash
# Show Argus server status
hpc argus status --endpoint http://localhost:9090

# Run PromQL query
hpc argus query "gpu_utilization{job='rustytorch'}" --range 1h

# List scrape targets
hpc argus targets --state up

# Show active alerts
hpc argus alerts --severity critical

# Query logs
hpc argus logs "{app='stratoswarm'}" --range 1h --limit 100

# View dashboards
hpc argus dashboards
```

### SLAI - GPU Management

```bash
# Detect GPUs
hpc slai detect

# Show GPU status
hpc slai status --detailed

# Allocate GPU resources
hpc slai allocate --gpus 2 --memory 16G

# Monitor GPU usage
hpc slai monitor --interval 1s
```

### StratoSwarm - Orchestration

```bash
# Show cluster status
hpc swarm status

# Deploy workload
hpc swarm deploy ./workload.yaml

# Scale services
hpc swarm scale my-service --replicas 4

# View logs
hpc swarm logs my-service --follow
```

---

## Shell Completions

Generate shell completions for your shell:

```bash
# Bash
hpc completions bash > ~/.local/share/bash-completion/completions/hpc

# Zsh
hpc completions zsh > ~/.zfunc/_hpc

# Fish
hpc completions fish > ~/.config/fish/completions/hpc.fish

# PowerShell
hpc completions powershell > ~/_hpc.ps1
```

---

## Architecture

```
src/
├── main.rs              # CLI/TUI dispatcher
├── cli.rs               # Clap command definitions
├── commands/            # Command implementations
│   ├── mod.rs
│   ├── deploy.rs        # Deploy commands
│   ├── picker.rs        # Interactive project picker
│   ├── stack/           # Stack management
│   │   └── mod.rs
│   ├── argus.rs         # Project-specific commands
│   ├── channels.rs
│   ├── nebula.rs
│   ├── parcode.rs
│   ├── rmpi.rs
│   ├── rnccl.rs
│   ├── rustg.rs
│   ├── slai.rs
│   ├── spark.rs
│   ├── stratoswarm.rs
│   ├── torch.rs
│   ├── vortex.rs
│   └── warp.rs
├── core/                # Shared state & config
│   ├── mod.rs
│   ├── config.rs        # Configuration management
│   ├── profile.rs       # Environment profiles
│   ├── project.rs       # Project definitions
│   └── state.rs         # Application state
└── tui/                 # Terminal UI
    ├── mod.rs
    ├── app.rs           # Main TUI application
    ├── events.rs        # Event handling
    └── panels/
        ├── mod.rs
        ├── dashboard.rs     # GPU/cluster panel
        ├── command_menu.rs  # Command/deploy panel
        └── log_viewer.rs    # Log streaming panel
```

---

## Development

### Build

```bash
cargo build           # Debug build
cargo build --release # Release build
```

### Test

```bash
cargo test            # Run all tests (36 tests)
cargo test --release  # Run tests in release mode
```

### Run

```bash
cargo run -- --help
cargo run -- deploy interactive
cargo run -- --tui
```

### Feature Flags

The CLI uses feature flags to enable optional integrations:

```toml
[features]
default = []
parcode = ["dep:parcode"]
rmpi = ["dep:rmpi"]
rnccl = ["dep:rnccl"]
slai = ["dep:slai"]
warp = ["dep:warp-cli"]
stratoswarm = ["dep:stratoswarm-cli"]
spark = ["dep:rustyspark"]
torch = ["dep:rustytorch"]
vortex = ["dep:vortex-cli"]
nebula = ["dep:nebula-cli"]
argus = ["dep:argus"]
full = ["parcode", "rmpi", "rnccl", "slai", "warp", "stratoswarm", "spark", "torch", "vortex", "nebula", "argus"]
```

Build with all features:

```bash
cargo build --release --features full
```

---

## License

MIT OR Apache-2.0
