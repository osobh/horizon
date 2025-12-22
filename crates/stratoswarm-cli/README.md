# StratoSwarm CLI

Unified command-line interface for StratoSwarm with natural language support and registry integration.

## Overview

The `stratoswarm-cli` crate provides the primary command-line interface for interacting with StratoSwarm. It combines traditional CLI commands with natural language processing, zero-configuration deployment, and comprehensive registry management. The CLI is designed to be intuitive for beginners while providing powerful features for advanced users.

## Features

- **Natural Language Commands**: Use plain English instead of complex flags
- **Zero-Config Deployment**: Deploy applications without configuration files
- **Registry Integration**: Build, push, pull, and manage container images
- **Interactive Shell**: Built-in REPL for exploration
- **Real-time Monitoring**: Watch deployments and system status
- **Multi-Format Output**: JSON, YAML, or human-readable output
- **Command Completion**: Shell completions for bash, zsh, and fish
- **Offline Mode**: Work without network connectivity

## Installation

```bash
# Install from source
cargo install --path crates/stratoswarm-cli

# Or download pre-built binary
curl -sSL https://stratoswarm.io/install.sh | bash
```

## Quick Start

```bash
# Deploy an application
stratoswarm deploy ./my-app

# Deploy from registry
stratoswarm deploy --image myapp:latest

# Use natural language
stratoswarm ai "deploy my Python web app with 2 replicas"

# Check status
stratoswarm status

# View logs
stratoswarm logs my-app

# Scale application
stratoswarm scale my-app --replicas 5
```

## Command Reference

### Core Commands

#### Deploy
Deploy applications with zero configuration:

```bash
# Deploy from source
stratoswarm deploy [PATH] [OPTIONS]

# Options:
#   --name <NAME>          Override auto-detected name
#   --replicas <N>         Number of replicas (default: auto)
#   --memory <SIZE>        Memory limit (e.g., 2G, 512M)
#   --cpu <CORES>          CPU cores (e.g., 0.5, 2)
#   --gpu <COUNT>          GPU count
#   --image <IMAGE>        Deploy from registry image
#   --stream-image         Stream image during deployment
#   --env <KEY=VALUE>      Environment variables
#   --port <PORT>          Expose ports
#   --personality <TYPE>   Agent personality (default: auto)

# Examples:
stratoswarm deploy ./webapp --replicas 3 --memory 4G
stratoswarm deploy --image nginx:latest --port 80
stratoswarm deploy . --personality aggressive --gpu 1
```

#### Status
View deployment and system status:

```bash
# All deployments
stratoswarm status

# Specific deployment
stratoswarm status my-app

# Detailed output
stratoswarm status --detailed

# JSON output
stratoswarm status --output json

# Watch mode
stratoswarm status --watch
```

#### Logs
Stream or view application logs:

```bash
# Stream logs
stratoswarm logs my-app

# Last N lines
stratoswarm logs my-app --tail 100

# Filter by time
stratoswarm logs my-app --since 1h

# Filter by level
stratoswarm logs my-app --level error

# Multiple apps
stratoswarm logs app1 app2 app3
```

#### Scale
Adjust application replicas:

```bash
# Scale to specific count
stratoswarm scale my-app --replicas 10

# Auto-scale based on load
stratoswarm scale my-app --auto --min 2 --max 20

# Scale to zero
stratoswarm scale my-app --replicas 0
```

### Registry Commands

#### Build
Build container images:

```bash
# Build from Dockerfile
stratoswarm build [PATH] --tag IMAGE:TAG

# Build with build args
stratoswarm build . --tag myapp:v1 --build-arg VERSION=1.0

# Multi-stage build
stratoswarm build . --tag myapp:latest --target production
```

#### Push
Push images to registry:

```bash
# Push to default registry
stratoswarm push IMAGE:TAG

# Push to specific registry
stratoswarm push registry.example.com/IMAGE:TAG

# Push with progress bar
stratoswarm push myapp:latest --progress
```

#### Pull
Pull images from registry:

```bash
# Pull latest
stratoswarm pull IMAGE

# Pull specific tag
stratoswarm pull IMAGE:TAG

# Pull to local cache
stratoswarm pull nginx:alpine --cache-only
```

#### List
List images in registry:

```bash
# List all images
stratoswarm list

# List with tags
stratoswarm list --tags

# Filter by name
stratoswarm list --filter "web*"

# Sort by date
stratoswarm list --sort date
```

### AI Assistant

Use natural language for any operation:

```bash
# Deploy
stratoswarm ai "deploy my Node.js app with 4GB RAM"

# Query
stratoswarm ai "show me CPU usage for the API service"

# Debug
stratoswarm ai "why is my service crashing?"

# Optimize
stratoswarm ai "optimize my app for better performance"
```

### Advanced Commands

#### Evolve
Trigger agent evolution:

```bash
# Start evolution
stratoswarm evolve my-app

# Evolution with constraints
stratoswarm evolve my-app --optimize latency --constraint "memory < 2G"

# View evolution history
stratoswarm evolve my-app --history
```

#### Debug
Advanced debugging tools:

```bash
# Time-travel debugging
stratoswarm debug my-app --time-travel

# Attach to container
stratoswarm debug my-app --attach

# Profile performance
stratoswarm debug my-app --profile cpu
```

#### Config
Manage CLI configuration:

```bash
# View config
stratoswarm config view

# Set registry
stratoswarm config set registry https://registry.example.com

# Set default values
stratoswarm config set defaults.memory 2G
stratoswarm config set defaults.personality balanced
```

## Interactive Shell

Start an interactive session:

```bash
stratoswarm shell

# In shell:
> deploy ./my-app
> status
> logs my-app
> help
> exit
```

Shell features:
- Command history
- Tab completion
- Syntax highlighting
- Context awareness

## Output Formats

Control output format:

```bash
# Human readable (default)
stratoswarm status

# JSON
stratoswarm status -o json

# YAML
stratoswarm status -o yaml

# Custom format
stratoswarm status -o "{{.Name}}: {{.Status}}"

# No headers
stratoswarm list --no-headers

# Quiet mode (IDs only)
stratoswarm list -q
```

## Configuration File

CLI configuration at `~/.stratoswarm/config.toml`:

```toml
[defaults]
registry = "registry.stratoswarm.io"
memory = "1G"
cpu = 1.0
personality = "balanced"
output_format = "table"

[ai]
enabled = true
model = "gpt-4"
temperature = 0.7

[cluster]
endpoint = "https://cluster.stratoswarm.io"
timeout = "30s"

[auth]
token_path = "~/.stratoswarm/token"
```

## Environment Variables

Override configuration with environment variables:

```bash
# Registry endpoint
export STRATOSWARM_REGISTRY=https://custom.registry.io

# Cluster endpoint
export STRATOSWARM_CLUSTER=https://cluster.local

# Default personality
export STRATOSWARM_PERSONALITY=explorer

# Disable color output
export NO_COLOR=1
```

## Shell Completions

Generate shell completions:

```bash
# Bash
stratoswarm completions bash > ~/.bash_completion.d/stratoswarm

# Zsh
stratoswarm completions zsh > ~/.zsh/completions/_stratoswarm

# Fish
stratoswarm completions fish > ~/.config/fish/completions/stratoswarm.fish
```

## Error Handling

The CLI provides helpful error messages:

```bash
$ stratoswarm deploy ./nonexistent
Error: Directory not found: ./nonexistent

Suggestion: Check if the path exists or try:
  stratoswarm deploy .  # Deploy current directory
  stratoswarm deploy --image nginx:latest  # Deploy from registry

$ stratoswarm scale my-app --replicas -1
Error: Invalid replica count: -1

Replica count must be >= 0
```

## Testing

The CLI has comprehensive test coverage:

```bash
# Run all tests
cargo test

# Unit tests
cargo test --lib

# Integration tests
cargo test --test '*'

# E2E tests
cargo test --features e2e
```

Test statistics:
- **Total Tests**: 95 passing
- **Unit Tests**: 63
- **Integration Tests**: 19
- **E2E Tests**: 13
- **Coverage**: 75% for registry module

## Plugin System

Extend the CLI with plugins:

```bash
# Install plugin
stratoswarm plugin install stratoswarm-kubectl-compat

# List plugins
stratoswarm plugin list

# Run plugin command
stratoswarm kubectl get pods
```

## Performance

CLI performance characteristics:
- **Startup Time**: <50ms
- **Command Execution**: <100ms (excluding network)
- **Completion Generation**: <10ms
- **Memory Usage**: <20MB

## Troubleshooting

Common issues and solutions:

```bash
# Debug mode
RUST_LOG=debug stratoswarm deploy .

# Verbose output
stratoswarm deploy . -vvv

# Dry run
stratoswarm deploy . --dry-run

# Check connectivity
stratoswarm cluster ping
```

## Contributing

See CONTRIBUTING.md for development setup:

```bash
# Development build
cargo build

# Run development version
cargo run -- deploy .

# Run with hot reload
cargo watch -x run
```

## License

MIT