# StratoSwarm DSL Parser

This crate provides parsing and compilation for the `.swarm` domain-specific language, enabling declarative infrastructure definitions without YAML.

## Features

- **Type-safe DSL**: Rust-like syntax with full type checking
- **Zero-config philosophy**: No YAML required
- **Agent-based model**: Define infrastructure as autonomous agents
- **Evolution support**: Built-in support for agent evolution
- **Template system**: Reusable infrastructure patterns
- **Comprehensive validation**: Catch errors at parse time

## Usage

```rust
use stratoswarm_dsl::{parse, compile, parse_and_compile};

// Parse a .swarm file
let input = r#"
    swarm myapp {
        agents {
            frontend: WebAgent {
                replicas: 3..10,
                tier_preference: ["GPU", "CPU"],
                
                resources {
                    cpu: 2.0,
                    memory: "4Gi",
                    gpu: optional(0.5),
                }
                
                network {
                    expose: 80,
                    mesh: true,
                }
            }
        }
    }
"#;

// Parse and compile in one step
let agent_specs = parse_and_compile(input)?;

// Or parse and compile separately
let ast = parse(input)?;
let agent_specs = compile(ast)?;
```

## DSL Syntax

### Basic Structure

```swarm
swarm application_name {
    agents {
        // Agent definitions
    }
    
    connections {
        // Inter-agent connections
    }
    
    policies {
        // Deployment policies
    }
}
```

### Agent Definition

```swarm
agent_name: AgentType {
    // Scaling
    replicas: 3,              // Fixed count
    replicas: 3..10,          // Auto-scaling range
    
    // Resources
    resources {
        cpu: 2.0,
        memory: "4Gi",
        gpu: 1.0,             // Required GPU
        gpu: optional(0.5),   // Optional GPU
    }
    
    // Networking
    network {
        expose: 8080,
        mesh: true,
        load_balance: "least_connections",
    }
    
    // Evolution
    evolution {
        strategy: "conservative",
        fitness: "latency < 100ms",
    }
    
    // Personality
    personality {
        risk_tolerance: 0.7,
        cooperation: 0.9,
        exploration: 0.5,
    }
    
    // Tier preferences
    tier_preference: ["GPU", "CPU", "NVMe"],
}
```

### Connections

```swarm
connections {
    frontend -> backend: {
        protocol: "grpc",
        retry: exponential_backoff(1, 30),
        circuit_breaker: true,
    }
}
```

### Templates

```swarm
template Microservice(name: String, port: Int) {
    swarm ${name}_service {
        agents {
            api: WebAgent {
                network {
                    expose: ${port},
                }
            }
        }
    }
}
```

## Supported Agent Types

- `WebAgent`: HTTP/web services
- `ComputeAgent`: CPU/GPU compute workloads
- `StorageAgent`: Persistent storage
- `NetworkAgent`: Network services
- `GPUAgent`: GPU-specific workloads

## Value Types

- **String**: `"hello"`
- **Number**: `42`, `3.14`
- **Boolean**: `true`, `false`
- **Range**: `3..10`
- **Array**: `[1, 2, 3]`, `["a", "b"]`
- **Object**: `{ key: value }`
- **Function calls**: `optional(1.0)`, `exponential_backoff(1, 30)`
- **Tier types**: `GPU`, `CPU`, `NVMe`, `Memory`

## Memory Units

- `Gi`: Gibibytes (1024³ bytes)
- `Mi`: Mebibytes (1024² bytes)
- `Ki`: Kibibytes (1024 bytes)
- No unit: bytes

## Error Handling

The parser provides detailed error messages with line and column information:

```rust
match parse(input) {
    Ok(ast) => // Process AST
    Err(DslError::ParseError { line, column, message }) => {
        eprintln!("Parse error at {}:{} - {}", line, column, message);
    }
    Err(DslError::ValidationError(msg)) => {
        eprintln!("Validation error: {}", msg);
    }
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
```

## License

MIT