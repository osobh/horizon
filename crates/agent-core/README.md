# Agent Core

Core agent infrastructure providing foundational components for autonomous agents in the StratoSwarm ecosystem.

## Overview

The `agent-core` crate is the central hub for agent functionality in StratoSwarm. It provides the fundamental building blocks for creating, managing, and coordinating autonomous agents that can learn, communicate, and evolve over time.

## Features

- **Agent Lifecycle Management**: Complete agent lifecycle from creation to termination with unique ID tracking
- **Inter-Agent Communication**: High-performance message bus for agent-to-agent communication
- **Goal Interpretation**: Natural language goal parsing and task planning capabilities
- **Multi-Type Memory System**: 
  - Working Memory: Short-term operational state
  - Episodic Memory: Event and experience storage
  - Semantic Memory: Knowledge and fact storage
  - Procedural Memory: Skill and behavior patterns
- **Resource Scheduling**: Modular scheduling architecture for efficient resource allocation
- **GPU Integration**: First-class GPU support for accelerated agent operations

## Usage

```rust
use agent_core::{Agent, AgentId, Goal, Message};
use agent_core::memory::{MemoryType, MemoryEntry};

// Create a new agent
let agent = Agent::new("assistant", AgentType::Assistant)?;

// Set a goal for the agent
let goal = Goal::from_natural_language("Optimize system performance")?;
agent.set_goal(goal).await?;

// Send a message to another agent
let message = Message::new(
    agent.id(),
    target_id,
    "Performance metrics updated".to_string(),
);
agent.send_message(message).await?;

// Store information in agent memory
let memory_entry = MemoryEntry::new(
    "system_metrics",
    serde_json::json!({"cpu": 45.2, "memory": 72.1}),
    MemoryType::Working,
);
agent.store_memory(memory_entry).await?;
```

## Architecture

The crate is organized into several key modules:

- `agent.rs`: Core agent implementation and lifecycle management
- `communication.rs`: Message passing and event bus infrastructure
- `goal.rs`: Goal representation and task decomposition
- `memory/`: Comprehensive memory system implementation
- `scheduler/`: Resource allocation and task scheduling

## Dependencies

### Internal Dependencies
- `exorust-cuda`: GPU acceleration support
- `exorust-runtime`: Container runtime integration
- `exorust-memory`: Memory management
- `exorust-storage`: Persistent storage
- `exorust-net`: Network communication
- `exorust-evolution`: Agent evolution capabilities

### Key External Dependencies
- `tokio`: Async runtime
- `serde`: Serialization
- `async-openai`: LLM integration for goal interpretation
- `dashmap`: Concurrent data structures
- `parking_lot`: Synchronization primitives

## Performance

- **Message Throughput**: 1M+ messages/second
- **Memory Operations**: <1μs latency for working memory
- **Goal Processing**: <100ms for typical natural language goals
- **Resource Scheduling**: O(log n) complexity for task allocation

## Testing

Run the test suite:

```bash
cargo test
```

Run benchmarks:

```bash
cargo bench
```

## Coverage

Current test coverage: ~85% (Good)

Areas with comprehensive testing:
- Agent lifecycle management
- Communication protocols
- Memory operations
- Basic scheduling

Areas needing additional tests:
- Complex goal decomposition
- Multi-agent coordination scenarios
- GPU acceleration paths

## Contributing

When adding new features:
1. Ensure compatibility with existing agent types
2. Maintain backward compatibility for message formats
3. Add comprehensive tests for new functionality
4. Update documentation and examples

## License

MIT