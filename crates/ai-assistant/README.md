# AI Assistant

Natural language interface for StratoSwarm operations with learning capabilities.

## Overview

The `ai-assistant` crate provides an intelligent natural language interface that allows users to interact with StratoSwarm using plain English commands. It features intent detection, entity extraction, command generation, and a learning system that improves over time based on user interactions.

## Features

- **Natural Language Understanding**: Parse complex user requests
- **Intent Detection**: Identify user intentions from natural language
- **Entity Extraction**: Extract URLs, numbers, resources, and identifiers
- **Command Generation**: Convert natural language to executable commands
- **Learning System**: Improve accuracy through user feedback
- **Template Engine**: Generate human-friendly responses
- **Multi-Operation Support**: Deploy, scale, query, debug, optimize, and more
- **Context Awareness**: Maintain conversation context
- **Error Recovery**: Graceful handling of ambiguous requests

## Usage

### Basic Natural Language Processing

```rust
use ai_assistant::{Assistant, Query};

// Create AI assistant
let assistant = Assistant::new().await?;

// Process natural language query
let query = Query::new("Deploy my web app with 2 replicas and 4GB RAM");
let response = assistant.process(query).await?;

// Generated command
assert_eq!(response.command, "stratoswarm deploy . --replicas 2 --memory 4G");
```

### Supported Operations

The assistant understands various natural language patterns:

```rust
// Deployment
"Deploy my Python app"
"Launch the web service from github.com/user/repo"
"Start my application with high availability"

// Scaling
"Scale my service to 5 instances"
"Increase replicas to handle more traffic"
"Scale down to save resources"

// Querying
"Show me the status of my agents"
"How much memory is the web service using?"
"List all running containers"

// Debugging
"Show logs for the API service"
"Debug why my service is crashing"
"What errors occurred in the last hour?"

// Optimization
"Optimize my service for better performance"
"Reduce memory usage of the workers"
"Make my app start faster"
```

### Intent Detection

```rust
use ai_assistant::{IntentDetector, Intent};

let detector = IntentDetector::new();
let intent = detector.detect("I want to deploy my app")?;

match intent {
    Intent::Deploy { source, options } => {
        // Handle deployment
    }
    Intent::Scale { target, replicas } => {
        // Handle scaling
    }
    Intent::Query { resource, filters } => {
        // Handle query
    }
    // ... other intents
}
```

### Entity Extraction

```rust
use ai_assistant::{EntityExtractor, Entity};

let extractor = EntityExtractor::new();
let entities = extractor.extract("Deploy github.com/user/repo with 3 replicas")?;

// Extracted entities:
// - URL: github.com/user/repo
// - Number: 3
// - Keyword: replicas
```

### Learning System

```rust
use ai_assistant::{LearningSystem, Feedback};

// Record user feedback
let feedback = Feedback {
    query: "deploy my app".to_string(),
    generated_command: "stratoswarm deploy .".to_string(),
    was_correct: true,
    user_correction: None,
};

assistant.record_feedback(feedback).await?;

// Learning system improves future predictions
```

### Template-Based Responses

```rust
use ai_assistant::{ResponseGenerator, ResponseTemplate};

let generator = ResponseGenerator::new();

// Generate human-friendly response
let response = generator.generate(ResponseTemplate::DeploymentSuccess {
    app_name: "web-service",
    replicas: 3,
    url: "https://web-service.stratoswarm.io",
});

// Output: "Successfully deployed web-service with 3 replicas. 
//          Access your application at https://web-service.stratoswarm.io"
```

## Architecture

The crate is organized into key modules:

- `parser.rs`: Natural language parsing and tokenization
- `embeddings.rs`: Semantic understanding using embeddings
- `command_generator.rs`: Command synthesis from intents
- `learning.rs`: Feedback-based learning system
- `query_engine.rs`: Infrastructure query capabilities
- `templates.rs`: Response generation templates

## Integration with StratoSwarm

The assistant integrates deeply with other components:

```rust
// Zero-config integration
assistant.analyze_and_deploy("./my-project").await?;

// Runtime queries
let status = assistant.query_runtime("Show container status").await?;

// Registry operations  
assistant.process("Build and push my Docker image").await?;

// Cluster management
assistant.process("Add new GPU node to cluster").await?;
```

## Advanced Features

### Context Management

```rust
use ai_assistant::{Context, ConversationManager};

let mut conversation = ConversationManager::new();

// First query establishes context
conversation.process("Deploy my web app").await?;

// Follow-up uses context
conversation.process("Now scale it to 5 replicas").await?;
// Knows "it" refers to the previously deployed web app
```

### Ambiguity Resolution

```rust
// Ambiguous query
let response = assistant.process("Deploy my app").await?;

if response.needs_clarification() {
    // Assistant asks: "I found multiple apps. Which one would you like to deploy?"
    // 1. web-frontend (React app)
    // 2. api-backend (Python Flask)
    // 3. worker-service (Go worker)
    
    let clarified = assistant.clarify("Deploy the Python one").await?;
}
```

### Custom Intents

```rust
use ai_assistant::{IntentRegistry, CustomIntent};

// Register custom intent
let mut registry = IntentRegistry::new();
registry.register(CustomIntent {
    name: "backup",
    patterns: vec![
        r"backup (\w+)",
        r"create backup of (\w+)",
        r"save snapshot of (\w+)",
    ],
    handler: |matches| {
        // Custom backup logic
    },
});
```

## Performance

- **Intent Detection**: <10ms for typical queries
- **Command Generation**: <50ms including validation
- **Learning Update**: <5ms per feedback item
- **Memory Usage**: ~50MB base + embeddings cache

## Testing

Comprehensive test suite with 40+ tests:

```bash
# Run all tests
cargo test

# Run specific test categories
cargo test intent_detection
cargo test entity_extraction
cargo test command_generation

# Run integration tests
cargo test --test integration
```

## Configuration

Customize assistant behavior:

```toml
[ai_assistant]
# Language model settings
embedding_model = "all-MiniLM-L6-v2"
embedding_cache_size = 10000

# Intent detection
confidence_threshold = 0.85
max_intent_candidates = 3

# Learning system
learning_rate = 0.1
feedback_batch_size = 100
min_feedback_for_update = 10

# Response generation
response_style = "concise"  # or "detailed"
include_examples = true
```

## Examples

See the `examples/` directory for common use cases:

```bash
# Basic usage
cargo run --example basic_assistant

# Learning system demo
cargo run --example learning_demo

# Custom intents
cargo run --example custom_intents
```

## Error Handling

The assistant handles various error cases gracefully:

```rust
match assistant.process(query).await {
    Ok(response) => {
        // Success
    }
    Err(AssistantError::AmbiguousQuery(options)) => {
        // Multiple interpretations possible
    }
    Err(AssistantError::UnknownIntent) => {
        // Couldn't understand the request
    }
    Err(AssistantError::InvalidEntity(entity)) => {
        // Entity extraction failed
    }
    Err(e) => {
        // Other errors
    }
}
```

## Future Enhancements

- Voice input/output support
- Multi-language support (currently English only)
- Advanced context tracking across sessions
- Integration with external knowledge bases
- Proactive suggestions based on patterns

## License

MIT