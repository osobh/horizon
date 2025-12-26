# Zero Config

Intelligent code analysis and behavioral learning system that eliminates configuration files.

## Overview

The `zero-config` crate is StratoSwarm's answer to configuration complexity. It automatically analyzes codebases, detects frameworks and dependencies, estimates resource requirements, and learns from deployment patterns to provide optimal configurations without any YAML files or manual setup.

## Features

- **Multi-Language Support**: Analyzes 8 programming languages
  - Rust, Python, JavaScript, TypeScript, Go, Java, C++, C
- **Framework Detection**: Identifies popular frameworks automatically
  - Web: Django, Flask, Express, FastAPI, Spring Boot, Gin
  - Frontend: React, Vue, Angular, Svelte
  - ML: TensorFlow, PyTorch, scikit-learn
  - Data: Pandas, NumPy, Apache Spark
- **Dependency Analysis**: Parses package files to understand requirements
- **Resource Estimation**: Predicts CPU, memory, and GPU needs
- **Behavioral Learning**: Learns from successful deployments
- **Pattern Recognition**: Identifies similar applications
- **Zero YAML**: Completely eliminates configuration files

## Usage

### Basic Analysis

```rust
use zero_config::{Analyzer, AnalysisConfig};

// Analyze a codebase
let config = AnalysisConfig::default();
let analyzer = Analyzer::new(config);

let result = analyzer.analyze_directory("/path/to/project").await?;

println!("Detected language: {:?}", result.primary_language);
println!("Frameworks: {:?}", result.frameworks);
println!("Estimated memory: {}MB", result.estimated_memory_mb);
```

### Automatic Configuration Generation

```rust
use zero_config::{ConfigGenerator, DeploymentConfig};

// Generate deployment configuration
let generator = ConfigGenerator::new();
let deployment_config = generator.generate(&analysis_result)?;

// Configuration includes:
// - Optimal resource allocation
// - Agent personality assignment
// - Scaling policies
// - Network requirements
// - Storage configuration
```

### Learning from Deployments

```rust
use zero_config::{LearningSystem, DeploymentOutcome};

// Create learning system
let mut learning = LearningSystem::new();

// Record deployment outcome
let outcome = DeploymentOutcome {
    success: true,
    performance_metrics: metrics,
    resource_usage: usage,
    configuration: config,
};

learning.record_outcome(&analysis_result, outcome).await?;

// Future deployments will benefit from this knowledge
```

## Language Analysis

### Supported Languages

Each language has specialized analysis:

```rust
// Python analysis
- Framework detection (Django, Flask, FastAPI)
- requirements.txt parsing
- Virtual environment detection
- ML library identification

// JavaScript/TypeScript analysis  
- package.json parsing
- Framework detection (React, Vue, Express)
- Build tool identification
- Node.js version requirements

// Rust analysis
- Cargo.toml parsing
- Feature flag detection
- Binary vs library detection
- WASM support identification

// Go analysis
- go.mod parsing
- Import analysis
- Binary detection
- Container awareness
```

### Pattern Recognition

```rust
use zero_config::{PatternEngine, ApplicationPattern};

// Find similar applications
let pattern_engine = PatternEngine::new();
let similar_apps = pattern_engine.find_similar(&analysis_result)?;

// Transfer knowledge from similar deployments
for app in similar_apps {
    let insights = pattern_engine.extract_insights(&app)?;
    config.apply_insights(insights);
}
```

## Resource Estimation

The system uses multiple signals to estimate resources:

1. **Code Complexity**: Cyclomatic complexity, lines of code
2. **Dependencies**: Heavy frameworks increase requirements
3. **Data Processing**: Presence of data libraries indicates memory needs
4. **Concurrency**: Thread pools and async operations affect CPU needs
5. **ML Workloads**: Deep learning frameworks trigger GPU allocation

```rust
// Resource estimation example
let resources = analyzer.estimate_resources(&analysis)?;

assert_eq!(resources.cpu_cores, 2.0);        // 2 CPU cores
assert_eq!(resources.memory_mb, 4096);       // 4GB RAM
assert_eq!(resources.gpu_count, 1);          // 1 GPU for ML
assert_eq!(resources.storage_gb, 10);        // 10GB storage
```

## Configuration Output

Generated configurations are comprehensive:

```rust
let config = generator.generate(&analysis)?;

// Agent configuration
config.agent_type              // e.g., AgentType::WebService
config.personality             // e.g., Personality::Balanced
config.resource_limits         // CPU, memory, GPU quotas

// Scaling configuration
config.scaling.min_instances   // Minimum instances
config.scaling.max_instances   // Maximum instances
config.scaling.target_cpu      // CPU threshold for scaling

// Network configuration
config.network.exposed_ports   // e.g., [80, 443]
config.network.protocols       // e.g., ["http", "https"]

// Storage configuration
config.storage.persistent_paths // Paths needing persistence
config.storage.cache_size       // Estimated cache requirements
```

## Learning System

The behavioral learning system continuously improves:

```rust
// Learning from patterns
- Successful Django apps typically need 2GB RAM
- React SPAs benefit from aggressive caching
- ML training workloads need checkpoint storage
- Microservices need service discovery

// Pattern database
patterns.save("./patterns.db")?;
patterns.load("./patterns.db")?;
```

## Testing

Comprehensive test suite with 150+ tests:

```bash
# Run all tests
cargo test

# Run specific test categories
cargo test analysis
cargo test pattern_recognition
cargo test resource_estimation

# Run with verbose output
cargo test -- --nocapture
```

## Coverage

Current test coverage: 80.9% (Below 90% target)

Well-tested areas:
- Language detection (100%)
- Framework detection (85%)
- Basic analysis (90%)
- Error handling (100%)

Areas needing improvement:
- Complex pattern recognition (73.4%)
- Advanced configuration generation (73.4%)
- Edge cases in resource estimation

## Performance

- **Analysis Speed**: ~1000 files/second
- **Pattern Matching**: O(log n) with indexed patterns  
- **Memory Usage**: <100MB for typical projects
- **Learning Update**: <10ms per deployment

## Configuration

Fine-tune analysis behavior:

```rust
let config = AnalysisConfig::builder()
    .max_file_size(10_485_760)       // 10MB max file size
    .exclude_patterns(vec![".git", "node_modules"])
    .parallel_analysis(true)
    .language_confidence_threshold(0.8)
    .framework_detection_depth(3)
    .build();
```

## CLI Integration

Used by stratoswarm-cli for zero-config deployments:

```bash
# Analyze and deploy without configuration
stratoswarm deploy /path/to/project

# Show analysis results
stratoswarm analyze /path/to/project
```

## Future Enhancements

- Support for more languages (Ruby, PHP, Kotlin)
- Advanced ML workload detection
- Database requirement inference
- Security configuration generation
- Multi-service dependency detection

## License

MIT