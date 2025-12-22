# Contributing to StratoSwarm

Welcome to StratoSwarm! This document outlines the development standards and contribution guidelines that maintain our high-quality, production-ready codebase.

## 🎯 Development Philosophy

StratoSwarm follows **strict Test-Driven Development (TDD)** with a focus on:

- **Production Quality**: Every line of code must be production-ready
- **Performance First**: GPU-native architecture with microsecond-level optimizations  
- **Zero Technical Debt**: No placeholder code, todos, or unfinished implementations
- **Maintainable Scale**: Clean, modular code that scales to millions of nodes

## 📋 Quality Standards

### **Code Quality Requirements**

All contributions must meet these **mandatory** standards:

| **Metric** | **Standard** | **Status** |
|------------|--------------|------------|
| **Test Coverage** | 100% for new code | ✅ Required |
| **unwrap() Calls** | 0 in production code | ✅ Enforced |
| **todo!() Macros** | 0 in production code | ✅ Enforced |
| **File Size** | <850 lines per file | ✅ Enforced |
| **Documentation** | All public APIs | ✅ Required |
| **Performance Tests** | All critical paths | ✅ Required |

### **Architecture Principles**

1. **GPU-Native Design**: Leverage CUDA, cudarc, Metal, and GPU memory pools
2. **Async-First**: All I/O operations must be async
3. **Zero-Copy**: Minimize memory allocations and copies
4. **Fault Tolerance**: Handle Byzantine failures and network partitions
5. **Observability**: Use `hpc-tracing` for OpenTelemetry and Prometheus metrics
6. **Security-First**: Use `hpc-auth` for mTLS, `hpc-crypto` for signatures
7. **Enterprise Patterns**: Follow HPC foundation patterns for new infrastructure

## 🛠️ Development Workflow

### **1. Setting Up Development Environment**

```bash
# Clone the repository
git clone https://github.com/StratoSwarm/stratoswarm.git
cd stratoswarm

# Install dependencies
cargo build --release --all-features

# Run the quality dashboard
python3 scripts/quality_dashboard.py

# Setup pre-commit hooks
cp pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### **2. TDD Development Cycle**

**Follow the strict RED-GREEN-REFACTOR cycle:**

#### **🔴 RED Phase: Write Failing Tests**
```bash
# Create failing tests first
cargo test specific_feature_test
# Should fail - this proves the test works

# Example test structure:
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_gpu_memory_allocation() -> Result<()> {
        let pool = GpuMemoryPool::new(1024 * 1024)?;
        let allocation = pool.allocate(512).await?;
        assert!(allocation.is_valid());
        Ok(())
    }
}
```

#### **🟢 GREEN Phase: Implement Minimum Code**
```bash
# Write minimal implementation to pass tests
cargo test specific_feature_test
# Should pass with real implementation (no todos or mocks)
```

#### **🔄 REFACTOR Phase: Optimize and Clean**
```bash
# Optimize performance and clean up code
cargo bench feature_benchmark
cargo clippy --all-targets -- -D warnings
```

### **3. Code Standards**

#### **Error Handling**
- **NO** `unwrap()` or `expect()` in production code
- Use `?` operator for error propagation
- Implement comprehensive error types with `thiserror`

```rust
// ❌ BAD
let result = risky_operation().unwrap();

// ✅ GOOD  
let result = risky_operation()
    .map_err(|e| ProcessingError::OperationFailed { 
        operation: "risky_operation",
        source: e 
    })?;
```

#### **Performance Requirements**
- **Memory operations**: >10,000 ops/second
- **GPU transfers**: <5μs P2P latency  
- **Consensus**: <100μs for routing decisions
- **Scaling**: Linear to 1M+ nodes

#### **GPU Programming Standards**
```rust
// Use cudarc for GPU operations
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};

// Prefer GPU memory pools
let device = CudaDevice::new(0)?;
let pool = GpuMemoryPool::new(&device, 1_000_000)?;

// Always handle GPU errors properly
let result = kernel.launch(config, (&input, &mut output))
    .map_err(|e| GpuError::KernelLaunchFailed { source: e })?;
```

#### **HPC Foundation Patterns**

When building infrastructure components, use the HPC foundation crates:

```rust
// Security: Use hpc-auth for mTLS
use hpc_auth::cert::{generate_ca_cert, generate_signed_cert, ServiceIdentity};
use hpc_auth::server::create_server_config_with_client_auth;

let ca = generate_ca_cert("Service CA")?;
let identity = ServiceIdentity::new("my-service");
let cert = generate_signed_cert(&identity, &ca)?;

// RPC: Use hpc-rpc for gRPC with mTLS
use hpc_rpc::grpc::{GrpcServerBuilder, GrpcClientBuilder};

let server = GrpcServerBuilder::new(addr)
    .with_tls(cert)?
    .with_client_auth(ca)?
    .add_service(MyService::new())
    .build()?;

// Observability: Use hpc-tracing
use hpc_tracing::{TracingConfig, init, init_metrics};

init_metrics("0.0.0.0:9090".parse()?)?;
let config = TracingConfig {
    service_name: "my-service".to_string(),
    log_level: "info".to_string(),
    otlp_endpoint: Some("http://collector:4317".to_string()),
};
let _guard = init(config)?;

// Errors: Use hpc-error for unified error handling
use hpc_error::{HpcError, Result};

fn operation() -> Result<()> {
    Err(HpcError::Database("connection failed".to_string()))
}

// Configuration: Use hpc-config for layered config
use hpc_config::ConfigBuilder;

let config: AppConfig = ConfigBuilder::new()
    .add_file("config.yaml")?
    .add_env_prefix("APP")
    .build()?;

// Policy: Use hpc-policy for authorization
use hpc_policy::{parse_policy, evaluate};

let allowed = evaluate(&policy, &context)?;
```

### **4. File Organization**

#### **Maximum File Size: 850 Lines**
When files exceed 850 lines, split them using this pattern:

```rust
// main_module/mod.rs - Public interface and orchestration
pub use config::*;
pub use types::*;
pub use implementation::*;

mod config;     // Configuration structs
mod types;      // Type definitions  
mod implementation; // Core logic
```

#### **Module Structure**
```
crate_name/
├── src/
│   ├── lib.rs              # Public API exports
│   ├── error.rs            # Error types
│   ├── config.rs           # Configuration
│   ├── types.rs            # Core types
│   ├── implementation.rs   # Main logic
│   └── tests.rs           # Integration tests
├── benches/               # Performance benchmarks
├── tests/                 # Integration tests
└── examples/              # Usage examples
```

## 🧪 Testing Guidelines

### **Test Categories**

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test module interactions
3. **Performance Tests**: Benchmark critical paths  
4. **End-to-End Tests**: Test complete workflows
5. **GPU Tests**: Validate CUDA kernel operations

### **Test Implementation Standards**

```rust
// Comprehensive test example
#[tokio::test]
async fn test_distributed_consensus_with_failures() -> Result<()> {
    // Setup realistic test environment
    let cluster = TestCluster::new(21).await?; // Byzantine fault tolerance
    let mut nodes = cluster.spawn_nodes().await?;
    
    // Inject realistic failures
    nodes[3].simulate_network_partition().await?;
    nodes[7].simulate_byzantine_behavior().await?;
    
    // Test the actual functionality
    let proposal = ConsensusProposal::new("test_transaction");
    let result = cluster.reach_consensus(proposal).await?;
    
    // Validate performance requirements
    assert!(result.latency < Duration::from_micros(100));
    assert!(result.agreement_percentage > 0.66);
    
    Ok(())
}
```

### **Benchmark Requirements**

```rust
// Performance benchmark template
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_gpu_memory_allocation(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let pool = rt.block_on(async { 
        GpuMemoryPool::new(1024 * 1024 * 1024).await.unwrap() 
    });
    
    c.bench_function("gpu_memory_allocation", |b| {
        b.to_async(&rt).iter(|| async {
            let allocation = pool.allocate(black_box(1024)).await.unwrap();
            pool.deallocate(allocation).await.unwrap();
        });
    });
}

criterion_group!(benches, benchmark_gpu_memory_allocation);
criterion_main!(benches);
```

## 🚀 Submission Process

### **Pre-Commit Validation**

Before submitting, ensure all checks pass:

```bash
# Automatic validation via pre-commit hook
git commit -m "feat: add gpu memory pooling"

# Manual validation  
cargo fmt --all                                    # Format code
cargo clippy --all-targets -- -D warnings          # Lint checks
cargo test --all-features                          # Run all tests
cargo bench --features gpu                         # Performance validation
python3 scripts/quality_dashboard.py               # Quality metrics
```

### **Pull Request Requirements**

Every PR must include:

1. **TDD Evidence**:
   - Screenshots of RED phase (failing tests)
   - Screenshots of GREEN phase (passing tests) 
   - Performance benchmark results

2. **Quality Metrics**:
   - Quality dashboard output showing improvements
   - Test coverage report
   - Performance regression analysis

3. **Documentation**:
   - Updated public API documentation
   - Usage examples for new features
   - Architecture decision records (ADRs) for significant changes

### **PR Description Template**

```markdown
## Summary
Brief description of changes and motivation.

## TDD Cycle Evidence
- [ ] RED: Failing tests implemented first
- [ ] GREEN: Implementation passes all tests  
- [ ] REFACTOR: Code optimized and cleaned

## Quality Checklist
- [ ] Zero unwrap() calls added
- [ ] Zero todo!() macros  
- [ ] All files <850 lines
- [ ] Performance tests included
- [ ] Documentation updated

## Performance Impact
- Benchmark results: [attach results]
- Memory usage: [before/after]
- Latency impact: [measurements]

## Testing
- [ ] Unit tests: XX% coverage
- [ ] Integration tests pass
- [ ] GPU tests validate CUDA kernels
- [ ] End-to-end scenarios tested
```

## 🔧 Tools and Automation

### **Quality Automation**

```bash
# Run quality dashboard for current state
python3 scripts/quality_dashboard.py

# Auto-fix certain quality issues
python3 scripts/fix_unwraps_safe.py --auto-fix

# Generate comprehensive test coverage
cargo tarpaulin --all-features --out html
```

### **Performance Profiling**

```bash
# GPU profiling with nvprof
nvprof --print-gpu-trace target/release/stratoswarm-gpu-test

# CPU profiling with perf
perf record -g target/release/stratoswarm-benchmark
perf report

# Memory profiling with valgrind
valgrind --tool=memcheck target/release/stratoswarm-memory-test
```

### **Continuous Integration**

Our CI pipeline enforces:

- All quality standards automatically
- Performance regression detection  
- Cross-platform compatibility testing
- GPU compatibility validation across RTX 4090, RTX 5090, H100

## 📚 Resources

### **Architecture Documentation**
- [System Overview](docs/ECOSYSTEM_OVERVIEW.md)
- [GPU Architecture](docs/architecture/gpu-native-design.md)
- [HPC Foundation](docs/HPC_FOUNDATION.md) - Enterprise infrastructure reference
- [Performance Benchmarks](benchmarks/README.md)

### **Development Guides**
- [Getting Started](docs/guides/getting-started.md)
- [Performance Tuning](docs/guides/performance-tuning.md)
- [GPU Programming Best Practices](docs/guides/gpu-programming.md)
- [Swarmlet Deployment](docs/swarmlet.md)

### **Examples and Templates**
- [Component Implementation Template](examples/component-template/)
- [TDD Workflow Examples](examples/tdd-examples/)
- [Benchmark Suite](benchmarks/component_benchmarks.md)

## ❓ Getting Help

- **Issues**: Create GitHub issues for bugs or feature requests
- **Discussions**: Use GitHub Discussions for architecture questions  
- **Performance**: Check existing benchmarks and profiling guides
- **GPU Questions**: Reference CUDA documentation and cudarc examples

## 🏆 Recognition

Contributors who consistently meet our quality standards and help improve system performance are recognized in:

- [Contributors Hall of Fame](CONTRIBUTORS.md) 
- Release notes and changelogs
- Performance achievement leaderboards

---

**Remember**: StratoSwarm is built for production use with millions of nodes and microsecond latencies. Every contribution should maintain our standards of performance, reliability, and maintainability.

**Quality is not an accident—it's the result of disciplined engineering practices.** 🚀