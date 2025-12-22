# StratoSwarm Performance Benchmarks

This directory contains comprehensive performance benchmarks for validating StratoSwarm's performance claims and identifying optimization opportunities.

## Quick Start

```bash
# Run all single-node benchmarks
./scripts/run_all_benchmarks.sh --single-node

# Run GPU consensus benchmark
./scripts/gpu_consensus_bench.sh

# Run concurrent stress tests
./scripts/concurrent_stress.sh --scenario production

# Run swarmlet scaling with RPi cluster
./scripts/swarmlet_scaling.sh --nodes 5
```

## Benchmark Categories

### 1. Single-Node Benchmarks (`single_node_benchmarks.md`)
Tests that can be run on a single system with GPU:
- **GPU Consensus**: Validate <100μs consensus latency
- **GPU Synthesis**: Validate 2.6B ops/sec throughput
- **Memory Tiers**: Test <50ms migration performance
- **Container Spawn**: Measure <500μs spawn times
- **Kernel Modules**: Overhead measurement

### 2. Swarmlet Scaling (`swarmlet_scaling_benchmarks.md`)
Tests using 5 Raspberry Pi nodes:
- **Join Performance**: Docker swarmlet join timing
- **Heterogeneous Coordination**: GPU + RPi workload distribution
- **Edge Device Stress**: Limited resources and network
- **Failure Recovery**: Node disconnect/reconnect scenarios

### 3. Concurrent Stress Tests (`concurrent_stress_tests.md`)
Multi-component stress testing:
- **Production Simulation**: Realistic mixed workloads
- **Maximum Throughput**: Push all components to limits
- **Resource Contention**: Identify bottlenecks under load
- **Long-Running Stability**: 24-72 hour endurance tests

### 4. Component Benchmarks (`component_benchmarks.md`)
Individual crate performance:
- **Knowledge Graph**: Query performance, GPU acceleration
- **AI Assistant**: NL parsing, command generation
- **Zero-Config**: Code analysis speed by language
- **Streaming Pipeline**: Throughput and GPU efficiency
- **Evolution Engines**: ADAS, DGM, SwarmAgentic performance

## Performance Targets

| Component | Metric | Target | Current | Status |
|-----------|--------|--------|---------|--------|
| GPU Consensus | Latency (p95) | <100μs | ~49μs | ✅ Needs validation |
| GPU Synthesis | Throughput | 2.6B ops/sec | 2.6B | ✅ Needs validation |
| Memory Migration | Latency | <1ms | <50ms | ⚠️ Optimization needed |
| Container Spawn | Time | <500μs | TBD | 📊 Testing required |
| GPU Utilization | Sustained | >90% | ~70% | ⚠️ Optimization needed |
| Agent Scale | Per node | >10K | TBD | 📊 Testing required |

## Running Benchmarks

### Prerequisites
- NVIDIA GPU with CUDA 12.x
- Linux kernel 6.14+ with StratoSwarm modules loaded
- Docker for swarmlet testing
- 5 Raspberry Pi nodes (optional, for scaling tests)
- Performance monitoring tools (perf, nvidia-smi, prometheus)

### Environment Setup
```bash
# Load kernel modules
sudo modprobe swarm_guard
sudo modprobe tier_watch
sudo modprobe gpu_dma_lock

# Set up monitoring
./scripts/setup_monitoring.sh

# Verify GPU
nvidia-smi
```

### Benchmark Execution

#### Single Component Test
```bash
# Test specific component
./scripts/gpu_consensus_bench.sh --iterations 1000 --duration 60s

# Results saved to results/gpu_consensus_$(date).json
```

#### Full Benchmark Suite
```bash
# Run complete benchmark suite (4-6 hours)
./scripts/run_all_benchmarks.sh --full

# Run quick validation (30 minutes)
./scripts/run_all_benchmarks.sh --quick
```

#### Stress Testing
```bash
# Production workload simulation
./scripts/concurrent_stress.sh --scenario production --duration 1h

# Maximum throughput test
./scripts/concurrent_stress.sh --scenario max-throughput --duration 30m

# Long-running stability
./scripts/concurrent_stress.sh --scenario stability --duration 24h
```

## Results Analysis

### Automated Reports
```bash
# Generate performance report
./scripts/generate_report.sh --input results/ --output performance_report.html

# Compare with previous results
./scripts/compare_results.sh --baseline results/baseline.json --current results/latest.json
```

### Key Metrics to Track
1. **Latency Percentiles**: p50, p95, p99 for all operations
2. **Throughput**: Operations per second, sustained rates
3. **Resource Utilization**: GPU, CPU, memory efficiency
4. **Scaling Efficiency**: Performance vs node count
5. **Stability**: Performance variance over time

## Benchmark Development

### Adding New Benchmarks
1. Create benchmark script in `scripts/`
2. Follow naming convention: `component_operation_bench.sh`
3. Output results in JSON format to `results/`
4. Update documentation with expected metrics

### Benchmark Standards
- Minimum 1000 iterations for statistical validity
- Warm-up period of 10 seconds
- Remove outliers >3 standard deviations
- Report percentiles, not just averages
- Include system configuration in results

## Troubleshooting

### Common Issues
- **GPU Not Found**: Ensure CUDA drivers are installed
- **Permission Denied**: Kernel modules require root access
- **Out of Memory**: Reduce concurrent test load
- **Network Issues**: Check firewall for swarmlet tests

### Debug Mode
```bash
# Enable verbose logging
export STRATOSWARM_BENCH_DEBUG=1

# Run with profiling
./scripts/gpu_consensus_bench.sh --profile
```

## CI/CD Integration

```yaml
# Example GitHub Actions workflow
- name: Run Performance Benchmarks
  run: |
    ./benchmarks/scripts/run_all_benchmarks.sh --ci
    ./benchmarks/scripts/check_regressions.sh --threshold 5
```

## Contributing

When adding benchmarks:
1. Ensure reproducibility across different hardware
2. Document all assumptions and requirements
3. Include both positive and negative test cases
4. Consider concurrent execution scenarios
5. Add to automated test suite

## References

- [Performance Testing Framework](../performance_testing_framework.md)
- [GPU Consensus Implementation](../crates/gpu-agents/src/kernels/consensus_kernel.cu)
- [Memory Tier Architecture](../docs/architecture/memory-tiers.md)
- [Swarmlet Design](../swarmlet.md)