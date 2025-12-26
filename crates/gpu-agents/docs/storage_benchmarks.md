# GPU Agent Storage Benchmarks

## Overview

The storage benchmark suite measures the performance of the GPU agent storage tier under various real-world scenarios. It helps identify bottlenecks, optimize cache configurations, and ensure the storage layer can handle massive agent swarms.

## Benchmark Scenarios

### 1. **Burst Write**
Tests rapid storage of many agents, simulating swarm initialization or mass agent creation.
- Measures: Write throughput, latency distribution
- Use case: Swarm startup, agent spawning events

### 2. **Random Access**
Tests random retrieval patterns across stored agents.
- Measures: Read throughput, cache effectiveness
- Use case: Diverse agent interactions, exploration phases

### 3. **Hot/Cold Pattern**
Tests realistic access patterns where 20% of agents are accessed 80% of the time.
- Measures: Cache hit rate, hot vs cold latency
- Use case: Elite agent selection, popular knowledge nodes

### 4. **Swarm Checkpoint**
Tests full swarm state persistence and restoration.
- Measures: Bulk write/read performance
- Use case: System checkpoints, migration, disaster recovery

### 5. **Knowledge Graph Update**
Tests storage of evolving knowledge graphs.
- Measures: Graph serialization performance
- Use case: Collective learning, knowledge accumulation

### 6. **Concurrent Access**
Tests multiple threads accessing storage simultaneously.
- Measures: Concurrent throughput, contention
- Use case: Parallel agent processing, multi-GPU scenarios

### 7. **Memory Pressure**
Tests storage with large agent states to stress memory systems.
- Measures: Memory efficiency, compression effectiveness
- Use case: Complex neural architectures, large memory agents

### 8. **Evolution Persistence**
Tests generational storage patterns for evolutionary algorithms.
- Measures: Generation checkpoint performance
- Use case: Genetic algorithms, evolutionary strategies

## Running Benchmarks

### Quick Test
```bash
# Run all scenarios with default settings
cargo run --release --bin storage-benchmark

# Run specific scenario
cargo run --release --bin storage-benchmark --scenario burst-write
```

### Performance Test
```bash
# Large-scale test with 1M agents
cargo run --release --bin storage-benchmark \
  --agents 1000000 \
  --iterations 1000 \
  --cache-mb 4096
```

### Production Test
```bash
# Test with production storage path
cargo run --release --bin storage-benchmark \
  --storage-path /magikdev/gpu \
  --agents 5000000 \
  --concurrent 16 \
  --compression
```

## CLI Options

```
GPU Agent Storage Benchmark

USAGE:
    storage-benchmark [OPTIONS]

OPTIONS:
    -s, --scenario <SCENARIO>          Benchmark scenario [default: all]
    -a, --agents <AGENTS>              Number of agents [default: 10000]
    -i, --iterations <ITERATIONS>      Number of iterations [default: 100]
        --gpu-cache <GPU_CACHE>        Enable GPU cache [default: true]
        --cache-mb <CACHE_MB>          GPU cache size in MB [default: 1024]
        --concurrent <CONCURRENT>      Concurrent tasks [default: 4]
        --state-size <STATE_SIZE>      Agent state size [default: 256]
        --compression                  Enable compression
        --storage-path <PATH>          Storage path
        --json                         Output as JSON
    -o, --output <OUTPUT>              Save results to file
    -h, --help                         Print help
```

## Metrics Explained

### Throughput Metrics
- **Store Throughput**: Agents stored per second
- **Retrieve Throughput**: Agents retrieved per second
- **Concurrent Throughput**: Agents processed per second with concurrent access

### Latency Metrics
- **Average Latency**: Mean time for operations
- **P99 Latency**: 99th percentile latency (worst 1%)
- **Hot Agent Latency**: Access time for frequently used agents
- **Cold Agent Latency**: Access time for rarely used agents

### Efficiency Metrics
- **Cache Hit Rate**: Percentage of requests served from cache
- **Memory Efficiency**: Ratio of useful data to allocated memory
- **Compression Ratio**: Data size reduction from compression
- **Contention Rate**: Frequency of concurrent access conflicts

## Example Results

```
🚀 GPU Agent Storage Benchmark
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

▶️  Running BurstWrite scenario...

📊 Results for BurstWrite:
   Duration: 2.34s
   Store throughput: 427,350 agents/sec
   Retrieve throughput: 0 agents/sec
   Avg store latency: 0.23ms (P99: 0.45ms)
   Avg retrieve latency: 0.00ms (P99: 0.00ms)

▶️  Running HotCold scenario...

📊 Results for HotCold:
   Duration: 5.67s
   Store throughput: 0 agents/sec
   Retrieve throughput: 1,234,567 agents/sec
   Cache hit rate: 89.3%
   Avg store latency: 0.00ms (P99: 0.00ms)
   Avg retrieve latency: 0.08ms (P99: 0.12ms)
```

## Performance Tuning

### Cache Optimization
```bash
# Find optimal cache size
for size in 256 512 1024 2048 4096; do
  cargo run --release --bin storage-benchmark \
    --scenario hot-cold \
    --cache-mb $size \
    -o results_${size}mb.json
done
```

### Concurrency Testing
```bash
# Test different concurrency levels
for threads in 1 2 4 8 16 32; do
  cargo run --release --bin storage-benchmark \
    --scenario concurrent \
    --concurrent $threads \
    -o results_${threads}threads.json
done
```

### Compression Analysis
```bash
# Compare with and without compression
cargo run --release --bin storage-benchmark \
  --scenario memory-pressure \
  --agents 100000 \
  --state-size 1024 \
  -o results_uncompressed.json

cargo run --release --bin storage-benchmark \
  --scenario memory-pressure \
  --agents 100000 \
  --state-size 1024 \
  --compression \
  -o results_compressed.json
```

## Integration with Swarm Benchmarks

The storage benchmarks can be combined with other GPU agent benchmarks:

```bash
# Full system benchmark
cargo run --release --bin benchmark -- --suite all --enable-storage
```

## Best Practices

1. **Warm-up Runs**: The benchmark includes automatic warm-up iterations
2. **Release Mode**: Always run benchmarks in release mode for accurate results
3. **Isolated Environment**: Close other applications to reduce interference
4. **Multiple Runs**: Run benchmarks multiple times and average results
5. **Monitor Resources**: Watch GPU memory, disk I/O, and CPU usage

## Interpreting Results

### Good Performance Indicators
- Cache hit rate > 80% for hot/cold scenarios
- P99 latency < 2x average latency
- Linear scaling with concurrent access
- Memory efficiency > 80%

### Warning Signs
- Cache hit rate < 50%
- P99 latency > 10x average
- Decreasing throughput with more threads
- High contention rate (> 20%)

## Troubleshooting

### Low Throughput
- Check disk I/O bandwidth
- Increase cache size
- Enable compression for large agents
- Verify storage path permissions

### High Latency
- Check cache configuration
- Monitor disk queue depth
- Consider NVMe vs SSD vs HDD
- Profile serialization overhead

### Memory Issues
- Reduce agent state size
- Enable compression
- Implement agent pooling
- Use memory-mapped files