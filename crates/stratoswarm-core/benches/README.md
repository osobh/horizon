# GPU Communication Benchmarks

Comprehensive benchmarks for stratoswarm-core GPU communication, comparing the modern channel-based approach with the legacy Mutex+VecDeque pattern.

## Running the Benchmarks

Run all benchmarks:
```bash
cargo bench --bench gpu_latency
```

Run specific benchmark group:
```bash
cargo bench --bench gpu_latency -- channel_send_latency
cargo bench --bench gpu_latency -- mutex_vecdeque_latency
cargo bench --bench gpu_latency -- throughput
```

List all available benchmarks:
```bash
cargo bench --bench gpu_latency -- --list
```

## Benchmark Groups

### 1. Channel Send Latency
**Purpose**: Measure the latency of sending GPU commands through broadcast channels.

**Benchmarks**:
- `small_command_bounded`: Small command (Synchronize) through bounded channel (capacity 100)
- `medium_command_bounded`: Medium command (LaunchKernel with 128B params)
- `large_command_bounded`: Large command (1MB data transfer with zero-copy Bytes)
- `small_command_unbounded`: Small command through larger capacity channel (1000)

**Metrics**: Nanoseconds per send operation

**Expected Results**: Channel sends should be sub-microsecond, with minimal overhead for large commands due to zero-copy Bytes.

### 2. Mutex+VecDeque Latency (Legacy Pattern)
**Purpose**: Benchmark the old synchronization pattern for comparison.

**Benchmarks**:
- `push_small_command`: Push small command to Mutex-protected VecDeque
- `push_medium_command`: Push medium command
- `push_large_command`: Push large command (1MB transfer)
- `pop_command`: Pop command from queue
- `push_pop_cycle`: Complete push+pop cycle

**Metrics**: Nanoseconds per operation

**Expected Results**: Higher latency than channels due to Mutex contention, especially under concurrent load.

### 3. Round-Trip Latency
**Purpose**: Measure end-to-end request/response latency.

**Benchmarks**:
- `channel_oneshot`: Send command via broadcast channel + receive response via oneshot
- `mutex_vecdeque`: Send via Mutex queue + poll for completion

**Metrics**: Microseconds for complete round trip

**Expected Results**: Channel approach should show 2-5x improvement due to async primitives and no polling.

### 4. Throughput
**Purpose**: Measure sustained throughput at different batch sizes.

**Benchmarks**:
- `channel/{size}`: Channel throughput at batch sizes 10, 100, 1000, 10000
- `mutex_vecdeque/{size}`: Mutex+VecDeque throughput at same batch sizes

**Metrics**: Elements per second

**Expected Results**:
- Channels should scale better with batch size
- Mutex approach degrades with larger batches due to contention
- Expected channel throughput: 100k-1M commands/sec
- Expected mutex throughput: 10k-100k commands/sec

### 5. Backpressure
**Purpose**: Test behavior under slow consumer scenarios.

**Benchmarks**:
- `channel_bounded_100`: Bounded channel (100) with slow consumer (10µs delay)
- `mutex_vecdeque_limited`: Mutex queue with size limit

**Metrics**: Total time to process 1000 commands

**Expected Results**: Channels handle backpressure more efficiently through built-in flow control.

### 6. Bytes Clone Latency
**Purpose**: Verify zero-copy characteristics of `bytes::Bytes`.

**Benchmarks**:
- Clone operations at 1KB, 64KB, 1MB sizes

**Metrics**: Nanoseconds per clone

**Expected Results**:
- Clone time should be constant regardless of size (< 100ns)
- Demonstrates zero-copy semantics

### 7. Memory Pool Operations
**Purpose**: Benchmark the unified memory pool for buffer management.

**Benchmarks**:
- `allocate_1kb`: Allocate 1KB buffer
- `allocate_1mb`: Allocate 1MB buffer
- `deallocate_1kb`: Deallocate buffer
- `write_1kb`: Write 1KB to buffer
- `read_1kb`: Read 1KB from buffer
- `alloc_dealloc_cycle`: Complete allocation/deallocation cycle

**Metrics**: Nanoseconds per operation

**Expected Results**:
- Allocations: < 1µs
- Deallocations: < 500ns
- Reads/writes: < 100ns for 1KB

### 8. Command Processing Patterns
**Purpose**: Compare sequential command processing patterns.

**Benchmarks**:
- `sequential_processing_channel`: Process 100 commands sequentially via channel
- `sequential_processing_mutex`: Process 100 commands via Mutex queue

**Metrics**: Microseconds to process 100 commands

**Expected Results**: Channels should be 20-30% faster due to reduced synchronization overhead.

### 9. Contention Scenarios
**Purpose**: Test multi-producer scenarios with concurrent load.

**Benchmarks**:
- `multi_producer_channel`: 4 producers × 250 commands each via channel
- `multi_producer_mutex`: 4 producers × 250 commands via Mutex queue

**Metrics**: Milliseconds to process 1000 total commands

**Expected Results**:
- Channels should show 3-10x improvement under contention
- Mutex approach suffers from lock contention as producer count increases

## Performance Targets

Based on the architecture goals, target performance metrics:

| Metric | Channel-Based | Mutex+VecDeque | Improvement Factor |
|--------|---------------|----------------|-------------------|
| Send Latency (small) | < 100ns | < 500ns | 5x |
| Round-Trip | < 5µs | < 25µs | 5x |
| Throughput (1000 batch) | > 500k/sec | > 50k/sec | 10x |
| Contention (4 producers) | < 5ms | < 50ms | 10x |
| Bytes Clone (1MB) | < 100ns | N/A | - |

## Interpreting Results

### Success Criteria

1. **Channel sends faster than Mutex pushes**: Demonstrates lower synchronization overhead
2. **Linear throughput scaling**: Channels should maintain performance as batch size increases
3. **Zero-copy verification**: Bytes clone time constant across sizes
4. **Contention resilience**: Channels should show minimal degradation with multiple producers

### Performance Regression Detection

Run benchmarks before and after changes:
```bash
# Baseline
cargo bench --bench gpu_latency -- --save-baseline main

# After changes
cargo bench --bench gpu_latency -- --baseline main
```

Criterion will automatically detect regressions and report percentage changes.

## Hardware Considerations

Benchmark results vary by hardware:
- **CPU**: Higher core count improves multi-producer scenarios
- **Memory**: Faster RAM reduces channel allocation overhead
- **OS**: MacOS, Linux, Windows show different synchronization primitives performance

Current benchmarks run on: Apple Silicon (M-series) / x86-64 Linux

## Profiling Integration

For detailed profiling:
```bash
# Generate flamegraphs
cargo flamegraph --bench gpu_latency -- --profile-time 30

# Profile with perf (Linux)
perf record --call-graph dwarf -- cargo bench --bench gpu_latency
perf report
```

## CI/CD Integration

Recommended CI workflow:
1. Run benchmarks on every PR
2. Compare against main branch baseline
3. Fail if regression > 10%
4. Track metrics over time

Example GitHub Actions:
```yaml
- name: Run benchmarks
  run: cargo bench --bench gpu_latency -- --save-baseline pr-${{ github.event.pull_request.number }}

- name: Compare with main
  run: cargo bench --bench gpu_latency -- --baseline main
```

## Troubleshooting

### Benchmark Variability
If results vary >5% between runs:
- Close background applications
- Disable CPU frequency scaling
- Run with higher sample count: `cargo bench -- --sample-size 1000`

### Out of Memory
For large throughput benchmarks:
- Reduce batch sizes in code
- Increase system swap
- Run individual benchmark groups

### Timeout Issues
Some benchmarks may timeout on slower systems:
- Increase Criterion timeout in code
- Reduce iteration count
- Skip heavy benchmarks: `cargo bench -- --skip contention`

## Future Enhancements

Planned benchmark additions:
- [ ] GPU kernel execution latency
- [ ] Multi-GPU coordination overhead
- [ ] Memory pressure scenarios
- [ ] Error path performance
- [ ] Metrics collection overhead
