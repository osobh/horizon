# Performance Tuning Guide

## Table of Contents

1. [Overview](#overview)
2. [GPU Utilization Optimization](#gpu-utilization-optimization)
3. [Memory Tier Optimization](#memory-tier-optimization)
4. [Kernel Optimization](#kernel-optimization)
5. [I/O Performance](#io-performance)
6. [Network Optimization](#network-optimization)
7. [Monitoring and Profiling](#monitoring-and-profiling)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Overview

This guide provides comprehensive strategies for optimizing ExoRust performance to achieve:
- 90%+ GPU utilization
- <100μs consensus latency
- <1ms memory tier migration
- 1M+ agent throughput

### Key Performance Metrics

| Metric | Target | Current Best |
|--------|--------|--------------|
| GPU Utilization | 90% | 85-92% |
| Consensus Latency | <100μs | 80-120μs |
| Memory Migration | <1ms | 0.8-1.2ms |
| Agent Throughput | 1M/s | 900K/s |
| Job Submission | <10ms | 8-12ms |

## GPU Utilization Optimization

### 1. Enable Auto-Tuning

```rust
use gpu_agents::{UtilizationController, ControllerConfig};

let config = ControllerConfig {
    target_utilization: 0.90,
    control_interval: Duration::from_millis(500),
    aggressive_mode: true,  // Enable for maximum performance
    predictive_mode: true,  // Enable ML-based prediction
    ..Default::default()
};

let controller = UtilizationController::new(device, config).await?;
controller.start().await?;
```

### 2. Workload Balancing

```rust
// Optimal batch sizes for RTX 5090
const OPTIMAL_BATCH_SIZES: &[u32] = &[
    256,   // Small workloads
    1024,  // Medium workloads  
    4096,  // Large workloads
    16384, // Very large workloads
];

// Dynamic batch size adjustment
workload_balancer.adjust_batch_size(current_utilization);
```

### 3. Kernel Configuration

```toml
# Optimal kernel configurations
[kernel.consensus]
block_size = 256
grid_size = 512
shared_memory = 16384

[kernel.evolution]
block_size = 512
grid_size = 1024
shared_memory = 32768

[kernel.synthesis]
block_size = 128
grid_size = 2048
shared_memory = 8192
```

### 4. Multi-Stream Execution

```rust
// Use multiple CUDA streams for concurrent execution
const NUM_STREAMS: usize = 4;

let streams: Vec<CudaStream> = (0..NUM_STREAMS)
    .map(|_| device.create_stream().unwrap())
    .collect();

// Distribute work across streams
for (i, batch) in work_batches.chunks(batch_size).enumerate() {
    let stream = &streams[i % NUM_STREAMS];
    launch_kernel_on_stream(batch, stream)?;
}
```

## Memory Tier Optimization

### 1. Tier Configuration

```rust
// Optimal tier thresholds
const TIER_THRESHOLDS: TierThresholds = TierThresholds {
    gpu_to_cpu: 0.85,      // Evict when GPU >85% full
    cpu_to_nvme: 0.90,     // Evict when CPU >90% full
    nvme_to_ssd: 0.95,     // Evict when NVMe >95% full
    ssd_to_hdd: 0.98,      // Evict when SSD >98% full
};

// Page sizes for each tier
const PAGE_SIZES: [usize; 5] = [
    4096,      // GPU: 4KB pages
    4096,      // CPU: 4KB pages
    65536,     // NVMe: 64KB pages
    262144,    // SSD: 256KB pages
    1048576,   // HDD: 1MB pages
];
```

### 2. Prefetching Strategy

```rust
// Configure prefetcher
let prefetch_config = PrefetchConfig {
    enable_ml_prediction: true,
    prefetch_distance: 16,      // Pages ahead
    confidence_threshold: 0.8,   // ML confidence
    max_prefetch_size: 64 * 1024 * 1024, // 64MB
};

// Pattern-specific prefetching
match access_pattern {
    AccessPattern::Sequential => prefetcher.set_distance(32),
    AccessPattern::Strided(stride) => prefetcher.set_stride(stride),
    AccessPattern::Random => prefetcher.disable(),
    AccessPattern::Temporal => prefetcher.enable_temporal(),
}
```

### 3. Compression Tuning

```rust
// Tier-specific compression
const COMPRESSION_CONFIG: [CompressionConfig; 5] = [
    CompressionConfig::None,           // GPU: No compression
    CompressionConfig::None,           // CPU: No compression  
    CompressionConfig::Lz4 { level: 1 }, // NVMe: Fast LZ4
    CompressionConfig::Zstd { level: 3 }, // SSD: Balanced ZSTD
    CompressionConfig::Zstd { level: 9 }, // HDD: Maximum ZSTD
];
```

## Kernel Optimization

### 1. Occupancy Optimization

```rust
// Calculate optimal block size for maximum occupancy
fn optimal_block_size(registers_per_thread: u32, shared_mem_per_block: u32) -> u32 {
    const MAX_THREADS_PER_SM: u32 = 2048;
    const MAX_REGISTERS_PER_SM: u32 = 65536;
    const MAX_SHARED_MEM_PER_SM: u32 = 164 * 1024;
    
    let threads_by_registers = MAX_REGISTERS_PER_SM / registers_per_thread;
    let threads_by_shared_mem = if shared_mem_per_block > 0 {
        MAX_SHARED_MEM_PER_SM / shared_mem_per_block * 32 // warp size
    } else {
        MAX_THREADS_PER_SM
    };
    
    threads_by_registers.min(threads_by_shared_mem).min(1024)
}
```

### 2. Kernel Fusion

```rust
// Enable kernel fusion for common patterns
let fusion_config = FusionConfig {
    enable_auto_fusion: true,
    min_ops_to_fuse: 2,
    max_fusion_depth: 4,
    patterns: vec![
        FusionPattern::MapReduce,
        FusionPattern::TransformFilter,
        FusionPattern::MatrixOps,
    ],
};

let fusion_engine = KernelFusionEngine::new(device, fusion_config)?;
fusion_engine.analyze_and_fuse(kernel_sequence)?;
```

### 3. Memory Access Optimization

```cuda
// Coalesced memory access pattern
__global__ void optimized_kernel(float* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Coalesced access - threads access consecutive addresses
    for (int i = tid; i < n; i += stride) {
        data[i] = process(data[i]);
    }
}

// Shared memory optimization
__global__ void shared_mem_kernel(float* input, float* output, int n) {
    __shared__ float tile[TILE_SIZE];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load to shared memory
    if (gid < n) {
        tile[tid] = input[gid];
    }
    __syncthreads();
    
    // Process in shared memory
    if (tid < TILE_SIZE && gid < n) {
        tile[tid] = complex_operation(tile[tid]);
    }
    __syncthreads();
    
    // Write back
    if (gid < n) {
        output[gid] = tile[tid];
    }
}
```

## I/O Performance

### 1. NVMe Optimization

```rust
// Direct I/O for NVMe
let nvme_config = NvmeConfig {
    direct_io: true,
    io_depth: 32,
    block_size: 4096,
    num_queues: 8,
};

// Aligned buffers for O_DIRECT
let aligned_buffer = AlignedBuffer::new(size, 4096)?;
```

### 2. Batch I/O Operations

```rust
// Batch small files
const BATCH_THRESHOLD: usize = 64 * 1024; // 64KB

let mut batch = Vec::new();
let mut batch_size = 0;

for file in files {
    if file.size() < BATCH_THRESHOLD {
        batch.push(file);
        batch_size += file.size();
        
        if batch_size >= 1024 * 1024 { // 1MB batch
            process_batch(&batch).await?;
            batch.clear();
            batch_size = 0;
        }
    } else {
        process_large_file(&file).await?;
    }
}
```

### 3. GPUDirect Storage

```rust
// Enable GPUDirect Storage when available
let gds_config = GpuDirectConfig {
    enable_gds: true,
    max_transfer_size: 256 * 1024 * 1024, // 256MB
    num_io_queues: 8,
    enable_async_io: true,
    batch_size: 32,
    enable_fallback: true,
};

let gds_manager = GpuDirectManager::new_with_fallback(gds_config)?;
```

## Network Optimization

### 1. Message Batching

```rust
// Batch network messages
const MESSAGE_BATCH_SIZE: usize = 100;
const BATCH_TIMEOUT: Duration = Duration::from_millis(10);

let mut message_batch = Vec::with_capacity(MESSAGE_BATCH_SIZE);
let mut last_send = Instant::now();

for message in message_stream {
    message_batch.push(message);
    
    if message_batch.len() >= MESSAGE_BATCH_SIZE || 
       last_send.elapsed() > BATCH_TIMEOUT {
        send_batch(&message_batch).await?;
        message_batch.clear();
        last_send = Instant::now();
    }
}
```

### 2. Connection Pooling

```rust
// Connection pool configuration
let pool_config = PoolConfig {
    min_connections: 10,
    max_connections: 100,
    connection_timeout: Duration::from_secs(5),
    idle_timeout: Duration::from_secs(60),
    max_lifetime: Duration::from_secs(300),
};

let connection_pool = ConnectionPool::new(pool_config)?;
```

## Monitoring and Profiling

### 1. Real-time Metrics

```rust
// Enable comprehensive monitoring
let monitor_config = MonitorConfig {
    gpu_metrics: true,
    memory_metrics: true,
    network_metrics: true,
    io_metrics: true,
    sample_interval: Duration::from_millis(100),
};

let monitor = PerformanceMonitor::new(monitor_config)?;
monitor.start().await?;

// Access metrics
let metrics = monitor.get_current_metrics();
println!("GPU Utilization: {:.1}%", metrics.gpu_utilization * 100.0);
println!("Memory Bandwidth: {:.1} GB/s", metrics.memory_bandwidth);
```

### 2. Profiling Tools

```bash
# NVIDIA Nsight Systems
nsys profile --stats=true ./exorust

# NVIDIA Nsight Compute
ncu --set full ./exorust

# Custom profiling
EXORUST_PROFILE=1 EXORUST_PROFILE_OUTPUT=profile.json ./exorust
```

### 3. Performance Dashboard

```rust
// Built-in performance dashboard
let dashboard = PerformanceDashboard::new()?;
dashboard.set_update_interval(Duration::from_secs(1));
dashboard.add_metric("gpu_utilization", MetricType::Percentage);
dashboard.add_metric("consensus_latency", MetricType::Microseconds);
dashboard.add_metric("agent_throughput", MetricType::CountPerSecond);
dashboard.start().await?;
```

## Best Practices

### 1. Agent Pool Management

```rust
// Maintain optimal agent pool sizes
const AGENT_POOL_CONFIG: AgentPoolConfig = AgentPoolConfig {
    gpu_agents: 64_000,    // GPU tier capacity
    cpu_agents: 200_000,   // CPU tier capacity
    min_free_gpu: 0.10,    // Keep 10% GPU free
    min_free_cpu: 0.15,    // Keep 15% CPU free
};
```

### 2. Resource Budgeting

```rust
// Set resource budgets
let resource_budget = ResourceBudget {
    max_gpu_memory: 28 * 1024 * 1024 * 1024,  // 28GB (leave 4GB free)
    max_cpu_memory: 80 * 1024 * 1024 * 1024,  // 80GB (leave 16GB free)
    max_io_bandwidth: 6 * 1024 * 1024 * 1024, // 6GB/s
    max_network_bandwidth: 1024 * 1024 * 1024, // 1GB/s
};
```

### 3. Error Recovery

```rust
// Graceful degradation
match gpu_operation().await {
    Ok(result) => result,
    Err(GpuError::OutOfMemory) => {
        // Reduce batch size and retry
        reduce_batch_size();
        retry_with_smaller_batch().await?
    }
    Err(GpuError::KernelTimeout) => {
        // Fall back to CPU
        cpu_fallback_operation().await?
    }
    Err(e) => return Err(e),
}
```

## Troubleshooting

### Common Performance Issues

1. **Low GPU Utilization**
   - Check batch sizes
   - Verify kernel occupancy
   - Look for CPU bottlenecks
   - Enable multi-stream execution

2. **High Memory Migration Latency**
   - Reduce page size for hot data
   - Enable prefetching
   - Check tier thresholds
   - Optimize access patterns

3. **Consensus Latency Spikes**
   - Check network latency
   - Verify atomic operation contention
   - Review leader election frequency
   - Monitor GPU memory bandwidth

4. **I/O Bottlenecks**
   - Enable GPUDirect Storage
   - Use larger batch sizes
   - Check filesystem alignment
   - Monitor NVMe queue depth

### Performance Debugging

```rust
// Enable detailed performance logging
env::set_var("EXORUST_PERF_LOG", "debug");
env::set_var("EXORUST_TRACE_KERNELS", "1");
env::set_var("EXORUST_PROFILE_MEMORY", "1");

// Capture performance trace
let trace = PerformanceTrace::start("operation_name");
// ... operation code ...
let report = trace.end();
println!("Operation took: {:?}", report.duration);
println!("GPU cycles: {}", report.gpu_cycles);
```

### Optimization Checklist

- [ ] GPU utilization >85%
- [ ] Kernel occupancy >60%
- [ ] Memory bandwidth utilization >70%
- [ ] Page fault rate <1000/s
- [ ] Network latency <1ms (local)
- [ ] I/O queue depth 16-32
- [ ] CPU usage <50% per core
- [ ] No memory leaks
- [ ] Error rate <0.01%
- [ ] Response time P99 within SLA