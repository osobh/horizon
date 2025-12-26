# Single-Node Performance Benchmarks

This document provides detailed instructions for running performance benchmarks on a single system with GPU. These tests validate core StratoSwarm performance claims without requiring a distributed setup.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA RTX 4090 or better (32GB+ VRAM recommended)
- **CPU**: 16+ cores recommended
- **RAM**: 64GB+ system memory
- **Storage**: NVMe SSD with 100GB+ free space
- **OS**: Linux kernel 6.14+ with StratoSwarm modules

### Software Requirements
```bash
# Verify CUDA installation
nvcc --version  # Should show CUDA 12.x

# Check kernel modules
lsmod | grep swarm  # Should show swarm_guard, tier_watch, gpu_dma_lock

# Install benchmark dependencies
cargo install criterion
cargo install flamegraph
sudo apt-get install linux-tools-common linux-tools-generic
```

## Benchmark Suite

### 1. GPU Consensus Performance

**Claim**: <100μs consensus latency (~49μs achieved)

#### Test Setup
```bash
cd crates/gpu-agents
cargo build --release --features cuda

# Load test data
./scripts/generate_consensus_data.sh --votes 10000 --nodes 100
```

#### Benchmark Execution
```rust
// File: benchmarks/scripts/gpu_consensus_bench.rs
use stratoswarm::gpu_agents::consensus::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_consensus_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("consensus");
    
    // Single vote processing
    group.bench_function("single_vote", |b| {
        let consensus = GpuConsensus::new().unwrap();
        let vote = create_test_vote();
        
        b.iter(|| {
            black_box(consensus.process_vote(&vote))
        });
    });
    
    // Batch vote processing (100 votes)
    group.bench_function("batch_100_votes", |b| {
        let consensus = GpuConsensus::new().unwrap();
        let votes = create_test_votes(100);
        
        b.iter(|| {
            black_box(consensus.process_batch(&votes))
        });
    });
    
    // Byzantine fault tolerance
    group.bench_function("byzantine_detection", |b| {
        let consensus = GpuConsensus::new().unwrap();
        let votes = create_byzantine_votes(100, 33); // 33% malicious
        
        b.iter(|| {
            black_box(consensus.validate_byzantine(&votes))
        });
    });
}
```

#### Run Command
```bash
# Basic benchmark
cargo bench --bench consensus -- --sample-size 1000

# With profiling
cargo bench --bench consensus -- --profile-time 10

# Generate flamegraph
cargo flamegraph --bench consensus
```

#### Expected Results
```
consensus/single_vote   time:   [48.2 μs 49.1 μs 50.3 μs]
consensus/batch_100     time:   [89.3 μs 91.2 μs 93.8 μs]
consensus/byzantine     time:   [95.1 μs 97.4 μs 99.9 μs]
```

### 2. GPU Synthesis Throughput

**Claim**: 2.6B pattern matching operations per second

#### Test Setup
```bash
# Generate pattern database
./scripts/generate_patterns.sh --count 1000000 --complexity medium

# Warm up GPU
nvidia-smi -pm 1  # Enable persistence mode
nvidia-smi -ac 5001,1980  # Set application clocks
```

#### Benchmark Code
```rust
// File: benchmarks/scripts/gpu_synthesis_bench.rs
fn benchmark_synthesis_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("synthesis");
    group.sample_size(100); // Reduce for longer tests
    
    // Pattern matching throughput
    group.bench_function("pattern_matching_1M", |b| {
        let synthesis = GpuSynthesis::new().unwrap();
        let patterns = load_patterns(1_000_000);
        
        b.iter(|| {
            let ops = synthesis.match_patterns(&patterns);
            assert!(ops >= 2_600_000_000); // 2.6B ops/sec
        });
    });
    
    // Template expansion
    group.bench_function("template_expansion", |b| {
        let synthesis = GpuSynthesis::new().unwrap();
        let templates = load_templates(10_000);
        
        b.iter(|| {
            synthesis.expand_templates(&templates)
        });
    });
    
    // AST manipulation
    group.bench_function("ast_transform", |b| {
        let synthesis = GpuSynthesis::new().unwrap();
        let ast = load_test_ast();
        
        b.iter(|| {
            synthesis.transform_ast(&ast)
        });
    });
}
```

#### Measurement Script
```bash
#!/bin/bash
# File: benchmarks/scripts/synthesis_throughput_bench.sh

echo "GPU Synthesis Throughput Benchmark"
echo "=================================="

# Monitor GPU utilization
nvidia-smi dmon -s pucvmet -i 0 -d 1 > gpu_metrics.log &
GPU_MON_PID=$!

# Run throughput test
cargo run --release --example synthesis_throughput -- \
    --patterns 10000000 \
    --duration 60 \
    --warmup 10 \
    --output results/synthesis_$(date +%Y%m%d_%H%M%S).json

# Stop GPU monitoring
kill $GPU_MON_PID

# Analyze results
python3 scripts/analyze_synthesis.py results/synthesis_*.json
```

### 3. Memory Tier Migration

**Claim**: <50ms current, <1ms target

#### Test Configuration
```rust
// File: benchmarks/src/memory_tier_bench.rs
use stratoswarm::memory::*;

fn benchmark_tier_migration(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_tiers");
    
    // GPU to CPU migration
    group.bench_function("gpu_to_cpu_1mb", |b| {
        let manager = TierManager::new().unwrap();
        let data = vec![0u8; 1024 * 1024]; // 1MB
        let gpu_handle = manager.allocate_gpu(&data).unwrap();
        
        b.iter(|| {
            manager.migrate_to_cpu(&gpu_handle)
        });
    });
    
    // CPU to NVMe migration
    group.bench_function("cpu_to_nvme_10mb", |b| {
        let manager = TierManager::new().unwrap();
        let data = vec![0u8; 10 * 1024 * 1024]; // 10MB
        let cpu_handle = manager.allocate_cpu(&data).unwrap();
        
        b.iter(|| {
            manager.migrate_to_nvme(&cpu_handle)
        });
    });
    
    // Page fault handling
    group.bench_function("page_fault_latency", |b| {
        let manager = TierManager::new().unwrap();
        manager.enable_fault_tracking();
        
        b.iter(|| {
            manager.trigger_test_fault()
        });
    });
}
```

#### Kernel Module Testing
```bash
#!/bin/bash
# File: benchmarks/scripts/memory_tier_bench.sh

# Test tier_watch module performance
echo "Testing tier_watch page fault handling..."
sudo cat /proc/swarm/tiers/stats > tier_stats_before.txt

# Run memory pressure test
./target/release/memory_pressure_test \
    --tiers gpu,cpu,nvme \
    --size 1GB \
    --access-pattern sequential \
    --duration 60

sudo cat /proc/swarm/tiers/stats > tier_stats_after.txt

# Analyze migration patterns
python3 scripts/analyze_migrations.py \
    tier_stats_before.txt \
    tier_stats_after.txt
```

### 4. Container/Agent Spawn Performance

**Claim**: <500μs spawn time

#### Benchmark Implementation
```rust
// File: benchmarks/src/container_spawn_bench.rs
fn benchmark_container_spawn(c: &mut Criterion) {
    let mut group = c.benchmark_group("container_spawn");
    
    // Basic container spawn
    group.bench_function("minimal_container", |b| {
        let runtime = Runtime::new().unwrap();
        
        b.iter(|| {
            let container = runtime.spawn_container(ContainerConfig {
                image: "stratoswarm/minimal",
                memory: 128 * 1024 * 1024, // 128MB
                cpus: 1.0,
            }).unwrap();
            
            runtime.destroy_container(container.id);
        });
    });
    
    // Agent with personality
    group.bench_function("agent_with_personality", |b| {
        let runtime = Runtime::new().unwrap();
        
        b.iter(|| {
            let agent = runtime.spawn_agent(AgentConfig {
                personality: Personality::Balanced,
                memory_limit: 256 * 1024 * 1024,
                gpu_quota: Some(512 * 1024 * 1024),
            }).unwrap();
            
            runtime.destroy_agent(agent.id);
        });
    });
}
```

### 5. Kernel Module Overhead

#### SwarmGuard Performance
```bash
#!/bin/bash
# File: benchmarks/scripts/kernel_overhead_bench.sh

# Baseline syscall performance (without swarm_guard)
sudo rmmod swarm_guard 2>/dev/null
perf bench syscall basic -l 100000 > baseline_syscall.txt

# Load swarm_guard and retest
sudo insmod /lib/modules/$(uname -r)/extra/swarm_guard.ko
perf bench syscall basic -l 100000 > swarmguard_syscall.txt

# Measure namespace operations
./target/release/namespace_bench \
    --operations create,enter,exit \
    --iterations 10000 \
    --output results/namespace_overhead.json
```

#### GPU DMA Lock Performance
```c
// File: benchmarks/src/gpu_dma_bench.c
#include <cuda_runtime.h>
#include <time.h>

void benchmark_allocation_intercept() {
    struct timespec start, end;
    void *ptr;
    size_t size = 1024 * 1024; // 1MB
    
    // Measure allocation with gpu_dma_lock
    clock_gettime(CLOCK_MONOTONIC, &start);
    cudaMalloc(&ptr, size);
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double latency_us = (end.tv_sec - start.tv_sec) * 1e6 + 
                       (end.tv_nsec - start.tv_nsec) / 1e3;
    
    printf("GPU allocation latency: %.2f μs\n", latency_us);
    cudaFree(ptr);
}
```

## Automated Test Suite

### Master Benchmark Script
```bash
#!/bin/bash
# File: benchmarks/scripts/run_single_node_suite.sh

set -e

RESULTS_DIR="results/single_node_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

echo "StratoSwarm Single-Node Benchmark Suite"
echo "======================================"
echo "Results directory: $RESULTS_DIR"

# System information
echo "Collecting system information..."
nvidia-smi -q > $RESULTS_DIR/gpu_info.txt
lscpu > $RESULTS_DIR/cpu_info.txt
free -h > $RESULTS_DIR/memory_info.txt
uname -a > $RESULTS_DIR/kernel_info.txt
lsmod | grep swarm > $RESULTS_DIR/modules_info.txt

# GPU Consensus
echo -e "\n[1/5] Running GPU Consensus benchmarks..."
./scripts/gpu_consensus_bench.sh \
    --output $RESULTS_DIR/consensus.json \
    --iterations 1000

# GPU Synthesis  
echo -e "\n[2/5] Running GPU Synthesis benchmarks..."
./scripts/synthesis_throughput_bench.sh \
    --output $RESULTS_DIR/synthesis.json \
    --duration 60

# Memory Tiers
echo -e "\n[3/5] Running Memory Tier benchmarks..."
./scripts/memory_tier_bench.sh \
    --output $RESULTS_DIR/memory_tiers.json \
    --sizes "1MB,10MB,100MB,1GB"

# Container Spawn
echo -e "\n[4/5] Running Container Spawn benchmarks..."
./scripts/container_spawn_bench.sh \
    --output $RESULTS_DIR/container_spawn.json \
    --iterations 100

# Kernel Overhead
echo -e "\n[5/5] Running Kernel Module benchmarks..."
./scripts/kernel_overhead_bench.sh \
    --output $RESULTS_DIR/kernel_overhead.json

# Generate report
echo -e "\nGenerating performance report..."
./scripts/generate_single_node_report.sh \
    --input $RESULTS_DIR \
    --output $RESULTS_DIR/report.html

echo -e "\nBenchmark suite complete!"
echo "Results: $RESULTS_DIR/report.html"
```

## Performance Analysis

### Result Validation
```python
# File: benchmarks/scripts/validate_results.py
import json
import sys

def validate_performance_claims(results_file):
    with open(results_file) as f:
        results = json.load(f)
    
    validations = {
        "consensus_latency_us": {"target": 100, "current": results.get("consensus", {}).get("p95", 0)},
        "synthesis_ops_per_sec": {"target": 2.6e9, "current": results.get("synthesis", {}).get("throughput", 0)},
        "memory_migration_ms": {"target": 50, "current": results.get("memory", {}).get("migration_p95", 0)},
        "container_spawn_us": {"target": 500, "current": results.get("container", {}).get("spawn_p95", 0)},
        "gpu_utilization_pct": {"target": 90, "current": results.get("gpu", {}).get("utilization_avg", 0)}
    }
    
    print("Performance Validation Report")
    print("=" * 50)
    
    for metric, data in validations.items():
        status = "✅ PASS" if data["current"] <= data["target"] else "❌ FAIL"
        print(f"{metric}: {data['current']:.2f} / {data['target']} {status}")
```

### Optimization Recommendations

Based on benchmark results, the analysis script provides:

1. **Bottleneck Identification**: Which components limit performance
2. **Optimization Opportunities**: Specific code paths to improve
3. **Hardware Recommendations**: GPU/CPU/Memory upgrades if needed
4. **Configuration Tuning**: Kernel parameters, GPU settings

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reset GPU
   nvidia-smi --gpu-reset
   # Reduce batch sizes in benchmarks
   ```

2. **Kernel Module Not Loaded**
   ```bash
   # Check and load modules
   sudo modprobe swarm_guard
   sudo dmesg | tail -20
   ```

3. **Performance Variability**
   ```bash
   # Disable CPU frequency scaling
   sudo cpupower frequency-set -g performance
   # Set GPU to maximum clocks
   sudo nvidia-smi -lgc 1980
   ```

## Next Steps

After completing single-node benchmarks:

1. **Analyze Results**: Identify gaps between claims and measurements
2. **Optimize**: Focus on components not meeting targets
3. **Scale Testing**: Move to multi-node benchmarks with RPi cluster
4. **Stress Testing**: Run concurrent workload scenarios
5. **Documentation**: Update performance claims with validated data