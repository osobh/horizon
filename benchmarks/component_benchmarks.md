# Component-Specific Performance Benchmarks

This document provides detailed benchmarking instructions for individual StratoSwarm components, leveraging existing benchmark implementations found in the crates.

## Overview

Many StratoSwarm crates already include benchmark suites using Criterion.rs. This guide shows how to run these benchmarks and interpret results for performance validation and optimization.

## Component Benchmarks

### 1. Knowledge Graph Performance

**Location**: `crates/knowledge-graph/benches/`  
**Files**: `query_benchmarks.rs`, `semantic_benchmarks.rs`

#### Running Knowledge Graph Benchmarks
```bash
cd crates/knowledge-graph

# Run all knowledge graph benchmarks
cargo bench

# Run specific benchmark
cargo bench query_benchmarks

# Run with specific features
cargo bench --features gpu-acceleration

# Generate HTML report
cargo bench -- --save-baseline current
```

#### Benchmark Scenarios

##### Node Operations
```rust
// Measures node creation, lookup, and deletion performance
benchmark_node_operations:
  - create_knowledge_graph: Graph initialization time
  - add_1000_nodes: Bulk node insertion performance
  - find_node_by_id: O(1) lookup validation
  - update_node_properties: Property modification overhead
  - delete_nodes: Cleanup performance
```

##### Query Performance
```rust
// Tests various query patterns and complexities
benchmark_query_operations:
  - simple_property_match: Basic property filtering
  - complex_graph_traversal: Multi-hop relationship queries
  - semantic_similarity_search: Vector-based similarity
  - pattern_matching: Subgraph pattern detection
  - aggregation_queries: Count, sum, average operations
```

##### GPU Acceleration (if available)
```bash
# Enable GPU benchmarks
cargo bench --features gpu-acceleration -- gpu_

# Expected improvements:
# - Semantic search: 10-50x speedup
# - Pattern matching: 5-20x speedup
# - Large graph traversal: 3-10x speedup
```

#### Performance Targets
| Operation | Target | Current | Notes |
|-----------|--------|---------|-------|
| Node Creation | <1μs | TBD | Per node |
| Simple Query | <10μs | TBD | 1-hop traversal |
| Complex Query | <1ms | TBD | 5-hop traversal |
| Semantic Search | <100μs | TBD | 1M nodes |

### 2. AI Assistant Natural Language Processing

**Location**: `crates/ai-assistant/benches/`  
**Files**: `nl_parsing.rs`, `command_generation.rs`

#### Running AI Assistant Benchmarks
```bash
cd crates/ai-assistant

# Run all AI benchmarks
cargo bench

# Profile intent detection
cargo bench -- intent_detection --profile-time 10

# Test with different corpus sizes
BENCH_CORPUS_SIZE=10000 cargo bench
```

#### Benchmark Components

##### Natural Language Parsing
```rust
// Measures NL understanding performance
nl_parsing_benchmarks:
  - tokenization_speed: Words per second
  - intent_classification: Intent detection latency
  - entity_extraction: Named entity recognition
  - command_parsing: Full parse pipeline
```

##### Command Generation
```rust
// Tests command synthesis performance
command_generation_benchmarks:
  - simple_command: Basic command generation
  - complex_command: Multi-parameter commands
  - batch_generation: Multiple commands
  - template_expansion: Dynamic template filling
```

#### Test Scenarios
```bash
#!/bin/bash
# File: benchmarks/scripts/ai_assistant_bench.sh

# Test different input complexities
for complexity in simple medium complex; do
    echo "Testing $complexity inputs..."
    cargo bench -- --complexity $complexity
done

# Test learning system performance
cargo bench learning_system -- --iterations 1000

# Measure memory usage during parsing
/usr/bin/time -v cargo bench nl_parsing 2>&1 | grep "Maximum resident"
```

### 3. Zero-Config Intelligence

**Location**: `crates/zero-config/`  
**Note**: No existing benchmarks, needs creation

#### Creating Zero-Config Benchmarks
```rust
// File: crates/zero-config/benches/analysis_bench.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use zero_config::*;

fn benchmark_code_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("code_analysis");
    
    // Language detection
    group.bench_function("detect_language", |b| {
        let files = load_test_files();
        b.iter(|| {
            for file in &files {
                black_box(detect_language(&file.content));
            }
        });
    });
    
    // Dependency analysis
    group.bench_function("analyze_dependencies_rust", |b| {
        let cargo_toml = include_str!("../fixtures/Cargo.toml");
        b.iter(|| {
            black_box(analyze_rust_dependencies(cargo_toml))
        });
    });
    
    // Pattern recognition
    group.bench_function("recognize_patterns", |b| {
        let codebase = load_test_codebase();
        b.iter(|| {
            black_box(recognize_deployment_patterns(&codebase))
        });
    });
    
    // Configuration generation
    group.bench_function("generate_config", |b| {
        let analysis = load_analysis_results();
        b.iter(|| {
            black_box(generate_agent_configuration(&analysis))
        });
    });
}

criterion_group!(benches, benchmark_code_analysis);
criterion_main!(benches);
```

#### Performance Targets
| Operation | Target | Lines/sec | Notes |
|-----------|--------|-----------|-------|
| Language Detection | <1ms | 100K+ | Per file |
| Dependency Analysis | <100ms | 10K+ | Per project |
| Pattern Recognition | <1s | 1K+ | Full codebase |
| Config Generation | <10ms | - | Per app |

### 4. Streaming Pipeline Performance

**Location**: `crates/streaming/`  
**Note**: Tests exist, benchmarks need creation

#### Streaming Benchmark Suite
```rust
// File: crates/streaming/benches/throughput_bench.rs

fn benchmark_streaming_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming");
    group.sample_size(10); // Reduce for longer benchmarks
    
    // Raw throughput
    group.bench_function("throughput_1gb", |b| {
        let data = vec![0u8; 1024 * 1024 * 1024]; // 1GB
        let pipeline = StreamingPipeline::new();
        
        b.iter(|| {
            pipeline.process(&data)
        });
    });
    
    // Compression performance
    group.bench_function("compression_huffman", |b| {
        let data = generate_compressible_data(10 * 1024 * 1024); // 10MB
        let compressor = HuffmanCompressor::new();
        
        b.iter(|| {
            black_box(compressor.compress(&data))
        });
    });
    
    // GPU-accelerated operations
    #[cfg(feature = "gpu")]
    group.bench_function("gpu_string_ops", |b| {
        let strings = generate_test_strings(100_000);
        let gpu_processor = GpuStringProcessor::new();
        
        b.iter(|| {
            black_box(gpu_processor.process_batch(&strings))
        });
    });
}
```

#### Throughput Testing Script
```bash
#!/bin/bash
# File: benchmarks/scripts/streaming_throughput_test.sh

echo "Streaming Pipeline Throughput Test"
echo "================================="

# Test different data types
for dtype in text binary mixed; do
    echo -e "\nTesting $dtype data..."
    cargo bench --bench streaming -- $dtype
done

# Test with different batch sizes
for batch_size in 1MB 10MB 100MB 1GB; do
    echo -e "\nBatch size: $batch_size"
    STREAMING_BATCH_SIZE=$batch_size cargo bench
done

# GPU vs CPU comparison
echo -e "\nGPU vs CPU Performance:"
cargo bench --features gpu --bench streaming > gpu_results.txt
cargo bench --no-default-features --bench streaming > cpu_results.txt
./scripts/compare_gpu_cpu.py gpu_results.txt cpu_results.txt
```

### 5. Memory Management Benchmarks

**Location**: `crates/memory/`

#### Memory Allocation Performance
```rust
// File: crates/memory/benches/allocation_bench.rs

fn benchmark_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory");
    
    // GPU allocation
    group.bench_function("gpu_allocate_1mb", |b| {
        let allocator = GpuMemoryAllocator::new();
        b.iter(|| {
            let handle = allocator.allocate(1024 * 1024).unwrap();
            allocator.deallocate(handle).unwrap();
        });
    });
    
    // Memory pool efficiency
    group.bench_function("pool_allocation", |b| {
        let pool = MemoryPool::new(100 * 1024 * 1024); // 100MB pool
        b.iter(|| {
            let mut handles = vec![];
            for _ in 0..100 {
                handles.push(pool.allocate(1024 * 1024).unwrap());
            }
            for handle in handles {
                pool.deallocate(handle).unwrap();
            }
        });
    });
    
    // Tier migration
    group.bench_function("tier_migration", |b| {
        let manager = TierManager::new();
        let data = vec![0u8; 10 * 1024 * 1024]; // 10MB
        let handle = manager.allocate_gpu(&data).unwrap();
        
        b.iter(|| {
            manager.migrate_to_cpu(&handle).unwrap();
            manager.migrate_to_gpu(&handle).unwrap();
        });
    });
}
```

### 6. Evolution Engine Benchmarks

**Location**: `crates/evolution-engines/`

#### Evolution Performance Testing
```rust
// File: crates/evolution-engines/benches/evolution_bench.rs

fn benchmark_evolution_engines(c: &mut Criterion) {
    let mut group = c.benchmark_group("evolution");
    
    // ADAS performance
    group.bench_function("adas_generation", |b| {
        let adas = ADASEngine::new();
        let population = generate_test_population(1000);
        
        b.iter(|| {
            adas.evolve_generation(&population)
        });
    });
    
    // DGM self-improvement
    group.bench_function("dgm_iteration", |b| {
        let dgm = DGMEngine::new();
        let model = generate_test_model();
        
        b.iter(|| {
            dgm.self_improve(&model)
        });
    });
    
    // SwarmAgentic coordination
    group.bench_function("swarm_coordination", |b| {
        let swarm = SwarmEngine::new();
        let agents = generate_test_agents(100);
        
        b.iter(|| {
            swarm.coordinate_evolution(&agents)
        });
    });
}
```

### 7. Time-Travel Debugger Performance

**Location**: `crates/time-travel-debugger/benches/`  
**Files**: `snapshot_performance.rs`, `event_replay_performance.rs`

#### Running Debugger Benchmarks
```bash
cd crates/time-travel-debugger

# Snapshot performance
cargo bench snapshot

# Event replay performance
cargo bench replay

# Memory usage analysis
cargo bench -- --measure-memory
```

### 8. Multi-Region Coordination

**Location**: `crates/multi-region/benches/`  
**File**: `multi_region_bench.rs`

#### Cross-Region Performance Testing
```bash
# Simulate different latencies
SIMULATED_LATENCY=50ms cargo bench multi_region

# Test with various region counts
for regions in 2 4 8 16; do
    REGION_COUNT=$regions cargo bench
done
```

### 9. Zero-Trust Security

**Location**: `crates/zero-trust/benches/`  
**File**: `zero_trust_bench.rs`

#### Security Operation Benchmarks
```bash
# Attestation performance
cargo bench attestation

# Certificate validation
cargo bench certificate_validation

# Trust score calculation
cargo bench trust_scoring
```

## Automated Component Benchmark Suite

### Master Component Benchmark Script
```bash
#!/bin/bash
# File: benchmarks/scripts/run_component_benchmarks.sh

set -e

RESULTS_DIR="results/components_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

echo "StratoSwarm Component Benchmark Suite"
echo "===================================="

# Function to run and collect benchmark results
run_component_bench() {
    local COMPONENT=$1
    local CRATE_PATH=$2
    
    echo -e "\n[$COMPONENT] Running benchmarks..."
    cd $CRATE_PATH
    
    # Run benchmarks and save results
    cargo bench --no-fail-fast 2>&1 | tee $RESULTS_DIR/${COMPONENT}_bench.log
    
    # Copy criterion results if they exist
    if [ -d "target/criterion" ]; then
        cp -r target/criterion $RESULTS_DIR/${COMPONENT}_criterion
    fi
    
    cd - > /dev/null
}

# Run all component benchmarks
run_component_bench "knowledge-graph" "crates/knowledge-graph"
run_component_bench "ai-assistant" "crates/ai-assistant"
run_component_bench "time-travel-debugger" "crates/time-travel-debugger"
run_component_bench "multi-region" "crates/multi-region"
run_component_bench "zero-trust" "crates/zero-trust"

# Components that need benchmark creation
echo -e "\n[zero-config] Creating and running benchmarks..."
./scripts/create_zero_config_bench.sh
run_component_bench "zero-config" "crates/zero-config"

echo -e "\n[streaming] Creating and running benchmarks..."
./scripts/create_streaming_bench.sh
run_component_bench "streaming" "crates/streaming"

# Generate consolidated report
echo -e "\nGenerating consolidated report..."
python3 scripts/analyze_component_benchmarks.py \
    --results-dir $RESULTS_DIR \
    --output $RESULTS_DIR/component_benchmark_report.html

echo -e "\nComponent benchmarks complete!"
echo "Report: $RESULTS_DIR/component_benchmark_report.html"
```

### Performance Comparison Tool
```python
#!/usr/bin/env python3
# File: benchmarks/scripts/compare_component_performance.py

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class ComponentPerformanceAnalyzer:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.components = {}
        self.load_results()
    
    def load_results(self):
        """Load benchmark results from all components"""
        for component_dir in self.results_dir.glob("*_criterion"):
            component_name = component_dir.name.replace("_criterion", "")
            self.components[component_name] = self.parse_criterion_results(component_dir)
    
    def parse_criterion_results(self, criterion_dir):
        """Parse Criterion benchmark results"""
        results = {}
        
        for bench_dir in criterion_dir.glob("*"):
            if bench_dir.is_dir():
                bench_name = bench_dir.name
                estimates_file = bench_dir / "base" / "estimates.json"
                
                if estimates_file.exists():
                    with open(estimates_file) as f:
                        data = json.load(f)
                        results[bench_name] = {
                            'mean': data['mean']['point_estimate'],
                            'std_dev': data['std_dev']['point_estimate'],
                            'median': data['median']['point_estimate']
                        }
        
        return results
    
    def generate_comparison_report(self):
        """Generate performance comparison report"""
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'components': {}
        }
        
        for component, benchmarks in self.components.items():
            report['components'][component] = {
                'benchmark_count': len(benchmarks),
                'benchmarks': benchmarks,
                'performance_summary': self.summarize_component_performance(benchmarks)
            }
        
        # Create visualizations
        self.create_performance_charts()
        
        return report
    
    def create_performance_charts(self):
        """Create performance comparison charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Component performance overview
        ax = axes[0, 0]
        components = []
        avg_latencies = []
        
        for comp, benches in self.components.items():
            components.append(comp)
            latencies = [b['mean'] for b in benches.values() if 'latency' in b]
            avg_latencies.append(np.mean(latencies) if latencies else 0)
        
        ax.bar(components, avg_latencies)
        ax.set_xlabel('Component')
        ax.set_ylabel('Average Latency (ns)')
        ax.set_title('Component Latency Comparison')
        ax.tick_params(axis='x', rotation=45)
        
        # Throughput comparison
        ax = axes[0, 1]
        throughputs = {}
        
        for comp, benches in self.components.items():
            for bench_name, data in benches.items():
                if 'throughput' in bench_name:
                    throughputs[f"{comp}/{bench_name}"] = data['mean']
        
        if throughputs:
            ax.bar(range(len(throughputs)), list(throughputs.values()))
            ax.set_xticks(range(len(throughputs)))
            ax.set_xticklabels(list(throughputs.keys()), rotation=45, ha='right')
            ax.set_ylabel('Operations/sec')
            ax.set_title('Throughput Comparison')
        
        # Performance variance
        ax = axes[1, 0]
        for comp, benches in self.components.items():
            variances = [b['std_dev'] / b['mean'] * 100 for b in benches.values()]
            if variances:
                ax.scatter([comp] * len(variances), variances, alpha=0.6)
        
        ax.set_xlabel('Component')
        ax.set_ylabel('Coefficient of Variation (%)')
        ax.set_title('Performance Stability')
        ax.tick_params(axis='x', rotation=45)
        
        # Target vs Actual
        ax = axes[1, 1]
        targets = {
            'knowledge-graph': {'query': 10000},  # 10μs target
            'ai-assistant': {'parse': 1000000},   # 1ms target
            'streaming': {'throughput': 1e9},     # 1GB/s target
        }
        
        for comp, target_dict in targets.items():
            if comp in self.components:
                for bench, target in target_dict.items():
                    actual = next((b['mean'] for n, b in self.components[comp].items() 
                                 if bench in n), None)
                    if actual:
                        ax.bar(f"{comp}/{bench}", [actual, target], 
                              color=['blue', 'green'], alpha=0.7)
        
        ax.set_ylabel('Performance (ns or ops/sec)')
        ax.set_title('Target vs Actual Performance')
        ax.legend(['Actual', 'Target'])
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'component_performance_comparison.png')

# Usage
if __name__ == "__main__":
    analyzer = ComponentPerformanceAnalyzer("results/components_20240801_120000")
    report = analyzer.generate_comparison_report()
    
    with open("results/component_performance_report.json", "w") as f:
        json.dump(report, f, indent=2)
```

## Performance Regression Detection

### Continuous Benchmark Monitoring
```bash
#!/bin/bash
# File: benchmarks/scripts/detect_regressions.sh

# Compare with baseline
BASELINE_DIR="results/baseline"
CURRENT_DIR="results/current"

# Run benchmarks and compare
for component in knowledge-graph ai-assistant streaming; do
    echo "Checking $component for regressions..."
    
    cd crates/$component
    
    # Save baseline if it doesn't exist
    if [ ! -d "$BASELINE_DIR/$component" ]; then
        cargo bench -- --save-baseline baseline
        cp -r target/criterion $BASELINE_DIR/$component
    fi
    
    # Run current benchmarks
    cargo bench -- --baseline baseline
    
    # Check for regressions
    if grep -q "Performance has regressed" target/criterion/*/report/index.html; then
        echo "⚠️  REGRESSION DETECTED in $component!"
        # Extract regression details
        ./scripts/extract_regression_details.py target/criterion
    else
        echo "✅ No regression in $component"
    fi
    
    cd - > /dev/null
done
```

## Integration with CI/CD

### GitHub Actions Benchmark Workflow
```yaml
# File: .github/workflows/benchmarks.yml
name: Component Benchmarks

on:
  pull_request:
    paths:
      - 'crates/**/*.rs'
      - 'Cargo.toml'
      - 'Cargo.lock'

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install Rust
        uses: actions-rust-lang/setup-rust-toolchain@v1
        
      - name: Run Component Benchmarks
        run: ./benchmarks/scripts/run_component_benchmarks.sh
        
      - name: Check for Regressions
        run: ./benchmarks/scripts/detect_regressions.sh
        
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: results/
          
      - name: Comment PR with Results
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const results = fs.readFileSync('results/summary.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: results
            });
```

## Optimization Workflow

Based on component benchmark results:

1. **Identify Bottlenecks**: Components not meeting performance targets
2. **Profile Deeply**: Use flamegraphs and detailed profiling
3. **Optimize**: Focus on hot paths identified by benchmarks
4. **Validate**: Re-run benchmarks to confirm improvements
5. **Document**: Update performance targets with achieved results

## Next Steps

1. **Create Missing Benchmarks**: Zero-config, streaming, memory components
2. **Establish Baselines**: Run all benchmarks to set current performance
3. **Set Targets**: Define acceptable performance for each component
4. **Automate**: Integrate into CI/CD pipeline
5. **Monitor**: Track performance over time to catch regressions