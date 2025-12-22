#!/bin/bash
# GPU Consensus Performance Benchmark Script
# Tests StratoSwarm's <100μs consensus latency claim

set -e

# Configuration
ITERATIONS=${ITERATIONS:-1000}
DURATION=${DURATION:-60}
OUTPUT_DIR=${OUTPUT_DIR:-"results/gpu_consensus_$(date +%Y%m%d_%H%M%S)"}
WARMUP=${WARMUP:-10}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}StratoSwarm GPU Consensus Benchmark${NC}"
echo "===================================="
echo "Iterations: $ITERATIONS"
echo "Duration: ${DURATION}s"
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check GPU availability
if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: No NVIDIA GPU detected${NC}"
    exit 1
fi

# Save GPU info
nvidia-smi -q > "$OUTPUT_DIR/gpu_info.txt"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Use existing benchmark binary with GPU evolution focus
if [ ! -f "../target/release/benchmark" ]; then
    echo "Building benchmark binary..."
    cd ../crates/gpu-agents
    cargo build --release --bins
    cd - > /dev/null
fi

# Set GPU to maximum performance
echo -e "\n${YELLOW}Setting GPU to maximum performance...${NC}"
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc $(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader | head -1)

# Monitor GPU during test
monitor_gpu() {
    nvidia-smi dmon -s pucvmet -i 0 -d 1 > "$OUTPUT_DIR/gpu_monitor.csv" &
    echo $! > "$OUTPUT_DIR/monitor.pid"
}

# Run warmup
echo -e "\n${YELLOW}Running warmup for ${WARMUP}s...${NC}"
../target/release/benchmark \
    --gpu-evolution-only --quick \
    --output /dev/null

# Start monitoring
monitor_gpu

# Run consensus benchmark
echo -e "\n${GREEN}Running consensus benchmark...${NC}"
../target/release/benchmark \
    --gpu-evolution-only --stress --verbose \
    --iterations $ITERATIONS \
    --duration $DURATION \
    --batch-sizes "1,10,100,1000" \
    --byzantine-ratios "0,0.1,0.33" \
    --output "$OUTPUT_DIR/consensus_results.json" \
    2>&1 | tee "$OUTPUT_DIR/benchmark.log"

# Stop monitoring
if [ -f "$OUTPUT_DIR/monitor.pid" ]; then
    kill $(cat "$OUTPUT_DIR/monitor.pid") 2>/dev/null || true
    rm "$OUTPUT_DIR/monitor.pid"
fi

# Analyze results
echo -e "\n${YELLOW}Analyzing results...${NC}"
python3 - << EOF
import json
import statistics

with open("$OUTPUT_DIR/consensus_results.json") as f:
    results = json.load(f)

print("\nConsensus Latency Results:")
print("=" * 50)

for test in results['tests']:
    name = test['name']
    latencies = test['latencies_us']
    
    if latencies:
        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99 = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
        
        # Check against target
        status = "✅ PASS" if p95 < 100 else "❌ FAIL"
        
        print(f"\n{name}:")
        print(f"  P50: {p50:.2f} μs")
        print(f"  P95: {p95:.2f} μs {status}")
        print(f"  P99: {p99:.2f} μs")
        print(f"  Min: {min(latencies):.2f} μs")
        print(f"  Max: {max(latencies):.2f} μs")

# GPU utilization analysis
print("\nGPU Utilization:")
try:
    import pandas as pd
    gpu_data = pd.read_csv("$OUTPUT_DIR/gpu_monitor.csv", skiprows=1, sep=r'\s+')
    print(f"  Average GPU: {gpu_data['gpu'].mean():.1f}%")
    print(f"  Average Memory: {gpu_data['mem'].mean():.1f}%")
except:
    pass

# Summary
consensus_p95 = statistics.quantiles(results['tests'][0]['latencies_us'], n=20)[18]
if consensus_p95 < 100:
    print(f"\n✅ SUCCESS: Consensus latency {consensus_p95:.2f}μs < 100μs target")
else:
    print(f"\n❌ FAILURE: Consensus latency {consensus_p95:.2f}μs > 100μs target")
EOF

# Generate plots if matplotlib is available
if python3 -c "import matplotlib" 2>/dev/null; then
    echo -e "\n${YELLOW}Generating performance plots...${NC}"
    python3 ../scripts/plot_consensus_results.py \
        "$OUTPUT_DIR/consensus_results.json" \
        "$OUTPUT_DIR/consensus_latency_plot.png"
fi

echo -e "\n${GREEN}Benchmark complete!${NC}"
echo "Results saved to: $OUTPUT_DIR"

# Reset GPU clocks
sudo nvidia-smi -rgc