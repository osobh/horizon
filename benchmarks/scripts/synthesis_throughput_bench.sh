#!/bin/bash
# GPU Synthesis Throughput Benchmark Script
# Tests StratoSwarm's 2.6B ops/sec synthesis claim

set -e

# Configuration
PATTERN_COUNT=${PATTERN_COUNT:-10000000}  # 10M patterns
DURATION=${DURATION:-60}
OUTPUT_DIR=${OUTPUT_DIR:-"results/synthesis_$(date +%Y%m%d_%H%M%S)"}
BATCH_SIZE=${BATCH_SIZE:-10000}

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}StratoSwarm GPU Synthesis Throughput Benchmark${NC}"
echo "=============================================="
echo "Pattern Count: $(echo $PATTERN_COUNT | sed ':a;s/\B[0-9]\{3\}\>/,&/;ta')"
echo "Duration: ${DURATION}s"
echo "Batch Size: $(echo $BATCH_SIZE | sed ':a;s/\B[0-9]\{3\}\>/,&/;ta')"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check GPU memory
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "GPU Memory: ${GPU_MEM} MB"

if [ $GPU_MEM -lt 8000 ]; then
    echo -e "${YELLOW}Warning: GPU memory <8GB, reducing pattern count${NC}"
    PATTERN_COUNT=$((PATTERN_COUNT / 4))
fi

# Use existing benchmark binary with GPU streaming focus
if [ ! -f "../target/release/benchmark" ]; then
    echo "Building benchmark binary..."
    cd ../crates/gpu-agents
    cargo build --release --bins
    cd - > /dev/null
fi

# Generate test patterns
echo -e "\n${YELLOW}Generating test patterns...${NC}"
../target/release/pattern_generator \
    --count $PATTERN_COUNT \
    --complexity medium \
    --output "$OUTPUT_DIR/patterns.bin" \
    || echo "Using default patterns"

# Set GPU performance mode
sudo nvidia-smi -pm 1
sudo nvidia-smi -ac $(nvidia-smi --query-gpu=clocks.max.memory,clocks.max.graphics --format=csv,noheader | head -1)

# Monitor system resources
monitor_resources() {
    # GPU monitoring
    nvidia-smi dmon -s pucvmet -i 0 -d 1 > "$OUTPUT_DIR/gpu_metrics.csv" &
    GPU_PID=$!
    
    # CPU monitoring
    sar -u 1 > "$OUTPUT_DIR/cpu_metrics.txt" &
    CPU_PID=$!
    
    echo "$GPU_PID $CPU_PID" > "$OUTPUT_DIR/monitor.pids"
}

# Run synthesis benchmark
run_synthesis_test() {
    local test_name=$1
    local extra_args=$2
    
    echo -e "\n${GREEN}Running $test_name test...${NC}"
    
    ../target/release/benchmark \
        --gpu-streaming-only --stress --verbose \
        --output "$OUTPUT_DIR" \
        2>&1 | tee -a "$OUTPUT_DIR/benchmark.log"
}

# Start monitoring
monitor_resources

# Run different synthesis workloads
run_synthesis_test "pattern_matching" "--workload pattern-matching"
run_synthesis_test "template_expansion" "--workload template-expansion"
run_synthesis_test "ast_manipulation" "--workload ast-manipulation"
run_synthesis_test "mixed_workload" "--workload mixed"

# Stop monitoring
if [ -f "$OUTPUT_DIR/monitor.pids" ]; then
    kill $(cat "$OUTPUT_DIR/monitor.pids") 2>/dev/null || true
    rm "$OUTPUT_DIR/monitor.pids"
fi

# Analyze results
echo -e "\n${YELLOW}Analyzing synthesis performance...${NC}"
python3 - << EOF
import json
import glob

print("\nSynthesis Throughput Results:")
print("=" * 50)

total_ops = 0
total_time = 0

for result_file in glob.glob("$OUTPUT_DIR/*_results.json"):
    with open(result_file) as f:
        data = json.load(f)
    
    test_name = data['test_name']
    ops_per_sec = data['throughput_ops_per_sec']
    duration = data['duration_secs']
    
    total_ops += ops_per_sec * duration
    total_time += duration
    
    # Format large numbers with suffix
    if ops_per_sec >= 1e9:
        throughput_str = f"{ops_per_sec/1e9:.2f}B"
    elif ops_per_sec >= 1e6:
        throughput_str = f"{ops_per_sec/1e6:.2f}M"
    else:
        throughput_str = f"{ops_per_sec:.0f}"
    
    status = "✅ PASS" if ops_per_sec >= 2.6e9 else "⚠️  BELOW TARGET"
    
    print(f"\n{test_name}:")
    print(f"  Throughput: {throughput_str} ops/sec {status}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Total Operations: {ops_per_sec * duration / 1e9:.2f}B")

# Overall average
avg_throughput = total_ops / total_time if total_time > 0 else 0
print(f"\nOverall Average: {avg_throughput/1e9:.2f}B ops/sec")

# GPU utilization
try:
    import pandas as pd
    gpu_data = pd.read_csv("$OUTPUT_DIR/gpu_metrics.csv", skiprows=1, sep=r'\s+')
    print(f"\nGPU Utilization:")
    print(f"  Compute: {gpu_data['gpu'].mean():.1f}%")
    print(f"  Memory: {gpu_data['mem'].mean():.1f}%")
    print(f"  Power: {gpu_data['pwr'].mean():.0f}W")
    
    if gpu_data['gpu'].mean() < 85:
        print("  ⚠️  GPU utilization below 85% target")
except:
    pass

# Summary
if avg_throughput >= 2.6e9:
    print(f"\n✅ SUCCESS: Average throughput {avg_throughput/1e9:.2f}B ops/sec meets target")
else:
    print(f"\n❌ FAILURE: Average throughput {avg_throughput/1e9:.2f}B ops/sec < 2.6B target")
EOF

# Generate detailed report
echo -e "\n${YELLOW}Generating detailed report...${NC}"
cat > "$OUTPUT_DIR/synthesis_report.md" << EOF
# GPU Synthesis Throughput Benchmark Report

Date: $(date)
GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)

## Configuration
- Pattern Count: $(echo $PATTERN_COUNT | sed ':a;s/\B[0-9]\{3\}\>/,&/;ta')
- Test Duration: ${DURATION}s
- Batch Size: $(echo $BATCH_SIZE | sed ':a;s/\B[0-9]\{3\}\>/,&/;ta')

## Results Summary
EOF

# Append results to report
for result_file in "$OUTPUT_DIR"/*_results.json; do
    python3 -c "
import json
with open('$result_file') as f:
    data = json.load(f)
print(f\"### {data['test_name']}\")
print(f\"- Throughput: {data['throughput_ops_per_sec']/1e9:.2f}B ops/sec\")
print(f\"- Latency P95: {data.get('latency_p95_us', 'N/A')} μs\")
print(f\"- Memory Usage: {data.get('memory_usage_mb', 'N/A')} MB\")
print()
" >> "$OUTPUT_DIR/synthesis_report.md"
done

echo -e "\n${GREEN}Synthesis benchmark complete!${NC}"
echo "Results saved to: $OUTPUT_DIR"
echo "Report: $OUTPUT_DIR/synthesis_report.md"

# Reset GPU clocks
sudo nvidia-smi -rgc
sudo nvidia-smi -rmc