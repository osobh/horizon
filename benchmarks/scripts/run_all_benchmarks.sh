#!/bin/bash
# Master script to run all StratoSwarm benchmarks
# Provides quick validation or comprehensive testing options

set -e

# Configuration
MODE=${1:-quick}  # quick, full, or ci
OUTPUT_BASE="results/full_benchmark_$(date +%Y%m%d_%H%M%S)"
PARALLEL=${PARALLEL:-false}

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}StratoSwarm Comprehensive Benchmark Suite${NC}"
echo "========================================"
echo "Mode: $MODE"
echo "Output: $OUTPUT_BASE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE"/{single_node,swarmlet,components,stress}

# Save system information
save_system_info() {
    echo -e "${YELLOW}Collecting system information...${NC}"
    
    {
        echo "=== System Information ==="
        echo "Date: $(date)"
        echo "Hostname: $(hostname)"
        echo "Kernel: $(uname -r)"
        echo ""
        echo "=== CPU ==="
        lscpu
        echo ""
        echo "=== Memory ==="
        free -h
        echo ""
        echo "=== GPU ==="
        nvidia-smi 2>/dev/null || echo "No NVIDIA GPU detected"
        echo ""
        echo "=== Disk ==="
        df -h .
        echo ""
        echo "=== StratoSwarm Modules ==="
        lsmod | grep swarm || echo "No kernel modules loaded"
    } > "$OUTPUT_BASE/system_info.txt"
}

# Quick mode benchmarks (30 minutes)
run_quick_benchmarks() {
    echo -e "\n${BLUE}Running Quick Validation Benchmarks${NC}"
    
    # Single-node GPU tests (10 minutes)
    echo -e "\n${YELLOW}[1/4] Single-node GPU benchmarks...${NC}"
    ITERATIONS=100 DURATION=30 OUTPUT_DIR="$OUTPUT_BASE/single_node/gpu_consensus" \
        ./scripts/gpu_consensus_bench.sh
    
    DURATION=30 OUTPUT_DIR="$OUTPUT_BASE/single_node/synthesis" \
        ./scripts/synthesis_throughput_bench.sh
    
    # Component benchmarks (10 minutes)
    echo -e "\n${YELLOW}[2/4] Component benchmarks...${NC}"
    cd ../crates/knowledge-graph && cargo bench --no-fail-fast -- --quick
    cd ../ai-assistant && cargo bench --no-fail-fast -- --quick
    cd ../../benchmarks
    
    # Basic stress test (5 minutes)
    echo -e "\n${YELLOW}[3/4] Basic stress test...${NC}"
    DURATION=300 OUTPUT_DIR="$OUTPUT_BASE/stress/quick" \
        ./scripts/concurrent_stress.sh production
    
    # Swarmlet basic test (5 minutes)
    echo -e "\n${YELLOW}[4/4] Swarmlet join test...${NC}"
    if [ -n "$RPI_NODES" ]; then
        OUTPUT_DIR="$OUTPUT_BASE/swarmlet" ./scripts/swarmlet_join_bench.sh
    else
        echo "Skipping swarmlet tests (no RPI_NODES configured)"
    fi
}

# Full mode benchmarks (4-6 hours)
run_full_benchmarks() {
    echo -e "\n${BLUE}Running Full Benchmark Suite${NC}"
    
    # Complete single-node tests (1 hour)
    echo -e "\n${YELLOW}[1/5] Complete single-node benchmarks...${NC}"
    OUTPUT_DIR="$OUTPUT_BASE/single_node" ./scripts/run_single_node_suite.sh
    
    # All component benchmarks (1 hour)
    echo -e "\n${YELLOW}[2/5] All component benchmarks...${NC}"
    OUTPUT_DIR="$OUTPUT_BASE/components" ./scripts/run_component_benchmarks.sh
    
    # Swarmlet scaling tests (30 minutes)
    echo -e "\n${YELLOW}[3/5] Swarmlet scaling tests...${NC}"
    if [ -n "$RPI_NODES" ]; then
        OUTPUT_DIR="$OUTPUT_BASE/swarmlet" ./scripts/swarmlet_scaling_suite.sh
    else
        echo "Skipping swarmlet tests (no RPI_NODES configured)"
    fi
    
    # Comprehensive stress tests (2 hours)
    echo -e "\n${YELLOW}[4/5] Comprehensive stress tests...${NC}"
    for scenario in production max-throughput; do
        echo -e "\n  Running $scenario scenario..."
        DURATION=3600 OUTPUT_DIR="$OUTPUT_BASE/stress/$scenario" \
            ./scripts/concurrent_stress.sh $scenario
    done
    
    # Long-term stability test (1 hour sample)
    echo -e "\n${YELLOW}[5/5] Stability test sample...${NC}"
    DURATION=3600 OUTPUT_DIR="$OUTPUT_BASE/stress/stability" \
        ./scripts/concurrent_stress.sh stability
}

# CI mode benchmarks (for automated testing)
run_ci_benchmarks() {
    echo -e "\n${BLUE}Running CI Benchmark Suite${NC}"
    
    # Fast validation tests
    ITERATIONS=10 DURATION=10 ./scripts/gpu_consensus_bench.sh
    
    # Check for performance regressions
    if [ -f "results/baseline/performance.json" ]; then
        ./scripts/check_regressions.sh \
            --baseline results/baseline/performance.json \
            --current "$OUTPUT_BASE/performance.json" \
            --threshold 5
    fi
}

# Parallel execution helper
run_parallel() {
    local pids=()
    
    # Start benchmarks in parallel
    OUTPUT_DIR="$OUTPUT_BASE/single_node/consensus" ./scripts/gpu_consensus_bench.sh &
    pids+=($!)
    
    OUTPUT_DIR="$OUTPUT_BASE/single_node/synthesis" ./scripts/synthesis_throughput_bench.sh &
    pids+=($!)
    
    # Wait for all to complete
    for pid in "${pids[@]}"; do
        wait $pid || echo "Process $pid failed"
    done
}

# Generate consolidated report
generate_report() {
    echo -e "\n${YELLOW}Generating consolidated benchmark report...${NC}"
    
    python3 scripts/generate_benchmark_report.py \
        --input "$OUTPUT_BASE" \
        --output "$OUTPUT_BASE/benchmark_report.html" \
        --format html
    
    # Generate summary for console
    python3 scripts/generate_benchmark_report.py \
        --input "$OUTPUT_BASE" \
        --output "$OUTPUT_BASE/summary.txt" \
        --format text
    
    cat "$OUTPUT_BASE/summary.txt"
}

# Check prerequisites
check_prerequisites() {
    local missing=()
    
    # Check for required binaries
    for cmd in nvidia-smi cargo python3 jq tmux; do
        if ! command -v $cmd &> /dev/null; then
            missing+=($cmd)
        fi
    done
    
    # Check for benchmark binaries
    if [ ! -f "../../target/release/benchmark" ]; then
        echo "Building benchmark binaries..."
        cd ../../
        cargo build --release --bins
        cd benchmarks/scripts
    fi
    
    if [ ${#missing[@]} -gt 0 ]; then
        echo -e "${RED}Missing prerequisites: ${missing[*]}${NC}"
        exit 1
    fi
}

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    
    # Kill any remaining benchmark processes
    pkill -f "gpu_consensus_bench" 2>/dev/null || true
    pkill -f "synthesis_bench" 2>/dev/null || true
    pkill -f "stress_test" 2>/dev/null || true
    
    # Reset system settings
    sudo nvidia-smi -rgc 2>/dev/null || true
    sudo cpupower frequency-set -g ondemand 2>/dev/null || true
}

# Set cleanup trap
trap cleanup EXIT INT TERM

# Main execution
check_prerequisites
save_system_info

# Run benchmarks based on mode
case $MODE in
    quick)
        time run_quick_benchmarks
        ;;
    full)
        time run_full_benchmarks
        ;;
    ci)
        run_ci_benchmarks
        ;;
    parallel)
        PARALLEL=true
        time run_quick_benchmarks
        ;;
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Valid modes: quick, full, ci, parallel"
        exit 1
        ;;
esac

# Generate report
generate_report

# Final summary
echo -e "\n${GREEN}Benchmark Suite Complete!${NC}"
echo "================================"
echo "Total Duration: $(date -d@$SECONDS -u +%H:%M:%S)"
echo "Results: $OUTPUT_BASE"
echo "Report: $OUTPUT_BASE/benchmark_report.html"

# Check for failures
if grep -q "FAIL\|ERROR" "$OUTPUT_BASE/summary.txt"; then
    echo -e "\n${RED}⚠️  Some benchmarks failed or showed errors${NC}"
    exit 1
else
    echo -e "\n${GREEN}✅ All benchmarks completed successfully${NC}"
fi