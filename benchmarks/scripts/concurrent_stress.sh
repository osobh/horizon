#!/bin/bash
# Concurrent Stress Testing Script
# Runs multiple StratoSwarm components simultaneously to identify bottlenecks

set -e

# Configuration
SCENARIO=${1:-production}  # production, max-throughput, or stability
DURATION=${DURATION:-3600} # 1 hour default
OUTPUT_DIR=${OUTPUT_DIR:-"results/stress_${SCENARIO}_$(date +%Y%m%d_%H%M%S)"}
COMPONENT_PIDS=()

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}StratoSwarm Concurrent Stress Test${NC}"
echo "=================================="
echo "Scenario: $SCENARIO"
echo "Duration: ${DURATION}s"
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directory structure
mkdir -p "$OUTPUT_DIR"/{logs,metrics,checkpoints}

# Check system resources
check_resources() {
    echo -e "${YELLOW}Checking system resources...${NC}"
    
    # CPU cores
    CPU_CORES=$(nproc)
    echo "CPU Cores: $CPU_CORES"
    
    # Memory
    MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
    echo "Memory: ${MEM_GB}GB"
    
    # GPU
    if nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        echo "GPU: $GPU_NAME (${GPU_MEM}MB)"
    else
        echo -e "${RED}Warning: No GPU detected${NC}"
    fi
    
    # Disk space
    DISK_FREE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    echo "Free Disk: ${DISK_FREE}GB"
    
    if [ $DISK_FREE -lt 50 ]; then
        echo -e "${RED}Error: Insufficient disk space (<50GB)${NC}"
        exit 1
    fi
}

# Prepare system for stress test
prepare_system() {
    echo -e "\n${YELLOW}Preparing system for stress test...${NC}"
    
    # Set CPU governor to performance
    if command -v cpupower &> /dev/null; then
        sudo cpupower frequency-set -g performance 2>/dev/null || true
    fi
    
    # Increase file descriptor limits
    ulimit -n 65536
    
    # Set GPU to maximum performance
    if nvidia-smi &> /dev/null; then
        sudo nvidia-smi -pm 1
        sudo nvidia-smi -lgc $(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader | head -1) 2>/dev/null || true
    fi
    
    # Clear caches
    sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
}

# Start monitoring infrastructure
start_monitoring() {
    echo -e "\n${YELLOW}Starting monitoring infrastructure...${NC}"
    
    # System metrics
    sar -A -o "$OUTPUT_DIR/metrics/sar.dat" 1 > /dev/null 2>&1 &
    COMPONENT_PIDS+=($!)
    
    # GPU metrics
    if nvidia-smi &> /dev/null; then
        nvidia-smi dmon -s pucvmet -i 0 -d 1 -f "$OUTPUT_DIR/metrics/gpu_metrics.csv" &
        COMPONENT_PIDS+=($!)
    fi
    
    # StratoSwarm metrics collector
    ../target/release/metrics_collector \
        --interval 1 \
        --output "$OUTPUT_DIR/metrics/stratoswarm_metrics.jsonl" &
    COMPONENT_PIDS+=($!)
    
    # Real-time dashboard (optional)
    if [ -t 1 ]; then  # If running in terminal
        tmux new-session -d -s stress-monitor \
            "watch -n 1 'tail -20 $OUTPUT_DIR/metrics/stratoswarm_metrics.jsonl | jq .'"
        echo "Monitor dashboard: tmux attach -t stress-monitor"
    fi
}

# Production workload scenario
run_production_scenario() {
    echo -e "\n${BLUE}Running Production Workload Scenario${NC}"
    
    # GPU Consensus workload
    echo "Starting GPU Consensus workload..."
    ../target/release/consensus_stress \
        --rate 100 \
        --duration $DURATION \
        --output "$OUTPUT_DIR/logs/consensus.log" &
    COMPONENT_PIDS+=($!)
    
    # Memory tier migrations
    echo "Starting Memory Migration workload..."
    ../target/release/memory_stress \
        --migrations-per-sec 50 \
        --size-range "1MB-100MB" \
        --duration $DURATION \
        --output "$OUTPUT_DIR/logs/memory.log" &
    COMPONENT_PIDS+=($!)
    
    # Knowledge Graph queries
    echo "Starting Knowledge Graph workload..."
    ../target/release/knowledge_graph_stress \
        --queries-per-sec 100 \
        --graph-size 1000000 \
        --duration $DURATION \
        --output "$OUTPUT_DIR/logs/knowledge_graph.log" &
    COMPONENT_PIDS+=($!)
    
    # AI Assistant requests
    echo "Starting AI Assistant workload..."
    ../target/release/ai_assistant_stress \
        --requests-per-sec 10 \
        --request-types "parse,generate,learn" \
        --duration $DURATION \
        --output "$OUTPUT_DIR/logs/ai_assistant.log" &
    COMPONENT_PIDS+=($!)
    
    # Streaming pipeline at 50% capacity
    echo "Starting Streaming Pipeline..."
    ../target/release/streaming_stress \
        --throughput "50%" \
        --compression enabled \
        --duration $DURATION \
        --output "$OUTPUT_DIR/logs/streaming.log" &
    COMPONENT_PIDS+=($!)
    
    # Agent lifecycle management
    echo "Starting Agent Lifecycle workload..."
    ../target/release/agent_lifecycle_stress \
        --spawn-rate 5 \
        --destroy-rate 3 \
        --target-population "1000-2000" \
        --duration $DURATION \
        --output "$OUTPUT_DIR/logs/agents.log" &
    COMPONENT_PIDS+=($!)
}

# Maximum throughput scenario
run_max_throughput_scenario() {
    echo -e "\n${BLUE}Running Maximum Throughput Scenario${NC}"
    
    # All components at maximum rate
    local components=(
        "consensus_stress --rate max --threads 32"
        "synthesis_stress --batch-size 10000 --threads 64"
        "memory_stress --allocations-per-sec 10000"
        "knowledge_graph_stress --concurrent-queries 1000"
        "evolution_stress --population 10000 --generations-per-sec 100"
        "streaming_stress --throughput 10GB/s"
    )
    
    for cmd in "${components[@]}"; do
        component=$(echo $cmd | cut -d' ' -f1)
        echo "Starting $component at maximum rate..."
        ../target/release/$cmd \
            --duration $DURATION \
            --output "$OUTPUT_DIR/logs/${component}.log" &
        COMPONENT_PIDS+=($!)
        sleep 1  # Stagger starts slightly
    done
}

# Long-running stability scenario
run_stability_scenario() {
    echo -e "\n${BLUE}Running Stability Test Scenario${NC}"
    
    # Moderate, sustainable load
    echo "Starting long-term stability workload..."
    ../target/release/stability_test \
        --config configs/stability_workload.yaml \
        --checkpoint-interval 300 \
        --checkpoint-dir "$OUTPUT_DIR/checkpoints" \
        --duration $DURATION \
        --output "$OUTPUT_DIR/logs/stability.log" &
    COMPONENT_PIDS+=($!)
}

# Monitor stress test progress
monitor_progress() {
    local start_time=$(date +%s)
    local last_checkpoint=$start_time
    
    while true; do
        sleep 30
        
        # Check if all processes are still running
        local running=0
        for pid in "${COMPONENT_PIDS[@]}"; do
            if kill -0 $pid 2>/dev/null; then
                ((running++))
            fi
        done
        
        local elapsed=$(($(date +%s) - start_time))
        echo -e "\n[$(date '+%H:%M:%S')] Progress: ${elapsed}s elapsed, $running/${#COMPONENT_PIDS[@]} components running"
        
        # Quick health check
        if [ -f "$OUTPUT_DIR/metrics/stratoswarm_metrics.jsonl" ]; then
            tail -1 "$OUTPUT_DIR/metrics/stratoswarm_metrics.jsonl" | jq -r '
                "CPU: \(.system.cpu_percent)%, " +
                "Memory: \(.system.memory_percent)%, " +
                "GPU: \(.gpu.utilization)%, " +
                "Errors: \(.errors.total)"
            ' 2>/dev/null || echo "Metrics unavailable"
        fi
        
        # Checkpoint every 5 minutes
        if [ $(($(date +%s) - last_checkpoint)) -gt 300 ]; then
            echo "Creating checkpoint..."
            ./create_checkpoint.sh "$OUTPUT_DIR/checkpoints/checkpoint_$(date +%s).tar.gz"
            last_checkpoint=$(date +%s)
        fi
        
        # Check if duration exceeded
        if [ $elapsed -ge $DURATION ]; then
            break
        fi
    done
}

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Stopping stress test components...${NC}"
    
    # Stop all component processes
    for pid in "${COMPONENT_PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            kill -TERM $pid 2>/dev/null || true
        fi
    done
    
    # Wait for graceful shutdown
    sleep 5
    
    # Force kill if needed
    for pid in "${COMPONENT_PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            kill -KILL $pid 2>/dev/null || true
        fi
    done
    
    # Stop tmux monitor if running
    tmux kill-session -t stress-monitor 2>/dev/null || true
    
    # Reset system settings
    sudo nvidia-smi -rgc 2>/dev/null || true
    sudo cpupower frequency-set -g ondemand 2>/dev/null || true
}

# Set cleanup trap
trap cleanup EXIT INT TERM

# Main execution
check_resources
prepare_system
start_monitoring

# Run selected scenario
case $SCENARIO in
    production)
        run_production_scenario
        ;;
    max-throughput)
        run_max_throughput_scenario
        ;;
    stability)
        run_stability_scenario
        ;;
    *)
        echo -e "${RED}Unknown scenario: $SCENARIO${NC}"
        echo "Valid scenarios: production, max-throughput, stability"
        exit 1
        ;;
esac

# Monitor progress
echo -e "\n${GREEN}Stress test started. Monitoring progress...${NC}"
monitor_progress

# Analyze results
echo -e "\n${YELLOW}Analyzing stress test results...${NC}"
python3 ../scripts/analyze_stress_results.py \
    --metrics-dir "$OUTPUT_DIR/metrics" \
    --logs-dir "$OUTPUT_DIR/logs" \
    --output "$OUTPUT_DIR/stress_test_report.html"

# Generate summary
echo -e "\n${GREEN}Stress Test Summary${NC}"
echo "==================="
python3 -c "
import json
import glob

# Load final metrics
metrics_files = sorted(glob.glob('$OUTPUT_DIR/metrics/stratoswarm_metrics.jsonl'))
if metrics_files:
    with open(metrics_files[-1]) as f:
        for line in f:
            pass  # Get last line
        final_metrics = json.loads(line)
    
    print(f\"Total Operations: {final_metrics.get('operations', {}).get('total', 'N/A')}\")
    print(f\"Error Rate: {final_metrics.get('errors', {}).get('rate', 'N/A')}%\")
    print(f\"Peak CPU: {final_metrics.get('system', {}).get('cpu_peak', 'N/A')}%\")
    print(f\"Peak Memory: {final_metrics.get('system', {}).get('memory_peak', 'N/A')}%\")
    print(f\"Peak GPU: {final_metrics.get('gpu', {}).get('utilization_peak', 'N/A')}%\")

# Check for bottlenecks
bottlenecks = []
if final_metrics.get('system', {}).get('cpu_peak', 0) > 90:
    bottlenecks.append('CPU')
if final_metrics.get('system', {}).get('memory_peak', 0) > 90:
    bottlenecks.append('Memory')
if final_metrics.get('gpu', {}).get('utilization_peak', 0) < 70:
    bottlenecks.append('GPU underutilized')

if bottlenecks:
    print(f\"\nBottlenecks detected: {', '.join(bottlenecks)}\")
else:
    print(\"\n✅ No major bottlenecks detected\")
"

echo -e "\n${GREEN}Stress test complete!${NC}"
echo "Full report: $OUTPUT_DIR/stress_test_report.html"
echo "Logs: $OUTPUT_DIR/logs/"
echo "Metrics: $OUTPUT_DIR/metrics/"