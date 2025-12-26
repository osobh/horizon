#!/bin/bash
# Basic system validation script
# Tests that all components are working at a basic level

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}StratoSwarm Basic System Validation${NC}"
echo "==================================="

# Test 1: Kernel modules
echo -e "\n${YELLOW}Test 1: Kernel Modules${NC}"
modules_loaded=0
for module in gpu_dma_simple swarm_guard_simple tier_watch_simple; do
    if lsmod | grep -q "$module"; then
        echo -e "✅ $module loaded"
        ((modules_loaded++))
    else
        echo -e "❌ $module not loaded"
    fi
done
echo "Modules loaded: $modules_loaded/3"

# Test 2: Proc interfaces
echo -e "\n${YELLOW}Test 2: Proc Interfaces${NC}"
proc_tests=0
if [ -f "/proc/swarm/agents" ]; then
    echo -e "✅ /proc/swarm/agents accessible"
    echo "Sample: $(head -1 /proc/swarm/agents)"
    ((proc_tests++))
fi

if [ -f "/proc/swarm/gpu/devices" ]; then
    echo -e "✅ /proc/swarm/gpu/devices accessible"
    echo "Sample: $(head -1 /proc/swarm/gpu/devices)"
    ((proc_tests++))
fi

echo "Proc interfaces working: $proc_tests/2"

# Test 3: GPU Detection
echo -e "\n${YELLOW}Test 3: GPU Detection${NC}"
if nvidia-smi &> /dev/null; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo -e "✅ GPU detected: $gpu_name"
    echo -e "✅ GPU memory: $gpu_memory"
    gpu_test=1
else
    echo -e "❌ No GPU detected"
    gpu_test=0
fi

# Test 4: Build system
echo -e "\n${YELLOW}Test 4: Build System${NC}"
build_tests=0
if [ -f "../target/release/benchmark" ]; then
    echo -e "✅ Benchmark binary exists"
    ((build_tests++))
fi

if [ -f "../target/release/monitor" ]; then
    echo -e "✅ Monitor binary exists"
    ((build_tests++))
fi

if [ -f "../target/release/storage-benchmark" ]; then
    echo -e "✅ Storage benchmark binary exists"
    ((build_tests++))
fi

echo "Build artifacts: $build_tests/3"

# Test 5: Basic storage test
echo -e "\n${YELLOW}Test 5: Basic Storage Test${NC}"
if timeout 10 ../target/release/storage-benchmark --quick 2>&1 | head -5; then
    echo -e "✅ Storage benchmark runs"
    storage_test=1
else
    echo -e "❌ Storage benchmark failed"
    storage_test=0
fi

# Test 6: System resources
echo -e "\n${YELLOW}Test 6: System Resources${NC}"
cpu_cores=$(nproc)
memory_gb=$(free -g | awk '/^Mem:/{print $2}')
disk_free=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')

echo "CPU cores: $cpu_cores"
echo "Memory: ${memory_gb}GB"
echo "Free disk: ${disk_free}GB"

resource_ok=1
if [ $cpu_cores -lt 4 ]; then
    echo -e "⚠️  Warning: Less than 4 CPU cores"
    resource_ok=0
fi

if [ $memory_gb -lt 16 ]; then
    echo -e "⚠️  Warning: Less than 16GB memory"
    resource_ok=0
fi

if [ $disk_free -lt 50 ]; then
    echo -e "⚠️  Warning: Less than 50GB free disk"
    resource_ok=0
fi

if [ $resource_ok -eq 1 ]; then
    echo -e "✅ System resources adequate"
fi

# Summary
echo -e "\n${GREEN}Validation Summary${NC}"
echo "=================="
total_score=$((modules_loaded + proc_tests + gpu_test + build_tests + storage_test + resource_ok))
max_score=12

echo "Overall Score: $total_score/$max_score"

if [ $total_score -eq $max_score ]; then
    echo -e "${GREEN}🎉 All tests passed! System ready for benchmarking.${NC}"
    exit 0
elif [ $total_score -ge 8 ]; then
    echo -e "${YELLOW}⚠️  Most tests passed. System mostly functional.${NC}"
    exit 0
else
    echo -e "${RED}❌ Multiple test failures. System needs attention.${NC}"
    exit 1
fi