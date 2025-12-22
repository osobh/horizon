#!/bin/bash
# StratoSwarm Benchmark Environment Setup and Verification
# This script prepares the system from scratch for benchmarking

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}StratoSwarm Benchmark Setup & Verification${NC}"
echo "=========================================="
echo "This script will:"
echo "1. Build kernel modules"
echo "2. Load and verify kernel modules"
echo "3. Build benchmark binaries"
echo "4. Verify GPU setup"
echo "5. Run basic validation tests"
echo ""

# Check if running as root when needed
check_root() {
    if [ "$EUID" -ne 0 ]; then 
        echo -e "${YELLOW}Some operations require root. You may be prompted for sudo password.${NC}"
    fi
}

# Step 1: Build Kernel Modules
build_kernel_modules() {
    echo -e "\n${BLUE}Step 1: Building Kernel Modules${NC}"
    
    if [ ! -d "../crates/kernel-modules" ]; then
        echo -e "${RED}Error: kernel-modules directory not found${NC}"
        echo "Expected location: ../crates/kernel-modules"
        return 1
    fi
    
    cd ../crates/kernel-modules
    
    # Check for kernel headers
    if [ ! -d "/lib/modules/$(uname -r)/build" ]; then
        echo -e "${YELLOW}Installing kernel headers...${NC}"
        sudo apt-get update
        sudo apt-get install -y linux-headers-$(uname -r) build-essential
    fi
    
    # Build each module
    for module in swarm_guard tier_watch gpu_dma_lock; do
        echo -e "\n${YELLOW}Building $module...${NC}"
        
        if [ -d "$module" ]; then
            cd $module
            
            # Clean previous builds
            make clean 2>/dev/null || true
            
            # Build module
            make
            
            if [ -f "${module}.ko" ]; then
                echo -e "${GREEN}✓ $module built successfully${NC}"
                # Copy to standard location
                sudo cp ${module}.ko /lib/modules/$(uname -r)/extra/ 2>/dev/null || \
                    sudo mkdir -p /lib/modules/$(uname -r)/extra/ && \
                    sudo cp ${module}.ko /lib/modules/$(uname -r)/extra/
            else
                echo -e "${RED}✗ Failed to build $module${NC}"
            fi
            
            cd ..
        else
            echo -e "${RED}Module directory not found: $module${NC}"
        fi
    done
    
    # Update module dependencies
    sudo depmod -a
    
    cd ../../benchmarks
}

# Step 2: Load and Verify Kernel Modules
load_kernel_modules() {
    echo -e "\n${BLUE}Step 2: Loading Kernel Modules${NC}"
    
    local modules=("swarm_guard" "tier_watch" "gpu_dma_lock")
    local loaded=0
    
    for module in "${modules[@]}"; do
        echo -e "\n${YELLOW}Loading $module...${NC}"
        
        # Check if already loaded
        if lsmod | grep -q "^$module"; then
            echo -e "${GREEN}✓ $module already loaded${NC}"
            ((loaded++))
            continue
        fi
        
        # Try to load module
        if sudo modprobe $module 2>/dev/null; then
            echo -e "${GREEN}✓ $module loaded successfully${NC}"
            ((loaded++))
        else
            # Try alternative loading method
            if [ -f "/lib/modules/$(uname -r)/extra/${module}.ko" ]; then
                if sudo insmod "/lib/modules/$(uname -r)/extra/${module}.ko" 2>/dev/null; then
                    echo -e "${GREEN}✓ $module loaded via insmod${NC}"
                    ((loaded++))
                else
                    echo -e "${RED}✗ Failed to load $module${NC}"
                    dmesg | tail -5
                fi
            else
                echo -e "${RED}✗ Module file not found: ${module}.ko${NC}"
            fi
        fi
    done
    
    echo -e "\n${YELLOW}Loaded $loaded/${#modules[@]} modules${NC}"
    
    # Show loaded modules
    echo -e "\n${YELLOW}Currently loaded StratoSwarm modules:${NC}"
    lsmod | grep swarm || echo "No swarm modules loaded"
    
    return $((${#modules[@]} - loaded))
}

# Step 3: Verify Kernel Module Functionality
verify_kernel_modules() {
    echo -e "\n${BLUE}Step 3: Verifying Kernel Module Functionality${NC}"
    
    # Check /proc/swarm interface
    echo -e "\n${YELLOW}Checking /proc/swarm interface...${NC}"
    if [ -d "/proc/swarm" ]; then
        echo -e "${GREEN}✓ /proc/swarm exists${NC}"
        
        # List contents
        echo "Contents:"
        ls -la /proc/swarm/ 2>/dev/null || echo "Unable to list contents"
        
        # Check specific interfaces
        for interface in agents tiers gpu; do
            if [ -e "/proc/swarm/$interface" ]; then
                echo -e "${GREEN}✓ /proc/swarm/$interface exists${NC}"
            else
                echo -e "${YELLOW}✗ /proc/swarm/$interface not found${NC}"
            fi
        done
    else
        echo -e "${RED}✗ /proc/swarm not found - modules may not be loaded correctly${NC}"
    fi
    
    # Check tier_watch specific interface
    echo -e "\n${YELLOW}Checking tier_watch functionality...${NC}"
    if [ -f "/proc/swarm/tiers/stats" ]; then
        echo -e "${GREEN}✓ Tier statistics available${NC}"
        echo "Sample stats:"
        head -5 /proc/swarm/tiers/stats 2>/dev/null || echo "Unable to read stats"
    fi
    
    # Check GPU module interface
    echo -e "\n${YELLOW}Checking gpu_dma_lock functionality...${NC}"
    if [ -f "/proc/swarm/gpu/devices" ]; then
        echo -e "${GREEN}✓ GPU devices interface available${NC}"
        cat /proc/swarm/gpu/devices 2>/dev/null || echo "Unable to read GPU devices"
    fi
}

# Step 4: Build Benchmark Binaries
build_benchmarks() {
    echo -e "\n${BLUE}Step 4: Building Benchmark Binaries${NC}"
    
    # Build GPU agents benchmarks
    echo -e "\n${YELLOW}Building GPU agent benchmarks...${NC}"
    cd ../crates/gpu-agents
    
    # Check for CUDA
    if ! command -v nvcc &> /dev/null; then
        echo -e "${RED}Warning: CUDA not found. GPU benchmarks may not work.${NC}"
        echo "Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads"
    else
        echo "CUDA version: $(nvcc --version | grep release | awk '{print $6}')"
    fi
    
    # Build with features
    cargo build --release --features cuda --bin gpu_consensus_bench 2>&1 | grep -E "(Compiling|Finished|error)" || true
    cargo build --release --features cuda --bin gpu_synthesis_bench 2>&1 | grep -E "(Compiling|Finished|error)" || true
    
    # Build other benchmark binaries
    echo -e "\n${YELLOW}Building stress test binaries...${NC}"
    cargo build --release --bin consensus_stress 2>&1 | grep -E "(Compiling|Finished|error)" || true
    cargo build --release --bin memory_stress 2>&1 | grep -E "(Compiling|Finished|error)" || true
    cargo build --release --bin metrics_collector 2>&1 | grep -E "(Compiling|Finished|error)" || true
    
    cd ../../benchmarks
    
    # Verify binaries exist
    echo -e "\n${YELLOW}Verifying benchmark binaries...${NC}"
    local binaries=(
        "gpu_consensus_bench"
        "gpu_synthesis_bench"
        "consensus_stress"
        "memory_stress"
        "metrics_collector"
    )
    
    local found=0
    for binary in "${binaries[@]}"; do
        if [ -f "../target/release/$binary" ]; then
            echo -e "${GREEN}✓ $binary found${NC}"
            ((found++))
        else
            echo -e "${RED}✗ $binary not found${NC}"
        fi
    done
    
    echo -e "\n${YELLOW}Found $found/${#binaries[@]} benchmark binaries${NC}"
}

# Step 5: Verify GPU Setup
verify_gpu_setup() {
    echo -e "\n${BLUE}Step 5: Verifying GPU Setup${NC}"
    
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}✗ nvidia-smi not found - NVIDIA drivers may not be installed${NC}"
        return 1
    fi
    
    # Check GPU
    if nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
        
        # Check CUDA
        if [ -f "/usr/local/cuda/version.txt" ]; then
            echo -e "${GREEN}✓ CUDA installed${NC}"
            cat /usr/local/cuda/version.txt
        elif command -v nvcc &> /dev/null; then
            echo -e "${GREEN}✓ CUDA compiler found${NC}"
            nvcc --version | grep release
        else
            echo -e "${YELLOW}⚠ CUDA toolkit not found - GPU benchmarks may fail${NC}"
        fi
    else
        echo -e "${RED}✗ No NVIDIA GPU accessible${NC}"
        return 1
    fi
}

# Step 6: Run Basic Validation
run_basic_validation() {
    echo -e "\n${BLUE}Step 6: Running Basic Validation Tests${NC}"
    
    # Test 1: Memory allocation via kernel module
    echo -e "\n${YELLOW}Test 1: Kernel module memory test...${NC}"
    if [ -f "/proc/swarm/agents" ]; then
        # Try to create a test agent
        echo "test_agent" | sudo tee /proc/swarm/agents/create > /dev/null 2>&1 && \
            echo -e "${GREEN}✓ Agent creation via kernel module works${NC}" || \
            echo -e "${YELLOW}⚠ Agent creation failed (may be normal)${NC}"
    fi
    
    # Test 2: GPU allocation test
    echo -e "\n${YELLOW}Test 2: Basic GPU allocation test...${NC}"
    if [ -f "../target/release/gpu_consensus_bench" ]; then
        timeout 5 ../target/release/gpu_consensus_bench --iterations 1 --quick-test 2>&1 | \
            grep -q "success" && echo -e "${GREEN}✓ GPU allocation works${NC}" || \
            echo -e "${YELLOW}⚠ GPU allocation test failed${NC}"
    fi
    
    # Test 3: Metrics collection
    echo -e "\n${YELLOW}Test 3: Metrics collection test...${NC}"
    if [ -f "../target/release/metrics_collector" ]; then
        timeout 5 ../target/release/metrics_collector --test 2>&1 | \
            grep -q "ok" && echo -e "${GREEN}✓ Metrics collection works${NC}" || \
            echo -e "${YELLOW}⚠ Metrics collection failed${NC}"
    fi
}

# Generate setup report
generate_setup_report() {
    local report_file="results/setup_report_$(date +%Y%m%d_%H%M%S).txt"
    mkdir -p results
    
    {
        echo "StratoSwarm Benchmark Setup Report"
        echo "=================================="
        echo "Date: $(date)"
        echo ""
        echo "System Information:"
        echo "- Kernel: $(uname -r)"
        echo "- CPU: $(lscpu | grep "Model name" | cut -d: -f2 | xargs)"
        echo "- Memory: $(free -h | awk '/^Mem:/{print $2}')"
        echo ""
        echo "Kernel Modules:"
        lsmod | grep swarm || echo "No swarm modules loaded"
        echo ""
        echo "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "No GPU found"
        echo ""
        echo "Benchmark Binaries:"
        ls -la ../target/release/*_bench ../target/release/*_stress 2>/dev/null | awk '{print $9}' || echo "No binaries found"
    } | tee "$report_file"
    
    echo -e "\n${GREEN}Setup report saved to: $report_file${NC}"
}

# Main execution
main() {
    check_root
    
    # Create results directory
    mkdir -p results
    
    # Run setup steps
    build_kernel_modules
    load_kernel_modules
    verify_kernel_modules
    build_benchmarks
    verify_gpu_setup
    run_basic_validation
    
    # Generate report
    generate_setup_report
    
    echo -e "\n${GREEN}Setup Complete!${NC}"
    echo "=============================="
    echo "You can now run benchmarks with:"
    echo "  ./scripts/run_all_benchmarks.sh --quick    # 30-minute validation"
    echo "  ./scripts/gpu_consensus_bench.sh           # GPU consensus test"
    echo "  ./scripts/concurrent_stress.sh production  # Stress test"
    
    # Check for critical failures
    if ! lsmod | grep -q swarm; then
        echo -e "\n${RED}⚠️  Warning: No kernel modules loaded. Benchmarks may fail.${NC}"
        echo "Try building modules manually in: ../crates/kernel-modules/"
        exit 1
    fi
    
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "\n${RED}⚠️  Warning: No GPU detected. GPU benchmarks will fail.${NC}"
        exit 1
    fi
}

# Run main
main "$@"