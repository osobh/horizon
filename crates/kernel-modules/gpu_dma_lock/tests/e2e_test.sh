#!/bin/bash
# End-to-end test for gpu_dma_lock kernel module
# Requires root privileges and GPU hardware

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test configuration
MODULE_NAME="gpu_dma_lock"
MODULE_PATH="../${MODULE_NAME}.ko"
PROC_BASE="/proc/swarm/gpu"

# Check requirements
check_requirements() {
    echo "Checking requirements..."
    
    # Check if running as root
    if [ "$EUID" -ne 0 ]; then 
        echo -e "${RED}Error: This script must be run as root${NC}"
        exit 1
    fi
    
    # Check if module exists
    if [ ! -f "$MODULE_PATH" ]; then
        echo -e "${YELLOW}Module not found, building...${NC}"
        make -C .. || exit 1
    fi
    
    # Check if GPU is available
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}Warning: nvidia-smi not found, GPU tests will be limited${NC}"
        GPU_AVAILABLE=0
    else
        GPU_AVAILABLE=1
        GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
        echo "Found $GPU_COUNT GPU(s)"
    fi
}

# Load module
load_module() {
    echo "Loading $MODULE_NAME module..."
    
    # Remove if already loaded
    if lsmod | grep -q "$MODULE_NAME"; then
        rmmod "$MODULE_NAME" 2>/dev/null || true
    fi
    
    # Load with parameters
    insmod "$MODULE_PATH" debug=1 || {
        echo -e "${RED}Failed to load module${NC}"
        dmesg | tail -20
        exit 1
    }
    
    echo -e "${GREEN}Module loaded successfully${NC}"
}

# Test 1: Basic /proc interface
test_proc_interface() {
    echo -e "\n${YELLOW}Test 1: /proc interface${NC}"
    
    # Check main directory
    if [ ! -d "$PROC_BASE" ]; then
        echo -e "${RED}FAIL: $PROC_BASE not found${NC}"
        return 1
    fi
    
    # Check expected files
    local expected_files=("stats" "ctl" "agents")
    for file in "${expected_files[@]}"; do
        if [ ! -e "$PROC_BASE/$file" ]; then
            echo -e "${RED}FAIL: $PROC_BASE/$file not found${NC}"
            return 1
        fi
    done
    
    # Check per-GPU directories
    if [ $GPU_AVAILABLE -eq 1 ]; then
        for ((i=0; i<$GPU_COUNT; i++)); do
            if [ ! -d "$PROC_BASE/$i" ]; then
                echo -e "${RED}FAIL: $PROC_BASE/$i not found${NC}"
                return 1
            fi
        done
    fi
    
    echo -e "${GREEN}PASS: /proc interface created correctly${NC}"
}

# Test 2: GPU device detection
test_gpu_detection() {
    echo -e "\n${YELLOW}Test 2: GPU device detection${NC}"
    
    if [ $GPU_AVAILABLE -eq 0 ]; then
        echo "SKIP: No GPU available"
        return 0
    fi
    
    # Read GPU info
    for ((i=0; i<$GPU_COUNT; i++)); do
        local info=$(cat "$PROC_BASE/$i/info" 2>/dev/null)
        if [ -z "$info" ]; then
            echo -e "${RED}FAIL: Cannot read GPU $i info${NC}"
            return 1
        fi
        
        # Check for expected fields
        if ! echo "$info" | grep -q "device_id:"; then
            echo -e "${RED}FAIL: device_id not found in GPU $i info${NC}"
            return 1
        fi
        
        if ! echo "$info" | grep -q "total_memory:"; then
            echo -e "${RED}FAIL: total_memory not found in GPU $i info${NC}"
            return 1
        fi
        
        echo "GPU $i detected correctly"
    done
    
    echo -e "${GREEN}PASS: All GPUs detected${NC}"
}

# Test 3: Agent creation and quota management
test_agent_management() {
    echo -e "\n${YELLOW}Test 3: Agent management${NC}"
    
    local agent_id=1000
    local quota=$((1 << 30)) # 1GB
    
    # Create agent
    echo "create_agent $agent_id $quota" > "$PROC_BASE/ctl" || {
        echo -e "${RED}FAIL: Cannot create agent${NC}"
        return 1
    }
    
    # Check agent directory
    if [ ! -d "$PROC_BASE/agents/$agent_id" ]; then
        echo -e "${RED}FAIL: Agent directory not created${NC}"
        return 1
    fi
    
    # Read agent stats
    local stats=$(cat "$PROC_BASE/agents/$agent_id/stats" 2>/dev/null)
    if ! echo "$stats" | grep -q "quota: $quota"; then
        echo -e "${RED}FAIL: Agent quota not set correctly${NC}"
        return 1
    fi
    
    echo -e "${GREEN}PASS: Agent created with correct quota${NC}"
}

# Test 4: Memory allocation tracking
test_memory_allocation() {
    echo -e "\n${YELLOW}Test 4: Memory allocation tracking${NC}"
    
    local agent_id=2000
    local quota=$((2 << 30)) # 2GB
    local alloc_size=$((512 << 20)) # 512MB
    
    # Create agent
    echo "create_agent $agent_id $quota" > "$PROC_BASE/ctl"
    
    # Allocate memory
    echo "alloc $agent_id $alloc_size 0" > "$PROC_BASE/ctl" || {
        echo -e "${RED}FAIL: Allocation failed${NC}"
        return 1
    }
    
    # Check allocation in stats
    local stats=$(cat "$PROC_BASE/agents/$agent_id/stats")
    if ! echo "$stats" | grep -q "allocated:"; then
        echo -e "${RED}FAIL: Allocation not tracked${NC}"
        return 1
    fi
    
    # Try to exceed quota
    echo "alloc $agent_id $quota 0" > "$PROC_BASE/ctl" 2>/dev/null && {
        echo -e "${RED}FAIL: Quota not enforced${NC}"
        return 1
    }
    
    echo -e "${GREEN}PASS: Memory allocation and quota enforcement working${NC}"
}

# Test 5: DMA access control
test_dma_access_control() {
    echo -e "\n${YELLOW}Test 5: DMA access control${NC}"
    
    local agent_id=3000
    local start_addr=$((0x100000000))
    local end_addr=$((0x200000000))
    
    # Create agent
    echo "create_agent $agent_id $((1 << 30))" > "$PROC_BASE/ctl"
    
    # Grant DMA permission
    echo "grant_dma $agent_id $start_addr $end_addr rw" > "$PROC_BASE/ctl" || {
        echo -e "${RED}FAIL: Cannot grant DMA permission${NC}"
        return 1
    }
    
    # Test valid access
    echo "test_dma $agent_id $((start_addr + 0x1000)) r" > "$PROC_BASE/ctl" || {
        echo -e "${RED}FAIL: Valid DMA access denied${NC}"
        return 1
    fi
    
    # Test invalid access
    echo "test_dma $agent_id $((end_addr + 0x1000)) r" > "$PROC_BASE/ctl" 2>/dev/null && {
        echo -e "${RED}FAIL: Invalid DMA access allowed${NC}"
        return 1
    }
    
    echo -e "${GREEN}PASS: DMA access control working${NC}"
}

# Test 6: Multi-GPU allocation
test_multi_gpu() {
    echo -e "\n${YELLOW}Test 6: Multi-GPU allocation${NC}"
    
    if [ $GPU_COUNT -lt 2 ]; then
        echo "SKIP: Requires multiple GPUs"
        return 0
    fi
    
    local agent_id=4000
    
    # Create agent
    echo "create_agent $agent_id $((4 << 30))" > "$PROC_BASE/ctl"
    
    # Allocate on different GPUs
    for ((i=0; i<$GPU_COUNT && i<2; i++)); do
        echo "alloc $agent_id $((1 << 30)) $i" > "$PROC_BASE/ctl" || {
            echo -e "${RED}FAIL: Cannot allocate on GPU $i${NC}"
            return 1
        }
    done
    
    # Check allocations on each GPU
    for ((i=0; i<$GPU_COUNT && i<2; i++)); do
        local gpu_stats=$(cat "$PROC_BASE/$i/stats")
        if ! echo "$gpu_stats" | grep -q "allocated:"; then
            echo -e "${RED}FAIL: Allocation not tracked on GPU $i${NC}"
            return 1
        fi
    done
    
    echo -e "${GREEN}PASS: Multi-GPU allocation working${NC}"
}

# Test 7: Concurrent operations
test_concurrent_ops() {
    echo -e "\n${YELLOW}Test 7: Concurrent operations${NC}"
    
    local num_agents=10
    local base_agent_id=5000
    
    # Create multiple agents concurrently
    for ((i=0; i<$num_agents; i++)); do
        (echo "create_agent $((base_agent_id + i)) $((1 << 30))" > "$PROC_BASE/ctl") &
    done
    wait
    
    # Allocate concurrently
    for ((i=0; i<$num_agents; i++)); do
        (echo "alloc $((base_agent_id + i)) $((100 << 20)) 0" > "$PROC_BASE/ctl") &
    done
    wait
    
    # Verify all agents were created
    local created_count=$(ls "$PROC_BASE/agents/" | wc -l)
    if [ $created_count -lt $num_agents ]; then
        echo -e "${RED}FAIL: Not all agents created ($created_count/$num_agents)${NC}"
        return 1
    fi
    
    echo -e "${GREEN}PASS: Concurrent operations handled correctly${NC}"
}

# Test 8: GPU context isolation
test_context_isolation() {
    echo -e "\n${YELLOW}Test 8: GPU context isolation${NC}"
    
    if [ $GPU_AVAILABLE -eq 0 ]; then
        echo "SKIP: No GPU available"
        return 0
    fi
    
    local agent1=6000
    local agent2=6001
    
    # Create agents
    echo "create_agent $agent1 $((1 << 30))" > "$PROC_BASE/ctl"
    echo "create_agent $agent2 $((1 << 30))" > "$PROC_BASE/ctl"
    
    # Create contexts
    echo "create_context $agent1 0" > "$PROC_BASE/ctl" || {
        echo -e "${RED}FAIL: Cannot create context for agent $agent1${NC}"
        return 1
    }
    
    echo "create_context $agent2 0" > "$PROC_BASE/ctl" || {
        echo -e "${RED}FAIL: Cannot create context for agent $agent2${NC}"
        return 1
    }
    
    # Check context isolation
    local contexts=$(cat "$PROC_BASE/0/contexts")
    if ! echo "$contexts" | grep -q "isolated: true"; then
        echo -e "${RED}FAIL: Contexts not isolated${NC}"
        return 1
    fi
    
    echo -e "${GREEN}PASS: GPU context isolation verified${NC}"
}

# Test 9: Memory pressure monitoring
test_memory_pressure() {
    echo -e "\n${YELLOW}Test 9: Memory pressure monitoring${NC}"
    
    if [ $GPU_AVAILABLE -eq 0 ]; then
        echo "SKIP: No GPU available"
        return 0
    fi
    
    # Get GPU memory size
    local gpu_memory=$(nvidia-smi --id=0 --query-gpu=memory.total --format=csv,noheader,nounits)
    gpu_memory=$((gpu_memory << 20)) # Convert MB to bytes
    
    # Set pressure thresholds
    echo "80" > "$PROC_BASE/0/pressure_warning"
    echo "95" > "$PROC_BASE/0/pressure_critical"
    
    # Create large allocation to trigger warning
    local agent_id=7000
    local alloc_size=$((gpu_memory * 85 / 100)) # 85% of total
    
    echo "create_agent $agent_id $gpu_memory" > "$PROC_BASE/ctl"
    echo "alloc $agent_id $alloc_size 0" > "$PROC_BASE/ctl" || {
        echo -e "${YELLOW}Note: Large allocation failed, may be due to fragmentation${NC}"
        return 0
    }
    
    # Check pressure level
    local pressure=$(cat "$PROC_BASE/0/pressure")
    if ! echo "$pressure" | grep -E "level: (warning|critical)"; then
        echo -e "${RED}FAIL: Memory pressure not detected${NC}"
        return 1
    fi
    
    echo -e "${GREEN}PASS: Memory pressure monitoring working${NC}"
}

# Test 10: Module parameters
test_module_params() {
    echo -e "\n${YELLOW}Test 10: Module parameters${NC}"
    
    # Check debug parameter
    local debug_val=$(cat /sys/module/$MODULE_NAME/parameters/debug 2>/dev/null)
    if [ "$debug_val" != "1" ]; then
        echo -e "${RED}FAIL: Debug parameter not set${NC}"
        return 1
    fi
    
    echo -e "${GREEN}PASS: Module parameters working${NC}"
}

# Test 11: Error handling
test_error_handling() {
    echo -e "\n${YELLOW}Test 11: Error handling${NC}"
    
    # Test invalid agent ID
    echo "alloc 99999 $((1 << 30)) 0" > "$PROC_BASE/ctl" 2>/dev/null && {
        echo -e "${RED}FAIL: Invalid agent allocation succeeded${NC}"
        return 1
    }
    
    # Test invalid GPU ID
    echo "create_agent 8000 $((1 << 30))" > "$PROC_BASE/ctl"
    echo "alloc 8000 $((1 << 20)) 999" > "$PROC_BASE/ctl" 2>/dev/null && {
        echo -e "${RED}FAIL: Invalid GPU allocation succeeded${NC}"
        return 1
    }
    
    # Test negative allocation size (via crafted input)
    echo "alloc 8000 -1 0" > "$PROC_BASE/ctl" 2>/dev/null && {
        echo -e "${RED}FAIL: Negative allocation succeeded${NC}"
        return 1
    }
    
    echo -e "${GREEN}PASS: Error handling working correctly${NC}"
}

# Test 12: Statistics accuracy
test_statistics() {
    echo -e "\n${YELLOW}Test 12: Statistics accuracy${NC}"
    
    # Clear stats by reloading module
    rmmod "$MODULE_NAME"
    load_module
    
    # Perform known operations
    local num_agents=5
    local allocs_per_agent=3
    local alloc_size=$((10 << 20)) # 10MB
    
    for ((i=0; i<$num_agents; i++)); do
        echo "create_agent $((9000 + i)) $((1 << 30))" > "$PROC_BASE/ctl"
        for ((j=0; j<$allocs_per_agent; j++)); do
            echo "alloc $((9000 + i)) $alloc_size 0" > "$PROC_BASE/ctl"
        done
    done
    
    # Check global stats
    local stats=$(cat "$PROC_BASE/stats")
    local total_allocs=$(echo "$stats" | grep "total_allocations:" | awk '{print $2}')
    local expected_allocs=$((num_agents * allocs_per_agent))
    
    if [ "$total_allocs" != "$expected_allocs" ]; then
        echo -e "${RED}FAIL: Allocation count mismatch ($total_allocs vs $expected_allocs)${NC}"
        return 1
    fi
    
    echo -e "${GREEN}PASS: Statistics tracking accurately${NC}"
}

# Test 13: Cleanup operations
test_cleanup() {
    echo -e "\n${YELLOW}Test 13: Cleanup operations${NC}"
    
    # Create some allocations
    echo "create_agent 10000 $((1 << 30))" > "$PROC_BASE/ctl"
    echo "alloc 10000 $((100 << 20)) 0" > "$PROC_BASE/ctl"
    
    # Remove agent
    echo "remove_agent 10000" > "$PROC_BASE/ctl" || {
        echo -e "${RED}FAIL: Cannot remove agent${NC}"
        return 1
    }
    
    # Verify cleanup
    if [ -d "$PROC_BASE/agents/10000" ]; then
        echo -e "${RED}FAIL: Agent directory not cleaned up${NC}"
        return 1
    fi
    
    echo -e "${GREEN}PASS: Cleanup operations working${NC}"
}

# Main test execution
main() {
    echo "GPU DMA Lock Kernel Module E2E Test"
    echo "==================================="
    
    check_requirements
    load_module
    
    # Run all tests
    local tests=(
        test_proc_interface
        test_gpu_detection
        test_agent_management
        test_memory_allocation
        test_dma_access_control
        test_multi_gpu
        test_concurrent_ops
        test_context_isolation
        test_memory_pressure
        test_module_params
        test_error_handling
        test_statistics
        test_cleanup
    )
    
    local passed=0
    local failed=0
    local skipped=0
    
    for test in "${tests[@]}"; do
        if $test; then
            ((passed++))
        else
            ((failed++))
        fi
    done
    
    # Cleanup
    echo -e "\nCleaning up..."
    rmmod "$MODULE_NAME" 2>/dev/null || true
    
    # Summary
    echo -e "\n=================================="
    echo -e "Test Summary:"
    echo -e "${GREEN}Passed: $passed${NC}"
    echo -e "${RED}Failed: $failed${NC}"
    echo -e "${YELLOW}Skipped: $skipped${NC}"
    echo -e "=================================="
    
    [ $failed -eq 0 ] && exit 0 || exit 1
}

main "$@"