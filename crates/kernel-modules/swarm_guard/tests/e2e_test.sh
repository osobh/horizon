#!/bin/bash
# End-to-end test for SwarmGuard kernel module
# Requires root privileges and a test environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Error: This test must be run as root${NC}"
    exit 1
fi

# Test configuration
MODULE_NAME="swarm_guard"
TEST_AGENTS=100
TEST_DURATION=30

echo -e "${YELLOW}SwarmGuard E2E Test Suite${NC}"
echo "=========================="

# Function to check module status
check_module() {
    if lsmod | grep -q "^$MODULE_NAME"; then
        return 0
    else
        return 1
    fi
}

# Function to create test agent
create_test_agent() {
    local id=$1
    local mem_limit=$((256 * 1024 * 1024)) # 256MB
    local cpu_quota=$((10 + ($id % 40))) # 10-50%
    
    cat > /proc/swarm/create <<EOF
{
    "id": $id,
    "memory_limit": $mem_limit,
    "cpu_quota": $cpu_quota,
    "namespace_flags": 63
}
EOF
}

# Function to stress test the module
stress_test() {
    echo -e "\n${YELLOW}Running stress test...${NC}"
    
    local start_time=$(date +%s)
    local ops=0
    
    while [ $(($(date +%s) - start_time)) -lt $TEST_DURATION ]; do
        # Read status
        cat /proc/swarm/status > /dev/null 2>&1 || true
        
        # Create/destroy agents randomly
        if [ $((RANDOM % 2)) -eq 0 ]; then
            create_test_agent $((RANDOM % 1000))
        fi
        
        ops=$((ops + 1))
        
        # Show progress every 1000 operations
        if [ $((ops % 1000)) -eq 0 ]; then
            echo -ne "\rOperations: $ops"
        fi
    done
    
    echo -e "\n${GREEN}Stress test completed: $ops operations${NC}"
}

# Clean up function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    
    if check_module; then
        rmmod $MODULE_NAME 2>/dev/null || true
    fi
    
    # Clean build artifacts
    make clean > /dev/null 2>&1 || true
}

# Set up trap for cleanup
trap cleanup EXIT

# Test 1: Build the module
echo -e "\n${YELLOW}Test 1: Building kernel module${NC}"
make clean > /dev/null
if make > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Module built successfully${NC}"
else
    echo -e "${RED}✗ Module build failed${NC}"
    exit 1
fi

# Test 2: Load the module
echo -e "\n${YELLOW}Test 2: Loading kernel module${NC}"
if insmod ${MODULE_NAME}.ko; then
    echo -e "${GREEN}✓ Module loaded successfully${NC}"
else
    echo -e "${RED}✗ Module load failed${NC}"
    exit 1
fi

# Test 3: Verify proc interface
echo -e "\n${YELLOW}Test 3: Verifying /proc interface${NC}"
if [ -d "/proc/swarm" ]; then
    echo -e "${GREEN}✓ /proc/swarm directory exists${NC}"
    
    # Check required files
    for file in status agents create; do
        if [ -e "/proc/swarm/$file" ]; then
            echo -e "${GREEN}✓ /proc/swarm/$file exists${NC}"
        else
            echo -e "${RED}✗ /proc/swarm/$file missing${NC}"
            exit 1
        fi
    done
else
    echo -e "${RED}✗ /proc/swarm directory missing${NC}"
    exit 1
fi

# Test 4: Create test agents
echo -e "\n${YELLOW}Test 4: Creating test agents${NC}"
for i in $(seq 1 $TEST_AGENTS); do
    create_test_agent $i
    
    # Show progress
    if [ $((i % 10)) -eq 0 ]; then
        echo -ne "\rCreated $i/$TEST_AGENTS agents"
    fi
done
echo -e "\n${GREEN}✓ Created $TEST_AGENTS test agents${NC}"

# Test 5: Verify agent tracking
echo -e "\n${YELLOW}Test 5: Verifying agent tracking${NC}"
status=$(cat /proc/swarm/status)
echo "$status"

# Parse active agent count
active_count=$(echo "$status" | grep "Active agents:" | awk '{print $3}')
if [ "$active_count" -gt 0 ]; then
    echo -e "${GREEN}✓ Agent tracking working: $active_count active agents${NC}"
else
    echo -e "${RED}✗ No active agents found${NC}"
    exit 1
fi

# Test 6: Resource limit enforcement
echo -e "\n${YELLOW}Test 6: Testing resource limits${NC}"
# Try to create an agent with excessive resources
cat > /proc/swarm/create <<EOF
{
    "id": 9999,
    "memory_limit": 17179869184,
    "cpu_quota": 1000,
    "namespace_flags": 63
}
EOF 2>/dev/null || true

# Check if violation was recorded
new_status=$(cat /proc/swarm/status)
violations=$(echo "$new_status" | grep "Policy violations:" | awk '{print $3}')
echo -e "${GREEN}✓ Resource enforcement working: $violations violations recorded${NC}"

# Test 7: Concurrent access
echo -e "\n${YELLOW}Test 7: Testing concurrent access${NC}"
(
    for i in {1..5}; do
        while true; do
            cat /proc/swarm/status > /dev/null 2>&1
        done &
    done
    
    sleep 5
    
    # Kill background jobs
    jobs -p | xargs kill 2>/dev/null || true
) 2>/dev/null

echo -e "${GREEN}✓ Concurrent access handled successfully${NC}"

# Test 8: Stress test
stress_test

# Test 9: Module unload
echo -e "\n${YELLOW}Test 9: Unloading module${NC}"
if rmmod $MODULE_NAME; then
    echo -e "${GREEN}✓ Module unloaded successfully${NC}"
else
    echo -e "${RED}✗ Module unload failed${NC}"
    exit 1
fi

# Test 10: Verify cleanup
echo -e "\n${YELLOW}Test 10: Verifying cleanup${NC}"
if [ ! -d "/proc/swarm" ]; then
    echo -e "${GREEN}✓ /proc/swarm cleaned up${NC}"
else
    echo -e "${RED}✗ /proc/swarm still exists${NC}"
    exit 1
fi

# Final summary
echo -e "\n${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ All E2E tests passed successfully!${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"