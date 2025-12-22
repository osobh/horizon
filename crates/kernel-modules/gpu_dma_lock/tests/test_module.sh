#!/bin/bash

# GPU DMA Lock Kernel Module Test Suite

echo "=== GPU DMA Lock Module Test Suite ==="
echo "Testing real kernel API integration..."
echo

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
test_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((TESTS_PASSED++))
}

test_fail() {
    echo -e "${RED}✗${NC} $1"
    ((TESTS_FAILED++))
}

# Check if module is loaded
if ! lsmod | grep -q gpu_dma_lock; then
    echo "ERROR: Module not loaded. Run: sudo insmod gpu_dma_lock.ko"
    exit 1
fi

echo "1. Testing /proc interface creation..."
if [ -d /proc/swarm/gpu ]; then
    test_pass "/proc/swarm/gpu directory exists"
else
    test_fail "/proc/swarm/gpu directory missing"
    exit 1
fi

echo
echo "2. Testing proc file existence..."
for file in stats quotas allocations dma_permissions contexts control; do
    if [ -f /proc/swarm/gpu/$file ]; then
        test_pass "/proc/swarm/gpu/$file exists"
    else
        test_fail "/proc/swarm/gpu/$file missing"
    fi
done

echo
echo "3. Testing read operations..."

# Test stats reading
if OUTPUT=$(cat /proc/swarm/gpu/stats 2>&1); then
    test_pass "Stats readable: $(echo $OUTPUT | head -c 50)..."
else
    test_fail "Stats read failed: $OUTPUT"
fi

# Test quotas reading
if OUTPUT=$(cat /proc/swarm/gpu/quotas 2>&1); then
    test_pass "Quotas readable"
else
    test_fail "Quotas read failed: $OUTPUT"
fi

echo
echo "4. Testing write operations..."

# Test control commands
echo "reset_stats" > /proc/swarm/gpu/control 2>&1
if [ $? -eq 0 ]; then
    test_pass "Control command 'reset_stats' accepted"
else
    test_fail "Control command 'reset_stats' failed"
fi

# Test DMA permissions
echo "1:0x10000:rw" > /proc/swarm/gpu/dma_permissions 2>&1
if [ $? -eq 0 ]; then
    test_pass "DMA permission write accepted"
    # Verify it was set
    if grep -q "0x10000" /proc/swarm/gpu/dma_permissions; then
        test_pass "DMA permission verified"
    else
        test_fail "DMA permission not set correctly"
    fi
else
    test_fail "DMA permission write failed"
fi

echo
echo "5. Testing quota operations..."

# Set quota
echo "set_quota 1 1048576" > /proc/swarm/gpu/control 2>&1
if [ $? -eq 0 ]; then
    test_pass "Quota set command accepted"
else
    test_fail "Quota set command failed"
fi

echo
echo "6. Testing error conditions..."

# Test invalid control command
echo "invalid_command" > /proc/swarm/gpu/control 2>&1
if [ $? -ne 0 ]; then
    test_pass "Invalid command properly rejected"
else
    test_fail "Invalid command not rejected"
fi

# Test invalid DMA permission format
echo "invalid:format" > /proc/swarm/gpu/dma_permissions 2>&1
if [ $? -ne 0 ]; then
    test_pass "Invalid DMA format properly rejected"
else
    test_fail "Invalid DMA format not rejected"
fi

echo
echo "7. Testing concurrent access..."

# Concurrent reads
(
    for i in {1..10}; do
        cat /proc/swarm/gpu/stats > /dev/null 2>&1 &
    done
    wait
)
if [ $? -eq 0 ]; then
    test_pass "Concurrent reads handled"
else
    test_fail "Concurrent reads failed"
fi

echo
echo "=== Test Summary ==="
echo "Passed: $TESTS_PASSED"
echo "Failed: $TESTS_FAILED"
echo

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi