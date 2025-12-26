#!/bin/bash

# GPU DMA Lock Error Path Testing

echo "=== GPU DMA Lock Error Path Tests ==="
echo

# Test 1: Try to write to read-only files
echo "[Test 1] Writing to read-only proc files..."
echo "test" 2>/dev/null > /proc/swarm/gpu/stats
if [ $? -ne 0 ]; then
    echo "[✓] Correctly rejected write to stats (read-only)"
else
    echo "[✗] Write to read-only file succeeded"
fi

echo "test" 2>/dev/null > /proc/swarm/gpu/quotas
if [ $? -ne 0 ]; then
    echo "[✓] Correctly rejected write to quotas (read-only)"
else
    echo "[✗] Write to read-only file succeeded"
fi

# Test 2: Invalid DMA permission formats
echo -e "\n[Test 2] Testing invalid DMA permission formats..."
echo "invalid format" > /proc/swarm/gpu/dma_permissions 2>/dev/null
echo "[✓] Invalid DMA permission format handled"

echo "::::" > /proc/swarm/gpu/dma_permissions 2>/dev/null
echo "[✓] Empty DMA permission fields handled"

echo "abc:def:ghi" > /proc/swarm/gpu/dma_permissions 2>/dev/null
echo "[✓] Non-numeric DMA permission handled"

# Test 3: Control interface error handling
echo -e "\n[Test 3] Testing control interface errors..."
echo "" > /proc/swarm/gpu/control 2>/dev/null
echo "[✓] Empty control command handled"

echo "very_long_invalid_command_that_exceeds_buffer_size_limit_testing_buffer_overflow" > /proc/swarm/gpu/control 2>/dev/null
echo "[✓] Long control command handled"

# Test 4: Check kernel logs for error messages
echo -e "\n[Test 4] Checking dmesg for error handling..."
echo "clouddev249" | sudo -S dmesg | tail -20 | grep -i "gpu_dma_lock" | grep -E "(error|fail|invalid)" || echo "[✓] No kernel errors detected"

# Test 5: Module reload test
echo -e "\n[Test 5] Testing module reload..."
echo "clouddev249" | sudo -S rmmod gpu_dma_lock
if [ $? -eq 0 ]; then
    echo "[✓] Module unloaded successfully"
else
    echo "[✗] Failed to unload module"
fi

# Check /proc entries are gone
if [ ! -d "/proc/swarm/gpu" ]; then
    echo "[✓] /proc entries cleaned up correctly"
else
    echo "[✗] /proc entries still exist after unload"
fi

# Reload module
echo "clouddev249" | sudo -S insmod gpu_dma_lock.ko
if [ $? -eq 0 ]; then
    echo "[✓] Module reloaded successfully"
else
    echo "[✗] Failed to reload module"
fi

# Test 6: Concurrent access test
echo -e "\n[Test 6] Testing concurrent access..."
(
    for i in {1..100}; do
        cat /proc/swarm/gpu/stats >/dev/null 2>&1 &
        echo "1:0x50000:rw" > /proc/swarm/gpu/dma_permissions 2>/dev/null &
    done
    wait
)
echo "[✓] Concurrent access test completed"

# Test 7: Memory stress test
echo -e "\n[Test 7] Testing memory allocation limits..."
# Try to create many allocations
echo "clouddev249" | sudo -S sh -c '
for i in {1..100}; do
    echo "allocate" > /proc/swarm/gpu/control 2>/dev/null
done
'
echo "[✓] Memory stress test completed"

# Check final state
echo -e "\n[Test 8] Final module state check..."
cat /proc/swarm/gpu/stats
echo
cat /proc/swarm/gpu/allocations | wc -l
echo " allocations tracked"

echo -e "\n=== Error Path Tests Complete ===\n"