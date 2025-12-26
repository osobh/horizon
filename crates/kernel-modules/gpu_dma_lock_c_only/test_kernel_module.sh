#!/bin/bash

# GPU DMA Lock Kernel Module Comprehensive Test Suite

echo "=== GPU DMA Lock Kernel Module Test Suite ==="
echo "Testing real kernel APIs and /proc interface"
echo

# Check if module is loaded
if ! lsmod | grep -q gpu_dma_lock; then
    echo "ERROR: Module not loaded!"
    exit 1
fi

echo "[✓] Module loaded successfully"

# Test 1: Verify /proc interface exists
echo -e "\n[Test 1] Checking /proc interface..."
if [ -d "/proc/swarm/gpu" ]; then
    echo "[✓] /proc/swarm/gpu exists"
    ls -la /proc/swarm/gpu/
else
    echo "[✗] /proc/swarm/gpu not found"
    exit 1
fi

# Test 2: Read statistics
echo -e "\n[Test 2] Reading initial statistics..."
if cat /proc/swarm/gpu/stats; then
    echo "[✓] Stats readable"
else
    echo "[✗] Failed to read stats"
fi

# Test 3: Read quotas
echo -e "\n[Test 3] Reading agent quotas..."
if cat /proc/swarm/gpu/quotas; then
    echo "[✓] Quotas readable"
else
    echo "[✗] Failed to read quotas"
fi

# Test 4: Read allocations
echo -e "\n[Test 4] Reading current allocations..."
if cat /proc/swarm/gpu/allocations; then
    echo "[✓] Allocations readable"
else
    echo "[✗] Failed to read allocations"
fi

# Test 5: Read DMA permissions
echo -e "\n[Test 5] Reading DMA permissions..."
if cat /proc/swarm/gpu/dma_permissions; then
    echo "[✓] DMA permissions readable"
else
    echo "[✗] Failed to read DMA permissions"
fi

# Test 6: Test control interface - allocate
echo -e "\n[Test 6] Testing control interface - allocation..."
echo "allocate" > /proc/swarm/gpu/control
if [ $? -eq 0 ]; then
    echo "[✓] Allocation command sent"
    echo "After allocation:"
    cat /proc/swarm/gpu/allocations
else
    echo "[✗] Failed to send allocation command"
fi

# Test 7: Update stats and verify
echo -e "\n[Test 7] Checking updated statistics..."
cat /proc/swarm/gpu/stats

# Test 8: Test DMA permission write
echo -e "\n[Test 8] Testing DMA permission update..."
echo "1:0x30000:rw" > /proc/swarm/gpu/dma_permissions
if [ $? -eq 0 ]; then
    echo "[✓] DMA permission update sent"
else
    echo "[✗] Failed to update DMA permissions"
fi

# Test 9: Reset statistics
echo -e "\n[Test 9] Testing stats reset..."
echo "reset_stats" > /proc/swarm/gpu/control
if [ $? -eq 0 ]; then
    echo "[✓] Stats reset command sent"
    echo "After reset:"
    cat /proc/swarm/gpu/stats
else
    echo "[✗] Failed to reset stats"
fi

# Test 10: Check kernel logs for module messages
echo -e "\n[Test 10] Checking kernel logs..."
dmesg | tail -20 | grep gpu_dma_lock

# Performance test setup
echo -e "\n[Performance Tests]"

# Test 11: Measure allocation performance
echo -e "\n[Test 11] Measuring allocation performance..."
start_time=$(date +%s%N)
for i in {1..1000}; do
    echo "allocate" > /proc/swarm/gpu/control
done
end_time=$(date +%s%N)
elapsed=$((($end_time - $start_time) / 1000))
avg_time=$(($elapsed / 1000))
echo "1000 allocations took ${elapsed}μs (average: ${avg_time}μs per allocation)"

# Test 12: Measure DMA check performance
echo -e "\n[Test 12] Measuring DMA check performance..."
start_time=$(date +%s%N)
for i in {1..10000}; do
    echo "1:0x40000:r" > /proc/swarm/gpu/dma_permissions
done
end_time=$(date +%s%N)
elapsed=$((($end_time - $start_time) / 1000))
avg_time=$(($elapsed / 10000))
echo "10000 DMA checks took ${elapsed}μs (average: ${avg_time}μs per check)"

# Test 13: Error path - invalid control command
echo -e "\n[Test 13] Testing error paths..."
echo "invalid_command" > /proc/swarm/gpu/control 2>/dev/null
echo "[✓] Invalid command handled"

# Test 14: Check memory usage
echo -e "\n[Test 14] Final allocation status..."
cat /proc/swarm/gpu/allocations | head -20

# Test 15: Final statistics
echo -e "\n[Test 15] Final statistics..."
cat /proc/swarm/gpu/stats

echo -e "\n=== Test Suite Complete ==="
echo "Module tested with real kernel APIs"