#!/bin/bash

# GPU DMA Lock Performance Benchmark

echo "=== GPU DMA Lock Performance Benchmark ==="
echo "Target: <10μs allocation, <1μs DMA checks"
echo

# Check if module is loaded
if ! lsmod | grep -q gpu_dma_lock; then
    echo "ERROR: Module not loaded. Run: sudo insmod gpu_dma_lock.ko"
    exit 1
fi

# Number of iterations
ITERATIONS=10000

echo "Running benchmarks with $ITERATIONS iterations..."
echo

# Benchmark 1: Allocation performance
echo "1. Testing allocation performance..."
START=$(date +%s%N)

for i in $(seq 1 $ITERATIONS); do
    echo "allocate 1 4096 0" > /proc/swarm/gpu/control 2>/dev/null
done

END=$(date +%s%N)
DURATION=$((($END - $START) / 1000)) # Convert to microseconds
AVG_TIME=$(($DURATION / $ITERATIONS))

echo "   Total time: ${DURATION}μs"
echo "   Average per allocation: ${AVG_TIME}μs"
if [ $AVG_TIME -lt 10 ]; then
    echo "   ✓ PASS: Under 10μs target"
else
    echo "   ✗ FAIL: Exceeds 10μs target"
fi

echo
echo "2. Testing DMA permission check performance..."
# First set some permissions
echo "1:0x10000:rw" > /proc/swarm/gpu/dma_permissions
echo "1:0x20000:r" > /proc/swarm/gpu/dma_permissions
echo "1:0x30000:w" > /proc/swarm/gpu/dma_permissions

START=$(date +%s%N)

for i in $(seq 1 $ITERATIONS); do
    cat /proc/swarm/gpu/dma_permissions > /dev/null
done

END=$(date +%s%N)
DURATION=$((($END - $START) / 1000))
AVG_TIME=$(($DURATION / $ITERATIONS))

echo "   Total time: ${DURATION}μs"
echo "   Average per check: ${AVG_TIME}μs"
if [ $AVG_TIME -lt 1 ]; then
    echo "   ✓ PASS: Under 1μs target"
else
    echo "   ✗ FAIL: Exceeds 1μs target"
fi

echo
echo "3. Testing stats collection overhead..."
START=$(date +%s%N)

for i in $(seq 1 $ITERATIONS); do
    cat /proc/swarm/gpu/stats > /dev/null
done

END=$(date +%s%N)
DURATION=$((($END - $START) / 1000))
AVG_TIME=$(($DURATION / $ITERATIONS))

echo "   Total time: ${DURATION}μs"
echo "   Average per stats read: ${AVG_TIME}μs"

echo
echo "4. Testing concurrent operation performance..."
echo "   Running 10 parallel threads..."

START=$(date +%s%N)

for thread in $(seq 1 10); do
    (
        for i in $(seq 1 1000); do
            cat /proc/swarm/gpu/stats > /dev/null
            echo "allocate $thread 4096 0" > /proc/swarm/gpu/control 2>/dev/null
        done
    ) &
done

wait

END=$(date +%s%N)
DURATION=$((($END - $START) / 1000000)) # Convert to milliseconds

echo "   Total time for 10,000 operations: ${DURATION}ms"
echo "   Operations per second: $((10000 * 1000 / $DURATION))"

echo
echo "=== Benchmark Summary ==="
cat /proc/swarm/gpu/stats | grep -E "allocations|dma_checks|total_time"

echo
echo "Resetting stats..."
echo "reset_stats" > /proc/swarm/gpu/control

echo "Done!"