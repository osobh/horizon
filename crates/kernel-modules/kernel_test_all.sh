#!/bin/bash
# Comprehensive Kernel Module Testing Script
# Run this inside the VM after copying gpu_dma_lock.tar.gz

set -e  # Exit on error

echo "=== StratoSwarm GPU DMA Lock Kernel Module Testing ==="
echo "Starting at: $(date)"
echo

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Step 1: Extract archive
if [ -f ~/gpu_dma_lock.tar.gz ]; then
    echo -e "${GREEN}✓${NC} Found gpu_dma_lock.tar.gz"
    cd ~
    tar -xzf gpu_dma_lock.tar.gz
    cd gpu_dma_lock
else
    echo -e "${RED}✗${NC} gpu_dma_lock.tar.gz not found!"
    echo "Please copy it first: scp -P 2222 gpu_dma_lock.tar.gz osobh@localhost:~/"
    exit 1
fi

# Step 2: Install dependencies
echo
echo "Installing kernel development tools..."
sudo apt-get update -qq
sudo apt-get install -y linux-headers-$(uname -r) build-essential make gcc

# Verify kernel headers
if [ -d /lib/modules/$(uname -r)/build ]; then
    echo -e "${GREEN}✓${NC} Kernel headers installed for $(uname -r)"
else
    echo -e "${RED}✗${NC} Kernel headers missing!"
    exit 1
fi

# Step 3: Build kernel module
echo
echo "Building kernel module..."
make clean
make

if [ -f gpu_dma_lock.ko ]; then
    echo -e "${GREEN}✓${NC} Module built successfully"
    ls -la gpu_dma_lock.ko
else
    echo -e "${RED}✗${NC} Build failed!"
    exit 1
fi

# Step 4: Load and test module
echo
echo "Loading kernel module..."
sudo insmod gpu_dma_lock.ko debug=1

# Verify loaded
if lsmod | grep -q gpu_dma_lock; then
    echo -e "${GREEN}✓${NC} Module loaded successfully"
else
    echo -e "${RED}✗${NC} Module failed to load!"
    sudo dmesg | tail -20
    exit 1
fi

# Check proc interface
echo
echo "Checking /proc interface..."
if [ -d /proc/swarm/gpu ]; then
    echo -e "${GREEN}✓${NC} Proc interface created"
    ls -la /proc/swarm/gpu/
else
    echo -e "${YELLOW}!${NC} Proc interface not found, checking dmesg..."
    sudo dmesg | grep gpu_dma
fi

# Run functional tests
echo
echo "Running functional tests..."
if [ -f tests/test_module.sh ]; then
    chmod +x tests/test_module.sh
    ./tests/test_module.sh
else
    echo -e "${YELLOW}!${NC} Test script not found, running manual tests..."
    
    echo "Testing stats..."
    cat /proc/swarm/gpu/stats 2>/dev/null || echo "Stats not readable"
    
    echo "Testing quotas..."
    cat /proc/swarm/gpu/quotas 2>/dev/null || echo "Quotas not readable"
    
    echo "Testing allocations..."
    cat /proc/swarm/gpu/allocations 2>/dev/null || echo "Allocations not readable"
fi

# Run performance benchmarks
echo
echo "Running performance benchmarks..."
if [ -f tests/benchmark.sh ]; then
    chmod +x tests/benchmark.sh
    ./tests/benchmark.sh
else
    echo -e "${YELLOW}!${NC} Benchmark script not found"
fi

# Check kernel logs
echo
echo "Kernel messages:"
sudo dmesg | grep gpu_dma | tail -10

# Unload module
echo
echo "Unloading module..."
sudo rmmod gpu_dma_lock

if lsmod | grep -q gpu_dma_lock; then
    echo -e "${RED}✗${NC} Failed to unload module!"
else
    echo -e "${GREEN}✓${NC} Module unloaded successfully"
fi

# Summary
echo
echo "=== Test Summary ==="
echo "Completed at: $(date)"
echo
echo "Next steps for coverage analysis:"
echo "1. make clean"
echo "2. make CFLAGS=\"-fprofile-arcs -ftest-coverage\""
echo "3. sudo insmod gpu_dma_lock.ko"
echo "4. ./tests/test_module.sh"
echo "5. sudo rmmod gpu_dma_lock"
echo "6. gcov src/*.c"

echo
echo "Done!"