#!/bin/bash

echo "=== Connecting to VM to run kernel module tests ==="
echo "Please enter password when prompted: clouddev249"
echo

# SSH command to run all tests
ssh -p 2222 osobh@localhost << 'EOF'
echo "=== Starting Kernel Module Testing ==="
cd ~/gpu_dma_lock

# Check if module was built
echo -e "\n1. Checking build output:"
ls -la *.ko

# Load the module
echo -e "\n2. Loading kernel module:"
sudo insmod gpu_dma_lock.ko debug=1

# Verify module loaded
echo -e "\n3. Verifying module loaded:"
lsmod | grep gpu_dma_lock

# Check kernel messages
echo -e "\n4. Kernel messages:"
sudo dmesg | tail -20 | grep -E "(gpu_dma|swarm)"

# Check proc interface
echo -e "\n5. Checking /proc interface:"
ls -la /proc/swarm/gpu/ 2>/dev/null || echo "Proc interface not found"

# Test basic operations
echo -e "\n6. Testing basic operations:"

# Read stats
echo "Reading stats..."
cat /proc/swarm/gpu/stats 2>&1 || echo "Failed to read stats"

# Read quotas  
echo -e "\nReading quotas..."
cat /proc/swarm/gpu/quotas 2>&1 || echo "Failed to read quotas"

# Read allocations
echo -e "\nReading allocations..."
cat /proc/swarm/gpu/allocations 2>&1 || echo "Failed to read allocations"

# Test write operations
echo -e "\n7. Testing write operations:"

# Set DMA permissions
echo "Setting DMA permissions..."
echo "1:0x10000:rw" > /proc/swarm/gpu/dma_permissions 2>&1
if [ $? -eq 0 ]; then
    echo "Success: DMA permission set"
    cat /proc/swarm/gpu/dma_permissions
else
    echo "Failed to set DMA permissions"
fi

# Test control commands
echo -e "\nTesting control commands..."
echo "reset_stats" > /proc/swarm/gpu/control 2>&1
if [ $? -eq 0 ]; then
    echo "Success: Stats reset"
else
    echo "Failed to reset stats"
fi

# Run test scripts if available
echo -e "\n8. Running test scripts:"
if [ -f tests/test_module.sh ]; then
    chmod +x tests/test_module.sh
    ./tests/test_module.sh
else
    echo "test_module.sh not found"
fi

if [ -f tests/benchmark.sh ]; then
    echo -e "\n9. Running benchmarks:"
    chmod +x tests/benchmark.sh
    ./tests/benchmark.sh
else
    echo "benchmark.sh not found"
fi

# Final kernel messages
echo -e "\n10. Final kernel messages:"
sudo dmesg | tail -30 | grep -E "(gpu_dma|swarm)"

# Unload module
echo -e "\n11. Unloading module:"
sudo rmmod gpu_dma_lock
if [ $? -eq 0 ]; then
    echo "Module unloaded successfully"
else
    echo "Failed to unload module"
    lsmod | grep gpu_dma_lock
fi

echo -e "\n=== Testing Complete ==="
EOF