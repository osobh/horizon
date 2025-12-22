# StratoSwarm Kernel Module Testing Guide

## Overview
This guide walks through testing the GPU DMA Lock kernel module in the VM to validate real kernel API integration beyond the 87.58% mock test coverage.

## VM Access
- SSH: `ssh -p 2222 osobh@localhost`
- Password: `clouddev249`

## Step 1: Install Kernel Development Tools
```bash
# Update packages
sudo apt-get update

# Install kernel headers and build tools
sudo apt-get install -y \
    linux-headers-$(uname -r) \
    build-essential \
    make \
    gcc \
    libelf-dev

# Verify installation
ls -la /lib/modules/$(uname -r)/build
```

## Step 2: Transfer Kernel Module Code
From the host machine:
```bash
# Create archive
tar -czf gpu_dma_lock.tar.gz gpu_dma_lock/

# Transfer to VM (you'll need to enter password: clouddev249)
scp -P 2222 gpu_dma_lock.tar.gz osobh@localhost:~/
```

## Step 3: Build Kernel Module in VM
```bash
# Extract archive
tar -xzf gpu_dma_lock.tar.gz
cd gpu_dma_lock

# Build the module
make

# Check if module built successfully
ls -la *.ko
```

## Step 4: Load and Test Module
```bash
# Load the module
sudo insmod gpu_dma_lock.ko debug=1

# Verify it loaded
lsmod | grep gpu_dma_lock

# Check kernel messages
sudo dmesg | tail -20

# Check proc interface
ls -la /proc/swarm/gpu/
```

## Step 5: Run Functional Tests
```bash
# Read statistics
cat /proc/swarm/gpu/stats

# Check quotas
cat /proc/swarm/gpu/quotas

# View allocations
cat /proc/swarm/gpu/allocations

# Test DMA permissions
echo "1:0xDEADBEEF:rw" > /proc/swarm/gpu/dma_permissions
cat /proc/swarm/gpu/dma_permissions

# Run the test suite
./tests/test_module.sh
```

## Step 6: Performance Testing
```bash
# Run performance benchmarks
sudo ./tests/benchmark.sh

# Monitor with perf
sudo perf stat -e cycles,instructions,cache-misses \
    ./tests/stress_test.sh
```

## Step 7: Coverage Analysis with gcov
```bash
# Build with coverage flags
make clean
make CFLAGS="-fprofile-arcs -ftest-coverage"

# Load module and run tests
sudo insmod gpu_dma_lock.ko
./tests/test_module.sh
sudo rmmod gpu_dma_lock

# Generate coverage report
gcov *.c
lcov --capture --directory . --output-file coverage.info
genhtml coverage.info --output-directory coverage_report
```

## Expected Results
- Module loads without errors
- /proc/swarm/gpu/ interface is created
- All proc files are readable/writable as expected
- No kernel panics or warnings
- Performance within targets (<10μs allocation, <1μs DMA checks)

## Troubleshooting
- If module fails to load: Check `dmesg` for errors
- If proc files missing: Verify module initialization
- If performance slow: Check debug mode is disabled
- If build fails: Ensure kernel headers match running kernel

## Clean Up
```bash
# Unload module
sudo rmmod gpu_dma_lock

# Clean build artifacts
make clean
```