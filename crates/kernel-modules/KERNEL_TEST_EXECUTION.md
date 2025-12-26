# Kernel Module Test Execution Guide

## Quick Start Commands

Execute these commands to run the complete kernel module test suite:

### 1. Transfer Files to VM
```bash
# Copy the archive and test script
scp -P 2222 gpu_dma_lock.tar.gz osobh@localhost:~/
scp -P 2222 kernel_test_all.sh osobh@localhost:~/
# Password for both: clouddev249
```

### 2. Connect to VM and Run Tests
```bash
# SSH into VM
ssh -p 2222 osobh@localhost
# Password: clouddev249

# Run the complete test suite
chmod +x kernel_test_all.sh
./kernel_test_all.sh
```

## What the Test Script Does

1. **Extracts** the kernel module archive
2. **Installs** kernel headers and build tools
3. **Builds** the gpu_dma_lock kernel module
4. **Loads** the module with debug enabled
5. **Verifies** /proc interface creation
6. **Runs** functional tests (test_module.sh)
7. **Runs** performance benchmarks
8. **Unloads** the module cleanly

## Expected Output

✅ **Success Indicators:**
- "Module built successfully"
- "Module loaded successfully"
- "Proc interface created"
- All tests in test_module.sh pass
- Performance under target (<10μs allocation, <1μs DMA)

❌ **Failure Indicators:**
- Build errors (missing headers)
- "Module failed to load" 
- Proc interface missing
- Test failures
- Kernel panics or warnings in dmesg

## Manual Testing (if automated fails)

```bash
# Inside VM, after extracting archive
cd ~/gpu_dma_lock

# Build
make

# Load
sudo insmod gpu_dma_lock.ko debug=1

# Test proc files
ls -la /proc/swarm/gpu/
cat /proc/swarm/gpu/stats
cat /proc/swarm/gpu/quotas
echo "1:0x10000:rw" > /proc/swarm/gpu/dma_permissions
cat /proc/swarm/gpu/dma_permissions

# Check kernel logs
sudo dmesg | grep gpu_dma

# Unload
sudo rmmod gpu_dma_lock
```

## Coverage Testing

For code coverage analysis:
```bash
# Build with coverage flags
make clean
make CFLAGS="-fprofile-arcs -ftest-coverage"

# Run tests
sudo insmod gpu_dma_lock.ko
./tests/test_module.sh
sudo rmmod gpu_dma_lock

# Generate coverage
gcov src/*.c
lcov --capture --directory . --output-file coverage.info
```

## Troubleshooting

- **Headers not found**: Ensure `linux-headers-$(uname -r)` matches kernel
- **Build fails**: Check gcc version and kernel compatibility
- **Module won't load**: Check `dmesg` for detailed errors
- **Proc missing**: Module initialization may have failed
- **Tests timeout**: VM may be resource constrained

## Results Location

Test results will be displayed in terminal. Key files:
- Build output: `gpu_dma_lock.ko`
- Test results: Terminal output from test_module.sh
- Performance: Benchmark timings in microseconds
- Kernel logs: `sudo dmesg | grep gpu_dma`

Ready to validate the kernel module beyond our 87.58% mock coverage!