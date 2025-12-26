# Kernel Module Test Instructions

## Current Status
- ✅ You've extracted the archive
- ✅ You've installed kernel headers
- ✅ You've built the module with `make`
- ⏳ Ready to load and test the module

## Next Steps in VM

### 1. Load the Module
```bash
sudo insmod gpu_dma_lock.ko debug=1
```

### 2. Verify It Loaded
```bash
lsmod | grep gpu_dma_lock
sudo dmesg | tail -20
```

### 3. Check /proc Interface
```bash
ls -la /proc/swarm/gpu/
```

### 4. Run Basic Tests
```bash
# Read operations
cat /proc/swarm/gpu/stats
cat /proc/swarm/gpu/quotas
cat /proc/swarm/gpu/allocations

# Write operations
echo '1:0x10000:rw' > /proc/swarm/gpu/dma_permissions
cat /proc/swarm/gpu/dma_permissions
echo 'reset_stats' > /proc/swarm/gpu/control
```

### 5. Run Test Scripts
```bash
chmod +x tests/*.sh
./tests/test_module.sh
./tests/benchmark.sh
```

### 6. Run Coverage Gap Tests (Optional)
```bash
# Copy this from host if needed:
# scp -P 2222 coverage_gap_tests.sh osobh@localhost:~/gpu_dma_lock/
./coverage_gap_tests.sh
```

### 7. Unload Module
```bash
sudo rmmod gpu_dma_lock
```

## What to Look For

### Success Indicators:
- Module loads without errors in dmesg
- /proc/swarm/gpu/ directory exists with all files
- test_module.sh shows all tests passing
- Benchmarks show performance within targets
- No kernel warnings or panics

### Potential Issues:
- "Unknown symbol" errors → Missing dependencies
- "File exists" errors → Module already loaded
- Proc files missing → Check initialization code
- Permission denied → Need sudo for some operations
- Kernel oops/panic → Serious bug in module

## Share Results
Please share the output of:
1. `lsmod | grep gpu_dma_lock`
2. `ls -la /proc/swarm/gpu/`
3. `./tests/test_module.sh` output
4. `./tests/benchmark.sh` output
5. Any errors from `dmesg`

This will help us understand how the real kernel environment differs from our mock tests and identify areas for coverage improvement beyond the current 87.58%.