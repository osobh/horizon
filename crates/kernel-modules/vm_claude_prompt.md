# Claude Code Prompt for Kernel Module Testing in VM

## Context
I'm running Claude Code directly in a VM where I need to test the GPU DMA Lock kernel module with real kernel APIs. The repository is cloned at ~/stratoswarm.

## Current Situation
1. I'm in a Ubuntu VM with kernel 5.15.0-151-generic
2. Kernel headers and build tools are installed
3. The GPU DMA Lock module at `crates/kernel-modules/gpu_dma_lock` needs to be built and tested
4. Previous attempts showed the module expects Rust functions but only C files are being compiled
5. We achieved 87.58% code coverage with mock tests on the host, now need to validate with real kernel APIs

## Task
Please help me:

1. First, read the memory bank files to understand the project context:
   - memory-bank/projectbrief.md
   - memory-bank/activeContext.md
   - memory-bank/systemPatterns.md
   - memory-bank/progress.md

2. Review the current kernel module code:
   - crates/kernel-modules/gpu_dma_lock/src/main.c
   - crates/kernel-modules/gpu_dma_lock/Makefile
   - The Rust files (*.rs) that should provide the external functions

3. Fix the build system to properly compile the kernel module:
   - The module appears to be a Rust/C hybrid but the Makefile only compiles C
   - Functions like `gpu_dma_lock_init`, `gpu_dma_register_device` are declared as extern but not implemented

4. Build and load the kernel module:
   - Fix compilation errors
   - Load with `insmod`
   - Verify /proc/swarm/gpu interface is created

5. Run comprehensive tests:
   - Execute tests/test_module.sh
   - Run tests/benchmark.sh
   - Test error paths and edge cases
   - Verify performance meets targets (<10μs allocation, <1μs DMA checks)

6. Compare real kernel behavior with our mock test coverage:
   - We have 87.58% coverage with mocks
   - Identify any gaps between mock and real kernel behavior
   - Test kernel-specific code paths that mocks couldn't cover

## Additional Context
- This is part of the StratoSwarm project - a kernel-integrated orchestration platform
- The GPU DMA Lock module provides GPU memory protection and DMA access control
- It should create /proc/swarm/gpu/ with stats, quotas, allocations, dma_permissions, contexts, and control files

## Expected Outcomes
1. Working kernel module that loads/unloads cleanly
2. Functional /proc interface
3. All tests passing
4. Performance within targets
5. Understanding of coverage gaps between mock and real testing

Please start by examining the current state of the code and help me get this kernel module working in the real kernel environment.