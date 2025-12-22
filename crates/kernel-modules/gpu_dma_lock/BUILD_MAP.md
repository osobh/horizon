# GPU DMA Lock Build Directory Map

## Overview
This document maps the directory structure and build artifact locations for the GPU DMA Lock kernel module, helping navigate the complex Rust/C kernel module build process.

## Directory Structure

```
/home/osobh/stratoswarm/                    # Workspace root
│
├── target/                                 # Workspace-wide target (when exists)
│   └── x86_64-unknown-none/release/       # Custom target builds
│       └── libgpu_dma_lock.a             # Rust static library
│
└── crates/kernel-modules/
    ├── gpu_dma_lock/                      # Main Rust/C hybrid module
    │   ├── Cargo.toml                     # Rust package config
    │   ├── Makefile                       # Kernel module makefile
    │   ├── x86_64-linux-kernel.json       # Custom Rust target
    │   ├── .cargo/
    │   │   └── config.toml               # Cargo build settings
    │   ├── src/
    │   │   ├── lib.rs                    # Main Rust library
    │   │   ├── main.c                    # C kernel entry point
    │   │   ├── kernel_ffi.rs             # Minimal FFI interface
    │   │   ├── kernel_stubs.S            # Assembly stubs
    │   │   ├── [modules].rs              # Other Rust modules
    │   │   └── [extracted files]         # After ar extraction:
    │   │       ├── gpu_dma_lock-*.o      # Rust object files
    │   │       ├── core-*.o              # Core library objects
    │   │       ├── alloc-*.o             # Alloc library objects
    │   │       └── rust_code.o           # Combined Rust object
    │   ├── target/                        # Local build cache
    │   └── [build outputs]                # After kernel build:
    │       ├── gpu_dma_lock.ko           # Kernel module
    │       ├── Module.symvers            # Module symbols
    │       └── *.mod.c                   # Generated files
    │
    └── gpu_dma_lock_c_only/              # Simplified C-only version
        ├── Makefile
        ├── gpu_dma_lock.c
        └── gpu_dma_lock.ko               # Working kernel module
```

## Build Artifact Locations

### Rust Compilation
1. **Source**: `crates/kernel-modules/gpu_dma_lock/src/*.rs`
2. **Target**: `target/x86_64-unknown-none/release/libgpu_dma_lock.a`
   - Note: This is 3 directories up from the module directory
   - Path from module: `../../../target/x86_64-unknown-none/release/`

### Object Extraction
1. **Static library**: `libgpu_dma_lock.a`
2. **Extracted to**: `src/` directory (mixed with source files)
3. **Files created**:
   - `gpu_dma_lock-*.o` - Main module objects
   - `core-*.o` - Rust core library
   - `alloc-*.o` - Rust alloc library
   - `compiler_builtins-*.o` - Compiler runtime

### Linking
1. **Input**: All `*.o` files in `src/`
2. **Output**: `src/rust_code.o`
3. **Cmd file**: `src/.rust_code.o.cmd` (for kernel build system)

### Kernel Module Build
1. **C source**: `src/main.c`
2. **Rust object**: `src/rust_code.o`
3. **Assembly**: `src/kernel_stubs.o`
4. **Output**: `gpu_dma_lock.ko` (in module root)

## Common Navigation Patterns

```bash
# Always return to module root
cd /home/osobh/stratoswarm/crates/kernel-modules/gpu_dma_lock

# Check Rust library location
ls -la ../../../target/x86_64-unknown-none/release/libgpu_dma_lock.a

# Work in src directory
cd src && pwd  # Should show .../gpu_dma_lock/src

# Return from any deep directory
cd $(git rev-parse --show-toplevel)/crates/kernel-modules/gpu_dma_lock
```

## Build Process Flow

```
1. Rust Compilation (cargo +nightly build)
   └─> Creates: target/x86_64-unknown-none/release/libgpu_dma_lock.a

2. Object Extraction (ar x)
   └─> Extracts to: src/*.o files

3. Object Linking (ld -r)
   └─> Creates: src/rust_code.o

4. Kernel Build (make modules)
   └─> Compiles: src/main.c -> src/main.o
   └─> Assembles: src/kernel_stubs.S -> src/kernel_stubs.o
   └─> Links all: gpu_dma_lock.ko

5. Module Loading (insmod)
   └─> Loads: gpu_dma_lock.ko
   └─> Creates: /proc/swarm/gpu/*
```

## Troubleshooting Paths

### "No such file or directory" errors
- Check current directory with `pwd`
- Verify relative paths from current location
- Use absolute paths when uncertain

### Missing libgpu_dma_lock.a
- Check if cargo build completed successfully
- Look in workspace target, not local target
- Path: `../../../target/x86_64-unknown-none/release/`

### Relocation errors (R_X86_64_GOTPCREL)
- Check RUSTFLAGS in Makefile
- Verify target JSON configuration
- Use readelf to inspect: `readelf -r src/rust_code.o`

### Module format errors
- Check kernel version: `uname -r`
- Verify module info: `modinfo gpu_dma_lock.ko`
- Check dmesg: `sudo dmesg | tail`

## Quick Reference

```bash
# Full rebuild
make clean && make

# Just Rust rebuild
cargo +nightly build --target x86_64-unknown-none -Z build-std=core,alloc --release

# Manual extraction and linking
cd src
ar x ../../../../target/x86_64-unknown-none/release/libgpu_dma_lock.a
ld -r -o rust_code.o gpu_dma_lock-*.o
cd ..

# Load module
sudo insmod gpu_dma_lock.ko

# Check proc interface
ls -la /proc/swarm/gpu/
```

## Important Notes

1. **Working Directory Matters**: Many commands assume you're in the module root
2. **Relative Paths**: The `../../../target` path assumes standard workspace structure
3. **Mixed Source/Objects**: The `src/` directory contains both source files and extracted objects
4. **Clean Builds**: Always `make clean` when switching between build configurations
5. **Two Implementations**: 
   - `gpu_dma_lock/` - Full Rust/C hybrid (relocation issues)
   - `gpu_dma_lock_c_only/` - Simplified C version (working)