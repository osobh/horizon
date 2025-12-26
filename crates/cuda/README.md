# CUDA

CUDA toolkit integration providing GPU detection, kernel compilation, and device management for StratoSwarm.

## Overview

The `cuda` crate provides safe Rust abstractions over CUDA operations, enabling GPU acceleration throughout the StratoSwarm ecosystem. It features automatic CUDA toolkit detection with graceful fallback to mock implementations for development and testing without GPU hardware.

## Features

- **Automatic CUDA Detection**: Finds CUDA installation automatically
- **Safe Abstractions**: Memory-safe wrappers around unsafe CUDA APIs
- **Kernel Management**: Load and execute PTX/CUBIN kernels
- **Memory Management**: Unified memory and explicit device allocations
- **Stream Support**: Asynchronous execution with CUDA streams
- **Context Management**: Multi-GPU context handling
- **Mock Implementation**: Full API compatibility without GPU hardware
- **Build-Time Configuration**: Compile-time CUDA detection and configuration

## Usage

### Basic GPU Operations

```rust
use exorust_cuda::{CudaDevice, CudaMemory, CudaStream};

// Initialize CUDA
let device = CudaDevice::new(0)?; // Device 0
let context = device.create_context()?;

// Allocate GPU memory
let mut gpu_mem = CudaMemory::new(1024 * 1024)?; // 1MB

// Copy data to GPU
gpu_mem.copy_from_host(&host_data)?;

// Create stream for async operations
let stream = CudaStream::new()?;

// Launch kernel
let kernel = device.load_kernel("my_kernel.ptx", "kernel_function")?;
kernel.launch(&[gpu_mem.as_ptr()], (256, 1, 1), (32, 1, 1), &stream)?;

// Synchronize
stream.synchronize()?;

// Copy results back
let mut results = vec![0u8; size];
gpu_mem.copy_to_host(&mut results)?;
```

### Kernel Compilation

```rust
use exorust_cuda::{KernelBuilder, CompileOptions};

// Compile CUDA C++ to PTX
let ptx = KernelBuilder::new()
    .source(cuda_source)
    .options(CompileOptions::default()
        .arch("sm_80")
        .opt_level(3))
    .compile()?;

// Load compiled kernel
let kernel = device.load_ptx(&ptx, "my_function")?;
```

### Multi-GPU Support

```rust
use exorust_cuda::multi_gpu::{GpuCluster, GpuTopology};

// Discover GPU topology
let topology = GpuTopology::detect()?;
println!("Found {} GPUs", topology.device_count());

// Check peer access
if topology.can_access_peer(0, 1)? {
    // Enable peer access for GPU-to-GPU transfers
    topology.enable_peer_access(0, 1)?;
}

// Create GPU cluster
let cluster = GpuCluster::new(topology)?;
```

## Architecture

The crate is structured as follows:

- `detection.rs`: CUDA toolkit detection and initialization
- `context.rs`: CUDA context management
- `kernel.rs`: Kernel loading and execution
- `memory.rs`: Device memory allocation and transfers
- `stream.rs`: Asynchronous execution streams
- `error.rs`: Comprehensive error types
- Build script handles compile-time configuration

## Build Configuration

The build script automatically detects CUDA:

```bash
# Force CUDA path
CUDA_PATH=/usr/local/cuda cargo build

# Disable CUDA (use mock)
STRATOSWARM_CUDA_MOCK=1 cargo build

# Specify compute capability
CUDA_COMPUTE_CAP=80 cargo build
```

## Performance Considerations

- **Memory Transfers**: PCIe bandwidth limited (~16GB/s PCIe 4.0)
- **Kernel Launch**: ~5μs overhead per launch
- **Context Switch**: ~10μs between contexts
- **Allocation**: ~100μs for first allocation, pooled after

## Error Handling

Comprehensive error types for all CUDA operations:

```rust
use exorust_cuda::CudaError;

match result {
    Err(CudaError::NoDevice) => {
        // No CUDA devices found
    }
    Err(CudaError::OutOfMemory) => {
        // GPU memory exhausted
    }
    Err(CudaError::InvalidKernel(name)) => {
        // Kernel not found or invalid
    }
    Err(e) => {
        // Other errors
    }
}
```

## Testing

```bash
# Run tests (uses mock if no GPU)
cargo test

# Force mock implementation
STRATOSWARM_CUDA_MOCK=1 cargo test

# Run with real GPU
cargo test --features gpu_required
```

## Known Issues

- **ContextProperties**: Struct definition missing (compilation issue)
- Mock implementation doesn't simulate all timing characteristics
- Some edge cases in multi-GPU scenarios need testing

## Coverage

Current test coverage: ~65% (Needs improvement)

Well-tested areas:
- Basic memory operations
- Kernel loading
- Error handling
- Mock implementation

Areas needing tests:
- Multi-GPU operations
- Complex kernel launches
- Memory pooling
- Stream synchronization edge cases

## Safety

All unsafe CUDA operations are wrapped in safe abstractions:
- Automatic memory deallocation
- Bounds checking on all operations
- Context lifetime management
- Thread safety via internal synchronization

## Integration Points

Foundation for GPU operations throughout StratoSwarm:
- `gpu-agents`: GPU computation backend
- `agent-core`: GPU-accelerated agent operations
- `memory`: GPU memory tier management
- `evolution-engines`: GPU-accelerated evolution

## Debugging

Enable debug output:

```bash
RUST_LOG=exorust_cuda=debug cargo run
```

CUDA debugging tools:
- `cuda-gdb`: GPU debugger
- `nsight-compute`: Kernel profiler
- `nsight-systems`: System-wide profiler

## License

MIT