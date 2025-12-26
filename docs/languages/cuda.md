# CUDA Development Rules

## Overview

This document defines comprehensive coding standards, length limits, linting configurations, and Test-Driven Development (TDD) practices for CUDA C/C++ projects. All development follows a strict TDD approach with GPU-specific testing requirements and performance considerations.

## CUDA Coding Standards

### Style Guide and Formatting

- **Style Guide**: Google C++ Style Guide + NVIDIA CUDA Best Practices
- **Formatter**: clang-format
- **Line Length**: 100 characters
- **Indentation**: 2 spaces
- **File Extensions**: `.cu` for CUDA source, `.cuh` for CUDA headers, `.cpp/.hpp` for host code
- **Naming**: Follow CUDA and C++ naming conventions

### Import Organization

- Group includes: System headers, CUDA headers, third-party libraries, project headers
- Use forward declarations when possible to reduce compilation time
- Include CUDA headers before other GPU libraries
- Separate device and host includes when applicable

```cuda
// System headers
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>

// CUDA headers
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Third-party libraries
#include <gtest/gtest.h>
#include <benchmark/benchmark.h>

// Project headers
#include "cuda_utils.cuh"
#include "device_memory.cuh"
#include "kernel_launcher.cuh"

// Local headers
#include "matrix_operations.cuh"
```

### Naming Conventions

| Element             | Convention            | Example                  |
| ------------------- | --------------------- | ------------------------ |
| Files               | snake_case            | `matrix_multiply.cu`     |
| Kernels             | snake_case + \_kernel | `matrix_mul_kernel`      |
| Device Functions    | snake_case + \_device | `reduce_sum_device`      |
| Host Functions      | snake_case            | `launch_matrix_multiply` |
| Classes             | PascalCase            | `CudaMatrixOps`          |
| Variables           | snake_case            | `block_size`             |
| Constants           | UPPER_SNAKE_CASE      | `MAX_THREADS_PER_BLOCK`  |
| Macros              | UPPER_SNAKE_CASE      | `CUDA_CHECK_ERROR`       |
| Template Parameters | PascalCase            | `typename DataType`      |
| Device Variables    | prefix with d\_       | `d_input`, `d_output`    |
| Host Variables      | prefix with h\_       | `h_input`, `h_output`    |
| Shared Memory       | prefix with s\_       | `s_data`, `s_temp`       |

### CUDA-Specific Standards

#### Error Handling

- Always check CUDA errors using proper error handling macros
- Use RAII for resource management (device memory, streams, contexts)
- Implement proper exception handling for GPU operations
- Provide meaningful error messages with context

```cuda
// Error checking macro
#define CUDA_CHECK_ERROR(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                << " - " << cudaGetErrorString(err) << std::endl; \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

// RAII wrapper for device memory
template<typename T>
class DeviceMemory {
private:
  T* d_ptr_;
  size_t size_;

public:
  explicit DeviceMemory(size_t count) : size_(count * sizeof(T)) {
    CUDA_CHECK_ERROR(cudaMalloc(&d_ptr_, size_));
  }

  ~DeviceMemory() {
    if (d_ptr_) {
      cudaFree(d_ptr_);
    }
  }

  // Move constructor
  DeviceMemory(DeviceMemory&& other) noexcept
    : d_ptr_(other.d_ptr_), size_(other.size_) {
    other.d_ptr_ = nullptr;
    other.size_ = 0;
  }

  // Delete copy constructor and assignment
  DeviceMemory(const DeviceMemory&) = delete;
  DeviceMemory& operator=(const DeviceMemory&) = delete;

  T* get() const { return d_ptr_; }
  size_t size() const { return size_; }

  void copy_from_host(const T* h_data, size_t count) {
    CUDA_CHECK_ERROR(cudaMemcpy(d_ptr_, h_data, count * sizeof(T),
                                cudaMemcpyHostToDevice));
  }

  void copy_to_host(T* h_data, size_t count) const {
    CUDA_CHECK_ERROR(cudaMemcpy(h_data, d_ptr_, count * sizeof(T),
                                cudaMemcpyDeviceToHost));
  }
};

// Exception-based error handling
class CudaException : public std::runtime_error {
public:
  CudaException(cudaError_t error, const std::string& message)
    : std::runtime_error(message + ": " + cudaGetErrorString(error)) {}
};

inline void cuda_check(cudaError_t error, const char* file, int line) {
  if (error != cudaSuccess) {
    std::string message = std::string(file) + ":" + std::to_string(line);
    throw CudaException(error, message);
  }
}

#define CUDA_THROW_ON_ERROR(call) \
  cuda_check(call, __FILE__, __LINE__)
```

#### Kernel Design

- Keep kernels focused on a single operation
- Use appropriate thread block sizes (multiples of 32)
- Implement proper bounds checking
- Optimize memory access patterns
- Use shared memory effectively

```cuda
// Good: Focused kernel with proper error checking
template<typename T>
__global__ void matrix_multiply_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ c,
    int m, int n, int k) {

  // Calculate thread indices
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Bounds checking
  if (row >= m || col >= n) return;

  // Compute matrix multiplication
  T sum = T(0);
  for (int i = 0; i < k; ++i) {
    sum += a[row * k + i] * b[i * n + col];
  }
  c[row * n + col] = sum;
}

// Good: Optimized kernel with shared memory
template<typename T, int TILE_SIZE = 16>
__global__ void matrix_multiply_shared_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ c,
    int m, int n, int k) {

  // Shared memory for tiles
  __shared__ T s_a[TILE_SIZE][TILE_SIZE];
  __shared__ T s_b[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  T sum = T(0);

  // Loop over tiles
  for (int tile = 0; tile < (k + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
    // Load tiles into shared memory
    int a_col = tile * TILE_SIZE + threadIdx.x;
    int b_row = tile * TILE_SIZE + threadIdx.y;

    s_a[threadIdx.y][threadIdx.x] =
        (row < m && a_col < k) ? a[row * k + a_col] : T(0);
    s_b[threadIdx.y][threadIdx.x] =
        (b_row < k && col < n) ? b[b_row * n + col] : T(0);

    __syncthreads();

    // Compute partial sum
    for (int i = 0; i < TILE_SIZE; ++i) {
      sum += s_a[threadIdx.y][i] * s_b[i][threadIdx.x];
    }

    __syncthreads();
  }

  // Write result
  if (row < m && col < n) {
    c[row * n + col] = sum;
  }
}
```

#### Memory Management

- Use appropriate memory types (global, shared, constant, texture)
- Implement memory coalescing for optimal performance
- Use pinned memory for host-device transfers when beneficial
- Consider unified memory for simpler memory management

```cuda
// Memory management utilities
namespace cuda_utils {

template<typename T>
class PinnedMemory {
private:
  T* h_ptr_;
  size_t size_;

public:
  explicit PinnedMemory(size_t count) : size_(count * sizeof(T)) {
    CUDA_CHECK_ERROR(cudaMallocHost(&h_ptr_, size_));
  }

  ~PinnedMemory() {
    if (h_ptr_) {
      cudaFreeHost(h_ptr_);
    }
  }

  T* get() const { return h_ptr_; }
  size_t size() const { return size_; }
};

// Unified memory wrapper
template<typename T>
class UnifiedMemory {
private:
  T* ptr_;
  size_t size_;

public:
  explicit UnifiedMemory(size_t count) : size_(count * sizeof(T)) {
    CUDA_CHECK_ERROR(cudaMallocManaged(&ptr_, size_));
  }

  ~UnifiedMemory() {
    if (ptr_) {
      cudaFree(ptr_);
    }
  }

  T* get() const { return ptr_; }
  size_t size() const { return size_; }

  void prefetch_to_device(int device = 0) {
    CUDA_CHECK_ERROR(cudaMemPrefetchAsync(ptr_, size_, device));
  }

  void prefetch_to_host() {
    CUDA_CHECK_ERROR(cudaMemPrefetchAsync(ptr_, size_, cudaCpuDeviceId));
  }
};

} // namespace cuda_utils
```

#### Stream and Event Management

- Use CUDA streams for concurrent execution
- Implement proper synchronization with events
- Leverage asynchronous operations for performance

```cuda
// Stream management utilities
class CudaStream {
private:
  cudaStream_t stream_;

public:
  CudaStream() {
    CUDA_CHECK_ERROR(cudaStreamCreate(&stream_));
  }

  ~CudaStream() {
    if (stream_) {
      cudaStreamDestroy(stream_);
    }
  }

  // Move constructor
  CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
    other.stream_ = nullptr;
  }

  cudaStream_t get() const { return stream_; }

  void synchronize() {
    CUDA_CHECK_ERROR(cudaStreamSynchronize(stream_));
  }

  bool is_complete() {
    cudaError_t status = cudaStreamQuery(stream_);
    if (status == cudaSuccess) return true;
    if (status == cudaErrorNotReady) return false;
    CUDA_THROW_ON_ERROR(status);
    return false;
  }
};

// Event management
class CudaEvent {
private:
  cudaEvent_t event_;

public:
  CudaEvent() {
    CUDA_CHECK_ERROR(cudaEventCreate(&event_));
  }

  ~CudaEvent() {
    if (event_) {
      cudaEventDestroy(event_);
    }
  }

  void record(cudaStream_t stream = 0) {
    CUDA_CHECK_ERROR(cudaEventRecord(event_, stream));
  }

  void wait(cudaStream_t stream = 0) {
    CUDA_CHECK_ERROR(cudaStreamWaitEvent(stream, event_, 0));
  }

  float elapsed_time(const CudaEvent& start) const {
    float ms;
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&ms, start.event_, event_));
    return ms;
  }
};
```

### Documentation Standards

#### CUDA Doc Comments

```cuda
/**
 * @file matrix_operations.cuh
 * @brief High-performance matrix operations using CUDA
 *
 * This file contains optimized CUDA kernels and host functions for
 * matrix operations including multiplication, addition, and reduction.
 * All kernels are designed for maximum memory bandwidth utilization
 * and computational throughput.
 */

/**
 * @brief Performs matrix multiplication C = A * B using CUDA
 *
 * This function launches an optimized CUDA kernel that uses shared memory
 * tiling to maximize memory bandwidth and computational efficiency.
 *
 * @tparam T Data type (float, double, int, etc.)
 * @param a Input matrix A (m x k) in row-major order
 * @param b Input matrix B (k x n) in row-major order
 * @param c Output matrix C (m x n) in row-major order
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A and rows in B
 * @param stream CUDA stream for asynchronous execution (default: 0)
 *
 * @pre All matrices must be allocated on device memory
 * @pre Dimensions must be positive and matrices must be compatible
 * @post Matrix C contains the result of A * B
 *
 * @throws CudaException if CUDA operations fail
 *
 * @note This function assumes row-major matrix storage
 * @note For best performance, ensure matrix dimensions are multiples of 16
 *
 * Example usage:
 * @code{.cpp}
 * DeviceMemory<float> d_a(m * k);
 * DeviceMemory<float> d_b(k * n);
 * DeviceMemory<float> d_c(m * n);
 *
 * // ... initialize matrices ...
 *
 * matrix_multiply(d_a.get(), d_b.get(), d_c.get(), m, n, k);
 * cudaDeviceSynchronize();
 * @endcode
 */
template<typename T>
void matrix_multiply(const T* a, const T* b, T* c,
                    int m, int n, int k, cudaStream_t stream = 0);

/**
 * @brief CUDA kernel for matrix multiplication with shared memory optimization
 *
 * This kernel implements tiled matrix multiplication using shared memory
 * to reduce global memory access and improve performance.
 *
 * @tparam T Data type for matrix elements
 * @tparam TILE_SIZE Size of the shared memory tile (must be power of 2)
 * @param a Input matrix A in device memory
 * @param b Input matrix B in device memory
 * @param c Output matrix C in device memory
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A and rows in B
 *
 * @note This kernel should be launched with block dimensions (TILE_SIZE, TILE_SIZE)
 * @note Grid dimensions should be ((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE)
 */
template<typename T, int TILE_SIZE = 16>
__global__ void matrix_multiply_shared_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ c,
    int m, int n, int k);
```

**Required Documentation**:

- All public functions and kernels
- All classes and templates
- All files (file-level comments)
- Complex algorithms with performance considerations
- Memory access patterns and optimization strategies

## Test-Driven Development (TDD) for CUDA

### Core TDD Requirements

**MANDATORY TDD WORKFLOW:**

1. **RED PHASE**: Write failing tests FIRST (both host and device tests)
2. **GREEN PHASE**: Write minimal CUDA code to make tests pass
3. **REFACTOR PHASE**: Optimize performance while keeping tests green

**Testing Requirements:**

- Write unit tests for ALL kernels and host functions
- Write integration tests for complete GPU workflows
- Write performance tests to verify optimization goals
- Write memory correctness tests (bounds checking, race conditions)
- Maintain minimum 80% code coverage (excluding CUDA runtime calls)
- All tests must pass before proceeding to next task

### Testing Framework and Structure

- **Framework**: Google Test (gtest) + custom CUDA test utilities
- **Performance Testing**: Google Benchmark + CUDA events
- **Memory Testing**: CUDA-memcheck integration
- **Property Testing**: Custom GPU property testing framework
- **Structure**: `tests/` directory with host and device test separation

### Test Organization

```cpp
// Project structure
src/
├── kernels/
│   ├── matrix_ops.cu
│   └── reduction.cu
├── host/
│   ├── cuda_manager.cpp
│   └── memory_manager.cpp
└── utils/
    └── cuda_utils.cuh

tests/
├── unit/
│   ├── test_matrix_kernels.cu
│   ├── test_reduction_kernels.cu
│   └── test_memory_manager.cpp
├── integration/
│   ├── test_matrix_pipeline.cu
│   └── test_multi_gpu.cu
├── performance/
│   ├── benchmark_matrix_ops.cu
│   └── benchmark_memory.cu
└── utils/
    ├── cuda_test_utils.cuh
    └── test_data_generator.cuh
```

### CUDA Test Utilities

```cuda
// tests/utils/cuda_test_utils.cuh

#pragma once

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cmath>

namespace cuda_test_utils {

// Test fixture for CUDA tests
class CudaTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize CUDA device
    int device_count;
    CUDA_CHECK_ERROR(cudaGetDeviceCount(&device_count));
    ASSERT_GT(device_count, 0) << "No CUDA devices available";

    CUDA_CHECK_ERROR(cudaSetDevice(0));

    // Get device properties
    CUDA_CHECK_ERROR(cudaGetDeviceProperties(&device_props_, 0));
  }

  void TearDown() override {
    // Reset device to clean up any remaining state
    CUDA_CHECK_ERROR(cudaDeviceReset());
  }

  cudaDeviceProperties device_props_;
};

// Floating point comparison for GPU results
template<typename T>
bool are_close(T a, T b, T tolerance = static_cast<T>(1e-5)) {
  return std::abs(a - b) <= tolerance * std::max(std::abs(a), std::abs(b));
}

// Vector comparison with tolerance
template<typename T>
bool vectors_are_close(const std::vector<T>& a, const std::vector<T>& b,
                      T tolerance = static_cast<T>(1e-5)) {
  if (a.size() != b.size()) return false;

  for (size_t i = 0; i < a.size(); ++i) {
    if (!are_close(a[i], b[i], tolerance)) {
      return false;
    }
  }
  return true;
}

// Test data generator
template<typename T>
class TestDataGenerator {
private:
  std::mt19937 gen_;
  std::uniform_real_distribution<T> dist_;

public:
  TestDataGenerator(T min_val = T(-1), T max_val = T(1), unsigned seed = 42)
    : gen_(seed), dist_(min_val, max_val) {}

  std::vector<T> generate_vector(size_t size) {
    std::vector<T> data(size);
    for (auto& val : data) {
      val = dist_(gen_);
    }
    return data;
  }

  std::vector<T> generate_matrix(size_t rows, size_t cols) {
    return generate_vector(rows * cols);
  }
};

// GPU memory checker
class GpuMemoryChecker {
public:
  static size_t get_free_memory() {
    size_t free, total;
    CUDA_CHECK_ERROR(cudaMemGetInfo(&free, &total));
    return free;
  }

  static void check_no_memory_leak(std::function<void()> test_func) {
    size_t initial_free = get_free_memory();
    test_func();
    cudaDeviceSynchronize();
    size_t final_free = get_free_memory();

    EXPECT_EQ(initial_free, final_free)
      << "Memory leak detected: " << (initial_free - final_free) << " bytes";
  }
};

// Performance timer using CUDA events
class CudaTimer {
private:
  cudaEvent_t start_event_, stop_event_;

public:
  CudaTimer() {
    CUDA_CHECK_ERROR(cudaEventCreate(&start_event_));
    CUDA_CHECK_ERROR(cudaEventCreate(&stop_event_));
  }

  ~CudaTimer() {
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
  }

  void start() {
    CUDA_CHECK_ERROR(cudaEventRecord(start_event_));
  }

  float stop() {
    CUDA_CHECK_ERROR(cudaEventRecord(stop_event_));
    CUDA_CHECK_ERROR(cudaEventSynchronize(stop_event_));

    float elapsed_ms;
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_));
    return elapsed_ms;
  }
};

} // namespace cuda_test_utils
```

### Unit Testing Patterns

```cuda
// tests/unit/test_matrix_kernels.cu

#include <gtest/gtest.h>
#include "cuda_test_utils.cuh"
#include "matrix_operations.cuh"
#include "device_memory.cuh"

using namespace cuda_test_utils;

class MatrixKernelTest : public CudaTest {
protected:
  void SetUp() override {
    CudaTest::SetUp();
    data_gen_ = std::make_unique<TestDataGenerator<float>>(0.0f, 1.0f);
  }

  std::unique_ptr<TestDataGenerator<float>> data_gen_;
};

TEST_F(MatrixKernelTest, MatrixMultiplyBasicFunctionality) {
  // Arrange - TDD RED PHASE: Test fails because kernel doesn't exist yet
  const int m = 4, n = 4, k = 4;

  // Generate test data
  auto h_a = data_gen_->generate_matrix(m, k);
  auto h_b = data_gen_->generate_matrix(k, n);
  std::vector<float> h_c(m * n, 0.0f);
  std::vector<float> h_c_expected(m * n, 0.0f);

  // Compute expected result on CPU
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int l = 0; l < k; ++l) {
        h_c_expected[i * n + j] += h_a[i * k + l] * h_b[l * n + j];
      }
    }
  }

  // Allocate device memory
  DeviceMemory<float> d_a(m * k);
  DeviceMemory<float> d_b(k * n);
  DeviceMemory<float> d_c(m * n);

  // Copy data to device
  d_a.copy_from_host(h_a.data(), h_a.size());
  d_b.copy_from_host(h_b.data(), h_b.size());

  // Act - Call the kernel (this will fail initially in RED phase)
  matrix_multiply(d_a.get(), d_b.get(), d_c.get(), m, n, k);
  CUDA_CHECK_ERROR(cudaDeviceSynchronize());

  // Copy result back to host
  d_c.copy_to_host(h_c.data(), h_c.size());

  // Assert
  EXPECT_TRUE(vectors_are_close(h_c, h_c_expected, 1e-5f))
    << "Matrix multiplication result doesn't match expected values";
}

TEST_F(MatrixKernelTest, MatrixMultiplyEmptyMatrix) {
  // Test edge case: empty matrices
  const int m = 0, n = 0, k = 0;

  // This should not crash and handle gracefully
  EXPECT_NO_THROW({
    matrix_multiply<float>(nullptr, nullptr, nullptr, m, n, k);
    cudaDeviceSynchronize();
  });
}

TEST_F(MatrixKernelTest, MatrixMultiplyLargeMatrix) {
  // Test with larger matrices to verify performance and correctness
  const int m = 128, n = 128, k = 128;

  auto h_a = data_gen_->generate_matrix(m, k);
  auto h_b = data_gen_->generate_matrix(k, n);
  std::vector<float> h_c(m * n);

  DeviceMemory<float> d_a(m * k);
  DeviceMemory<float> d_b(k * n);
  DeviceMemory<float> d_c(m * n);

  d_a.copy_from_host(h_a.data(), h_a.size());
  d_b.copy_from_host(h_b.data(), h_b.size());

  // Time the operation
  CudaTimer timer;
  timer.start();
  matrix_multiply(d_a.get(), d_b.get(), d_c.get(), m, n, k);
  float elapsed_ms = timer.stop();

  d_c.copy_to_host(h_c.data(), h_c.size());

  // Verify performance is reasonable (should be much faster than CPU)
  EXPECT_LT(elapsed_ms, 10.0f) << "Matrix multiplication took too long: "
                               << elapsed_ms << " ms";

  // Verify no NaN or infinite values
  for (const auto& val : h_c) {
    EXPECT_TRUE(std::isfinite(val)) << "Result contains non-finite values";
  }
}

TEST_F(MatrixKernelTest, MatrixMultiplyMemoryLeak) {
  // Test for memory leaks
  const int m = 32, n = 32, k = 32;

  GpuMemoryChecker::check_no_memory_leak([&]() {
    auto h_a = data_gen_->generate_matrix(m, k);
    auto h_b = data_gen_->generate_matrix(k, n);

    DeviceMemory<float> d_a(m * k);
    DeviceMemory<float> d_b(k * n);
    DeviceMemory<float> d_c(m * n);

    d_a.copy_from_host(h_a.data(), h_a.size());
    d_b.copy_from_host(h_b.data(), h_b.size());

    matrix_multiply(d_a.get(), d_b.get(), d_c.get(), m, n, k);
    cudaDeviceSynchronize();
  });
}

// Property-based testing
TEST_F(MatrixKernelTest, MatrixMultiplyProperties) {
  // Test mathematical properties
  const int size = 16;

  auto h_a = data_gen_->generate_matrix(size, size);
  auto h_identity = std::vector<float>(size * size, 0.0f);

  // Create identity matrix
  for (int i = 0; i < size; ++i) {
    h_identity[i * size + i] = 1.0f;
  }

  std::vector<float> h_result(size * size);

  DeviceMemory<float> d_a(size * size);
  DeviceMemory<float> d_identity(size * size);
  DeviceMemory<float> d_result(size * size);

  d_a.copy_from_host(h_a.data(), h_a.size());
  d_identity.copy_from_host(h_identity.data(), h_identity.size());

  // Test: A * I = A
  matrix_multiply(d_a.get(), d_identity.get(), d_result.get(), size, size, size);
  cudaDeviceSynchronize();

  d_result.copy_to_host(h_result.data(), h_result.size());

  EXPECT_TRUE(vectors_are_close(h_a, h_result, 1e-5f))
    << "Matrix multiplication with identity failed";
}
```

### Integration Testing

```cuda
// tests/integration/test_matrix_pipeline.cu

#include <gtest/gtest.h>
#include "cuda_test_utils.cuh"
#include "matrix_pipeline.cuh"

class MatrixPipelineTest : public CudaTest {
protected:
  void SetUp() override {
    CudaTest::SetUp();
    pipeline_ = std::make_unique<MatrixPipeline>();
  }

  std::unique_ptr<MatrixPipeline> pipeline_;
};

TEST_F(MatrixPipelineTest, CompleteMatrixWorkflow) {
  // Test complete pipeline: load -> process -> multiply -> reduce
  const int matrix_size = 64;

  // Create test matrices
  TestDataGenerator<float> gen(0.0f, 1.0f);
  auto input_a = gen.generate_matrix(matrix_size, matrix_size);
  auto input_b = gen.generate_matrix(matrix_size, matrix_size);

  // Run complete pipeline
  auto result = pipeline_->process_matrices(input_a, input_b, matrix_size);

  // Verify pipeline completed successfully
  ASSERT_TRUE(result.success) << "Pipeline failed: " << result.error_message;
  EXPECT_GT(result.final_value, 0.0f);
  EXPECT_LT(result.processing_time_ms, 100.0f);
}

TEST_F(MatrixPipelineTest, MultiStreamPipeline) {
  // Test pipeline with multiple CUDA streams
  const int num_streams = 4;
  const int matrix_size = 32;

  std::vector<CudaStream> streams(num_streams);
  std::vector<std::future<float>> results;

  TestDataGenerator<float> gen(0.0f, 1.0f);

  // Launch multiple parallel operations
  for (int i = 0; i < num_streams; ++i) {
    auto input_a = gen.generate_matrix(matrix_size, matrix_size);
    auto input_b = gen.generate_matrix(matrix_size, matrix_size);

    results.push_back(std::async(std::launch::async, [&, i]() {
      return pipeline_->process_matrices_async(
        input_a, input_b, matrix_size, streams[i].get()
      );
    }));
  }

  // Collect results
  for (auto& future : results) {
    float result = future.get();
    EXPECT_TRUE(std::isfinite(result));
  }
}
```

### Performance Testing

```cuda
// tests/performance/benchmark_matrix_ops.cu

#include <benchmark/benchmark.h>
#include "cuda_test_utils.cuh"
#include "matrix_operations.cuh"

using namespace cuda_test_utils;

class MatrixBenchmark : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State& state) override {
    // Initialize CUDA
    cudaSetDevice(0);

    // Generate test data
    int size = state.range(0);
    TestDataGenerator<float> gen(0.0f, 1.0f);

    h_a = gen.generate_matrix(size, size);
    h_b = gen.generate_matrix(size, size);

    // Allocate device memory
    d_a = std::make_unique<DeviceMemory<float>>(size * size);
    d_b = std::make_unique<DeviceMemory<float>>(size * size);
    d_c = std::make_unique<DeviceMemory<float>>(size * size);

    // Copy data to device
    d_a->copy_from_host(h_a.data(), h_a.size());
    d_b->copy_from_host(h_b.data(), h_b.size());
  }

  void TearDown(const benchmark::State& state) override {
    cudaDeviceReset();
  }

protected:
  std::vector<float> h_a, h_b;
  std::unique_ptr<DeviceMemory<float>> d_a, d_b, d_c;
};

BENCHMARK_DEFINE_F(MatrixBenchmark, MatrixMultiply)(benchmark::State& state) {
  int size = state.range(0);

  for (auto _ : state) {
    // Benchmark the matrix multiplication
    CudaTimer timer;
    timer.start();

    matrix_multiply(d_a->get(), d_b->get(), d_c->get(), size, size, size);

    float elapsed_ms = timer.stop();
    state.SetIterationTime(elapsed_ms / 1000.0); // Convert to seconds
  }

  // Calculate FLOPS
  double flops = 2.0 * size * size * size; // 2*N^3 operations
  state.counters["GFLOPS"] = benchmark::Counter(
    flops, benchmark::Counter::kIsRate, benchmark::Counter::kIs1000
  );

  state.SetItemsProcessed(state.iterations() * size * size);
}

// Register benchmarks for different matrix sizes
BENCHMARK_REGISTER_F(MatrixBenchmark, MatrixMultiply)
  ->RangeMultiplier(2)
  ->Range(64, 2048)
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond);

// Memory bandwidth benchmark
static void BM_MemoryBandwidth(benchmark::State& state) {
  int size = state.range(0);
  size_t bytes = size * sizeof(float);

  DeviceMemory<float> d_src(size);
  DeviceMemory<float> d_dst(size);

  for (auto _ : state) {
    CudaTimer timer;
    timer.start();

    CUDA_CHECK_ERROR(cudaMemcpy(d_dst.get(), d_src.get(), bytes, cudaMemcpyDeviceToDevice));

    float elapsed_ms = timer.stop();
    state.SetIterationTime(elapsed_ms / 1000.0);
  }

  state.counters["Bandwidth_GB/s"] = benchmark::Counter(
    bytes, benchmark::Counter::kIsRate, benchmark::Counter::kIs1024
  );
}

BENCHMARK(BM_MemoryBandwidth)
  ->Range(1<<20, 1<<28)
  ->UseManualTime();
```

## Length Restrictions

### Files

- **Maximum Lines**: 750
- **Enforcement**: Strict
- **Exceptions**:
  - Generated CUDA code (thrust algorithms, cooperative groups)
  - Large kernel implementations with extensive shared memory optimization
  - Comprehensive integration test files

**Refactoring Strategies**:

- Split kernels into separate `.cu` files
- Extract device functions to separate compilation units
- Use header files (`.cuh`) for device function declarations
- Move host utility functions to separate `.cpp` files
- Extract template specializations to separate files

### Functions and Kernels

- **Maximum Lines per Kernel**: 100
- **Maximum Lines per Host Function**: 80
- **Enforcement**: Strict
- **Exceptions**:
  - Complex optimized kernels with detailed comments
  - Large template instantiations
  - Generated code from CUDA libraries

**Refactoring Strategies**:

- Extract device helper functions
- Use template specialization for different data types
- Break complex kernels into phases
- Use cooperative groups for complex synchronization
- Extract shared memory initialization to separate functions

### Classes and Templates

- **Maximum Methods per Class**: 20
- **Maximum Template Parameters**: 8
- **Guidance**: Prefer composition over inheritance, use CUDA-aware RAII patterns

## Linting and Static Analysis

### Primary Tools

- **Primary Linter**: clang-tidy with CUDA support
- **Code Formatter**: clang-format
- **Static Analysis**: PVS-Studio, Cppcheck with CUDA extensions
- **CUDA-Specific**: nvcc compiler warnings, CUDA-memcheck

### clang-format Configuration

**Configuration** (`.clang-format`):

```yaml
Language: Cpp
BasedOnStyle: Google
AccessModifierOffset: -2
AlignAfterOpenBracket: Align
AlignConsecutiveMacros: true
AlignConsecutiveAssignments: false
AlignEscapedNewlines: Left
AlignOperands: true
AlignTrailingComments: true
AllowAllArgumentsOnNextLine: true
AllowAllConstructorInitializersOnNextLine: true
AllowAllParametersOfDeclarationOnNextLine: true
AllowShortBlocksOnASingleLine: Never
AllowShortCaseLabelsOnASingleLine: false
AllowShortFunctionsOnASingleLine: All
AllowShortLambdasOnASingleLine: All
AllowShortIfStatementsOnASingleLine: WithoutElse
AllowShortLoopsOnASingleLine: true
AlwaysBreakAfterDefinitionReturnType: None
AlwaysBreakAfterReturnType: None
AlwaysBreakBeforeMultilineStrings: true
AlwaysBreakTemplateDeclarations: Yes
BinPackArguments: true
BinPackParameters: true
BraceWrapping:
  AfterCaseLabel: false
  AfterClass: false
  AfterControlStatement: Never
  AfterEnum: false
  AfterFunction: false
  AfterNamespace: false
  AfterObjCDeclaration: false
  AfterStruct: false
  AfterUnion: false
  AfterExternBlock: false
  BeforeCatch: false
  BeforeElse: false
  BeforeLambdaBody: false
  BeforeWhile: false
  IndentBraces: false
  SplitEmptyFunction: true
  SplitEmptyRecord: true
  SplitEmptyNamespace: true
BreakBeforeBinaryOperators: None
BreakBeforeBraces: Attach
BreakBeforeInheritanceComma: false
BreakInheritanceList: BeforeColon
BreakBeforeTernaryOperators: true
BreakConstructorInitializersBeforeComma: false
BreakConstructorInitializers: BeforeColon
BreakAfterJavaFieldAnnotations: false
BreakStringLiterals: true
ColumnLimit: 100
CommentPragmas: "^ IWYU pragma:"
CompactNamespaces: false
ConstructorInitializerAllOnOneLineOrOnePerLine: true
ConstructorInitializerIndentWidth: 4
ContinuationIndentWidth: 4
Cpp11BracedListStyle: true
DeriveLineEnding: true
DerivePointerAlignment: true
DisableFormat: false
ExperimentalAutoDetectBinPacking: false
FixNamespaceComments: true
ForEachMacros:
  - foreach
  - Q_FOREACH
  - BOOST_FOREACH
IncludeBlocks: Regroup
IncludeCategories:
  - Regex: '^<ext/.*\.h>'
    Priority: 2
    SortPriority: 0
  - Regex: '^<.*\.h>'
    Priority: 1
    SortPriority: 0
  - Regex: "^<.*"
    Priority: 2
    SortPriority: 0
  - Regex: "^<cuda.*"
    Priority: 3
    SortPriority: 0
  - Regex: "^<thrust.*"
    Priority: 4
    SortPriority: 0
  - Regex: ".*"
    Priority: 5
    SortPriority: 0
IncludeIsMainRegex: "([-_](test|unittest))?$"
IncludeIsMainSourceRegex: ""
IndentCaseLabels: true
IndentGotoLabels: true
IndentPPDirectives: None
IndentWidth: 2
IndentWrappedFunctionNames: false
InsertTrailingCommas: None
JavaScriptQuotes: Leave
JavaScriptWrapImports: true
KeepEmptyLinesAtTheStartOfBlocks: false
MacroBlockBegin: ""
MacroBlockEnd: ""
MaxEmptyLinesToKeep: 1
NamespaceIndentation: None
ObjCBinPackProtocolList: Never
ObjCBlockIndentWidth: 2
ObjCBreakBeforeNestedBlockParam: true
ObjCSpaceAfterProperty: false
ObjCSpaceBeforeProtocolList: true
PenaltyBreakAssignment: 2
PenaltyBreakBeforeFirstCallParameter: 1
PenaltyBreakComment: 300
PenaltyBreakFirstLessLess: 120
PenaltyBreakString: 1000
PenaltyBreakTemplateDeclaration: 10
PenaltyExcessCharacter: 1000000
PenaltyReturnTypeOnItsOwnLine: 200
PointerAlignment: Left
ReflowComments: true
SortIncludes: true
SortUsingDeclarations: true
SpaceAfterCStyleCast: false
SpaceAfterLogicalNot: false
SpaceAfterTemplateKeyword: true
SpaceBeforeAssignmentOperators: true
SpaceBeforeCpp11BracedList: false
SpaceBeforeCtorInitializerColon: true
SpaceBeforeInheritanceColon: true
SpaceBeforeParens: ControlStatements
SpaceBeforeRangeBasedForLoopColon: true
SpaceInEmptyBlock: false
SpaceInEmptyParentheses: false
SpacesBeforeTrailingComments: 2
SpacesInAngles: false
SpacesInConditionalStatement: false
SpacesInContainerLiterals: true
SpacesInCStyleCastParentheses: false
SpacesInParentheses: false
SpacesInSquareBrackets: false
SpaceBeforeSquareBrackets: false
Standard: Auto
StatementMacros:
  - Q_UNUSED
  - QT_REQUIRE_VERSION
TabWidth: 8
UseCRLF: false
UseTab: Never
```

### clang-tidy Configuration

**Configuration** (`.clang-tidy`):

```yaml
Checks: >
  -*,
  bugprone-*,
  cert-*,
  clang-analyzer-*,
  cppcoreguidelines-*,
  google-*,
  hicpp-*,
  llvm-*,
  misc-*,
  modernize-*,
  performance-*,
  portability-*,
  readability-*,
  -bugprone-easily-swappable-parameters,
  -cppcoreguidelines-avoid-magic-numbers,
  -cppcoreguidelines-pro-bounds-pointer-arithmetic,
  -cppcoreguidelines-pro-type-reinterpret-cast,
  -google-readability-todo,
  -hicpp-avoid-magic-numbers,
  -llvm-header-guard,
  -misc-non-private-member-variables-in-classes,
  -modernize-use-trailing-return-type,
  -readability-magic-numbers

WarningsAsErrors: >
  bugprone-use-after-move,
  cert-err34-c,
  cert-err52-cpp,
  cert-err60-cpp,
  clang-analyzer-core.DivideZero,
  clang-analyzer-core.NullDereference,
  cppcoreguidelines-init-variables,
  cppcoreguidelines-slicing,
  google-explicit-constructor,
  misc-unused-parameters,
  performance-*

CheckOptions:
  - key: readability-identifier-naming.NamespaceCase
    value: snake_case
  - key: readability-identifier-naming.ClassCase
    value: CamelCase
  - key: readability-identifier-naming.StructCase
    value: CamelCase
  - key: readability-identifier-naming.FunctionCase
    value: snake_case
  - key: readability-identifier-naming.VariableCase
    value: snake_case
  - key: readability-identifier-naming.ConstantCase
    value: UPPER_CASE
  - key: readability-identifier-naming.EnumCase
    value: CamelCase
  - key: readability-identifier-naming.EnumConstantCase
    value: UPPER_CASE
  - key: readability-identifier-naming.GlobalConstantCase
    value: UPPER_CASE
  - key: readability-identifier-naming.MemberCase
    value: snake_case
  - key: readability-identifier-naming.PrivateMemberSuffix
    value: _
  - key: readability-identifier-naming.ProtectedMemberSuffix
    value: _
  - key: cppcoreguidelines-special-member-functions.AllowSoleDefaultDtor
    value: true
  - key: performance-for-range-copy.WarnOnAllAutoCopies
    value: true
```

### CUDA-Specific Compiler Flags

**CMake Configuration**:

```cmake
# CMakeLists.txt for CUDA project
cmake_minimum_required(VERSION 3.18)
project(CudaProject LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

# Compiler-specific options
if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Werror>
    )
endif()

# CUDA compiler options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror cross-execution-space-call")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror reorder")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror deprecated-declarations")

# Debug flags
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")

# Release flags
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Create main library
add_library(${PROJECT_NAME} STATIC
    src/kernels/matrix_ops.cu
    src/kernels/reduction.cu
    src/host/cuda_manager.cpp
    src/host/memory_manager.cpp
    src/utils/cuda_utils.cu
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::curand
)

# Testing
enable_testing()
find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)

# Unit tests
add_executable(unit_tests
    tests/unit/test_matrix_kernels.cu
    tests/unit/test_reduction_kernels.cu
    tests/unit/test_memory_manager.cpp
)

target_link_libraries(unit_tests
    PRIVATE
        ${PROJECT_NAME}
        GTest::gtest_main
        CUDA::cudart
)

# Integration tests
add_executable(integration_tests
    tests/integration/test_matrix_pipeline.cu
    tests/integration/test_multi_gpu.cu
)

target_link_libraries(integration_tests
    PRIVATE
        ${PROJECT_NAME}
        GTest::gtest_main
        CUDA::cudart
)

# Performance benchmarks
add_executable(benchmarks
    tests/performance/benchmark_matrix_ops.cu
    tests/performance/benchmark_memory.cu
)

target_link_libraries(benchmarks
    PRIVATE
        ${PROJECT_NAME}
        benchmark::benchmark_main
        CUDA::cudart
)

# Add tests to CTest
add_test(NAME UnitTests COMMAND unit_tests)
add_test(NAME IntegrationTests COMMAND integration_tests)
add_test(NAME Benchmarks COMMAND benchmarks --benchmark_min_time=1)
```

### Makefile Integration

**Makefile**:

```makefile
# CUDA project Makefile

# Compiler settings
NVCC := nvcc
CXX := g++
CUDA_ARCH := -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80

# Directories
SRC_DIR := src
TEST_DIR := tests
BUILD_DIR := build
BIN_DIR := bin

# Flags
NVCC_FLAGS := -std=c++17 --extended-lambda --expt-relaxed-constexpr -Werror cross-execution-space-call
CXX_FLAGS := -std=c++17 -Wall -Wextra -Wpedantic -Werror

# Debug/Release flags
DEBUG_FLAGS := -g -G -O0 -DDEBUG
RELEASE_FLAGS := -O3 -DNDEBUG

# Libraries
CUDA_LIBS := -lcudart -lcublas -lcurand
TEST_LIBS := -lgtest -lgtest_main -lpthread
BENCHMARK_LIBS := -lbenchmark -lpthread

# Source files
CUDA_SOURCES := $(wildcard $(SRC_DIR)/**/*.cu)
CPP_SOURCES := $(wildcard $(SRC_DIR)/**/*.cpp)
TEST_SOURCES := $(wildcard $(TEST_DIR)/**/*.cu) $(wildcard $(TEST_DIR)/**/*.cpp)

# Object files
CUDA_OBJECTS := $(CUDA_SOURCES:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)
CPP_OBJECTS := $(CPP_SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

.PHONY: all clean test benchmark lint format check debug release

# Default target
all: release

# Debug build
debug: NVCC_FLAGS += $(DEBUG_FLAGS)
debug: CXX_FLAGS += $(DEBUG_FLAGS)
debug: $(BIN_DIR)/cuda_project

# Release build
release: NVCC_FLAGS += $(RELEASE_FLAGS)
release: CXX_FLAGS += $(RELEASE_FLAGS)
release: $(BIN_DIR)/cuda_project

# Create directories
$(BUILD_DIR) $(BIN_DIR):
	mkdir -p $@

# Compile CUDA sources
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) -c $< -o $@

# Compile C++ sources
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Link main executable
$(BIN_DIR)/cuda_project: $(CUDA_OBJECTS) $(CPP_OBJECTS) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $^ -o $@ $(CUDA_LIBS)

# Format code
format:
	find $(SRC_DIR) $(TEST_DIR) -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.hpp" | \
		xargs clang-format -i

# Check formatting
format-check:
	find $(SRC_DIR) $(TEST_DIR) -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.hpp" | \
		xargs clang-format --dry-run --Werror

# Lint code
lint:
	find $(SRC_DIR) -name "*.cpp" -o -name "*.hpp" | \
		xargs clang-tidy --config-file=.clang-tidy

# Compile tests
test-compile: $(BIN_DIR)/unit_tests $(BIN_DIR)/integration_tests

$(BIN_DIR)/unit_tests: $(CUDA_OBJECTS) $(CPP_OBJECTS) $(wildcard $(TEST_DIR)/unit/*.cu) $(wildcard $(TEST_DIR)/unit/*.cpp) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $^ -o $@ $(CUDA_LIBS) $(TEST_LIBS)

$(BIN_DIR)/integration_tests: $(CUDA_OBJECTS) $(CPP_OBJECTS) $(wildcard $(TEST_DIR)/integration/*.cu) $(wildcard $(TEST_DIR)/integration/*.cpp) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $^ -o $@ $(CUDA_LIBS) $(TEST_LIBS)

# Run tests
test: test-compile
	@echo "Running unit tests..."
	$(BIN_DIR)/unit_tests
	@echo "Running integration tests..."
	$(BIN_DIR)/integration_tests

# Memory check
memcheck: test-compile
	cuda-memcheck --tool memcheck $(BIN_DIR)/unit_tests
	cuda-memcheck --tool racecheck $(BIN_DIR)/unit_tests

# Compile benchmarks
benchmark-compile: $(BIN_DIR)/benchmarks

$(BIN_DIR)/benchmarks: $(CUDA_OBJECTS) $(CPP_OBJECTS) $(wildcard $(TEST_DIR)/performance/*.cu) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $^ -o $@ $(CUDA_LIBS) $(BENCHMARK_LIBS)

# Run benchmarks
benchmark: benchmark-compile
	$(BIN_DIR)/benchmarks --benchmark_format=json > benchmark_results.json

# Complete quality check
check: format-check lint test memcheck
	@echo "All quality checks passed!"

# TDD workflow - run after each change
tdd: format test
	@echo "TDD cycle complete - tests passing!"

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Install development tools
install-tools:
	@echo "Installing CUDA development tools..."
	sudo apt-get update
	sudo apt-get install -y clang-format clang-tidy
	pip install pre-commit

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Build release version (default)"
	@echo "  debug        - Build debug version"
	@echo "  release      - Build release version"
	@echo "  test         - Run all tests"
	@echo "  benchmark    - Run performance benchmarks"
	@echo "  memcheck     - Run CUDA memory checker"
	@echo "  format       - Format all source files"
	@echo "  format-check - Check code formatting"
	@echo "  lint         - Run static analysis"
	@echo "  check        - Run all quality checks"
	@echo "  tdd          - Quick TDD cycle (format + test)"
	@echo "  clean        - Clean build artifacts"
	@echo "  help         - Show this help message"
```

## Quality Gates

### Pre-commit Requirements (Enforced by TDD)

- Code must be formatted with clang-format
- All clang-tidy warnings must be resolved
- All unit tests must pass (100% success rate)
- Integration tests must pass
- Memory checks must pass (no leaks, no race conditions)
- CUDA compilation must succeed with zero warnings
- Performance benchmarks must not regress by more than 5%

### Continuous Integration

```yaml
# .github/workflows/cuda-ci.yml
name: CUDA CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  format-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install clang-format
        run: sudo apt-get install -y clang-format
      - name: Check formatting
        run: make format-check

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install clang-tidy
        run: sudo apt-get install -y clang-tidy
      - name: Run linter
        run: make lint

  build-and-test:
    runs-on: ubuntu-latest
    container:
      image: nvidia/cuda:12.0-devel-ubuntu20.04

    steps:
      - uses: actions/checkout@v3

      - name: Install dependencies
        run: |
          apt-get update
          apt-get install -y cmake build-essential libgtest-dev libbenchmark-dev

      - name: Configure CMake
        run: cmake -B build -DCMAKE_BUILD_TYPE=Debug

      - name: Build
        run: cmake --build build --parallel

      - name: Run unit tests
        run: |
          cd build
          ctest --output-on-failure --parallel
        env:
          CUDA_VISIBLE_DEVICES: 0

      - name: Run memory checks
        run: |
          cuda-memcheck --tool memcheck ./build/unit_tests
          cuda-memcheck --tool racecheck ./build/unit_tests

  performance:
    runs-on: ubuntu-latest
    container:
      image: nvidia/cuda:12.0-devel-ubuntu20.04

    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: |
          apt-get update
          apt-get install -y cmake build-essential libbenchmark-dev

      - name: Build benchmarks
        run: |
          cmake -B build -DCMAKE_BUILD_TYPE=Release
          cmake --build build --target benchmarks

      - name: Run benchmarks
        run: ./build/benchmarks --benchmark_format=json > benchmark_results.json
        env:
          CUDA_VISIBLE_DEVICES: 0

      - name: Upload benchmark results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmark_results.json

  coverage:
    runs-on: ubuntu-latest
    container:
      image: nvidia/cuda:12.0-devel-ubuntu20.04

    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: |
          apt-get update
          apt-get install -y cmake build-essential lcov

      - name: Build with coverage
        run: |
          cmake -B build -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON
          cmake --build build

      - name: Generate coverage report
        run: |
          cd build
          make coverage

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: build/coverage.info
```

## Performance Optimization Guidelines

### Memory Optimization

- Use coalesced memory access patterns
- Minimize global memory transactions
- Leverage shared memory for data reuse
- Consider texture memory for read-only data
- Use pinned memory for faster host-device transfers

### Compute Optimization

- Maximize occupancy through optimal block sizes
- Minimize thread divergence in conditional code
- Use appropriate synchronization primitives
- Leverage warp-level primitives when possible
- Consider using cooperative groups for complex synchronization

### Profiling Integration

```cuda
// Performance measurement utilities
namespace cuda_profiling {

class NvtxRange {
private:
  const char* name_;

public:
  explicit NvtxRange(const char* name) : name_(name) {
    nvtxRangePushA(name_);
  }

  ~NvtxRange() {
    nvtxRangePop();
  }
};

#define NVTX_RANGE(name) cuda_profiling::NvtxRange nvtx_range(name)

// Kernel launch wrapper with profiling
template<typename... Args>
void launch_kernel_with_profiling(
    const char* kernel_name,
    dim3 grid_size,
    dim3 block_size,
    size_t shared_mem,
    cudaStream_t stream,
    void (*kernel)(Args...),
    Args... args) {

  NVTX_RANGE(kernel_name);

  CudaTimer timer;
  timer.start();

  kernel<<<grid_size, block_size, shared_mem, stream>>>(args...);

  float elapsed_ms = timer.stop();

  std::cout << "Kernel " << kernel_name << " executed in "
            << elapsed_ms << " ms" << std::endl;
}

} // namespace cuda_profiling
```

---

_This document serves as the comprehensive coding standard for CUDA C/C++ projects with mandatory Test-Driven Development. All GPU code must follow the TDD workflow: write tests FIRST, then implementation, maintaining the quality gates and performance standards defined herein._
