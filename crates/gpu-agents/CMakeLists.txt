cmake_minimum_required(VERSION 3.18)
project(gpu_agents CUDA CXX)

# Set C++ and CUDA standards (C++17 required by CUDA 13.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architecture 
# 75 = RTX 2080, 80 = A100, 86 = RTX 3090, 89 = RTX 4090, 110 = RTX 5090 (Blackwell)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90 110)

# Add compile options - Enhanced for CUDA 13.0
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>  # Enable relocatable device code for CUDA 13.0
)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

# CUDA kernels library
add_library(gpu_agent_kernels STATIC
    src/kernels/agent_kernel.cu
    src/kernels/memory_kernel.cu
    src/kernels/swarm_kernel.cu
    src/kernels/evolution_kernel.cu
    src/kernels/evolution_kernels.cu
    src/kernels/llm_kernel.cu
    src/kernels/knowledge_graph_kernel.cu
    src/kernels/knowledge_graph_fp8.cu      # FP8 optimized knowledge graph
    src/kernels/atomic_knowledge_kernel.cu
    src/kernels/consensus_kernel.cu
    src/kernels/synthesis_kernel.cu
    src/kernels/synthesis_kernel_safe.cu
    src/kernels/synthesis_kernel_fast.cu
    src/memory/memory_mgmt_kernel.cu
    # src/memory/async_memory_cuda13.cu  # CUDA 13.0 async memory management - disabled due to kernel design issues
    src/kernels/tensor_core_ops.cu     # CUDA 13.0 Tensor Core operations
    src/kernels/cluster_swarm.cu       # Thread block clusters for swarm
    # src/multi_gpu/nccl_consensus.cu    # NCCL collective operations - disabled (NCCL not installed)
    # src/multi_gpu/gpudirect_rdma.cu    # GPUDirect RDMA support - disabled (InfiniBand not installed)
    # src/neural/cublas_cudnn_ops.cu     # cuBLAS/cuDNN v9.0 operations - disabled (cuDNN not installed)
    src/jit/nvjitlink_compiler.cu      # nvJitLink JIT compilation
    src/kernels/stubs.cu
    src/consensus_kernels.cu
)

target_link_libraries(gpu_agent_kernels
    CUDA::cudart
    CUDA::curand
    CUDA::cuda_driver  # CUDA driver API for async operations
    CUDA::cublas       # cuBLAS for linear algebra
    CUDA::cublasLt     # cuBLASLt for advanced features
)

# Find and link optional libraries
find_library(CUDNN_LIB cudnn PATHS ${CUDAToolkit_LIBRARY_DIR})
if(CUDNN_LIB)
    target_link_libraries(gpu_agent_kernels ${CUDNN_LIB})
    message(STATUS "Found cuDNN for neural network operations")
endif()

find_library(NCCL_LIB nccl PATHS ${CUDAToolkit_LIBRARY_DIR})
if(NCCL_LIB)
    target_link_libraries(gpu_agent_kernels ${NCCL_LIB})
    message(STATUS "Found NCCL for multi-GPU collective operations")
endif()

# Link cudadevrt for relocatable device code (CUDA 13.0)
if(EXISTS "${CUDAToolkit_LIBRARY_DIR}/libcudadevrt.a")
    target_link_libraries(gpu_agent_kernels ${CUDAToolkit_LIBRARY_DIR}/libcudadevrt.a)
endif()

# Optional CUDA 13.0 libraries
find_library(NVRTC_LIB nvrtc PATHS ${CUDAToolkit_LIBRARY_DIR})
if(NVRTC_LIB)
    target_link_libraries(gpu_agent_kernels ${NVRTC_LIB})
    message(STATUS "Found NVRTC for runtime compilation")
endif()

find_library(NVJITLINK_LIB nvJitLink PATHS ${CUDAToolkit_LIBRARY_DIR})
if(NVJITLINK_LIB)
    target_link_libraries(gpu_agent_kernels ${NVJITLINK_LIB})
    message(STATUS "Found nvJitLink for CUDA 13.0 JIT compilation")
endif()

# Enable separable compilation and device symbol resolution
set_target_properties(gpu_agent_kernels PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Enable testing - commented out for now to simplify build
# enable_testing()
# find_package(GTest REQUIRED)

# Test executable
# add_executable(gpu_agent_tests
#     tests/test_main.cpp
#     tests/test_agent_kernel.cpp
#     tests/test_memory_kernel.cpp
#     tests/test_swarm_kernel.cpp
# )

# target_link_libraries(gpu_agent_tests
#     gpu_agent_kernels
#     GTest::GTest
#     GTest::Main
#     CUDA::cudart
# )

# add_test(NAME gpu_agent_tests COMMAND gpu_agent_tests)

# Multi-GPU Performance Tests for Phase 3
add_executable(test_multi_gpu_performance
    tests/test_multi_gpu_performance.cu
)

target_link_libraries(test_multi_gpu_performance
    gpu_agent_kernels
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    CUDA::cublasLt
)

# Link optional libraries only if found
if(NCCL_LIB)
    target_link_libraries(test_multi_gpu_performance ${NCCL_LIB})
endif()

if(CUDNN_LIB)
    target_link_libraries(test_multi_gpu_performance ${CUDNN_LIB})
endif()

if(NVRTC_LIB)
    target_link_libraries(test_multi_gpu_performance ${NVRTC_LIB})
endif()

if(NVJITLINK_LIB)
    target_link_libraries(test_multi_gpu_performance ${NVJITLINK_LIB})
endif()

set_target_properties(test_multi_gpu_performance PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Benchmarks - commented out as C++ benchmark files don't exist
# find_package(benchmark REQUIRED)
# 
# add_executable(gpu_agent_benchmarks
#     benchmarks/bench_main.cpp
#     benchmarks/bench_agent_kernel.cpp
#     benchmarks/bench_swarm_scaling.cpp
# )
# 
# target_link_libraries(gpu_agent_benchmarks
#     gpu_agent_kernels
#     benchmark::benchmark
#     CUDA::cudart
# )

# Custom targets for linting
add_custom_target(format
    COMMAND clang-format -i ${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.cu
    COMMAND clang-format -i ${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.cuh
    COMMAND clang-format -i ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp
    COMMAND clang-format -i ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.cpp
    COMMENT "Running clang-format"
)

add_custom_target(tidy
    COMMAND clang-tidy ${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.cu -p ${CMAKE_BINARY_DIR}
    COMMAND clang-tidy ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp -p ${CMAKE_BINARY_DIR}
    COMMENT "Running clang-tidy"
)