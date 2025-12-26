#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <iostream>

// Custom test environment for CUDA
class CudaTestEnvironment : public ::testing::Environment
{
public:
    void SetUp() override
    {
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        
        if (error != cudaSuccess || device_count == 0)
        {
            std::cerr << "No CUDA devices available. Skipping GPU tests." << std::endl;
            GTEST_SKIP();
        }
        
        // Set device and print info
        cudaSetDevice(0);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        std::cout << "Running tests on GPU: " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    }
    
    void TearDown() override
    {
        // Reset device after tests
        cudaDeviceReset();
    }
};

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new CudaTestEnvironment);
    return RUN_ALL_TESTS();
}