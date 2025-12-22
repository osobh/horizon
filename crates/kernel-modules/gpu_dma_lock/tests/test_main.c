/*
 * Test harness for gpu_dma_lock kernel module
 * 
 * This file contains kernel-level tests that verify the module's
 * integration with kernel subsystems.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include <linux/device.h>
#include <linux/pci.h>

#define TEST_AGENT_ID 1000
#define TEST_ALLOCATION_SIZE (1 << 20) /* 1MB */
#define TEST_GPU_DEVICE_ID 0

/* External functions from gpu_dma_lock module */
extern int gpu_dma_lock_create_agent(u64 agent_id, size_t quota);
extern int gpu_dma_lock_allocate(u64 agent_id, size_t size, u32 device_id);
extern int gpu_dma_lock_deallocate(u64 agent_id, void *addr);
extern int gpu_dma_lock_grant_dma(u64 agent_id, dma_addr_t start, dma_addr_t end, int mode);
extern int gpu_dma_lock_check_dma(u64 agent_id, dma_addr_t addr, int mode);
extern void gpu_dma_lock_remove_agent(u64 agent_id);

/* Test results */
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(condition, message) do { \
    if (!(condition)) { \
        pr_err("TEST FAILED: %s\n", message); \
        tests_failed++; \
        return -1; \
    } else { \
        tests_passed++; \
    } \
} while (0)

/* Test 1: Basic agent creation and removal */
static int test_agent_lifecycle(void)
{
    int ret;
    size_t quota = 1ULL << 30; /* 1GB */

    pr_info("Test 1: Agent lifecycle\n");

    /* Create agent */
    ret = gpu_dma_lock_create_agent(TEST_AGENT_ID, quota);
    TEST_ASSERT(ret == 0, "Failed to create agent");

    /* Try to create duplicate agent */
    ret = gpu_dma_lock_create_agent(TEST_AGENT_ID, quota);
    TEST_ASSERT(ret != 0, "Duplicate agent creation should fail");

    /* Remove agent */
    gpu_dma_lock_remove_agent(TEST_AGENT_ID);

    /* Recreate should succeed now */
    ret = gpu_dma_lock_create_agent(TEST_AGENT_ID, quota);
    TEST_ASSERT(ret == 0, "Failed to recreate agent after removal");

    /* Cleanup */
    gpu_dma_lock_remove_agent(TEST_AGENT_ID);

    return 0;
}

/* Test 2: Memory allocation and quota enforcement */
static int test_memory_allocation(void)
{
    int ret;
    size_t quota = 100 << 20; /* 100MB */
    size_t alloc_size = 50 << 20; /* 50MB */

    pr_info("Test 2: Memory allocation and quota\n");

    /* Create agent with limited quota */
    ret = gpu_dma_lock_create_agent(TEST_AGENT_ID, quota);
    TEST_ASSERT(ret == 0, "Failed to create agent");

    /* First allocation should succeed */
    ret = gpu_dma_lock_allocate(TEST_AGENT_ID, alloc_size, TEST_GPU_DEVICE_ID);
    TEST_ASSERT(ret == 0, "First allocation failed");

    /* Second allocation should succeed */
    ret = gpu_dma_lock_allocate(TEST_AGENT_ID, alloc_size, TEST_GPU_DEVICE_ID);
    TEST_ASSERT(ret == 0, "Second allocation failed");

    /* Third allocation should fail (exceeds quota) */
    ret = gpu_dma_lock_allocate(TEST_AGENT_ID, alloc_size, TEST_GPU_DEVICE_ID);
    TEST_ASSERT(ret != 0, "Allocation exceeding quota should fail");

    /* Cleanup */
    gpu_dma_lock_remove_agent(TEST_AGENT_ID);

    return 0;
}

/* Test 3: DMA access control */
static int test_dma_access_control(void)
{
    int ret;
    dma_addr_t test_addr_start = 0x100000000ULL;
    dma_addr_t test_addr_end = 0x200000000ULL;
    dma_addr_t valid_addr = 0x150000000ULL;
    dma_addr_t invalid_addr = 0x250000000ULL;

    pr_info("Test 3: DMA access control\n");

    /* Create agent */
    ret = gpu_dma_lock_create_agent(TEST_AGENT_ID, 1ULL << 30);
    TEST_ASSERT(ret == 0, "Failed to create agent");

    /* Grant DMA permission */
    ret = gpu_dma_lock_grant_dma(TEST_AGENT_ID, test_addr_start, test_addr_end, 0x3); /* RW */
    TEST_ASSERT(ret == 0, "Failed to grant DMA permission");

    /* Check valid access */
    ret = gpu_dma_lock_check_dma(TEST_AGENT_ID, valid_addr, 0x1); /* Read */
    TEST_ASSERT(ret == 0, "Valid DMA read access denied");

    ret = gpu_dma_lock_check_dma(TEST_AGENT_ID, valid_addr, 0x2); /* Write */
    TEST_ASSERT(ret == 0, "Valid DMA write access denied");

    /* Check invalid access (outside range) */
    ret = gpu_dma_lock_check_dma(TEST_AGENT_ID, invalid_addr, 0x1);
    TEST_ASSERT(ret != 0, "Invalid DMA access should be denied");

    /* Check unauthorized agent */
    ret = gpu_dma_lock_check_dma(TEST_AGENT_ID + 1, valid_addr, 0x1);
    TEST_ASSERT(ret != 0, "Unauthorized agent DMA access should be denied");

    /* Cleanup */
    gpu_dma_lock_remove_agent(TEST_AGENT_ID);

    return 0;
}

/* Test 4: Multiple agents */
static int test_multiple_agents(void)
{
    int ret, i;
    const int num_agents = 10;
    size_t quota_per_agent = 50 << 20; /* 50MB each */
    size_t alloc_size = 10 << 20; /* 10MB */

    pr_info("Test 4: Multiple agents\n");

    /* Create multiple agents */
    for (i = 0; i < num_agents; i++) {
        ret = gpu_dma_lock_create_agent(TEST_AGENT_ID + i, quota_per_agent);
        TEST_ASSERT(ret == 0, "Failed to create agent");
    }

    /* Each agent allocates memory */
    for (i = 0; i < num_agents; i++) {
        ret = gpu_dma_lock_allocate(TEST_AGENT_ID + i, alloc_size, TEST_GPU_DEVICE_ID);
        TEST_ASSERT(ret == 0, "Allocation failed for agent");
    }

    /* Cleanup all agents */
    for (i = 0; i < num_agents; i++) {
        gpu_dma_lock_remove_agent(TEST_AGENT_ID + i);
    }

    return 0;
}

/* Test 5: Invalid parameters */
static int test_invalid_parameters(void)
{
    int ret;

    pr_info("Test 5: Invalid parameters\n");

    /* Test invalid agent ID */
    ret = gpu_dma_lock_allocate(99999, 1 << 20, TEST_GPU_DEVICE_ID);
    TEST_ASSERT(ret != 0, "Allocation with invalid agent should fail");

    /* Test zero quota */
    ret = gpu_dma_lock_create_agent(TEST_AGENT_ID, 0);
    TEST_ASSERT(ret != 0, "Creating agent with zero quota should fail");

    /* Test zero allocation size */
    ret = gpu_dma_lock_create_agent(TEST_AGENT_ID, 1 << 30);
    if (ret == 0) {
        ret = gpu_dma_lock_allocate(TEST_AGENT_ID, 0, TEST_GPU_DEVICE_ID);
        TEST_ASSERT(ret != 0, "Zero size allocation should fail");
        gpu_dma_lock_remove_agent(TEST_AGENT_ID);
    }

    /* Test invalid GPU device ID */
    ret = gpu_dma_lock_create_agent(TEST_AGENT_ID, 1 << 30);
    if (ret == 0) {
        ret = gpu_dma_lock_allocate(TEST_AGENT_ID, 1 << 20, 999);
        TEST_ASSERT(ret != 0, "Allocation on invalid GPU should fail");
        gpu_dma_lock_remove_agent(TEST_AGENT_ID);
    }

    return 0;
}

/* Test 6: Memory pressure simulation */
static int test_memory_pressure(void)
{
    int ret, i;
    size_t large_quota = 10ULL << 30; /* 10GB */
    size_t chunk_size = 100 << 20; /* 100MB chunks */
    int allocations = 0;

    pr_info("Test 6: Memory pressure simulation\n");

    /* Create agent with large quota */
    ret = gpu_dma_lock_create_agent(TEST_AGENT_ID, large_quota);
    TEST_ASSERT(ret == 0, "Failed to create agent");

    /* Allocate until failure */
    for (i = 0; i < 200; i++) { /* Try up to 20GB */
        ret = gpu_dma_lock_allocate(TEST_AGENT_ID, chunk_size, TEST_GPU_DEVICE_ID);
        if (ret != 0) {
            break;
        }
        allocations++;
    }

    pr_info("Successfully allocated %d chunks (%zu MB total)\n", 
            allocations, (allocations * chunk_size) >> 20);

    /* Should have hit either quota or device limit */
    TEST_ASSERT(allocations > 0, "Should have made at least one allocation");
    TEST_ASSERT(ret != 0, "Should eventually fail due to limits");

    /* Cleanup */
    gpu_dma_lock_remove_agent(TEST_AGENT_ID);

    return 0;
}

/* Test 7: Concurrent access stress test */
static int test_concurrent_access(void)
{
    /* Note: This is a simplified version. Real concurrent testing
     * would require multiple kernel threads */
    int ret, i;
    const int iterations = 1000;
    size_t small_alloc = 1 << 10; /* 1KB */

    pr_info("Test 7: Concurrent access simulation\n");

    /* Create agent */
    ret = gpu_dma_lock_create_agent(TEST_AGENT_ID, 100 << 20);
    TEST_ASSERT(ret == 0, "Failed to create agent");

    /* Rapid allocations to test locking */
    for (i = 0; i < iterations; i++) {
        ret = gpu_dma_lock_allocate(TEST_AGENT_ID, small_alloc, TEST_GPU_DEVICE_ID);
        if (ret != 0) {
            pr_warn("Allocation %d failed\n", i);
            break;
        }
    }

    pr_info("Completed %d rapid allocations\n", i);
    TEST_ASSERT(i > 0, "Should complete at least some allocations");

    /* Cleanup */
    gpu_dma_lock_remove_agent(TEST_AGENT_ID);

    return 0;
}

/* Main test runner */
static int __init gpu_dma_lock_test_init(void)
{
    pr_info("GPU DMA Lock kernel test suite starting...\n");

    /* Run all tests */
    test_agent_lifecycle();
    test_memory_allocation();
    test_dma_access_control();
    test_multiple_agents();
    test_invalid_parameters();
    test_memory_pressure();
    test_concurrent_access();

    /* Print summary */
    pr_info("\n=== Test Summary ===\n");
    pr_info("Tests passed: %d\n", tests_passed);
    pr_info("Tests failed: %d\n", tests_failed);
    pr_info("===================\n");

    /* Return error if any test failed */
    return tests_failed > 0 ? -EFAULT : 0;
}

static void __exit gpu_dma_lock_test_exit(void)
{
    pr_info("GPU DMA Lock kernel test suite exiting\n");
}

module_init(gpu_dma_lock_test_init);
module_exit(gpu_dma_lock_test_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("StratoSwarm Team");
MODULE_DESCRIPTION("GPU DMA Lock kernel module test suite");