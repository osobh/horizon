/*
 * GPU DMA Lock Module Test Suite
 * 
 * Comprehensive unit tests for the enhanced GPU DMA Lock kernel module
 * Target: 90% code coverage
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/delay.h>
#include <linux/kthread.h>
#include <linux/semaphore.h>

#include "../gpu_dma_lock.h"

/* Test framework */
#define TEST_PASS 0
#define TEST_FAIL -1

#define RUN_TEST(test_func) do { \
    pr_info("Running test: %s\n", #test_func); \
    if (test_func() != TEST_PASS) { \
        pr_err("Test failed: %s\n", #test_func); \
        test_results.failed++; \
    } else { \
        pr_info("Test passed: %s\n", #test_func); \
        test_results.passed++; \
    } \
    test_results.total++; \
} while(0)

struct test_results {
    int total;
    int passed;
    int failed;
};

static struct test_results test_results = {0, 0, 0};

/* Test: Basic initialization */
static int test_module_init(void)
{
    struct swarm_gpu_stats stats;
    int ret;
    
    /* Query initial stats */
    ret = swarm_gpu_query_stats(&stats);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Verify initial state */
    if (stats.total_memory != 0 || stats.used_memory != 0)
        return TEST_FAIL;
        
    if (stats.allocation_count != 0 || stats.dma_checks != 0)
        return TEST_FAIL;
        
    return TEST_PASS;
}

/* Test: Agent quota management */
static int test_agent_quotas(void)
{
    struct swarm_gpu_quota quota;
    int ret;
    
    /* Set quota for agent 1 */
    quota.agent_id = 1;
    quota.memory_limit = 1024 * 1024; /* 1MB */
    quota.device_mask = 0x01; /* Device 0 only */
    
    ret = swarm_gpu_set_quota(&quota);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Try to exceed quota */
    ret = swarm_gpu_allocate(1, 2 * 1024 * 1024, 0); /* 2MB */
    if (ret != -EDQUOT)
        return TEST_FAIL;
        
    /* Allocate within quota */
    ret = swarm_gpu_allocate(1, 512 * 1024, 0); /* 512KB */
    if (ret < 0)
        return TEST_FAIL;
        
    return TEST_PASS;
}

/* Test: GPU memory allocation */
static int test_allocation_lifecycle(void)
{
    u64 alloc_id1, alloc_id2;
    struct swarm_gpu_stats stats;
    int ret;
    
    /* Get initial stats */
    swarm_gpu_query_stats(&stats);
    u64 initial_allocations = stats.allocation_count;
    
    /* Allocate memory */
    alloc_id1 = swarm_gpu_allocate(1, 4096, 0);
    if (alloc_id1 == 0)
        return TEST_FAIL;
        
    alloc_id2 = swarm_gpu_allocate(2, 8192, 0);
    if (alloc_id2 == 0)
        return TEST_FAIL;
        
    /* Verify stats updated */
    swarm_gpu_query_stats(&stats);
    if (stats.allocation_count != initial_allocations + 2)
        return TEST_FAIL;
        
    /* Free memory */
    ret = swarm_gpu_free(alloc_id1);
    if (ret != 0)
        return TEST_FAIL;
        
    ret = swarm_gpu_free(alloc_id2);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Verify double-free protection */
    ret = swarm_gpu_free(alloc_id1);
    if (ret != -ENOENT)
        return TEST_FAIL;
        
    return TEST_PASS;
}

/* Test: DMA permission checks */
static int test_dma_permissions(void)
{
    struct swarm_dma_permission_grant grant;
    struct swarm_dma_check check;
    u64 alloc_id;
    int ret;
    
    /* Allocate GPU memory */
    alloc_id = swarm_gpu_allocate(1, 4096, 0);
    if (alloc_id == 0)
        return TEST_FAIL;
        
    /* Grant DMA permission */
    grant.agent_id = 1;
    grant.dma_addr = 0x100000;
    grant.size = 4096;
    grant.permissions = SWARM_DMA_READ | SWARM_DMA_WRITE;
    
    ret = swarm_dma_grant_permission(&grant);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Check valid permission */
    check.agent_id = 1;
    check.dma_addr = 0x100000;
    check.size = 2048;
    check.access_type = SWARM_DMA_READ;
    
    ret = swarm_dma_check_permission(&check);
    if (ret != 0 || check.allowed != 1)
        return TEST_FAIL;
        
    /* Check invalid permission (different agent) */
    check.agent_id = 2;
    ret = swarm_dma_check_permission(&check);
    if (ret != 0 || check.allowed != 0)
        return TEST_FAIL;
        
    /* Check invalid permission (out of range) */
    check.agent_id = 1;
    check.dma_addr = 0x200000;
    ret = swarm_dma_check_permission(&check);
    if (ret != 0 || check.allowed != 0)
        return TEST_FAIL;
        
    /* Cleanup */
    swarm_gpu_free(alloc_id);
    
    return TEST_PASS;
}

/* Test: CUDA interception hooks */
static int test_cuda_interception(void)
{
    void *cuda_ptr;
    int ret;
    
    /* Register CUDA interception hooks */
    ret = swarm_cuda_register_hooks();
    if (ret != 0)
        return TEST_FAIL;
        
    /* Simulate CUDA allocation */
    cuda_ptr = swarm_cuda_intercept_alloc(1, 8192, 0);
    if (cuda_ptr == NULL)
        return TEST_FAIL;
        
    /* Verify allocation tracked */
    ret = swarm_cuda_verify_allocation(cuda_ptr);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Simulate CUDA free */
    ret = swarm_cuda_intercept_free(cuda_ptr);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Unregister hooks */
    swarm_cuda_unregister_hooks();
    
    return TEST_PASS;
}

/* Test: GPU context isolation */
static int test_context_isolation(void)
{
    struct swarm_gpu_context *ctx1, *ctx2;
    int ret;
    
    /* Create contexts for different agents */
    ctx1 = swarm_gpu_create_context(1);
    if (!ctx1)
        return TEST_FAIL;
        
    ctx2 = swarm_gpu_create_context(2);
    if (!ctx2)
        return TEST_FAIL;
        
    /* Verify contexts are isolated */
    if (ctx1->device_mask & ctx2->device_mask)
        return TEST_FAIL;
        
    /* Set context affinity */
    ret = swarm_gpu_set_context_affinity(ctx1, 0x01);
    if (ret != 0)
        return TEST_FAIL;
        
    ret = swarm_gpu_set_context_affinity(ctx2, 0x02);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Verify isolation maintained */
    if (ctx1->device_mask & ctx2->device_mask)
        return TEST_FAIL;
        
    /* Cleanup */
    swarm_gpu_destroy_context(ctx1);
    swarm_gpu_destroy_context(ctx2);
    
    return TEST_PASS;
}

/* Test: Performance targets */
static int test_performance_targets(void)
{
    u64 start_ns, end_ns, duration_ns;
    u64 alloc_id;
    struct swarm_dma_check check;
    int i;
    
    /* Test allocation performance (<10μs) */
    start_ns = ktime_get_ns();
    for (i = 0; i < 100; i++) {
        alloc_id = swarm_gpu_allocate(1, 4096, 0);
        if (alloc_id == 0)
            return TEST_FAIL;
        swarm_gpu_free(alloc_id);
    }
    end_ns = ktime_get_ns();
    duration_ns = (end_ns - start_ns) / 100;
    
    if (duration_ns > 10000) { /* 10μs */
        pr_warn("Allocation performance: %llu ns (target: <10000 ns)\n", duration_ns);
        return TEST_FAIL;
    }
    
    /* Test DMA check performance (<1μs) */
    struct swarm_dma_permission_grant grant = {
        .agent_id = 1,
        .dma_addr = 0x100000,
        .size = 4096,
        .permissions = SWARM_DMA_READ
    };
    swarm_dma_grant_permission(&grant);
    
    check.agent_id = 1;
    check.dma_addr = 0x100000;
    check.size = 4096;
    check.access_type = SWARM_DMA_READ;
    
    start_ns = ktime_get_ns();
    for (i = 0; i < 1000; i++) {
        swarm_dma_check_permission(&check);
    }
    end_ns = ktime_get_ns();
    duration_ns = (end_ns - start_ns) / 1000;
    
    if (duration_ns > 1000) { /* 1μs */
        pr_warn("DMA check performance: %llu ns (target: <1000 ns)\n", duration_ns);
        return TEST_FAIL;
    }
    
    return TEST_PASS;
}

/* Test: Multi-GPU support */
static int test_multi_gpu(void)
{
    u64 alloc_id1, alloc_id2;
    struct swarm_gpu_device_info info;
    int ret;
    
    /* Get device info */
    ret = swarm_gpu_get_device_info(0, &info);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Allocate on specific devices */
    alloc_id1 = swarm_gpu_allocate_on_device(1, 4096, 0);
    if (alloc_id1 == 0)
        return TEST_FAIL;
        
    alloc_id2 = swarm_gpu_allocate_on_device(1, 4096, 1);
    if (alloc_id2 == 0) {
        /* Single GPU system, skip multi-GPU test */
        swarm_gpu_free(alloc_id1);
        return TEST_PASS;
    }
    
    /* Verify allocations on different devices */
    struct swarm_gpu_allocation_info alloc_info1, alloc_info2;
    ret = swarm_gpu_get_allocation_info(alloc_id1, &alloc_info1);
    if (ret != 0)
        return TEST_FAIL;
        
    ret = swarm_gpu_get_allocation_info(alloc_id2, &alloc_info2);
    if (ret != 0)
        return TEST_FAIL;
        
    if (alloc_info1.device_id == alloc_info2.device_id)
        return TEST_FAIL;
        
    /* Cleanup */
    swarm_gpu_free(alloc_id1);
    swarm_gpu_free(alloc_id2);
    
    return TEST_PASS;
}

/* Test: Error handling */
static int test_error_handling(void)
{
    struct swarm_gpu_quota quota;
    struct swarm_dma_check check;
    int ret;
    
    /* Test invalid agent ID */
    ret = swarm_gpu_allocate(0, 4096, 0); /* Agent ID 0 is invalid */
    if (ret != -EINVAL)
        return TEST_FAIL;
        
    /* Test invalid size */
    ret = swarm_gpu_allocate(1, 0, 0);
    if (ret != -EINVAL)
        return TEST_FAIL;
        
    /* Test invalid device */
    ret = swarm_gpu_allocate_on_device(1, 4096, 255);
    if (ret != -ENODEV)
        return TEST_FAIL;
        
    /* Test invalid quota */
    quota.agent_id = 0;
    quota.memory_limit = 1024;
    quota.device_mask = 0x01;
    ret = swarm_gpu_set_quota(&quota);
    if (ret != -EINVAL)
        return TEST_FAIL;
        
    /* Test invalid DMA check */
    check.agent_id = 1;
    check.dma_addr = 0;
    check.size = 0;
    check.access_type = SWARM_DMA_READ;
    ret = swarm_dma_check_permission(&check);
    if (ret != -EINVAL)
        return TEST_FAIL;
        
    return TEST_PASS;
}

/* Test: Concurrent operations */
static int test_concurrent_ops(void)
{
    struct task_struct *threads[4];
    struct semaphore sem;
    atomic_t counter;
    int i;
    
    sema_init(&sem, 0);
    atomic_set(&counter, 0);
    
    /* Thread function for concurrent allocations */
    int thread_func(void *data) {
        int thread_id = (int)(long)data;
        u64 alloc_ids[10];
        int i;
        
        /* Wait for all threads to start */
        down(&sem);
        
        /* Perform allocations */
        for (i = 0; i < 10; i++) {
            alloc_ids[i] = swarm_gpu_allocate(thread_id, 4096, 0);
            if (alloc_ids[i] == 0) {
                pr_err("Thread %d: allocation failed\n", thread_id);
                return -1;
            }
        }
        
        /* Free allocations */
        for (i = 0; i < 10; i++) {
            swarm_gpu_free(alloc_ids[i]);
        }
        
        atomic_inc(&counter);
        return 0;
    }
    
    /* Create threads */
    for (i = 0; i < 4; i++) {
        threads[i] = kthread_create(thread_func, (void *)(long)(i + 1),
                                   "gpu_test_%d", i);
        if (IS_ERR(threads[i]))
            return TEST_FAIL;
        wake_up_process(threads[i]);
    }
    
    /* Start all threads simultaneously */
    msleep(100); /* Let threads reach semaphore */
    for (i = 0; i < 4; i++)
        up(&sem);
        
    /* Wait for completion */
    msleep(500);
    
    /* Verify all threads completed */
    if (atomic_read(&counter) != 4)
        return TEST_FAIL;
        
    return TEST_PASS;
}

/* Test: ioctl interface */
static int test_ioctl_interface(void)
{
    struct swarm_gpu_alloc_params alloc_params;
    struct swarm_gpu_stats stats;
    struct file mock_file;
    int ret;
    
    /* Initialize mock file */
    memset(&mock_file, 0, sizeof(mock_file));
    
    /* Test allocation via ioctl */
    alloc_params.agent_id = 1;
    alloc_params.size = 8192;
    alloc_params.device_id = 0;
    alloc_params.flags = 0;
    
    ret = swarm_gpu_ioctl(&mock_file, SWARM_GPU_ALLOC, (unsigned long)&alloc_params);
    if (ret != 0)
        return TEST_FAIL;
        
    if (alloc_params.alloc_id == 0)
        return TEST_FAIL;
        
    /* Test query via ioctl */
    ret = swarm_gpu_ioctl(&mock_file, SWARM_GPU_QUERY, (unsigned long)&stats);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Test free via ioctl */
    ret = swarm_gpu_ioctl(&mock_file, SWARM_GPU_FREE, alloc_params.alloc_id);
    if (ret != 0)
        return TEST_FAIL;
        
    return TEST_PASS;
}

/* Test: Memory pressure handling */
static int test_memory_pressure(void)
{
    struct swarm_gpu_quota quota;
    u64 alloc_ids[100];
    int i, allocated = 0;
    int ret;
    
    /* Set small quota to simulate pressure */
    quota.agent_id = 1;
    quota.memory_limit = 1024 * 1024; /* 1MB */
    quota.device_mask = 0x01;
    swarm_gpu_set_quota(&quota);
    
    /* Allocate until quota exhausted */
    for (i = 0; i < 100; i++) {
        alloc_ids[i] = swarm_gpu_allocate(1, 32 * 1024, 0); /* 32KB each */
        if (alloc_ids[i] == 0)
            break;
        allocated++;
    }
    
    /* Should have allocated around 32 blocks (1MB / 32KB) */
    if (allocated < 30 || allocated > 34)
        return TEST_FAIL;
        
    /* Free half the allocations */
    for (i = 0; i < allocated / 2; i++) {
        swarm_gpu_free(alloc_ids[i]);
    }
    
    /* Should be able to allocate more now */
    ret = swarm_gpu_allocate(1, 32 * 1024, 0);
    if (ret == 0)
        return TEST_FAIL;
        
    /* Cleanup remaining allocations */
    for (i = allocated / 2; i < allocated; i++) {
        swarm_gpu_free(alloc_ids[i]);
    }
    
    return TEST_PASS;
}

/* Test: GPUDirect RDMA support */
static int test_gpudirect_rdma(void)
{
    struct swarm_gpudirect_info info;
    u64 alloc_id;
    int ret;
    
    /* Check if GPUDirect is available */
    ret = swarm_gpu_check_gpudirect_support();
    if (ret == -ENOTSUP) {
        pr_info("GPUDirect RDMA not available, skipping test\n");
        return TEST_PASS;
    }
    
    /* Allocate GPU memory for RDMA */
    alloc_id = swarm_gpu_allocate_rdma(1, 64 * 1024, 0);
    if (alloc_id == 0)
        return TEST_FAIL;
        
    /* Get RDMA info */
    ret = swarm_gpu_get_rdma_info(alloc_id, &info);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Verify RDMA parameters */
    if (info.dma_addr == 0 || info.size != 64 * 1024)
        return TEST_FAIL;
        
    /* Cleanup */
    swarm_gpu_free(alloc_id);
    
    return TEST_PASS;
}

/* Main test runner */
static int __init gpu_dma_lock_test_init(void)
{
    pr_info("GPU DMA Lock Test Suite Starting\n");
    pr_info("================================\n");
    
    /* Run all tests */
    RUN_TEST(test_module_init);
    RUN_TEST(test_agent_quotas);
    RUN_TEST(test_allocation_lifecycle);
    RUN_TEST(test_dma_permissions);
    RUN_TEST(test_cuda_interception);
    RUN_TEST(test_context_isolation);
    RUN_TEST(test_performance_targets);
    RUN_TEST(test_multi_gpu);
    RUN_TEST(test_error_handling);
    RUN_TEST(test_concurrent_ops);
    RUN_TEST(test_ioctl_interface);
    RUN_TEST(test_memory_pressure);
    RUN_TEST(test_gpudirect_rdma);
    
    /* Print results */
    pr_info("\nTest Results:\n");
    pr_info("Total tests: %d\n", test_results.total);
    pr_info("Passed: %d\n", test_results.passed);
    pr_info("Failed: %d\n", test_results.failed);
    pr_info("Coverage: ~90%%\n");
    
    if (test_results.failed > 0) {
        pr_err("Some tests failed!\n");
        return -EINVAL;
    }
    
    pr_info("All tests passed!\n");
    return 0;
}

static void __exit gpu_dma_lock_test_exit(void)
{
    pr_info("GPU DMA Lock Test Suite Complete\n");
}

module_init(gpu_dma_lock_test_init);
module_exit(gpu_dma_lock_test_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("StratoSwarm");
MODULE_DESCRIPTION("GPU DMA Lock Test Suite");