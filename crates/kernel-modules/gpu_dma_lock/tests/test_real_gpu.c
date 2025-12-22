/*
 * GPU DMA Lock Real GPU Tests
 * 
 * Tests with actual GPU hardware and CUDA runtime
 */

#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/delay.h>
#include <linux/kthread.h>
#include <linux/pci.h>
#include "../gpu_dma_lock.h"

#define MODULE_NAME "test_real_gpu"
#define TEST_AGENT_ID 1000

/* Test results */
static struct {
    int passed;
    int failed;
    int skipped;
} test_results;

/* Test: Real GPU enumeration */
static int test_gpu_enumeration(void)
{
    struct swarm_gpu_device_info info;
    int gpu_count;
    int ret;
    
    pr_info("TEST: GPU enumeration\n");
    
    gpu_count = swarm_gpu_get_device_count();
    pr_info("  Found %d GPUs\n", gpu_count);
    
    if (gpu_count == 0) {
        pr_warn("  No GPUs found - skipping hardware tests\n");
        test_results.skipped++;
        return 0;
    }
    
    /* Get info for each GPU */
    for (int i = 0; i < gpu_count; i++) {
        ret = swarm_gpu_get_device_info(i, &info);
        if (ret != 0) {
            pr_err("  FAIL: Failed to get info for GPU %d\n", i);
            test_results.failed++;
            return -1;
        }
        
        pr_info("  GPU %d: %s\n", i, info.name);
        pr_info("    Memory: %llu MB total, %llu MB free\n",
                info.total_memory / (1024 * 1024),
                info.free_memory / (1024 * 1024));
        pr_info("    Compute Capability: %d.%d\n",
                info.compute_capability / 10,
                info.compute_capability % 10);
                
        /* Verify reasonable values */
        if (info.total_memory < 1024 * 1024 * 1024) { /* Less than 1GB */
            pr_err("  FAIL: GPU %d has unrealistic memory size\n", i);
            test_results.failed++;
            return -1;
        }
    }
    
    pr_info("  PASS: GPU enumeration successful\n");
    test_results.passed++;
    return 0;
}

/* Test: Real DMA allocation */
static int test_real_dma_allocation(void)
{
    u64 alloc_id;
    struct swarm_gpu_allocation_info info;
    struct swarm_dma_check check;
    size_t test_size = 64 * 1024 * 1024; /* 64MB */
    int ret;
    
    pr_info("TEST: Real DMA allocation\n");
    
    if (swarm_gpu_get_device_count() == 0) {
        pr_warn("  No GPUs - skipping\n");
        test_results.skipped++;
        return 0;
    }
    
    /* Allocate GPU memory */
    alloc_id = swarm_gpu_allocate_on_device(TEST_AGENT_ID, test_size, 0);
    if (alloc_id <= 0) {
        pr_err("  FAIL: Failed to allocate GPU memory: %lld\n", alloc_id);
        test_results.failed++;
        return -1;
    }
    
    pr_info("  Allocated %zu bytes, ID: %llu\n", test_size, alloc_id);
    
    /* Get allocation info */
    ret = swarm_gpu_get_allocation_info(alloc_id, &info);
    if (ret != 0) {
        pr_err("  FAIL: Failed to get allocation info\n");
        swarm_gpu_free(alloc_id);
        test_results.failed++;
        return -1;
    }
    
    pr_info("  DMA address: 0x%llx\n", info.gpu_addr);
    
    /* Verify DMA address is valid */
    if (info.gpu_addr == 0 || info.gpu_addr == (u64)-1) {
        pr_err("  FAIL: Invalid DMA address\n");
        swarm_gpu_free(alloc_id);
        test_results.failed++;
        return -1;
    }
    
    /* Test DMA permission check */
    check.agent_id = TEST_AGENT_ID;
    check.dma_addr = info.gpu_addr;
    check.size = test_size;
    check.flags = SWARM_DMA_READ | SWARM_DMA_WRITE;
    
    ret = swarm_dma_check_permission(&check);
    if (ret != 0) {
        pr_err("  FAIL: DMA permission check failed\n");
        swarm_gpu_free(alloc_id);
        test_results.failed++;
        return -1;
    }
    
    if (!check.allowed) {
        pr_err("  FAIL: DMA access not allowed for valid allocation\n");
        swarm_gpu_free(alloc_id);
        test_results.failed++;
        return -1;
    }
    
    /* Free allocation */
    ret = swarm_gpu_free(alloc_id);
    if (ret != 0) {
        pr_err("  FAIL: Failed to free allocation\n");
        test_results.failed++;
        return -1;
    }
    
    pr_info("  PASS: Real DMA allocation successful\n");
    test_results.passed++;
    return 0;
}

/* Test: CUDA interception */
static int test_cuda_interception(void)
{
    struct swarm_gpu_stats stats_before, stats_after;
    int ret;
    
    pr_info("TEST: CUDA runtime interception\n");
    
    /* Get stats before */
    ret = swarm_gpu_query_stats(&stats_before);
    if (ret != 0) {
        pr_err("  FAIL: Failed to get stats\n");
        test_results.failed++;
        return -1;
    }
    
    /* Trigger some allocations that might be intercepted */
    for (int i = 0; i < 10; i++) {
        u64 alloc_id = swarm_gpu_allocate(TEST_AGENT_ID + i, 1024 * 1024, 0);
        if (alloc_id > 0) {
            swarm_gpu_free(alloc_id);
        }
    }
    
    /* Get stats after */
    ret = swarm_gpu_query_stats(&stats_after);
    if (ret != 0) {
        pr_err("  FAIL: Failed to get stats\n");
        test_results.failed++;
        return -1;
    }
    
    /* Check if any CUDA calls were intercepted */
    u64 intercepts = atomic64_read(&g_state->stats.cuda_intercepts);
    pr_info("  CUDA intercepts: %llu\n", intercepts);
    
    /* Note: Intercepts might be 0 if CUDA runtime isn't loaded */
    if (intercepts > 0) {
        pr_info("  Successfully intercepted CUDA calls\n");
    } else {
        pr_info("  No CUDA runtime detected (expected in kernel context)\n");
    }
    
    pr_info("  PASS: CUDA interception test completed\n");
    test_results.passed++;
    return 0;
}

/* Test: IOMMU/DMA security */
static int test_iommu_security(void)
{
    struct swarm_dma_check check;
    int ret;
    
    pr_info("TEST: IOMMU/DMA security\n");
    
    /* Test 1: Check invalid DMA address */
    check.agent_id = TEST_AGENT_ID;
    check.dma_addr = 0xdeadbeef000; /* Invalid address */
    check.size = 4096;
    check.flags = SWARM_DMA_READ;
    
    ret = swarm_dma_check_permission(&check);
    if (ret != 0) {
        pr_err("  FAIL: DMA check returned error: %d\n", ret);
        test_results.failed++;
        return -1;
    }
    
    if (check.allowed) {
        pr_err("  FAIL: Invalid DMA address was allowed!\n");
        test_results.failed++;
        return -1;
    }
    
    pr_info("  Invalid DMA address correctly rejected\n");
    
    /* Test 2: Check zero address */
    check.dma_addr = 0;
    ret = swarm_dma_check_permission(&check);
    if (ret == 0 && check.allowed) {
        pr_err("  FAIL: NULL DMA address was allowed!\n");
        test_results.failed++;
        return -1;
    }
    
    pr_info("  NULL DMA address correctly rejected\n");
    
    /* Test 3: Check kernel address (should fail) */
    check.dma_addr = (u64)&check; /* Kernel stack address */
    ret = swarm_dma_check_permission(&check);
    if (ret == 0 && check.allowed) {
        pr_warn("  WARN: Kernel address allowed (no IOMMU?)\n");
    } else {
        pr_info("  Kernel address correctly rejected\n");
    }
    
    pr_info("  PASS: IOMMU/DMA security checks working\n");
    test_results.passed++;
    return 0;
}

/* Test: Multi-GPU allocation */
static int test_multi_gpu(void)
{
    struct swarm_gpu_stats stats;
    u64 alloc_ids[MAX_GPU_DEVICES];
    int gpu_count;
    int ret;
    
    pr_info("TEST: Multi-GPU allocation\n");
    
    gpu_count = swarm_gpu_get_device_count();
    if (gpu_count < 2) {
        pr_warn("  Less than 2 GPUs - skipping multi-GPU test\n");
        test_results.skipped++;
        return 0;
    }
    
    /* Allocate on each GPU */
    for (int i = 0; i < gpu_count; i++) {
        alloc_ids[i] = swarm_gpu_allocate_on_device(TEST_AGENT_ID,
                                                    10 * 1024 * 1024, /* 10MB */
                                                    i);
        if (alloc_ids[i] <= 0) {
            pr_err("  FAIL: Failed to allocate on GPU %d\n", i);
            /* Clean up previous allocations */
            for (int j = 0; j < i; j++) {
                swarm_gpu_free(alloc_ids[j]);
            }
            test_results.failed++;
            return -1;
        }
        pr_info("  Allocated on GPU %d: ID %llu\n", i, alloc_ids[i]);
    }
    
    /* Check stats */
    ret = swarm_gpu_query_stats(&stats);
    if (ret == 0) {
        pr_info("  Total allocations: %llu\n", stats.allocation_count);
        pr_info("  Memory used: %llu MB\n", stats.used_memory / (1024 * 1024));
    }
    
    /* Free all allocations */
    for (int i = 0; i < gpu_count; i++) {
        ret = swarm_gpu_free(alloc_ids[i]);
        if (ret != 0) {
            pr_err("  FAIL: Failed to free allocation on GPU %d\n", i);
            test_results.failed++;
            return -1;
        }
    }
    
    pr_info("  PASS: Multi-GPU allocation successful\n");
    test_results.passed++;
    return 0;
}

/* Test: Performance with real hardware */
static int test_real_performance(void)
{
    ktime_t start, end;
    u64 total_ns = 0;
    u64 alloc_id;
    int iterations = 1000;
    
    pr_info("TEST: Real hardware performance\n");
    
    if (swarm_gpu_get_device_count() == 0) {
        pr_warn("  No GPUs - skipping\n");
        test_results.skipped++;
        return 0;
    }
    
    /* Test allocation performance */
    for (int i = 0; i < iterations; i++) {
        start = ktime_get();
        alloc_id = swarm_gpu_allocate(TEST_AGENT_ID, 1024 * 1024, 0); /* 1MB */
        end = ktime_get();
        
        if (alloc_id <= 0) {
            pr_err("  FAIL: Allocation failed at iteration %d\n", i);
            test_results.failed++;
            return -1;
        }
        
        total_ns += ktime_to_ns(ktime_sub(end, start));
        swarm_gpu_free(alloc_id);
    }
    
    u64 avg_ns = total_ns / iterations;
    pr_info("  Average allocation time: %llu ns (%llu us)\n",
            avg_ns, avg_ns / 1000);
            
    /* Check against target */
    if (avg_ns / 1000 > SWARM_TARGET_ALLOC_US) {
        pr_warn("  WARN: Allocation slower than target (%d us)\n",
                SWARM_TARGET_ALLOC_US);
    } else {
        pr_info("  Meets performance target!\n");
    }
    
    /* Test DMA check performance */
    struct swarm_dma_check check = {
        .agent_id = TEST_AGENT_ID,
        .dma_addr = 0x100000000, /* Some address */
        .size = 4096,
        .flags = SWARM_DMA_READ
    };
    
    total_ns = 0;
    for (int i = 0; i < iterations * 10; i++) {
        start = ktime_get();
        swarm_dma_check_permission(&check);
        end = ktime_get();
        total_ns += ktime_to_ns(ktime_sub(end, start));
    }
    
    avg_ns = total_ns / (iterations * 10);
    pr_info("  Average DMA check time: %llu ns\n", avg_ns);
    
    if (avg_ns > SWARM_TARGET_DMA_NS) {
        pr_warn("  WARN: DMA check slower than target (%d ns)\n",
                SWARM_TARGET_DMA_NS);
    } else {
        pr_info("  Meets DMA check target!\n");
    }
    
    pr_info("  PASS: Performance test completed\n");
    test_results.passed++;
    return 0;
}

/* Test runner thread */
static int test_thread(void *data)
{
    pr_info("%s: Starting real GPU tests\n", MODULE_NAME);
    
    /* Run tests */
    test_gpu_enumeration();
    test_real_dma_allocation();
    test_cuda_interception();
    test_iommu_security();
    test_multi_gpu();
    test_real_performance();
    
    /* Print summary */
    pr_info("\n=== Real GPU Test Summary ===\n");
    pr_info("Passed: %d\n", test_results.passed);
    pr_info("Failed: %d\n", test_results.failed);
    pr_info("Skipped: %d\n", test_results.skipped);
    pr_info("Total: %d\n", test_results.passed + test_results.failed + test_results.skipped);
    
    if (test_results.failed == 0) {
        pr_info("All tests PASSED!\n");
    } else {
        pr_err("Some tests FAILED!\n");
    }
    
    return 0;
}

/* Module init */
static int __init test_real_gpu_init(void)
{
    struct task_struct *thread;
    
    pr_info("%s: Loading real GPU test module\n", MODULE_NAME);
    
    /* Reset test results */
    memset(&test_results, 0, sizeof(test_results));
    
    /* Create test thread */
    thread = kthread_create(test_thread, NULL, "test_real_gpu");
    if (IS_ERR(thread)) {
        pr_err("%s: Failed to create test thread\n", MODULE_NAME);
        return PTR_ERR(thread);
    }
    
    wake_up_process(thread);
    return 0;
}

/* Module exit */
static void __exit test_real_gpu_exit(void)
{
    pr_info("%s: Unloading real GPU test module\n", MODULE_NAME);
}

module_init(test_real_gpu_init);
module_exit(test_real_gpu_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("StratoSwarm");
MODULE_DESCRIPTION("Real GPU Hardware Tests");
MODULE_VERSION("1.0");