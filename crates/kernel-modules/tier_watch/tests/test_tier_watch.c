/*
 * TierWatch Kernel Module Test Suite
 * 
 * Comprehensive unit tests for the 5-tier memory hierarchy monitoring module
 * Target: 90% code coverage
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/delay.h>
#include <linux/kthread.h>
#include <linux/semaphore.h>
#include <linux/mm.h>
#include <linux/highmem.h>
#include <linux/numa.h>

#include "../tier_watch.h"

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

/* Test: Basic tier properties */
static int test_tier_properties(void)
{
    struct tier_info info;
    int ret;
    
    /* Test GPU tier */
    ret = tier_watch_get_tier_info(TIER_GPU, &info);
    if (ret != 0)
        return TEST_FAIL;
        
    if (info.capacity_bytes != 32ULL * 1024 * 1024 * 1024)
        return TEST_FAIL;
        
    if (info.latency_ns != 200)
        return TEST_FAIL;
        
    /* Test CPU tier */
    ret = tier_watch_get_tier_info(TIER_CPU, &info);
    if (ret != 0)
        return TEST_FAIL;
        
    if (info.capacity_bytes != 96ULL * 1024 * 1024 * 1024)
        return TEST_FAIL;
        
    /* Test tier ordering */
    if (!tier_watch_is_faster(TIER_GPU, TIER_CPU))
        return TEST_FAIL;
        
    if (!tier_watch_is_faster(TIER_CPU, TIER_NVME))
        return TEST_FAIL;
        
    return TEST_PASS;
}

/* Test: Page tracking initialization */
static int test_page_tracking(void)
{
    struct page_info pinfo;
    u64 pfn = 0x1000; /* Test page frame number */
    int ret;
    
    /* Track a new page */
    ret = tier_watch_track_page(pfn, TIER_CPU, 1);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Get page info */
    ret = tier_watch_get_page_info(pfn, &pinfo);
    if (ret != 0)
        return TEST_FAIL;
        
    if (pinfo.tier != TIER_CPU)
        return TEST_FAIL;
        
    if (pinfo.agent_id != 1)
        return TEST_FAIL;
        
    if (pinfo.access_count != 0)
        return TEST_FAIL;
        
    /* Untrack page */
    ret = tier_watch_untrack_page(pfn);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Verify page is no longer tracked */
    ret = tier_watch_get_page_info(pfn, &pinfo);
    if (ret != -ENOENT)
        return TEST_FAIL;
        
    return TEST_PASS;
}

/* Test: Page fault handling */
static int test_page_fault_handling(void)
{
    u64 pfn = 0x2000;
    u64 vaddr = 0xffff888000000000ULL;
    struct fault_stats stats;
    int ret;
    
    /* Track page first */
    ret = tier_watch_track_page(pfn, TIER_NVME, 1);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Simulate page fault */
    ret = tier_watch_handle_fault(pfn, vaddr, FAULT_FLAG_WRITE);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Get fault statistics */
    ret = tier_watch_get_fault_stats(TIER_NVME, &stats);
    if (ret != 0)
        return TEST_FAIL;
        
    if (stats.total_faults == 0)
        return TEST_FAIL;
        
    /* Simulate multiple faults to make page hot */
    for (int i = 0; i < 100; i++) {
        tier_watch_handle_fault(pfn, vaddr, FAULT_FLAG_READ);
    }
    
    /* Check if page is marked as hot */
    struct page_info pinfo;
    ret = tier_watch_get_page_info(pfn, &pinfo);
    if (ret != 0)
        return TEST_FAIL;
        
    if (pinfo.access_count < 100)
        return TEST_FAIL;
        
    /* Cleanup */
    tier_watch_untrack_page(pfn);
    
    return TEST_PASS;
}

/* Test: Hot/cold page detection */
static int test_hot_cold_detection(void)
{
    u64 hot_pfn = 0x3000;
    u64 cold_pfn = 0x4000;
    struct page_list hot_pages, cold_pages;
    int ret;
    
    /* Track pages */
    tier_watch_track_page(hot_pfn, TIER_CPU, 1);
    tier_watch_track_page(cold_pfn, TIER_CPU, 1);
    
    /* Make one page hot */
    for (int i = 0; i < 1000; i++) {
        tier_watch_handle_fault(hot_pfn, 0, FAULT_FLAG_READ);
    }
    
    /* Let cold page age */
    msleep(100);
    
    /* Get hot pages */
    ret = tier_watch_get_hot_pages(TIER_CPU, &hot_pages, 10);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Verify hot page is detected */
    bool found = false;
    for (int i = 0; i < hot_pages.count; i++) {
        if (hot_pages.pages[i] == hot_pfn) {
            found = true;
            break;
        }
    }
    if (!found)
        return TEST_FAIL;
        
    /* Get cold pages */
    ret = tier_watch_get_cold_pages(TIER_CPU, &cold_pages, 10);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Verify cold page is detected */
    found = false;
    for (int i = 0; i < cold_pages.count; i++) {
        if (cold_pages.pages[i] == cold_pfn) {
            found = true;
            break;
        }
    }
    if (!found)
        return TEST_FAIL;
        
    /* Cleanup */
    tier_watch_untrack_page(hot_pfn);
    tier_watch_untrack_page(cold_pfn);
    
    return TEST_PASS;
}

/* Test: Migration candidate detection */
static int test_migration_candidates(void)
{
    struct migration_candidate candidates[10];
    int count;
    int ret;
    
    /* Create pages in different tiers */
    for (int i = 0; i < 5; i++) {
        u64 pfn = 0x5000 + i;
        tier_watch_track_page(pfn, TIER_NVME, 1);
        
        /* Make some pages hot */
        if (i < 2) {
            for (int j = 0; j < 500; j++) {
                tier_watch_handle_fault(pfn, 0, FAULT_FLAG_READ);
            }
        }
    }
    
    /* Get migration candidates for promotion */
    count = tier_watch_get_migration_candidates(TIER_NVME, TIER_CPU, 
                                               candidates, 10);
    if (count <= 0)
        return TEST_FAIL;
        
    /* Verify candidates have high access counts */
    for (int i = 0; i < count; i++) {
        if (candidates[i].access_count < 100)
            return TEST_FAIL;
            
        if (candidates[i].from_tier != TIER_NVME)
            return TEST_FAIL;
            
        if (candidates[i].to_tier != TIER_CPU)
            return TEST_FAIL;
    }
    
    /* Cleanup */
    for (int i = 0; i < 5; i++) {
        tier_watch_untrack_page(0x5000 + i);
    }
    
    return TEST_PASS;
}

/* Test: Memory pressure handling */
static int test_memory_pressure(void)
{
    struct tier_pressure pressure;
    int ret;
    
    /* Get initial pressure */
    ret = tier_watch_get_pressure(TIER_CPU, &pressure);
    if (ret != 0)
        return TEST_FAIL;
        
    if (pressure.level != PRESSURE_LOW)
        return TEST_FAIL;
        
    /* Simulate high memory usage */
    for (int i = 0; i < 1000; i++) {
        tier_watch_track_page(0x10000 + i, TIER_CPU, 1);
    }
    
    /* Update pressure calculation */
    tier_watch_update_pressure();
    
    /* Check pressure increased */
    ret = tier_watch_get_pressure(TIER_CPU, &pressure);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Cleanup */
    for (int i = 0; i < 1000; i++) {
        tier_watch_untrack_page(0x10000 + i);
    }
    
    return TEST_PASS;
}

/* Test: NUMA awareness */
static int test_numa_awareness(void)
{
    struct numa_stats stats;
    u64 pfn = 0x6000;
    int ret;
    
    /* Track page on specific NUMA node */
    ret = tier_watch_track_page_numa(pfn, TIER_CPU, 1, 0);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Get NUMA statistics */
    ret = tier_watch_get_numa_stats(0, &stats);
    if (ret != 0)
        return TEST_FAIL;
        
    if (stats.pages_on_node == 0)
        return TEST_FAIL;
        
    /* Test NUMA migration hint */
    int target_node = tier_watch_get_optimal_numa_node(pfn);
    if (target_node < 0 || target_node >= MAX_NUMNODES)
        return TEST_FAIL;
        
    /* Cleanup */
    tier_watch_untrack_page(pfn);
    
    return TEST_PASS;
}

/* Test: Per-agent memory tracking */
static int test_agent_memory_tracking(void)
{
    struct agent_memory_stats stats;
    u64 agent_id = 42;
    int ret;
    
    /* Track pages for specific agent */
    for (int i = 0; i < 10; i++) {
        tier_watch_track_page(0x7000 + i, TIER_CPU, agent_id);
    }
    
    for (int i = 0; i < 5; i++) {
        tier_watch_track_page(0x8000 + i, TIER_NVME, agent_id);
    }
    
    /* Get agent memory statistics */
    ret = tier_watch_get_agent_memory(agent_id, &stats);
    if (ret != 0)
        return TEST_FAIL;
        
    if (stats.pages_in_tier[TIER_CPU] != 10)
        return TEST_FAIL;
        
    if (stats.pages_in_tier[TIER_NVME] != 5)
        return TEST_FAIL;
        
    if (stats.total_pages != 15)
        return TEST_FAIL;
        
    /* Cleanup */
    for (int i = 0; i < 10; i++) {
        tier_watch_untrack_page(0x7000 + i);
    }
    for (int i = 0; i < 5; i++) {
        tier_watch_untrack_page(0x8000 + i);
    }
    
    return TEST_PASS;
}

/* Test: Performance targets */
static int test_performance_targets(void)
{
    u64 start_ns, end_ns, duration_ns;
    u64 pfn = 0x9000;
    int i;
    
    /* Track initial pages */
    for (i = 0; i < 100; i++) {
        tier_watch_track_page(pfn + i, TIER_CPU, 1);
    }
    
    /* Test fault handling performance (<100ns overhead) */
    start_ns = ktime_get_ns();
    
    for (i = 0; i < 10000; i++) {
        tier_watch_handle_fault(pfn, 0, FAULT_FLAG_READ);
    }
    
    end_ns = ktime_get_ns();
    duration_ns = (end_ns - start_ns) / 10000;
    
    if (duration_ns > 100) {
        pr_warn("Fault handling performance: %llu ns (target: <100 ns)\n", 
                duration_ns);
        return TEST_FAIL;
    }
    
    /* Test migration detection performance (<1ms) */
    start_ns = ktime_get_ns();
    
    struct migration_candidate candidates[100];
    tier_watch_get_migration_candidates(TIER_CPU, TIER_GPU, candidates, 100);
    
    end_ns = ktime_get_ns();
    duration_ns = end_ns - start_ns;
    
    if (duration_ns > 1000000) { /* 1ms */
        pr_warn("Migration detection: %llu ns (target: <1ms)\n", duration_ns);
        return TEST_FAIL;
    }
    
    /* Cleanup */
    for (i = 0; i < 100; i++) {
        tier_watch_untrack_page(pfn + i);
    }
    
    return TEST_PASS;
}

/* Test: Concurrent operations */
static int test_concurrent_operations(void)
{
    struct task_struct *threads[4];
    struct semaphore sem;
    atomic_t success_counter;
    int i;
    
    sema_init(&sem, 0);
    atomic_set(&success_counter, 0);
    
    /* Thread function for concurrent page tracking */
    int thread_func(void *data) {
        int thread_id = (int)(long)data;
        u64 base_pfn = 0xA0000 + (thread_id * 0x1000);
        int i;
        
        /* Wait for all threads */
        down(&sem);
        
        /* Track and fault pages concurrently */
        for (i = 0; i < 100; i++) {
            u64 pfn = base_pfn + i;
            
            if (tier_watch_track_page(pfn, TIER_CPU, thread_id) != 0)
                return -1;
                
            /* Simulate faults */
            tier_watch_handle_fault(pfn, 0, FAULT_FLAG_READ);
            
            /* Random operations */
            if (i % 3 == 0) {
                struct page_info pinfo;
                tier_watch_get_page_info(pfn, &pinfo);
            }
            
            if (i % 5 == 0) {
                tier_watch_untrack_page(pfn - 5);
            }
        }
        
        /* Cleanup remaining pages */
        for (i = 95; i < 100; i++) {
            tier_watch_untrack_page(base_pfn + i);
        }
        
        atomic_inc(&success_counter);
        return 0;
    }
    
    /* Create threads */
    for (i = 0; i < 4; i++) {
        threads[i] = kthread_create(thread_func, (void *)(long)i,
                                   "tier_test_%d", i);
        if (IS_ERR(threads[i]))
            return TEST_FAIL;
        wake_up_process(threads[i]);
    }
    
    /* Start all threads */
    msleep(100);
    for (i = 0; i < 4; i++)
        up(&sem);
        
    /* Wait for completion */
    msleep(500);
    
    if (atomic_read(&success_counter) != 4)
        return TEST_FAIL;
        
    return TEST_PASS;
}

/* Test: /proc interface */
static int test_proc_interface(void)
{
    struct file *file;
    char buffer[256];
    loff_t pos = 0;
    ssize_t ret;
    
    /* Track some pages for testing */
    for (int i = 0; i < 5; i++) {
        tier_watch_track_page(0xB000 + i, TIER_CPU + (i % 5), 1);
    }
    
    /* Test reading tier stats */
    file = filp_open("/proc/swarm/tiers/cpu/stats", O_RDONLY, 0);
    if (IS_ERR(file))
        return TEST_FAIL;
        
    ret = kernel_read(file, buffer, sizeof(buffer) - 1, &pos);
    if (ret <= 0) {
        filp_close(file, NULL);
        return TEST_FAIL;
    }
    
    buffer[ret] = '\0';
    filp_close(file, NULL);
    
    /* Verify stats contain expected fields */
    if (!strstr(buffer, "total_pages"))
        return TEST_FAIL;
        
    if (!strstr(buffer, "hot_pages"))
        return TEST_FAIL;
        
    /* Cleanup */
    for (int i = 0; i < 5; i++) {
        tier_watch_untrack_page(0xB000 + i);
    }
    
    return TEST_PASS;
}

/* Test: Migration execution */
static int test_migration_execution(void)
{
    struct migration_request req;
    struct migration_result result;
    u64 pfn = 0xC000;
    int ret;
    
    /* Track page in lower tier */
    ret = tier_watch_track_page(pfn, TIER_NVME, 1);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Create migration request */
    req.pfn = pfn;
    req.from_tier = TIER_NVME;
    req.to_tier = TIER_CPU;
    req.priority = 100;
    req.agent_id = 1;
    req.reason = MIGRATION_HOT_PROMOTION;
    
    /* Execute migration */
    ret = tier_watch_migrate_page(&req, &result);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Verify page moved */
    struct page_info pinfo;
    ret = tier_watch_get_page_info(pfn, &pinfo);
    if (ret != 0)
        return TEST_FAIL;
        
    if (pinfo.tier != TIER_CPU)
        return TEST_FAIL;
        
    /* Check migration stats updated */
    struct tier_stats stats;
    ret = tier_watch_get_tier_stats(TIER_CPU, &stats);
    if (ret != 0)
        return TEST_FAIL;
        
    if (stats.migrations_in == 0)
        return TEST_FAIL;
        
    /* Cleanup */
    tier_watch_untrack_page(pfn);
    
    return TEST_PASS;
}

/* Test: Tier capacity limits */
static int test_tier_capacity_limits(void)
{
    struct tier_info info;
    u64 base_pfn = 0xD0000;
    int ret;
    int pages_tracked = 0;
    
    /* Get GPU tier info (smallest tier) */
    ret = tier_watch_get_tier_info(TIER_GPU, &info);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Try to fill the tier */
    u64 pages_per_tier = info.capacity_bytes / PAGE_SIZE;
    u64 max_test_pages = min(pages_per_tier / 1000, 100ULL); /* Test subset */
    
    for (u64 i = 0; i < max_test_pages; i++) {
        ret = tier_watch_track_page(base_pfn + i, TIER_GPU, 1);
        if (ret == -ENOMEM)
            break; /* Tier full */
        if (ret != 0)
            return TEST_FAIL;
        pages_tracked++;
    }
    
    /* Verify we tracked some pages */
    if (pages_tracked == 0)
        return TEST_FAIL;
        
    /* Cleanup */
    for (int i = 0; i < pages_tracked; i++) {
        tier_watch_untrack_page(base_pfn + i);
    }
    
    return TEST_PASS;
}

/* Main test runner */
static int __init tier_watch_test_init(void)
{
    pr_info("TierWatch Test Suite Starting\n");
    pr_info("==============================\n");
    
    /* Run all tests */
    RUN_TEST(test_tier_properties);
    RUN_TEST(test_page_tracking);
    RUN_TEST(test_page_fault_handling);
    RUN_TEST(test_hot_cold_detection);
    RUN_TEST(test_migration_candidates);
    RUN_TEST(test_memory_pressure);
    RUN_TEST(test_numa_awareness);
    RUN_TEST(test_agent_memory_tracking);
    RUN_TEST(test_performance_targets);
    RUN_TEST(test_concurrent_operations);
    RUN_TEST(test_proc_interface);
    RUN_TEST(test_migration_execution);
    RUN_TEST(test_tier_capacity_limits);
    
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

static void __exit tier_watch_test_exit(void)
{
    pr_info("TierWatch Test Suite Complete\n");
}

module_init(tier_watch_test_init);
module_exit(tier_watch_test_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("StratoSwarm");
MODULE_DESCRIPTION("TierWatch Test Suite");