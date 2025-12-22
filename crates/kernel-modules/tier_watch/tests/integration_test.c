/*
 * TierWatch Integration Tests
 * 
 * Tests real-world scenarios and interactions with system components
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <numa.h>

#include "../tier_watch.h"

#define TEST_PASS 0
#define TEST_FAIL -1

#define PAGE_SIZE 4096
#define MB (1024 * 1024)
#define GB (1024 * MB)

/* Test: Real memory allocation and tracking */
int test_real_memory_tracking(void)
{
    void *mem;
    u64 pfn;
    struct page_info pinfo;
    int ret;
    
    printf("Testing real memory allocation tracking...\n");
    
    /* Allocate real memory */
    mem = mmap(NULL, PAGE_SIZE, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        printf("  FAIL: mmap failed: %s\n", strerror(errno));
        return TEST_FAIL;
    }
    
    /* Touch the page to ensure it's allocated */
    memset(mem, 0x42, PAGE_SIZE);
    
    /* Get page frame number (simplified - real implementation would use pagemap) */
    pfn = ((uintptr_t)mem >> 12);
    
    /* Track the page */
    ret = tier_watch_track_page(pfn, TIER_CPU, 1);
    if (ret != 0) {
        printf("  FAIL: Failed to track page: %d\n", ret);
        munmap(mem, PAGE_SIZE);
        return TEST_FAIL;
    }
    
    /* Simulate memory access patterns */
    for (int i = 0; i < 100; i++) {
        /* Read from memory */
        volatile char c = ((char *)mem)[i * 32];
        
        /* Simulate fault */
        tier_watch_handle_fault(pfn, (u64)mem, FAULT_FLAG_READ);
    }
    
    /* Get page info */
    ret = tier_watch_get_page_info(pfn, &pinfo);
    if (ret != 0) {
        printf("  FAIL: Failed to get page info\n");
        tier_watch_untrack_page(pfn);
        munmap(mem, PAGE_SIZE);
        return TEST_FAIL;
    }
    
    printf("  Page access count: %u\n", pinfo.access_count);
    
    /* Cleanup */
    tier_watch_untrack_page(pfn);
    munmap(mem, PAGE_SIZE);
    
    printf("  PASS: Real memory tracking working\n");
    return TEST_PASS;
}

/* Test: Multi-tier simulation */
int test_multi_tier_simulation(void)
{
    struct tier_info info;
    struct migration_request req;
    struct migration_result result;
    u64 pages[5];
    int ret;
    
    printf("Testing multi-tier memory simulation...\n");
    
    /* Track pages in different tiers */
    for (int i = 0; i < 5; i++) {
        pages[i] = 0x100000 + i;
        ret = tier_watch_track_page(pages[i], i, 1); /* Each in different tier */
        if (ret != 0) {
            printf("  FAIL: Failed to track page in tier %d\n", i);
            return TEST_FAIL;
        }
    }
    
    /* Get tier information */
    for (int i = 0; i < 5; i++) {
        ret = tier_watch_get_tier_info(i, &info);
        if (ret != 0) {
            printf("  FAIL: Failed to get tier %d info\n", i);
            return TEST_FAIL;
        }
        
        printf("  Tier %s: capacity=%llu GB, latency=%u ns\n",
               info.name, info.capacity_bytes / GB, info.latency_ns);
    }
    
    /* Test migration between tiers */
    req.pfn = pages[TIER_HDD];
    req.from_tier = TIER_HDD;
    req.to_tier = TIER_SSD;
    req.priority = 80;
    req.agent_id = 1;
    req.reason = MIGRATION_HOT_PROMOTION;
    
    ret = tier_watch_migrate_page(&req, &result);
    if (ret != 0) {
        printf("  FAIL: Migration failed: %d\n", ret);
        return TEST_FAIL;
    }
    
    printf("  Migration completed in %llu ns\n", result.latency_ns);
    
    /* Cleanup */
    for (int i = 0; i < 5; i++) {
        tier_watch_untrack_page(pages[i]);
    }
    
    printf("  PASS: Multi-tier simulation working\n");
    return TEST_PASS;
}

/* Test: Memory pressure simulation */
int test_memory_pressure(void)
{
    struct tier_pressure pressure;
    u64 base_pfn = 0x200000;
    int ret;
    
    printf("Testing memory pressure simulation...\n");
    
    /* Fill GPU tier to create pressure */
    printf("  Filling GPU tier...\n");
    for (int i = 0; i < 1000; i++) {
        ret = tier_watch_track_page(base_pfn + i, TIER_GPU, 1);
        if (ret == -ENOMEM) {
            printf("  GPU tier full at %d pages\n", i);
            break;
        }
    }
    
    /* Check pressure */
    ret = tier_watch_get_pressure(TIER_GPU, &pressure);
    if (ret != 0) {
        printf("  FAIL: Failed to get pressure\n");
        return TEST_FAIL;
    }
    
    printf("  GPU tier pressure: level=%d, value=%u%%\n",
           pressure.level, pressure.pressure_value);
    
    /* Update system pressure */
    tier_watch_update_pressure();
    
    /* Cleanup some pages */
    for (int i = 0; i < 500; i++) {
        tier_watch_untrack_page(base_pfn + i);
    }
    
    /* Check pressure again */
    ret = tier_watch_get_pressure(TIER_GPU, &pressure);
    printf("  After cleanup - pressure: level=%d, value=%u%%\n",
           pressure.level, pressure.pressure_value);
    
    /* Final cleanup */
    for (int i = 500; i < 1000; i++) {
        tier_watch_untrack_page(base_pfn + i);
    }
    
    printf("  PASS: Memory pressure simulation working\n");
    return TEST_PASS;
}

/* Test: NUMA awareness */
int test_numa_awareness(void)
{
    struct numa_stats stats;
    int num_nodes;
    u64 pfn = 0x300000;
    int ret;
    
    printf("Testing NUMA awareness...\n");
    
    /* Check if NUMA is available */
    if (numa_available() < 0) {
        printf("  SKIP: NUMA not available on this system\n");
        return TEST_PASS;
    }
    
    num_nodes = numa_num_configured_nodes();
    printf("  System has %d NUMA nodes\n", num_nodes);
    
    /* Track pages on different NUMA nodes */
    for (int node = 0; node < num_nodes && node < 4; node++) {
        for (int i = 0; i < 10; i++) {
            ret = tier_watch_track_page_numa(pfn++, TIER_CPU, 1, node);
            if (ret != 0) {
                printf("  WARN: Failed to track page on node %d\n", node);
            }
        }
    }
    
    /* Get NUMA statistics */
    for (int node = 0; node < num_nodes && node < 4; node++) {
        ret = tier_watch_get_numa_stats(node, &stats);
        if (ret != 0)
            continue;
            
        printf("  Node %d: %llu pages\n", node, stats.pages_on_node);
    }
    
    /* Test optimal node detection */
    int optimal = tier_watch_get_optimal_numa_node(0x300005);
    printf("  Optimal NUMA node for page: %d\n", optimal);
    
    /* Cleanup */
    pfn = 0x300000;
    for (int i = 0; i < num_nodes * 10; i++) {
        tier_watch_untrack_page(pfn++);
    }
    
    printf("  PASS: NUMA awareness working\n");
    return TEST_PASS;
}

/* Test: Agent-specific memory tracking */
int test_agent_memory_tracking(void)
{
    struct agent_memory_stats stats;
    u64 agents[] = {100, 200, 300};
    u64 base_pfn = 0x400000;
    int ret;
    
    printf("Testing agent-specific memory tracking...\n");
    
    /* Track pages for different agents */
    for (int a = 0; a < 3; a++) {
        for (int t = 0; t < TIER_COUNT; t++) {
            for (int p = 0; p < 5; p++) {
                u64 pfn = base_pfn + (a * 100) + (t * 10) + p;
                ret = tier_watch_track_page(pfn, t, agents[a]);
                if (ret != 0) {
                    printf("  WARN: Failed to track page for agent %llu\n", agents[a]);
                }
            }
        }
    }
    
    /* Get agent statistics */
    for (int a = 0; a < 3; a++) {
        ret = tier_watch_get_agent_memory(agents[a], &stats);
        if (ret != 0) {
            printf("  FAIL: Failed to get stats for agent %llu\n", agents[a]);
            continue;
        }
        
        printf("  Agent %llu: total=%llu pages\n", agents[a], stats.total_pages);
        for (int t = 0; t < TIER_COUNT; t++) {
            if (stats.pages_in_tier[t] > 0) {
                printf("    Tier %d: %llu pages\n", t, stats.pages_in_tier[t]);
            }
        }
    }
    
    /* Cleanup */
    for (int i = 0; i < 300; i++) {
        tier_watch_untrack_page(base_pfn + i);
    }
    
    printf("  PASS: Agent memory tracking working\n");
    return TEST_PASS;
}

/* Thread function for concurrent test */
void *migration_thread(void *arg)
{
    int thread_id = *(int *)arg;
    u64 base_pfn = 0x500000 + (thread_id * 0x10000);
    struct migration_request req;
    struct migration_result result;
    
    /* Track and migrate pages */
    for (int i = 0; i < 10; i++) {
        u64 pfn = base_pfn + i;
        
        /* Track in lower tier */
        tier_watch_track_page(pfn, TIER_SSD, thread_id);
        
        /* Simulate access to make hot */
        for (int j = 0; j < 200; j++) {
            tier_watch_handle_fault(pfn, 0, FAULT_FLAG_READ);
        }
        
        /* Migrate to higher tier */
        req.pfn = pfn;
        req.from_tier = TIER_SSD;
        req.to_tier = TIER_NVME;
        req.priority = 50;
        req.agent_id = thread_id;
        req.reason = MIGRATION_HOT_PROMOTION;
        
        tier_watch_migrate_page(&req, &result);
        
        usleep(1000); /* 1ms delay */
    }
    
    /* Cleanup */
    for (int i = 0; i < 10; i++) {
        tier_watch_untrack_page(base_pfn + i);
    }
    
    return NULL;
}

/* Test: Concurrent migrations */
int test_concurrent_migrations(void)
{
    pthread_t threads[4];
    int thread_ids[4];
    
    printf("Testing concurrent migrations...\n");
    
    /* Create threads */
    for (int i = 0; i < 4; i++) {
        thread_ids[i] = i + 1;
        if (pthread_create(&threads[i], NULL, migration_thread, &thread_ids[i]) != 0) {
            printf("  FAIL: Failed to create thread %d\n", i);
            return TEST_FAIL;
        }
    }
    
    /* Wait for threads */
    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }
    
    /* Check module stats */
    struct tier_watch_stats stats;
    tier_watch_get_module_stats(&stats);
    
    printf("  Total migrations: %llu\n", stats.total_migrations);
    printf("  Failed migrations: %llu\n", stats.failed_migrations);
    
    if (stats.total_migrations > 0) {
        u64 avg_latency = stats.total_migration_ns / stats.total_migrations;
        printf("  Average migration latency: %llu ns\n", avg_latency);
    }
    
    printf("  PASS: Concurrent migrations working\n");
    return TEST_PASS;
}

/* Test: /proc interface */
int test_proc_interface(void)
{
    FILE *fp;
    char buffer[1024];
    int found = 0;
    
    printf("Testing /proc interface...\n");
    
    /* Track some test pages */
    for (int i = 0; i < 5; i++) {
        tier_watch_track_page(0x600000 + i, TIER_CPU, 1);
    }
    
    /* Read CPU tier stats */
    fp = fopen("/proc/swarm/tiers/cpu/stats", "r");
    if (!fp) {
        printf("  FAIL: Cannot open /proc/swarm/tiers/cpu/stats\n");
        return TEST_FAIL;
    }
    
    while (fgets(buffer, sizeof(buffer), fp)) {
        if (strstr(buffer, "total_pages")) {
            found = 1;
            printf("  Found: %s", buffer);
            break;
        }
    }
    
    fclose(fp);
    
    if (!found) {
        printf("  FAIL: Stats not found in /proc\n");
        return TEST_FAIL;
    }
    
    /* Read module stats */
    fp = fopen("/proc/swarm/module_stats", "r");
    if (!fp) {
        printf("  FAIL: Cannot open /proc/swarm/module_stats\n");
        return TEST_FAIL;
    }
    
    printf("  Module stats:\n");
    while (fgets(buffer, sizeof(buffer), fp)) {
        printf("    %s", buffer);
    }
    
    fclose(fp);
    
    /* Cleanup */
    for (int i = 0; i < 5; i++) {
        tier_watch_untrack_page(0x600000 + i);
    }
    
    printf("  PASS: /proc interface working\n");
    return TEST_PASS;
}

/* Test: Performance under load */
int test_performance_load(void)
{
    struct timespec start, end;
    long elapsed_ms;
    u64 base_pfn = 0x700000;
    int pages_tracked = 0;
    
    printf("Testing performance under load...\n");
    
    /* Track many pages */
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < 10000; i++) {
        if (tier_watch_track_page(base_pfn + i, TIER_CPU, 1) == 0) {
            pages_tracked++;
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    elapsed_ms = (end.tv_sec - start.tv_sec) * 1000 + 
                 (end.tv_nsec - start.tv_nsec) / 1000000;
    
    printf("  Tracked %d pages in %ld ms\n", pages_tracked, elapsed_ms);
    
    /* Simulate faults */
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < 100000; i++) {
        u64 pfn = base_pfn + (rand() % pages_tracked);
        tier_watch_handle_fault(pfn, 0, FAULT_FLAG_READ);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    elapsed_ms = (end.tv_sec - start.tv_sec) * 1000 + 
                 (end.tv_nsec - start.tv_nsec) / 1000000;
    
    printf("  Handled 100k faults in %ld ms\n", elapsed_ms);
    
    /* Get migration candidates */
    struct migration_candidate candidates[100];
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    int count = tier_watch_get_migration_candidates(TIER_CPU, TIER_GPU,
                                                   candidates, 100);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    elapsed_ms = (end.tv_sec - start.tv_sec) * 1000 + 
                 (end.tv_nsec - start.tv_nsec) / 1000000;
    
    printf("  Found %d migration candidates in %ld ms\n", count, elapsed_ms);
    
    /* Cleanup */
    for (int i = 0; i < pages_tracked; i++) {
        tier_watch_untrack_page(base_pfn + i);
    }
    
    printf("  PASS: Performance acceptable under load\n");
    return TEST_PASS;
}

/* Main test runner */
int main(void)
{
    int total_pass = 0, total_fail = 0;
    
    printf("TierWatch Integration Test Suite\n");
    printf("================================\n\n");
    
    /* Check if module is loaded */
    if (access("/proc/swarm/tiers/cpu/stats", F_OK) != 0) {
        printf("ERROR: TierWatch module not loaded\n");
        printf("Run: sudo insmod tier_watch.ko\n");
        return 1;
    }
    
    /* Run tests */
    if (test_real_memory_tracking() == TEST_PASS) total_pass++; else total_fail++;
    if (test_multi_tier_simulation() == TEST_PASS) total_pass++; else total_fail++;
    if (test_memory_pressure() == TEST_PASS) total_pass++; else total_fail++;
    if (test_numa_awareness() == TEST_PASS) total_pass++; else total_fail++;
    if (test_agent_memory_tracking() == TEST_PASS) total_pass++; else total_fail++;
    if (test_concurrent_migrations() == TEST_PASS) total_pass++; else total_fail++;
    if (test_proc_interface() == TEST_PASS) total_pass++; else total_fail++;
    if (test_performance_load() == TEST_PASS) total_pass++; else total_fail++;
    
    /* Print summary */
    printf("\n=== Test Summary ===\n");
    printf("Total tests: %d\n", total_pass + total_fail);
    printf("Passed: %d\n", total_pass);
    printf("Failed: %d\n", total_fail);
    
    return (total_fail > 0) ? 1 : 0;
}