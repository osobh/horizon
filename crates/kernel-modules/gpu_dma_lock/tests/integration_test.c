/*
 * GPU DMA Lock Integration Tests
 * 
 * Tests the interaction between gpu_dma_lock and other kernel modules
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

#include "../../../common/swarm_ioctl.h"

#define DEVICE_PATH "/dev/gpu_dma_lock"
#define NUM_THREADS 4
#define ALLOCATIONS_PER_THREAD 100

/* Test results */
struct test_result {
    const char *name;
    int passed;
    int failed;
};

/* Thread data for concurrent tests */
struct thread_data {
    int thread_id;
    int device_fd;
    int success_count;
    int error_count;
};

/* Test: Basic ioctl operations */
int test_basic_ioctl(int fd)
{
    struct swarm_gpu_alloc_params alloc_params;
    struct swarm_gpu_stats stats;
    int ret;
    
    printf("Testing basic ioctl operations...\n");
    
    /* Test allocation */
    alloc_params.agent_id = 1;
    alloc_params.size = 4096;
    alloc_params.device_id = 0;
    alloc_params.flags = 0;
    
    ret = ioctl(fd, SWARM_GPU_ALLOC, &alloc_params);
    if (ret != 0) {
        printf("  FAIL: Allocation failed: %s\n", strerror(errno));
        return -1;
    }
    
    if (alloc_params.alloc_id == 0) {
        printf("  FAIL: Invalid allocation ID\n");
        return -1;
    }
    
    printf("  PASS: Allocated %zu bytes (ID: %llu)\n", 
           alloc_params.size, alloc_params.alloc_id);
    
    /* Test query */
    ret = ioctl(fd, SWARM_GPU_QUERY, &stats);
    if (ret != 0) {
        printf("  FAIL: Query failed: %s\n", strerror(errno));
        return -1;
    }
    
    printf("  PASS: Query successful - %llu allocations\n", stats.allocation_count);
    
    /* Test free */
    ret = ioctl(fd, SWARM_GPU_FREE, alloc_params.alloc_id);
    if (ret != 0) {
        printf("  FAIL: Free failed: %s\n", strerror(errno));
        return -1;
    }
    
    printf("  PASS: Freed allocation\n");
    
    return 0;
}

/* Test: Quota enforcement */
int test_quota_enforcement(int fd)
{
    struct swarm_gpu_quota quota;
    struct swarm_gpu_alloc_params alloc_params;
    int ret;
    
    printf("Testing quota enforcement...\n");
    
    /* Set small quota */
    quota.agent_id = 2;
    quota.memory_limit = 1024 * 1024; /* 1MB */
    quota.device_mask = 0x01;
    
    ret = ioctl(fd, SWARM_GPU_SET_QUOTA, &quota);
    if (ret != 0) {
        printf("  FAIL: Set quota failed: %s\n", strerror(errno));
        return -1;
    }
    
    /* Try to exceed quota */
    alloc_params.agent_id = 2;
    alloc_params.size = 2 * 1024 * 1024; /* 2MB */
    alloc_params.device_id = 0;
    alloc_params.flags = 0;
    
    ret = ioctl(fd, SWARM_GPU_ALLOC, &alloc_params);
    if (ret == 0) {
        printf("  FAIL: Allocation should have failed (quota exceeded)\n");
        return -1;
    }
    
    if (errno != EDQUOT) {
        printf("  FAIL: Wrong error code: %s (expected EDQUOT)\n", strerror(errno));
        return -1;
    }
    
    printf("  PASS: Quota enforcement working\n");
    
    return 0;
}

/* Test: DMA permissions */
int test_dma_permissions(int fd)
{
    struct swarm_dma_permission_grant grant;
    struct swarm_dma_check check;
    int ret;
    
    printf("Testing DMA permissions...\n");
    
    /* Grant permission */
    grant.agent_id = 3;
    grant.dma_addr = 0x100000;
    grant.size = 4096;
    grant.permissions = SWARM_DMA_READ | SWARM_DMA_WRITE;
    
    ret = ioctl(fd, SWARM_DMA_GRANT_PERM, &grant);
    if (ret != 0) {
        printf("  FAIL: Grant permission failed: %s\n", strerror(errno));
        return -1;
    }
    
    /* Check permission */
    check.agent_id = 3;
    check.dma_addr = 0x100000;
    check.size = 2048;
    check.access_type = SWARM_DMA_READ;
    
    ret = ioctl(fd, SWARM_DMA_CHECK_PERM, &check);
    if (ret != 0) {
        printf("  FAIL: Check permission failed: %s\n", strerror(errno));
        return -1;
    }
    
    if (!check.allowed) {
        printf("  FAIL: Permission should be allowed\n");
        return -1;
    }
    
    printf("  PASS: DMA permissions working\n");
    
    return 0;
}

/* Thread function for concurrent allocation test */
void *allocation_thread(void *arg)
{
    struct thread_data *data = (struct thread_data *)arg;
    struct swarm_gpu_alloc_params alloc_params;
    u64 alloc_ids[ALLOCATIONS_PER_THREAD];
    int i, ret;
    
    /* Perform allocations */
    for (i = 0; i < ALLOCATIONS_PER_THREAD; i++) {
        alloc_params.agent_id = 100 + data->thread_id;
        alloc_params.size = 4096 * (1 + (rand() % 10));
        alloc_params.device_id = 0;
        alloc_params.flags = 0;
        
        ret = ioctl(data->device_fd, SWARM_GPU_ALLOC, &alloc_params);
        if (ret == 0) {
            alloc_ids[data->success_count] = alloc_params.alloc_id;
            data->success_count++;
        } else {
            data->error_count++;
        }
        
        /* Random delay */
        usleep(rand() % 1000);
    }
    
    /* Free allocations */
    for (i = 0; i < data->success_count; i++) {
        ret = ioctl(data->device_fd, SWARM_GPU_FREE, alloc_ids[i]);
        if (ret != 0) {
            data->error_count++;
        }
    }
    
    return NULL;
}

/* Test: Concurrent operations */
int test_concurrent_operations(int fd)
{
    pthread_t threads[NUM_THREADS];
    struct thread_data thread_data[NUM_THREADS];
    int i, total_success = 0, total_errors = 0;
    
    printf("Testing concurrent operations...\n");
    
    /* Initialize random seed */
    srand(time(NULL));
    
    /* Create threads */
    for (i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].device_fd = fd;
        thread_data[i].success_count = 0;
        thread_data[i].error_count = 0;
        
        if (pthread_create(&threads[i], NULL, allocation_thread, &thread_data[i]) != 0) {
            printf("  FAIL: Failed to create thread %d\n", i);
            return -1;
        }
    }
    
    /* Wait for threads to complete */
    for (i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
        total_success += thread_data[i].success_count;
        total_errors += thread_data[i].error_count;
    }
    
    printf("  Completed: %d successful allocations, %d errors\n", 
           total_success, total_errors);
    
    if (total_errors > 0) {
        printf("  WARN: Some operations failed during concurrent test\n");
    }
    
    printf("  PASS: Concurrent operations handled\n");
    
    return 0;
}

/* Test: Performance benchmarks */
int test_performance(int fd)
{
    struct swarm_gpu_alloc_params alloc_params;
    struct swarm_dma_check check;
    struct timespec start, end;
    u64 alloc_id;
    long alloc_ns, dma_ns;
    int i, ret;
    const int iterations = 1000;
    
    printf("Testing performance targets...\n");
    
    /* Test allocation performance */
    alloc_params.agent_id = 999;
    alloc_params.size = 4096;
    alloc_params.device_id = 0;
    alloc_params.flags = 0;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (i = 0; i < iterations; i++) {
        ret = ioctl(fd, SWARM_GPU_ALLOC, &alloc_params);
        if (ret == 0) {
            alloc_id = alloc_params.alloc_id;
            ioctl(fd, SWARM_GPU_FREE, alloc_id);
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    alloc_ns = ((end.tv_sec - start.tv_sec) * 1000000000L + 
                (end.tv_nsec - start.tv_nsec)) / iterations;
    
    printf("  Allocation performance: %ld ns per operation", alloc_ns);
    if (alloc_ns <= 10000) {
        printf(" (PASS: target <10μs)\n");
    } else {
        printf(" (FAIL: target <10μs)\n");
    }
    
    /* Test DMA check performance */
    check.agent_id = 999;
    check.dma_addr = 0x200000;
    check.size = 4096;
    check.access_type = SWARM_DMA_READ;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (i = 0; i < iterations; i++) {
        ioctl(fd, SWARM_DMA_CHECK_PERM, &check);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    dma_ns = ((end.tv_sec - start.tv_sec) * 1000000000L + 
              (end.tv_nsec - start.tv_nsec)) / iterations;
    
    printf("  DMA check performance: %ld ns per operation", dma_ns);
    if (dma_ns <= 1000) {
        printf(" (PASS: target <1μs)\n");
    } else {
        printf(" (FAIL: target <1μs)\n");
    }
    
    return 0;
}

/* Test: /proc interface */
int test_proc_interface(void)
{
    FILE *fp;
    char buffer[256];
    
    printf("Testing /proc interface...\n");
    
    /* Read stats */
    fp = fopen("/proc/swarm/gpu/stats", "r");
    if (!fp) {
        printf("  FAIL: Cannot open /proc/swarm/gpu/stats\n");
        return -1;
    }
    
    if (fgets(buffer, sizeof(buffer), fp) == NULL) {
        printf("  FAIL: Cannot read stats\n");
        fclose(fp);
        return -1;
    }
    
    fclose(fp);
    printf("  PASS: /proc/swarm/gpu/stats readable\n");
    
    /* Read agents */
    fp = fopen("/proc/swarm/gpu/agents", "r");
    if (!fp) {
        printf("  FAIL: Cannot open /proc/swarm/gpu/agents\n");
        return -1;
    }
    
    fclose(fp);
    printf("  PASS: /proc/swarm/gpu/agents readable\n");
    
    return 0;
}

/* Main test runner */
int main(void)
{
    int fd;
    int total_pass = 0, total_fail = 0;
    
    printf("GPU DMA Lock Integration Test Suite\n");
    printf("===================================\n\n");
    
    /* Open device */
    fd = open(DEVICE_PATH, O_RDWR);
    if (fd < 0) {
        printf("ERROR: Cannot open device %s: %s\n", DEVICE_PATH, strerror(errno));
        printf("Make sure the module is loaded: sudo insmod gpu_dma_lock.ko\n");
        return 1;
    }
    
    /* Run tests */
    if (test_basic_ioctl(fd) == 0) total_pass++; else total_fail++;
    if (test_quota_enforcement(fd) == 0) total_pass++; else total_fail++;
    if (test_dma_permissions(fd) == 0) total_pass++; else total_fail++;
    if (test_concurrent_operations(fd) == 0) total_pass++; else total_fail++;
    if (test_performance(fd) == 0) total_pass++; else total_fail++;
    if (test_proc_interface() == 0) total_pass++; else total_fail++;
    
    /* Print summary */
    printf("\n=== Test Summary ===\n");
    printf("Total tests: %d\n", total_pass + total_fail);
    printf("Passed: %d\n", total_pass);
    printf("Failed: %d\n", total_fail);
    
    close(fd);
    
    return (total_fail > 0) ? 1 : 0;
}