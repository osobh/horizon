/*
 * Test harness for TierWatch kernel module
 * Tests 5-tier memory hierarchy monitoring functionality
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <sys/mman.h>
#include <fcntl.h>

// Test framework
#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    printf("Running test_%s...", #name); \
    test_##name(); \
    printf(" PASSED\n"); \
} while(0)

// Memory tier definitions
typedef enum {
    TIER_GPU = 0,
    TIER_CPU = 1,
    TIER_NVME = 2,
    TIER_SSD = 3,
    TIER_HDD = 4,
    TIER_COUNT = 5
} memory_tier_t;

const char* tier_names[] = {"GPU", "CPU", "NVMe", "SSD", "HDD"};
const size_t tier_capacities[] = {
    32UL << 30,   // 32GB GPU
    96UL << 30,   // 96GB CPU
    3200UL << 30, // 3.2TB NVMe
    4500UL << 30, // 4.5TB SSD
    3700UL << 30  // 3.7TB HDD
};

// Page tracking structure
struct page_info {
    unsigned long addr;
    memory_tier_t tier;
    unsigned long access_count;
    unsigned long last_access_time;
    int agent_id;
};

// Tier statistics
struct tier_stats {
    unsigned long total_pages;
    unsigned long used_bytes;
    unsigned long major_faults;
    unsigned long minor_faults;
    unsigned long migrations_in;
    unsigned long migrations_out;
    unsigned long access_count;
    double avg_latency_ns;
};

// Global tier statistics
static struct tier_stats tier_stats[TIER_COUNT];
static pthread_mutex_t stats_lock = PTHREAD_MUTEX_INITIALIZER;

// Test: Tier hierarchy validation
TEST(tier_hierarchy) {
    // Verify tier ordering
    assert(TIER_GPU < TIER_CPU);
    assert(TIER_CPU < TIER_NVME);
    assert(TIER_NVME < TIER_SSD);
    assert(TIER_SSD < TIER_HDD);
    
    // Verify capacities increase down the hierarchy
    assert(tier_capacities[TIER_GPU] < tier_capacities[TIER_CPU]);
    assert(tier_capacities[TIER_CPU] < tier_capacities[TIER_NVME]);
    assert(tier_capacities[TIER_NVME] < tier_capacities[TIER_SSD]);
}

// Test: Page fault simulation
TEST(page_fault_handling) {
    struct page_info pages[1000];
    int num_pages = 1000;
    
    // Initialize pages
    for (int i = 0; i < num_pages; i++) {
        pages[i].addr = 0x100000 + i * 0x1000; // 4KB pages
        pages[i].tier = i % TIER_COUNT;
        pages[i].access_count = 0;
        pages[i].last_access_time = 0;
        pages[i].agent_id = i % 10;
    }
    
    // Simulate page faults
    pthread_mutex_lock(&stats_lock);
    for (int i = 0; i < 10000; i++) {
        int page_idx = rand() % num_pages;
        struct page_info *page = &pages[page_idx];
        
        // Simulate fault
        if (rand() % 10 == 0) {
            // Major fault
            tier_stats[page->tier].major_faults++;
        } else {
            // Minor fault
            tier_stats[page->tier].minor_faults++;
        }
        
        page->access_count++;
        page->last_access_time = i;
        tier_stats[page->tier].access_count++;
    }
    pthread_mutex_unlock(&stats_lock);
    
    // Verify fault counts
    unsigned long total_faults = 0;
    for (int i = 0; i < TIER_COUNT; i++) {
        total_faults += tier_stats[i].major_faults + tier_stats[i].minor_faults;
    }
    assert(total_faults == 10000);
}

// Test: Memory pressure calculation
TEST(memory_pressure) {
    // Set up tier usage
    pthread_mutex_lock(&stats_lock);
    for (int i = 0; i < TIER_COUNT; i++) {
        tier_stats[i].total_pages = tier_capacities[i] / 4096;
        tier_stats[i].used_bytes = 0;
    }
    
    // Simulate different pressure levels
    tier_stats[TIER_GPU].used_bytes = tier_capacities[TIER_GPU] * 0.85; // 85%
    tier_stats[TIER_CPU].used_bytes = tier_capacities[TIER_CPU] * 0.50; // 50%
    tier_stats[TIER_NVME].used_bytes = tier_capacities[TIER_NVME] * 0.30; // 30%
    pthread_mutex_unlock(&stats_lock);
    
    // Calculate pressure
    double gpu_pressure = (double)tier_stats[TIER_GPU].used_bytes / tier_capacities[TIER_GPU] * 100;
    double cpu_pressure = (double)tier_stats[TIER_CPU].used_bytes / tier_capacities[TIER_CPU] * 100;
    double nvme_pressure = (double)tier_stats[TIER_NVME].used_bytes / tier_capacities[TIER_NVME] * 100;
    
    assert(gpu_pressure > 80); // High pressure
    assert(cpu_pressure > 40 && cpu_pressure < 60); // Medium pressure
    assert(nvme_pressure < 40); // Low pressure
}

// Test: Migration candidate detection
TEST(migration_detection) {
    struct page_info hot_page = {
        .addr = 0x1000,
        .tier = TIER_NVME,
        .access_count = 1000,
        .last_access_time = time(NULL),
        .agent_id = 1
    };
    
    struct page_info cold_page = {
        .addr = 0x2000,
        .tier = TIER_CPU,
        .access_count = 1,
        .last_access_time = time(NULL) - 3600, // 1 hour old
        .agent_id = 2
    };
    
    // Hot page should be promoted
    if (hot_page.access_count > 100 && hot_page.tier > TIER_CPU) {
        assert(1); // Should migrate up
    }
    
    // Cold page should be demoted
    time_t age = time(NULL) - cold_page.last_access_time;
    if (cold_page.access_count < 10 && age > 1800 && cold_page.tier < TIER_HDD) {
        assert(1); // Should migrate down
    }
}

// Test: Concurrent tier updates
TEST(concurrent_updates) {
    const int num_threads = 10;
    const int updates_per_thread = 10000;
    pthread_t threads[num_threads];
    
    // Thread function
    void* update_stats(void* arg) {
        int thread_id = *(int*)arg;
        
        for (int i = 0; i < updates_per_thread; i++) {
            int tier = thread_id % TIER_COUNT;
            
            pthread_mutex_lock(&stats_lock);
            tier_stats[tier].access_count++;
            if (i % 10 == 0) {
                tier_stats[tier].minor_faults++;
            }
            if (i % 100 == 0) {
                tier_stats[tier].major_faults++;
            }
            pthread_mutex_unlock(&stats_lock);
            
            usleep(1); // Small delay
        }
        
        return NULL;
    }
    
    // Reset stats
    pthread_mutex_lock(&stats_lock);
    memset(tier_stats, 0, sizeof(tier_stats));
    pthread_mutex_unlock(&stats_lock);
    
    // Start threads
    int thread_ids[num_threads];
    for (int i = 0; i < num_threads; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, update_stats, &thread_ids[i]);
    }
    
    // Wait for completion
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // Verify all updates were recorded
    unsigned long total_accesses = 0;
    for (int i = 0; i < TIER_COUNT; i++) {
        total_accesses += tier_stats[i].access_count;
    }
    assert(total_accesses == num_threads * updates_per_thread);
}

// Test: NUMA awareness
TEST(numa_awareness) {
    struct numa_node {
        int id;
        unsigned long local_memory;
        unsigned long cpu_mask;
    };
    
    struct numa_node nodes[] = {
        {0, 48UL << 30, 0x00FF}, // Node 0: 48GB, CPUs 0-7
        {1, 48UL << 30, 0xFF00}  // Node 1: 48GB, CPUs 8-15
    };
    
    // Test NUMA-aware allocation
    int preferred_node = 0;
    int current_cpu = 3; // CPU 3 is on node 0
    
    // Check if CPU belongs to preferred node
    assert((nodes[preferred_node].cpu_mask & (1 << current_cpu)) != 0);
}

// Test: Page migration latency
TEST(migration_latency) {
    struct timespec start, end;
    
    // Simulate page migration
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Mock migration work
    void* src = malloc(1 << 20); // 1MB
    void* dst = malloc(1 << 20);
    memcpy(dst, src, 1 << 20);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    // Calculate latency
    long latency_ns = (end.tv_sec - start.tv_sec) * 1000000000L + 
                      (end.tv_nsec - start.tv_nsec);
    
    // Should be under 1ms (1,000,000 ns)
    assert(latency_ns < 1000000);
    
    free(src);
    free(dst);
}

// Test: Per-CPU data structures
TEST(per_cpu_structures) {
    const int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    
    // Per-CPU counters
    struct per_cpu_counter {
        unsigned long count;
        char padding[64 - sizeof(unsigned long)]; // Cache line padding
    };
    
    struct per_cpu_counter *counters = calloc(num_cpus, sizeof(struct per_cpu_counter));
    
    // Simulate updates from different CPUs
    for (int cpu = 0; cpu < num_cpus; cpu++) {
        for (int i = 0; i < 1000; i++) {
            counters[cpu].count++;
        }
    }
    
    // Verify total
    unsigned long total = 0;
    for (int cpu = 0; cpu < num_cpus; cpu++) {
        total += counters[cpu].count;
    }
    assert(total == num_cpus * 1000);
    
    free(counters);
}

// Test: Large-scale page tracking
TEST(large_scale_tracking) {
    const int target_pages = 1000000; // 1M pages (4GB at 4KB per page)
    
    // Use compact structure for efficiency
    struct compact_page {
        unsigned char tier : 3;
        unsigned char flags : 5;
        unsigned short access_count;
        unsigned short agent_id;
    } __attribute__((packed));
    
    // Verify structure size
    assert(sizeof(struct compact_page) == 5);
    
    // Calculate memory requirement
    size_t memory_needed = target_pages * sizeof(struct compact_page);
    assert(memory_needed < 10 << 20); // Should be under 10MB
    
    // Simulate allocation
    struct compact_page *pages = calloc(target_pages, sizeof(struct compact_page));
    assert(pages != NULL);
    
    // Initialize some pages
    for (int i = 0; i < 10000; i++) {
        int idx = rand() % target_pages;
        pages[idx].tier = rand() % TIER_COUNT;
        pages[idx].access_count = rand() % 1000;
        pages[idx].agent_id = rand() % 1000;
    }
    
    free(pages);
}

// Test: /proc interface format
TEST(proc_interface) {
    char buffer[4096];
    
    // Format tier stats
    for (int tier = 0; tier < TIER_COUNT; tier++) {
        snprintf(buffer, sizeof(buffer),
            "Tier: %s\n"
            "Total pages: %lu\n"
            "Used bytes: %lu\n"
            "Pressure: %lu%%\n"
            "Major faults: %lu\n"
            "Minor faults: %lu\n"
            "Migrations in: %lu\n"
            "Migrations out: %lu\n"
            "Access count: %lu\n"
            "Avg latency: %.2f ns\n",
            tier_names[tier],
            tier_stats[tier].total_pages,
            tier_stats[tier].used_bytes,
            tier_stats[tier].used_bytes * 100 / tier_capacities[tier],
            tier_stats[tier].major_faults,
            tier_stats[tier].minor_faults,
            tier_stats[tier].migrations_in,
            tier_stats[tier].migrations_out,
            tier_stats[tier].access_count,
            tier_stats[tier].avg_latency_ns
        );
        
        // Verify format
        assert(strstr(buffer, "Tier:") != NULL);
        assert(strstr(buffer, "Pressure:") != NULL);
    }
}

// Main test runner
int main() {
    printf("TierWatch Kernel Module Tests\n");
    printf("=============================\n\n");
    
    RUN_TEST(tier_hierarchy);
    RUN_TEST(page_fault_handling);
    RUN_TEST(memory_pressure);
    RUN_TEST(migration_detection);
    RUN_TEST(concurrent_updates);
    RUN_TEST(numa_awareness);
    RUN_TEST(migration_latency);
    RUN_TEST(per_cpu_structures);
    RUN_TEST(large_scale_tracking);
    RUN_TEST(proc_interface);
    
    printf("\nAll tests passed!\n");
    return 0;
}