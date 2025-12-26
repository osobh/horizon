/*
 * TierWatch Kernel Module Header
 * 
 * 5-tier memory hierarchy monitoring and migration for StratoSwarm
 */

#ifndef _TIER_WATCH_H
#define _TIER_WATCH_H

#include <linux/types.h>
#include <linux/list.h>
#include <linux/spinlock.h>
#include <linux/rbtree.h>
#include <linux/mm_types.h>
#include <linux/numa.h>
#include "../../common/swarm_core.h"
#include "../../common/swarm_ioctl.h"

/* Module configuration */
#define TIER_WATCH_VERSION "2.0"
#define MAX_TRACKED_PAGES 10000000 /* 10M pages = 40GB at 4KB/page */
#define MAX_MIGRATION_BATCH 256
#define HOT_PAGE_THRESHOLD 100
#define COLD_PAGE_THRESHOLD 10
#define MIGRATION_COST_NS 10000 /* 10μs per page migration */

/* Memory tier definitions */
enum memory_tier {
    TIER_GPU = 0,
    TIER_CPU = 1,
    TIER_NVME = 2,
    TIER_SSD = 3,
    TIER_HDD = 4,
    TIER_COUNT = 5
};

/* Pressure levels */
enum pressure_level {
    PRESSURE_LOW = 0,
    PRESSURE_MEDIUM = 1,
    PRESSURE_HIGH = 2,
    PRESSURE_CRITICAL = 3
};

/* Migration reasons */
enum migration_reason {
    MIGRATION_HOT_PROMOTION = 0,
    MIGRATION_COLD_DEMOTION = 1,
    MIGRATION_MEMORY_PRESSURE = 2,
    MIGRATION_NUMA_BALANCE = 3,
    MIGRATION_AGENT_REQUEST = 4
};

/* Tier information */
struct tier_info {
    enum memory_tier tier;
    u64 capacity_bytes;
    u64 used_bytes;
    u64 free_bytes;
    u32 latency_ns;
    u32 bandwidth_mbps;
    const char *name;
};

/* Page information for tracking */
struct page_info {
    u64 pfn;                    /* Physical page frame number */
    u64 vaddr;                  /* Virtual address */
    enum memory_tier tier;      /* Current tier */
    u32 access_count;          /* Access count */
    u64 last_access_jiffies;   /* Last access time */
    u64 agent_id;              /* Owner agent ID (0 = unassigned) */
    u8 numa_node;              /* NUMA node */
    u8 flags;                  /* Page flags */
};

/* Compact page info for efficient storage */
struct compact_page_info {
    u8 tier_and_flags;         /* Tier (3 bits) + flags (5 bits) */
    u16 access_count;          /* Access count (capped) */
    u16 agent_id;              /* Agent ID (limited to 65535) */
} __packed;

/* Fault statistics per tier */
struct fault_stats {
    u64 total_faults;
    u64 read_faults;
    u64 write_faults;
    u64 minor_faults;
    u64 major_faults;
    u64 avg_latency_ns;
};

/* Migration candidate */
struct migration_candidate {
    u64 pfn;
    enum memory_tier from_tier;
    enum memory_tier to_tier;
    u32 access_count;
    u32 score;                 /* Migration benefit score */
    u64 agent_id;
};

/* Migration request */
struct migration_request {
    u64 pfn;
    enum memory_tier from_tier;
    enum memory_tier to_tier;
    u8 priority;               /* 0=low, 100=high */
    u64 agent_id;
    enum migration_reason reason;
};

/* Migration result */
struct migration_result {
    int status;                /* 0=success, <0=error */
    u64 latency_ns;           /* Migration latency */
    u32 pages_moved;          /* Number of pages moved */
};

/* Tier statistics */
struct tier_stats {
    u64 total_pages;
    u64 used_pages;
    u64 hot_pages;
    u64 cold_pages;
    u64 migrations_in;
    u64 migrations_out;
    u64 total_faults;
    enum pressure_level pressure;
};

/* NUMA statistics */
struct numa_stats {
    u32 node_id;
    u64 pages_on_node;
    u64 local_accesses;
    u64 remote_accesses;
    u64 migrations_to_node;
    u64 migrations_from_node;
};

/* Agent memory statistics */
struct agent_memory_stats {
    u64 agent_id;
    u64 total_pages;
    u64 pages_in_tier[TIER_COUNT];
    u64 total_faults;
    u64 migrations;
};

/* Memory pressure info */
struct tier_pressure {
    enum memory_tier tier;
    enum pressure_level level;
    u64 free_pages;
    u64 reclaimable_pages;
    u32 pressure_value;        /* 0-100 */
};

/* Page list for batch operations */
struct page_list {
    u64 *pages;
    u32 count;
    u32 capacity;
};

/* Module statistics */
struct tier_watch_stats {
    atomic64_t pages_tracked;
    atomic64_t total_faults;
    atomic64_t total_migrations;
    atomic64_t total_migration_ns;
    atomic64_t failed_migrations;
};

/* Function prototypes */

/* Module initialization */
int tier_watch_init(void);
void tier_watch_exit(void);

/* Tier information */
int tier_watch_get_tier_info(enum memory_tier tier, struct tier_info *info);
bool tier_watch_is_faster(enum memory_tier tier1, enum memory_tier tier2);
enum memory_tier tier_watch_get_next_tier(enum memory_tier tier);
enum memory_tier tier_watch_get_prev_tier(enum memory_tier tier);

/* Page tracking */
int tier_watch_track_page(u64 pfn, enum memory_tier tier, u64 agent_id);
int tier_watch_track_page_numa(u64 pfn, enum memory_tier tier, u64 agent_id, int node);
int tier_watch_untrack_page(u64 pfn);
int tier_watch_get_page_info(u64 pfn, struct page_info *info);

/* Fault handling */
int tier_watch_handle_fault(u64 pfn, u64 vaddr, unsigned int flags);
int tier_watch_get_fault_stats(enum memory_tier tier, struct fault_stats *stats);
void tier_watch_reset_fault_stats(void);

/* Hot/cold detection */
int tier_watch_get_hot_pages(enum memory_tier tier, struct page_list *pages, u32 max_pages);
int tier_watch_get_cold_pages(enum memory_tier tier, struct page_list *pages, u32 max_pages);
int tier_watch_mark_page_hot(u64 pfn);
int tier_watch_mark_page_cold(u64 pfn);

/* Migration */
int tier_watch_get_migration_candidates(enum memory_tier from_tier, 
                                       enum memory_tier to_tier,
                                       struct migration_candidate *candidates,
                                       u32 max_candidates);
int tier_watch_migrate_page(struct migration_request *req, struct migration_result *result);
int tier_watch_migrate_pages_batch(struct migration_request *reqs, u32 count,
                                  struct migration_result *results);

/* Memory pressure */
int tier_watch_get_pressure(enum memory_tier tier, struct tier_pressure *pressure);
void tier_watch_update_pressure(void);
int tier_watch_set_pressure_threshold(enum memory_tier tier, enum pressure_level level);

/* NUMA support */
int tier_watch_get_numa_stats(int node, struct numa_stats *stats);
int tier_watch_get_optimal_numa_node(u64 pfn);
int tier_watch_balance_numa(void);

/* Agent memory tracking */
int tier_watch_get_agent_memory(u64 agent_id, struct agent_memory_stats *stats);
int tier_watch_set_agent_tier_limit(u64 agent_id, enum memory_tier tier, u64 max_pages);
int tier_watch_migrate_agent_pages(u64 agent_id, enum memory_tier from_tier,
                                  enum memory_tier to_tier, u32 max_pages);

/* Tier statistics */
int tier_watch_get_tier_stats(enum memory_tier tier, struct tier_stats *stats);
int tier_watch_get_module_stats(struct tier_watch_stats *stats);
void tier_watch_reset_stats(void);

/* /proc interface */
int tier_watch_proc_init(void);
void tier_watch_proc_exit(void);

/* Performance monitoring */
u64 tier_watch_get_avg_fault_latency(enum memory_tier tier);
u64 tier_watch_get_avg_migration_latency(void);

/* Tier capacity information (static) */
static inline u64 tier_capacity_bytes(enum memory_tier tier)
{
    switch (tier) {
    case TIER_GPU:  return 32ULL << 30;    /* 32GB */
    case TIER_CPU:  return 96ULL << 30;    /* 96GB */
    case TIER_NVME: return 3200ULL << 30;  /* 3.2TB */
    case TIER_SSD:  return 4500ULL << 30;  /* 4.5TB */
    case TIER_HDD:  return 3700ULL << 30;  /* 3.7TB */
    default: return 0;
    }
}

/* Tier latency information (static) */
static inline u32 tier_latency_ns(enum memory_tier tier)
{
    switch (tier) {
    case TIER_GPU:  return 200;         /* 200ns */
    case TIER_CPU:  return 50;          /* 50ns */
    case TIER_NVME: return 20000;       /* 20μs */
    case TIER_SSD:  return 100000;      /* 100μs */
    case TIER_HDD:  return 10000000;    /* 10ms */
    default: return UINT_MAX;
    }
}

#endif /* _TIER_WATCH_H */