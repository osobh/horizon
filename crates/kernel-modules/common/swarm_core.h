/*
 * StratoSwarm Core Definitions
 * 
 * Common definitions shared across all StratoSwarm kernel modules
 */

#ifndef _SWARM_CORE_H
#define _SWARM_CORE_H

#include <linux/types.h>
#include <linux/limits.h>
#include <linux/list.h>
#include <linux/spinlock.h>
#include <linux/atomic.h>

/* Module name prefixes */
#define SWARM_MODULE_PREFIX "swarm_"
#define SWARM_PROC_ROOT "swarm"

/* Maximum limits */
#define SWARM_MAX_AGENTS      200000
#define SWARM_MAX_GPU_DEVICES 8
#define SWARM_MAX_MEMORY_TIERS 5
#define SWARM_MAX_AGENT_NAME  64

/* Performance targets */
#define SWARM_TARGET_ALLOC_US 10
#define SWARM_TARGET_DMA_NS   1000
#define SWARM_TARGET_SYSCALL_NS 500

/* Agent structure - shared across modules */
struct swarm_agent {
    u64 id;
    pid_t pid;
    char name[SWARM_MAX_AGENT_NAME];
    
    /* Resource limits */
    u64 memory_limit;
    u32 cpu_quota;
    u64 gpu_memory_limit;
    
    /* Statistics */
    atomic64_t memory_usage;
    atomic64_t gpu_memory_usage;
    atomic_t violation_count;
    
    /* Namespace info */
    u32 namespace_flags;
    
    /* List management */
    struct list_head list;
    struct rcu_head rcu;
};

/* Memory tier definitions */
enum swarm_memory_tier {
    SWARM_TIER_GPU = 0,
    SWARM_TIER_CPU = 1,
    SWARM_TIER_NVME = 2,
    SWARM_TIER_SSD = 3,
    SWARM_TIER_HDD = 4,
};

/* GPU allocation tracking */
struct swarm_gpu_allocation {
    u64 id;
    u64 agent_id;
    u64 gpu_addr;
    size_t size;
    u32 device_id;
    struct list_head list;
};

/* DMA permission entry */
struct swarm_dma_permission {
    u64 agent_id;
    u64 dma_addr;
    size_t size;
    u32 permissions; /* Read/Write flags */
    struct list_head list;
};

/* Common error codes */
#define SWARM_SUCCESS         0
#define SWARM_ERR_NOMEM      -ENOMEM
#define SWARM_ERR_QUOTA      -EDQUOT
#define SWARM_ERR_PERM       -EPERM
#define SWARM_ERR_NOENT      -ENOENT
#define SWARM_ERR_BUSY       -EBUSY
#define SWARM_ERR_INVALID    -EINVAL

/* Helper macros */
#define SWARM_DEBUG(fmt, ...) \
    pr_debug("swarm: " fmt "\n", ##__VA_ARGS__)

#define SWARM_INFO(fmt, ...) \
    pr_info("swarm: " fmt "\n", ##__VA_ARGS__)

#define SWARM_ERROR(fmt, ...) \
    pr_err("swarm: " fmt "\n", ##__VA_ARGS__)

/* Timing helpers for performance measurement */
static inline u64 swarm_get_time_ns(void)
{
    return ktime_get_ns();
}

static inline u64 swarm_time_diff_ns(u64 start)
{
    return swarm_get_time_ns() - start;
}

#endif /* _SWARM_CORE_H */