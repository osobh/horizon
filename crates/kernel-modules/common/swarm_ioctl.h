/*
 * StratoSwarm IOCTL Interface
 * 
 * Unified ioctl interface for all StratoSwarm kernel modules
 */

#ifndef _SWARM_IOCTL_H
#define _SWARM_IOCTL_H

#include <linux/ioctl.h>
#include <linux/types.h>

/* IOCTL magic number */
#define SWARM_IOC_MAGIC 'S'

/* GPU DMA Lock IOCTLs */
#define SWARM_GPU_ALLOC      _IOWR(SWARM_IOC_MAGIC, 1, struct swarm_gpu_alloc_params)
#define SWARM_GPU_FREE       _IOW(SWARM_IOC_MAGIC, 2, u64)
#define SWARM_GPU_QUERY      _IOR(SWARM_IOC_MAGIC, 3, struct swarm_gpu_stats)
#define SWARM_GPU_SET_QUOTA  _IOW(SWARM_IOC_MAGIC, 4, struct swarm_gpu_quota)
#define SWARM_DMA_CHECK_PERM _IOWR(SWARM_IOC_MAGIC, 5, struct swarm_dma_check)
#define SWARM_DMA_GRANT_PERM _IOW(SWARM_IOC_MAGIC, 6, struct swarm_dma_permission_grant)

/* SwarmGuard IOCTLs */
#define SWARM_AGENT_CREATE   _IOWR(SWARM_IOC_MAGIC, 10, struct swarm_agent_config)
#define SWARM_AGENT_DESTROY  _IOW(SWARM_IOC_MAGIC, 11, u64)
#define SWARM_AGENT_QUERY    _IOWR(SWARM_IOC_MAGIC, 12, struct swarm_agent_query)
#define SWARM_AGENT_SET_NS   _IOW(SWARM_IOC_MAGIC, 13, struct swarm_namespace_config)

/* Mount Isolation IOCTLs */
#define SWARM_MOUNT_SETUP    _IOW(SWARM_IOC_MAGIC, 14, struct swarm_mount_ioctl)
#define SWARM_MOUNT_TEARDOWN _IOW(SWARM_IOC_MAGIC, 15, u64)
#define SWARM_MOUNT_ADD_BIND _IOW(SWARM_IOC_MAGIC, 16, struct swarm_bind_mount_ioctl)
#define SWARM_MOUNT_OVERLAYFS _IOW(SWARM_IOC_MAGIC, 17, struct swarm_overlayfs_ioctl)

/* TierWatch IOCTLs */
#define SWARM_TIER_STATS     _IOR(SWARM_IOC_MAGIC, 20, struct swarm_tier_stats)
#define SWARM_TIER_PRESSURE  _IOR(SWARM_IOC_MAGIC, 21, struct swarm_memory_pressure)
#define SWARM_TIER_MIGRATE   _IOW(SWARM_IOC_MAGIC, 22, struct swarm_migration_request)

/* Structure definitions for IOCTLs */

/* GPU allocation parameters */
struct swarm_gpu_alloc_params {
    u64 agent_id;
    size_t size;
    u32 device_id;
    u32 flags;
    u64 alloc_id; /* Output: allocation ID */
};

/* GPU statistics */
struct swarm_gpu_stats {
    u64 total_memory;
    u64 used_memory;
    u64 allocation_count;
    u64 dma_checks;
    u64 quota_violations;
};

/* GPU quota configuration */
struct swarm_gpu_quota {
    u64 agent_id;
    u64 memory_limit;
    u32 device_mask; /* Bitmap of allowed devices */
};

/* DMA permission check */
struct swarm_dma_check {
    u64 agent_id;
    u64 dma_addr;
    size_t size;
    u32 access_type; /* Read/Write */
    u32 allowed;     /* Output: 1 if allowed, 0 if denied */
};

/* DMA permission grant */
struct swarm_dma_permission_grant {
    u64 agent_id;
    u64 dma_addr;
    size_t size;
    u32 permissions;
};

/* Agent configuration */
struct swarm_agent_config {
    char name[64];
    u64 memory_limit;
    u32 cpu_quota;
    u64 gpu_memory_limit;
    u32 namespace_flags;
    u64 agent_id; /* Output: created agent ID */
};

/* Agent query */
struct swarm_agent_query {
    u64 agent_id;
    /* Output fields */
    pid_t pid;
    u64 memory_usage;
    u64 gpu_memory_usage;
    u32 violation_count;
    u32 state;
};

/* Namespace configuration */
struct swarm_namespace_config {
    u64 agent_id;
    u32 namespace_flags;
};

/* Tier statistics */
struct swarm_tier_stats {
    enum swarm_memory_tier tier;
    u64 total_pages;
    u64 used_pages;
    u64 hot_pages;
    u64 cold_pages;
    u64 migrations_in;
    u64 migrations_out;
};

/* Memory pressure info */
struct swarm_memory_pressure {
    enum swarm_memory_tier tier;
    u32 pressure_level; /* 0-100 */
    u64 free_pages;
    u64 reclaimable_pages;
};

/* Migration request */
struct swarm_migration_request {
    u64 agent_id;
    enum swarm_memory_tier from_tier;
    enum swarm_memory_tier to_tier;
    size_t page_count;
};

/* Mount isolation structures */
#define SWARM_MOUNT_MAX_PATH 256
#define SWARM_MOUNT_MAX_BINDS 16

/* Mount flags */
#define SWARM_MNT_READONLY  0x01
#define SWARM_MNT_NOSUID    0x02
#define SWARM_MNT_NOEXEC    0x04
#define SWARM_MNT_NODEV     0x08

/* Bind mount configuration for ioctl */
struct swarm_bind_mount_ioctl {
    u64 agent_id;
    char source[SWARM_MOUNT_MAX_PATH];
    char target[SWARM_MOUNT_MAX_PATH];
    u32 flags;
};

/* OverlayFS configuration for ioctl */
struct swarm_overlayfs_ioctl {
    u64 agent_id;
    char lower_dir[SWARM_MOUNT_MAX_PATH];
    char upper_dir[SWARM_MOUNT_MAX_PATH];
    char work_dir[SWARM_MOUNT_MAX_PATH];
    char merged_dir[SWARM_MOUNT_MAX_PATH];
};

/* Complete mount isolation setup for ioctl */
struct swarm_mount_ioctl {
    u64 agent_id;
    /* OverlayFS config */
    char lower_dir[SWARM_MOUNT_MAX_PATH];
    char upper_dir[SWARM_MOUNT_MAX_PATH];
    char work_dir[SWARM_MOUNT_MAX_PATH];
    char merged_dir[SWARM_MOUNT_MAX_PATH];
    /* Bind mounts */
    struct {
        char source[SWARM_MOUNT_MAX_PATH];
        char target[SWARM_MOUNT_MAX_PATH];
        u32 flags;
    } bind_mounts[SWARM_MOUNT_MAX_BINDS];
    u32 num_bind_mounts;
    /* Old root for pivot_root cleanup */
    char old_root[SWARM_MOUNT_MAX_PATH];
    /* Flags */
    u32 use_overlayfs;
    u32 private_mounts;
};

#endif /* _SWARM_IOCTL_H */