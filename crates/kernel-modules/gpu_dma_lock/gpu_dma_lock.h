/*
 * GPU DMA Lock Kernel Module Header
 * 
 * Provides GPU memory protection, DMA access control, and CUDA interception
 * for the StratoSwarm orchestration platform
 */

#ifndef _GPU_DMA_LOCK_H
#define _GPU_DMA_LOCK_H

#include <linux/types.h>
#include <linux/ioctl.h>
#include <linux/list.h>
#include <linux/spinlock.h>
#include <linux/rbtree.h>
#include "../../common/swarm_core.h"
#include "../../common/swarm_ioctl.h"

/* Module configuration */
#define GPU_DMA_LOCK_VERSION "2.0"
#define MAX_GPU_DEVICES 8
#define MAX_ALLOCATIONS_PER_AGENT 1000
#define DEFAULT_AGENT_QUOTA (256 * 1024 * 1024) /* 256MB */

/* DMA permission flags */
#define SWARM_DMA_READ  0x01
#define SWARM_DMA_WRITE 0x02
#define SWARM_DMA_EXEC  0x04

/* GPU allocation flags */
#define SWARM_GPU_FLAG_PINNED     0x01
#define SWARM_GPU_FLAG_RDMA       0x02
#define SWARM_GPU_FLAG_PERSISTENT 0x04

/* GPU allocation tracking */
struct gpu_allocation {
    u64 id;
    u64 agent_id;
    u64 gpu_addr;
    size_t size;
    u32 device_id;
    u32 flags;
    ktime_t allocated_at;
    struct list_head agent_list;
    struct rb_node rb_node;
};

/* Per-agent GPU state */
struct gpu_agent_state {
    u64 agent_id;
    u64 memory_limit;
    u64 memory_used;
    u32 device_mask;
    u32 allocation_count;
    struct list_head allocations;
    spinlock_t lock;
    struct list_head list;
};

/* GPU device information */
struct swarm_gpu_device_info {
    u32 device_id;
    u64 total_memory;
    u64 free_memory;
    u32 compute_capability;
    char name[64];
};

/* GPU context for isolation */
struct swarm_gpu_context {
    u64 agent_id;
    u32 device_mask;
    void *cuda_context;
    struct list_head list;
};

/* DMA permission entry */
struct dma_permission {
    u64 agent_id;
    u64 dma_addr;
    size_t size;
    u32 permissions;
    struct rb_node rb_node;
};

/* CUDA interception hooks */
struct cuda_hooks {
    void *(*alloc_hook)(size_t size, unsigned int flags);
    int (*free_hook)(void *ptr);
    int (*memcpy_hook)(void *dst, const void *src, size_t size);
};

/* GPUDirect RDMA info */
struct swarm_gpudirect_info {
    u64 dma_addr;
    size_t size;
    u32 device_id;
    void *rdma_handle;
};

/* Module statistics */
struct gpu_module_stats {
    atomic64_t total_allocations;
    atomic64_t total_deallocations;
    atomic64_t total_bytes_allocated;
    atomic64_t total_bytes_freed;
    atomic64_t dma_checks_performed;
    atomic64_t dma_violations;
    atomic64_t quota_violations;
    atomic64_t cuda_intercepts;
};

/* Global module state */
struct gpu_dma_lock_state {
    struct gpu_module_stats stats;
    struct list_head agents;
    struct rb_root allocations_tree;
    struct rb_root dma_permissions_tree;
    struct list_head contexts;
    spinlock_t agents_lock;
    spinlock_t allocations_lock;
    spinlock_t permissions_lock;
    spinlock_t contexts_lock;
    struct cuda_hooks *cuda_hooks;
    bool gpudirect_enabled;
};

/* Function prototypes */

/* Core allocation functions */
u64 swarm_gpu_allocate(u64 agent_id, size_t size, u32 flags);
u64 swarm_gpu_allocate_on_device(u64 agent_id, size_t size, u32 device_id);
u64 swarm_gpu_allocate_rdma(u64 agent_id, size_t size, u32 device_id);
int swarm_gpu_free(u64 alloc_id);
int swarm_gpu_get_allocation_info(u64 alloc_id, struct swarm_gpu_allocation_info *info);

/* Quota management */
int swarm_gpu_set_quota(struct swarm_gpu_quota *quota);
int swarm_gpu_get_quota(u64 agent_id, struct swarm_gpu_quota *quota);

/* DMA permission management */
int swarm_dma_grant_permission(struct swarm_dma_permission_grant *grant);
int swarm_dma_revoke_permission(u64 agent_id, u64 dma_addr);
int swarm_dma_check_permission(struct swarm_dma_check *check);

/* CUDA interception */
int swarm_cuda_register_hooks(void);
void swarm_cuda_unregister_hooks(void);
void *swarm_cuda_intercept_alloc(u64 agent_id, size_t size, u32 device_id);
int swarm_cuda_intercept_free(void *ptr);
int swarm_cuda_verify_allocation(void *ptr);

/* GPU context management */
struct swarm_gpu_context *swarm_gpu_create_context(u64 agent_id);
void swarm_gpu_destroy_context(struct swarm_gpu_context *ctx);
int swarm_gpu_set_context_affinity(struct swarm_gpu_context *ctx, u32 device_mask);

/* Device information */
int swarm_gpu_get_device_count(void);
int swarm_gpu_get_device_info(u32 device_id, struct swarm_gpu_device_info *info);

/* Statistics */
int swarm_gpu_query_stats(struct swarm_gpu_stats *stats);
void swarm_gpu_reset_stats(void);

/* GPUDirect RDMA */
int swarm_gpu_check_gpudirect_support(void);
int swarm_gpu_get_rdma_info(u64 alloc_id, struct swarm_gpudirect_info *info);

/* ioctl handler */
long swarm_gpu_ioctl(struct file *file, unsigned int cmd, unsigned long arg);

/* Allocation info structure */
struct swarm_gpu_allocation_info {
    u64 id;
    u64 agent_id;
    u64 gpu_addr;
    size_t size;
    u32 device_id;
    u32 flags;
    u64 allocated_ns;
};

/* Module initialization/cleanup */
int gpu_dma_lock_init(void);
void gpu_dma_lock_exit(void);

#endif /* _GPU_DMA_LOCK_H */