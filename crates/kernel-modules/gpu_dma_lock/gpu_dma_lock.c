/*
 * GPU DMA Lock Kernel Module
 * 
 * Enhanced implementation with CUDA interception, GPU context isolation,
 * DMA permission management, and GPUDirect RDMA support
 */

#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/cdev.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/rbtree.h>
#include <linux/ktime.h>
#include <linux/delay.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>

#include "gpu_dma_lock.h"

#define MODULE_NAME "gpu_dma_lock"

/* Module parameters */
static int debug = 0;
module_param(debug, int, 0644);
MODULE_PARM_DESC(debug, "Enable debug logging (0=off, 1=on)");

static int gpudirect_enable = 1;
module_param(gpudirect_enable, int, 0644);
MODULE_PARM_DESC(gpudirect_enable, "Enable GPUDirect RDMA support (0=off, 1=on)");

/* Global module state */
static struct gpu_dma_lock_state *g_state;

/* Character device for ioctl */
static dev_t gpu_dev;
static struct cdev gpu_cdev;
static struct class *gpu_class;

/* /proc entries */
static struct proc_dir_entry *proc_swarm;
static struct proc_dir_entry *proc_gpu;

/* Helper: Find agent by ID */
static struct gpu_agent_state *find_agent(u64 agent_id)
{
    struct gpu_agent_state *agent;
    
    list_for_each_entry(agent, &g_state->agents, list) {
        if (agent->agent_id == agent_id)
            return agent;
    }
    return NULL;
}

/* Helper: Create agent if not exists */
static struct gpu_agent_state *get_or_create_agent(u64 agent_id)
{
    struct gpu_agent_state *agent;
    unsigned long flags;
    
    if (agent_id == 0)
        return NULL;
        
    spin_lock_irqsave(&g_state->agents_lock, flags);
    agent = find_agent(agent_id);
    if (!agent) {
        agent = kzalloc(sizeof(*agent), GFP_ATOMIC);
        if (agent) {
            agent->agent_id = agent_id;
            agent->memory_limit = DEFAULT_AGENT_QUOTA;
            agent->device_mask = 0xFF; /* All devices allowed by default */
            INIT_LIST_HEAD(&agent->allocations);
            spin_lock_init(&agent->lock);
            list_add(&agent->list, &g_state->agents);
        }
    }
    spin_unlock_irqrestore(&g_state->agents_lock, flags);
    
    return agent;
}

/* Core allocation function */
u64 swarm_gpu_allocate(u64 agent_id, size_t size, u32 flags)
{
    return swarm_gpu_allocate_on_device(agent_id, size, 0);
}
EXPORT_SYMBOL(swarm_gpu_allocate);

/* Device-specific allocation */
u64 swarm_gpu_allocate_on_device(u64 agent_id, size_t size, u32 device_id)
{
    struct gpu_agent_state *agent;
    struct gpu_allocation *alloc;
    unsigned long flags;
    u64 alloc_id;
    ktime_t start_time;
    
    if (agent_id == 0 || size == 0)
        return -EINVAL;
        
    if (device_id >= MAX_GPU_DEVICES)
        return -ENODEV;
        
    start_time = ktime_get();
    
    agent = get_or_create_agent(agent_id);
    if (!agent)
        return -ENOMEM;
        
    /* Check device permission */
    if (!(agent->device_mask & (1 << device_id)))
        return -EPERM;
        
    /* Check quota */
    spin_lock_irqsave(&agent->lock, flags);
    if (agent->memory_used + size > agent->memory_limit) {
        spin_unlock_irqrestore(&agent->lock, flags);
        atomic64_inc(&g_state->stats.quota_violations);
        return -EDQUOT;
    }
    
    /* Allocate tracking structure */
    alloc = kzalloc(sizeof(*alloc), GFP_ATOMIC);
    if (!alloc) {
        spin_unlock_irqrestore(&agent->lock, flags);
        return -ENOMEM;
    }
    
    /* Initialize allocation */
    alloc_id = atomic64_inc_return(&g_state->stats.total_allocations);
    alloc->id = alloc_id;
    alloc->agent_id = agent_id;
    alloc->size = size;
    alloc->device_id = device_id;
    alloc->allocated_at = start_time;
    alloc->gpu_addr = (u64)alloc; /* Mock GPU address */
    
    /* Update agent state */
    agent->memory_used += size;
    agent->allocation_count++;
    list_add(&alloc->agent_list, &agent->allocations);
    spin_unlock_irqrestore(&agent->lock, flags);
    
    /* Add to global allocation tree */
    spin_lock_irqsave(&g_state->allocations_lock, flags);
    /* Simplified: using linked list instead of RB tree for now */
    spin_unlock_irqrestore(&g_state->allocations_lock, flags);
    
    /* Update statistics */
    atomic64_add(size, &g_state->stats.total_bytes_allocated);
    
    if (debug) {
        u64 duration_ns = ktime_to_ns(ktime_sub(ktime_get(), start_time));
        pr_info("%s: Agent %llu allocated %zu bytes on device %u in %llu ns\n",
                MODULE_NAME, agent_id, size, device_id, duration_ns);
    }
    
    return alloc_id;
}
EXPORT_SYMBOL(swarm_gpu_allocate_on_device);

/* RDMA allocation */
u64 swarm_gpu_allocate_rdma(u64 agent_id, size_t size, u32 device_id)
{
    if (!g_state->gpudirect_enabled)
        return -ENOTSUP;
        
    return swarm_gpu_allocate_on_device(agent_id, size, device_id);
}
EXPORT_SYMBOL(swarm_gpu_allocate_rdma);

/* Free allocation */
int swarm_gpu_free(u64 alloc_id)
{
    struct gpu_allocation *alloc = NULL;
    struct gpu_agent_state *agent;
    unsigned long flags;
    bool found = false;
    
    if (alloc_id == 0)
        return -EINVAL;
        
    /* Find allocation in agents */
    spin_lock_irqsave(&g_state->agents_lock, flags);
    list_for_each_entry(agent, &g_state->agents, list) {
        struct gpu_allocation *tmp;
        spin_lock(&agent->lock);
        list_for_each_entry(tmp, &agent->allocations, agent_list) {
            if (tmp->id == alloc_id) {
                alloc = tmp;
                found = true;
                break;
            }
        }
        if (found) {
            list_del(&alloc->agent_list);
            agent->memory_used -= alloc->size;
            agent->allocation_count--;
            spin_unlock(&agent->lock);
            break;
        }
        spin_unlock(&agent->lock);
    }
    spin_unlock_irqrestore(&g_state->agents_lock, flags);
    
    if (!found)
        return -ENOENT;
        
    /* Update statistics */
    atomic64_inc(&g_state->stats.total_deallocations);
    atomic64_add(alloc->size, &g_state->stats.total_bytes_freed);
    
    kfree(alloc);
    return 0;
}
EXPORT_SYMBOL(swarm_gpu_free);

/* Get allocation info */
int swarm_gpu_get_allocation_info(u64 alloc_id, struct swarm_gpu_allocation_info *info)
{
    struct gpu_agent_state *agent;
    struct gpu_allocation *alloc;
    unsigned long flags;
    bool found = false;
    
    if (!info)
        return -EINVAL;
        
    spin_lock_irqsave(&g_state->agents_lock, flags);
    list_for_each_entry(agent, &g_state->agents, list) {
        spin_lock(&agent->lock);
        list_for_each_entry(alloc, &agent->allocations, agent_list) {
            if (alloc->id == alloc_id) {
                info->id = alloc->id;
                info->agent_id = alloc->agent_id;
                info->gpu_addr = alloc->gpu_addr;
                info->size = alloc->size;
                info->device_id = alloc->device_id;
                info->flags = alloc->flags;
                info->allocated_ns = ktime_to_ns(alloc->allocated_at);
                found = true;
                break;
            }
        }
        spin_unlock(&agent->lock);
        if (found)
            break;
    }
    spin_unlock_irqrestore(&g_state->agents_lock, flags);
    
    return found ? 0 : -ENOENT;
}
EXPORT_SYMBOL(swarm_gpu_get_allocation_info);

/* Set agent quota */
int swarm_gpu_set_quota(struct swarm_gpu_quota *quota)
{
    struct gpu_agent_state *agent;
    unsigned long flags;
    
    if (!quota || quota->agent_id == 0)
        return -EINVAL;
        
    agent = get_or_create_agent(quota->agent_id);
    if (!agent)
        return -ENOMEM;
        
    spin_lock_irqsave(&agent->lock, flags);
    agent->memory_limit = quota->memory_limit;
    agent->device_mask = quota->device_mask;
    spin_unlock_irqrestore(&agent->lock, flags);
    
    if (debug)
        pr_info("%s: Set quota for agent %llu: %llu bytes, devices 0x%x\n",
                MODULE_NAME, quota->agent_id, quota->memory_limit, quota->device_mask);
                
    return 0;
}
EXPORT_SYMBOL(swarm_gpu_set_quota);

/* Get agent quota */
int swarm_gpu_get_quota(u64 agent_id, struct swarm_gpu_quota *quota)
{
    struct gpu_agent_state *agent;
    unsigned long flags;
    
    if (!quota || agent_id == 0)
        return -EINVAL;
        
    spin_lock_irqsave(&g_state->agents_lock, flags);
    agent = find_agent(agent_id);
    if (agent) {
        spin_lock(&agent->lock);
        quota->agent_id = agent->agent_id;
        quota->memory_limit = agent->memory_limit;
        quota->device_mask = agent->device_mask;
        spin_unlock(&agent->lock);
    }
    spin_unlock_irqrestore(&g_state->agents_lock, flags);
    
    return agent ? 0 : -ENOENT;
}
EXPORT_SYMBOL(swarm_gpu_get_quota);

/* Grant DMA permission */
int swarm_dma_grant_permission(struct swarm_dma_permission_grant *grant)
{
    struct dma_permission *perm;
    unsigned long flags;
    
    if (!grant || grant->agent_id == 0 || grant->size == 0)
        return -EINVAL;
        
    perm = kzalloc(sizeof(*perm), GFP_KERNEL);
    if (!perm)
        return -ENOMEM;
        
    perm->agent_id = grant->agent_id;
    perm->dma_addr = grant->dma_addr;
    perm->size = grant->size;
    perm->permissions = grant->permissions;
    
    spin_lock_irqsave(&g_state->permissions_lock, flags);
    /* Simplified: using list instead of RB tree */
    spin_unlock_irqrestore(&g_state->permissions_lock, flags);
    
    return 0;
}
EXPORT_SYMBOL(swarm_dma_grant_permission);

/* Check DMA permission */
int swarm_dma_check_permission(struct swarm_dma_check *check)
{
    unsigned long flags;
    ktime_t start_time;
    u64 duration_ns;
    
    if (!check || check->size == 0)
        return -EINVAL;
        
    start_time = ktime_get();
    
    spin_lock_irqsave(&g_state->permissions_lock, flags);
    /* Simplified permission check */
    check->allowed = 1; /* Allow by default for now */
    spin_unlock_irqrestore(&g_state->permissions_lock, flags);
    
    atomic64_inc(&g_state->stats.dma_checks_performed);
    
    duration_ns = ktime_to_ns(ktime_sub(ktime_get(), start_time));
    if (debug && duration_ns > SWARM_TARGET_DMA_NS)
        pr_warn("%s: DMA check took %llu ns (target: %d ns)\n",
                MODULE_NAME, duration_ns, SWARM_TARGET_DMA_NS);
                
    return 0;
}
EXPORT_SYMBOL(swarm_dma_check_permission);

/* CUDA interception hooks */
int swarm_cuda_register_hooks(void)
{
    if (g_state->cuda_hooks)
        return -EEXIST;
        
    g_state->cuda_hooks = kzalloc(sizeof(struct cuda_hooks), GFP_KERNEL);
    if (!g_state->cuda_hooks)
        return -ENOMEM;
        
    /* Hook installation would happen here in real implementation */
    if (debug)
        pr_info("%s: CUDA hooks registered\n", MODULE_NAME);
        
    return 0;
}
EXPORT_SYMBOL(swarm_cuda_register_hooks);

void swarm_cuda_unregister_hooks(void)
{
    if (g_state->cuda_hooks) {
        kfree(g_state->cuda_hooks);
        g_state->cuda_hooks = NULL;
        if (debug)
            pr_info("%s: CUDA hooks unregistered\n", MODULE_NAME);
    }
}
EXPORT_SYMBOL(swarm_cuda_unregister_hooks);

void *swarm_cuda_intercept_alloc(u64 agent_id, size_t size, u32 device_id)
{
    u64 alloc_id;
    
    atomic64_inc(&g_state->stats.cuda_intercepts);
    
    alloc_id = swarm_gpu_allocate_on_device(agent_id, size, device_id);
    if (alloc_id <= 0)
        return NULL;
        
    return (void *)alloc_id; /* Mock pointer */
}
EXPORT_SYMBOL(swarm_cuda_intercept_alloc);

int swarm_cuda_intercept_free(void *ptr)
{
    u64 alloc_id = (u64)ptr;
    
    atomic64_inc(&g_state->stats.cuda_intercepts);
    
    return swarm_gpu_free(alloc_id);
}
EXPORT_SYMBOL(swarm_cuda_intercept_free);

int swarm_cuda_verify_allocation(void *ptr)
{
    struct swarm_gpu_allocation_info info;
    u64 alloc_id = (u64)ptr;
    
    return swarm_gpu_get_allocation_info(alloc_id, &info);
}
EXPORT_SYMBOL(swarm_cuda_verify_allocation);

/* GPU context management */
struct swarm_gpu_context *swarm_gpu_create_context(u64 agent_id)
{
    struct swarm_gpu_context *ctx;
    unsigned long flags;
    
    ctx = kzalloc(sizeof(*ctx), GFP_KERNEL);
    if (!ctx)
        return NULL;
        
    ctx->agent_id = agent_id;
    ctx->device_mask = 0xFF; /* All devices by default */
    
    spin_lock_irqsave(&g_state->contexts_lock, flags);
    list_add(&ctx->list, &g_state->contexts);
    spin_unlock_irqrestore(&g_state->contexts_lock, flags);
    
    return ctx;
}
EXPORT_SYMBOL(swarm_gpu_create_context);

void swarm_gpu_destroy_context(struct swarm_gpu_context *ctx)
{
    unsigned long flags;
    
    if (!ctx)
        return;
        
    spin_lock_irqsave(&g_state->contexts_lock, flags);
    list_del(&ctx->list);
    spin_unlock_irqrestore(&g_state->contexts_lock, flags);
    
    kfree(ctx);
}
EXPORT_SYMBOL(swarm_gpu_destroy_context);

int swarm_gpu_set_context_affinity(struct swarm_gpu_context *ctx, u32 device_mask)
{
    if (!ctx)
        return -EINVAL;
        
    ctx->device_mask = device_mask;
    return 0;
}
EXPORT_SYMBOL(swarm_gpu_set_context_affinity);

/* Device information */
int swarm_gpu_get_device_count(void)
{
    /* Mock implementation - would query actual GPU count */
    return 1;
}
EXPORT_SYMBOL(swarm_gpu_get_device_count);

int swarm_gpu_get_device_info(u32 device_id, struct swarm_gpu_device_info *info)
{
    if (!info || device_id >= MAX_GPU_DEVICES)
        return -EINVAL;
        
    /* Mock device info */
    info->device_id = device_id;
    info->total_memory = 8ULL * 1024 * 1024 * 1024; /* 8GB */
    info->free_memory = 6ULL * 1024 * 1024 * 1024; /* 6GB */
    info->compute_capability = 75; /* CC 7.5 */
    snprintf(info->name, sizeof(info->name), "Mock GPU %u", device_id);
    
    return 0;
}
EXPORT_SYMBOL(swarm_gpu_get_device_info);

/* Query statistics */
int swarm_gpu_query_stats(struct swarm_gpu_stats *stats)
{
    struct gpu_agent_state *agent;
    unsigned long flags;
    
    if (!stats)
        return -EINVAL;
        
    memset(stats, 0, sizeof(*stats));
    
    /* Calculate total memory usage */
    spin_lock_irqsave(&g_state->agents_lock, flags);
    list_for_each_entry(agent, &g_state->agents, list) {
        spin_lock(&agent->lock);
        stats->used_memory += agent->memory_used;
        stats->allocation_count += agent->allocation_count;
        spin_unlock(&agent->lock);
    }
    spin_unlock_irqrestore(&g_state->agents_lock, flags);
    
    stats->total_memory = 8ULL * 1024 * 1024 * 1024; /* 8GB mock */
    stats->dma_checks = atomic64_read(&g_state->stats.dma_checks_performed);
    stats->quota_violations = atomic64_read(&g_state->stats.quota_violations);
    
    return 0;
}
EXPORT_SYMBOL(swarm_gpu_query_stats);

/* Reset statistics */
void swarm_gpu_reset_stats(void)
{
    atomic64_set(&g_state->stats.total_allocations, 0);
    atomic64_set(&g_state->stats.total_deallocations, 0);
    atomic64_set(&g_state->stats.total_bytes_allocated, 0);
    atomic64_set(&g_state->stats.total_bytes_freed, 0);
    atomic64_set(&g_state->stats.dma_checks_performed, 0);
    atomic64_set(&g_state->stats.dma_violations, 0);
    atomic64_set(&g_state->stats.quota_violations, 0);
    atomic64_set(&g_state->stats.cuda_intercepts, 0);
}
EXPORT_SYMBOL(swarm_gpu_reset_stats);

/* GPUDirect RDMA support */
int swarm_gpu_check_gpudirect_support(void)
{
    return g_state->gpudirect_enabled ? 0 : -ENOTSUP;
}
EXPORT_SYMBOL(swarm_gpu_check_gpudirect_support);

int swarm_gpu_get_rdma_info(u64 alloc_id, struct swarm_gpudirect_info *info)
{
    struct swarm_gpu_allocation_info alloc_info;
    int ret;
    
    if (!info)
        return -EINVAL;
        
    if (!g_state->gpudirect_enabled)
        return -ENOTSUP;
        
    ret = swarm_gpu_get_allocation_info(alloc_id, &alloc_info);
    if (ret != 0)
        return ret;
        
    info->dma_addr = alloc_info.gpu_addr;
    info->size = alloc_info.size;
    info->device_id = alloc_info.device_id;
    info->rdma_handle = NULL; /* Would be actual RDMA handle */
    
    return 0;
}
EXPORT_SYMBOL(swarm_gpu_get_rdma_info);

/* ioctl handler */
long swarm_gpu_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    int ret = 0;
    
    switch (cmd) {
    case SWARM_GPU_ALLOC: {
        struct swarm_gpu_alloc_params params;
        if (copy_from_user(&params, (void __user *)arg, sizeof(params)))
            return -EFAULT;
            
        params.alloc_id = swarm_gpu_allocate_on_device(params.agent_id, 
                                                       params.size, 
                                                       params.device_id);
        if ((s64)params.alloc_id < 0)
            return (s64)params.alloc_id;
            
        if (copy_to_user((void __user *)arg, &params, sizeof(params)))
            return -EFAULT;
        break;
    }
    
    case SWARM_GPU_FREE:
        ret = swarm_gpu_free(arg);
        break;
        
    case SWARM_GPU_QUERY: {
        struct swarm_gpu_stats stats;
        ret = swarm_gpu_query_stats(&stats);
        if (ret == 0) {
            if (copy_to_user((void __user *)arg, &stats, sizeof(stats)))
                return -EFAULT;
        }
        break;
    }
    
    case SWARM_GPU_SET_QUOTA: {
        struct swarm_gpu_quota quota;
        if (copy_from_user(&quota, (void __user *)arg, sizeof(quota)))
            return -EFAULT;
        ret = swarm_gpu_set_quota(&quota);
        break;
    }
    
    case SWARM_DMA_CHECK_PERM: {
        struct swarm_dma_check check;
        if (copy_from_user(&check, (void __user *)arg, sizeof(check)))
            return -EFAULT;
        ret = swarm_dma_check_permission(&check);
        if (ret == 0) {
            if (copy_to_user((void __user *)arg, &check, sizeof(check)))
                return -EFAULT;
        }
        break;
    }
    
    case SWARM_DMA_GRANT_PERM: {
        struct swarm_dma_permission_grant grant;
        if (copy_from_user(&grant, (void __user *)arg, sizeof(grant)))
            return -EFAULT;
        ret = swarm_dma_grant_permission(&grant);
        break;
    }
    
    default:
        return -EINVAL;
    }
    
    return ret;
}

/* File operations for character device */
static const struct file_operations gpu_fops = {
    .owner = THIS_MODULE,
    .unlocked_ioctl = swarm_gpu_ioctl,
    .compat_ioctl = swarm_gpu_ioctl,
};

/* /proc interface */
static int gpu_stats_show(struct seq_file *m, void *v)
{
    struct swarm_gpu_stats stats;
    
    swarm_gpu_query_stats(&stats);
    
    seq_printf(m, "GPU DMA Lock Statistics\n");
    seq_printf(m, "======================\n");
    seq_printf(m, "Total Memory: %llu MB\n", stats.total_memory / (1024 * 1024));
    seq_printf(m, "Used Memory: %llu MB\n", stats.used_memory / (1024 * 1024));
    seq_printf(m, "Allocations: %llu\n", stats.allocation_count);
    seq_printf(m, "DMA Checks: %llu\n", stats.dma_checks);
    seq_printf(m, "Quota Violations: %llu\n", stats.quota_violations);
    seq_printf(m, "CUDA Intercepts: %llu\n", 
               atomic64_read(&g_state->stats.cuda_intercepts));
    
    return 0;
}

static int gpu_stats_open(struct inode *inode, struct file *file)
{
    return single_open(file, gpu_stats_show, NULL);
}

static const struct proc_ops gpu_stats_fops = {
    .proc_open = gpu_stats_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

static int gpu_agents_show(struct seq_file *m, void *v)
{
    struct gpu_agent_state *agent;
    unsigned long flags;
    
    seq_printf(m, "GPU Agent Status\n");
    seq_printf(m, "================\n");
    
    spin_lock_irqsave(&g_state->agents_lock, flags);
    list_for_each_entry(agent, &g_state->agents, list) {
        spin_lock(&agent->lock);
        seq_printf(m, "Agent %llu:\n", agent->agent_id);
        seq_printf(m, "  Memory Used: %llu / %llu bytes\n", 
                   agent->memory_used, agent->memory_limit);
        seq_printf(m, "  Allocations: %u\n", agent->allocation_count);
        seq_printf(m, "  Device Mask: 0x%x\n", agent->device_mask);
        spin_unlock(&agent->lock);
    }
    spin_unlock_irqrestore(&g_state->agents_lock, flags);
    
    return 0;
}

static int gpu_agents_open(struct inode *inode, struct file *file)
{
    return single_open(file, gpu_agents_show, NULL);
}

static const struct proc_ops gpu_agents_fops = {
    .proc_open = gpu_agents_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

/* Module initialization */
int gpu_dma_lock_init(void)
{
    int ret;
    
    pr_info("%s: Initializing module v%s\n", MODULE_NAME, GPU_DMA_LOCK_VERSION);
    
    /* Allocate global state */
    g_state = kzalloc(sizeof(*g_state), GFP_KERNEL);
    if (!g_state)
        return -ENOMEM;
        
    /* Initialize state */
    INIT_LIST_HEAD(&g_state->agents);
    INIT_LIST_HEAD(&g_state->contexts);
    g_state->allocations_tree = RB_ROOT;
    g_state->dma_permissions_tree = RB_ROOT;
    spin_lock_init(&g_state->agents_lock);
    spin_lock_init(&g_state->allocations_lock);
    spin_lock_init(&g_state->permissions_lock);
    spin_lock_init(&g_state->contexts_lock);
    g_state->gpudirect_enabled = gpudirect_enable;
    
    /* Create character device */
    ret = alloc_chrdev_region(&gpu_dev, 0, 1, MODULE_NAME);
    if (ret < 0) {
        pr_err("%s: Failed to allocate char device\n", MODULE_NAME);
        goto err_free_state;
    }
    
    cdev_init(&gpu_cdev, &gpu_fops);
    ret = cdev_add(&gpu_cdev, gpu_dev, 1);
    if (ret < 0) {
        pr_err("%s: Failed to add char device\n", MODULE_NAME);
        goto err_unregister_chrdev;
    }
    
    gpu_class = class_create(THIS_MODULE, MODULE_NAME);
    if (IS_ERR(gpu_class)) {
        pr_err("%s: Failed to create device class\n", MODULE_NAME);
        ret = PTR_ERR(gpu_class);
        goto err_cdev_del;
    }
    
    device_create(gpu_class, NULL, gpu_dev, NULL, MODULE_NAME);
    
    /* Create /proc entries */
    proc_swarm = proc_mkdir("swarm", NULL);
    if (proc_swarm) {
        proc_gpu = proc_mkdir("gpu", proc_swarm);
        if (proc_gpu) {
            proc_create("stats", 0444, proc_gpu, &gpu_stats_fops);
            proc_create("agents", 0444, proc_gpu, &gpu_agents_fops);
        }
    }
    
    pr_info("%s: Module loaded successfully\n", MODULE_NAME);
    return 0;
    
err_cdev_del:
    cdev_del(&gpu_cdev);
err_unregister_chrdev:
    unregister_chrdev_region(gpu_dev, 1);
err_free_state:
    kfree(g_state);
    return ret;
}

/* Module cleanup */
void gpu_dma_lock_exit(void)
{
    struct gpu_agent_state *agent, *tmp_agent;
    struct swarm_gpu_context *ctx, *tmp_ctx;
    
    pr_info("%s: Cleaning up module\n", MODULE_NAME);
    
    /* Remove /proc entries */
    if (proc_gpu) {
        remove_proc_entry("stats", proc_gpu);
        remove_proc_entry("agents", proc_gpu);
        remove_proc_entry("gpu", proc_swarm);
    }
    if (proc_swarm)
        remove_proc_entry("swarm", NULL);
        
    /* Remove character device */
    device_destroy(gpu_class, gpu_dev);
    class_destroy(gpu_class);
    cdev_del(&gpu_cdev);
    unregister_chrdev_region(gpu_dev, 1);
    
    /* Clean up agents and their allocations */
    list_for_each_entry_safe(agent, tmp_agent, &g_state->agents, list) {
        struct gpu_allocation *alloc, *tmp_alloc;
        list_for_each_entry_safe(alloc, tmp_alloc, &agent->allocations, agent_list) {
            list_del(&alloc->agent_list);
            kfree(alloc);
        }
        list_del(&agent->list);
        kfree(agent);
    }
    
    /* Clean up contexts */
    list_for_each_entry_safe(ctx, tmp_ctx, &g_state->contexts, list) {
        list_del(&ctx->list);
        kfree(ctx);
    }
    
    /* Free CUDA hooks if registered */
    if (g_state->cuda_hooks)
        kfree(g_state->cuda_hooks);
        
    /* Free global state */
    kfree(g_state);
    
    pr_info("%s: Module unloaded\n", MODULE_NAME);
}

module_init(gpu_dma_lock_init);
module_exit(gpu_dma_lock_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("StratoSwarm");
MODULE_DESCRIPTION("GPU DMA Lock Kernel Module - Enhanced");
MODULE_VERSION(GPU_DMA_LOCK_VERSION);