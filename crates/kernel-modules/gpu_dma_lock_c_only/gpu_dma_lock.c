/*
 * Simplified GPU DMA Lock Kernel Module
 * This is a C-only version that implements the basic /proc interface
 */

#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/proc_fs.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/list.h>

#define MODULE_NAME "gpu_dma_lock"
#define PROC_DIR "swarm/gpu"

/* Module parameters */
static int debug = 0;
module_param(debug, int, 0644);
MODULE_PARM_DESC(debug, "Enable debug logging (0=off, 1=on)");

/* Proc filesystem entries */
static struct proc_dir_entry *proc_root;
static struct proc_dir_entry *proc_swarm;
static struct proc_dir_entry *proc_gpu;
static struct proc_dir_entry *proc_stats;
static struct proc_dir_entry *proc_quotas;
static struct proc_dir_entry *proc_allocations;
static struct proc_dir_entry *proc_dma_perms;
static struct proc_dir_entry *proc_control;

/* Forward declaration */
static void __exit gpu_dma_lock_cleanup(void);

/* Statistics */
static struct {
    unsigned long allocations;
    unsigned long deallocations;
    unsigned long dma_checks;
    unsigned long quota_hits;
    spinlock_t lock;
} stats;

/* Simple allocation tracking */
struct allocation {
    struct list_head list;
    unsigned long agent_id;
    unsigned long size;
    unsigned long address;
};

static LIST_HEAD(allocation_list);
static DEFINE_SPINLOCK(allocation_lock);

/* Proc read functions */
static ssize_t stats_read(struct file *file, char __user *buf, size_t count, loff_t *pos)
{
    char buffer[512];
    int len;
    
    if (*pos > 0)
        return 0;
    
    spin_lock(&stats.lock);
    len = snprintf(buffer, sizeof(buffer),
        "GPU DMA Lock Statistics\n"
        "======================\n"
        "Allocations: %lu\n"
        "Deallocations: %lu\n"
        "DMA Checks: %lu\n"
        "Quota Hits: %lu\n",
        stats.allocations,
        stats.deallocations,
        stats.dma_checks,
        stats.quota_hits);
    spin_unlock(&stats.lock);
    
    if (len > count)
        len = count;
    
    if (copy_to_user(buf, buffer, len))
        return -EFAULT;
    
    *pos += len;
    return len;
}

static ssize_t quotas_read(struct file *file, char __user *buf, size_t count, loff_t *pos)
{
    char buffer[256];
    int len;
    
    if (*pos > 0)
        return 0;
    
    len = snprintf(buffer, sizeof(buffer),
        "Agent Quotas\n"
        "============\n"
        "Agent 1: 1048576 bytes\n"
        "Agent 2: 2097152 bytes\n");
    
    if (len > count)
        len = count;
    
    if (copy_to_user(buf, buffer, len))
        return -EFAULT;
    
    *pos += len;
    return len;
}

static ssize_t allocations_read(struct file *file, char __user *buf, size_t count, loff_t *pos)
{
    char buffer[1024];
    int len = 0;
    struct allocation *alloc;
    
    if (*pos > 0)
        return 0;
    
    len += snprintf(buffer + len, sizeof(buffer) - len,
        "Current Allocations\n"
        "==================\n");
    
    spin_lock(&allocation_lock);
    list_for_each_entry(alloc, &allocation_list, list) {
        len += snprintf(buffer + len, sizeof(buffer) - len,
            "Agent %lu: %lu bytes at 0x%lx\n",
            alloc->agent_id, alloc->size, alloc->address);
    }
    spin_unlock(&allocation_lock);
    
    if (len > count)
        len = count;
    
    if (copy_to_user(buf, buffer, len))
        return -EFAULT;
    
    *pos += len;
    return len;
}

static ssize_t dma_perms_read(struct file *file, char __user *buf, size_t count, loff_t *pos)
{
    char buffer[256];
    int len;
    
    if (*pos > 0)
        return 0;
    
    len = snprintf(buffer, sizeof(buffer),
        "DMA Permissions\n"
        "===============\n"
        "1:0x10000:rw\n"
        "2:0x20000:r\n");
    
    if (len > count)
        len = count;
    
    if (copy_to_user(buf, buffer, len))
        return -EFAULT;
    
    *pos += len;
    return len;
}

static ssize_t dma_perms_write(struct file *file, const char __user *buf, size_t count, loff_t *pos)
{
    char buffer[128];
    
    if (count > sizeof(buffer) - 1)
        count = sizeof(buffer) - 1;
    
    if (copy_from_user(buffer, buf, count))
        return -EFAULT;
    
    buffer[count] = '\0';
    
    if (debug)
        printk(KERN_INFO "%s: DMA permission write: %s", MODULE_NAME, buffer);
    
    spin_lock(&stats.lock);
    stats.dma_checks++;
    spin_unlock(&stats.lock);
    
    return count;
}

static ssize_t control_write(struct file *file, const char __user *buf, size_t count, loff_t *pos)
{
    char buffer[64];
    
    if (count > sizeof(buffer) - 1)
        count = sizeof(buffer) - 1;
    
    if (copy_from_user(buffer, buf, count))
        return -EFAULT;
    
    buffer[count] = '\0';
    
    if (strncmp(buffer, "reset_stats", 11) == 0) {
        spin_lock(&stats.lock);
        stats.allocations = 0;
        stats.deallocations = 0;
        stats.dma_checks = 0;
        stats.quota_hits = 0;
        spin_unlock(&stats.lock);
        
        if (debug)
            printk(KERN_INFO "%s: Stats reset\n", MODULE_NAME);
    } else if (strncmp(buffer, "allocate", 8) == 0) {
        struct allocation *alloc;
        unsigned long agent_id = 1, size = 4096;
        
        alloc = kzalloc(sizeof(*alloc), GFP_KERNEL);
        if (alloc) {
            alloc->agent_id = agent_id;
            alloc->size = size;
            alloc->address = (unsigned long)alloc;
            
            spin_lock(&allocation_lock);
            list_add_tail(&alloc->list, &allocation_list);
            spin_unlock(&allocation_lock);
            
            spin_lock(&stats.lock);
            stats.allocations++;
            spin_unlock(&stats.lock);
        }
    }
    
    return count;
}

/* Proc ops structures */
static const struct proc_ops stats_fops = {
    .proc_read = stats_read,
};

static const struct proc_ops quotas_fops = {
    .proc_read = quotas_read,
};

static const struct proc_ops allocations_fops = {
    .proc_read = allocations_read,
};

static const struct proc_ops dma_perms_fops = {
    .proc_read = dma_perms_read,
    .proc_write = dma_perms_write,
};

static const struct proc_ops control_fops = {
    .proc_write = control_write,
};

static int __init gpu_dma_lock_init(void)
{
    printk(KERN_INFO "%s: Initializing module\n", MODULE_NAME);
    
    /* Initialize statistics */
    spin_lock_init(&stats.lock);
    stats.allocations = 0;
    stats.deallocations = 0;
    stats.dma_checks = 0;
    stats.quota_hits = 0;
    
    /* Create /proc/swarm directory */
    proc_swarm = proc_mkdir("swarm", NULL);
    if (!proc_swarm) {
        printk(KERN_ERR "%s: Failed to create /proc/swarm\n", MODULE_NAME);
        return -ENOMEM;
    }
    
    /* Create /proc/swarm/gpu directory */
    proc_gpu = proc_mkdir("gpu", proc_swarm);
    if (!proc_gpu) {
        printk(KERN_ERR "%s: Failed to create /proc/swarm/gpu\n", MODULE_NAME);
        proc_remove(proc_swarm);
        return -ENOMEM;
    }
    
    /* Create proc files */
    proc_stats = proc_create("stats", 0444, proc_gpu, &stats_fops);
    proc_quotas = proc_create("quotas", 0444, proc_gpu, &quotas_fops);
    proc_allocations = proc_create("allocations", 0444, proc_gpu, &allocations_fops);
    proc_dma_perms = proc_create("dma_permissions", 0666, proc_gpu, &dma_perms_fops);
    proc_control = proc_create("control", 0200, proc_gpu, &control_fops);
    
    if (!proc_stats || !proc_quotas || !proc_allocations || 
        !proc_dma_perms || !proc_control) {
        printk(KERN_ERR "%s: Failed to create proc files\n", MODULE_NAME);
        gpu_dma_lock_cleanup();
        return -ENOMEM;
    }
    
    printk(KERN_INFO "%s: Module loaded successfully\n", MODULE_NAME);
    return 0;
}

static void __exit gpu_dma_lock_cleanup(void)
{
    struct allocation *alloc, *tmp;
    
    printk(KERN_INFO "%s: Cleaning up module\n", MODULE_NAME);
    
    /* Remove proc files */
    proc_remove(proc_control);
    proc_remove(proc_dma_perms);
    proc_remove(proc_allocations);
    proc_remove(proc_quotas);
    proc_remove(proc_stats);
    proc_remove(proc_gpu);
    proc_remove(proc_swarm);
    
    /* Clean up allocations */
    spin_lock(&allocation_lock);
    list_for_each_entry_safe(alloc, tmp, &allocation_list, list) {
        list_del(&alloc->list);
        kfree(alloc);
    }
    spin_unlock(&allocation_lock);
    
    printk(KERN_INFO "%s: Module unloaded\n", MODULE_NAME);
}

module_init(gpu_dma_lock_init);
module_exit(gpu_dma_lock_cleanup);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("StratoSwarm");
MODULE_DESCRIPTION("GPU DMA Lock Kernel Module");
MODULE_VERSION("1.0");