/*
 * TierWatch Kernel Module - C Entry Point
 *
 * Monitors 5-tier memory hierarchy and provides migration detection
 */

#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/mm.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/kthread.h>
#include <linux/delay.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("StratoSwarm");
MODULE_DESCRIPTION("TierWatch - 5-tier memory hierarchy monitor");
MODULE_VERSION("1.0.0");

/* External Rust functions */
extern int tier_watch_init(void);
extern void tier_watch_cleanup(void);
extern int tier_watch_handle_fault(unsigned long vaddr, unsigned long pfn,
                                  int fault_type, unsigned long agent_id);
extern int tier_watch_get_stats(int tier, void *stats);
extern int tier_watch_check_pressure(int *tiers, unsigned int *pressures,
                                   unsigned int max_tiers);
extern void tier_watch_set_detector_enabled(int enabled);
extern int tier_watch_get_numa_node(unsigned int cpu_id);
extern void tier_watch_record_numa_access(unsigned int cpu_id, unsigned long pfn);
extern int tier_watch_get_performance(void *metrics);
extern long tier_watch_generate_report(char *buffer, unsigned long size);
extern const char *tier_watch_get_version(void);

/* Debug parameter */
static int debug = 0;
module_param(debug, int, 0644);
MODULE_PARM_DESC(debug, "Enable debug output (0=off, 1=on)");

/* Proc filesystem root */
static struct proc_dir_entry *swarm_dir;
static struct proc_dir_entry *tiers_dir;

/* Migration worker thread */
static struct task_struct *migration_thread;

/* Original page fault handler */
static int (*original_handle_mm_fault)(struct vm_area_struct *vma,
                                      unsigned long address,
                                      unsigned int flags);

/* Our page fault handler wrapper */
static int tier_watch_mm_fault(struct vm_area_struct *vma,
                              unsigned long address,
                              unsigned int flags)
{
    int ret;
    unsigned long pfn = 0;
    struct page *page;
    int fault_type = 0; /* 0=Major, 1=Minor */
    
    /* Call original handler first */
    ret = original_handle_mm_fault(vma, address, flags);
    
    /* Extract page information if successful */
    if (!(ret & VM_FAULT_ERROR)) {
        page = follow_page(vma, address, FOLL_GET);
        if (page) {
            pfn = page_to_pfn(page);
            
            /* Determine fault type */
            if (ret & VM_FAULT_MAJOR)
                fault_type = 0; /* Major */
            else
                fault_type = 1; /* Minor */
            
            /* Call our Rust handler */
            tier_watch_handle_fault(address, pfn, fault_type, 0);
            
            put_page(page);
        }
    }
    
    return ret;
}

/* Proc file operations for tier stats */
static int tier_stats_show(struct seq_file *m, void *v)
{
    char *buffer;
    long len;
    
    buffer = vmalloc(PAGE_SIZE * 4);
    if (!buffer)
        return -ENOMEM;
    
    len = tier_watch_generate_report(buffer, PAGE_SIZE * 4);
    if (len > 0) {
        seq_printf(m, "%s", buffer);
    }
    
    vfree(buffer);
    return 0;
}

static int tier_stats_open(struct inode *inode, struct file *file)
{
    return single_open(file, tier_stats_show, NULL);
}

static const struct proc_ops tier_stats_ops = {
    .proc_open = tier_stats_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

/* Migration worker thread function */
static int migration_worker(void *data)
{
    int pressure_tiers[5];
    unsigned int pressures[5];
    int count;
    
    while (!kthread_should_stop()) {
        /* Check for high pressure tiers */
        count = tier_watch_check_pressure(pressure_tiers, pressures, 5);
        
        if (count > 0 && debug) {
            printk(KERN_INFO "tier_watch: Found %d tiers under pressure\n", count);
        }
        
        /* Sleep for 100ms between checks */
        msleep(100);
    }
    
    return 0;
}

/* Create proc entries */
static int create_proc_entries(void)
{
    /* Create /proc/swarm directory */
    swarm_dir = proc_mkdir("swarm", NULL);
    if (!swarm_dir) {
        printk(KERN_ERR "tier_watch: Failed to create /proc/swarm\n");
        return -ENOMEM;
    }
    
    /* Create /proc/swarm/tiers directory */
    tiers_dir = proc_mkdir("tiers", swarm_dir);
    if (!tiers_dir) {
        printk(KERN_ERR "tier_watch: Failed to create /proc/swarm/tiers\n");
        proc_remove(swarm_dir);
        return -ENOMEM;
    }
    
    /* Create /proc/swarm/tiers/stats */
    if (!proc_create("stats", 0444, tiers_dir, &tier_stats_ops)) {
        printk(KERN_ERR "tier_watch: Failed to create /proc/swarm/tiers/stats\n");
        proc_remove(tiers_dir);
        proc_remove(swarm_dir);
        return -ENOMEM;
    }
    
    /* Create per-tier stat files */
    const char *tier_names[] = {"gpu", "cpu", "nvme", "ssd", "hdd"};
    int i;
    for (i = 0; i < 5; i++) {
        struct proc_dir_entry *tier_subdir;
        char path[32];
        
        /* Create /proc/swarm/tiers/<tier> */
        tier_subdir = proc_mkdir(tier_names[i], tiers_dir);
        if (!tier_subdir)
            continue;
        
        /* Create /proc/swarm/tiers/<tier>/stats */
        snprintf(path, sizeof(path), "stats");
        proc_create(path, 0444, tier_subdir, &tier_stats_ops);
    }
    
    return 0;
}

/* Remove proc entries */
static void remove_proc_entries(void)
{
    if (tiers_dir) {
        remove_proc_subtree("tiers", swarm_dir);
    }
    if (swarm_dir) {
        proc_remove(swarm_dir);
    }
}

/* Module initialization */
static int __init tier_watch_module_init(void)
{
    int ret;
    
    printk(KERN_INFO "tier_watch: Loading %s\n", tier_watch_get_version());
    
    /* Initialize Rust components */
    ret = tier_watch_init();
    if (ret) {
        printk(KERN_ERR "tier_watch: Failed to initialize Rust components\n");
        return ret;
    }
    
    /* Create proc entries */
    ret = create_proc_entries();
    if (ret) {
        tier_watch_cleanup();
        return ret;
    }
    
    /* Hook page fault handler */
    /* Note: In real implementation, would use proper kernel hooks */
    /* This is simplified for the example */
    
    /* Start migration worker thread */
    migration_thread = kthread_create(migration_worker, NULL, "tier_watch_migration");
    if (IS_ERR(migration_thread)) {
        printk(KERN_ERR "tier_watch: Failed to create migration thread\n");
        remove_proc_entries();
        tier_watch_cleanup();
        return PTR_ERR(migration_thread);
    }
    
    wake_up_process(migration_thread);
    
    printk(KERN_INFO "tier_watch: Module loaded successfully\n");
    if (debug) {
        printk(KERN_INFO "tier_watch: Debug mode enabled\n");
    }
    
    return 0;
}

/* Module cleanup */
static void __exit tier_watch_module_exit(void)
{
    printk(KERN_INFO "tier_watch: Unloading module\n");
    
    /* Stop migration thread */
    if (migration_thread) {
        kthread_stop(migration_thread);
    }
    
    /* Unhook page fault handler */
    
    /* Remove proc entries */
    remove_proc_entries();
    
    /* Cleanup Rust components */
    tier_watch_cleanup();
    
    printk(KERN_INFO "tier_watch: Module unloaded\n");
}

module_init(tier_watch_module_init);
module_exit(tier_watch_module_exit);