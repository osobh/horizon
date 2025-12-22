/*
 * SwarmGuard Kernel Module
 * Main entry point for the kernel module
 */

#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/slab.h>
#include <linux/uaccess.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("StratoSwarm Team");
MODULE_DESCRIPTION("SwarmGuard - Resource enforcement and namespace management for StratoSwarm");
MODULE_VERSION("0.1.0");

/* Module parameters */
static unsigned int max_agents = 200000;
module_param(max_agents, uint, 0644);
MODULE_PARM_DESC(max_agents, "Maximum number of concurrent agents");

static int debug = 0;
module_param(debug, int, 0644);
MODULE_PARM_DESC(debug, "Enable debug logging (0=off, 1=on)");

/* Proc filesystem root */
static struct proc_dir_entry *swarm_proc_root;

/* Forward declarations */
static int swarm_guard_init(void);
static void swarm_guard_cleanup(void);

/* Debug logging macro */
#define swarm_debug(fmt, ...) \
    do { if (debug) pr_info("[swarm_guard] " fmt, ##__VA_ARGS__); } while (0)

#define swarm_info(fmt, ...) \
    pr_info("[swarm_guard] " fmt, ##__VA_ARGS__)

#define swarm_err(fmt, ...) \
    pr_err("[swarm_guard] " fmt, ##__VA_ARGS__)

/* /proc/swarm/status implementation */
static int status_show(struct seq_file *m, void *v)
{
    /* Call into Rust code via FFI */
    extern char* swarm_guard_get_status(void);
    char *status = swarm_guard_get_status();
    
    if (status) {
        seq_printf(m, "%s", status);
        kfree(status);
    }
    
    return 0;
}

static int status_open(struct inode *inode, struct file *file)
{
    return single_open(file, status_show, NULL);
}

static const struct proc_ops status_proc_ops = {
    .proc_open = status_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

/* /proc/swarm/create implementation */
static ssize_t create_write(struct file *file, const char __user *buf,
                          size_t count, loff_t *ppos)
{
    char *kbuf;
    int ret;
    
    if (count > 4096) /* Reasonable limit */
        return -EINVAL;
    
    kbuf = kmalloc(count + 1, GFP_KERNEL);
    if (!kbuf)
        return -ENOMEM;
    
    if (copy_from_user(kbuf, buf, count)) {
        kfree(kbuf);
        return -EFAULT;
    }
    kbuf[count] = '\0';
    
    /* Call into Rust code to create agent */
    extern int swarm_guard_create_agent(const char *config);
    ret = swarm_guard_create_agent(kbuf);
    
    kfree(kbuf);
    
    return ret < 0 ? ret : count;
}

static const struct proc_ops create_proc_ops = {
    .proc_write = create_write,
};

/* Create /proc/swarm directory structure */
static int create_proc_entries(void)
{
    /* Create /proc/swarm */
    swarm_proc_root = proc_mkdir("swarm", NULL);
    if (!swarm_proc_root) {
        swarm_err("Failed to create /proc/swarm\n");
        return -ENOMEM;
    }
    
    /* Create /proc/swarm/status */
    if (!proc_create("status", 0444, swarm_proc_root, &status_proc_ops)) {
        swarm_err("Failed to create /proc/swarm/status\n");
        return -ENOMEM;
    }
    
    /* Create /proc/swarm/create */
    if (!proc_create("create", 0200, swarm_proc_root, &create_proc_ops)) {
        swarm_err("Failed to create /proc/swarm/create\n");
        return -ENOMEM;
    }
    
    swarm_debug("Created /proc/swarm interface\n");
    return 0;
}

/* Remove /proc entries */
static void remove_proc_entries(void)
{
    if (swarm_proc_root) {
        remove_proc_subtree("swarm", NULL);
        swarm_proc_root = NULL;
    }
}

/* Module initialization */
static int __init swarm_guard_init_module(void)
{
    int ret;
    
    swarm_info("Loading SwarmGuard kernel module (max_agents=%u)\n", max_agents);
    
    /* Initialize Rust subsystems */
    ret = swarm_guard_init();
    if (ret < 0) {
        swarm_err("Failed to initialize SwarmGuard: %d\n", ret);
        return ret;
    }
    
    /* Create proc entries */
    ret = create_proc_entries();
    if (ret < 0) {
        swarm_guard_cleanup();
        return ret;
    }
    
    swarm_info("SwarmGuard kernel module loaded successfully\n");
    return 0;
}

/* Module cleanup */
static void __exit swarm_guard_exit_module(void)
{
    swarm_info("Unloading SwarmGuard kernel module\n");
    
    /* Remove proc entries */
    remove_proc_entries();
    
    /* Cleanup Rust subsystems */
    swarm_guard_cleanup();
    
    swarm_info("SwarmGuard kernel module unloaded\n");
}

/* Rust FFI functions (implemented in Rust) */
extern int swarm_guard_init(void);
extern void swarm_guard_cleanup(void);

module_init(swarm_guard_init_module);
module_exit(swarm_guard_exit_module);