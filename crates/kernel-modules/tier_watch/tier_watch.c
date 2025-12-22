/*
 * TierWatch Kernel Module
 * 
 * 5-tier memory hierarchy monitoring and migration for StratoSwarm
 */

#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/rbtree.h>
#include <linux/list.h>
#include <linux/hashtable.h>
#include <linux/ktime.h>
#include <linux/jiffies.h>
#include <linux/mm.h>
#include <linux/highmem.h>
#include <linux/numa.h>
#include <linux/migrate.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/uaccess.h>
#include <linux/workqueue.h>
#include <linux/sched.h>

#include "tier_watch.h"

#define MODULE_NAME "tier_watch"

/* Module parameters */
static int debug = 0;
module_param(debug, int, 0644);
MODULE_PARM_DESC(debug, "Enable debug logging (0=off, 1=on)");

static int enable_auto_migration = 1;
module_param(enable_auto_migration, int, 0644);
MODULE_PARM_DESC(enable_auto_migration, "Enable automatic page migration");

/* Page tracking hash table */
#define PAGE_HASH_BITS 20
DEFINE_HASHTABLE(page_hash, PAGE_HASH_BITS);

/* Per-tier page lists */
struct tier_state {
    struct list_head pages;
    spinlock_t lock;
    u64 total_pages;
    u64 hot_pages;
    u64 cold_pages;
    struct tier_stats stats;
};

/* Global state */
static struct {
    struct tier_state tiers[TIER_COUNT];
    struct tier_watch_stats stats;
    spinlock_t global_lock;
    
    /* Migration workqueue */
    struct workqueue_struct *migration_wq;
    struct work_struct migration_work;
    
    /* Pressure monitoring */
    struct timer_list pressure_timer;
    
    /* /proc entries */
    struct proc_dir_entry *proc_root;
    struct proc_dir_entry *proc_tiers;
} g_state;

/* Page tracking entry */
struct page_entry {
    struct hlist_node hash_node;
    struct list_head tier_list;
    u64 pfn;
    u64 vaddr;
    enum memory_tier tier;
    u32 access_count;
    u64 last_access_jiffies;
    u64 agent_id;
    u8 numa_node;
    u8 flags;
    spinlock_t lock;
};

/* Helper: Get tier name */
static const char *tier_name(enum memory_tier tier)
{
    static const char *names[] = {
        "gpu", "cpu", "nvme", "ssd", "hdd"
    };
    
    if (tier >= TIER_COUNT)
        return "unknown";
    return names[tier];
}

/* Helper: Find page entry by PFN */
static struct page_entry *find_page_entry(u64 pfn)
{
    struct page_entry *entry;
    
    hash_for_each_possible(page_hash, entry, hash_node, pfn) {
        if (entry->pfn == pfn)
            return entry;
    }
    
    return NULL;
}

/* Get tier information */
int tier_watch_get_tier_info(enum memory_tier tier, struct tier_info *info)
{
    unsigned long flags;
    
    if (!info || tier >= TIER_COUNT)
        return -EINVAL;
        
    info->tier = tier;
    info->capacity_bytes = tier_capacity_bytes(tier);
    info->latency_ns = tier_latency_ns(tier);
    info->name = tier_name(tier);
    
    /* Calculate bandwidth based on tier */
    switch (tier) {
    case TIER_GPU:
        info->bandwidth_mbps = 900000; /* 900 GB/s */
        break;
    case TIER_CPU:
        info->bandwidth_mbps = 100000; /* 100 GB/s */
        break;
    case TIER_NVME:
        info->bandwidth_mbps = 7000;   /* 7 GB/s */
        break;
    case TIER_SSD:
        info->bandwidth_mbps = 550;    /* 550 MB/s */
        break;
    case TIER_HDD:
        info->bandwidth_mbps = 200;    /* 200 MB/s */
        break;
    }
    
    /* Get current usage */
    spin_lock_irqsave(&g_state.tiers[tier].lock, flags);
    info->used_bytes = g_state.tiers[tier].total_pages * PAGE_SIZE;
    info->free_bytes = info->capacity_bytes - info->used_bytes;
    spin_unlock_irqrestore(&g_state.tiers[tier].lock, flags);
    
    return 0;
}
EXPORT_SYMBOL(tier_watch_get_tier_info);

/* Check if tier1 is faster than tier2 */
bool tier_watch_is_faster(enum memory_tier tier1, enum memory_tier tier2)
{
    if (tier1 >= TIER_COUNT || tier2 >= TIER_COUNT)
        return false;
    return tier1 < tier2;
}
EXPORT_SYMBOL(tier_watch_is_faster);

/* Get next slower tier */
enum memory_tier tier_watch_get_next_tier(enum memory_tier tier)
{
    if (tier >= TIER_HDD)
        return TIER_COUNT; /* Invalid */
    return tier + 1;
}
EXPORT_SYMBOL(tier_watch_get_next_tier);

/* Get previous faster tier */
enum memory_tier tier_watch_get_prev_tier(enum memory_tier tier)
{
    if (tier <= TIER_GPU)
        return TIER_COUNT; /* Invalid */
    return tier - 1;
}
EXPORT_SYMBOL(tier_watch_get_prev_tier);

/* Track a page */
int tier_watch_track_page(u64 pfn, enum memory_tier tier, u64 agent_id)
{
    return tier_watch_track_page_numa(pfn, tier, agent_id, numa_node_id());
}
EXPORT_SYMBOL(tier_watch_track_page);

/* Track a page with NUMA node */
int tier_watch_track_page_numa(u64 pfn, enum memory_tier tier, u64 agent_id, int node)
{
    struct page_entry *entry;
    unsigned long flags;
    
    if (tier >= TIER_COUNT)
        return -EINVAL;
        
    if (atomic64_read(&g_state.stats.pages_tracked) >= MAX_TRACKED_PAGES)
        return -ENOMEM;
        
    /* Check if already tracked */
    spin_lock_irqsave(&g_state.global_lock, flags);
    entry = find_page_entry(pfn);
    if (entry) {
        spin_unlock_irqrestore(&g_state.global_lock, flags);
        return -EEXIST;
    }
    spin_unlock_irqrestore(&g_state.global_lock, flags);
    
    /* Allocate new entry */
    entry = kzalloc(sizeof(*entry), GFP_KERNEL);
    if (!entry)
        return -ENOMEM;
        
    /* Initialize entry */
    entry->pfn = pfn;
    entry->tier = tier;
    entry->agent_id = agent_id;
    entry->numa_node = node;
    entry->last_access_jiffies = jiffies;
    spin_lock_init(&entry->lock);
    
    /* Add to hash table */
    spin_lock_irqsave(&g_state.global_lock, flags);
    hash_add(page_hash, &entry->hash_node, pfn);
    spin_unlock_irqrestore(&g_state.global_lock, flags);
    
    /* Add to tier list */
    spin_lock_irqsave(&g_state.tiers[tier].lock, flags);
    list_add(&entry->tier_list, &g_state.tiers[tier].pages);
    g_state.tiers[tier].total_pages++;
    spin_unlock_irqrestore(&g_state.tiers[tier].lock, flags);
    
    atomic64_inc(&g_state.stats.pages_tracked);
    
    if (debug)
        pr_info("%s: Tracked page %llx in tier %s for agent %llu\n",
                MODULE_NAME, pfn, tier_name(tier), agent_id);
                
    return 0;
}
EXPORT_SYMBOL(tier_watch_track_page_numa);

/* Untrack a page */
int tier_watch_untrack_page(u64 pfn)
{
    struct page_entry *entry;
    unsigned long flags;
    
    spin_lock_irqsave(&g_state.global_lock, flags);
    entry = find_page_entry(pfn);
    if (!entry) {
        spin_unlock_irqrestore(&g_state.global_lock, flags);
        return -ENOENT;
    }
    
    /* Remove from hash */
    hash_del(&entry->hash_node);
    spin_unlock_irqrestore(&g_state.global_lock, flags);
    
    /* Remove from tier list */
    spin_lock_irqsave(&g_state.tiers[entry->tier].lock, flags);
    list_del(&entry->tier_list);
    g_state.tiers[entry->tier].total_pages--;
    spin_unlock_irqrestore(&g_state.tiers[entry->tier].lock, flags);
    
    atomic64_dec(&g_state.stats.pages_tracked);
    
    kfree(entry);
    return 0;
}
EXPORT_SYMBOL(tier_watch_untrack_page);

/* Get page information */
int tier_watch_get_page_info(u64 pfn, struct page_info *info)
{
    struct page_entry *entry;
    unsigned long flags;
    
    if (!info)
        return -EINVAL;
        
    spin_lock_irqsave(&g_state.global_lock, flags);
    entry = find_page_entry(pfn);
    if (!entry) {
        spin_unlock_irqrestore(&g_state.global_lock, flags);
        return -ENOENT;
    }
    
    spin_lock(&entry->lock);
    info->pfn = entry->pfn;
    info->vaddr = entry->vaddr;
    info->tier = entry->tier;
    info->access_count = entry->access_count;
    info->last_access_jiffies = entry->last_access_jiffies;
    info->agent_id = entry->agent_id;
    info->numa_node = entry->numa_node;
    info->flags = entry->flags;
    spin_unlock(&entry->lock);
    
    spin_unlock_irqrestore(&g_state.global_lock, flags);
    
    return 0;
}
EXPORT_SYMBOL(tier_watch_get_page_info);

/* Handle page fault */
int tier_watch_handle_fault(u64 pfn, u64 vaddr, unsigned int flags)
{
    struct page_entry *entry;
    unsigned long iflags;
    ktime_t start_time;
    u64 latency_ns;
    
    start_time = ktime_get();
    
    spin_lock_irqsave(&g_state.global_lock, iflags);
    entry = find_page_entry(pfn);
    if (!entry) {
        spin_unlock_irqrestore(&g_state.global_lock, iflags);
        return -ENOENT;
    }
    
    spin_lock(&entry->lock);
    entry->access_count++;
    entry->last_access_jiffies = jiffies;
    if (vaddr)
        entry->vaddr = vaddr;
        
    /* Update hot/cold status */
    if (entry->access_count > HOT_PAGE_THRESHOLD) {
        spin_lock(&g_state.tiers[entry->tier].lock);
        g_state.tiers[entry->tier].hot_pages++;
        spin_unlock(&g_state.tiers[entry->tier].lock);
    }
    
    spin_unlock(&entry->lock);
    spin_unlock_irqrestore(&g_state.global_lock, iflags);
    
    /* Update statistics */
    atomic64_inc(&g_state.stats.total_faults);
    spin_lock_irqsave(&g_state.tiers[entry->tier].lock, iflags);
    g_state.tiers[entry->tier].stats.total_faults++;
    spin_unlock_irqrestore(&g_state.tiers[entry->tier].lock, iflags);
    
    latency_ns = ktime_to_ns(ktime_sub(ktime_get(), start_time));
    
    if (debug && latency_ns > 100)
        pr_warn("%s: Page fault handling took %llu ns\n", MODULE_NAME, latency_ns);
        
    return 0;
}
EXPORT_SYMBOL(tier_watch_handle_fault);

/* Get fault statistics */
int tier_watch_get_fault_stats(enum memory_tier tier, struct fault_stats *stats)
{
    unsigned long flags;
    
    if (!stats || tier >= TIER_COUNT)
        return -EINVAL;
        
    spin_lock_irqsave(&g_state.tiers[tier].lock, flags);
    stats->total_faults = g_state.tiers[tier].stats.total_faults;
    /* Mock other stats for now */
    stats->read_faults = stats->total_faults * 7 / 10;
    stats->write_faults = stats->total_faults * 3 / 10;
    stats->minor_faults = stats->total_faults * 9 / 10;
    stats->major_faults = stats->total_faults / 10;
    stats->avg_latency_ns = tier_latency_ns(tier);
    spin_unlock_irqrestore(&g_state.tiers[tier].lock, flags);
    
    return 0;
}
EXPORT_SYMBOL(tier_watch_get_fault_stats);

/* Get hot pages */
int tier_watch_get_hot_pages(enum memory_tier tier, struct page_list *pages, u32 max_pages)
{
    struct page_entry *entry;
    unsigned long flags;
    u32 count = 0;
    
    if (!pages || !pages->pages || tier >= TIER_COUNT)
        return -EINVAL;
        
    spin_lock_irqsave(&g_state.tiers[tier].lock, flags);
    
    list_for_each_entry(entry, &g_state.tiers[tier].pages, tier_list) {
        if (count >= max_pages)
            break;
            
        spin_lock(&entry->lock);
        if (entry->access_count > HOT_PAGE_THRESHOLD) {
            pages->pages[count++] = entry->pfn;
        }
        spin_unlock(&entry->lock);
    }
    
    spin_unlock_irqrestore(&g_state.tiers[tier].lock, flags);
    
    pages->count = count;
    return 0;
}
EXPORT_SYMBOL(tier_watch_get_hot_pages);

/* Get cold pages */
int tier_watch_get_cold_pages(enum memory_tier tier, struct page_list *pages, u32 max_pages)
{
    struct page_entry *entry;
    unsigned long flags;
    u32 count = 0;
    u64 age_threshold = jiffies - (HZ * 60); /* 60 seconds old */
    
    if (!pages || !pages->pages || tier >= TIER_COUNT)
        return -EINVAL;
        
    spin_lock_irqsave(&g_state.tiers[tier].lock, flags);
    
    list_for_each_entry(entry, &g_state.tiers[tier].pages, tier_list) {
        if (count >= max_pages)
            break;
            
        spin_lock(&entry->lock);
        if (entry->access_count < COLD_PAGE_THRESHOLD &&
            time_before64(entry->last_access_jiffies, age_threshold)) {
            pages->pages[count++] = entry->pfn;
        }
        spin_unlock(&entry->lock);
    }
    
    spin_unlock_irqrestore(&g_state.tiers[tier].lock, flags);
    
    pages->count = count;
    return 0;
}
EXPORT_SYMBOL(tier_watch_get_cold_pages);

/* Get migration candidates */
int tier_watch_get_migration_candidates(enum memory_tier from_tier,
                                       enum memory_tier to_tier,
                                       struct migration_candidate *candidates,
                                       u32 max_candidates)
{
    struct page_entry *entry;
    unsigned long flags;
    u32 count = 0;
    bool is_promotion = tier_watch_is_faster(to_tier, from_tier);
    
    if (!candidates || from_tier >= TIER_COUNT || to_tier >= TIER_COUNT)
        return -EINVAL;
        
    spin_lock_irqsave(&g_state.tiers[from_tier].lock, flags);
    
    list_for_each_entry(entry, &g_state.tiers[from_tier].pages, tier_list) {
        if (count >= max_candidates)
            break;
            
        spin_lock(&entry->lock);
        
        bool is_candidate = false;
        if (is_promotion && entry->access_count > HOT_PAGE_THRESHOLD) {
            is_candidate = true;
        } else if (!is_promotion && entry->access_count < COLD_PAGE_THRESHOLD) {
            is_candidate = true;
        }
        
        if (is_candidate) {
            candidates[count].pfn = entry->pfn;
            candidates[count].from_tier = from_tier;
            candidates[count].to_tier = to_tier;
            candidates[count].access_count = entry->access_count;
            candidates[count].agent_id = entry->agent_id;
            candidates[count].score = entry->access_count * tier_latency_ns(from_tier);
            count++;
        }
        
        spin_unlock(&entry->lock);
    }
    
    spin_unlock_irqrestore(&g_state.tiers[from_tier].lock, flags);
    
    return count;
}
EXPORT_SYMBOL(tier_watch_get_migration_candidates);

/* Migrate a page */
int tier_watch_migrate_page(struct migration_request *req, struct migration_result *result)
{
    struct page_entry *entry;
    unsigned long flags;
    ktime_t start_time;
    
    if (!req || !result || req->from_tier >= TIER_COUNT || req->to_tier >= TIER_COUNT)
        return -EINVAL;
        
    start_time = ktime_get();
    
    spin_lock_irqsave(&g_state.global_lock, flags);
    entry = find_page_entry(req->pfn);
    if (!entry || entry->tier != req->from_tier) {
        spin_unlock_irqrestore(&g_state.global_lock, flags);
        result->status = -ENOENT;
        return -ENOENT;
    }
    
    /* Remove from current tier */
    spin_lock(&g_state.tiers[entry->tier].lock);
    list_del(&entry->tier_list);
    g_state.tiers[entry->tier].total_pages--;
    g_state.tiers[entry->tier].stats.migrations_out++;
    spin_unlock(&g_state.tiers[entry->tier].lock);
    
    /* Update entry */
    spin_lock(&entry->lock);
    entry->tier = req->to_tier;
    spin_unlock(&entry->lock);
    
    /* Add to new tier */
    spin_lock(&g_state.tiers[req->to_tier].lock);
    list_add(&entry->tier_list, &g_state.tiers[req->to_tier].pages);
    g_state.tiers[req->to_tier].total_pages++;
    g_state.tiers[req->to_tier].stats.migrations_in++;
    spin_unlock(&g_state.tiers[req->to_tier].lock);
    
    spin_unlock_irqrestore(&g_state.global_lock, flags);
    
    /* Simulate migration delay */
    if (debug)
        udelay(MIGRATION_COST_NS / 1000);
        
    /* Update result */
    result->status = 0;
    result->latency_ns = ktime_to_ns(ktime_sub(ktime_get(), start_time));
    result->pages_moved = 1;
    
    /* Update statistics */
    atomic64_inc(&g_state.stats.total_migrations);
    atomic64_add(result->latency_ns, &g_state.stats.total_migration_ns);
    
    if (debug)
        pr_info("%s: Migrated page %llx from %s to %s in %llu ns\n",
                MODULE_NAME, req->pfn, tier_name(req->from_tier),
                tier_name(req->to_tier), result->latency_ns);
                
    return 0;
}
EXPORT_SYMBOL(tier_watch_migrate_page);

/* Get memory pressure */
int tier_watch_get_pressure(enum memory_tier tier, struct tier_pressure *pressure)
{
    struct tier_info info;
    unsigned long flags;
    int ret;
    
    if (!pressure || tier >= TIER_COUNT)
        return -EINVAL;
        
    ret = tier_watch_get_tier_info(tier, &info);
    if (ret != 0)
        return ret;
        
    pressure->tier = tier;
    pressure->free_pages = info.free_bytes / PAGE_SIZE;
    pressure->reclaimable_pages = 0; /* Would need to scan */
    
    /* Calculate pressure level */
    u64 usage_percent = (info.used_bytes * 100) / info.capacity_bytes;
    
    if (usage_percent >= 95) {
        pressure->level = PRESSURE_CRITICAL;
        pressure->pressure_value = 100;
    } else if (usage_percent >= 85) {
        pressure->level = PRESSURE_HIGH;
        pressure->pressure_value = 80;
    } else if (usage_percent >= 70) {
        pressure->level = PRESSURE_MEDIUM;
        pressure->pressure_value = 50;
    } else {
        pressure->level = PRESSURE_LOW;
        pressure->pressure_value = usage_percent / 2;
    }
    
    return 0;
}
EXPORT_SYMBOL(tier_watch_get_pressure);

/* Update pressure (called by timer) */
void tier_watch_update_pressure(void)
{
    int tier;
    struct tier_pressure pressure;
    
    for (tier = 0; tier < TIER_COUNT; tier++) {
        tier_watch_get_pressure(tier, &pressure);
        
        if (pressure.level >= PRESSURE_HIGH && enable_auto_migration) {
            /* Schedule migration work */
            queue_work(g_state.migration_wq, &g_state.migration_work);
        }
    }
}
EXPORT_SYMBOL(tier_watch_update_pressure);

/* Get NUMA statistics */
int tier_watch_get_numa_stats(int node, struct numa_stats *stats)
{
    struct page_entry *entry;
    unsigned long flags;
    int tier;
    
    if (!stats || node >= MAX_NUMNODES)
        return -EINVAL;
        
    memset(stats, 0, sizeof(*stats));
    stats->node_id = node;
    
    /* Count pages on this node */
    for (tier = 0; tier < TIER_COUNT; tier++) {
        spin_lock_irqsave(&g_state.tiers[tier].lock, flags);
        
        list_for_each_entry(entry, &g_state.tiers[tier].pages, tier_list) {
            if (entry->numa_node == node) {
                stats->pages_on_node++;
            }
        }
        
        spin_unlock_irqrestore(&g_state.tiers[tier].lock, flags);
    }
    
    return 0;
}
EXPORT_SYMBOL(tier_watch_get_numa_stats);

/* Get optimal NUMA node for page */
int tier_watch_get_optimal_numa_node(u64 pfn)
{
    struct page_entry *entry;
    unsigned long flags;
    int node;
    
    spin_lock_irqsave(&g_state.global_lock, flags);
    entry = find_page_entry(pfn);
    if (!entry) {
        spin_unlock_irqrestore(&g_state.global_lock, flags);
        return numa_node_id(); /* Default to current node */
    }
    
    node = entry->numa_node;
    spin_unlock_irqrestore(&g_state.global_lock, flags);
    
    /* In real implementation, would analyze access patterns */
    return node;
}
EXPORT_SYMBOL(tier_watch_get_optimal_numa_node);

/* Get agent memory statistics */
int tier_watch_get_agent_memory(u64 agent_id, struct agent_memory_stats *stats)
{
    struct page_entry *entry;
    unsigned long flags;
    int tier;
    
    if (!stats)
        return -EINVAL;
        
    memset(stats, 0, sizeof(*stats));
    stats->agent_id = agent_id;
    
    /* Count pages for this agent */
    for (tier = 0; tier < TIER_COUNT; tier++) {
        spin_lock_irqsave(&g_state.tiers[tier].lock, flags);
        
        list_for_each_entry(entry, &g_state.tiers[tier].pages, tier_list) {
            if (entry->agent_id == agent_id) {
                stats->pages_in_tier[tier]++;
                stats->total_pages++;
            }
        }
        
        spin_unlock_irqrestore(&g_state.tiers[tier].lock, flags);
    }
    
    return 0;
}
EXPORT_SYMBOL(tier_watch_get_agent_memory);

/* Get tier statistics */
int tier_watch_get_tier_stats(enum memory_tier tier, struct tier_stats *stats)
{
    unsigned long flags;
    
    if (!stats || tier >= TIER_COUNT)
        return -EINVAL;
        
    spin_lock_irqsave(&g_state.tiers[tier].lock, flags);
    *stats = g_state.tiers[tier].stats;
    stats->total_pages = g_state.tiers[tier].total_pages;
    stats->hot_pages = g_state.tiers[tier].hot_pages;
    stats->cold_pages = g_state.tiers[tier].cold_pages;
    spin_unlock_irqrestore(&g_state.tiers[tier].lock, flags);
    
    /* Get current pressure */
    struct tier_pressure pressure;
    tier_watch_get_pressure(tier, &pressure);
    stats->pressure = pressure.level;
    
    return 0;
}
EXPORT_SYMBOL(tier_watch_get_tier_stats);

/* Get module statistics */
int tier_watch_get_module_stats(struct tier_watch_stats *stats)
{
    if (!stats)
        return -EINVAL;
        
    stats->pages_tracked = atomic64_read(&g_state.stats.pages_tracked);
    stats->total_faults = atomic64_read(&g_state.stats.total_faults);
    stats->total_migrations = atomic64_read(&g_state.stats.total_migrations);
    stats->total_migration_ns = atomic64_read(&g_state.stats.total_migration_ns);
    stats->failed_migrations = atomic64_read(&g_state.stats.failed_migrations);
    
    return 0;
}
EXPORT_SYMBOL(tier_watch_get_module_stats);

/* Migration work handler */
static void migration_work_handler(struct work_struct *work)
{
    struct migration_candidate candidates[MAX_MIGRATION_BATCH];
    struct migration_request req;
    struct migration_result result;
    int tier, count;
    
    if (debug)
        pr_info("%s: Running automatic migration\n", MODULE_NAME);
        
    /* Check each tier for migration candidates */
    for (tier = TIER_HDD; tier > TIER_GPU; tier--) {
        /* Promote hot pages */
        count = tier_watch_get_migration_candidates(tier, tier - 1,
                                                   candidates, MAX_MIGRATION_BATCH);
        
        for (int i = 0; i < count && i < 10; i++) { /* Limit migrations */
            req.pfn = candidates[i].pfn;
            req.from_tier = candidates[i].from_tier;
            req.to_tier = candidates[i].to_tier;
            req.priority = 50;
            req.agent_id = candidates[i].agent_id;
            req.reason = MIGRATION_HOT_PROMOTION;
            
            tier_watch_migrate_page(&req, &result);
        }
    }
}

/* Pressure timer callback */
static void pressure_timer_callback(struct timer_list *timer)
{
    tier_watch_update_pressure();
    
    /* Reschedule timer */
    mod_timer(&g_state.pressure_timer, jiffies + HZ * 5); /* 5 seconds */
}

/* /proc interface */
static int tier_stats_show(struct seq_file *m, void *v)
{
    struct tier_stats stats;
    enum memory_tier tier = (enum memory_tier)(long)m->private;
    
    if (tier_watch_get_tier_stats(tier, &stats) != 0)
        return -EIO;
        
    seq_printf(m, "Tier: %s\n", tier_name(tier));
    seq_printf(m, "==================\n");
    seq_printf(m, "total_pages: %llu\n", stats.total_pages);
    seq_printf(m, "hot_pages: %llu\n", stats.hot_pages);
    seq_printf(m, "cold_pages: %llu\n", stats.cold_pages);
    seq_printf(m, "migrations_in: %llu\n", stats.migrations_in);
    seq_printf(m, "migrations_out: %llu\n", stats.migrations_out);
    seq_printf(m, "total_faults: %llu\n", stats.total_faults);
    seq_printf(m, "pressure: %s\n",
               stats.pressure == PRESSURE_LOW ? "low" :
               stats.pressure == PRESSURE_MEDIUM ? "medium" :
               stats.pressure == PRESSURE_HIGH ? "high" : "critical");
    
    return 0;
}

static int tier_stats_open(struct inode *inode, struct file *file)
{
    return single_open(file, tier_stats_show, PDE_DATA(inode));
}

static const struct proc_ops tier_stats_fops = {
    .proc_open = tier_stats_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

static int module_stats_show(struct seq_file *m, void *v)
{
    struct tier_watch_stats stats;
    
    tier_watch_get_module_stats(&stats);
    
    seq_printf(m, "TierWatch Module Statistics\n");
    seq_printf(m, "==========================\n");
    seq_printf(m, "Pages tracked: %llu\n", stats.pages_tracked);
    seq_printf(m, "Total faults: %llu\n", stats.total_faults);
    seq_printf(m, "Total migrations: %llu\n", stats.total_migrations);
    seq_printf(m, "Failed migrations: %llu\n", stats.failed_migrations);
    
    if (stats.total_migrations > 0) {
        u64 avg_latency = stats.total_migration_ns / stats.total_migrations;
        seq_printf(m, "Avg migration latency: %llu ns\n", avg_latency);
    }
    
    return 0;
}

static int module_stats_open(struct inode *inode, struct file *file)
{
    return single_open(file, module_stats_show, NULL);
}

static const struct proc_ops module_stats_fops = {
    .proc_open = module_stats_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

/* Initialize /proc interface */
int tier_watch_proc_init(void)
{
    int tier;
    
    g_state.proc_root = proc_mkdir("swarm", NULL);
    if (!g_state.proc_root)
        return -ENOMEM;
        
    g_state.proc_tiers = proc_mkdir("tiers", g_state.proc_root);
    if (!g_state.proc_tiers)
        return -ENOMEM;
        
    /* Create per-tier directories */
    for (tier = 0; tier < TIER_COUNT; tier++) {
        struct proc_dir_entry *tier_dir;
        
        tier_dir = proc_mkdir(tier_name(tier), g_state.proc_tiers);
        if (!tier_dir)
            continue;
            
        proc_create_data("stats", 0444, tier_dir, &tier_stats_fops, (void *)(long)tier);
    }
    
    /* Create module stats */
    proc_create("module_stats", 0444, g_state.proc_root, &module_stats_fops);
    
    return 0;
}

/* Cleanup /proc interface */
void tier_watch_proc_exit(void)
{
    if (g_state.proc_tiers) {
        int tier;
        for (tier = 0; tier < TIER_COUNT; tier++) {
            remove_proc_subtree(tier_name(tier), g_state.proc_tiers);
        }
        remove_proc_entry("tiers", g_state.proc_root);
    }
    
    if (g_state.proc_root) {
        remove_proc_entry("module_stats", g_state.proc_root);
        remove_proc_entry("swarm", NULL);
    }
}

/* Module initialization */
int tier_watch_init(void)
{
    int tier;
    
    pr_info("%s: Initializing module v%s\n", MODULE_NAME, TIER_WATCH_VERSION);
    
    /* Initialize per-tier state */
    for (tier = 0; tier < TIER_COUNT; tier++) {
        INIT_LIST_HEAD(&g_state.tiers[tier].pages);
        spin_lock_init(&g_state.tiers[tier].lock);
        g_state.tiers[tier].total_pages = 0;
        g_state.tiers[tier].hot_pages = 0;
        g_state.tiers[tier].cold_pages = 0;
        memset(&g_state.tiers[tier].stats, 0, sizeof(struct tier_stats));
    }
    
    /* Initialize global state */
    spin_lock_init(&g_state.global_lock);
    atomic64_set(&g_state.stats.pages_tracked, 0);
    atomic64_set(&g_state.stats.total_faults, 0);
    atomic64_set(&g_state.stats.total_migrations, 0);
    atomic64_set(&g_state.stats.total_migration_ns, 0);
    atomic64_set(&g_state.stats.failed_migrations, 0);
    
    /* Create migration workqueue */
    g_state.migration_wq = create_singlethread_workqueue("tier_watch_migration");
    if (!g_state.migration_wq)
        return -ENOMEM;
        
    INIT_WORK(&g_state.migration_work, migration_work_handler);
    
    /* Initialize pressure timer */
    timer_setup(&g_state.pressure_timer, pressure_timer_callback, 0);
    mod_timer(&g_state.pressure_timer, jiffies + HZ * 5);
    
    /* Initialize /proc interface */
    tier_watch_proc_init();
    
    pr_info("%s: Module loaded successfully\n", MODULE_NAME);
    return 0;
}

/* Module cleanup */
void tier_watch_exit(void)
{
    struct page_entry *entry;
    struct hlist_node *tmp;
    int bkt;
    
    pr_info("%s: Cleaning up module\n", MODULE_NAME);
    
    /* Stop pressure timer */
    del_timer_sync(&g_state.pressure_timer);
    
    /* Stop migration workqueue */
    if (g_state.migration_wq) {
        cancel_work_sync(&g_state.migration_work);
        destroy_workqueue(g_state.migration_wq);
    }
    
    /* Remove /proc entries */
    tier_watch_proc_exit();
    
    /* Free all tracked pages */
    hash_for_each_safe(page_hash, bkt, tmp, entry, hash_node) {
        hash_del(&entry->hash_node);
        list_del(&entry->tier_list);
        kfree(entry);
    }
    
    pr_info("%s: Module unloaded\n", MODULE_NAME);
}

module_init(tier_watch_init);
module_exit(tier_watch_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("StratoSwarm");
MODULE_DESCRIPTION("5-Tier Memory Hierarchy Monitoring");
MODULE_VERSION(TIER_WATCH_VERSION);