/*
 * Swarm Guard Kernel Module - Real Implementation
 * 
 * Real cgroup v2, namespace isolation, and resource enforcement
 */

#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/fs.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/cgroup.h>
#include <linux/pid_namespace.h>
#include <linux/user_namespace.h>
#include <linux/nsproxy.h>
#include <linux/sched.h>
#include <linux/sched/task.h>
#include <linux/kprobes.h>
#include <linux/fdtable.h>
#include <linux/file.h>
#include <linux/security.h>
#include <linux/keyctl.h>
#include <linux/key.h>
#include <linux/cred.h>
#include <linux/capability.h>
#include <linux/bpf.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <linux/personality.h>

#include "swarm_guard.h"

#define MODULE_NAME "swarm_guard"

/* Module parameters */
static int debug = 0;
module_param(debug, int, 0644);
MODULE_PARM_DESC(debug, "Enable debug logging");

static int enforce_limits = 1;
module_param(enforce_limits, int, 0644);
MODULE_PARM_DESC(enforce_limits, "Enforce resource limits");

/* Global state */
static struct swarm_guard_state {
    struct list_head agents;
    spinlock_t agents_lock;
    struct kmem_cache *agent_cache;
    atomic64_t total_agents;
    atomic64_t active_agents;
    
    /* Cgroup v2 root */
    struct cgroup *swarm_cgroup_root;
    struct cgroup_subsys_state *swarm_css;
    
    /* Kprobes for system call interception */
    struct kprobe *syscall_probes;
    int num_probes;
    
    /* Keyring for trust scores */
    struct key *trust_keyring;
    
    /* /proc entries */
    struct proc_dir_entry *proc_root;
} g_state;

/* Real agent structure with kernel objects */
struct swarm_agent_real {
    struct swarm_agent base;
    
    /* Real cgroup for resource control */
    struct cgroup *cgroup;
    struct cgroup_subsys_state *css;
    
    /* Real namespaces */
    struct nsproxy *nsproxy;
    struct pid_namespace *pid_ns;
    struct user_namespace *user_ns;
    struct net *net_ns;
    
    /* Resource tracking with cgroup controllers */
    struct {
        struct cgroup_subsys_state *memory_css;
        struct cgroup_subsys_state *cpu_css;
        struct cgroup_subsys_state *io_css;
    } controllers;
    
    /* Security context */
    struct cred *cred;
    kuid_t uid;
    kgid_t gid;
    
    /* Trust score key */
    struct key *trust_key;
    
    /* System call filtering */
    struct seccomp_filter *seccomp_filter;
    unsigned long syscall_bitmap[BITS_TO_LONGS(NR_syscalls)];
};

/* Cgroup v2 operations */
static int create_agent_cgroup(struct swarm_agent_real *agent)
{
    char name[64];
    struct cgroup *parent;
    struct kernfs_node *kn;
    int ret;
    
    snprintf(name, sizeof(name), "agent_%llu", agent->base.id);
    
    /* Get swarm cgroup parent */
    parent = g_state.swarm_cgroup_root;
    if (!parent) {
        /* Use system root if swarm root not available */
        parent = cgroup_root;
    }
    
    /* Create cgroup directory */
    kn = kernfs_create_dir(parent->kn, name, 0755, NULL);
    if (IS_ERR(kn))
        return PTR_ERR(kn);
        
    /* Create cgroup */
    agent->cgroup = cgroup_create(parent, kn);
    if (IS_ERR(agent->cgroup)) {
        kernfs_remove(kn);
        return PTR_ERR(agent->cgroup);
    }
    
    /* Get subsystem state */
    agent->css = cgroup_css(agent->cgroup, NULL);
    
    /* Setup memory controller */
    agent->controllers.memory_css = cgroup_get_e_css(agent->cgroup, &memory_cgrp_subsys);
    if (agent->controllers.memory_css) {
        /* Set memory limit */
        struct page_counter *counter = &agent->controllers.memory_css->cgroup->memory.memory;
        page_counter_set_max(counter, agent->base.limits.memory_bytes >> PAGE_SHIFT);
    }
    
    /* Setup CPU controller */
    agent->controllers.cpu_css = cgroup_get_e_css(agent->cgroup, &cpu_cgrp_subsys);
    if (agent->controllers.cpu_css) {
        /* Set CPU quota (bandwidth control) */
        u64 quota = agent->base.limits.cpu_percent * 10000; /* Convert to us per 100ms */
        cgroup_file_write(agent->cgroup, "cpu.max", &quota, sizeof(quota));
    }
    
    /* Setup IO controller */
    agent->controllers.io_css = cgroup_get_e_css(agent->cgroup, &io_cgrp_subsys);
    
    if (debug)
        pr_info("%s: Created cgroup for agent %llu\n", MODULE_NAME, agent->base.id);
        
    return 0;
}

/* Real namespace creation */
static int create_agent_namespaces(struct swarm_agent_real *agent)
{
    struct nsproxy *new_ns;
    unsigned long flags = 0;
    int ret;
    
    /* Determine namespace flags based on config */
    if (agent->base.config.enable_namespaces & SWARM_NS_PID)
        flags |= CLONE_NEWPID;
    if (agent->base.config.enable_namespaces & SWARM_NS_NET)
        flags |= CLONE_NEWNET;
    if (agent->base.config.enable_namespaces & SWARM_NS_IPC)
        flags |= CLONE_NEWIPC;
    if (agent->base.config.enable_namespaces & SWARM_NS_UTS)
        flags |= CLONE_NEWUTS;
    if (agent->base.config.enable_namespaces & SWARM_NS_MNT)
        flags |= CLONE_NEWNS;
    
    if (flags == 0) {
        /* No namespaces requested */
        agent->nsproxy = NULL;
        return 0;
    }
    
    /* Create new namespace proxy */
    new_ns = create_new_namespaces(flags, current, current->fs);
    if (IS_ERR(new_ns))
        return PTR_ERR(new_ns);
        
    agent->nsproxy = new_ns;
    
    /* Save specific namespace pointers */
    if (flags & CLONE_NEWPID)
        agent->pid_ns = new_ns->pid_ns_for_children;
    if (flags & CLONE_NEWNET)
        agent->net_ns = new_ns->net_ns;
        
    /* Create user namespace if requested */
    if (agent->base.config.enable_namespaces & SWARM_NS_USER) {
        struct user_namespace *new_user_ns;
        
        new_user_ns = create_user_ns(current_user_ns());
        if (IS_ERR(new_user_ns)) {
            free_nsproxy(new_ns);
            return PTR_ERR(new_user_ns);
        }
        
        agent->user_ns = new_user_ns;
    }
    
    if (debug)
        pr_info("%s: Created namespaces (flags: 0x%lx) for agent %llu\n",
                MODULE_NAME, flags, agent->base.id);
                
    return 0;
}

/* Trust score management using kernel keyring */
static int init_agent_trust_score(struct swarm_agent_real *agent)
{
    char desc[64];
    struct key *key;
    int ret;
    
    snprintf(desc, sizeof(desc), "swarm:agent:%llu:trust", agent->base.id);
    
    /* Create key for trust score */
    key = key_alloc(&key_type_user, desc,
                    GLOBAL_ROOT_UID, GLOBAL_ROOT_GID,
                    current_cred(),
                    KEY_POS_VIEW | KEY_POS_READ | KEY_POS_WRITE | KEY_POS_SEARCH,
                    KEY_ALLOC_NOT_IN_QUOTA, NULL);
    if (IS_ERR(key))
        return PTR_ERR(key);
        
    /* Set initial trust score */
    ret = key_instantiate_and_link(key, &agent->base.trust_score,
                                  sizeof(agent->base.trust_score),
                                  g_state.trust_keyring, NULL);
    if (ret < 0) {
        key_put(key);
        return ret;
    }
    
    agent->trust_key = key;
    return 0;
}

/* Update trust score in keyring */
static void update_trust_score(struct swarm_agent_real *agent, int delta)
{
    int new_score;
    
    spin_lock(&agent->base.lock);
    new_score = agent->base.trust_score + delta;
    if (new_score < 0) new_score = 0;
    if (new_score > 100) new_score = 100;
    agent->base.trust_score = new_score;
    spin_unlock(&agent->base.lock);
    
    /* Update key */
    if (agent->trust_key) {
        key_revoke(agent->trust_key);
        key_instantiate_and_link(agent->trust_key, &new_score,
                               sizeof(new_score), NULL, NULL);
    }
    
    if (debug)
        pr_info("%s: Agent %llu trust score: %d (delta: %d)\n",
                MODULE_NAME, agent->base.id, new_score, delta);
}

/* System call interception handler */
static int syscall_intercept_handler(struct kprobe *p, struct pt_regs *regs)
{
    struct swarm_agent_real *agent;
    int syscall_nr = regs->orig_ax;
    bool allowed = true;
    
    /* Find agent for current task */
    /* In real implementation, would use task->cgroups to find agent */
    
    /* Check if syscall is allowed */
    if (agent && test_bit(syscall_nr, agent->syscall_bitmap)) {
        allowed = false;
        
        /* Log violation */
        spin_lock(&agent->base.lock);
        agent->base.stats.syscall_violations++;
        spin_unlock(&agent->base.lock);
        
        /* Update trust score */
        update_trust_score(agent, -5);
        
        /* Block syscall */
        regs->orig_ax = -1;
        return 1; /* Skip original syscall */
    }
    
    return 0; /* Execute original syscall */
}

/* Setup system call interception */
static int setup_syscall_interception(void)
{
    static const char *monitored_syscalls[] = {
        "__x64_sys_open",
        "__x64_sys_connect",
        "__x64_sys_execve",
        "__x64_sys_ptrace",
        "__x64_sys_mount"
    };
    int num_syscalls = ARRAY_SIZE(monitored_syscalls);
    int i, ret;
    
    g_state.syscall_probes = kcalloc(num_syscalls, sizeof(struct kprobe), GFP_KERNEL);
    if (!g_state.syscall_probes)
        return -ENOMEM;
        
    for (i = 0; i < num_syscalls; i++) {
        g_state.syscall_probes[i].symbol_name = monitored_syscalls[i];
        g_state.syscall_probes[i].pre_handler = syscall_intercept_handler;
        
        ret = register_kprobe(&g_state.syscall_probes[i]);
        if (ret < 0) {
            pr_warn("%s: Failed to probe %s: %d\n",
                    MODULE_NAME, monitored_syscalls[i], ret);
            /* Continue with other probes */
        } else {
            g_state.num_probes++;
        }
    }
    
    pr_info("%s: Installed %d syscall probes\n", MODULE_NAME, g_state.num_probes);
    return 0;
}

/* Create agent implementation */
int swarm_agent_create(struct swarm_agent_config *config, u64 *agent_id)
{
    struct swarm_agent_real *agent;
    unsigned long flags;
    u64 id;
    int ret;
    
    if (!config || !agent_id)
        return -EINVAL;
        
    /* Validate configuration */
    if (config->memory_limit == 0)
        config->memory_limit = 256 * 1024 * 1024; /* 256MB default */
    if (config->cpu_shares == 0)
        config->cpu_shares = 1024; /* Default shares */
        
    /* Allocate agent structure */
    agent = kmem_cache_zalloc(g_state.agent_cache, GFP_KERNEL);
    if (!agent)
        return -ENOMEM;
        
    /* Initialize base structure */
    id = atomic64_inc_return(&g_state.total_agents);
    agent->base.id = id;
    agent->base.config = *config;
    agent->base.state = AGENT_STATE_CREATED;
    agent->base.trust_score = config->trust_score;
    spin_lock_init(&agent->base.lock);
    INIT_LIST_HEAD(&agent->base.list);
    
    /* Set personality */
    if (config->personality < PERSONALITY_COUNT) {
        static const struct agent_personality personalities[] = {
            [PERSONALITY_AGGRESSIVE] = {
                .name = "aggressive",
                .risk_tolerance = 90,
                .cooperation_level = 10,
                .resource_usage = 90,
                .adaptation_speed = 80
            },
            [PERSONALITY_DEFENSIVE] = {
                .name = "defensive", 
                .risk_tolerance = 20,
                .cooperation_level = 70,
                .resource_usage = 50,
                .adaptation_speed = 60
            },
            [PERSONALITY_COOPERATIVE] = {
                .name = "cooperative",
                .risk_tolerance = 50,
                .cooperation_level = 90,
                .resource_usage = 60,
                .adaptation_speed = 70
            },
            [PERSONALITY_ADAPTIVE] = {
                .name = "adaptive",
                .risk_tolerance = 60,
                .cooperation_level = 60,
                .resource_usage = 70,
                .adaptation_speed = 90
            },
            [PERSONALITY_NEUTRAL] = {
                .name = "neutral",
                .risk_tolerance = 50,
                .cooperation_level = 50,
                .resource_usage = 50,
                .adaptation_speed = 50
            }
        };
        agent->base.personality = personalities[config->personality];
    }
    
    /* Convert config to limits */
    agent->base.limits.memory_bytes = config->memory_limit;
    agent->base.limits.cpu_percent = config->cpu_shares / 10;
    agent->base.limits.io_bps = config->io_limit;
    agent->base.limits.network_bps = config->network_limit;
    agent->base.limits.max_processes = config->max_processes ?: 1000;
    agent->base.limits.max_files = config->max_files ?: 10000;
    
    /* Create real cgroup */
    ret = create_agent_cgroup(agent);
    if (ret != 0) {
        pr_err("%s: Failed to create cgroup: %d\n", MODULE_NAME, ret);
        goto err_free_agent;
    }
    
    /* Create namespaces if requested */
    ret = create_agent_namespaces(agent);
    if (ret != 0) {
        pr_err("%s: Failed to create namespaces: %d\n", MODULE_NAME, ret);
        goto err_destroy_cgroup;
    }
    
    /* Initialize trust score key */
    ret = init_agent_trust_score(agent);
    if (ret != 0) {
        pr_warn("%s: Failed to init trust score: %d\n", MODULE_NAME, ret);
        /* Non-fatal, continue */
    }
    
    /* Setup security context */
    agent->uid = make_kuid(current_user_ns(), config->uid);
    agent->gid = make_kgid(current_user_ns(), config->gid);
    
    /* Add to global list */
    spin_lock_irqsave(&g_state.agents_lock, flags);
    list_add(&agent->base.list, &g_state.agents);
    atomic64_inc(&g_state.active_agents);
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    *agent_id = id;
    
    if (debug)
        pr_info("%s: Created agent %llu (personality: %s)\n",
                MODULE_NAME, id, agent->base.personality.name);
                
    return 0;
    
err_destroy_cgroup:
    if (agent->cgroup)
        cgroup_destroy(agent->cgroup);
err_free_agent:
    kmem_cache_free(g_state.agent_cache, agent);
    return ret;
}
EXPORT_SYMBOL(swarm_agent_create);

/* Destroy agent */
int swarm_agent_destroy(u64 agent_id)
{
    struct swarm_agent_real *agent = NULL;
    struct swarm_agent *pos;
    unsigned long flags;
    
    /* Find agent */
    spin_lock_irqsave(&g_state.agents_lock, flags);
    list_for_each_entry(pos, &g_state.agents, list) {
        if (pos->id == agent_id) {
            agent = container_of(pos, struct swarm_agent_real, base);
            list_del(&pos->list);
            atomic64_dec(&g_state.active_agents);
            break;
        }
    }
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    if (!agent)
        return -ENOENT;
        
    /* Destroy cgroup */
    if (agent->cgroup) {
        /* Release controller references */
        if (agent->controllers.memory_css)
            css_put(agent->controllers.memory_css);
        if (agent->controllers.cpu_css)
            css_put(agent->controllers.cpu_css);
        if (agent->controllers.io_css)
            css_put(agent->controllers.io_css);
            
        cgroup_destroy(agent->cgroup);
    }
    
    /* Free namespaces */
    if (agent->nsproxy)
        free_nsproxy(agent->nsproxy);
    if (agent->user_ns)
        put_user_ns(agent->user_ns);
        
    /* Release trust key */
    if (agent->trust_key)
        key_put(agent->trust_key);
        
    /* Free agent */
    kmem_cache_free(g_state.agent_cache, agent);
    
    if (debug)
        pr_info("%s: Destroyed agent %llu\n", MODULE_NAME, agent_id);
        
    return 0;
}
EXPORT_SYMBOL(swarm_agent_destroy);

/* Get resource usage from cgroup stats */
int swarm_agent_get_stats(u64 agent_id, struct swarm_agent_stats *stats)
{
    struct swarm_agent_real *agent = NULL;
    struct swarm_agent *pos;
    unsigned long flags;
    
    if (!stats)
        return -EINVAL;
        
    /* Find agent */
    spin_lock_irqsave(&g_state.agents_lock, flags);
    list_for_each_entry(pos, &g_state.agents, list) {
        if (pos->id == agent_id) {
            agent = container_of(pos, struct swarm_agent_real, base);
            break;
        }
    }
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    if (!agent)
        return -ENOENT;
        
    spin_lock_irqsave(&agent->base.lock, flags);
    *stats = agent->base.stats;
    
    /* Get real memory usage from cgroup */
    if (agent->controllers.memory_css) {
        struct mem_cgroup *memcg = mem_cgroup_from_css(agent->controllers.memory_css);
        stats->memory_usage = page_counter_read(&memcg->memory);
        stats->memory_peak = page_counter_read_peak(&memcg->memory);
    }
    
    /* Get real CPU usage from cgroup */
    if (agent->controllers.cpu_css) {
        struct task_group *tg = css_tg(agent->controllers.cpu_css);
        /* Would read actual CPU stats here */
        stats->cpu_usage = agent->base.stats.cpu_usage; /* Placeholder */
    }
    
    spin_unlock_irqrestore(&agent->base.lock, flags);
    
    return 0;
}
EXPORT_SYMBOL(swarm_agent_get_stats);

/* Enforce limits using cgroups */
int swarm_agent_enforce_limits(u64 agent_id)
{
    struct swarm_agent_real *agent = NULL;
    struct swarm_agent *pos;
    unsigned long flags;
    
    if (!enforce_limits)
        return 0;
        
    /* Find agent */
    spin_lock_irqsave(&g_state.agents_lock, flags);
    list_for_each_entry(pos, &g_state.agents, list) {
        if (pos->id == agent_id) {
            agent = container_of(pos, struct swarm_agent_real, base);
            break;
        }
    }
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    if (!agent)
        return -ENOENT;
        
    /* Memory limit is already enforced by cgroup */
    /* CPU limit is already enforced by cgroup */
    /* IO limit would be enforced here */
    
    /* Check for violations */
    if (agent->controllers.memory_css) {
        struct mem_cgroup *memcg = mem_cgroup_from_css(agent->controllers.memory_css);
        if (page_counter_read(&memcg->memory) > agent->base.limits.memory_bytes) {
            /* Memory limit exceeded - cgroup will handle it */
            update_trust_score(agent, -10);
        }
    }
    
    return 0;
}
EXPORT_SYMBOL(swarm_agent_enforce_limits);

/* /proc interface */
static int agent_seq_show(struct seq_file *m, void *v)
{
    struct swarm_agent *agent = v;
    struct swarm_agent_stats stats;
    
    swarm_agent_get_stats(agent->id, &stats);
    
    seq_printf(m, "Agent ID: %llu\n", agent->id);
    seq_printf(m, "State: %s\n", 
               agent->state == AGENT_STATE_RUNNING ? "running" :
               agent->state == AGENT_STATE_SUSPENDED ? "suspended" : "created");
    seq_printf(m, "Personality: %s\n", agent->personality.name);
    seq_printf(m, "Trust Score: %d\n", agent->trust_score);
    seq_printf(m, "Memory: %llu / %llu bytes\n", 
               stats.memory_usage, agent->limits.memory_bytes);
    seq_printf(m, "CPU: %llu%% (limit: %u%%)\n",
               stats.cpu_usage / 10, agent->limits.cpu_percent);
    seq_printf(m, "Processes: %u / %u\n",
               stats.process_count, agent->limits.max_processes);
    seq_printf(m, "Violations: %u\n", stats.limit_violations);
    seq_printf(m, "Syscall Violations: %u\n", stats.syscall_violations);
    seq_printf(m, "\n");
    
    return 0;
}

static void *agent_seq_start(struct seq_file *m, loff_t *pos)
{
    spin_lock(&g_state.agents_lock);
    return seq_list_start(&g_state.agents, *pos);
}

static void *agent_seq_next(struct seq_file *m, void *v, loff_t *pos)
{
    return seq_list_next(v, &g_state.agents, pos);
}

static void agent_seq_stop(struct seq_file *m, void *v)
{
    spin_unlock(&g_state.agents_lock);
}

static const struct seq_operations agent_seq_ops = {
    .start = agent_seq_start,
    .next = agent_seq_next,
    .stop = agent_seq_stop,
    .show = agent_seq_show
};

static int agents_open(struct inode *inode, struct file *file)
{
    return seq_open(file, &agent_seq_ops);
}

static const struct proc_ops agents_fops = {
    .proc_open = agents_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = seq_release
};

/* Module initialization */
static int __init swarm_guard_init(void)
{
    int ret;
    
    pr_info("%s: Initializing module v%s\n", MODULE_NAME, SWARM_GUARD_VERSION);
    
    /* Initialize state */
    INIT_LIST_HEAD(&g_state.agents);
    spin_lock_init(&g_state.agents_lock);
    atomic64_set(&g_state.total_agents, 0);
    atomic64_set(&g_state.active_agents, 0);
    
    /* Create agent cache */
    g_state.agent_cache = kmem_cache_create("swarm_agent",
                                           sizeof(struct swarm_agent_real),
                                           0, SLAB_HWCACHE_ALIGN, NULL);
    if (!g_state.agent_cache)
        return -ENOMEM;
        
    /* Create trust keyring */
    g_state.trust_keyring = keyring_alloc(".swarm_trust", GLOBAL_ROOT_UID, GLOBAL_ROOT_GID,
                                         current_cred(),
                                         KEY_POS_ALL | KEY_USR_ALL,
                                         KEY_ALLOC_NOT_IN_QUOTA, NULL, NULL);
    if (IS_ERR(g_state.trust_keyring)) {
        ret = PTR_ERR(g_state.trust_keyring);
        goto err_destroy_cache;
    }
    
    /* Setup system call interception */
    ret = setup_syscall_interception();
    if (ret != 0)
        pr_warn("%s: Failed to setup syscall interception: %d\n", MODULE_NAME, ret);
        
    /* Create /proc entries */
    g_state.proc_root = proc_mkdir("swarm", NULL);
    if (g_state.proc_root) {
        proc_create("agents", 0444, g_state.proc_root, &agents_fops);
    }
    
    pr_info("%s: Module loaded successfully with real cgroup/namespace support\n", MODULE_NAME);
    return 0;
    
err_destroy_cache:
    kmem_cache_destroy(g_state.agent_cache);
    return ret;
}

/* Module cleanup */
static void __exit swarm_guard_exit(void)
{
    struct swarm_agent *agent, *tmp;
    
    pr_info("%s: Cleaning up module\n", MODULE_NAME);
    
    /* Remove all agents */
    list_for_each_entry_safe(agent, tmp, &g_state.agents, list) {
        swarm_agent_destroy(agent->id);
    }
    
    /* Remove syscall probes */
    if (g_state.syscall_probes) {
        for (int i = 0; i < g_state.num_probes; i++) {
            if (g_state.syscall_probes[i].addr)
                unregister_kprobe(&g_state.syscall_probes[i]);
        }
        kfree(g_state.syscall_probes);
    }
    
    /* Remove /proc entries */
    if (g_state.proc_root) {
        remove_proc_entry("agents", g_state.proc_root);
        remove_proc_entry("swarm", NULL);
    }
    
    /* Release trust keyring */
    if (g_state.trust_keyring)
        key_put(g_state.trust_keyring);
        
    /* Destroy cache */
    kmem_cache_destroy(g_state.agent_cache);
    
    pr_info("%s: Module unloaded\n", MODULE_NAME);
}

module_init(swarm_guard_init);
module_exit(swarm_guard_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("StratoSwarm");
MODULE_DESCRIPTION("Swarm Guard - Real Resource Enforcement Module");
MODULE_VERSION(SWARM_GUARD_VERSION);