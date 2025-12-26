/*
 * SwarmGuard Kernel Module
 * 
 * Resource enforcement and namespace management for StratoSwarm agent containers
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
#include <linux/sched.h>
#include <linux/sched/signal.h>
#include <linux/pid_namespace.h>
#include <linux/nsproxy.h>
#include <linux/cgroup.h>
#include <linux/capability.h>
#include <linux/security.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/syscalls.h>
#include <linux/tracepoint.h>
#include <linux/ftrace.h>
#include <linux/uaccess.h>

#include "swarm_guard.h"

#define MODULE_NAME "swarm_guard"

/* Module parameters */
static int debug = 0;
module_param(debug, int, 0644);
MODULE_PARM_DESC(debug, "Enable debug logging (0=off, 1=on)");

static int max_agents = SWARM_MAX_AGENTS;
module_param(max_agents, int, 0444);
MODULE_PARM_DESC(max_agents, "Maximum number of agents");

/* Global state */
static struct {
    struct list_head agents;
    struct rb_root agents_tree;
    spinlock_t agents_lock;
    
    struct swarm_guard_stats stats;
    
    /* Syscall interception */
    bool syscall_intercept_enabled;
    struct ftrace_ops syscall_ftrace_ops;
    
    /* /proc entries */
    struct proc_dir_entry *proc_root;
    struct proc_dir_entry *proc_agents;
} g_state;

/* Helper: Find agent by ID */
static struct swarm_agent *find_agent(u64 agent_id)
{
    struct rb_node *node = g_state.agents_tree.rb_node;
    
    while (node) {
        struct swarm_agent *agent = rb_entry(node, struct swarm_agent, rb_node);
        
        if (agent_id < agent->id)
            node = node->rb_left;
        else if (agent_id > agent->id)
            node = node->rb_right;
        else
            return agent;
    }
    
    return NULL;
}

/* Helper: Insert agent into RB tree */
static void insert_agent(struct swarm_agent *agent)
{
    struct rb_node **new = &g_state.agents_tree.rb_node;
    struct rb_node *parent = NULL;
    
    while (*new) {
        struct swarm_agent *this = rb_entry(*new, struct swarm_agent, rb_node);
        parent = *new;
        
        if (agent->id < this->id)
            new = &((*new)->rb_left);
        else
            new = &((*new)->rb_right);
    }
    
    rb_link_node(&agent->rb_node, parent, new);
    rb_insert_color(&agent->rb_node, &g_state.agents_tree);
}

/* Generate unique agent ID */
static u64 generate_agent_id(void)
{
    static atomic64_t counter = ATOMIC64_INIT(1);
    return atomic64_inc_return(&counter);
}

/* Initialize agent with defaults */
static void init_agent_defaults(struct swarm_agent *agent)
{
    /* Personality defaults */
    agent->personality.risk_tolerance = 0.5f;
    agent->personality.cooperation = 0.7f;
    agent->personality.exploration = 0.3f;
    agent->personality.efficiency_focus = 0.6f;
    agent->personality.stability_preference = 0.8f;
    
    /* Trust score */
    agent->trust_score = DEFAULT_TRUST_SCORE;
    
    /* Resource limits */
    agent->memory_limit = 256 * 1024 * 1024; /* 256MB */
    agent->cpu_quota = 25; /* 25% */
    agent->gpu_memory_limit = 0;
    agent->max_fds = 1024;
    agent->network_bps = 10 * 1024 * 1024; /* 10MB/s */
    
    /* Security defaults */
    agent->security.capabilities = 0;
    agent->security.no_new_privs = 1;
    strcpy(agent->security.seccomp_profile, "default");
    agent->security.device_rule_count = 0;
    
    /* Syscall policy - all allowed by default */
    memset(&agent->syscall_policy, 0, sizeof(agent->syscall_policy));
    agent->syscall_policy.default_action = SWARM_SYSCALL_ALLOW;
    
    /* State */
    agent->state = SWARM_AGENT_CREATED;
    agent->violation_action = SWARM_VIOLATION_LOG;
    
    /* Lists */
    INIT_LIST_HEAD(&agent->children);
    INIT_LIST_HEAD(&agent->sibling);
    INIT_LIST_HEAD(&agent->list);
    
    /* Timestamps */
    agent->created_at = ktime_get_ns();
    agent->last_active = 0;
    
    spin_lock_init(&agent->lock);
}

/* Create new agent */
u64 swarm_agent_create(struct swarm_agent_config *config)
{
    struct swarm_agent *agent;
    unsigned long flags;
    
    if (!config || strlen(config->name) == 0)
        return 0;
        
    /* Check agent limit */
    if (atomic64_read(&g_state.stats.active_agents) >= max_agents)
        return 0;
        
    agent = kzalloc(sizeof(*agent), GFP_KERNEL);
    if (!agent)
        return 0;
        
    /* Initialize agent */
    agent->id = generate_agent_id();
    strncpy(agent->name, config->name, MAX_AGENT_NAME - 1);
    init_agent_defaults(agent);
    
    /* Apply config */
    if (config->memory_limit > 0)
        agent->memory_limit = config->memory_limit;
    if (config->cpu_quota > 0 && config->cpu_quota <= 100)
        agent->cpu_quota = config->cpu_quota;
    if (config->gpu_memory_limit > 0)
        agent->gpu_memory_limit = config->gpu_memory_limit;
    
    agent->namespace_flags = config->namespace_flags;
    
    /* Copy personality if provided */
    if (config->personality.risk_tolerance >= 0.0f)
        agent->personality = config->personality;
        
    /* Add to global list */
    spin_lock_irqsave(&g_state.agents_lock, flags);
    list_add(&agent->list, &g_state.agents);
    insert_agent(agent);
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    /* Update stats */
    atomic64_inc(&g_state.stats.total_agents_created);
    atomic64_inc(&g_state.stats.active_agents);
    
    if (debug)
        pr_info("%s: Created agent %llu (%s)\n", MODULE_NAME, agent->id, agent->name);
        
    return agent->id;
}
EXPORT_SYMBOL(swarm_agent_create);

/* Create child agent */
u64 swarm_agent_create_child(struct swarm_agent_config *config)
{
    struct swarm_agent *parent, *child;
    unsigned long flags;
    u64 child_id;
    
    if (!config || !config->parent_id)
        return 0;
        
    spin_lock_irqsave(&g_state.agents_lock, flags);
    parent = find_agent(config->parent_id);
    if (!parent) {
        spin_unlock_irqrestore(&g_state.agents_lock, flags);
        return 0;
    }
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    /* Create child with inherited limits */
    child_id = swarm_agent_create(config);
    if (!child_id)
        return 0;
        
    spin_lock_irqsave(&g_state.agents_lock, flags);
    child = find_agent(child_id);
    if (child) {
        child->parent_id = config->parent_id;
        
        /* Inherit parent's limits if not specified */
        if (config->memory_limit == 0)
            child->memory_limit = parent->memory_limit;
        if (config->cpu_quota == 0)
            child->cpu_quota = parent->cpu_quota;
            
        /* Add to parent's children list */
        spin_lock(&parent->lock);
        list_add(&child->sibling, &parent->children);
        spin_unlock(&parent->lock);
    }
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    return child_id;
}
EXPORT_SYMBOL(swarm_agent_create_child);

/* Destroy agent */
int swarm_agent_destroy(u64 agent_id)
{
    struct swarm_agent *agent, *child, *tmp;
    unsigned long flags;
    
    spin_lock_irqsave(&g_state.agents_lock, flags);
    agent = find_agent(agent_id);
    if (!agent) {
        spin_unlock_irqrestore(&g_state.agents_lock, flags);
        return -ENOENT;
    }
    
    /* Remove from trees and lists */
    rb_erase(&agent->rb_node, &g_state.agents_tree);
    list_del(&agent->list);
    
    /* Remove from parent's children list */
    if (agent->parent_id) {
        struct swarm_agent *parent = find_agent(agent->parent_id);
        if (parent) {
            spin_lock(&parent->lock);
            list_del(&agent->sibling);
            spin_unlock(&parent->lock);
        }
    }
    
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    /* Destroy all children */
    list_for_each_entry_safe(child, tmp, &agent->children, sibling) {
        swarm_agent_destroy(child->id);
    }
    
    /* Cleanup cgroups if active */
    if (agent->state == SWARM_AGENT_ACTIVE) {
        swarm_cgroup_teardown(agent_id);
    }
    
    /* Update stats */
    atomic64_inc(&g_state.stats.total_agents_destroyed);
    atomic64_dec(&g_state.stats.active_agents);
    
    if (debug)
        pr_info("%s: Destroyed agent %llu (%s)\n", MODULE_NAME, agent->id, agent->name);
        
    kfree(agent);
    return 0;
}
EXPORT_SYMBOL(swarm_agent_destroy);

/* Query agent status */
int swarm_agent_query(struct swarm_agent_query *query)
{
    struct swarm_agent *agent;
    unsigned long flags;
    
    if (!query)
        return -EINVAL;
        
    spin_lock_irqsave(&g_state.agents_lock, flags);
    agent = find_agent(query->agent_id);
    if (!agent) {
        spin_unlock_irqrestore(&g_state.agents_lock, flags);
        return -ENOENT;
    }
    
    /* Fill query results */
    spin_lock(&agent->lock);
    query->pid = agent->pid;
    query->memory_usage = 0; /* Would get from cgroups */
    query->gpu_memory_usage = 0; /* Would get from GPU module */
    query->violation_count = agent->violations.total_violations;
    query->state = agent->state;
    query->memory_limit = agent->memory_limit;
    spin_unlock(&agent->lock);
    
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    return 0;
}
EXPORT_SYMBOL(swarm_agent_query);

/* Activate agent with process */
int swarm_agent_activate(u64 agent_id, pid_t pid)
{
    struct swarm_agent *agent;
    unsigned long flags;
    int ret = 0;
    
    spin_lock_irqsave(&g_state.agents_lock, flags);
    agent = find_agent(agent_id);
    if (!agent) {
        spin_unlock_irqrestore(&g_state.agents_lock, flags);
        return -ENOENT;
    }
    
    spin_lock(&agent->lock);
    
    if (agent->state != SWARM_AGENT_CREATED) {
        ret = -EBUSY;
        goto out;
    }
    
    agent->pid = pid;
    agent->tgid = current->tgid;
    agent->state = SWARM_AGENT_ACTIVE;
    agent->last_active = ktime_get_ns();
    
    /* Set up namespaces if requested */
    if (agent->namespace_flags) {
        /* Would set up namespaces here */
    }
    
    /* Set up cgroups */
    ret = swarm_cgroup_setup(agent_id);
    if (ret != 0) {
        agent->state = SWARM_AGENT_CREATED;
        agent->pid = 0;
    }
    
out:
    spin_unlock(&agent->lock);
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    if (debug && ret == 0)
        pr_info("%s: Activated agent %llu with PID %d\n", MODULE_NAME, agent_id, pid);
        
    return ret;
}
EXPORT_SYMBOL(swarm_agent_activate);

/* Get agent personality */
int swarm_agent_get_personality(u64 agent_id, struct swarm_personality *personality)
{
    struct swarm_agent *agent;
    unsigned long flags;
    
    if (!personality)
        return -EINVAL;
        
    spin_lock_irqsave(&g_state.agents_lock, flags);
    agent = find_agent(agent_id);
    if (!agent) {
        spin_unlock_irqrestore(&g_state.agents_lock, flags);
        return -ENOENT;
    }
    
    spin_lock(&agent->lock);
    *personality = agent->personality;
    spin_unlock(&agent->lock);
    
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    return 0;
}
EXPORT_SYMBOL(swarm_agent_get_personality);

/* Update personality based on behavior */
int swarm_agent_update_personality(u64 agent_id, enum swarm_behavior_type behavior, float intensity)
{
    struct swarm_agent *agent;
    unsigned long flags;
    
    if (intensity < 0.0f || intensity > 1.0f)
        return -EINVAL;
        
    spin_lock_irqsave(&g_state.agents_lock, flags);
    agent = find_agent(agent_id);
    if (!agent) {
        spin_unlock_irqrestore(&g_state.agents_lock, flags);
        return -ENOENT;
    }
    
    spin_lock(&agent->lock);
    
    /* Update personality based on behavior */
    switch (behavior) {
    case SWARM_BEHAVIOR_AGGRESSIVE:
        agent->personality.risk_tolerance = 
            (agent->personality.risk_tolerance + intensity) / 2.0f;
        break;
    case SWARM_BEHAVIOR_COOPERATIVE:
        agent->personality.cooperation = 
            (agent->personality.cooperation + intensity) / 2.0f;
        break;
    case SWARM_BEHAVIOR_EXPLORATORY:
        agent->personality.exploration = 
            (agent->personality.exploration + intensity) / 2.0f;
        break;
    case SWARM_BEHAVIOR_CONSERVATIVE:
        agent->personality.risk_tolerance = 
            (agent->personality.risk_tolerance + (1.0f - intensity)) / 2.0f;
        break;
    case SWARM_BEHAVIOR_EFFICIENT:
        agent->personality.efficiency_focus = 
            (agent->personality.efficiency_focus + intensity) / 2.0f;
        break;
    }
    
    /* Clamp values to [0, 1] */
    #define CLAMP(x) ((x) < 0.0f ? 0.0f : ((x) > 1.0f ? 1.0f : (x)))
    agent->personality.risk_tolerance = CLAMP(agent->personality.risk_tolerance);
    agent->personality.cooperation = CLAMP(agent->personality.cooperation);
    agent->personality.exploration = CLAMP(agent->personality.exploration);
    agent->personality.efficiency_focus = CLAMP(agent->personality.efficiency_focus);
    agent->personality.stability_preference = CLAMP(agent->personality.stability_preference);
    #undef CLAMP
    
    spin_unlock(&agent->lock);
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    return 0;
}
EXPORT_SYMBOL(swarm_agent_update_personality);

/* Get trust score */
int swarm_agent_get_trust_score(u64 agent_id, float *score)
{
    struct swarm_agent *agent;
    unsigned long flags;
    
    if (!score)
        return -EINVAL;
        
    spin_lock_irqsave(&g_state.agents_lock, flags);
    agent = find_agent(agent_id);
    if (!agent) {
        spin_unlock_irqrestore(&g_state.agents_lock, flags);
        return -ENOENT;
    }
    
    spin_lock(&agent->lock);
    *score = agent->trust_score;
    spin_unlock(&agent->lock);
    
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    return 0;
}
EXPORT_SYMBOL(swarm_agent_get_trust_score);

/* Update trust score */
int swarm_agent_update_trust(u64 agent_id, float delta)
{
    struct swarm_agent *agent;
    unsigned long flags;
    
    spin_lock_irqsave(&g_state.agents_lock, flags);
    agent = find_agent(agent_id);
    if (!agent) {
        spin_unlock_irqrestore(&g_state.agents_lock, flags);
        return -ENOENT;
    }
    
    spin_lock(&agent->lock);
    agent->trust_score += delta;
    
    /* Clamp to [0, 1] */
    if (agent->trust_score < 0.0f)
        agent->trust_score = 0.0f;
    else if (agent->trust_score > 1.0f)
        agent->trust_score = 1.0f;
        
    spin_unlock(&agent->lock);
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    return 0;
}
EXPORT_SYMBOL(swarm_agent_update_trust);

/* Get namespace info */
int swarm_agent_get_namespaces(u64 agent_id, struct swarm_namespace_info *info)
{
    struct swarm_agent *agent;
    unsigned long flags;
    
    if (!info)
        return -EINVAL;
        
    spin_lock_irqsave(&g_state.agents_lock, flags);
    agent = find_agent(agent_id);
    if (!agent) {
        spin_unlock_irqrestore(&g_state.agents_lock, flags);
        return -ENOENT;
    }
    
    spin_lock(&agent->lock);
    *info = agent->namespaces;
    info->active_namespaces = agent->namespace_flags;
    spin_unlock(&agent->lock);
    
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    return 0;
}
EXPORT_SYMBOL(swarm_agent_get_namespaces);

/* Cgroup setup */
int swarm_cgroup_setup(u64 agent_id)
{
    struct swarm_agent *agent;
    unsigned long flags;
    
    spin_lock_irqsave(&g_state.agents_lock, flags);
    agent = find_agent(agent_id);
    if (!agent) {
        spin_unlock_irqrestore(&g_state.agents_lock, flags);
        return -ENOENT;
    }
    
    /* In real implementation, would create cgroups here */
    /* For now, just mark as successful */
    
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    if (debug)
        pr_info("%s: Set up cgroups for agent %llu\n", MODULE_NAME, agent_id);
        
    return 0;
}
EXPORT_SYMBOL(swarm_cgroup_setup);

/* Cgroup teardown */
int swarm_cgroup_teardown(u64 agent_id)
{
    struct swarm_agent *agent;
    unsigned long flags;
    
    spin_lock_irqsave(&g_state.agents_lock, flags);
    agent = find_agent(agent_id);
    if (!agent) {
        spin_unlock_irqrestore(&g_state.agents_lock, flags);
        return -ENOENT;
    }
    
    /* In real implementation, would remove cgroups here */
    
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    return 0;
}
EXPORT_SYMBOL(swarm_cgroup_teardown);

/* Get cgroup stats */
int swarm_cgroup_get_stats(u64 agent_id, struct swarm_cgroup_stats *stats)
{
    struct swarm_agent *agent;
    unsigned long flags;
    
    if (!stats)
        return -EINVAL;
        
    spin_lock_irqsave(&g_state.agents_lock, flags);
    agent = find_agent(agent_id);
    if (!agent) {
        spin_unlock_irqrestore(&g_state.agents_lock, flags);
        return -ENOENT;
    }
    
    /* Fill mock stats */
    spin_lock(&agent->lock);
    stats->memory_limit = agent->memory_limit;
    stats->memory_usage = 0; /* Would get from cgroups */
    stats->memory_max_usage = 0;
    stats->cpu_quota = agent->cpu_quota;
    stats->cpu_usage_ns = 0;
    stats->io_read_bytes = 0;
    stats->io_write_bytes = 0;
    spin_unlock(&agent->lock);
    
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    return 0;
}
EXPORT_SYMBOL(swarm_cgroup_get_stats);

/* Notify memory pressure */
int swarm_cgroup_notify_pressure(u64 agent_id, enum swarm_pressure_level level)
{
    struct swarm_agent *agent;
    unsigned long flags;
    
    spin_lock_irqsave(&g_state.agents_lock, flags);
    agent = find_agent(agent_id);
    if (!agent) {
        spin_unlock_irqrestore(&g_state.agents_lock, flags);
        return -ENOENT;
    }
    
    /* Handle pressure notification */
    if (level >= SWARM_PRESSURE_HIGH) {
        /* Could trigger OOM killer or other actions */
        if (debug)
            pr_info("%s: High memory pressure for agent %llu\n", MODULE_NAME, agent_id);
    }
    
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    return 0;
}
EXPORT_SYMBOL(swarm_cgroup_notify_pressure);

/* Set syscall policy */
int swarm_agent_set_syscall_policy(u64 agent_id, struct swarm_syscall_policy *policy)
{
    struct swarm_agent *agent;
    unsigned long flags;
    
    if (!policy)
        return -EINVAL;
        
    spin_lock_irqsave(&g_state.agents_lock, flags);
    agent = find_agent(agent_id);
    if (!agent) {
        spin_unlock_irqrestore(&g_state.agents_lock, flags);
        return -ENOENT;
    }
    
    spin_lock(&agent->lock);
    agent->syscall_policy = *policy;
    spin_unlock(&agent->lock);
    
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    return 0;
}
EXPORT_SYMBOL(swarm_agent_set_syscall_policy);

/* Check if syscall is allowed */
int swarm_syscall_check(u64 agent_id, int syscall_nr)
{
    struct swarm_agent *agent;
    unsigned long flags;
    int allowed;
    ktime_t start_time;
    
    if (syscall_nr < 0 || syscall_nr >= MAX_SYSCALLS)
        return -EINVAL;
        
    start_time = ktime_get();
    
    spin_lock_irqsave(&g_state.agents_lock, flags);
    agent = find_agent(agent_id);
    if (!agent) {
        spin_unlock_irqrestore(&g_state.agents_lock, flags);
        return -ENOENT;
    }
    
    spin_lock(&agent->lock);
    
    /* Check bitmap */
    allowed = agent->syscall_policy.allowed_syscalls[syscall_nr / 64] & 
              (1ULL << (syscall_nr % 64));
    
    if (!allowed && agent->syscall_policy.default_action == SWARM_SYSCALL_ALLOW)
        allowed = 1;
    else if (allowed && agent->syscall_policy.default_action == SWARM_SYSCALL_DENY)
        allowed = 0;
        
    spin_unlock(&agent->lock);
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    /* Update stats */
    atomic64_inc(&g_state.stats.syscalls_intercepted);
    if (!allowed)
        atomic64_inc(&g_state.stats.syscalls_denied);
        
    if (debug) {
        u64 duration_ns = ktime_to_ns(ktime_sub(ktime_get(), start_time));
        if (duration_ns > SWARM_TARGET_SYSCALL_NS)
            pr_warn("%s: Syscall check took %llu ns (target: %d ns)\n",
                    MODULE_NAME, duration_ns, SWARM_TARGET_SYSCALL_NS);
    }
    
    return allowed ? 0 : -EPERM;
}
EXPORT_SYMBOL(swarm_syscall_check);

/* Set security policy */
int swarm_agent_set_security_policy(u64 agent_id, struct swarm_security_policy *policy)
{
    struct swarm_agent *agent;
    unsigned long flags;
    
    if (!policy)
        return -EINVAL;
        
    spin_lock_irqsave(&g_state.agents_lock, flags);
    agent = find_agent(agent_id);
    if (!agent) {
        spin_unlock_irqrestore(&g_state.agents_lock, flags);
        return -ENOENT;
    }
    
    spin_lock(&agent->lock);
    agent->security = *policy;
    spin_unlock(&agent->lock);
    
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    return 0;
}
EXPORT_SYMBOL(swarm_agent_set_security_policy);

/* Check capability */
int swarm_security_check_capability(u64 agent_id, int capability)
{
    struct swarm_agent *agent;
    unsigned long flags;
    int allowed;
    
    if (capability < 0 || capability >= CAP_LAST_CAP)
        return -EINVAL;
        
    spin_lock_irqsave(&g_state.agents_lock, flags);
    agent = find_agent(agent_id);
    if (!agent) {
        spin_unlock_irqrestore(&g_state.agents_lock, flags);
        return -ENOENT;
    }
    
    spin_lock(&agent->lock);
    allowed = agent->security.capabilities & (1ULL << capability);
    spin_unlock(&agent->lock);
    
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    return allowed ? 0 : -EPERM;
}
EXPORT_SYMBOL(swarm_security_check_capability);

/* Check device access */
int swarm_security_check_device(u64 agent_id, u32 major, u32 minor, u32 access)
{
    struct swarm_agent *agent;
    unsigned long flags;
    int i, allowed = 0;
    
    spin_lock_irqsave(&g_state.agents_lock, flags);
    agent = find_agent(agent_id);
    if (!agent) {
        spin_unlock_irqrestore(&g_state.agents_lock, flags);
        return -ENOENT;
    }
    
    spin_lock(&agent->lock);
    
    for (i = 0; i < agent->security.device_rule_count; i++) {
        struct swarm_device_rule *rule = &agent->security.device_rules[i];
        if (rule->major == major && rule->minor == minor) {
            if ((rule->access & access) == access) {
                allowed = 1;
                break;
            }
        }
    }
    
    spin_unlock(&agent->lock);
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    return allowed ? 0 : -EPERM;
}
EXPORT_SYMBOL(swarm_security_check_device);

/* Record violation */
int swarm_agent_record_violation(u64 agent_id, enum swarm_violation_type type)
{
    struct swarm_agent *agent;
    unsigned long flags;
    
    spin_lock_irqsave(&g_state.agents_lock, flags);
    agent = find_agent(agent_id);
    if (!agent) {
        spin_unlock_irqrestore(&g_state.agents_lock, flags);
        return -ENOENT;
    }
    
    spin_lock(&agent->lock);
    
    switch (type) {
    case SWARM_VIOLATION_MEMORY:
        agent->violations.memory_violations++;
        break;
    case SWARM_VIOLATION_CPU:
        agent->violations.cpu_violations++;
        break;
    case SWARM_VIOLATION_SYSCALL:
        agent->violations.syscall_violations++;
        break;
    case SWARM_VIOLATION_DEVICE:
        agent->violations.device_violations++;
        break;
    case SWARM_VIOLATION_NETWORK:
        agent->violations.network_violations++;
        break;
    case SWARM_VIOLATION_CAPABILITY:
        agent->violations.capability_violations++;
        break;
    }
    
    agent->violations.total_violations++;
    
    /* Take action based on violation handler */
    switch (agent->violation_action) {
    case SWARM_VIOLATION_KILL:
        /* Would send SIGKILL to process */
        if (debug)
            pr_info("%s: Killing agent %llu due to violation\n", MODULE_NAME, agent_id);
        break;
    case SWARM_VIOLATION_SUSPEND:
        agent->state = SWARM_AGENT_SUSPENDED;
        break;
    case SWARM_VIOLATION_THROTTLE:
        /* Would reduce resource limits */
        break;
    case SWARM_VIOLATION_LOG:
    default:
        /* Just log */
        break;
    }
    
    spin_unlock(&agent->lock);
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    /* Update global stats */
    atomic64_inc(&g_state.stats.total_violations);
    
    return 0;
}
EXPORT_SYMBOL(swarm_agent_record_violation);

/* Get violation stats */
int swarm_agent_get_violations(u64 agent_id, struct swarm_violation_stats *stats)
{
    struct swarm_agent *agent;
    unsigned long flags;
    
    if (!stats)
        return -EINVAL;
        
    spin_lock_irqsave(&g_state.agents_lock, flags);
    agent = find_agent(agent_id);
    if (!agent) {
        spin_unlock_irqrestore(&g_state.agents_lock, flags);
        return -ENOENT;
    }
    
    spin_lock(&agent->lock);
    *stats = agent->violations;
    spin_unlock(&agent->lock);
    
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    return 0;
}
EXPORT_SYMBOL(swarm_agent_get_violations);

/* Set violation handler */
int swarm_agent_set_violation_handler(u64 agent_id, enum swarm_violation_action action)
{
    struct swarm_agent *agent;
    unsigned long flags;
    
    spin_lock_irqsave(&g_state.agents_lock, flags);
    agent = find_agent(agent_id);
    if (!agent) {
        spin_unlock_irqrestore(&g_state.agents_lock, flags);
        return -ENOENT;
    }
    
    spin_lock(&agent->lock);
    agent->violation_action = action;
    spin_unlock(&agent->lock);
    
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    return 0;
}
EXPORT_SYMBOL(swarm_agent_set_violation_handler);

/* Get module stats */
int swarm_guard_get_stats(struct swarm_guard_stats *stats)
{
    if (!stats)
        return -EINVAL;
        
    stats->total_agents_created = atomic64_read(&g_state.stats.total_agents_created);
    stats->total_agents_destroyed = atomic64_read(&g_state.stats.total_agents_destroyed);
    stats->active_agents = atomic64_read(&g_state.stats.active_agents);
    stats->total_violations = atomic64_read(&g_state.stats.total_violations);
    stats->syscalls_intercepted = atomic64_read(&g_state.stats.syscalls_intercepted);
    stats->syscalls_denied = atomic64_read(&g_state.stats.syscalls_denied);
    
    return 0;
}
EXPORT_SYMBOL(swarm_guard_get_stats);

/* /proc interface */
static int agents_stats_show(struct seq_file *m, void *v)
{
    struct swarm_guard_stats stats;
    
    swarm_guard_get_stats(&stats);
    
    seq_printf(m, "SwarmGuard Agent Statistics\n");
    seq_printf(m, "==========================\n");
    seq_printf(m, "Total agents created: %llu\n", stats.total_agents_created);
    seq_printf(m, "Total agents destroyed: %llu\n", stats.total_agents_destroyed);
    seq_printf(m, "Active agents: %llu\n", stats.active_agents);
    seq_printf(m, "Total violations: %llu\n", stats.total_violations);
    seq_printf(m, "Syscalls intercepted: %llu\n", stats.syscalls_intercepted);
    seq_printf(m, "Syscalls denied: %llu\n", stats.syscalls_denied);
    
    return 0;
}

static int agents_stats_open(struct inode *inode, struct file *file)
{
    return single_open(file, agents_stats_show, NULL);
}

static const struct proc_ops agents_stats_fops = {
    .proc_open = agents_stats_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

static int agents_list_show(struct seq_file *m, void *v)
{
    struct swarm_agent *agent;
    unsigned long flags;
    
    seq_printf(m, "Active SwarmGuard Agents\n");
    seq_printf(m, "=======================\n");
    seq_printf(m, "ID\tPID\tName\t\tState\tTrust\tViolations\n");
    
    spin_lock_irqsave(&g_state.agents_lock, flags);
    list_for_each_entry(agent, &g_state.agents, list) {
        const char *state_str = "unknown";
        
        switch (agent->state) {
        case SWARM_AGENT_CREATED:
            state_str = "created";
            break;
        case SWARM_AGENT_ACTIVE:
            state_str = "active";
            break;
        case SWARM_AGENT_SUSPENDED:
            state_str = "suspended";
            break;
        case SWARM_AGENT_TERMINATED:
            state_str = "terminated";
            break;
        }
        
        seq_printf(m, "%llu\t%d\t%-16s%s\t%.2f\t%llu\n",
                   agent->id,
                   agent->pid,
                   agent->name,
                   state_str,
                   agent->trust_score,
                   agent->violations.total_violations);
    }
    spin_unlock_irqrestore(&g_state.agents_lock, flags);
    
    return 0;
}

static int agents_list_open(struct inode *inode, struct file *file)
{
    return single_open(file, agents_list_show, NULL);
}

static const struct proc_ops agents_list_fops = {
    .proc_open = agents_list_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

/* Module initialization */
int swarm_guard_init(void)
{
    pr_info("%s: Initializing module v%s\n", MODULE_NAME, SWARM_GUARD_VERSION);
    
    /* Initialize global state */
    INIT_LIST_HEAD(&g_state.agents);
    g_state.agents_tree = RB_ROOT;
    spin_lock_init(&g_state.agents_lock);
    
    /* Initialize stats */
    atomic64_set(&g_state.stats.total_agents_created, 0);
    atomic64_set(&g_state.stats.total_agents_destroyed, 0);
    atomic64_set(&g_state.stats.active_agents, 0);
    atomic64_set(&g_state.stats.total_violations, 0);
    atomic64_set(&g_state.stats.syscalls_intercepted, 0);
    atomic64_set(&g_state.stats.syscalls_denied, 0);
    
    /* Create /proc entries */
    g_state.proc_root = proc_mkdir("swarm", NULL);
    if (g_state.proc_root) {
        g_state.proc_agents = proc_mkdir("agents", g_state.proc_root);
        if (g_state.proc_agents) {
            proc_create("stats", 0444, g_state.proc_agents, &agents_stats_fops);
            proc_create("list", 0444, g_state.proc_agents, &agents_list_fops);
        }
    }
    
    pr_info("%s: Module loaded successfully\n", MODULE_NAME);
    return 0;
}

/* Module cleanup */
void swarm_guard_exit(void)
{
    struct swarm_agent *agent, *tmp;
    
    pr_info("%s: Cleaning up module\n", MODULE_NAME);
    
    /* Remove /proc entries */
    if (g_state.proc_agents) {
        remove_proc_entry("stats", g_state.proc_agents);
        remove_proc_entry("list", g_state.proc_agents);
        remove_proc_entry("agents", g_state.proc_root);
    }
    if (g_state.proc_root)
        remove_proc_entry("swarm", NULL);
        
    /* Destroy all agents */
    list_for_each_entry_safe(agent, tmp, &g_state.agents, list) {
        swarm_agent_destroy(agent->id);
    }
    
    pr_info("%s: Module unloaded\n", MODULE_NAME);
}

module_init(swarm_guard_init);
module_exit(swarm_guard_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("StratoSwarm");
MODULE_DESCRIPTION("SwarmGuard Resource Enforcement Module");
MODULE_VERSION(SWARM_GUARD_VERSION);