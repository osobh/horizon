/*
 * SwarmGuard Kernel Module Header
 * 
 * Resource enforcement and namespace management for StratoSwarm agent containers
 */

#ifndef _SWARM_GUARD_H
#define _SWARM_GUARD_H

#include <linux/types.h>
#include <linux/list.h>
#include <linux/spinlock.h>
#include <linux/rbtree.h>
#include <linux/cgroup.h>
#include <linux/nsproxy.h>
#include <linux/capability.h>
#include "../../common/swarm_core.h"
#include "../../common/swarm_ioctl.h"

/* Module configuration */
#define SWARM_GUARD_VERSION "2.0"
#define MAX_SYSCALLS 512
#define MAX_DEVICE_RULES 32
#define MAX_AGENT_NAME 64
#define DEFAULT_TRUST_SCORE 0.5f

/* Namespace flags */
#define SWARM_NS_PID    0x01
#define SWARM_NS_NET    0x02
#define SWARM_NS_IPC    0x04
#define SWARM_NS_UTS    0x08
#define SWARM_NS_MNT    0x10
#define SWARM_NS_USER   0x20
#define SWARM_NS_CGROUP 0x40
#define SWARM_NS_ALL    0x7F

/* Agent states */
enum swarm_agent_state {
    SWARM_AGENT_CREATED = 0,
    SWARM_AGENT_ACTIVE = 1,
    SWARM_AGENT_SUSPENDED = 2,
    SWARM_AGENT_TERMINATED = 3,
};

/* Violation types */
enum swarm_violation_type {
    SWARM_VIOLATION_MEMORY = 0,
    SWARM_VIOLATION_CPU = 1,
    SWARM_VIOLATION_SYSCALL = 2,
    SWARM_VIOLATION_DEVICE = 3,
    SWARM_VIOLATION_NETWORK = 4,
    SWARM_VIOLATION_CAPABILITY = 5,
};

/* Violation actions */
enum swarm_violation_action {
    SWARM_VIOLATION_LOG = 0,
    SWARM_VIOLATION_THROTTLE = 1,
    SWARM_VIOLATION_SUSPEND = 2,
    SWARM_VIOLATION_KILL = 3,
};

/* Syscall actions */
enum swarm_syscall_action {
    SWARM_SYSCALL_ALLOW = 0,
    SWARM_SYSCALL_DENY = 1,
    SWARM_SYSCALL_TRACE = 2,
    SWARM_SYSCALL_KILL = 3,
};

/* Device access flags */
#define SWARM_DEVICE_READ   0x01
#define SWARM_DEVICE_WRITE  0x02
#define SWARM_DEVICE_MKNOD  0x04

/* Behavior types for personality updates */
enum swarm_behavior_type {
    SWARM_BEHAVIOR_AGGRESSIVE = 0,
    SWARM_BEHAVIOR_COOPERATIVE = 1,
    SWARM_BEHAVIOR_EXPLORATORY = 2,
    SWARM_BEHAVIOR_CONSERVATIVE = 3,
    SWARM_BEHAVIOR_EFFICIENT = 4,
};

/* Pressure levels */
enum swarm_pressure_level {
    SWARM_PRESSURE_LOW = 0,
    SWARM_PRESSURE_MEDIUM = 1,
    SWARM_PRESSURE_HIGH = 2,
    SWARM_PRESSURE_CRITICAL = 3,
};

/* Agent personality structure */
struct swarm_personality {
    float risk_tolerance;       /* 0.0 = conservative, 1.0 = aggressive */
    float cooperation;         /* 0.0 = selfish, 1.0 = collaborative */
    float exploration;         /* 0.0 = exploit, 1.0 = explore */
    float efficiency_focus;    /* 0.0 = thorough, 1.0 = fast */
    float stability_preference; /* 0.0 = dynamic, 1.0 = stable */
};

/* Device access rule */
struct swarm_device_rule {
    u32 major;
    u32 minor;
    u32 access; /* SWARM_DEVICE_* flags */
};

/* Security policy */
struct swarm_security_policy {
    u64 capabilities;                                    /* Linux capabilities */
    u32 no_new_privs;                                   /* No new privileges flag */
    char seccomp_profile[64];                           /* Seccomp profile name */
    struct swarm_device_rule device_rules[MAX_DEVICE_RULES];
    u32 device_rule_count;
};

/* Syscall policy */
struct swarm_syscall_policy {
    u64 allowed_syscalls[MAX_SYSCALLS / 64];           /* Bitmap of allowed syscalls */
    enum swarm_syscall_action default_action;
};

/* Namespace information */
struct swarm_namespace_info {
    u32 active_namespaces;  /* Bitmask of SWARM_NS_* */
    struct pid_namespace *pid_ns;
    struct net *net_ns;
    struct ipc_namespace *ipc_ns;
    struct uts_namespace *uts_ns;
    struct mnt_namespace *mnt_ns;
    struct user_namespace *user_ns;
    struct cgroup_namespace *cgroup_ns;
};

/* Cgroup statistics */
struct swarm_cgroup_stats {
    u64 memory_limit;
    u64 memory_usage;
    u64 memory_max_usage;
    u32 cpu_quota;
    u64 cpu_usage_ns;
    u64 io_read_bytes;
    u64 io_write_bytes;
};

/* Violation statistics */
struct swarm_violation_stats {
    u64 memory_violations;
    u64 cpu_violations;
    u64 syscall_violations;
    u64 device_violations;
    u64 network_violations;
    u64 capability_violations;
    u64 total_violations;
};

/* Agent structure */
struct swarm_agent {
    u64 id;
    pid_t pid;
    pid_t tgid;
    char name[MAX_AGENT_NAME];
    
    /* Hierarchy */
    u64 parent_id;
    struct list_head children;
    struct list_head sibling;
    
    /* Personality and behavior */
    struct swarm_personality personality;
    float trust_score;
    
    /* Resource limits */
    u64 memory_limit;
    u32 cpu_quota;
    u64 gpu_memory_limit;
    u32 max_fds;
    u64 network_bps;
    
    /* Security */
    struct swarm_security_policy security;
    struct swarm_syscall_policy syscall_policy;
    
    /* Namespaces */
    struct swarm_namespace_info namespaces;
    u32 namespace_flags;
    
    /* Cgroups */
    struct cgroup *memory_cgrp;
    struct cgroup *cpu_cgrp;
    struct cgroup *io_cgrp;
    
    /* Statistics */
    struct swarm_violation_stats violations;
    u64 created_at;
    u64 last_active;
    
    /* State */
    enum swarm_agent_state state;
    enum swarm_violation_action violation_action;
    
    /* Management */
    struct list_head list;
    struct rb_node rb_node;
    spinlock_t lock;
    struct rcu_head rcu;
};

/* Module statistics */
struct swarm_guard_stats {
    atomic64_t total_agents_created;
    atomic64_t total_agents_destroyed;
    atomic64_t active_agents;
    atomic64_t total_violations;
    atomic64_t syscalls_intercepted;
    atomic64_t syscalls_denied;
};

/* Function prototypes */

/* Agent management */
u64 swarm_agent_create(struct swarm_agent_config *config);
u64 swarm_agent_create_child(struct swarm_agent_config *config);
int swarm_agent_destroy(u64 agent_id);
int swarm_agent_query(struct swarm_agent_query *query);
int swarm_agent_activate(u64 agent_id, pid_t pid);
int swarm_agent_suspend(u64 agent_id);
int swarm_agent_resume(u64 agent_id);

/* Personality management */
int swarm_agent_get_personality(u64 agent_id, struct swarm_personality *personality);
int swarm_agent_set_personality(u64 agent_id, struct swarm_personality *personality);
int swarm_agent_update_personality(u64 agent_id, enum swarm_behavior_type behavior, float intensity);

/* Trust management */
int swarm_agent_get_trust_score(u64 agent_id, float *score);
int swarm_agent_update_trust(u64 agent_id, float delta);

/* Namespace management */
int swarm_agent_get_namespaces(u64 agent_id, struct swarm_namespace_info *info);
int swarm_agent_set_namespace_flags(u64 agent_id, u32 flags);
int swarm_namespace_enter(u64 agent_id);
int swarm_namespace_exit(void);

/* Cgroup management */
int swarm_cgroup_setup(u64 agent_id);
int swarm_cgroup_teardown(u64 agent_id);
int swarm_cgroup_get_stats(u64 agent_id, struct swarm_cgroup_stats *stats);
int swarm_cgroup_update_limits(u64 agent_id, u64 memory_limit, u32 cpu_quota);
int swarm_cgroup_notify_pressure(u64 agent_id, enum swarm_pressure_level level);

/* Syscall interception */
int swarm_agent_set_syscall_policy(u64 agent_id, struct swarm_syscall_policy *policy);
int swarm_syscall_check(u64 agent_id, int syscall_nr);
int swarm_syscall_intercept_enable(void);
void swarm_syscall_intercept_disable(void);

/* Security policy */
int swarm_agent_set_security_policy(u64 agent_id, struct swarm_security_policy *policy);
int swarm_security_check_capability(u64 agent_id, int capability);
int swarm_security_check_device(u64 agent_id, u32 major, u32 minor, u32 access);

/* Violation handling */
int swarm_agent_record_violation(u64 agent_id, enum swarm_violation_type type);
int swarm_agent_get_violations(u64 agent_id, struct swarm_violation_stats *stats);
int swarm_agent_set_violation_handler(u64 agent_id, enum swarm_violation_action action);

/* Module statistics */
int swarm_guard_get_stats(struct swarm_guard_stats *stats);
void swarm_guard_reset_stats(void);

/* Module initialization */
int swarm_guard_init(void);
void swarm_guard_exit(void);

#endif /* _SWARM_GUARD_H */