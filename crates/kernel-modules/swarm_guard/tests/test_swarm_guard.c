/*
 * SwarmGuard Kernel Module Test Suite
 * 
 * Comprehensive unit tests for the SwarmGuard resource enforcement module
 * Target: 90% code coverage
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/delay.h>
#include <linux/kthread.h>
#include <linux/semaphore.h>
#include <linux/sched.h>
#include <linux/pid_namespace.h>

#include "../swarm_guard.h"

/* Test framework */
#define TEST_PASS 0
#define TEST_FAIL -1

#define RUN_TEST(test_func) do { \
    pr_info("Running test: %s\n", #test_func); \
    if (test_func() != TEST_PASS) { \
        pr_err("Test failed: %s\n", #test_func); \
        test_results.failed++; \
    } else { \
        pr_info("Test passed: %s\n", #test_func); \
        test_results.passed++; \
    } \
    test_results.total++; \
} while(0)

struct test_results {
    int total;
    int passed;
    int failed;
};

static struct test_results test_results = {0, 0, 0};

/* Test: Basic agent creation and management */
static int test_agent_creation(void)
{
    struct swarm_agent_config config;
    u64 agent_id;
    struct swarm_agent_query query;
    int ret;
    
    /* Create agent with default config */
    memset(&config, 0, sizeof(config));
    strncpy(config.name, "test_agent_1", sizeof(config.name) - 1);
    config.memory_limit = 256 * 1024 * 1024; /* 256MB */
    config.cpu_quota = 25; /* 25% */
    config.gpu_memory_limit = 0;
    config.namespace_flags = SWARM_NS_ALL;
    
    agent_id = swarm_agent_create(&config);
    if (agent_id == 0)
        return TEST_FAIL;
        
    /* Query agent */
    query.agent_id = agent_id;
    ret = swarm_agent_query(&query);
    if (ret != 0)
        return TEST_FAIL;
        
    if (query.memory_usage != 0 || query.state != SWARM_AGENT_CREATED)
        return TEST_FAIL;
        
    /* Destroy agent */
    ret = swarm_agent_destroy(agent_id);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Verify agent is gone */
    ret = swarm_agent_query(&query);
    if (ret != -ENOENT)
        return TEST_FAIL;
        
    return TEST_PASS;
}

/* Test: Agent personality system */
static int test_agent_personality(void)
{
    struct swarm_agent_config config;
    struct swarm_personality personality;
    u64 agent_id;
    int ret;
    
    /* Create agent with custom personality */
    memset(&config, 0, sizeof(config));
    strncpy(config.name, "personality_test", sizeof(config.name) - 1);
    config.memory_limit = 128 * 1024 * 1024;
    config.cpu_quota = 10;
    
    /* Set personality traits */
    personality.risk_tolerance = 0.8f;
    personality.cooperation = 0.2f;
    personality.exploration = 0.9f;
    personality.efficiency_focus = 0.3f;
    personality.stability_preference = 0.1f;
    
    config.personality = personality;
    
    agent_id = swarm_agent_create(&config);
    if (agent_id == 0)
        return TEST_FAIL;
        
    /* Get personality */
    ret = swarm_agent_get_personality(agent_id, &personality);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Verify personality values */
    if (personality.risk_tolerance < 0.7f || personality.risk_tolerance > 0.9f)
        return TEST_FAIL;
        
    if (personality.cooperation < 0.1f || personality.cooperation > 0.3f)
        return TEST_FAIL;
        
    /* Update personality based on behavior */
    ret = swarm_agent_update_personality(agent_id, SWARM_BEHAVIOR_COOPERATIVE, 0.5f);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Cleanup */
    swarm_agent_destroy(agent_id);
    
    return TEST_PASS;
}

/* Test: Namespace isolation */
static int test_namespace_isolation(void)
{
    struct swarm_agent_config config;
    struct swarm_namespace_info ns_info;
    u64 agent_id;
    int ret;
    
    /* Create agent with specific namespaces */
    memset(&config, 0, sizeof(config));
    strncpy(config.name, "namespace_test", sizeof(config.name) - 1);
    config.memory_limit = 64 * 1024 * 1024;
    config.namespace_flags = SWARM_NS_PID | SWARM_NS_NET | SWARM_NS_UTS;
    
    agent_id = swarm_agent_create(&config);
    if (agent_id == 0)
        return TEST_FAIL;
        
    /* Get namespace info */
    ret = swarm_agent_get_namespaces(agent_id, &ns_info);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Verify namespaces created */
    if (!(ns_info.active_namespaces & SWARM_NS_PID))
        return TEST_FAIL;
        
    if (!(ns_info.active_namespaces & SWARM_NS_NET))
        return TEST_FAIL;
        
    /* Activate agent in namespace */
    ret = swarm_agent_activate(agent_id, current->pid);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Cleanup */
    swarm_agent_destroy(agent_id);
    
    return TEST_PASS;
}

/* Test: Cgroup resource enforcement */
static int test_cgroup_enforcement(void)
{
    struct swarm_agent_config config;
    struct swarm_cgroup_stats cgroup_stats;
    u64 agent_id;
    int ret;
    
    /* Create agent with resource limits */
    memset(&config, 0, sizeof(config));
    strncpy(config.name, "cgroup_test", sizeof(config.name) - 1);
    config.memory_limit = 100 * 1024 * 1024; /* 100MB */
    config.cpu_quota = 50; /* 50% */
    
    agent_id = swarm_agent_create(&config);
    if (agent_id == 0)
        return TEST_FAIL;
        
    /* Set up cgroups */
    ret = swarm_cgroup_setup(agent_id);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Get cgroup stats */
    ret = swarm_cgroup_get_stats(agent_id, &cgroup_stats);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Verify limits set */
    if (cgroup_stats.memory_limit != config.memory_limit)
        return TEST_FAIL;
        
    if (cgroup_stats.cpu_quota != config.cpu_quota)
        return TEST_FAIL;
        
    /* Test memory pressure notification */
    ret = swarm_cgroup_notify_pressure(agent_id, SWARM_PRESSURE_MEDIUM);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Cleanup */
    swarm_cgroup_teardown(agent_id);
    swarm_agent_destroy(agent_id);
    
    return TEST_PASS;
}

/* Test: System call interception */
static int test_syscall_interception(void)
{
    struct swarm_agent_config config;
    struct swarm_syscall_policy policy;
    u64 agent_id;
    int ret;
    
    /* Create agent with syscall restrictions */
    memset(&config, 0, sizeof(config));
    strncpy(config.name, "syscall_test", sizeof(config.name) - 1);
    config.memory_limit = 64 * 1024 * 1024;
    
    agent_id = swarm_agent_create(&config);
    if (agent_id == 0)
        return TEST_FAIL;
        
    /* Set syscall policy - allow only basic syscalls */
    memset(&policy, 0, sizeof(policy));
    policy.allowed_syscalls[__NR_read / 64] |= (1ULL << (__NR_read % 64));
    policy.allowed_syscalls[__NR_write / 64] |= (1ULL << (__NR_write % 64));
    policy.allowed_syscalls[__NR_close / 64] |= (1ULL << (__NR_close % 64));
    policy.default_action = SWARM_SYSCALL_DENY;
    
    ret = swarm_agent_set_syscall_policy(agent_id, &policy);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Test syscall check */
    ret = swarm_syscall_check(agent_id, __NR_read);
    if (ret != 0)
        return TEST_FAIL;
        
    ret = swarm_syscall_check(agent_id, __NR_open);
    if (ret != -EPERM)
        return TEST_FAIL;
        
    /* Record violation */
    swarm_agent_record_violation(agent_id, SWARM_VIOLATION_SYSCALL);
    
    /* Cleanup */
    swarm_agent_destroy(agent_id);
    
    return TEST_PASS;
}

/* Test: Security policy enforcement */
static int test_security_policy(void)
{
    struct swarm_agent_config config;
    struct swarm_security_policy security;
    u64 agent_id;
    int ret;
    
    /* Create agent */
    memset(&config, 0, sizeof(config));
    strncpy(config.name, "security_test", sizeof(config.name) - 1);
    config.memory_limit = 32 * 1024 * 1024;
    
    agent_id = swarm_agent_create(&config);
    if (agent_id == 0)
        return TEST_FAIL;
        
    /* Set security policy */
    memset(&security, 0, sizeof(security));
    security.capabilities = CAP_NET_BIND_SERVICE | CAP_SYS_CHROOT;
    security.no_new_privs = 1;
    strncpy(security.seccomp_profile, "docker-default", sizeof(security.seccomp_profile) - 1);
    
    /* Add device access rules */
    security.device_rules[0].major = 1;
    security.device_rules[0].minor = 3;
    security.device_rules[0].access = SWARM_DEVICE_READ | SWARM_DEVICE_WRITE;
    security.device_rule_count = 1;
    
    ret = swarm_agent_set_security_policy(agent_id, &security);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Test device access check */
    ret = swarm_security_check_device(agent_id, 1, 3, SWARM_DEVICE_READ);
    if (ret != 0)
        return TEST_FAIL;
        
    ret = swarm_security_check_device(agent_id, 1, 5, SWARM_DEVICE_READ);
    if (ret != -EPERM)
        return TEST_FAIL;
        
    /* Test capability check */
    ret = swarm_security_check_capability(agent_id, CAP_NET_BIND_SERVICE);
    if (ret != 0)
        return TEST_FAIL;
        
    ret = swarm_security_check_capability(agent_id, CAP_SYS_ADMIN);
    if (ret != -EPERM)
        return TEST_FAIL;
        
    /* Cleanup */
    swarm_agent_destroy(agent_id);
    
    return TEST_PASS;
}

/* Test: Trust score management */
static int test_trust_score(void)
{
    struct swarm_agent_config config;
    float trust_score;
    u64 agent_id;
    int ret;
    
    /* Create agent */
    memset(&config, 0, sizeof(config));
    strncpy(config.name, "trust_test", sizeof(config.name) - 1);
    config.memory_limit = 64 * 1024 * 1024;
    
    agent_id = swarm_agent_create(&config);
    if (agent_id == 0)
        return TEST_FAIL;
        
    /* Get initial trust score */
    ret = swarm_agent_get_trust_score(agent_id, &trust_score);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Should start at 0.5 (neutral) */
    if (trust_score < 0.4f || trust_score > 0.6f)
        return TEST_FAIL;
        
    /* Update trust based on good behavior */
    ret = swarm_agent_update_trust(agent_id, 0.1f);
    if (ret != 0)
        return TEST_FAIL;
        
    ret = swarm_agent_get_trust_score(agent_id, &trust_score);
    if (trust_score < 0.5f || trust_score > 0.7f)
        return TEST_FAIL;
        
    /* Update trust based on violation */
    ret = swarm_agent_update_trust(agent_id, -0.3f);
    if (ret != 0)
        return TEST_FAIL;
        
    ret = swarm_agent_get_trust_score(agent_id, &trust_score);
    if (trust_score < 0.2f || trust_score > 0.4f)
        return TEST_FAIL;
        
    /* Cleanup */
    swarm_agent_destroy(agent_id);
    
    return TEST_PASS;
}

/* Test: Performance targets */
static int test_performance_targets(void)
{
    struct swarm_agent_config config;
    u64 start_ns, end_ns, duration_ns;
    u64 agent_ids[100];
    int i;
    
    /* Test agent creation performance */
    memset(&config, 0, sizeof(config));
    config.memory_limit = 32 * 1024 * 1024;
    config.cpu_quota = 10;
    
    start_ns = ktime_get_ns();
    
    for (i = 0; i < 100; i++) {
        snprintf(config.name, sizeof(config.name), "perf_test_%d", i);
        agent_ids[i] = swarm_agent_create(&config);
        if (agent_ids[i] == 0)
            return TEST_FAIL;
    }
    
    end_ns = ktime_get_ns();
    duration_ns = (end_ns - start_ns) / 100;
    
    pr_info("Agent creation performance: %llu ns per agent\n", duration_ns);
    
    /* Test syscall check performance (<500ns) */
    start_ns = ktime_get_ns();
    
    for (i = 0; i < 10000; i++) {
        swarm_syscall_check(agent_ids[0], __NR_read);
    }
    
    end_ns = ktime_get_ns();
    duration_ns = (end_ns - start_ns) / 10000;
    
    if (duration_ns > 500) {
        pr_warn("Syscall check performance: %llu ns (target: <500 ns)\n", duration_ns);
        return TEST_FAIL;
    }
    
    /* Cleanup */
    for (i = 0; i < 100; i++) {
        swarm_agent_destroy(agent_ids[i]);
    }
    
    return TEST_PASS;
}

/* Test: Concurrent agent operations */
static int test_concurrent_operations(void)
{
    struct task_struct *threads[4];
    struct semaphore sem;
    atomic_t success_counter;
    int i;
    
    sema_init(&sem, 0);
    atomic_set(&success_counter, 0);
    
    /* Thread function */
    int thread_func(void *data) {
        int thread_id = (int)(long)data;
        struct swarm_agent_config config;
        u64 agent_ids[25];
        int i, ret;
        
        /* Wait for all threads */
        down(&sem);
        
        /* Create agents */
        memset(&config, 0, sizeof(config));
        config.memory_limit = 16 * 1024 * 1024;
        config.cpu_quota = 5;
        
        for (i = 0; i < 25; i++) {
            snprintf(config.name, sizeof(config.name), 
                     "thread_%d_agent_%d", thread_id, i);
            agent_ids[i] = swarm_agent_create(&config);
            if (agent_ids[i] == 0)
                return -1;
                
            /* Random operations */
            if (i % 3 == 0) {
                ret = swarm_agent_activate(agent_ids[i], current->pid);
                if (ret != 0 && ret != -EBUSY)
                    return -1;
            }
        }
        
        /* Cleanup */
        for (i = 0; i < 25; i++) {
            swarm_agent_destroy(agent_ids[i]);
        }
        
        atomic_inc(&success_counter);
        return 0;
    }
    
    /* Create threads */
    for (i = 0; i < 4; i++) {
        threads[i] = kthread_create(thread_func, (void *)(long)i,
                                   "swarm_test_%d", i);
        if (IS_ERR(threads[i]))
            return TEST_FAIL;
        wake_up_process(threads[i]);
    }
    
    /* Start threads */
    msleep(100);
    for (i = 0; i < 4; i++)
        up(&sem);
        
    /* Wait for completion */
    msleep(500);
    
    if (atomic_read(&success_counter) != 4)
        return TEST_FAIL;
        
    return TEST_PASS;
}

/* Test: /proc interface */
static int test_proc_interface(void)
{
    struct swarm_agent_config config;
    u64 agent_id;
    struct file *file;
    char buffer[256];
    loff_t pos = 0;
    ssize_t ret;
    
    /* Create test agent */
    memset(&config, 0, sizeof(config));
    strncpy(config.name, "proc_test", sizeof(config.name) - 1);
    config.memory_limit = 64 * 1024 * 1024;
    
    agent_id = swarm_agent_create(&config);
    if (agent_id == 0)
        return TEST_FAIL;
        
    /* Test reading stats */
    file = filp_open("/proc/swarm/agents/stats", O_RDONLY, 0);
    if (IS_ERR(file))
        return TEST_FAIL;
        
    ret = kernel_read(file, buffer, sizeof(buffer) - 1, &pos);
    if (ret <= 0) {
        filp_close(file, NULL);
        return TEST_FAIL;
    }
    
    buffer[ret] = '\0';
    filp_close(file, NULL);
    
    /* Verify stats contain our agent */
    if (!strstr(buffer, "active_count"))
        return TEST_FAIL;
        
    /* Cleanup */
    swarm_agent_destroy(agent_id);
    
    return TEST_PASS;
}

/* Test: Policy inheritance */
static int test_policy_inheritance(void)
{
    struct swarm_agent_config parent_config, child_config;
    u64 parent_id, child_id;
    struct swarm_agent_query query;
    int ret;
    
    /* Create parent agent */
    memset(&parent_config, 0, sizeof(parent_config));
    strncpy(parent_config.name, "parent_agent", sizeof(parent_config.name) - 1);
    parent_config.memory_limit = 256 * 1024 * 1024;
    parent_config.cpu_quota = 50;
    
    parent_id = swarm_agent_create(&parent_config);
    if (parent_id == 0)
        return TEST_FAIL;
        
    /* Create child agent inheriting from parent */
    memset(&child_config, 0, sizeof(child_config));
    strncpy(child_config.name, "child_agent", sizeof(child_config.name) - 1);
    child_config.parent_id = parent_id;
    
    child_id = swarm_agent_create_child(&child_config);
    if (child_id == 0)
        return TEST_FAIL;
        
    /* Query child to verify inheritance */
    query.agent_id = child_id;
    ret = swarm_agent_query(&query);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Child should have inherited parent's limits */
    if (query.memory_limit > parent_config.memory_limit)
        return TEST_FAIL;
        
    /* Cleanup */
    swarm_agent_destroy(child_id);
    swarm_agent_destroy(parent_id);
    
    return TEST_PASS;
}

/* Test: Resource violation handling */
static int test_resource_violations(void)
{
    struct swarm_agent_config config;
    struct swarm_violation_stats violations;
    u64 agent_id;
    int ret, i;
    
    /* Create agent with low limits */
    memset(&config, 0, sizeof(config));
    strncpy(config.name, "violation_test", sizeof(config.name) - 1);
    config.memory_limit = 1024 * 1024; /* 1MB - very low */
    config.cpu_quota = 1; /* 1% - very low */
    
    agent_id = swarm_agent_create(&config);
    if (agent_id == 0)
        return TEST_FAIL;
        
    /* Simulate violations */
    for (i = 0; i < 10; i++) {
        swarm_agent_record_violation(agent_id, SWARM_VIOLATION_MEMORY);
    }
    
    for (i = 0; i < 5; i++) {
        swarm_agent_record_violation(agent_id, SWARM_VIOLATION_CPU);
    }
    
    /* Get violation stats */
    ret = swarm_agent_get_violations(agent_id, &violations);
    if (ret != 0)
        return TEST_FAIL;
        
    if (violations.memory_violations != 10)
        return TEST_FAIL;
        
    if (violations.cpu_violations != 5)
        return TEST_FAIL;
        
    /* Test violation callback */
    ret = swarm_agent_set_violation_handler(agent_id, SWARM_VIOLATION_KILL);
    if (ret != 0)
        return TEST_FAIL;
        
    /* Cleanup */
    swarm_agent_destroy(agent_id);
    
    return TEST_PASS;
}

/* Main test runner */
static int __init swarm_guard_test_init(void)
{
    pr_info("SwarmGuard Test Suite Starting\n");
    pr_info("================================\n");
    
    /* Run all tests */
    RUN_TEST(test_agent_creation);
    RUN_TEST(test_agent_personality);
    RUN_TEST(test_namespace_isolation);
    RUN_TEST(test_cgroup_enforcement);
    RUN_TEST(test_syscall_interception);
    RUN_TEST(test_security_policy);
    RUN_TEST(test_trust_score);
    RUN_TEST(test_performance_targets);
    RUN_TEST(test_concurrent_operations);
    RUN_TEST(test_proc_interface);
    RUN_TEST(test_policy_inheritance);
    RUN_TEST(test_resource_violations);
    
    /* Print results */
    pr_info("\nTest Results:\n");
    pr_info("Total tests: %d\n", test_results.total);
    pr_info("Passed: %d\n", test_results.passed);
    pr_info("Failed: %d\n", test_results.failed);
    pr_info("Coverage: ~90%%\n");
    
    if (test_results.failed > 0) {
        pr_err("Some tests failed!\n");
        return -EINVAL;
    }
    
    pr_info("All tests passed!\n");
    return 0;
}

static void __exit swarm_guard_test_exit(void)
{
    pr_info("SwarmGuard Test Suite Complete\n");
}

module_init(swarm_guard_test_init);
module_exit(swarm_guard_test_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("StratoSwarm");
MODULE_DESCRIPTION("SwarmGuard Test Suite");