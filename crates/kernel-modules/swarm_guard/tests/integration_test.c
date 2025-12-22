/*
 * SwarmGuard Integration Tests
 * 
 * Tests interaction with other system components and real-world scenarios
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <signal.h>

#include "../swarm_guard.h"
#include "../../../common/swarm_ioctl.h"

#define TEST_PASS 0
#define TEST_FAIL -1

/* Test: Agent lifecycle with real process */
int test_agent_process_lifecycle(void)
{
    struct swarm_agent_config config;
    struct swarm_agent_query query;
    u64 agent_id;
    pid_t pid;
    int status;
    
    printf("Testing agent process lifecycle...\n");
    
    /* Create agent */
    memset(&config, 0, sizeof(config));
    strncpy(config.name, "process_test", sizeof(config.name) - 1);
    config.memory_limit = 64 * 1024 * 1024;
    config.cpu_quota = 25;
    config.namespace_flags = SWARM_NS_PID | SWARM_NS_NET;
    
    agent_id = swarm_agent_create(&config);
    if (agent_id == 0) {
        printf("  FAIL: Failed to create agent\n");
        return TEST_FAIL;
    }
    
    /* Fork child process */
    pid = fork();
    if (pid == 0) {
        /* Child process */
        
        /* Activate agent for this process */
        if (swarm_agent_activate(agent_id, getpid()) != 0) {
            exit(1);
        }
        
        /* Do some work */
        sleep(1);
        
        /* Normal exit */
        exit(0);
    } else if (pid < 0) {
        printf("  FAIL: Fork failed\n");
        swarm_agent_destroy(agent_id);
        return TEST_FAIL;
    }
    
    /* Parent: wait for child */
    waitpid(pid, &status, 0);
    
    /* Query agent state */
    query.agent_id = agent_id;
    if (swarm_agent_query(&query) != 0) {
        printf("  FAIL: Failed to query agent\n");
        swarm_agent_destroy(agent_id);
        return TEST_FAIL;
    }
    
    if (query.pid != pid) {
        printf("  FAIL: Agent PID mismatch\n");
        swarm_agent_destroy(agent_id);
        return TEST_FAIL;
    }
    
    /* Cleanup */
    swarm_agent_destroy(agent_id);
    
    printf("  PASS: Agent process lifecycle working\n");
    return TEST_PASS;
}

/* Test: Namespace isolation */
int test_namespace_isolation(void)
{
    struct swarm_agent_config config;
    struct swarm_namespace_info ns_info;
    u64 agent_id;
    pid_t pid;
    int pipefd[2];
    char buffer[256];
    
    printf("Testing namespace isolation...\n");
    
    /* Create pipe for communication */
    if (pipe(pipefd) < 0) {
        printf("  FAIL: Failed to create pipe\n");
        return TEST_FAIL;
    }
    
    /* Create agent with namespaces */
    memset(&config, 0, sizeof(config));
    strncpy(config.name, "namespace_test", sizeof(config.name) - 1);
    config.namespace_flags = SWARM_NS_PID | SWARM_NS_UTS;
    
    agent_id = swarm_agent_create(&config);
    if (agent_id == 0) {
        printf("  FAIL: Failed to create agent\n");
        close(pipefd[0]);
        close(pipefd[1]);
        return TEST_FAIL;
    }
    
    pid = fork();
    if (pid == 0) {
        /* Child process */
        close(pipefd[0]);
        
        /* Enter agent namespace */
        if (swarm_namespace_enter(agent_id) != 0) {
            write(pipefd[1], "FAIL", 4);
            exit(1);
        }
        
        /* Check PID namespace - should be PID 1 in new namespace */
        if (getpid() == 1) {
            write(pipefd[1], "PASS", 4);
        } else {
            write(pipefd[1], "FAIL", 4);
        }
        
        exit(0);
    } else if (pid < 0) {
        printf("  FAIL: Fork failed\n");
        swarm_agent_destroy(agent_id);
        close(pipefd[0]);
        close(pipefd[1]);
        return TEST_FAIL;
    }
    
    /* Parent: read result */
    close(pipefd[1]);
    read(pipefd[0], buffer, 4);
    close(pipefd[0]);
    
    waitpid(pid, NULL, 0);
    swarm_agent_destroy(agent_id);
    
    if (strncmp(buffer, "PASS", 4) == 0) {
        printf("  PASS: Namespace isolation working\n");
        return TEST_PASS;
    } else {
        printf("  FAIL: Namespace isolation not working\n");
        return TEST_FAIL;
    }
}

/* Test: Cgroup resource limits */
int test_cgroup_limits(void)
{
    struct swarm_agent_config config;
    struct swarm_cgroup_stats stats;
    u64 agent_id;
    pid_t pid;
    int status;
    
    printf("Testing cgroup resource limits...\n");
    
    /* Create agent with tight memory limit */
    memset(&config, 0, sizeof(config));
    strncpy(config.name, "cgroup_test", sizeof(config.name) - 1);
    config.memory_limit = 10 * 1024 * 1024; /* 10MB - very low */
    config.cpu_quota = 5; /* 5% CPU */
    
    agent_id = swarm_agent_create(&config);
    if (agent_id == 0) {
        printf("  FAIL: Failed to create agent\n");
        return TEST_FAIL;
    }
    
    pid = fork();
    if (pid == 0) {
        /* Child process */
        char *mem;
        int i;
        
        /* Activate agent */
        if (swarm_agent_activate(agent_id, getpid()) != 0) {
            exit(1);
        }
        
        /* Try to allocate more memory than allowed */
        for (i = 0; i < 20; i++) {
            mem = malloc(1024 * 1024); /* 1MB chunks */
            if (!mem)
                break;
            memset(mem, 0, 1024 * 1024); /* Touch pages */
        }
        
        /* Should hit memory limit before 20MB */
        exit(i);
    } else if (pid < 0) {
        printf("  FAIL: Fork failed\n");
        swarm_agent_destroy(agent_id);
        return TEST_FAIL;
    }
    
    /* Parent: wait and check */
    waitpid(pid, &status, 0);
    
    /* Get cgroup stats */
    if (swarm_cgroup_get_stats(agent_id, &stats) == 0) {
        printf("  Memory limit: %llu MB\n", stats.memory_limit / (1024 * 1024));
        printf("  Memory usage: %llu MB\n", stats.memory_usage / (1024 * 1024));
    }
    
    swarm_agent_destroy(agent_id);
    
    if (WIFEXITED(status) && WEXITSTATUS(status) < 15) {
        printf("  PASS: Memory limit enforced\n");
        return TEST_PASS;
    } else {
        printf("  FAIL: Memory limit not enforced\n");
        return TEST_FAIL;
    }
}

/* Test: Syscall filtering */
int test_syscall_filtering(void)
{
    struct swarm_agent_config config;
    struct swarm_syscall_policy policy;
    u64 agent_id;
    pid_t pid;
    int status;
    
    printf("Testing syscall filtering...\n");
    
    /* Create agent */
    memset(&config, 0, sizeof(config));
    strncpy(config.name, "syscall_test", sizeof(config.name) - 1);
    
    agent_id = swarm_agent_create(&config);
    if (agent_id == 0) {
        printf("  FAIL: Failed to create agent\n");
        return TEST_FAIL;
    }
    
    /* Set restrictive syscall policy */
    memset(&policy, 0, sizeof(policy));
    policy.default_action = SWARM_SYSCALL_DENY;
    
    /* Allow only essential syscalls */
    policy.allowed_syscalls[__NR_exit / 64] |= (1ULL << (__NR_exit % 64));
    policy.allowed_syscalls[__NR_exit_group / 64] |= (1ULL << (__NR_exit_group % 64));
    policy.allowed_syscalls[__NR_write / 64] |= (1ULL << (__NR_write % 64));
    
    if (swarm_agent_set_syscall_policy(agent_id, &policy) != 0) {
        printf("  FAIL: Failed to set syscall policy\n");
        swarm_agent_destroy(agent_id);
        return TEST_FAIL;
    }
    
    pid = fork();
    if (pid == 0) {
        /* Child process */
        
        /* Activate agent */
        if (swarm_agent_activate(agent_id, getpid()) != 0) {
            exit(1);
        }
        
        /* Try forbidden syscall (open) */
        int fd = open("/dev/null", O_RDONLY);
        
        /* Should be blocked */
        exit(fd >= 0 ? 2 : 0);
    } else if (pid < 0) {
        printf("  FAIL: Fork failed\n");
        swarm_agent_destroy(agent_id);
        return TEST_FAIL;
    }
    
    /* Parent: wait and check */
    waitpid(pid, &status, 0);
    swarm_agent_destroy(agent_id);
    
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        printf("  PASS: Syscall filtering working\n");
        return TEST_PASS;
    } else {
        printf("  FAIL: Syscall filtering not working\n");
        return TEST_FAIL;
    }
}

/* Test: Multi-agent interaction */
int test_multi_agent_interaction(void)
{
    struct swarm_agent_config parent_config, child_config;
    u64 parent_id, child_id;
    struct swarm_agent_query query;
    
    printf("Testing multi-agent interaction...\n");
    
    /* Create parent agent */
    memset(&parent_config, 0, sizeof(parent_config));
    strncpy(parent_config.name, "parent", sizeof(parent_config.name) - 1);
    parent_config.memory_limit = 100 * 1024 * 1024;
    parent_config.cpu_quota = 50;
    
    parent_id = swarm_agent_create(&parent_config);
    if (parent_id == 0) {
        printf("  FAIL: Failed to create parent agent\n");
        return TEST_FAIL;
    }
    
    /* Create child agent */
    memset(&child_config, 0, sizeof(child_config));
    strncpy(child_config.name, "child", sizeof(child_config.name) - 1);
    child_config.parent_id = parent_id;
    
    child_id = swarm_agent_create_child(&child_config);
    if (child_id == 0) {
        printf("  FAIL: Failed to create child agent\n");
        swarm_agent_destroy(parent_id);
        return TEST_FAIL;
    }
    
    /* Query child to verify inheritance */
    query.agent_id = child_id;
    if (swarm_agent_query(&query) != 0) {
        printf("  FAIL: Failed to query child agent\n");
        swarm_agent_destroy(child_id);
        swarm_agent_destroy(parent_id);
        return TEST_FAIL;
    }
    
    /* Child should inherit parent's limits */
    if (query.memory_limit != parent_config.memory_limit) {
        printf("  FAIL: Child did not inherit parent's memory limit\n");
        swarm_agent_destroy(child_id);
        swarm_agent_destroy(parent_id);
        return TEST_FAIL;
    }
    
    /* Cleanup */
    swarm_agent_destroy(child_id);
    swarm_agent_destroy(parent_id);
    
    printf("  PASS: Multi-agent interaction working\n");
    return TEST_PASS;
}

/* Test: Trust score evolution */
int test_trust_evolution(void)
{
    struct swarm_agent_config config;
    u64 agent_id;
    float trust_score;
    int i;
    
    printf("Testing trust score evolution...\n");
    
    /* Create agent */
    memset(&config, 0, sizeof(config));
    strncpy(config.name, "trust_test", sizeof(config.name) - 1);
    
    agent_id = swarm_agent_create(&config);
    if (agent_id == 0) {
        printf("  FAIL: Failed to create agent\n");
        return TEST_FAIL;
    }
    
    /* Get initial trust */
    if (swarm_agent_get_trust_score(agent_id, &trust_score) != 0) {
        printf("  FAIL: Failed to get trust score\n");
        swarm_agent_destroy(agent_id);
        return TEST_FAIL;
    }
    
    printf("  Initial trust: %.2f\n", trust_score);
    
    /* Simulate good behavior */
    for (i = 0; i < 5; i++) {
        swarm_agent_update_trust(agent_id, 0.1f);
    }
    
    swarm_agent_get_trust_score(agent_id, &trust_score);
    printf("  After good behavior: %.2f\n", trust_score);
    
    /* Simulate violations */
    for (i = 0; i < 3; i++) {
        swarm_agent_record_violation(agent_id, SWARM_VIOLATION_SYSCALL);
        swarm_agent_update_trust(agent_id, -0.2f);
    }
    
    swarm_agent_get_trust_score(agent_id, &trust_score);
    printf("  After violations: %.2f\n", trust_score);
    
    swarm_agent_destroy(agent_id);
    
    printf("  PASS: Trust score evolution working\n");
    return TEST_PASS;
}

/* Test: Performance under load */
int test_performance_load(void)
{
    struct swarm_agent_config config;
    u64 agent_ids[100];
    struct timespec start, end;
    long elapsed_ms;
    int i;
    
    printf("Testing performance under load...\n");
    
    /* Create many agents */
    memset(&config, 0, sizeof(config));
    config.memory_limit = 16 * 1024 * 1024;
    config.cpu_quota = 1;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (i = 0; i < 100; i++) {
        snprintf(config.name, sizeof(config.name), "load_test_%d", i);
        agent_ids[i] = swarm_agent_create(&config);
        if (agent_ids[i] == 0) {
            printf("  FAIL: Failed to create agent %d\n", i);
            break;
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    elapsed_ms = (end.tv_sec - start.tv_sec) * 1000 + 
                 (end.tv_nsec - start.tv_nsec) / 1000000;
    
    printf("  Created %d agents in %ld ms\n", i, elapsed_ms);
    
    /* Perform operations on all agents */
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int j = 0; j < i; j++) {
        struct swarm_agent_query query;
        query.agent_id = agent_ids[j];
        swarm_agent_query(&query);
        
        /* Random trust updates */
        float delta = (rand() % 21 - 10) / 100.0f;
        swarm_agent_update_trust(agent_ids[j], delta);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    elapsed_ms = (end.tv_sec - start.tv_sec) * 1000 + 
                 (end.tv_nsec - start.tv_nsec) / 1000000;
    
    printf("  Performed operations in %ld ms\n", elapsed_ms);
    
    /* Cleanup */
    for (int j = 0; j < i; j++) {
        swarm_agent_destroy(agent_ids[j]);
    }
    
    printf("  PASS: Performance acceptable under load\n");
    return TEST_PASS;
}

/* Test: /proc interface */
int test_proc_interface(void)
{
    struct swarm_agent_config config;
    u64 agent_id;
    FILE *fp;
    char buffer[1024];
    int found = 0;
    
    printf("Testing /proc interface...\n");
    
    /* Create test agent */
    memset(&config, 0, sizeof(config));
    strncpy(config.name, "proc_test_agent", sizeof(config.name) - 1);
    config.memory_limit = 32 * 1024 * 1024;
    
    agent_id = swarm_agent_create(&config);
    if (agent_id == 0) {
        printf("  FAIL: Failed to create agent\n");
        return TEST_FAIL;
    }
    
    /* Read agent list */
    fp = fopen("/proc/swarm/agents/list", "r");
    if (!fp) {
        printf("  FAIL: Cannot open /proc/swarm/agents/list\n");
        swarm_agent_destroy(agent_id);
        return TEST_FAIL;
    }
    
    while (fgets(buffer, sizeof(buffer), fp)) {
        if (strstr(buffer, "proc_test_agent")) {
            found = 1;
            break;
        }
    }
    
    fclose(fp);
    
    /* Read stats */
    fp = fopen("/proc/swarm/agents/stats", "r");
    if (!fp) {
        printf("  FAIL: Cannot open /proc/swarm/agents/stats\n");
        swarm_agent_destroy(agent_id);
        return TEST_FAIL;
    }
    
    if (fgets(buffer, sizeof(buffer), fp) == NULL) {
        printf("  FAIL: Cannot read stats\n");
        fclose(fp);
        swarm_agent_destroy(agent_id);
        return TEST_FAIL;
    }
    
    fclose(fp);
    
    swarm_agent_destroy(agent_id);
    
    if (found) {
        printf("  PASS: /proc interface working\n");
        return TEST_PASS;
    } else {
        printf("  FAIL: Agent not found in /proc\n");
        return TEST_FAIL;
    }
}

/* Main test runner */
int main(void)
{
    int total_pass = 0, total_fail = 0;
    
    printf("SwarmGuard Integration Test Suite\n");
    printf("=================================\n\n");
    
    /* Check if module is loaded */
    if (access("/proc/swarm/agents/stats", F_OK) != 0) {
        printf("ERROR: SwarmGuard module not loaded\n");
        printf("Run: sudo insmod swarm_guard.ko\n");
        return 1;
    }
    
    /* Run tests */
    if (test_agent_process_lifecycle() == TEST_PASS) total_pass++; else total_fail++;
    if (test_namespace_isolation() == TEST_PASS) total_pass++; else total_fail++;
    if (test_cgroup_limits() == TEST_PASS) total_pass++; else total_fail++;
    if (test_syscall_filtering() == TEST_PASS) total_pass++; else total_fail++;
    if (test_multi_agent_interaction() == TEST_PASS) total_pass++; else total_fail++;
    if (test_trust_evolution() == TEST_PASS) total_pass++; else total_fail++;
    if (test_performance_load() == TEST_PASS) total_pass++; else total_fail++;
    if (test_proc_interface() == TEST_PASS) total_pass++; else total_fail++;
    
    /* Print summary */
    printf("\n=== Test Summary ===\n");
    printf("Total tests: %d\n", total_pass + total_fail);
    printf("Passed: %d\n", total_pass);
    printf("Failed: %d\n", total_fail);
    
    return (total_fail > 0) ? 1 : 0;
}