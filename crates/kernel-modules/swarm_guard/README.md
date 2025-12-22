# SwarmGuard Kernel Module

SwarmGuard is the core kernel module for StratoSwarm that provides resource enforcement, namespace management, and system call interception for agent containers.

## Features

- **System Call Interception**: Intercepts clone(), fork(), and execve() to enforce policies
- **Namespace Enforcement**: Ensures all agents run in proper namespace isolation (all 7 types)
- **Resource Management**: Integrates with cgroups v2 for CPU and memory limits
- **Device Whitelisting**: Controls which devices containers can access
- **Real-time Statistics**: Provides monitoring via /proc/swarm interface
- **Lock-free Tracking**: Uses RCU for high-performance agent management

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   SwarmGuard Module                      │
├─────────────────────────────────────────────────────────┤
│  System Call       Namespace         Resource           │
│  Interception      Management        Enforcement        │
│      ↓                 ↓                  ↓             │
│  ┌────────┐      ┌──────────┐     ┌──────────┐        │
│  │ Hooks  │      │ NS Types │     │ Cgroups  │        │
│  │ Table  │      │ Creation │     │ v2 API   │        │
│  └────────┘      └──────────┘     └──────────┘        │
│       ↓                ↓                 ↓              │
│  ┌─────────────────────────────────────────┐           │
│  │         Agent Policy Engine             │           │
│  └─────────────────────────────────────────┘           │
│                        ↓                                │
│  ┌─────────────────────────────────────────┐           │
│  │          /proc/swarm Interface          │           │
│  └─────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘
```

## Building

### Requirements

- Linux kernel 6.14+ with headers
- Rust for Linux support (experimental)
- Root privileges for module operations

### Build Steps

```bash
# Build the module
make

# Install (requires root)
sudo make install

# Load the module
sudo insmod swarm_guard.ko

# Verify it's loaded
lsmod | grep swarm_guard
```

## Testing

### Unit Tests (userspace)

```bash
# Run Rust unit tests
cargo test

# Run C unit tests
gcc -o test_main tests/test_main.c -lpthread
./test_main
```

### Integration Tests (requires module)

```bash
# Run integration tests (requires root)
sudo cargo test --features integration -- --ignored

# Run E2E test suite
sudo ./tests/e2e_test.sh
```

## /proc Interface

The module creates the following proc files:

### /proc/swarm/status
Shows global statistics:
```
Active agents: 42
Total created: 150
Total destroyed: 108
Policy violations: 3
```

### /proc/swarm/agents
Lists all active agents with their configurations:
```
ID: 1, PID: 1234, Memory: 256MB, CPU: 25%, Namespaces: 0x3F
ID: 2, PID: 5678, Memory: 512MB, CPU: 50%, Namespaces: 0x3F
```

### /proc/swarm/create
Write JSON to create a new agent:
```json
{
    "memory_limit": 268435456,
    "cpu_quota": 25,
    "namespace_flags": 63
}
```

## Performance Targets

- System call interception: <1μs overhead
- Agent creation: <10μs
- Concurrent agents: 200K+
- Memory overhead: <1KB per agent

## Security

- All agents run in separate namespaces
- Resource limits enforced at kernel level
- Device access whitelisted
- System calls filtered based on policy

## Module Parameters

```bash
# Load with custom parameters
sudo insmod swarm_guard.ko max_agents=500000 debug=1
```

- `max_agents`: Maximum number of concurrent agents (default: 200000)
- `debug`: Enable debug logging (default: 0)

## Troubleshooting

### Module won't load
- Check kernel version: `uname -r` (must be 6.14+)
- Verify kernel headers installed
- Check dmesg for error messages

### Performance issues
- Check `/proc/swarm/status` for high violation counts
- Monitor CPU usage of kernel threads
- Verify RCU grace periods aren't too long

### Agent creation failures
- Check system memory availability
- Verify cgroup hierarchy is properly set up
- Ensure namespace types are supported

## Development

### Adding new features

1. Update the policy engine in `src/policy.rs`
2. Add new proc interface if needed
3. Write comprehensive tests
4. Update performance benchmarks

### Debugging

```bash
# Enable debug output
echo 1 > /sys/module/swarm_guard/parameters/debug

# View kernel logs
dmesg | grep swarm_guard

# Trace system calls
strace -e trace=clone,fork,execve -p <pid>
```

## License

MIT OR Apache-2.0