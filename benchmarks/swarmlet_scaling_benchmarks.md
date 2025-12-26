# Swarmlet Scaling & Raspberry Pi Cluster Benchmarks

This document provides instructions for testing StratoSwarm's heterogeneous computing capabilities using a cluster of Raspberry Pi nodes with the lightweight swarmlet container system.

## Test Environment

### Hardware Setup
- **Primary Node**: x86_64 system with GPU (control plane)
- **Worker Nodes**: 5x Raspberry Pi 4 (4GB/8GB RAM)
- **Network**: Gigabit Ethernet switch or Wi-Fi mesh
- **Total Cluster**: 6 nodes (1 GPU + 5 RPi)

### Raspberry Pi Preparation
```bash
# On each Raspberry Pi
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Enable cgroups
sudo sed -i '$ s/$/ cgroup_enable=cpuset cgroup_enable=memory cgroup_memory=1/' /boot/cmdline.txt
sudo reboot

# Set up monitoring
sudo apt-get update
sudo apt-get install -y htop iotop nmon
```

## Swarmlet Performance Tests

### 1. Individual Swarmlet Join Performance

**Goal**: Measure time from `docker run` to fully joined cluster member

#### Test Script
```bash
#!/bin/bash
# File: benchmarks/scripts/swarmlet_join_bench.sh

# Generate join token on primary node
PRIMARY_IP="192.168.1.100"
JOIN_TOKEN=$(stratoswarm cluster create-join-token)

echo "Swarmlet Join Performance Test"
echo "=============================="
echo "Token: $JOIN_TOKEN"
echo "Cluster: $PRIMARY_IP"

# Function to test single join
test_single_join() {
    local NODE_IP=$1
    local NODE_NAME=$2
    
    echo -e "\n[$NODE_NAME] Starting join test..."
    
    # SSH to RPi and measure join time
    ssh pi@$NODE_IP << EOF
        # Clean any previous swarmlet
        docker rm -f swarmlet 2>/dev/null
        
        # Measure join time
        START_TIME=\$(date +%s.%N)
        
        docker run -d \
            --name swarmlet \
            --network host \
            --privileged \
            -v /var/run/docker.sock:/var/run/docker.sock \
            stratoswarm/swarmlet:latest \
            join --token $JOIN_TOKEN --cluster $PRIMARY_IP
        
        # Wait for join completion
        while ! docker logs swarmlet 2>&1 | grep -q "Successfully joined cluster"; do
            sleep 0.5
        done
        
        END_TIME=\$(date +%s.%N)
        JOIN_TIME=\$(echo "\$END_TIME - \$START_TIME" | bc)
        
        echo "Join completed in: \${JOIN_TIME}s"
        
        # Collect metrics
        docker logs swarmlet > /tmp/swarmlet_metrics.log
        grep "Hardware profile:" /tmp/swarmlet_metrics.log
        grep "Network latency:" /tmp/swarmlet_metrics.log
EOF
}

# Test each RPi
NODES=(
    "192.168.1.101:rpi-1"
    "192.168.1.102:rpi-2"
    "192.168.1.103:rpi-3"
    "192.168.1.104:rpi-4"
    "192.168.1.105:rpi-5"
)

for node in "${NODES[@]}"; do
    IFS=':' read -r ip name <<< "$node"
    test_single_join $ip $name
done
```

#### Expected Metrics
```
Join Time Targets:
- Hardware Profiling: <5 seconds
- Network Discovery: <3 seconds
- Certificate Exchange: <2 seconds
- Total Join Time: <30 seconds

Resource Usage:
- Memory Footprint: <50MB
- CPU Usage: <10% average
- Network Traffic: <10MB total
```

### 2. Heterogeneous Workload Distribution

**Goal**: Test GPU + RPi coordination and workload scheduling

#### Test Configuration
```yaml
# File: benchmarks/configs/heterogeneous_workload.yaml
workloads:
  - name: "gpu-inference"
    type: "gpu-compute"
    requirements:
      gpu: true
      memory: "4GB"
    expected_node: "primary-gpu"
    
  - name: "edge-sensor-processing"
    type: "cpu-light"
    requirements:
      cpu: 1.0
      memory: "256MB"
    replicas: 5
    expected_nodes: ["rpi-1", "rpi-2", "rpi-3", "rpi-4", "rpi-5"]
    
  - name: "data-aggregation"
    type: "cpu-moderate"
    requirements:
      cpu: 2.0
      memory: "1GB"
    expected_node: "any-capable"
```

#### Benchmark Script
```bash
#!/bin/bash
# File: benchmarks/scripts/heterogeneous_workload_bench.sh

echo "Heterogeneous Workload Distribution Test"
echo "======================================="

# Deploy mixed workloads
stratoswarm deploy benchmarks/configs/heterogeneous_workload.yaml \
    --measure-scheduling-time \
    --output results/heterogeneous_scheduling.json

# Monitor workload placement
watch_placement() {
    echo -e "\nWorkload Placement:"
    stratoswarm cluster nodes --format=table
    echo -e "\nRunning Workloads:"
    stratoswarm ps --all-nodes
}

# Measure scheduling efficiency
measure_scheduling() {
    local START=$(date +%s.%N)
    
    # Deploy 50 mixed workloads
    for i in {1..50}; do
        if (( $i % 10 == 0 )); then
            # Every 10th is GPU workload
            stratoswarm run --gpu nvidia/cuda:11.0-base nvidia-smi
        else
            # CPU workloads distributed to RPis
            stratoswarm run --cpu 0.5 --memory 128M alpine sleep 60
        fi
    done
    
    local END=$(date +%s.%N)
    local DURATION=$(echo "$END - $START" | bc)
    
    echo "Scheduled 50 workloads in: ${DURATION}s"
    echo "Average scheduling time: $(echo "scale=3; $DURATION / 50" | bc)s"
}

# Test workload migration
test_migration() {
    echo -e "\nTesting workload migration..."
    
    # Start workload on RPi
    WORKLOAD_ID=$(stratoswarm run --constraint node.name==rpi-1 \
        alpine sh -c "while true; do date; sleep 1; done")
    
    sleep 5
    
    # Migrate to different RPi
    START=$(date +%s.%N)
    stratoswarm migrate $WORKLOAD_ID --to rpi-2
    END=$(date +%s.%N)
    
    MIGRATION_TIME=$(echo "$END - $START" | bc)
    echo "Migration completed in: ${MIGRATION_TIME}s"
}

# Run tests
watch_placement
measure_scheduling
test_migration
```

### 3. Swarmlet Network Mesh Formation

**Goal**: Measure mesh network formation and coordination latency

#### Mesh Testing Script
```python
#!/usr/bin/env python3
# File: benchmarks/scripts/mesh_formation_test.py

import time
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

class MeshBenchmark:
    def __init__(self, nodes):
        self.nodes = nodes
        self.results = {}
    
    def test_discovery_time(self):
        """Measure time for all nodes to discover each other"""
        print("\n=== Node Discovery Test ===")
        start_time = time.time()
        
        # Start swarmlets simultaneously
        with ThreadPoolExecutor(max_workers=len(self.nodes)) as executor:
            futures = []
            for node in self.nodes:
                future = executor.submit(self.start_swarmlet, node)
                futures.append(future)
            
            # Wait for all to start
            for future in as_completed(futures):
                result = future.result()
                print(f"Started swarmlet on {result['node']}")
        
        # Wait for full mesh formation
        while not self.is_mesh_complete():
            time.sleep(0.5)
        
        discovery_time = time.time() - start_time
        print(f"Full mesh discovery completed in: {discovery_time:.2f}s")
        self.results['discovery_time'] = discovery_time
    
    def test_coordination_latency(self):
        """Measure cross-node coordination latency"""
        print("\n=== Coordination Latency Test ===")
        
        latencies = []
        
        # Test pairwise latencies
        for i, node1 in enumerate(self.nodes):
            for node2 in self.nodes[i+1:]:
                latency = self.measure_latency(node1, node2)
                latencies.append(latency)
                print(f"{node1['name']} <-> {node2['name']}: {latency:.2f}ms")
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        print(f"\nAverage latency: {avg_latency:.2f}ms")
        print(f"Maximum latency: {max_latency:.2f}ms")
        
        self.results['avg_coordination_latency'] = avg_latency
        self.results['max_coordination_latency'] = max_latency
    
    def test_bandwidth(self):
        """Test mesh network bandwidth"""
        print("\n=== Bandwidth Test ===")
        
        # GPU to RPi bandwidth
        gpu_node = self.nodes[0]  # Assuming first is GPU node
        
        for rpi_node in self.nodes[1:]:
            bandwidth = self.measure_bandwidth(gpu_node, rpi_node)
            print(f"GPU -> {rpi_node['name']}: {bandwidth:.2f} MB/s")
        
    def stress_test_mesh(self):
        """Stress test with high message volume"""
        print("\n=== Mesh Stress Test ===")
        
        # Generate high volume of cross-node messages
        message_count = 10000
        message_size = 1024  # 1KB messages
        
        start_time = time.time()
        
        # Broadcast messages from each node
        with ThreadPoolExecutor(max_workers=len(self.nodes)) as executor:
            futures = []
            for node in self.nodes:
                future = executor.submit(
                    self.broadcast_messages, 
                    node, 
                    message_count // len(self.nodes),
                    message_size
                )
                futures.append(future)
            
            for future in as_completed(futures):
                result = future.result()
        
        duration = time.time() - start_time
        throughput = (message_count * message_size) / (duration * 1024 * 1024)
        
        print(f"Processed {message_count} messages in {duration:.2f}s")
        print(f"Throughput: {throughput:.2f} MB/s")
        
        self.results['stress_test_throughput'] = throughput

# Run benchmark
if __name__ == "__main__":
    nodes = [
        {"name": "gpu-primary", "ip": "192.168.1.100", "type": "gpu"},
        {"name": "rpi-1", "ip": "192.168.1.101", "type": "rpi"},
        {"name": "rpi-2", "ip": "192.168.1.102", "type": "rpi"},
        {"name": "rpi-3", "ip": "192.168.1.103", "type": "rpi"},
        {"name": "rpi-4", "ip": "192.168.1.104", "type": "rpi"},
        {"name": "rpi-5", "ip": "192.168.1.105", "type": "rpi"},
    ]
    
    benchmark = MeshBenchmark(nodes)
    benchmark.test_discovery_time()
    benchmark.test_coordination_latency()
    benchmark.test_bandwidth()
    benchmark.stress_test_mesh()
    
    # Save results
    with open('results/mesh_benchmark.json', 'w') as f:
        json.dump(benchmark.results, f, indent=2)
```

### 4. Edge Device Resource Constraints

**Goal**: Test swarmlet performance under RPi resource limitations

#### Resource Limit Testing
```bash
#!/bin/bash
# File: benchmarks/scripts/edge_resource_limits.sh

echo "Edge Device Resource Constraint Testing"
echo "====================================="

# Test memory pressure
test_memory_pressure() {
    local NODE=$1
    echo -e "\n[$NODE] Testing memory pressure..."
    
    ssh pi@$NODE << 'EOF'
        # Get baseline memory
        FREE_BEFORE=$(free -m | awk '/^Mem:/ {print $4}')
        
        # Run memory-intensive workload
        docker run --rm \
            --memory="3g" \
            --memory-swap="3g" \
            progrium/stress \
            --vm 1 --vm-bytes 3G --timeout 30s
        
        # Check swarmlet behavior under pressure
        docker stats --no-stream swarmlet
        
        FREE_AFTER=$(free -m | awk '/^Mem:/ {print $4}')
        echo "Memory impact: $((FREE_BEFORE - FREE_AFTER)) MB"
EOF
}

# Test CPU limits
test_cpu_limits() {
    local NODE=$1
    echo -e "\n[$NODE] Testing CPU limits..."
    
    ssh pi@$NODE << 'EOF'
        # Run CPU-intensive workload
        docker run --rm \
            --cpus="3.5" \
            progrium/stress \
            --cpu 4 --timeout 30s &
        
        STRESS_PID=$!
        
        # Monitor swarmlet performance
        for i in {1..10}; do
            docker stats --no-stream swarmlet
            sleep 3
        done
        
        wait $STRESS_PID
EOF
}

# Test network bandwidth limits
test_network_limits() {
    local NODE=$1
    echo -e "\n[$NODE] Testing network limits..."
    
    # Limit bandwidth using tc
    ssh pi@$NODE << 'EOF'
        # Apply bandwidth limit (10 Mbps)
        sudo tc qdisc add dev eth0 root handle 1: htb default 30
        sudo tc class add dev eth0 parent 1: classid 1:1 htb rate 10mbit
        
        # Test swarmlet communication
        time docker exec swarmlet stratoswarm-test bandwidth --duration 10
        
        # Remove limit
        sudo tc qdisc del dev eth0 root
EOF
}

# Test thermal throttling
test_thermal_throttling() {
    local NODE=$1
    echo -e "\n[$NODE] Testing thermal behavior..."
    
    ssh pi@$NODE << 'EOF'
        # Monitor temperature during stress
        (while true; do 
            echo "$(date +%s),$(cat /sys/class/thermal/thermal_zone0/temp)"
            sleep 1
        done) > thermal_log.csv &
        TEMP_PID=$!
        
        # Run stress test
        stress-ng --cpu 4 --cpu-method matrixprod --timeout 120s
        
        kill $TEMP_PID
        
        # Analyze throttling
        echo "Temperature data saved to thermal_log.csv"
        MAX_TEMP=$(cut -d',' -f2 thermal_log.csv | sort -n | tail -1)
        echo "Maximum temperature: $((MAX_TEMP/1000))°C"
EOF
}

# Run tests on each RPi
for i in {1..5}; do
    NODE="192.168.1.10$i"
    echo -e "\n=== Testing RPi-$i ($NODE) ==="
    test_memory_pressure $NODE
    test_cpu_limits $NODE
    test_network_limits $NODE
    test_thermal_throttling $NODE
done
```

### 5. Failure Recovery & Resilience

**Goal**: Test swarmlet behavior during node failures and network partitions

#### Failure Injection Tests
```bash
#!/bin/bash
# File: benchmarks/scripts/swarmlet_failure_recovery.sh

echo "Swarmlet Failure Recovery Testing"
echo "================================"

# Test single node failure
test_node_failure() {
    echo -e "\n=== Single Node Failure Test ==="
    
    # Deploy workload to specific RPi
    WORKLOAD_ID=$(stratoswarm run --constraint node.name==rpi-3 \
        alpine sh -c "while true; do date > /data/heartbeat; sleep 1; done")
    
    sleep 10
    
    # Simulate node failure
    echo "Simulating node failure on rpi-3..."
    ssh pi@192.168.1.103 "sudo poweroff"
    
    # Measure detection time
    START=$(date +%s)
    while stratoswarm node ls | grep -q "rpi-3.*Ready"; do
        sleep 0.5
    done
    END=$(date +%s)
    
    DETECTION_TIME=$((END - START))
    echo "Failure detected in: ${DETECTION_TIME}s"
    
    # Check workload rescheduling
    sleep 5
    NEW_NODE=$(stratoswarm ps | grep $WORKLOAD_ID | awk '{print $3}')
    echo "Workload rescheduled to: $NEW_NODE"
}

# Test network partition
test_network_partition() {
    echo -e "\n=== Network Partition Test ==="
    
    # Create partition between GPU node and 2 RPis
    echo "Creating network partition..."
    ssh pi@192.168.1.101 "sudo iptables -A INPUT -s 192.168.1.100 -j DROP"
    ssh pi@192.168.1.102 "sudo iptables -A INPUT -s 192.168.1.100 -j DROP"
    
    # Monitor mesh reformation
    START=$(date +%s)
    
    # Wait for detection and mesh reformation
    sleep 30
    
    # Check mesh topology
    stratoswarm cluster mesh-status > mesh_during_partition.txt
    
    # Heal partition
    echo "Healing network partition..."
    ssh pi@192.168.1.101 "sudo iptables -D INPUT -s 192.168.1.100 -j DROP"
    ssh pi@192.168.1.102 "sudo iptables -D INPUT -s 192.168.1.100 -j DROP"
    
    # Wait for mesh to reform
    sleep 10
    
    END=$(date +%s)
    RECOVERY_TIME=$((END - START))
    
    echo "Partition recovery completed in: ${RECOVERY_TIME}s"
    stratoswarm cluster mesh-status > mesh_after_recovery.txt
    
    # Compare mesh states
    diff mesh_during_partition.txt mesh_after_recovery.txt
}

# Test cascading failures
test_cascading_failure() {
    echo -e "\n=== Cascading Failure Test ==="
    
    # Deploy critical workload with replicas
    stratoswarm deploy - <<EOF
name: critical-service
replicas: 3
placement:
  constraints:
    - node.type == rpi
spec:
  image: alpine
  command: ["sh", "-c", "while true; do echo alive; sleep 1; done"]
EOF
    
    sleep 10
    
    # Fail nodes progressively
    for i in 1 2; do
        echo "Failing rpi-$i..."
        ssh pi@192.168.1.10$i "sudo systemctl stop docker"
        sleep 10
        
        # Check service availability
        RUNNING=$(stratoswarm ps | grep critical-service | grep Running | wc -l)
        echo "Running replicas: $RUNNING/3"
    done
    
    # Verify service maintains availability
    if [ $RUNNING -ge 1 ]; then
        echo "✓ Service maintained availability"
    else
        echo "✗ Service failed completely"
    fi
}

# Run all failure tests
test_node_failure
test_network_partition
test_cascading_failure

# Generate failure recovery report
./scripts/analyze_failure_recovery.py \
    --detection-times detection_times.json \
    --recovery-times recovery_times.json \
    --output results/failure_recovery_report.html
```

## Performance Metrics Collection

### Continuous Monitoring Setup
```bash
#!/bin/bash
# File: benchmarks/scripts/setup_rpi_monitoring.sh

# Install monitoring on each RPi
for i in {1..5}; do
    ssh pi@192.168.1.10$i << 'EOF'
        # Install Prometheus node exporter
        wget https://github.com/prometheus/node_exporter/releases/download/v1.3.1/node_exporter-1.3.1.linux-armv7.tar.gz
        tar xvf node_exporter-1.3.1.linux-armv7.tar.gz
        sudo cp node_exporter-1.3.1.linux-armv7/node_exporter /usr/local/bin/
        
        # Create systemd service
        sudo tee /etc/systemd/system/node_exporter.service > /dev/null <<'SERVICE'
[Unit]
Description=Node Exporter
After=network.target

[Service]
User=pi
Group=pi
Type=simple
ExecStart=/usr/local/bin/node_exporter

[Install]
WantedBy=multi-user.target
SERVICE
        
        sudo systemctl daemon-reload
        sudo systemctl enable --now node_exporter
        
        # Custom swarmlet metrics
        docker exec swarmlet stratoswarm-metrics --prometheus-format
EOF
done
```

### Results Analysis Script
```python
#!/usr/bin/env python3
# File: benchmarks/scripts/analyze_swarmlet_results.py

import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

class SwarmletAnalyzer:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        
    def analyze_join_performance(self):
        """Analyze swarmlet join times"""
        with open(f"{self.results_dir}/join_times.json") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        # Plot join time breakdown
        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(x='node', y=['hardware_profiling', 'network_discovery', 
                             'cert_exchange', 'total'], kind='bar', ax=ax)
        ax.set_title('Swarmlet Join Time Breakdown')
        ax.set_ylabel('Time (seconds)')
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/join_time_analysis.png")
        
        # Statistics
        print(f"Average join time: {df['total'].mean():.2f}s")
        print(f"Max join time: {df['total'].max():.2f}s")
        print(f"Join time variance: {df['total'].std():.2f}s")
    
    def analyze_resource_usage(self):
        """Analyze RPi resource utilization"""
        metrics = pd.read_csv(f"{self.results_dir}/resource_metrics.csv")
        
        # Plot resource usage over time
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # CPU usage
        for node in metrics['node'].unique():
            node_data = metrics[metrics['node'] == node]
            axes[0].plot(node_data['timestamp'], node_data['cpu_percent'], 
                        label=node, alpha=0.7)
        axes[0].set_title('CPU Usage Over Time')
        axes[0].set_ylabel('CPU %')
        axes[0].legend()
        
        # Memory usage
        for node in metrics['node'].unique():
            node_data = metrics[metrics['node'] == node]
            axes[1].plot(node_data['timestamp'], node_data['memory_mb'], 
                        label=node, alpha=0.7)
        axes[1].set_title('Memory Usage Over Time')
        axes[1].set_ylabel('Memory (MB)')
        
        # Network throughput
        for node in metrics['node'].unique():
            node_data = metrics[metrics['node'] == node]
            axes[2].plot(node_data['timestamp'], node_data['network_mbps'], 
                        label=node, alpha=0.7)
        axes[2].set_title('Network Throughput')
        axes[2].set_ylabel('Mbps')
        axes[2].set_xlabel('Time')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/resource_usage_analysis.png")
    
    def generate_report(self):
        """Generate comprehensive benchmark report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "cluster_configuration": {
                "gpu_nodes": 1,
                "rpi_nodes": 5,
                "total_nodes": 6
            },
            "performance_summary": {
                "avg_join_time": self.get_avg_join_time(),
                "mesh_formation_time": self.get_mesh_formation_time(),
                "coordination_latency": self.get_coordination_latency(),
                "workload_scheduling_time": self.get_scheduling_time(),
                "failure_recovery_time": self.get_recovery_time()
            },
            "resource_efficiency": {
                "swarmlet_memory_footprint": self.get_memory_footprint(),
                "idle_cpu_usage": self.get_idle_cpu(),
                "network_overhead": self.get_network_overhead()
            },
            "recommendations": self.generate_recommendations()
        }
        
        with open(f"{self.results_dir}/swarmlet_benchmark_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

# Run analysis
if __name__ == "__main__":
    analyzer = SwarmletAnalyzer("results/swarmlet_benchmarks")
    analyzer.analyze_join_performance()
    analyzer.analyze_resource_usage()
    report = analyzer.generate_report()
    print("Benchmark report generated:", report)
```

## Expected Results & Validation

### Performance Targets

| Metric | Target | Acceptable | Current |
|--------|--------|------------|---------|
| Swarmlet Join Time | <30s | <60s | TBD |
| Memory Footprint | <50MB | <100MB | TBD |
| Coordination Latency | <10ms | <50ms | TBD |
| Failure Detection | <10s | <30s | TBD |
| Recovery Time | <30s | <60s | TBD |
| Network Overhead | <5% | <10% | TBD |

### Success Criteria

1. **Heterogeneous Integration**: Seamless GPU + RPi coordination
2. **Resource Efficiency**: Swarmlet runs efficiently on 1GB RPi
3. **Network Resilience**: Handles intermittent connectivity
4. **Scalability**: Linear performance with 5+ nodes
5. **Failure Recovery**: Automatic workload migration

## Troubleshooting

### Common Issues

1. **RPi Out of Memory**
   ```bash
   # Increase swap
   sudo dphys-swapfile swapoff
   sudo nano /etc/dphys-swapfile  # Set CONF_SWAPSIZE=2048
   sudo dphys-swapfile setup
   sudo dphys-swapfile swapon
   ```

2. **Network Discovery Failures**
   ```bash
   # Check mDNS
   sudo systemctl status avahi-daemon
   # Check firewall
   sudo iptables -L
   ```

3. **Docker Permission Issues**
   ```bash
   # Fix docker group
   sudo usermod -aG docker $USER
   newgrp docker
   ```

## Next Steps

After completing swarmlet benchmarks:

1. **Scale Testing**: Add more RPi nodes (10-20)
2. **Real Workloads**: Deploy actual applications
3. **Power Efficiency**: Measure power consumption
4. **Edge Scenarios**: Test with limited bandwidth (3G/4G)
5. **Production Validation**: Long-term stability testing