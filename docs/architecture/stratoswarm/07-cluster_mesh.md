# Stratoswarm Cluster Mesh - Adding Heterogeneous Nodes

## Overview

Stratoswarm's cluster mesh is designed to seamlessly integrate any compute resource - from powerful GPU servers to Raspberry Pis to your laptop. Each node contributes its unique capabilities to the swarm while maintaining security and performance isolation.

## Zero-Config Node Addition

### 1. Single Command Join

```bash
# On your laptop
curl -sSL https://get.stratoswarm.io | sh
stratoswarm join cluster.example.com

# On Raspberry Pi
wget -qO- https://get.stratoswarm.io | sh
stratoswarm join cluster.example.com --node-class=edge

# That's it. No configuration needed.
```

### 2. Automatic Node Discovery and Profiling

```rust
pub struct NodeDiscovery {
    hardware_profiler: HardwareProfiler,
    network_prober: NetworkProber,
    capability_detector: CapabilityDetector,
    trust_evaluator: TrustEvaluator,
}

impl NodeDiscovery {
    pub async fn profile_node(&self) -> Result<NodeProfile, Error> {
        // Detect hardware capabilities
        let hardware = self.hardware_profiler.profile()?;

        let profile = NodeProfile {
            // CPU capabilities
            cpu: CpuProfile {
                cores: hardware.cpu_cores,
                architecture: hardware.arch, // x86_64, arm64, etc
                features: hardware.cpu_features, // AVX, NEON, etc
                performance_class: self.classify_cpu_performance(&hardware)?,
            },

            // Memory tiers
            memory_tiers: self.detect_memory_tiers()?,

            // GPU capabilities (if any)
            gpu: self.detect_gpu_capabilities()?,

            // Network characteristics
            network: NetworkProfile {
                bandwidth: self.network_prober.measure_bandwidth().await?,
                latency_map: self.network_prober.measure_latency_to_peers().await?,
                nat_type: self.network_prober.detect_nat_type()?,
                ipv6_capable: self.network_prober.has_ipv6()?,
            },

            // Storage capabilities
            storage: self.profile_storage()?,

            // Special capabilities
            capabilities: Capabilities {
                has_tpm: hardware.has_tpm,
                has_secure_enclave: hardware.has_secure_enclave,
                can_run_kernel_modules: self.check_kernel_module_support()?,
                power_profile: self.detect_power_profile()?, // battery, plugged, etc
            },

            // Trust and security
            trust: TrustProfile {
                attestation: self.perform_attestation()?,
                owner_verified: false, // Will be verified during join
                security_level: self.evaluate_security_level()?,
            },
        };

        Ok(profile)
    }

    fn classify_cpu_performance(&self, hardware: &Hardware) -> PerformanceClass {
        match (hardware.cpu_cores, hardware.cpu_freq_ghz) {
            (cores, freq) if cores >= 32 && freq >= 3.0 => PerformanceClass::DataCenter,
            (cores, freq) if cores >= 8 && freq >= 2.5 => PerformanceClass::Workstation,
            (cores, freq) if cores >= 4 && freq >= 2.0 => PerformanceClass::Desktop,
            (cores, freq) if cores >= 2 && freq >= 1.5 => PerformanceClass::Laptop,
            _ => PerformanceClass::Edge, // Raspberry Pi, embedded
        }
    }
}
```

### 3. Heterogeneous Node Classes

```rust
pub enum NodeClass {
    // High-performance nodes
    DataCenter {
        gpus: Vec<GpuInfo>,
        numa_nodes: u8,
        network_bandwidth: Bandwidth,
    },

    // Developer machines
    Workstation {
        gpu: Option<GpuInfo>,
        available_hours: Schedule, // "9am-5pm weekdays"
        reserved_resources: Resources, // Keep some for local use
    },

    // Personal devices
    Laptop {
        battery_powered: bool,
        mobility_pattern: MobilityPattern, // stationary, mobile, etc
        sleep_schedule: Schedule,
    },

    // Edge devices
    Edge {
        device_type: EdgeType, // RaspberryPi, Jetson, etc
        power_budget: Watts,
        environmental: Environmental, // temperature range, outdoor, etc
    },

    // Specialized
    Storage {
        capacity: ByteSize,
        tier: StorageTier,
        reliability: Reliability,
    },
}

impl NodeClass {
    pub fn auto_detect() -> Self {
        // Intelligent detection based on hardware profile
        let profile = NodeDiscovery::profile_node().unwrap();

        match profile {
            p if p.has_datacenter_features() => NodeClass::DataCenter { /* ... */ },
            p if p.is_laptop() => NodeClass::Laptop { /* ... */ },
            p if p.is_edge_device() => NodeClass::Edge { /* ... */ },
            _ => NodeClass::Workstation { /* ... */ },
        }
    }
}
```

## Intelligent Node Integration

### 1. Capability-Based Work Assignment

```rust
pub struct CapabilityScheduler {
    node_registry: NodeRegistry,
    capability_matcher: CapabilityMatcher,
    work_distributor: WorkDistributor,
}

impl CapabilityScheduler {
    pub async fn assign_work(&self, job: Job) -> Result<Assignment, Error> {
        // Match job requirements to node capabilities
        let suitable_nodes = self.capability_matcher.find_suitable_nodes(&job)?;

        // Consider node-specific constraints
        let assignments = suitable_nodes.into_iter().map(|node| {
            match node.class {
                NodeClass::Laptop { battery_powered: true, .. } => {
                    // Don't assign long-running jobs to battery-powered devices
                    if job.expected_duration > Duration::from_hours(2) {
                        return None;
                    }

                    // Prefer CPU-only tasks for battery devices
                    if job.requires_gpu() {
                        return None;
                    }

                    Some(Assignment {
                        node: node.id,
                        resources: job.resources.scale(0.5), // Use only 50% to preserve battery
                        priority: Priority::Low,
                    })
                }

                NodeClass::Edge { power_budget, .. } => {
                    // Check if job fits within power budget
                    if job.estimated_power() > power_budget {
                        return None;
                    }

                    Some(Assignment {
                        node: node.id,
                        resources: job.resources,
                        priority: Priority::Normal,
                    })
                }

                NodeClass::DataCenter { .. } => {
                    // Data center nodes can handle anything
                    Some(Assignment {
                        node: node.id,
                        resources: job.resources,
                        priority: Priority::High,
                    })
                }

                _ => Some(Assignment::default()),
            }
        }).flatten().collect();

        Ok(self.work_distributor.optimize_assignment(assignments)?)
    }
}
```

### 2. Network-Aware Mesh Formation

```rust
pub struct MeshTopology {
    nodes: HashMap<NodeId, NodeInfo>,
    latency_matrix: LatencyMatrix,
    bandwidth_map: BandwidthMap,
    topology_optimizer: TopologyOptimizer,
}

impl MeshTopology {
    pub async fn add_node(&mut self, new_node: NodeInfo) -> Result<MeshUpdate, Error> {
        // Measure network characteristics to existing nodes
        let latencies = self.measure_latencies_to_all(&new_node).await?;
        let bandwidth = self.probe_bandwidth(&new_node).await?;

        // Determine optimal mesh connections
        let connections = match new_node.network_profile {
            NetworkProfile::HighBandwidth { .. } => {
                // Connect to multiple peers for redundancy
                self.topology_optimizer.full_mesh_connections(&new_node, 5)?
            }

            NetworkProfile::Mobile { .. } => {
                // Limited connections, prefer geographically close
                self.topology_optimizer.geographic_connections(&new_node, 3)?
            }

            NetworkProfile::NATRestricted { .. } => {
                // Use relay nodes
                self.topology_optimizer.relay_connections(&new_node)?
            }

            NetworkProfile::Edge { bandwidth_limit, .. } => {
                // Minimal connections to preserve bandwidth
                self.topology_optimizer.minimal_connections(&new_node, 2)?
            }
        };

        // Update mesh topology
        self.establish_connections(&new_node, connections).await?;

        // Propagate node information via CRDT
        self.broadcast_node_addition(&new_node).await?;

        Ok(MeshUpdate {
            added_node: new_node.id,
            new_connections: connections,
            topology_changes: self.rebalance_after_addition(&new_node)?,
        })
    }
}
```

### 3. Security for Untrusted Nodes

```rust
pub struct UntrustedNodeSecurity {
    trust_manager: TrustManager,
    isolation_enforcer: IsolationEnforcer,
    work_verifier: WorkVerifier,
}

impl UntrustedNodeSecurity {
    pub async fn handle_untrusted_node(&self, node: &NodeInfo) -> Result<SecurityPolicy, Error> {
        let trust_level = self.trust_manager.evaluate_node(node)?;

        let policy = match trust_level {
            TrustLevel::Untrusted => {
                SecurityPolicy {
                    // Only stateless, verifiable work
                    allowed_workloads: vec![WorkloadType::Stateless, WorkloadType::Verifiable],

                    // Strong isolation
                    isolation: IsolationLevel::Maximum,
                    network_policy: NetworkPolicy::NoDirectAccess,

                    // All work must be verified
                    verification: VerificationPolicy::MandatoryWithRedundancy(3),

                    // No access to sensitive data
                    data_access: DataAccess::PublicOnly,

                    // Limited resource allocation
                    resource_limit: ResourceLimit::Minimal,
                }
            }

            TrustLevel::Personal => {
                // Your own devices get more trust
                SecurityPolicy {
                    allowed_workloads: vec![WorkloadType::All],
                    isolation: IsolationLevel::Standard,
                    network_policy: NetworkPolicy::MeshAccess,
                    verification: VerificationPolicy::Periodic,
                    data_access: DataAccess::PersonalData,
                    resource_limit: ResourceLimit::Fair,
                }
            }

            TrustLevel::Verified => {
                // TPM-attested or known good nodes
                SecurityPolicy {
                    allowed_workloads: vec![WorkloadType::All],
                    isolation: IsolationLevel::Minimal,
                    network_policy: NetworkPolicy::FullAccess,
                    verification: VerificationPolicy::TrustButVerify,
                    data_access: DataAccess::All,
                    resource_limit: ResourceLimit::Unlimited,
                }
            }
        };

        // Enforce policy at kernel level
        self.isolation_enforcer.enforce(&node.id, &policy).await?;

        Ok(policy)
    }
}
```

## Dynamic Resource Contribution

### 1. Laptop-Aware Scheduling

```rust
pub struct LaptopAwareScheduler {
    power_monitor: PowerMonitor,
    user_activity: UserActivityMonitor,
    thermal_monitor: ThermalMonitor,
}

impl LaptopAwareScheduler {
    pub async fn laptop_scheduling_policy(&self, laptop: &LaptopNode) -> SchedulingPolicy {
        let mut policy = SchedulingPolicy::default();

        // Check power state
        if self.power_monitor.on_battery() {
            policy.max_cpu_usage = 25.0; // Preserve battery
            policy.disable_gpu = true;
            policy.prefer_suspended_work = true;
        }

        // Check user activity
        if self.user_activity.user_active() {
            policy.max_cpu_usage = 10.0; // Don't interfere with user
            policy.nice_level = 19; // Lowest priority
            policy.pause_on_user_input = true;
        }

        // Check thermal state
        let temp = self.thermal_monitor.cpu_temp();
        if temp > 70.0 {
            policy.thermal_throttle = true;
            policy.max_cpu_usage *= 0.5;
        }

        // Time-based policies
        let hour = Local::now().hour();
        match hour {
            9..=17 => policy.work_hours_mode = true, // Reduced during work
            22..=6 => policy.night_mode = true, // Can use more resources
            _ => {}
        }

        policy
    }
}
```

### 2. Raspberry Pi Optimization

```rust
pub struct EdgeNodeOptimizer {
    thermal_limits: ThermalLimits,
    power_monitor: PowerMonitor,
    storage_manager: StorageManager,
}

impl EdgeNodeOptimizer {
    pub fn optimize_for_pi(&self, pi_node: &EdgeNode) -> OptimizationStrategy {
        OptimizationStrategy {
            // Use ARM-optimized agents
            prefer_architecture: Architecture::ARM64,

            // Memory is precious on Pi
            memory_optimization: MemoryStrategy::Aggressive,
            swap_strategy: SwapStrategy::ZRAM, // Compressed swap in RAM

            // SD card optimization
            storage_strategy: StorageStrategy {
                minimize_writes: true,
                use_ram_disk: true,
                log_aggregation: Duration::from_hours(1),
            },

            // Thermal management
            thermal_strategy: ThermalStrategy {
                throttle_at: 70.0, // Celsius
                shutdown_at: 80.0,
                prefer_bursty_workloads: true,
            },

            // Network optimization
            network_strategy: NetworkStrategy {
                compress_traffic: true,
                batch_communications: true,
                prefer_local_work: true,
            },

            // Power efficiency
            power_strategy: PowerStrategy {
                idle_sleep: true,
                cpu_governor: "powersave",
                disable_unnecessary_peripherals: true,
            },
        }
    }
}
```

## Mesh Communication Protocol

### 1. Heterogeneous Node Communication

```rust
pub struct HeterogeneousMesh {
    transport: AdaptiveTransport,
    protocol: MeshProtocol,
    compressor: AdaptiveCompressor,
}

impl HeterogeneousMesh {
    pub async fn setup_communication(&self, node_a: &Node, node_b: &Node) -> Result<Channel, Error> {
        // Choose optimal transport based on node types
        let transport = match (node_a.class, node_b.class) {
            (NodeClass::DataCenter, NodeClass::DataCenter) => {
                // High bandwidth, low latency
                Transport::RDMA
            }
            (_, NodeClass::Edge { .. }) => {
                // Bandwidth conscious
                Transport::QUIC
            }
            (NodeClass::Laptop { .. }, _) => {
                // May have NAT, firewall issues
                Transport::WebRTC
            }
            _ => Transport::TCP,
        };

        // Adaptive compression based on bandwidth
        let compression = match self.measure_bandwidth(node_a, node_b).await? {
            bw if bw < Bandwidth::from_mbps(1) => Compression::Aggressive,
            bw if bw < Bandwidth::from_mbps(10) => Compression::Standard,
            _ => Compression::None,
        };

        // Create channel with appropriate settings
        let channel = Channel {
            transport,
            compression,
            encryption: self.select_encryption(node_a, node_b)?,
            qos: self.determine_qos(node_a, node_b)?,
        };

        Ok(channel)
    }
}
```

### 2. Work Migration Between Nodes

```rust
pub struct WorkMigration {
    migration_planner: MigrationPlanner,
    state_serializer: StateSerializer,
    network_optimizer: NetworkOptimizer,
}

impl WorkMigration {
    pub async fn migrate_work(&self, work: &Work, from: &Node, to: &Node) -> Result<(), Error> {
        // Check if migration makes sense
        let migration_benefit = self.migration_planner.calculate_benefit(work, from, to)?;

        if migration_benefit < 0.0 {
            return Err(Error::MigrationNotBeneficial);
        }

        // Optimize based on node types
        match (from.class, to.class) {
            (NodeClass::Edge { .. }, NodeClass::DataCenter { .. }) => {
                // Edge to datacenter - probably offloading heavy computation
                self.migrate_edge_to_datacenter(work, from, to).await?
            }

            (NodeClass::Laptop { battery_powered: true, .. }, _) => {
                // Laptop on battery - migrate ASAP to preserve power
                self.emergency_migrate(work, from, to).await?
            }

            (NodeClass::DataCenter { .. }, NodeClass::Edge { .. }) => {
                // Datacenter to edge - probably for locality
                self.migrate_with_compression(work, from, to).await?
            }

            _ => self.standard_migration(work, from, to).await?,
        }

        Ok(())
    }

    async fn migrate_edge_to_datacenter(&self, work: &Work, from: &Node, to: &Node) -> Result<(), Error> {
        // Serialize state efficiently for limited bandwidth
        let state = self.state_serializer.serialize_compressed(work)?;

        // Use incremental transfer if possible
        if let Some(checkpoint) = work.last_checkpoint() {
            let delta = self.state_serializer.create_delta(checkpoint, &state)?;
            self.network_optimizer.transfer_delta(from, to, delta).await?;
        } else {
            self.network_optimizer.transfer_full(from, to, state).await?;
        }

        Ok(())
    }
}
```

## Node Lifecycle Management

### 1. Graceful Node Departure

```rust
pub struct NodeLifecycle {
    work_redistributor: WorkRedistributor,
    mesh_rebalancer: MeshRebalancer,
    state_manager: StateManager,
}

impl NodeLifecycle {
    pub async fn handle_node_departure(&self, departing_node: &Node) -> Result<(), Error> {
        match departing_node.departure_type() {
            DepartureType::Planned => {
                // Laptop shutting down, Pi being unplugged
                self.planned_departure(departing_node).await?
            }

            DepartureType::LowBattery => {
                // Emergency migration
                self.emergency_departure(departing_node).await?
            }

            DepartureType::UserRequested => {
                // User wants their laptop back for gaming
                self.immediate_departure(departing_node).await?
            }

            DepartureType::Unexpected => {
                // Network failure, crash
                self.handle_node_failure(departing_node).await?
            }
        }

        Ok(())
    }

    async fn planned_departure(&self, node: &Node) -> Result<(), Error> {
        // Gracefully migrate all work
        let work_items = self.state_manager.get_node_work(node)?;

        for work in work_items {
            // Find best alternative node
            let target = self.work_redistributor.find_best_target(&work, node)?;

            // Migrate with full state
            self.migrate_work(&work, node, &target).await?;
        }

        // Update mesh topology
        self.mesh_rebalancer.remove_node(node).await?;

        // Mark node as gracefully departed
        node.set_state(NodeState::Departed);

        Ok(())
    }
}
```

### 2. Intermittent Availability

```rust
pub struct IntermittentNodeManager {
    availability_predictor: AvailabilityPredictor,
    work_scheduler: WorkScheduler,
    state_cache: StateCache,
}

impl IntermittentNodeManager {
    pub async fn handle_intermittent_node(&self, node: &Node) -> Result<(), Error> {
        // Learn availability patterns
        let pattern = self.availability_predictor.learn_pattern(node).await?;

        match pattern {
            AvailabilityPattern::BusinessHours => {
                // Laptop that's available 9-5
                self.work_scheduler.schedule_only_during(node, "9am-5pm")?;
            }

            AvailabilityPattern::Overnight => {
                // Desktop that's idle at night
                self.work_scheduler.schedule_batch_work(node, "10pm-6am")?;
            }

            AvailabilityPattern::Sporadic => {
                // Mobile device with unpredictable availability
                self.work_scheduler.only_stateless_work(node)?;
            }

            AvailabilityPattern::Reliable => {
                // Always-on Pi or server
                self.work_scheduler.any_work(node)?;
            }
        }

        // Cache state for quick resume
        if pattern.has_predictable_offline_periods() {
            self.state_cache.enable_aggressive_caching(node)?;
        }

        Ok(())
    }
}
```

## Example Scenarios

### Scenario 1: Adding Your Laptop

```bash
# Install Stratoswarm
curl -sSL https://get.stratoswarm.io | sh

# Join with preferences
stratoswarm join cluster.example.com \
  --when="nights-and-weekends" \
  --reserve-cpu="50%" \
  --no-work-during="zoom"

# Stratoswarm automatically:
# - Detects 8-core CPU, 32GB RAM, RTX 3070
# - Notices you're on WiFi (reduces network usage)
# - Learns your usage patterns
# - Only schedules work when laptop is idle
# - Migrates work before you unplug
```

### Scenario 2: Raspberry Pi Cluster

```bash
# On each Pi
stratoswarm join cluster.example.com --node-class=edge

# Stratoswarm automatically:
# - Detects ARM64 architecture
# - Optimizes for SD card longevity
# - Manages thermal throttling
# - Assigns edge-appropriate workloads
# - Forms mesh with nearby Pis for redundancy
```

### Scenario 3: Mixed Infrastructure

```yaml
# Your cluster might have:
- 2x GPU servers (datacenter)
- 5x developer workstations (office)
- 20x laptops (remote workers)
- 50x Raspberry Pis (edge locations)
- 3x Mac Studios (creative team)

# Stratoswarm automatically:
- Routes GPU work to servers
- Uses workstations during off-hours
- Leverages laptops when available
- Deploys edge work to Pis
- Assigns macOS-specific work to Macs
```

## Benefits of Heterogeneous Mesh

1. **Resource Efficiency** - Use idle capacity everywhere
2. **Cost Optimization** - Reduce cloud spending by using existing hardware
3. **Edge Computing** - Process data where it's generated
4. **Resilience** - No single point of failure
5. **Flexibility** - Work runs wherever it makes sense

## Security Considerations

```rust
pub struct HeterogeneousSecurityPolicy {
    pub untrusted_nodes: UntrustedPolicy {
        // Public compute only
        workload_types: vec![WorkloadType::PublicCompute],
        verification: VerificationPolicy::AlwaysVerify,
        isolation: IsolationLevel::Maximum,
    },

    pub personal_devices: PersonalPolicy {
        // Your own devices
        workload_types: vec![WorkloadType::Personal],
        data_encryption: EncryptionPolicy::AtRest,
        network_isolation: NetworkPolicy::VPN,
    },

    pub corporate_devices: CorporatePolicy {
        // Company-owned hardware
        workload_types: vec![WorkloadType::Any],
        compliance: CompliancePolicy::Corporate,
        audit_logging: AuditPolicy::Full,
    },
}
```

## Conclusion

Stratoswarm's heterogeneous mesh seamlessly integrates any compute resource:

- **Zero-config joining** - One command to add any node
- **Intelligent work placement** - Right work on right hardware
- **Adaptive behavior** - Responds to node characteristics
- **Graceful handling** - Manages intermittent availability
- **Security-first** - Trust-based isolation and verification

Whether it's your laptop contributing spare cycles or a Raspberry Pi at the edge, every node becomes a valuable part of the swarm.
