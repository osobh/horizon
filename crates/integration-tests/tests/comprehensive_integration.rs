//! Comprehensive integration tests for ExoRust Phase 1 components
//! Tests cross-component workflows including runtime, isolation, storage, networking, and evolution

use exorust_evolution::{
    EvolutionEngine, GeneticEvolutionEngine, Population, SimpleFitnessFunction,
};
use exorust_memory::{GpuMemoryAllocator, MemoryManager};
use exorust_net::{MemoryNetwork, Message, MessageType, Network, ZeroCopyTransport};
use exorust_runtime::{ContainerConfig, ContainerRuntime, KernelSignature, SecureContainerRuntime};
use exorust_storage::{MemoryStorage, NvmeConfig, NvmeStorage, Storage};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Test complete container lifecycle with isolation
#[tokio::test]
async fn test_secure_container_lifecycle_integration() {
    let runtime = SecureContainerRuntime::new();

    // Create container with custom configuration
    let config = ContainerConfig {
        memory_limit_bytes: 1024 * 1024 * 50, // 50MB
        gpu_compute_units: 2,
        timeout_seconds: Some(120),
        agent_type: "test-agent".to_string(),
        environment: {
            let mut env = std::collections::HashMap::new();
            env.insert("TEST_MODE".to_string(), "true".to_string());
            env
        },
    };

    // Create container with isolation
    let container = runtime
        .create_container(config)
        .await
        .expect("Failed to create container");
    let container_id = container.id();

    // Verify isolation context was created
    let isolation_stats = runtime
        .get_isolation_stats(container_id)
        .await
        .expect("Failed to get isolation stats");
    assert_eq!(isolation_stats.container_id, container_id);
    assert_eq!(isolation_stats.memory_usage_bytes, 0);
    assert_eq!(isolation_stats.kernels_executed, 0);

    // Start container
    runtime
        .start_container(container_id)
        .await
        .expect("Failed to start container");

    // Verify container state
    let stats = runtime
        .container_stats(container_id)
        .await
        .expect("Failed to get container stats");
    assert_eq!(stats.state, "Running");

    // Allocate memory within quota
    let memory_address = runtime
        .allocate_memory(container_id, 1024 * 1024)
        .await
        .expect("Failed to allocate memory");
    assert!(memory_address > 0);

    // Launch verified kernel
    let kernel_sig = KernelSignature {
        prompt_hash: "deadbeefcafebabe".to_string(),
        ptx_hash: "0123456789abcdef".to_string(),
        agent_id: Some("test-agent".to_string()),
        signature: None,
        created_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };

    runtime
        .launch_kernel(container_id, kernel_sig)
        .await
        .expect("Failed to launch kernel");

    // Verify execution was recorded
    let updated_isolation_stats = runtime
        .get_isolation_stats(container_id)
        .await
        .expect("Failed to get updated isolation stats");
    assert_eq!(updated_isolation_stats.kernels_executed, 1);

    // Stop and remove container
    runtime
        .stop_container(container_id)
        .await
        .expect("Failed to stop container");
    runtime
        .remove_container(container_id)
        .await
        .expect("Failed to remove container");

    // Verify cleanup
    let containers = runtime
        .list_containers()
        .await
        .expect("Failed to list containers");
    assert!(containers.is_empty());
}

/// Test memory-storage-network pipeline
#[tokio::test]
async fn test_memory_storage_network_pipeline() {
    // Initialize components
    let memory_allocator = Arc::new(
        GpuMemoryAllocator::new(1024 * 1024 * 100).expect("Failed to create memory allocator"),
    );
    let storage = Arc::new(MemoryStorage::new(1024 * 1024 * 50));
    let network = Arc::new(MemoryNetwork::new("pipeline-node".to_string()));

    // Stage 1: Allocate memory for data processing
    let memory_handle = memory_allocator
        .allocate(1024 * 1024 * 10)
        .await
        .expect("Failed to allocate memory");

    // Stage 2: Generate and store processed data
    let processed_data = (0..1000).map(|i| (i % 256) as u8).collect::<Vec<u8>>();
    let data_key = format!("processed_data_{memory_handle.id}");

    storage
        .store(&data_key, &processed_data)
        .await
        .expect("Failed to store processed data");

    // Stage 3: Send notification via network
    let notification_msg = Message {
        msg_type: MessageType::ResourceResponse,
        payload: data_key.as_bytes().to_vec(),
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };

    network
        .send("processing-complete", notification_msg)
        .await
        .expect("Failed to send notification");

    // Stage 4: Simulate remote node receiving and processing notification
    let remote_network = MemoryNetwork::new("remote-node".to_string());
    remote_network
        .simulate_receive(
            "pipeline-node".to_string(),
            Message {
                msg_type: MessageType::ResourceResponse,
                payload: data_key.as_bytes().to_vec(),
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        )
        .await
        .expect("Failed to simulate remote receive");

    let (from, msg) = remote_network
        .receive()
        .await
        .expect("Failed to receive at remote node");
    assert_eq!(from, "pipeline-node");

    // Stage 5: Remote node retrieves data using received key
    let received_key = String::from_utf8(msg.payload).expect("Invalid key in message");
    let retrieved_data = storage
        .retrieve(&received_key)
        .await
        .expect("Failed to retrieve data");
    assert_eq!(retrieved_data, processed_data);

    // Stage 6: Cleanup
    memory_allocator
        .deallocate(memory_handle)
        .await
        .expect("Failed to deallocate memory");
    storage
        .delete(&data_key)
        .await
        .expect("Failed to delete stored data");

    // Verify final state
    let storage_stats = storage.stats().await.expect("Failed to get storage stats");
    assert_eq!(storage_stats.total_files, 0);

    let network_stats = network.stats().await.expect("Failed to get network stats");
    assert_eq!(network_stats.messages_sent, 1);
}

/// Test evolution engine with storage persistence
#[tokio::test]
async fn test_evolution_storage_integration() {
    let fitness_fn = Arc::new(SimpleFitnessFunction);
    let evolution_engine = GeneticEvolutionEngine::with_defaults(fitness_fn);
    let storage = Arc::new(MemoryStorage::new(1024 * 1024 * 10));

    // Initialize population
    let mut population = evolution_engine
        .initialize_population(20)
        .await
        .expect("Failed to initialize population");

    // Store initial population
    let population_data = serde_json::to_vec(&population).expect("Failed to serialize population");
    storage
        .store("population_gen_0", &population_data)
        .await
        .expect("Failed to store initial population");

    // Evolve for several generations
    for generation in 1..=5 {
        evolution_engine
            .evolve_generation(&mut population)
            .await
            .expect("Failed to evolve generation");

        // Store population state after each generation
        let gen_data = serde_json::to_vec(&population).expect("Failed to serialize population");
        let gen_key = format!("population_gen_{generation}");
        storage
            .store(&gen_key, &gen_data)
            .await
            .expect("Failed to store population generation");
    }

    // Verify evolution progress by comparing generations
    let gen_0_data = storage
        .retrieve("population_gen_0")
        .await
        .expect("Failed to retrieve gen 0");
    let gen_5_data = storage
        .retrieve("population_gen_5")
        .await
        .expect("Failed to retrieve gen 5");

    let gen_0_pop: Population =
        serde_json::from_slice(&gen_0_data).expect("Failed to deserialize gen 0");
    let gen_5_pop: Population =
        serde_json::from_slice(&gen_5_data).expect("Failed to deserialize gen 5");

    // Verify population evolved (generation numbers should be different)
    assert_eq!(gen_0_pop.generation, 0);
    assert_eq!(gen_5_pop.generation, 5);

    // Check that fitness values exist and potentially improved
    let gen_0_avg_fitness = gen_0_pop.average_fitness();
    let gen_5_avg_fitness = gen_5_pop.average_fitness();

    // Both should have valid fitness scores
    assert!(gen_0_avg_fitness >= 0.0);
    assert!(gen_5_avg_fitness >= 0.0);

    // Verify storage contains all generations
    let storage_stats = storage.stats().await.expect("Failed to get storage stats");
    assert_eq!(storage_stats.total_files, 6); // gen_0 through gen_5
}

/// Test multi-container isolation and resource management
#[tokio::test]
async fn test_multi_container_isolation() {
    let runtime = SecureContainerRuntime::new();

    // Create multiple containers with different quotas
    let configs = vec![
        ContainerConfig {
            memory_limit_bytes: 1024 * 1024 * 10, // 10MB
            gpu_compute_units: 1,
            timeout_seconds: Some(60),
            agent_type: "agent-a".to_string(),
            environment: std::collections::HashMap::new(),
        },
        ContainerConfig {
            memory_limit_bytes: 1024 * 1024 * 20, // 20MB
            gpu_compute_units: 2,
            timeout_seconds: Some(120),
            agent_type: "agent-b".to_string(),
            environment: std::collections::HashMap::new(),
        },
        ContainerConfig {
            memory_limit_bytes: 1024 * 1024 * 5, // 5MB
            gpu_compute_units: 1,
            timeout_seconds: Some(30),
            agent_type: "agent-c".to_string(),
            environment: std::collections::HashMap::new(),
        },
    ];

    let mut container_ids = Vec::new();

    // Create and start all containers
    for (i, config) in configs.into_iter().enumerate() {
        let container = runtime
            .create_container(config)
            .await
            .expect(&format!("Failed to create container {i}"));
        let container_id = container.id().to_string();

        runtime
            .start_container(&container_id)
            .await
            .expect(&format!("Failed to start container {i}"));
        container_ids.push(container_id);
    }

    // Test isolation by attempting memory allocation within each container's quota
    let allocations = vec![
        (0, 1024 * 1024 * 5),  // 5MB within 10MB quota
        (1, 1024 * 1024 * 15), // 15MB within 20MB quota
        (2, 1024 * 1024 * 3),  // 3MB within 5MB quota
    ];

    for (container_idx, allocation_size) in allocations {
        let container_id = &container_ids[container_idx];
        let result = runtime.allocate_memory(container_id, allocation_size).await;
        assert!(
            result.is_ok(),
            "Container {} should be able to allocate {} bytes",
            container_idx,
            allocation_size
        );
    }

    // Test quota enforcement by trying to exceed limits
    let container_c_id = &container_ids[2];
    let over_quota_result = runtime
        .allocate_memory(container_c_id, 1024 * 1024 * 10)
        .await; // 10MB > 5MB quota
    assert!(
        over_quota_result.is_err(),
        "Container C should not be able to exceed its 5MB quota"
    );

    // Verify isolation statistics
    for (i, container_id) in container_ids.iter().enumerate() {
        let isolation_stats = runtime
            .get_isolation_stats(container_id)
            .await
            .expect(&format!(
                "Failed to get isolation stats for container {}",
                i
            ));
        assert_eq!(isolation_stats.container_id, *container_id);
        assert!(
            isolation_stats.memory_usage_bytes > 0,
            "Container {} should have allocated memory",
            i
        );
    }

    // Test kernel execution in isolated containers
    for (i, container_id) in container_ids.iter().enumerate() {
        let kernel_sig = KernelSignature {
            prompt_hash: format!("{:08x}{:08x}", i * 16, i * 16),
            ptx_hash: format!("{:08x}{:08x}", i * 32, i * 32),
            agent_id: Some(format!("agent_{}", (b'a' + i as u8) as char)),
            signature: None,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        let result = runtime.launch_kernel(container_id, kernel_sig).await;
        assert!(
            result.is_ok(),
            "Container {} should be able to launch kernels",
            i
        );
    }

    // Verify runtime statistics
    let runtime_stats = runtime
        .get_runtime_stats()
        .await
        .expect("Failed to get runtime stats");
    assert_eq!(runtime_stats.total_containers, 3);
    assert_eq!(runtime_stats.running_containers, 3);
    assert_eq!(runtime_stats.quota_violations, 1); // From the over-quota attempt
    assert_eq!(runtime_stats.security_violations, 0); // No security violations expected

    // Cleanup all containers
    for container_id in container_ids {
        runtime
            .stop_container(&container_id)
            .await
            .expect("Failed to stop container");
        runtime
            .remove_container(&container_id)
            .await
            .expect("Failed to remove container");
    }

    // Verify cleanup
    let final_containers = runtime
        .list_containers()
        .await
        .expect("Failed to list containers");
    assert!(final_containers.is_empty());
}

/// Test zero-copy networking with GPU memory
#[tokio::test]
async fn test_zero_copy_gpu_memory_integration() {
    let memory_allocator = Arc::new(
        GpuMemoryAllocator::new(1024 * 1024 * 100).expect("Failed to create memory allocator"),
    );
    let transport = ZeroCopyTransport::new();

    // Allocate GPU memory buffers
    let buffer1_handle = memory_allocator
        .allocate(1024 * 1024 * 10)
        .await
        .expect("Failed to allocate buffer 1");
    let buffer2_handle = memory_allocator
        .allocate(1024 * 1024 * 10)
        .await
        .expect("Failed to allocate buffer 2");

    // Set up zero-copy transport with shared buffers
    transport
        .allocate_shared_buffer("gpu_buffer_1", buffer1_handle.size)
        .expect("Failed to allocate shared buffer 1");
    transport
        .allocate_shared_buffer("gpu_buffer_2", buffer2_handle.size)
        .expect("Failed to allocate shared buffer 2");

    // Create messages referencing GPU buffers
    let buffer_ref_msg = Message {
        msg_type: MessageType::KnowledgeSync,
        payload: format!("gpu_buffer_1:{buffer1_handle.id}")
            .as_bytes()
            .to_vec(),
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };

    // Send via zero-copy transport
    transport
        .send("gpu_node_1", buffer_ref_msg.clone())
        .await
        .expect("Failed to send buffer reference");

    // Verify message was sent (zero-copy should show in stats)
    let transport_stats = transport
        .stats()
        .await
        .expect("Failed to get transport stats");
    assert_eq!(transport_stats.messages_sent, 1);
    assert_eq!(transport_stats.throughput_mbps, 40000.0); // High throughput for zero-copy

    // Simulate receiving and processing the buffer reference
    let (endpoint, received_msg) = transport
        .receive()
        .await
        .expect("Failed to receive buffer reference");
    assert_eq!(endpoint, "gpu_node_1");

    let buffer_info = String::from_utf8(received_msg.payload).expect("Invalid buffer info");
    assert!(buffer_info.contains("gpu_buffer_1"));
    assert!(buffer_info.contains(&buffer1_handle.id.to_string()));

    // Cleanup
    memory_allocator
        .deallocate(buffer1_handle)
        .await
        .expect("Failed to deallocate buffer 1");
    memory_allocator
        .deallocate(buffer2_handle)
        .await
        .expect("Failed to deallocate buffer 2");
}

/// Test storage persistence with NVMe simulation
#[tokio::test]
async fn test_nvme_storage_persistence() {
    let config = NvmeConfig {
        base_path: PathBuf::from("/tmp/exorust_nvme_test"),
        block_size: 4096,
        cache_size: 1024 * 1024, // 1MB cache
        sync_writes: true,
    };

    let nvme_storage = NvmeStorage::with_config(config)
        .await
        .expect("Failed to create NVMe storage");

    // Test large data persistence
    let large_dataset = (0..1000000).map(|i| (i % 256) as u8).collect::<Vec<u8>>();

    // Store with compression
    nvme_storage
        .store("large_dataset", &large_dataset)
        .await
        .expect("Failed to store large dataset");

    // Verify persistence stats
    let stats = nvme_storage
        .stats()
        .await
        .expect("Failed to get NVMe stats");
    assert_eq!(stats.total_files, 1);
    assert!(stats.used_bytes > 0);

    // Test retrieval
    let retrieved_data = nvme_storage
        .retrieve("large_dataset")
        .await
        .expect("Failed to retrieve large dataset");
    assert_eq!(retrieved_data, large_dataset);

    // Test multiple files for performance
    let mut file_keys = Vec::new();
    for i in 0..10 {
        let data = vec![(i % 256) as u8; 10000]; // 10KB each
        let key = format!("file_{:03}", i);
        nvme_storage
            .store(&key, &data)
            .await
            .expect(&format!("Failed to store file {i}"));
        file_keys.push(key);
    }

    // Verify all files stored
    let final_stats = nvme_storage
        .stats()
        .await
        .expect("Failed to get final NVMe stats");
    assert_eq!(final_stats.total_files, 11); // 1 large + 10 small files

    // Test concurrent access by creating multiple NVMe instances
    let mut handles = Vec::new();
    for key in file_keys.clone() {
        let config_clone = NvmeConfig {
            base_path: PathBuf::from("/tmp/exorust_nvme_test"),
            block_size: 4096,
            cache_size: 1024 * 1024, // 1MB cache
            sync_writes: true,
        };
        let handle = tokio::spawn(async move {
            let storage = NvmeStorage::with_config(config_clone)
                .await
                .expect("Failed to create storage");
            storage.retrieve(&key).await
        });
        handles.push(handle);
    }

    // Wait for all retrievals
    for handle in handles {
        let result = handle.await.expect("Task failed");
        assert!(result.is_ok(), "Concurrent retrieval should succeed");
    }

    // Cleanup
    nvme_storage
        .delete("large_dataset")
        .await
        .expect("Failed to delete large dataset");
    for key in file_keys {
        nvme_storage
            .delete(&key)
            .await
            .expect(&format!("Failed to delete {key}"));
    }

    let cleanup_stats = nvme_storage
        .stats()
        .await
        .expect("Failed to get cleanup stats");
    assert_eq!(cleanup_stats.total_files, 0);
}

/// Test complete agent workflow simulation
#[tokio::test]
async fn test_complete_agent_workflow() {
    // Initialize all systems
    let runtime = SecureContainerRuntime::new();
    let _memory_allocator = Arc::new(
        GpuMemoryAllocator::new(1024 * 1024 * 200).expect("Failed to create memory allocator"),
    );
    let storage = Arc::new(MemoryStorage::new(1024 * 1024 * 100));
    let network = Arc::new(MemoryNetwork::new("agent-node".to_string()));
    let fitness_fn = Arc::new(SimpleFitnessFunction);
    let evolution_engine = GeneticEvolutionEngine::with_defaults(fitness_fn);

    // Step 1: Create secure container for agent
    let agent_config = ContainerConfig {
        memory_limit_bytes: 1024 * 1024 * 50,
        gpu_compute_units: 2,
        timeout_seconds: Some(300),
        agent_type: "self-evolving-agent".to_string(),
        environment: {
            let mut env = std::collections::HashMap::new();
            env.insert("EVOLUTION_MODE".to_string(), "active".to_string());
            env
        },
    };

    let container = runtime
        .create_container(agent_config)
        .await
        .expect("Failed to create agent container");
    let container_id = container.id();
    runtime
        .start_container(container_id)
        .await
        .expect("Failed to start agent container");

    // Step 2: Allocate memory for agent operations
    let agent_memory = runtime
        .allocate_memory(container_id, 1024 * 1024 * 20)
        .await
        .expect("Failed to allocate agent memory");
    assert!(agent_memory > 0);

    // Step 3: Initialize agent population
    let mut population = evolution_engine
        .initialize_population(10)
        .await
        .expect("Failed to initialize agent population");

    // Step 4: Store initial agent state
    let agent_state = serde_json::to_vec(&population).expect("Failed to serialize agent state");
    storage
        .store(&format!("agent_{}_state_0", container_id), &agent_state)
        .await
        .expect("Failed to store initial agent state");

    // Step 5: Agent evolution cycle
    for cycle in 1..=3 {
        // Evolve agent population
        evolution_engine
            .evolve_generation(&mut population)
            .await
            .expect("Failed to evolve agent population");

        // Store evolved state
        let evolved_state =
            serde_json::to_vec(&population).expect("Failed to serialize evolved state");
        storage
            .store(
                &format!("agent_{}_state_{container_id, cycle}"),
                &evolved_state,
            )
            .await
            .expect("Failed to store evolved state");

        // Launch evolved kernels (simulate agent self-modification)
        let kernel_sig = KernelSignature {
            prompt_hash: format!("{:08x}{:08x}", cycle, cycle),
            ptx_hash: format!("{:08x}{:08x}", cycle * 16, cycle * 16),
            agent_id: Some(format!("agent_{container_id}")),
            signature: None,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        runtime
            .launch_kernel(container_id, kernel_sig)
            .await
            .expect("Failed to launch evolved kernel");

        // Send evolution status via network
        let status_msg = Message {
            msg_type: MessageType::KnowledgeSync,
            payload: format!("agent_{}_evolution_cycle_{}_complete", container_id, cycle)
                .as_bytes()
                .to_vec(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        network
            .send("evolution-monitor", status_msg)
            .await
            .expect("Failed to send evolution status");
    }

    // Step 6: Verify agent evolution progress
    let isolation_stats = runtime
        .get_isolation_stats(container_id)
        .await
        .expect("Failed to get final isolation stats");
    assert_eq!(isolation_stats.kernels_executed, 3); // One kernel per evolution cycle
    assert!(isolation_stats.memory_usage_bytes > 0);

    // Step 7: Verify network communication
    let network_stats = network.stats().await.expect("Failed to get network stats");
    assert_eq!(network_stats.messages_sent, 3); // One message per evolution cycle

    // Step 8: Verify storage persistence
    let storage_stats = storage.stats().await.expect("Failed to get storage stats");
    assert_eq!(storage_stats.total_files, 4); // Initial + 3 evolution cycles

    // Step 9: Agent shutdown and cleanup
    runtime
        .stop_container(container_id)
        .await
        .expect("Failed to stop agent container");
    runtime
        .remove_container(container_id)
        .await
        .expect("Failed to remove agent container");

    // Verify final cleanup
    let final_containers = runtime
        .list_containers()
        .await
        .expect("Failed to list containers");
    assert!(final_containers.is_empty());

    let final_runtime_stats = runtime
        .get_runtime_stats()
        .await
        .expect("Failed to get final runtime stats");
    assert_eq!(final_runtime_stats.total_containers, 0);
    assert_eq!(final_runtime_stats.running_containers, 0);
}

/// Test system-wide resource management and limits
#[tokio::test]
async fn test_system_resource_management() {
    // Test resource contention and management across multiple components
    let runtime = SecureContainerRuntime::new();
    let _memory_allocator = Arc::new(
        GpuMemoryAllocator::new(1024 * 1024 * 50)
            .expect("Failed to create limited memory allocator"),
    );
    let storage = Arc::new(MemoryStorage::new(1024 * 1024 * 10)); // Limited storage

    // Create multiple containers competing for resources
    let mut containers = Vec::new();
    for i in 0..3 {
        let config = ContainerConfig {
            memory_limit_bytes: 1024 * 1024 * 20, // Each wants 20MB, but only 50MB total available
            gpu_compute_units: 1,
            timeout_seconds: Some(60),
            agent_type: format!("resource-agent-{i}"),
            environment: std::collections::HashMap::new(),
        };

        let container = runtime
            .create_container(config)
            .await
            .expect(&format!("Failed to create container {i}"));
        runtime
            .start_container(container.id())
            .await
            .expect(&format!("Failed to start container {i}"));
        containers.push(container);
    }

    // Test memory allocation competition
    let mut successful_allocations = 0;
    let mut failed_allocations = 0;

    for (i, container) in containers.iter().enumerate() {
        // Each tries to allocate their full quota
        match runtime
            .allocate_memory(container.id(), 1024 * 1024 * 15)
            .await
        {
            Ok(_) => {
                successful_allocations += 1;
                println!("Container {} successfully allocated 15MB", i);
            }
            Err(_) => {
                failed_allocations += 1;
                println!(
                    "Container {} failed to allocate 15MB (expected due to system limits)",
                    i
                );
            }
        }
    }

    // Some allocations should succeed, some should fail due to system limits
    assert!(
        successful_allocations > 0,
        "At least some allocations should succeed"
    );
    println!(
        "Resource allocation results: {} successful, {} failed",
        successful_allocations, failed_allocations
    );

    // Test storage competition
    let mut storage_success = 0;
    let mut storage_failures = 0;

    for i in 0..5 {
        let large_data = vec![i as u8; 1024 * 1024 * 3]; // 3MB each
        match storage.store(&format!("large_file_{i}"), &large_data).await {
            Ok(_) => storage_success += 1,
            Err(_) => storage_failures += 1,
        }
    }

    // Storage should hit capacity limits
    assert!(
        storage_success > 0,
        "Some storage operations should succeed"
    );
    assert!(
        storage_failures > 0,
        "Some storage operations should fail due to capacity"
    );
    println!(
        "Storage results: {} successful, {} failed",
        storage_success, storage_failures
    );

    // Verify system is still stable
    let runtime_stats = runtime
        .get_runtime_stats()
        .await
        .expect("Failed to get runtime stats");
    assert_eq!(runtime_stats.total_containers, 3);

    // Cleanup
    for container in containers {
        runtime
            .stop_container(container.id())
            .await
            .expect("Failed to stop container");
        runtime
            .remove_container(container.id())
            .await
            .expect("Failed to remove container");
    }
}
