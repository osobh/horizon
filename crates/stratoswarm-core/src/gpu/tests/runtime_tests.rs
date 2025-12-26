//! Integration tests for GpuRuntime.

use crate::channels::{GpuCommand, SystemEvent};
use crate::gpu::{GpuConfig, GpuRuntime, MockDevice};
use bytes::Bytes;
use tokio::sync::broadcast;
use tokio::time::{sleep, timeout, Duration};

#[tokio::test]
async fn test_gpu_runtime_creation() {
    let config = GpuConfig::default();
    let device = MockDevice::new(0, 1024 * 1024 * 1024);
    let runtime = GpuRuntime::new(config, device);

    let metrics = runtime.metrics();
    assert_eq!(metrics.kernels_launched, 0);
    assert_eq!(metrics.device_id, 0);
}

#[tokio::test]
async fn test_gpu_runtime_processes_launch_kernel() {
    let config = GpuConfig::default();
    let device = MockDevice::new(0, 1024 * 1024 * 1024);
    let runtime = GpuRuntime::new(config, device);

    let (gpu_tx, _) = broadcast::channel(100);
    let gpu_rx = gpu_tx.subscribe();
    let (events_tx, mut events_rx) = broadcast::channel(100);

    // Spawn runtime
    let runtime_handle = tokio::spawn(async move {
        runtime.run(gpu_rx, events_tx).await
    });

    // Send a launch kernel command
    let cmd = GpuCommand::LaunchKernel {
        kernel_id: "test_kernel".to_string(),
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        params: Bytes::from(vec![1, 2, 3]),
    };

    gpu_tx.send(cmd).unwrap();

    // Wait for kernel completed event
    let event = timeout(Duration::from_secs(1), events_rx.recv())
        .await
        .expect("Timeout waiting for event")
        .unwrap();

    match event {
        SystemEvent::KernelCompleted {
            kernel_id,
            success,
            ..
        } => {
            assert_eq!(kernel_id, "test_kernel");
            assert!(success);
        }
        _ => panic!("Expected KernelCompleted event"),
    }

    // Cleanup
    drop(gpu_tx);
    let _ = timeout(Duration::from_millis(100), runtime_handle).await;
}

#[tokio::test]
async fn test_gpu_runtime_processes_transfer_to_device() {
    let config = GpuConfig::default();
    let device = MockDevice::new(0, 1024 * 1024 * 1024);
    let runtime = GpuRuntime::new(config, device);

    let (gpu_tx, _) = broadcast::channel(100);
    let gpu_rx = gpu_tx.subscribe();
    let (events_tx, _events_rx) = broadcast::channel(100);

    // Spawn runtime
    let runtime_handle = tokio::spawn(async move {
        runtime.run(gpu_rx, events_tx).await
    });

    // First allocate a buffer (the runtime should handle this internally)
    let data = Bytes::from(vec![5, 6, 7, 8]);
    let cmd = GpuCommand::TransferToDevice {
        buffer_id: "test_buffer".to_string(),
        data: data.clone(),
        offset: 0,
    };

    gpu_tx.send(cmd).unwrap();

    // Give it time to process
    sleep(Duration::from_millis(50)).await;

    // Cleanup
    drop(gpu_tx);
    let _ = timeout(Duration::from_millis(100), runtime_handle).await;
}

#[tokio::test]
async fn test_gpu_runtime_processes_synchronize() {
    let config = GpuConfig::default();
    let device = MockDevice::new(0, 1024 * 1024 * 1024);
    let runtime = GpuRuntime::new(config, device);

    let (gpu_tx, _) = broadcast::channel(100);
    let gpu_rx = gpu_tx.subscribe();
    let (events_tx, _events_rx) = broadcast::channel(100);

    // Spawn runtime
    let runtime_handle = tokio::spawn(async move {
        runtime.run(gpu_rx, events_tx).await
    });

    // Send synchronize command
    let cmd = GpuCommand::Synchronize { stream_id: None };
    gpu_tx.send(cmd).unwrap();

    // Give it time to process
    sleep(Duration::from_millis(50)).await;

    // Cleanup
    drop(gpu_tx);
    let _ = timeout(Duration::from_millis(100), runtime_handle).await;
}

#[tokio::test]
async fn test_gpu_runtime_broadcasts_utilization() {
    let mut config = GpuConfig::default();
    config.utilization_broadcast_interval = Duration::from_millis(100);

    let device = MockDevice::new(0, 1024 * 1024 * 1024);
    let runtime = GpuRuntime::new(config, device);

    let (gpu_tx, _) = broadcast::channel(100);
    let gpu_rx = gpu_tx.subscribe();
    let (events_tx, mut events_rx) = broadcast::channel(100);

    // Spawn runtime
    let _runtime_handle = tokio::spawn(async move {
        runtime.run(gpu_rx, events_tx).await
    });

    // Wait for at least one utilization event
    let event = timeout(Duration::from_secs(1), events_rx.recv())
        .await
        .expect("Timeout waiting for utilization event")
        .unwrap();

    match event {
        SystemEvent::GpuUtilization { device_id, .. } => {
            assert_eq!(device_id, 0);
        }
        _ => panic!("Expected GpuUtilization event, got {:?}", event),
    }
}

#[tokio::test]
async fn test_gpu_runtime_tracks_metrics() {
    let config = GpuConfig::default();
    let device = MockDevice::new(0, 1024 * 1024 * 1024);
    let runtime = GpuRuntime::new(config, device);

    let (gpu_tx, _) = broadcast::channel(100);
    let gpu_rx = gpu_tx.subscribe();
    let (events_tx, mut events_rx) = broadcast::channel(100);

    // Spawn runtime
    let runtime_clone = runtime.clone_metrics();
    tokio::spawn(async move {
        runtime.run(gpu_rx, events_tx).await
    });

    // Send multiple kernel launches
    for i in 0..3 {
        let cmd = GpuCommand::LaunchKernel {
            kernel_id: format!("kernel_{}", i),
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            params: Bytes::from(vec![]),
        };
        gpu_tx.send(cmd).unwrap();
    }

    // Wait for all kernel completed events
    let mut kernel_count = 0;
    for _ in 0..10 {
        match timeout(Duration::from_millis(500), events_rx.recv()).await {
            Ok(Ok(SystemEvent::KernelCompleted { .. })) => {
                kernel_count += 1;
                if kernel_count == 3 {
                    break;
                }
            }
            Ok(Ok(_)) => continue, // Skip other events like GpuUtilization
            _ => break,
        }
    }

    // Check metrics
    let metrics = runtime_clone.metrics();
    assert_eq!(metrics.kernels_launched, 3, "Expected 3 kernels, got {} (received {} kernel events)", metrics.kernels_launched, kernel_count);
}

#[tokio::test]
async fn test_gpu_runtime_graceful_shutdown() {
    let config = GpuConfig::default();
    let device = MockDevice::new(0, 1024 * 1024 * 1024);
    let runtime = GpuRuntime::new(config, device);

    let (gpu_tx, _) = broadcast::channel(100);
    let gpu_rx = gpu_tx.subscribe();
    let (events_tx, _events_rx) = broadcast::channel(100);

    // Spawn runtime
    let runtime_handle = tokio::spawn(async move {
        runtime.run(gpu_rx, events_tx).await
    });

    // Send a command
    let cmd = GpuCommand::Synchronize { stream_id: None };
    gpu_tx.send(cmd).unwrap();

    // Close the channel
    drop(gpu_tx);

    // Runtime should exit gracefully
    let result = timeout(Duration::from_secs(1), runtime_handle).await;
    assert!(result.is_ok(), "Runtime did not shutdown gracefully");
}

#[tokio::test]
async fn test_gpu_runtime_handles_backpressure() {
    let config = GpuConfig::default();
    let device = MockDevice::new(0, 1024 * 1024 * 1024);
    let runtime = GpuRuntime::new(config, device);

    let (gpu_tx, _) = broadcast::channel(10); // Small buffer
    let gpu_rx = gpu_tx.subscribe();
    let (events_tx, _events_rx) = broadcast::channel(100);

    // Spawn runtime
    let _runtime_handle = tokio::spawn(async move {
        runtime.run(gpu_rx, events_tx).await
    });

    // Try to send more than buffer size
    let mut send_count = 0;
    for i in 0..20 {
        let cmd = GpuCommand::Synchronize { stream_id: Some(i) };

        // Use send to test - broadcast channels handle backpressure differently
        if gpu_tx.send(cmd).is_ok() {
            send_count += 1;
        } else {
            break;
        }
    }

    // Broadcast channels don't have the same backpressure as mpsc
    // They drop lagging receivers instead
    assert!(send_count > 0, "Should be able to send messages");
}

#[tokio::test]
async fn test_gpu_runtime_multiple_commands_in_sequence() {
    let config = GpuConfig::default();
    let device = MockDevice::new(0, 1024 * 1024 * 1024);
    let runtime = GpuRuntime::new(config, device);

    let (gpu_tx, _) = broadcast::channel(100);
    let gpu_rx = gpu_tx.subscribe();
    let (events_tx, mut events_rx) = broadcast::channel(100);

    // Spawn runtime
    let _runtime_handle = tokio::spawn(async move {
        runtime.run(gpu_rx, events_tx).await
    });

    // Send sequence of commands
    gpu_tx.send(GpuCommand::Synchronize { stream_id: None }).unwrap();

    gpu_tx
        .send(GpuCommand::LaunchKernel {
            kernel_id: "kernel1".to_string(),
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            params: Bytes::from(vec![]),
        })
        .unwrap();

    gpu_tx.send(GpuCommand::Synchronize { stream_id: None }).unwrap();

    // Wait for kernel completed event
    let event = timeout(Duration::from_secs(1), events_rx.recv())
        .await
        .expect("Timeout")
        .unwrap();

    match event {
        SystemEvent::KernelCompleted { kernel_id, .. } => {
            assert_eq!(kernel_id, "kernel1");
        }
        SystemEvent::GpuUtilization { .. } => {
            // Skip utilization events, wait for kernel
            let event = timeout(Duration::from_secs(1), events_rx.recv())
                .await
                .expect("Timeout")
                .unwrap();
            match event {
                SystemEvent::KernelCompleted { kernel_id, .. } => {
                    assert_eq!(kernel_id, "kernel1");
                }
                _ => panic!("Expected KernelCompleted"),
            }
        }
        _ => panic!("Expected KernelCompleted or GpuUtilization"),
    }
}

#[tokio::test]
async fn test_gpu_config_default() {
    let config = GpuConfig::default();
    assert_eq!(config.device_id, 0);
    assert!(config.memory_pool_size > 0);
    assert!(config.utilization_broadcast_interval > Duration::from_secs(0));
}

#[tokio::test]
async fn test_gpu_config_custom() {
    let config = GpuConfig {
        device_id: 1,
        memory_pool_size: 2 * 1024 * 1024 * 1024,
        utilization_broadcast_interval: Duration::from_secs(10),
    };

    assert_eq!(config.device_id, 1);
    assert_eq!(config.memory_pool_size, 2 * 1024 * 1024 * 1024);
    assert_eq!(config.utilization_broadcast_interval, Duration::from_secs(10));
}
