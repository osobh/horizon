//! Tests for GPUDirect Storage implementation
//!
//! These tests verify direct GPU-to-storage data transfer capabilities
//! using NVIDIA GPUDirect Storage (GDS) APIs.

use super::*;
use anyhow::Result;
use std::time::Instant;

#[cfg(test)]
mod tests {
    use super::*;

    /// Test GPUDirect Storage configuration
    #[test]
    fn test_gpudirect_config() {
        let config = GpuDirectConfig::default();
        
        assert!(config.enable_gds);
        assert_eq!(config.max_transfer_size, 256 * 1024 * 1024); // 256MB
        assert_eq!(config.io_alignment, 4096); // 4KB alignment
        assert_eq!(config.num_io_queues, 4);
        assert!(config.enable_async_io);
        assert_eq!(config.batch_size, 16);
    }

    /// Test GDS availability check
    #[test]
    fn test_gds_availability() {
        let checker = GdsAvailabilityChecker::new();
        
        // Check if GDS is available on the system
        let available = checker.is_available();
        
        if available {
            println!("GPUDirect Storage is available");
            assert!(checker.check_driver_version().is_ok());
            assert!(checker.check_cuda_version().is_ok());
            assert!(checker.check_filesystem_support().is_ok());
        } else {
            println!("GPUDirect Storage not available - using fallback");
        }
    }

    /// Test GPUDirect Storage manager creation
    #[test]
    fn test_gpudirect_manager_creation() {
        let config = GpuDirectConfig::default();
        
        match GpuDirectManager::new(config) {
            Ok(manager) => {
                assert!(manager.is_initialized());
                assert_eq!(manager.get_num_io_queues(), 4);
            }
            Err(e) => {
                // GDS not available, ensure fallback works
                println!("GDS not available: {}, using fallback", e);
            }
        }
    }

    /// Test direct GPU memory allocation for I/O
    #[test]
    fn test_gpu_io_buffer_allocation() {
        let size = 16 * 1024 * 1024; // 16MB
        
        match GpuIoBuffer::allocate(size) {
            Ok(buffer) => {
                assert_eq!(buffer.size(), size);
                assert!(buffer.is_aligned());
                assert!(buffer.is_pinned());
                assert!(buffer.device_ptr().is_some());
            }
            Err(e) => {
                println!("GPU I/O buffer allocation not supported: {}", e);
            }
        }
    }

    /// Test basic read operation
    #[tokio::test]
    async fn test_gpudirect_read() -> Result<()> {
        let config = GpuDirectConfig::default();
        let manager = match GpuDirectManager::new(config) {
            Ok(m) => m,
            Err(_) => {
                println!("Skipping test - GDS not available");
                return Ok(());
            }
        };

        // Allocate GPU buffer
        let buffer_size = 4 * 1024 * 1024; // 4MB
        let mut gpu_buffer = GpuIoBuffer::allocate(buffer_size)?;

        // Create test file
        let test_path = "/tmp/gpudirect_test_read.bin";
        let test_data = vec![42u8; buffer_size];
        tokio::fs::write(test_path, &test_data).await?;

        // Read directly to GPU
        let bytes_read = manager.read_to_gpu(
            test_path,
            &mut gpu_buffer,
            0, // offset
            buffer_size,
        ).await?;

        assert_eq!(bytes_read, buffer_size);

        // Verify data (copy back to host for verification)
        let host_data = gpu_buffer.to_host_vec()?;
        assert_eq!(host_data.len(), buffer_size);
        assert_eq!(host_data[0], 42);

        // Cleanup
        tokio::fs::remove_file(test_path).await?;

        Ok(())
    }

    /// Test basic write operation
    #[tokio::test]
    async fn test_gpudirect_write() -> Result<()> {
        let config = GpuDirectConfig::default();
        let manager = match GpuDirectManager::new(config) {
            Ok(m) => m,
            Err(_) => {
                println!("Skipping test - GDS not available");
                return Ok(());
            }
        };

        // Allocate and fill GPU buffer
        let buffer_size = 4 * 1024 * 1024; // 4MB
        let gpu_buffer = GpuIoBuffer::allocate_with_data(&vec![99u8; buffer_size])?;

        // Write directly from GPU
        let test_path = "/tmp/gpudirect_test_write.bin";
        let bytes_written = manager.write_from_gpu(
            test_path,
            &gpu_buffer,
            0, // offset
            buffer_size,
        ).await?;

        assert_eq!(bytes_written, buffer_size);

        // Verify file contents
        let file_data = tokio::fs::read(test_path).await?;
        assert_eq!(file_data.len(), buffer_size);
        assert_eq!(file_data[0], 99);

        // Cleanup
        tokio::fs::remove_file(test_path).await?;

        Ok(())
    }

    /// Test batch I/O operations
    #[tokio::test]
    async fn test_batch_io_operations() -> Result<()> {
        let config = GpuDirectConfig {
            batch_size: 4,
            ..Default::default()
        };

        let manager = match GpuDirectManager::new(config) {
            Ok(m) => m,
            Err(_) => {
                println!("Skipping test - GDS not available");
                return Ok(());
            }
        };

        // Prepare batch operations
        let mut batch = GdsBatchOperation::new();
        let buffer_size = 1024 * 1024; // 1MB per operation

        // Add read operations
        for i in 0..4 {
            let path = format!("/tmp/gpudirect_batch_{}.bin", i);
            let data = vec![(i + 1) as u8; buffer_size];
            tokio::fs::write(&path, &data).await?;

            let gpu_buffer = GpuIoBuffer::allocate(buffer_size)?;
            batch.add_read(path, gpu_buffer, 0, buffer_size);
        }

        // Execute batch
        let results = manager.execute_batch(&mut batch).await?;
        
        assert_eq!(results.len(), 4);
        for (i, result) in results.iter().enumerate() {
            assert!(result.is_ok());
            assert_eq!(result.as_ref().unwrap().bytes_transferred, buffer_size);
        }

        // Cleanup
        for i in 0..4 {
            let path = format!("/tmp/gpudirect_batch_{}.bin", i);
            tokio::fs::remove_file(path).await?;
        }

        Ok(())
    }

    /// Test async I/O operations
    #[tokio::test]
    async fn test_async_io() -> Result<()> {
        let config = GpuDirectConfig {
            enable_async_io: true,
            ..Default::default()
        };

        let manager = match GpuDirectManager::new(config) {
            Ok(m) => m,
            Err(_) => {
                println!("Skipping test - GDS not available");
                return Ok(());
            }
        };

        let buffer_size = 8 * 1024 * 1024; // 8MB
        let gpu_buffer = GpuIoBuffer::allocate_with_data(&vec![77u8; buffer_size])?;
        let test_path = "/tmp/gpudirect_async.bin";

        // Start async write
        let handle = manager.write_from_gpu_async(
            test_path,
            &gpu_buffer,
            0,
            buffer_size,
        )?;

        // Do other work while I/O is in progress
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Wait for completion
        let bytes_written = handle.await?;
        assert_eq!(bytes_written, buffer_size);

        // Verify
        let file_data = tokio::fs::read(test_path).await?;
        assert_eq!(file_data[0], 77);

        // Cleanup
        tokio::fs::remove_file(test_path).await?;

        Ok(())
    }

    /// Test performance comparison
    #[tokio::test]
    async fn test_performance_comparison() -> Result<()> {
        let buffer_size = 64 * 1024 * 1024; // 64MB
        let test_path = "/tmp/gpudirect_perf_test.bin";

        // Create test file
        let test_data = vec![123u8; buffer_size];
        tokio::fs::write(test_path, &test_data).await?;

        // Test with GPUDirect
        let gds_manager = match GpuDirectManager::new(GpuDirectConfig::default()) {
            Ok(m) => Some(m),
            Err(_) => None,
        };

        if let Some(manager) = gds_manager {
            let mut gpu_buffer = GpuIoBuffer::allocate(buffer_size)?;
            
            let start = Instant::now();
            manager.read_to_gpu(test_path, &mut gpu_buffer, 0, buffer_size).await?;
            let gds_time = start.elapsed();
            
            println!("GPUDirect read time: {:?}", gds_time);
            println!("GPUDirect throughput: {:.2} GB/s", 
                buffer_size as f64 / gds_time.as_secs_f64() / 1_073_741_824.0);
        }

        // Test with traditional approach (for comparison)
        let start = Instant::now();
        let data = tokio::fs::read(test_path).await?;
        let traditional_time = start.elapsed();
        
        println!("Traditional read time: {:?}", traditional_time);
        println!("Traditional throughput: {:.2} GB/s", 
            buffer_size as f64 / traditional_time.as_secs_f64() / 1_073_741_824.0);

        // Cleanup
        tokio::fs::remove_file(test_path).await?;

        Ok(())
    }

    /// Test error handling
    #[tokio::test]
    async fn test_error_handling() -> Result<()> {
        let manager = match GpuDirectManager::new(GpuDirectConfig::default()) {
            Ok(m) => m,
            Err(_) => {
                println!("Skipping test - GDS not available");
                return Ok(());
            }
        };

        // Test invalid file path
        let gpu_buffer = GpuIoBuffer::allocate(1024)?;
        let result = manager.read_to_gpu(
            "/nonexistent/path/file.bin",
            &gpu_buffer,
            0,
            1024,
        ).await;
        
        assert!(result.is_err());

        Ok(())
    }

    /// Test multi-queue operations
    #[tokio::test]
    async fn test_multi_queue_operations() -> Result<()> {
        let config = GpuDirectConfig {
            num_io_queues: 4,
            ..Default::default()
        };

        let manager = match GpuDirectManager::new(config) {
            Ok(m) => m,
            Err(_) => {
                println!("Skipping test - GDS not available");
                return Ok(());
            }
        };

        // Spawn multiple concurrent I/O operations
        let mut handles = vec![];
        let buffer_size = 16 * 1024 * 1024; // 16MB

        for i in 0..4 {
            let manager = manager.clone();
            let handle = tokio::spawn(async move {
                let path = format!("/tmp/gpudirect_queue_{}.bin", i);
                let data = vec![(i * 10) as u8; buffer_size];
                tokio::fs::write(&path, &data).await?;

                let gpu_buffer = GpuIoBuffer::allocate_with_data(&data)?;
                manager.write_from_gpu(&path, &gpu_buffer, 0, buffer_size).await?;

                tokio::fs::remove_file(&path).await?;
                Ok::<_, anyhow::Error>(())
            });
            handles.push(handle);
        }

        // Wait for all operations
        for handle in handles {
            handle.await??;
        }

        Ok(())
    }
}

/// Integration tests
#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Test integration with existing storage system
    #[tokio::test]
    async fn test_storage_integration() -> Result<()> {
        // Create GPU storage with GDS support
        let storage_config = GpuStorageConfig {
            enable_gpudirect: true,
            ..Default::default()
        };

        let storage = GpuAgentStorage::new(storage_config)?;

        // Test agent data storage with GDS
        let agent_id = "test_agent_gds";
        let agent_data = GpuAgentData {
            state: vec![1.0; 1000],
            memory: vec![2.0; 1000],
            fitness: 0.95,
            generation: 100,
            metadata: HashMap::new(),
        };

        // Store using GDS if available
        storage.store_agent_gds(agent_id, &agent_data).await?;

        // Load using GDS if available
        let loaded = storage.load_agent_gds(agent_id).await?;
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().fitness, 0.95);

        Ok(())
    }

    /// Test fallback behavior
    #[tokio::test]
    async fn test_fallback_behavior() -> Result<()> {
        let config = GpuDirectConfig {
            enable_fallback: true,
            ..Default::default()
        };

        // This should work even without GDS
        let manager = GpuDirectManager::new_with_fallback(config)?;
        
        let buffer_size = 1024 * 1024; // 1MB
        let test_path = "/tmp/gpudirect_fallback.bin";
        let test_data = vec![88u8; buffer_size];
        tokio::fs::write(test_path, &test_data).await?;

        // Should use fallback if GDS not available
        let buffer = GpuIoBuffer::allocate_with_fallback(buffer_size)?;
        let bytes_read = manager.read_to_gpu(test_path, &buffer, 0, buffer_size).await?;
        
        assert_eq!(bytes_read, buffer_size);

        tokio::fs::remove_file(test_path).await?;
        Ok(())
    }
}

/// Benchmark tests
#[cfg(test)]
mod benchmark_tests {
    use super::*;
    use std::time::Duration;

    /// Benchmark different transfer sizes
    #[tokio::test]
    #[ignore] // Run with --ignored flag
    async fn benchmark_transfer_sizes() -> Result<()> {
        let manager = match GpuDirectManager::new(GpuDirectConfig::default()) {
            Ok(m) => m,
            Err(_) => {
                println!("Skipping benchmark - GDS not available");
                return Ok(());
            }
        };

        let sizes = vec![
            1024 * 1024,        // 1MB
            16 * 1024 * 1024,   // 16MB
            64 * 1024 * 1024,   // 64MB
            256 * 1024 * 1024,  // 256MB
        ];

        for size in sizes {
            let test_path = format!("/tmp/gpudirect_bench_{}.bin", size);
            let test_data = vec![0u8; size];
            tokio::fs::write(&test_path, &test_data).await?;

            let gpu_buffer = GpuIoBuffer::allocate(size)?;
            
            // Warm up
            manager.read_to_gpu(&test_path, &gpu_buffer, 0, size).await?;

            // Benchmark
            let iterations = 10;
            let start = Instant::now();
            
            for _ in 0..iterations {
                manager.read_to_gpu(&test_path, &gpu_buffer, 0, size).await?;
            }
            
            let elapsed = start.elapsed();
            let avg_time = elapsed / iterations;
            let throughput = size as f64 / avg_time.as_secs_f64() / 1_073_741_824.0;

            println!("Size: {} MB, Avg time: {:?}, Throughput: {:.2} GB/s",
                size / 1024 / 1024, avg_time, throughput);

            tokio::fs::remove_file(&test_path).await?;
        }

        Ok(())
    }
}