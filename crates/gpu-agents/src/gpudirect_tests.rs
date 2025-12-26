#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpudirect::*;
    use crate::storage::{GpuAgentData, GpuAgentStorage, GpuStorageConfig};
    use anyhow::Result;
    use cudarc::driver::CudaDevice;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tempfile::TempDir;
    use tokio;

    #[test]
    fn test_gpudirect_config() {
        let config = GpuDirectConfig::default();
        assert!(config.enable_gds);
        assert_eq!(config.max_transfer_size, 256 * 1024 * 1024); // 256MB
        assert_eq!(config.io_alignment, 4096); // 4KB alignment
        assert_eq!(config.num_io_queues, 4);
        assert!(config.enable_async_io);
        assert_eq!(config.batch_size, 16);
        assert!(config.enable_fallback);
    }

    #[test]
    fn test_gds_availability_checker() {
        let checker = GdsAvailabilityChecker::new();

        // Should be false by default
        assert!(!checker.is_available());

        // Check individual components
        assert!(checker.check_driver_version().is_ok());
        assert!(checker.check_cuda_version().is_ok());
        assert!(checker.check_filesystem_support().is_ok());
    }

    #[test]
    fn test_gpu_io_buffer_allocation() -> Result<()> {
        let size = 4096;
        let buffer = GpuIoBuffer::allocate(size)?;

        assert_eq!(buffer.size(), size);
        assert!(buffer.is_aligned());
        assert!(buffer.is_pinned());
        assert!(buffer.device_ptr().is_some());

        Ok(())
    }

    #[test]
    fn test_gpu_io_buffer_with_data() -> Result<()> {
        let data = vec![1u8; 1024];
        let buffer = GpuIoBuffer::allocate_with_data(&data)?;

        assert_eq!(buffer.size(), data.len());
        assert!(buffer.is_aligned());
        assert!(buffer.is_pinned());

        // Verify data was copied
        let host_data = buffer.to_host_vec()?;
        assert_eq!(host_data.len(), data.len());
        assert_eq!(host_data[0], 1);

        Ok(())
    }

    #[test]
    fn test_gpu_io_buffer_fallback() -> Result<()> {
        let size = 4096;
        let buffer = GpuIoBuffer::allocate_with_fallback(size)?;

        assert_eq!(buffer.size(), size);
        // Fallback may not be aligned
        assert!(buffer.device_ptr().is_some());

        Ok(())
    }

    #[test]
    fn test_gpu_direct_manager_creation() -> Result<()> {
        let config = GpuDirectConfig::default();
        let manager = GpuDirectManager::new(config)?;

        assert!(manager.is_initialized());
        assert_eq!(manager.get_num_io_queues(), 4);

        Ok(())
    }

    #[test]
    fn test_gpu_direct_manager_with_fallback() -> Result<()> {
        let mut config = GpuDirectConfig::default();
        config.enable_fallback = false;

        // Should fail without fallback when GDS is not available
        let result = GpuDirectManager::new(config.clone());
        assert!(result.is_err());

        // Should succeed with fallback
        let manager = GpuDirectManager::new_with_fallback(config)?;
        assert!(manager.is_initialized());

        Ok(())
    }

    #[tokio::test]
    async fn test_read_write_operations() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let test_file = temp_dir.path().join("test_data.bin");

        // Create test data
        let test_data = vec![42u8; 8192];
        tokio::fs::write(&test_file, &test_data).await?;

        // Create manager and buffer
        let config = GpuDirectConfig::default();
        let manager = GpuDirectManager::new_with_fallback(config)?;
        let mut buffer = GpuIoBuffer::allocate(test_data.len())?;

        // Test read
        let bytes_read = manager
            .read_to_gpu(&test_file, &mut buffer, 0, test_data.len())
            .await?;
        assert_eq!(bytes_read, test_data.len());

        // Verify data
        let read_data = buffer.to_host_vec()?;
        assert_eq!(read_data[0], 42);
        assert_eq!(read_data[100], 42);

        // Test write
        let output_file = temp_dir.path().join("output.bin");
        let bytes_written = manager
            .write_from_gpu(&output_file, &buffer, 0, test_data.len())
            .await?;
        assert_eq!(bytes_written, test_data.len());

        // Verify written data
        let written_data = tokio::fs::read(&output_file).await?;
        assert_eq!(written_data.len(), test_data.len());
        assert_eq!(written_data[0], 42);

        Ok(())
    }

    #[tokio::test]
    async fn test_batch_operations() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = GpuDirectConfig::default();
        let manager = GpuDirectManager::new_with_fallback(config)?;

        // Create test files
        for i in 0..3 {
            let file = temp_dir.path().join(format!("batch_test_{}.bin", i));
            let data = vec![i as u8; 1024];
            tokio::fs::write(&file, &data).await?;
        }

        // Create batch operation
        let mut batch = GdsBatchOperation::new();

        // Add read operations
        for i in 0..3 {
            let file = temp_dir.path().join(format!("batch_test_{}.bin", i));
            let buffer = GpuIoBuffer::allocate(1024)?;
            batch.add_read(file, buffer, 0, 1024);
        }

        // Execute batch
        let results = manager.execute_batch(&mut batch).await?;
        assert_eq!(results.len(), 3);

        for (i, result) in results.iter().enumerate() {
            match result {
                Ok(io_result) => {
                    assert_eq!(io_result.bytes_transferred, 1024);
                    assert!(io_result.duration.as_millis() >= 0);
                }
                Err(e) => panic!("Batch operation {} failed: {}", i, e),
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_async_write_operation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let output_file = temp_dir.path().join("async_output.bin");

        let config = GpuDirectConfig::default();
        let manager = GpuDirectManager::new_with_fallback(config)?;

        // Create buffer with test data
        let test_data = vec![99u8; 4096];
        let buffer = GpuIoBuffer::allocate_with_data(&test_data)?;

        // Start async write
        let handle = manager.write_from_gpu_async(&output_file, &buffer, 0, test_data.len())?;

        // Wait for completion
        let bytes_written = handle.await??;
        assert_eq!(bytes_written, test_data.len());

        // Verify written data
        let written_data = tokio::fs::read(&output_file).await?;
        assert_eq!(written_data.len(), test_data.len());
        assert_eq!(written_data[0], 99);

        Ok(())
    }

    #[tokio::test]
    async fn test_performance_comparison() -> Result<()> {
        use std::time::Instant;

        let temp_dir = TempDir::new()?;
        let test_file = temp_dir.path().join("perf_test.bin");

        // Create large test file (10MB)
        let size = 10 * 1024 * 1024;
        let test_data = vec![0u8; size];
        tokio::fs::write(&test_file, &test_data).await?;

        let config = GpuDirectConfig::default();
        let manager = GpuDirectManager::new_with_fallback(config)?;
        let mut buffer = GpuIoBuffer::allocate(size)?;

        // Measure GDS/fallback read performance
        let start = Instant::now();
        let bytes_read = manager
            .read_to_gpu(&test_file, &mut buffer, 0, size)
            .await?;
        let gds_duration = start.elapsed();

        assert_eq!(bytes_read, size);
        println!("GDS/Fallback read time for 10MB: {:?}", gds_duration);

        // Measure write performance
        let output_file = temp_dir.path().join("perf_output.bin");
        let start = Instant::now();
        let bytes_written = manager
            .write_from_gpu(&output_file, &buffer, 0, size)
            .await?;
        let write_duration = start.elapsed();

        assert_eq!(bytes_written, size);
        println!("GDS/Fallback write time for 10MB: {:?}", write_duration);

        Ok(())
    }

    #[test]
    fn test_error_handling() -> Result<()> {
        // Test invalid buffer size
        let result = GpuIoBuffer::allocate(0);
        assert!(result.is_err() || result.unwrap().size() == 0);

        // Test manager with no fallback and no GDS
        let mut config = GpuDirectConfig::default();
        config.enable_fallback = false;
        let result = GpuDirectManager::new(config);
        assert!(result.is_err());

        Ok(())
    }

    #[tokio::test]
    async fn test_multi_queue_operations() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let mut config = GpuDirectConfig::default();
        config.num_io_queues = 8;

        let manager = GpuDirectManager::new_with_fallback(config)?;
        assert_eq!(manager.get_num_io_queues(), 8);

        // Create multiple files for concurrent operations
        let mut handles = Vec::new();

        for i in 0..8 {
            let file = temp_dir.path().join(format!("queue_test_{}.bin", i));
            let data = vec![i as u8; 1024];
            let buffer = GpuIoBuffer::allocate_with_data(&data)?;

            let handle = manager.write_from_gpu_async(&file, &buffer, 0, data.len())?;
            handles.push(handle);
        }

        // Wait for all operations
        for (i, handle) in handles.into_iter().enumerate() {
            let bytes = handle.await??;
            assert_eq!(bytes, 1024, "Queue {} operation failed", i);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_integration_with_storage() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let mut config = GpuStorageConfig::development();
        config.base_path = temp_dir.path().to_path_buf();
        config.enable_gpudirect = true;

        let storage = GpuAgentStorage::new(config)?;

        // Create test agent
        let agent = GpuAgentData {
            id: "test_agent_gds".to_string(),
            state: vec![1.0; 256],
            memory: vec![2.0; 128],
            generation: 42,
            fitness: 0.95,
            metadata: Default::default(),
        };

        // Store using GDS
        storage.store_agent_gds(&agent.id, &agent).await?;

        // Load using GDS
        let loaded = storage.load_agent_gds(&agent.id).await?;
        assert!(loaded.is_some());

        let loaded_agent = loaded.unwrap();
        assert_eq!(loaded_agent.id, agent.id);
        assert_eq!(loaded_agent.generation, agent.generation);
        assert_eq!(loaded_agent.fitness, agent.fitness);
        assert_eq!(loaded_agent.state.len(), agent.state.len());

        Ok(())
    }

    #[tokio::test]
    async fn test_large_batch_operations() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = GpuDirectConfig {
            batch_size: 32,
            ..Default::default()
        };
        let manager = GpuDirectManager::new_with_fallback(config)?;

        // Create 100 test files
        for i in 0..100 {
            let file = temp_dir.path().join(format!("large_batch_{}.bin", i));
            let data = vec![(i % 256) as u8; 512];
            tokio::fs::write(&file, &data).await?;
        }

        // Create large batch
        let mut batch = GdsBatchOperation::new();

        for i in 0..100 {
            let file = temp_dir.path().join(format!("large_batch_{}.bin", i));
            let buffer = GpuIoBuffer::allocate(512)?;
            batch.add_read(file, buffer, 0, 512);
        }

        // Execute and verify
        let results = manager.execute_batch(&mut batch).await?;
        assert_eq!(results.len(), 100);

        let successful: Vec<_> = results.iter().filter(|r| r.is_ok()).collect();
        assert_eq!(successful.len(), 100, "Some batch operations failed");

        Ok(())
    }

    #[test]
    fn test_benchmarks() -> Result<()> {
        use std::time::Instant;

        // Benchmark buffer allocation
        let sizes = vec![4096, 1024 * 1024, 10 * 1024 * 1024];

        for size in sizes {
            let start = Instant::now();
            let _buffer = GpuIoBuffer::allocate(size)?;
            let duration = start.elapsed();
            println!("Buffer allocation for {} bytes: {:?}", size, duration);
        }

        // Benchmark manager creation
        let start = Instant::now();
        let config = GpuDirectConfig::default();
        let _manager = GpuDirectManager::new_with_fallback(config)?;
        let duration = start.elapsed();
        println!("Manager creation time: {:?}", duration);

        Ok(())
    }
}
