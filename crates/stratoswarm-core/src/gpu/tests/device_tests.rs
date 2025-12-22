//! Tests for GpuDevice trait and MockDevice implementation.

use crate::gpu::{GpuDevice, MockDevice};
use bytes::Bytes;

#[tokio::test]
async fn test_mock_device_creation() {
    let device = MockDevice::new(0, 1024 * 1024 * 1024); // 1GB
    assert_eq!(device.device_id(), 0);
    assert_eq!(device.total_memory(), 1024 * 1024 * 1024);
    assert_eq!(device.used_memory(), 0);
    assert_eq!(device.utilization(), 0.0);
}

#[tokio::test]
async fn test_mock_device_allocate_buffer() {
    let mut device = MockDevice::new(0, 1024 * 1024 * 1024);

    // Allocate a buffer
    let result = device.allocate_buffer("buffer1", 1024).await;
    assert!(result.is_ok());
    assert_eq!(device.used_memory(), 1024);

    // Allocate another buffer
    let result = device.allocate_buffer("buffer2", 2048).await;
    assert!(result.is_ok());
    assert_eq!(device.used_memory(), 1024 + 2048);
}

#[tokio::test]
async fn test_mock_device_allocate_duplicate_buffer_fails() {
    let mut device = MockDevice::new(0, 1024 * 1024);

    device.allocate_buffer("buffer1", 1024).await.unwrap();
    let result = device.allocate_buffer("buffer1", 1024).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("already exists"));
}

#[tokio::test]
async fn test_mock_device_allocate_exceeds_memory() {
    let mut device = MockDevice::new(0, 1024);

    let result = device.allocate_buffer("huge", 2048).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("out of memory"));
}

#[tokio::test]
async fn test_mock_device_deallocate_buffer() {
    let mut device = MockDevice::new(0, 1024 * 1024);

    device.allocate_buffer("buffer1", 1024).await.unwrap();
    device.allocate_buffer("buffer2", 2048).await.unwrap();
    assert_eq!(device.used_memory(), 3072);

    let result = device.deallocate_buffer("buffer1").await;
    assert!(result.is_ok());
    assert_eq!(device.used_memory(), 2048);
}

#[tokio::test]
async fn test_mock_device_deallocate_nonexistent_buffer() {
    let mut device = MockDevice::new(0, 1024 * 1024);

    let result = device.deallocate_buffer("nonexistent").await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not found"));
}

#[tokio::test]
async fn test_mock_device_transfer_to_device() {
    let mut device = MockDevice::new(0, 1024 * 1024);

    // Allocate buffer first
    device.allocate_buffer("buffer1", 1024).await.unwrap();

    // Transfer data
    let data = Bytes::from(vec![1, 2, 3, 4]);
    let result = device.transfer_to_device("buffer1", data.clone(), 0).await;
    assert!(result.is_ok());

    // Verify data was stored
    let stored = device.get_buffer_data("buffer1").unwrap();
    assert_eq!(stored.len(), 1024); // Buffer size
    assert_eq!(&stored[0..4], &[1, 2, 3, 4]);
}

#[tokio::test]
async fn test_mock_device_transfer_to_nonexistent_buffer() {
    let mut device = MockDevice::new(0, 1024 * 1024);

    let data = Bytes::from(vec![1, 2, 3, 4]);
    let result = device.transfer_to_device("nonexistent", data, 0).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not found"));
}

#[tokio::test]
async fn test_mock_device_transfer_exceeds_buffer_size() {
    let mut device = MockDevice::new(0, 1024 * 1024);

    device.allocate_buffer("small", 10).await.unwrap();

    let data = Bytes::from(vec![0; 20]);
    let result = device.transfer_to_device("small", data, 0).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("exceeds buffer size"));
}

#[tokio::test]
async fn test_mock_device_transfer_with_offset() {
    let mut device = MockDevice::new(0, 1024 * 1024);

    device.allocate_buffer("buffer1", 1024).await.unwrap();

    // Transfer data at offset 10
    let data = Bytes::from(vec![1, 2, 3, 4]);
    device.transfer_to_device("buffer1", data.clone(), 10).await.unwrap();

    // Verify data is at correct offset
    let stored = device.get_buffer_data("buffer1").unwrap();
    assert_eq!(&stored[10..14], &[1, 2, 3, 4]);
}

#[tokio::test]
async fn test_mock_device_transfer_from_device() {
    let mut device = MockDevice::new(0, 1024 * 1024);

    device.allocate_buffer("buffer1", 1024).await.unwrap();

    // Put some data in
    let data = Bytes::from(vec![5, 6, 7, 8]);
    device.transfer_to_device("buffer1", data, 0).await.unwrap();

    // Read it back
    let result = device.transfer_from_device("buffer1", 4, 0).await;
    assert!(result.is_ok());
    let read_data = result.unwrap();
    assert_eq!(read_data.as_ref(), &[5, 6, 7, 8]);
}

#[tokio::test]
async fn test_mock_device_transfer_from_device_with_offset() {
    let mut device = MockDevice::new(0, 1024 * 1024);

    device.allocate_buffer("buffer1", 1024).await.unwrap();

    // Put data at offset 5
    let data = Bytes::from(vec![10, 11, 12, 13]);
    device.transfer_to_device("buffer1", data, 5).await.unwrap();

    // Read from offset 5
    let result = device.transfer_from_device("buffer1", 4, 5).await.unwrap();
    assert_eq!(result.as_ref(), &[10, 11, 12, 13]);
}

#[tokio::test]
async fn test_mock_device_launch_kernel() {
    let mut device = MockDevice::new(0, 1024 * 1024);

    let params = Bytes::from(vec![1, 2, 3, 4]);
    let result = device
        .launch_kernel("test_kernel", (1, 1, 1), (32, 1, 1), params)
        .await;

    assert!(result.is_ok());
    let duration = result.unwrap();
    assert!(duration > 0); // Should simulate some execution time
}

#[tokio::test]
async fn test_mock_device_kernel_tracking() {
    let mut device = MockDevice::new(0, 1024 * 1024);

    assert_eq!(device.kernel_count(), 0);

    let params = Bytes::from(vec![]);
    device.launch_kernel("kernel1", (1, 1, 1), (32, 1, 1), params.clone()).await.unwrap();
    assert_eq!(device.kernel_count(), 1);

    device.launch_kernel("kernel2", (2, 1, 1), (64, 1, 1), params).await.unwrap();
    assert_eq!(device.kernel_count(), 2);
}

#[tokio::test]
async fn test_mock_device_synchronize() {
    let mut device = MockDevice::new(0, 1024 * 1024);

    // Launch some kernels
    let params = Bytes::from(vec![]);
    device.launch_kernel("k1", (1, 1, 1), (32, 1, 1), params.clone()).await.unwrap();
    device.launch_kernel("k2", (1, 1, 1), (32, 1, 1), params).await.unwrap();

    // Synchronize should succeed
    let result = device.synchronize(None).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_mock_device_utilization_increases() {
    let mut device = MockDevice::new(0, 1024 * 1024);

    assert_eq!(device.utilization(), 0.0);

    // Launch kernel should increase utilization
    let params = Bytes::from(vec![]);
    device.launch_kernel("kernel", (10, 10, 1), (256, 1, 1), params).await.unwrap();

    let util = device.utilization();
    assert!(util > 0.0 && util <= 100.0);
}

#[tokio::test]
async fn test_mock_device_memory_pressure() {
    let mut device = MockDevice::new(0, 1000);

    // Use 85% of memory
    device.allocate_buffer("buffer", 850).await.unwrap();

    let pressure = device.memory_pressure();
    assert!(pressure > 0.80 && pressure < 0.90);
}

#[tokio::test]
async fn test_mock_device_reset() {
    let mut device = MockDevice::new(0, 1024 * 1024);

    // Add some state
    device.allocate_buffer("buffer1", 1024).await.unwrap();
    let params = Bytes::from(vec![]);
    device.launch_kernel("kernel", (1, 1, 1), (32, 1, 1), params).await.unwrap();

    assert_eq!(device.used_memory(), 1024);
    assert_eq!(device.kernel_count(), 1);

    // Reset
    device.reset().await;

    assert_eq!(device.used_memory(), 0);
    assert_eq!(device.kernel_count(), 0);
    assert_eq!(device.utilization(), 0.0);
}
