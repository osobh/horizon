//! Tests for unified memory pool.

use crate::gpu::{MemoryStats, UnifiedMemoryPool};
use bytes::Bytes;

#[test]
fn test_memory_pool_creation() {
    let pool = UnifiedMemoryPool::new(1024 * 1024); // 1MB
    assert_eq!(pool.capacity(), 1024 * 1024);
    assert_eq!(pool.used(), 0);
    assert_eq!(pool.available(), 1024 * 1024);
}

#[test]
fn test_memory_pool_allocate() {
    let mut pool = UnifiedMemoryPool::new(1024 * 1024);

    let result = pool.allocate("buffer1", 1024);
    assert!(result.is_ok());
    assert_eq!(pool.used(), 1024);
    assert_eq!(pool.available(), 1024 * 1024 - 1024);
}

#[test]
fn test_memory_pool_allocate_multiple() {
    let mut pool = UnifiedMemoryPool::new(10000);

    pool.allocate("buf1", 1000).unwrap();
    pool.allocate("buf2", 2000).unwrap();
    pool.allocate("buf3", 3000).unwrap();

    assert_eq!(pool.used(), 6000);
    assert_eq!(pool.available(), 4000);
}

#[test]
fn test_memory_pool_allocate_duplicate_fails() {
    let mut pool = UnifiedMemoryPool::new(1024 * 1024);

    pool.allocate("buffer1", 1024).unwrap();
    let result = pool.allocate("buffer1", 1024);

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("already allocated"));
}

#[test]
fn test_memory_pool_allocate_exceeds_capacity() {
    let mut pool = UnifiedMemoryPool::new(1000);

    let result = pool.allocate("huge", 2000);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("insufficient memory"));
}

#[test]
fn test_memory_pool_deallocate() {
    let mut pool = UnifiedMemoryPool::new(10000);

    pool.allocate("buf1", 1000).unwrap();
    pool.allocate("buf2", 2000).unwrap();
    assert_eq!(pool.used(), 3000);

    let result = pool.deallocate("buf1");
    assert!(result.is_ok());
    assert_eq!(pool.used(), 2000);
    assert_eq!(pool.available(), 8000);
}

#[test]
fn test_memory_pool_deallocate_nonexistent() {
    let mut pool = UnifiedMemoryPool::new(1024 * 1024);

    let result = pool.deallocate("nonexistent");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not found"));
}

#[test]
fn test_memory_pool_write_data() {
    let mut pool = UnifiedMemoryPool::new(1024 * 1024);

    pool.allocate("buffer1", 1024).unwrap();

    let data = Bytes::from(vec![1, 2, 3, 4, 5]);
    let result = pool.write("buffer1", data.clone(), 0);
    assert!(result.is_ok());

    // Verify data was written
    let read_data = pool.read("buffer1", 5, 0).unwrap();
    assert_eq!(read_data.as_ref(), &[1, 2, 3, 4, 5]);
}

#[test]
fn test_memory_pool_write_to_nonexistent_buffer() {
    let mut pool = UnifiedMemoryPool::new(1024 * 1024);

    let data = Bytes::from(vec![1, 2, 3]);
    let result = pool.write("nonexistent", data, 0);
    assert!(result.is_err());
}

#[test]
fn test_memory_pool_write_exceeds_buffer_size() {
    let mut pool = UnifiedMemoryPool::new(1024 * 1024);

    pool.allocate("small", 10).unwrap();

    let data = Bytes::from(vec![0; 20]);
    let result = pool.write("small", data, 0);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("exceeds buffer size"));
}

#[test]
fn test_memory_pool_write_with_offset() {
    let mut pool = UnifiedMemoryPool::new(1024 * 1024);

    pool.allocate("buffer1", 1024).unwrap();

    let data = Bytes::from(vec![10, 11, 12]);
    pool.write("buffer1", data.clone(), 5).unwrap();

    let read_data = pool.read("buffer1", 3, 5).unwrap();
    assert_eq!(read_data.as_ref(), &[10, 11, 12]);
}

#[test]
fn test_memory_pool_read_from_nonexistent_buffer() {
    let pool = UnifiedMemoryPool::new(1024 * 1024);

    let result = pool.read("nonexistent", 10, 0);
    assert!(result.is_err());
}

#[test]
fn test_memory_pool_read_exceeds_buffer_size() {
    let mut pool = UnifiedMemoryPool::new(1024 * 1024);

    pool.allocate("buffer1", 10).unwrap();

    let result = pool.read("buffer1", 20, 0);
    assert!(result.is_err());
}

#[test]
fn test_memory_pool_read_with_offset() {
    let mut pool = UnifiedMemoryPool::new(1024 * 1024);

    pool.allocate("buffer1", 100).unwrap();

    let data = Bytes::from(vec![1, 2, 3, 4, 5]);
    pool.write("buffer1", data, 10).unwrap();

    let read_data = pool.read("buffer1", 5, 10).unwrap();
    assert_eq!(read_data.as_ref(), &[1, 2, 3, 4, 5]);
}

#[test]
fn test_memory_pool_pressure() {
    let mut pool = UnifiedMemoryPool::new(1000);

    assert_eq!(pool.pressure(), 0.0);

    pool.allocate("buf1", 500).unwrap();
    assert_eq!(pool.pressure(), 0.5);

    pool.allocate("buf2", 300).unwrap();
    assert_eq!(pool.pressure(), 0.8);

    pool.deallocate("buf1").unwrap();
    assert_eq!(pool.pressure(), 0.3);
}

#[test]
fn test_memory_pool_stats() {
    let mut pool = UnifiedMemoryPool::new(10000);

    pool.allocate("buf1", 1000).unwrap();
    pool.allocate("buf2", 2000).unwrap();

    let stats = pool.stats();
    assert_eq!(stats.capacity, 10000);
    assert_eq!(stats.used, 3000);
    assert_eq!(stats.available, 7000);
    assert_eq!(stats.pressure, 0.3);
    assert_eq!(stats.allocation_count, 2);
}

#[test]
fn test_memory_pool_is_allocated() {
    let mut pool = UnifiedMemoryPool::new(1024 * 1024);

    assert!(!pool.is_allocated("buffer1"));

    pool.allocate("buffer1", 1024).unwrap();
    assert!(pool.is_allocated("buffer1"));

    pool.deallocate("buffer1").unwrap();
    assert!(!pool.is_allocated("buffer1"));
}

#[test]
fn test_memory_pool_get_buffer_size() {
    let mut pool = UnifiedMemoryPool::new(1024 * 1024);

    assert!(pool.get_buffer_size("buffer1").is_none());

    pool.allocate("buffer1", 1024).unwrap();
    assert_eq!(pool.get_buffer_size("buffer1"), Some(1024));
}

#[test]
fn test_memory_pool_clear() {
    let mut pool = UnifiedMemoryPool::new(10000);

    pool.allocate("buf1", 1000).unwrap();
    pool.allocate("buf2", 2000).unwrap();
    assert_eq!(pool.used(), 3000);

    pool.clear();
    assert_eq!(pool.used(), 0);
    assert_eq!(pool.available(), 10000);
    assert!(!pool.is_allocated("buf1"));
    assert!(!pool.is_allocated("buf2"));
}

#[test]
fn test_memory_pool_zero_copy() {
    let mut pool = UnifiedMemoryPool::new(1024 * 1024);
    pool.allocate("buffer1", 1024).unwrap();

    // Create data with known address
    let data = Bytes::from(vec![1, 2, 3, 4, 5]);

    pool.write("buffer1", data.clone(), 0).unwrap();

    // Read back - should be zero-copy with Bytes
    let read_data = pool.read("buffer1", 5, 0).unwrap();

    // Verify data is correct
    assert_eq!(read_data.as_ref(), &[1, 2, 3, 4, 5]);
}

#[test]
fn test_memory_stats_default() {
    let stats = MemoryStats::default();
    assert_eq!(stats.capacity, 0);
    assert_eq!(stats.used, 0);
    assert_eq!(stats.available, 0);
    assert_eq!(stats.pressure, 0.0);
    assert_eq!(stats.allocation_count, 0);
}
