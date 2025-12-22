//! Tests for CUDA stream management

use crate::error::CudaResult;
use crate::stream::*;

#[test]
fn test_stream_flags() {
    let default_flags = StreamFlags::default();
    assert_eq!(default_flags.bits(), 0);

    let non_blocking = StreamFlags::NON_BLOCKING;
    assert!(non_blocking.contains(StreamFlags::NON_BLOCKING));

    let combined = StreamFlags::NON_BLOCKING | StreamFlags::PRIORITY_HIGH;
    assert!(combined.contains(StreamFlags::NON_BLOCKING));
    assert!(combined.contains(StreamFlags::PRIORITY_HIGH));
}

#[cfg(cuda_mock)]
#[test]
fn test_stream_creation() {
    let stream = Stream::new(StreamFlags::default()).unwrap();
    assert!(!stream.is_null());
    assert_eq!(stream.get_flags(), StreamFlags::default());
}

#[cfg(cuda_mock)]
#[test]
fn test_stream_with_priority() {
    let high_priority_stream = Stream::with_priority(1).unwrap();
    assert_eq!(high_priority_stream.get_priority(), 1);

    let low_priority_stream = Stream::with_priority(-1).unwrap();
    assert_eq!(low_priority_stream.get_priority(), -1);
}

#[cfg(cuda_mock)]
#[test]
fn test_stream_synchronization() {
    let stream = Stream::new(StreamFlags::default()).unwrap();

    // Should be able to synchronize empty stream
    assert!(stream.synchronize().is_ok());

    // Query should return true for empty stream
    assert!(stream.query()?);
}

#[cfg(cuda_mock)]
#[test]
fn test_stream_callbacks() {
    let stream = Stream::new(StreamFlags::default()).unwrap();

    let callback_called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let callback_called_clone = callback_called.clone();

    let callback = Box::new(move |status: CudaResult<()>| {
        assert!(status.is_ok());
        callback_called_clone.store(true, std::sync::atomic::Ordering::Relaxed);
    });

    stream.add_callback(callback).unwrap();
    stream.synchronize().unwrap();

    // Callback should have been called
    assert!(callback_called.load(std::sync::atomic::Ordering::Relaxed));
}

#[cfg(cuda_mock)]
#[test]
fn test_stream_events() {
    let stream = Stream::new(StreamFlags::default()).unwrap();
    let event = Event::new().unwrap();

    // Record event in stream
    event.record(&stream)?;

    // Wait for event
    stream.wait_event(&event)?;

    // Event should be completed
    assert!(event.query().unwrap());
}

#[cfg(cuda_mock)]
#[test]
fn test_multiple_streams() {
    let stream1 = Stream::new(StreamFlags::default()).unwrap();
    let stream2 = Stream::new(StreamFlags::NON_BLOCKING).unwrap();
    let stream3 = Stream::with_priority(1).unwrap();

    // All streams should be different
    assert_ne!(stream1.as_ptr(), stream2.as_ptr());
    assert_ne!(stream2.as_ptr(), stream3.as_ptr());
    assert_ne!(stream1.as_ptr(), stream3.as_ptr());
}

#[cfg(cuda_mock)]
#[test]
fn test_stream_memory_operations() {
    let stream = Stream::new(StreamFlags::default()).unwrap();

    // Test memory operations in stream
    let size = 1024;
    let pattern = 0x42u8;

    let dst_ptr = 0x1000 as *mut u8;
    stream.memset_async(dst_ptr, pattern, size)?;

    let src_ptr = 0x2000 as *const u8;
    stream.memcpy_async(dst_ptr, src_ptr, size).unwrap();

    stream.synchronize().unwrap();
}

#[test]
fn test_event_creation() {
    let event = Event::new().unwrap();
    assert!(!event.is_null());

    let event_with_flags = Event::with_flags(EventFlags::DISABLE_TIMING).unwrap();
    assert_eq!(event_with_flags.get_flags(), EventFlags::DISABLE_TIMING);
}

#[cfg(cuda_mock)]
#[test]
fn test_event_timing() {
    let start_event = Event::new().unwrap();
    let end_event = Event::new().unwrap();
    let stream = Stream::new(StreamFlags::default()).unwrap();

    start_event.record(&stream)?;

    // Simulate some work
    std::thread::sleep(std::time::Duration::from_millis(10));

    end_event.record(&stream)?;
    stream.synchronize().unwrap();

    let elapsed_ms = start_event.elapsed_time(&end_event).unwrap();
    assert!(elapsed_ms >= 0.0);
}

#[cfg(cuda_mock)]
#[test]
fn test_event_synchronization() {
    let event = Event::new().unwrap();
    let stream = Stream::new(StreamFlags::default()).unwrap();

    event.record(&stream).unwrap();
    event.synchronize()?;

    assert!(event.query()?);
}

#[test]
fn test_event_flags() {
    let flags = EventFlags::default();
    assert_eq!(flags.bits(), 0);

    let timing_disabled = EventFlags::DISABLE_TIMING;
    assert!(timing_disabled.contains(EventFlags::DISABLE_TIMING));

    let blocking_sync = EventFlags::BLOCKING_SYNC;
    assert!(blocking_sync.contains(EventFlags::BLOCKING_SYNC));

    let combined = EventFlags::DISABLE_TIMING | EventFlags::BLOCKING_SYNC;
    assert!(combined.contains(EventFlags::DISABLE_TIMING));
    assert!(combined.contains(EventFlags::BLOCKING_SYNC));
}

#[cfg(cuda_mock)]
#[test]
fn test_stream_priority_range() {
    let (least, greatest) = Stream::get_priority_range().unwrap();

    // In CUDA, lower numbers = higher priority
    assert!(least >= greatest);

    // Should be able to create streams with priorities in this range
    let high_priority = Stream::with_priority(greatest)?;
    let low_priority = Stream::with_priority(least)?;

    assert_eq!(high_priority.get_priority(), greatest);
    assert_eq!(low_priority.get_priority(), least);
}

#[cfg(cuda_mock)]
#[test]
fn test_stream_attributes() {
    let stream = Stream::new(StreamFlags::default()).unwrap();

    let attrs = stream.get_attributes().unwrap();
    assert!(attrs.synchronization_policy.is_some());
}

#[cfg(cuda_mock)]
#[test]
fn test_stream_begin_capture() {
    let stream = Stream::new(StreamFlags::default()).unwrap();

    // Begin graph capture
    stream.begin_capture(CaptureMode::Global).unwrap();

    // Add some operations (would be actual CUDA operations)
    let dst_ptr = 0x1000 as *mut u8;
    let pattern = 0x00u8;
    stream.memset_async(dst_ptr, pattern, 1024)?;

    // End capture and get graph
    let graph = stream.end_capture().unwrap();
    assert!(!graph.is_null());
}

#[cfg(cuda_mock)]
#[test]
fn test_stream_dependency() {
    let stream1 = Stream::new(StreamFlags::default()).unwrap();
    let stream2 = Stream::new(StreamFlags::default()).unwrap();
    let event = Event::new().unwrap();

    // Record event in stream1
    event.record(&stream1)?;

    // Make stream2 wait for the event
    stream2.wait_event(&event)?;

    // Both streams should synchronize correctly
    stream1.synchronize().unwrap();
    stream2.synchronize().unwrap();
}

#[cfg(cuda_mock)]
#[test]
fn test_stream_memory_pool() {
    let stream = Stream::new(StreamFlags::default()).unwrap();
    let mem_pool = crate::memory::MemoryPool::new();

    // Associate stream with memory pool
    stream.set_memory_pool(&mem_pool)?;

    let retrieved_pool = stream.get_memory_pool()?;
    assert_eq!(retrieved_pool.id(), mem_pool.id());
}

#[cfg(cuda_mock)]
#[test]
fn test_stream_batch_operations() {
    let stream = Stream::new(StreamFlags::default()).unwrap();

    // Batch multiple operations
    let operations = vec![
        StreamOperation::Memset {
            dst: 0x1000 as *mut u8,
            value: 0x00,
            size: 1024,
        },
        StreamOperation::Memcpy {
            dst: 0x2000 as *mut u8,
            src: 0x1000 as *const u8,
            size: 1024,
        },
        StreamOperation::KernelLaunch {
            kernel: "test_kernel".to_string(),
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
        },
    ];

    stream.batch_submit(operations).unwrap();
    stream.synchronize().unwrap();
}

#[cfg(cuda_mock)]
#[test]
fn test_stream_error_handling() {
    let stream = Stream::new(StreamFlags::default()).unwrap();

    // Test invalid operations
    let null_ptr = std::ptr::null_mut();
    let result = stream.memset_async(null_ptr, 0, 1024);
    assert!(result.is_err());

    // Stream should still be usable after error
    assert!(stream.query()?);
}

#[cfg(cuda_mock)]
#[test]
fn test_concurrent_streams() {
    use std::sync::Arc;
    use std::thread;

    let mut handles = Vec::new();

    // Create multiple streams concurrently
    for i in 0..10 {
        let handle = thread::spawn(move || {
            let stream = Stream::new(StreamFlags::default())?;

            // Perform some operations
            let dst_ptr = (0x1000 + i * 1024) as *mut u8;
            stream.memset_async(dst_ptr, i as u8, 1024).unwrap();
            stream.synchronize().unwrap();

            stream
        });
        handles.push(handle);
    }

    let streams: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(streams.len(), 10);
}

#[cfg(cuda_mock)]
#[test]
fn test_stream_lifecycle() {
    // Test stream creation and destruction
    {
        let _stream = Stream::new(StreamFlags::default()).unwrap();
        // Stream dropped here
    }

    // Should be able to create new stream
    let _stream2 = Stream::new(StreamFlags::default())?;
}

#[cfg(cuda_mock)]
#[test]
fn test_event_lifecycle() {
    // Test event creation and destruction
    {
        let _event = Event::new().unwrap();
        // Event dropped here
    }

    // Should be able to create new event
    let _event2 = Event::new()?;
}

#[test]
fn test_stream_flags_combinations() {
    let flag_combinations = vec![
        StreamFlags::default(),
        StreamFlags::NON_BLOCKING,
        StreamFlags::PRIORITY_HIGH,
        StreamFlags::PRIORITY_LOW,
        StreamFlags::NON_BLOCKING | StreamFlags::PRIORITY_HIGH,
        StreamFlags::NON_BLOCKING | StreamFlags::PRIORITY_LOW,
    ];

    for flags in flag_combinations {
        // All combinations should be valid
        assert!(flags.bits() <= (StreamFlags::all().bits()));
    }
}

#[test]
fn test_event_flags_combinations() {
    let flag_combinations = vec![
        EventFlags::default(),
        EventFlags::DISABLE_TIMING,
        EventFlags::BLOCKING_SYNC,
        EventFlags::INTERPROCESS,
        EventFlags::DISABLE_TIMING | EventFlags::BLOCKING_SYNC,
    ];

    for flags in flag_combinations {
        // All combinations should be valid
        assert!(flags.bits() <= EventFlags::all().bits());
    }
}

#[cfg(cuda_mock)]
#[test]
fn test_stream_performance_monitoring() {
    let stream = Stream::new(StreamFlags::default()).unwrap();

    // Enable performance monitoring
    stream.enable_profiling().unwrap();

    // Perform some operations
    let dst_ptr = 0x1000 as *mut u8;
    stream.memset_async(dst_ptr, 0xFF, 2048)?;
    stream.synchronize()?;

    // Get performance metrics
    let metrics = stream.get_performance_metrics().unwrap();
    assert!(metrics.operations_count > 0);
    assert!(metrics.total_time_ns > 0);
}

#[cfg(cuda_mock)]
#[test]
fn test_stream_resource_limits() {
    let stream = Stream::new(StreamFlags::default()).unwrap();

    // Test setting resource limits
    stream.set_max_pending_operations(1000).unwrap();
    stream.set_memory_limit(1024 * 1024 * 1024)?; // 1GB

    let limits = stream.get_resource_limits()?;
    assert_eq!(limits.max_pending_operations, 1000);
    assert_eq!(limits.memory_limit, 1024 * 1024 * 1024);
}
