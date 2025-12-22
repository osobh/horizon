//! Edge case tests for test-utils crate to enhance coverage to 90%+

#[cfg(test)]
mod edge_case_tests {
    use crate::{catch_panic, test_mutex, with_poisoned_mutex, TestMutex};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    // TestMutex edge cases

    #[test]
    fn test_mutex_with_various_types() {
        // Test with different data types
        let int_mutex = TestMutex::new(42i32);
        let string_mutex = TestMutex::new(String::from("hello"));
        let vec_mutex = TestMutex::new(vec![1, 2, 3, 4, 5]);
        let option_mutex = TestMutex::new(Some("test"));

        // Verify all can be locked and accessed
        assert_eq!(*int_mutex.lock().unwrap(), 42);
        assert_eq!(*string_mutex.lock().unwrap(), "hello");
        assert_eq!(*vec_mutex.lock().unwrap(), vec![1, 2, 3, 4, 5]);
        assert_eq!(*option_mutex.lock().unwrap(), Some("test"));
    }

    #[test]
    fn test_mutex_large_data() {
        // Test with large data structures
        let large_vec = vec![0u8; 1_000_000]; // 1MB
        let mutex = TestMutex::new(large_vec.clone());

        {
            let guard = mutex.lock().unwrap();
            assert_eq!(guard.len(), 1_000_000);
        }

        // Modify the data
        {
            let mut guard = mutex.lock().unwrap();
            guard[0] = 255;
            guard[999_999] = 255;
        }

        // Verify modifications
        let guard = mutex.lock().unwrap();
        assert_eq!(guard[0], 255);
        assert_eq!(guard[999_999], 255);
    }

    #[test]
    fn test_mutex_zero_sized_type() {
        // Test with zero-sized type
        #[derive(Debug, PartialEq)]
        struct ZeroSized;

        let mutex = TestMutex::new(ZeroSized);
        let guard = mutex.lock().unwrap();
        assert_eq!(*guard, ZeroSized);
    }

    #[test]
    fn test_try_lock_edge_cases() {
        let mutex = Arc::new(TestMutex::new(100));
        let mutex_clone = mutex.clone();

        // Hold a lock in another thread
        let handle = thread::spawn(move || {
            let _guard = mutex_clone.lock().unwrap();
            thread::sleep(Duration::from_millis(100));
        });

        // Give the thread time to acquire the lock
        thread::sleep(Duration::from_millis(10));

        // Try lock should fail
        match mutex.try_lock() {
            Ok(_) => panic!("Expected try_lock to fail"),
            Err(e) => {
                // Verify it's WouldBlock error
                assert!(matches!(e, std::sync::TryLockError::WouldBlock));
            }
        }

        handle.join().unwrap();

        // Now try_lock should succeed
        let guard = mutex.try_lock().unwrap();
        assert_eq!(*guard, 100);
    }

    #[test]
    fn test_poison_next_multiple_times() {
        let mutex = TestMutex::new("test");

        // Set poison flag multiple times
        mutex.poison_next();
        mutex.poison_next();
        mutex.poison_next();

        // Only the first drop should poison
        let result = catch_panic(|| {
            let _guard = mutex.lock().unwrap();
            // Drop will trigger panic
        });

        assert!(result.is_err());
        assert!(mutex.is_poisoned());

        // Subsequent locks should fail due to poison
        assert!(mutex.lock().is_err());
    }

    #[test]
    fn test_get_mut_edge_cases() {
        let mut mutex = TestMutex::new(vec![1, 2, 3]);

        // Get mutable reference
        if let Some(data) = mutex.get_mut() {
            data.push(4);
            data.push(5);
        }

        // Verify changes
        let guard = mutex.lock().unwrap();
        assert_eq!(*guard, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_concurrent_poison_and_lock() {
        let mutex = Arc::new(TestMutex::new(0));
        let mut handles = vec![];

        // Spawn multiple threads trying to lock and poison
        for i in 0..5 {
            let mutex_clone = mutex.clone();
            let handle = thread::spawn(move || {
                if i == 2 {
                    // One thread will poison
                    mutex_clone.poison_next();
                    let _ = catch_panic(|| {
                        let _guard = mutex_clone.lock().unwrap();
                    });
                } else {
                    // Others will try to lock
                    thread::sleep(Duration::from_millis(10));
                    let _ = mutex_clone.lock();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Mutex should be poisoned
        assert!(mutex.is_poisoned());
    }

    // catch_panic edge cases

    #[test]
    fn test_catch_panic_various_types() {
        // Panic with string slice
        let result = catch_panic(|| {
            panic!("string slice panic");
        });
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "string slice panic");

        // Panic with String
        let result = catch_panic(|| {
            panic!("{}", String::from("owned string panic"));
        });
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("owned string panic"));

        // Panic with custom type (falls back to "Unknown panic")
        let result = catch_panic(|| {
            panic!(42); // Panic with integer
        });
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Unknown panic");

        // No panic
        let result = catch_panic(|| 42);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_catch_panic_nested() {
        let result = catch_panic(|| {
            let inner_result = catch_panic(|| {
                panic!("inner panic");
            });

            // This should not panic
            assert!(inner_result.is_err());
            "outer success"
        });

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "outer success");
    }

    // with_poisoned_mutex edge cases

    #[test]
    fn test_with_poisoned_mutex_complex_type() {
        #[derive(Debug)]
        struct ComplexType {
            data: Vec<String>,
            count: usize,
        }

        let complex = ComplexType {
            data: vec!["a".to_string(), "b".to_string()],
            count: 2,
        };

        with_poisoned_mutex(complex, |mutex| {
            assert!(mutex.is_poisoned());

            // Try to recover from poison
            match mutex.lock() {
                Ok(_) => panic!("Expected poison error"),
                Err(poison_err) => {
                    // Can still access data through poison error
                    let guard = poison_err.into_inner();
                    assert_eq!(guard.count, 2);
                }
            }
        });
    }

    // test_mutex! macro edge cases

    #[test]
    fn test_macro_usage() {
        let mutex1 = test_mutex!(42);
        let mutex2 = test_mutex!(String::from("macro test"));
        let mutex3 = test_mutex!(vec![1, 2, 3]);

        assert_eq!(*mutex1.lock().unwrap(), 42);
        assert_eq!(*mutex2.lock().unwrap(), "macro test");
        assert_eq!(*mutex3.lock().unwrap(), vec![1, 2, 3]);
    }

    // Thread safety tests

    #[test]
    fn test_send_sync_traits() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        // Verify TestMutex is Send + Sync for Send + Sync types
        assert_send::<TestMutex<i32>>();
        assert_sync::<TestMutex<i32>>();
        assert_send::<TestMutex<String>>();
        assert_sync::<TestMutex<String>>();
    }

    // Edge case: multiple panic_on_lock calls

    #[test]
    fn test_multiple_panic_on_lock() {
        let mutex = TestMutex::new("test");

        // Set panic flag multiple times
        mutex.panic_on_lock();
        mutex.panic_on_lock();
        mutex.panic_on_lock();

        // Should still panic only once
        let result = catch_panic(|| {
            let _guard = mutex.lock();
        });

        assert!(result.is_err());

        // Flag should be cleared after panic
        mutex.panic_on_lock();
        let result2 = catch_panic(|| {
            let _guard = mutex.lock();
        });
        assert!(result2.is_err());
    }

    // Complex concurrent scenario

    #[test]
    fn test_complex_concurrent_scenario() {
        let mutex = Arc::new(TestMutex::new(vec![0; 100]));
        let mut handles = vec![];

        // Multiple readers
        for i in 0..3 {
            let mutex_clone = mutex.clone();
            let handle = thread::spawn(move || {
                for _ in 0..10 {
                    let guard = mutex_clone.lock().unwrap();
                    let sum: i32 = guard.iter().sum();
                    drop(guard);
                    thread::sleep(Duration::from_micros(100));
                }
            });
            handles.push(handle);
        }

        // Multiple writers
        for i in 0..2 {
            let mutex_clone = mutex.clone();
            let handle = thread::spawn(move || {
                for j in 0..5 {
                    let mut guard = mutex_clone.lock().unwrap();
                    guard[i * 10 + j] = 1;
                    drop(guard);
                    thread::sleep(Duration::from_micros(150));
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify some writes occurred
        let guard = mutex.lock().unwrap();
        let sum: i32 = guard.iter().sum();
        assert!(sum > 0);
    }

    // Test panic message preservation

    #[test]
    fn test_panic_message_preservation() {
        let special_messages = vec![
            "Simple message",
            "Message with\nnewlines\nand\ttabs",
            "Unicode: 错误信息 🚨",
            "Very long message: ".to_string() + &"x".repeat(1000),
            "", // Empty message
        ];

        for msg in special_messages {
            let msg_clone = msg.to_string();
            let result = catch_panic(move || {
                panic!("{}", msg_clone);
            });

            assert!(result.is_err());
            let err_msg = result.unwrap_err();

            if msg.is_empty() {
                // Empty panic might have different representation
                assert!(!err_msg.is_empty());
            } else {
                assert!(err_msg.contains(&msg));
            }
        }
    }
}
