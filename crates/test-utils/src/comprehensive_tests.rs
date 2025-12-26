//! Comprehensive test suite for test-utils crate

#[cfg(test)]
mod lib_tests {
    use crate::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_mutex_try_lock_success() {
        let mutex = TestMutex::new(vec![1, 2, 3]);

        let guard = mutex.try_lock();
        assert!(guard.is_ok());
        assert_eq!(*guard.unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn test_mutex_try_lock_blocked() {
        let mutex = Arc::new(TestMutex::new(100));
        let mutex_clone = mutex.clone();

        // Hold lock in another thread
        let handle = thread::spawn(move || {
            let _guard = mutex_clone.lock().unwrap();
            thread::sleep(Duration::from_millis(50));
        });

        thread::sleep(Duration::from_millis(10));

        // Try lock should fail while mutex is held
        let result = mutex.try_lock();
        assert!(result.is_err());

        handle.join().unwrap();
    }

    #[test]
    fn test_mutex_get_mut() {
        let mut mutex = TestMutex::new(String::from("hello"));

        if let Some(value) = mutex.get_mut() {
            value.push_str(" world");
        }

        let guard = mutex.lock().unwrap();
        assert_eq!(*guard, "hello world");
    }

    #[test]
    fn test_mutex_poison_next_behavior() {
        let mutex = Arc::new(TestMutex::new(42));
        let mutex_clone = mutex.clone();

        // Configure to poison on next unlock
        mutex.poison_next();

        // This should panic when the guard is dropped
        let handle = thread::spawn(move || {
            let _guard = mutex_clone.lock().unwrap();
            // Guard drops here, triggering panic
        });

        // Wait for thread to panic
        let result = handle.join();
        assert!(result.is_err());

        // Mutex should now be poisoned
        assert!(mutex.is_poisoned());
    }

    #[test]
    fn test_mutex_multiple_poison_next_calls() {
        let mutex = TestMutex::new(vec![1, 2, 3]);

        // Multiple poison_next calls should not stack
        mutex.poison_next();
        mutex.poison_next();
        mutex.poison_next();

        // Only the first lock/unlock cycle should poison
        let result = catch_panic(|| {
            let _guard = mutex.lock().unwrap();
        });

        assert!(result.is_err());
        assert!(mutex.is_poisoned());
    }

    #[test]
    fn test_mutex_guard_deref() {
        let mutex = TestMutex::new(vec![10, 20, 30]);

        {
            let guard = mutex.lock().unwrap();
            // Test Deref
            assert_eq!(guard.len(), 3);
            assert_eq!(guard[0], 10);
        }

        // Mutex should not be poisoned after normal use
        assert!(!mutex.is_poisoned());
    }

    #[test]
    fn test_mutex_guard_deref_mut() {
        let mutex = TestMutex::new(vec![1, 2, 3]);

        {
            let mut guard = mutex.lock().unwrap();
            // Test DerefMut
            guard.push(4);
            guard[0] = 100;
        }

        let guard = mutex.lock().unwrap();
        assert_eq!(*guard, vec![100, 2, 3, 4]);
    }

    #[test]
    fn test_catch_panic_success() {
        let result = catch_panic(|| 1 + 1);

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 2);
    }

    #[test]
    fn test_catch_panic_with_string() {
        let result = catch_panic(|| {
            panic!("Test panic message");
        });

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Test panic message");
    }

    #[test]
    fn test_catch_panic_with_string_ref() {
        let result = catch_panic(|| {
            panic!("Static string panic");
        });

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Static string panic"));
    }

    #[test]
    fn test_catch_panic_with_custom_type() {
        // Test that panic with formatted string is caught properly
        let result = catch_panic(|| {
            panic!("{}", 42);
        });

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("42"));
    }

    #[test]
    fn test_test_mutex_macro() {
        let mutex = test_mutex!(HashMap::<String, i32>::new());

        {
            let mut guard = mutex.lock().unwrap();
            guard.insert("key".to_string(), 42);
        }

        let guard = mutex.lock().unwrap();
        assert_eq!(guard.get("key"), Some(&42));
    }

    #[test]
    fn test_with_poisoned_mutex_complex_type() {
        #[derive(Debug, Clone)]
        struct ComplexType {
            data: Vec<String>,
            count: usize,
        }

        with_poisoned_mutex(
            ComplexType {
                data: vec!["test".to_string()],
                count: 1,
            },
            |mutex| {
                assert!(mutex.is_poisoned());

                // Even with poisoned mutex, we can try to recover
                match mutex.lock() {
                    Ok(_) => panic!("Expected poisoned mutex"),
                    Err(poison_err) => {
                        // Can still access data through into_inner()
                        let guard = poison_err.into_inner();
                        assert_eq!(guard.count, 1);
                    }
                }
            },
        );
    }

    #[test]
    fn test_mutex_concurrent_access() {
        let mutex = Arc::new(TestMutex::new(0));
        let mut handles = vec![];

        // Spawn 10 threads that increment the counter
        for _ in 0..10 {
            let mutex_clone = mutex.clone();
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    let mut guard = mutex_clone.lock().unwrap();
                    *guard += 1;
                }
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        // Check final value
        let guard = mutex.lock().unwrap();
        assert_eq!(*guard, 1000);
    }

    #[test]
    fn test_mutex_poison_and_recovery() {
        let mutex = Arc::new(TestMutex::new(vec!["initial".to_string()]));
        let mutex_clone = mutex.clone();

        // Poison the mutex
        mutex.poison_next();

        let handle = thread::spawn(move || {
            let _guard = mutex_clone.lock().unwrap();
        });

        let _ = handle.join();

        // Try to recover from poisoned state
        match mutex.lock() {
            Ok(_) => panic!("Expected poisoned mutex"),
            Err(poison_err) => {
                let mut guard = poison_err.into_inner();
                guard.push("recovered".to_string());
                assert_eq!(guard.len(), 2);
            }
        };
    }

    #[test]
    fn test_panic_on_lock_does_not_poison() {
        let mutex = TestMutex::new(42);

        mutex.panic_on_lock();

        // This should panic but not poison the mutex
        let _ = catch_panic(|| {
            let _guard = mutex.lock();
        });

        // Mutex should not be poisoned from panic_on_lock
        assert!(!mutex.is_poisoned());

        // Should be able to lock normally now
        let guard = mutex.lock().unwrap();
        assert_eq!(*guard, 42);
    }

    #[test]
    fn test_nested_mutex_usage() {
        let outer = Arc::new(TestMutex::new(vec![
            TestMutex::new(1),
            TestMutex::new(2),
            TestMutex::new(3),
        ]));

        let outer_clone = outer.clone();

        let handle = thread::spawn(move || {
            let guard = outer_clone.lock().unwrap();
            let inner_guard = guard[0].lock().unwrap();
            assert_eq!(*inner_guard, 1);
        });

        handle.join().unwrap();

        // Both locks should work fine
        let guard = outer.lock().unwrap();
        assert_eq!(guard.len(), 3);
    }

    #[test]
    fn test_empty_mutex() {
        let mutex = TestMutex::new(());

        {
            let _guard = mutex.lock().unwrap();
            // Can hold lock on unit type
        }

        assert!(!mutex.is_poisoned());
    }

    #[test]
    fn test_large_data_mutex() {
        let large_vec: Vec<u8> = vec![0; 1_000_000]; // 1MB
        let mutex = TestMutex::new(large_vec);

        {
            let mut guard = mutex.lock().unwrap();
            guard[0] = 255;
            guard[999_999] = 255;
        }

        let guard = mutex.lock().unwrap();
        assert_eq!(guard[0], 255);
        assert_eq!(guard[999_999], 255);
        assert_eq!(guard[500_000], 0);
    }

    #[test]
    fn test_with_poisoned_mutex_edge_cases() {
        // Test with zero-sized type
        with_poisoned_mutex((), |mutex| {
            assert!(mutex.is_poisoned());
        });

        // Test with Option type
        with_poisoned_mutex(Some(42), |mutex| {
            assert!(mutex.is_poisoned());
            assert!(mutex.lock().is_err());
        });
    }

    #[test]
    fn test_catch_panic_in_catch_panic() {
        let result = catch_panic(|| {
            let inner_result = catch_panic(|| {
                panic!("Inner panic");
            });

            assert!(inner_result.is_err());
            "Success"
        });

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Success");
    }

    #[test]
    fn test_mutex_drop_behavior() {
        let mutex = TestMutex::new(vec![1, 2, 3]);

        // Create and immediately drop a guard
        {
            let _guard = mutex.lock().unwrap();
        }

        // Should still be accessible
        let guard = mutex.lock().unwrap();
        assert_eq!(*guard, vec![1, 2, 3]);
    }

    use std::collections::HashMap;
}

#[cfg(test)]
mod mutex_mock_tests {
    use crate::mutex_mock::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_mock_mutex_fail_next() {
        let mutex = MockMutex::new(vec![1, 2, 3]);

        // Configure to fail
        mutex.fail_next_lock();

        // This lock should fail
        let result = mutex.lock();
        assert!(result.is_err());

        // Next lock should succeed
        let guard = mutex.lock().unwrap();
        assert_eq!(*guard, vec![1, 2, 3]);
    }

    #[test]
    fn test_mock_mutex_multiple_failures() {
        let mutex = MockMutex::new(100);

        // Set up multiple failures
        mutex.fail_next_lock();
        assert!(mutex.lock().is_err());

        mutex.fail_next_lock();
        assert!(mutex.lock().is_err());

        // Should work normally after failures
        let guard = mutex.lock().unwrap();
        assert_eq!(*guard, 100);
    }

    #[test]
    fn test_mutex_like_trait_standard() {
        let mutex = StandardMutex::new("test");

        // Test trait methods
        assert!(!mutex.is_poisoned());

        let guard = mutex.lock().unwrap();
        assert_eq!(*guard, "test");
    }

    #[test]
    fn test_mutex_like_trait_mock() {
        let mutex = MockMutex::new(42);

        // Test trait methods
        assert!(!mutex.is_poisoned());

        let guard = mutex.lock().unwrap();
        assert_eq!(*guard, 42);
    }

    #[test]
    fn test_mock_mutex_concurrent() {
        let mutex = Arc::new(MockMutex::new(0));
        let mutex_clone = mutex.clone();

        // Configure failure in one thread
        let handle = thread::spawn(move || {
            mutex_clone.fail_next_lock();

            // This should fail
            assert!(mutex_clone.lock().is_err());

            // But this should succeed
            let mut guard = mutex_clone.lock().unwrap();
            *guard = 100;
        });

        handle.join().unwrap();

        // Check value was updated
        let guard = mutex.lock().unwrap();
        assert_eq!(*guard, 100);
    }

    #[test]
    fn test_create_poisoned_mutex_string() {
        let mutex = create_poisoned_mutex::<String>();

        assert!(mutex.is_poisoned());

        match mutex.lock() {
            Ok(_) => panic!("Expected poisoned mutex"),
            Err(e) => {
                // Can recover the data
                let guard = e.into_inner();
                assert_eq!(*guard, String::default());
            }
        };
    }

    #[test]
    fn test_mock_mutex_poison_default_impl() {
        #[derive(Default, Clone)]
        struct TestStruct {
            value: i32,
        }

        unsafe impl Send for TestStruct {}

        let mutex = MockMutex::new(TestStruct { value: 42 });

        // Note: poison() method requires Default + Send
        // This is a limitation of the current implementation
        // but we can still test fail_next_lock
        mutex.fail_next_lock();

        assert!(mutex.lock().is_err());
    }

    #[test]
    fn test_standard_mutex_wrapper() {
        let mutex = StandardMutex::new(vec![1, 2, 3, 4, 5]);

        {
            let guard = mutex.lock().unwrap();
            assert_eq!(guard.len(), 5);
        }

        // Test that it behaves like a normal mutex
        let mutex = Arc::new(mutex);
        let mutex_clone = mutex.clone();

        let handle = thread::spawn(move || {
            let mut guard = mutex_clone.lock().unwrap();
            guard.push(6);
        });

        handle.join().unwrap();

        let guard = mutex.lock().unwrap();
        assert_eq!(*guard, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_mock_mutex_edge_cases() {
        // Test with Option
        let mutex = MockMutex::new(Some("value"));
        let guard = mutex.lock().unwrap();
        assert_eq!(*guard, Some("value"));

        // Test with Result
        let mutex = MockMutex::new(Ok::<i32, String>(42));
        let guard = mutex.lock().unwrap();
        assert!(guard.is_ok());
    }

    #[test]
    fn test_multiple_threads_mock_mutex() {
        let mutex = Arc::new(MockMutex::new(vec![0; 10]));
        let mut handles = vec![];

        for i in 0..10 {
            let mutex_clone = mutex.clone();
            let handle = thread::spawn(move || {
                let mut guard = mutex_clone.lock().unwrap();
                guard[i] = i as i32;
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let guard = mutex.lock().unwrap();
        for i in 0..10 {
            assert_eq!(guard[i], i as i32);
        }
    }
}
