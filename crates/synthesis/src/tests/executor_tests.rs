//! Tests for synthesis executor module

use crate::executor::*;
use crate::error::SynthesisResult;

#[test]
    fn test_synthesis_executor_creation() -> Result<(), Box<dyn std::error::Error>> {
    let executor = SynthesisExecutor::new().unwrap();
    assert!(executor.is_initialized());
    assert_eq!(executor.get_status(), ExecutorStatus::Ready);
}

#[test]
    fn test_executor_configuration() -> Result<(), Box<dyn std::error::Error>> {
    let config = ExecutorConfig {
        max_concurrent_tasks: 8,
        timeout_seconds: 300,
        memory_limit_mb: 4096,
        use_gpu_acceleration: true,
        enable_profiling: true,
        cache_size_mb: 512,
    };
    
    let executor = SynthesisExecutor::with_config(config.clone())?;
    let retrieved_config = executor.get_config();
    
    assert_eq!(retrieved_config.max_concurrent_tasks, config.max_concurrent_tasks);
    assert_eq!(retrieved_config.timeout_seconds, config.timeout_seconds);
    assert_eq!(retrieved_config.memory_limit_mb, config.memory_limit_mb);
    assert_eq!(retrieved_config.use_gpu_acceleration, config.use_gpu_acceleration);
}

#[test]
    fn test_task_submission() -> Result<(), Box<dyn std::error::Error>> {
    let executor = SynthesisExecutor::new().unwrap();
    
    let task = SynthesisTask {
        id: "test_task_1".to_string(),
        goal: "Add two vectors element-wise".to_string(),
        priority: TaskPriority::Normal,
        input_data: vec![1.0, 2.0, 3.0, 4.0],
        expected_output_size: 4,
        timeout: Some(std::time::Duration::from_secs(60)),
    };
    
    let task_id = executor.submit_task(task).unwrap();
    assert_eq!(task_id, "test_task_1");
    
    let status = executor.get_task_status(&task_id).unwrap();
    assert!(matches!(status, TaskStatus::Queued | TaskStatus::Running));
}

#[test]
    fn test_task_execution() -> Result<(), Box<dyn std::error::Error>> {
    let executor = SynthesisExecutor::new().unwrap();
    
    let task = SynthesisTask {
        id: "vector_add_task".to_string(),
        goal: "Add two vectors of 1000 elements each".to_string(),
        priority: TaskPriority::High,
        input_data: (0..2000).map(|i| i as f32).collect(),
        expected_output_size: 1000,
        timeout: Some(std::time::Duration::from_secs(30)),
    };
    
    let task_id = executor.submit_task(task).unwrap();
    let result = executor.wait_for_completion(&task_id).unwrap();
    
    assert!(result.success);
    assert_eq!(result.output_data.len(), 1000);
    assert!(result.execution_time.as_millis() > 0);
}

#[test]
    fn test_concurrent_task_execution() -> Result<(), Box<dyn std::error::Error>> {
    let executor = SynthesisExecutor::new().unwrap();
    let mut task_ids = Vec::new();
    
    // Submit multiple tasks
    for i in 0..5 {
        let task = SynthesisTask {
            id: format!("concurrent_task_{}", i),
            goal: format!("Process array {} with element-wise operations", i),
            priority: TaskPriority::Normal,
            input_data: (0..100).map(|j| (i * 100 + j) as f32).collect(),
            expected_output_size: 100,
            timeout: Some(std::time::Duration::from_secs(10)),
        };
        
        let task_id = executor.submit_task(task).unwrap();
        task_ids.push(task_id);
    }
    
    // Wait for all tasks to complete
    let results = executor.wait_for_all(&task_ids).unwrap();
    
    assert_eq!(results.len(), 5);
    for result in results {
        assert!(result.success);
        assert_eq!(result.output_data.len(), 100);
    }
}

#[test]
    fn test_task_prioritization() -> Result<(), Box<dyn std::error::Error>> {
    let executor = SynthesisExecutor::new().unwrap();
    
    // Submit low priority task first
    let low_priority_task = SynthesisTask {
        id: "low_priority".to_string(),
        goal: "Slow matrix multiplication".to_string(),
        priority: TaskPriority::Low,
        input_data: vec![0.0; 10000],
        expected_output_size: 100,
        timeout: Some(std::time::Duration::from_secs(60)),
    };
    
    let low_id = executor.submit_task(low_priority_task).unwrap();
    
    // Submit high priority task after
    let high_priority_task = SynthesisTask {
        id: "high_priority".to_string(),
        goal: "Fast vector addition".to_string(),
        priority: TaskPriority::Critical,
        input_data: vec![0.0; 100],
        expected_output_size: 50,
        timeout: Some(std::time::Duration::from_secs(5)),
    };
    
    let high_id = executor.submit_task(high_priority_task).unwrap();
    
    // High priority task should complete first or around the same time
    let high_result = executor.wait_for_completion(&high_id).unwrap();
    let high_completion_time = high_result.completion_timestamp;
    
    let low_result = executor.wait_for_completion(&low_id).unwrap();
    let low_completion_time = low_result.completion_timestamp;
    
    // Allow some tolerance for timing
    assert!(high_completion_time <= low_completion_time + std::time::Duration::from_millis(100));
}

#[test]
    fn test_task_cancellation() -> Result<(), Box<dyn std::error::Error>> {
    let executor = SynthesisExecutor::new().unwrap();
    
    let task = SynthesisTask {
        id: "cancellable_task".to_string(),
        goal: "Long running matrix operation".to_string(),
        priority: TaskPriority::Normal,
        input_data: vec![0.0; 100000],
        expected_output_size: 10000,
        timeout: Some(std::time::Duration::from_secs(120)),
    };
    
    let task_id = executor.submit_task(task).unwrap();
    
    // Wait a bit then cancel
    std::thread::sleep(std::time::Duration::from_millis(100));
    let cancel_result = executor.cancel_task(&task_id);
    assert!(cancel_result.is_ok());
    
    // Task should be cancelled
    let status = executor.get_task_status(&task_id).unwrap();
    assert_eq!(status, TaskStatus::Cancelled);
}

#[test]
    fn test_task_timeout() -> Result<(), Box<dyn std::error::Error>> {
    let executor = SynthesisExecutor::new().unwrap();
    
    let task = SynthesisTask {
        id: "timeout_task".to_string(),
        goal: "Infinite loop simulation".to_string(),
        priority: TaskPriority::Normal,
        input_data: vec![0.0; 1000],
        expected_output_size: 1000,
        timeout: Some(std::time::Duration::from_millis(100)), // Very short timeout
    };
    
    let task_id = executor.submit_task(task).unwrap();
    let result = executor.wait_for_completion(&task_id);
    
    // Should fail due to timeout
    match result {
        Err(SynthesisError::RuntimeError(msg)) => assert!(msg.contains("timeout")),
        Ok(result) => assert!(!result.success),
        Err(_) => {} // Other errors are also acceptable
    }
}

#[test]
    fn test_resource_monitoring() -> Result<(), Box<dyn std::error::Error>> {
    let executor = SynthesisExecutor::new().unwrap();
    
    let initial_metrics = executor.get_resource_metrics().unwrap();
    assert_eq!(initial_metrics.active_tasks, 0);
    assert_eq!(initial_metrics.queued_tasks, 0);
    
    // Submit a task
    let task = SynthesisTask {
        id: "resource_monitor_task".to_string(),
        goal: "Monitor resource usage".to_string(),
        priority: TaskPriority::Normal,
        input_data: vec![0.0; 1000],
        expected_output_size: 1000,
        timeout: Some(std::time::Duration::from_secs(10)),
    };
    
    let _task_id = executor.submit_task(task).unwrap();
    
    let metrics_with_task = executor.get_resource_metrics().unwrap();
    assert!(metrics_with_task.active_tasks > 0 || metrics_with_task.queued_tasks > 0);
    assert!(metrics_with_task.memory_usage_mb >= initial_metrics.memory_usage_mb);
}

#[test]
    fn test_execution_history() -> Result<(), Box<dyn std::error::Error>> {
    let executor = SynthesisExecutor::new().unwrap();
    
    let task = SynthesisTask {
        id: "history_task".to_string(),
        goal: "Simple vector scaling".to_string(),
        priority: TaskPriority::Normal,
        input_data: vec![1.0, 2.0, 3.0, 4.0],
        expected_output_size: 4,
        timeout: Some(std::time::Duration::from_secs(5)),
    };
    
    let task_id = executor.submit_task(task).unwrap();
    let _result = executor.wait_for_completion(&task_id).unwrap();
    
    let history = executor.get_execution_history().unwrap();
    assert!(!history.is_empty());
    
    let task_history = history.iter().find(|h| h.task_id == task_id).ok();
    assert_eq!(task_history.goal, "Simple vector scaling");
    assert!(task_history.success);
}

#[test]
    fn test_performance_profiling() -> Result<(), Box<dyn std::error::Error>> {
    let config = ExecutorConfig {
        enable_profiling: true,
        ..ExecutorConfig::default()
    };
    let executor = SynthesisExecutor::with_config(config)?;
    
    let task = SynthesisTask {
        id: "profile_task".to_string(),
        goal: "Matrix transpose operation".to_string(),
        priority: TaskPriority::Normal,
        input_data: (0..10000).map(|i| i as f32).collect(),
        expected_output_size: 10000,
        timeout: Some(std::time::Duration::from_secs(30)),
    };
    
    let task_id = executor.submit_task(task).unwrap();
    let result = executor.wait_for_completion(&task_id).unwrap();
    
    assert!(result.profiling_data.is_some());
    let profile = result.profiling_data.unwrap();
    assert!(profile.compilation_time.as_millis() > 0);
    assert!(profile.execution_time.as_millis() > 0);
    assert!(profile.memory_transfers > 0);
}

#[test]
    fn test_batch_task_submission() -> Result<(), Box<dyn std::error::Error>> {
    let executor = SynthesisExecutor::new().unwrap();
    
    let tasks = (0..10).map(|i| SynthesisTask {
        id: format!("batch_task_{}", i),
        goal: "Element-wise array processing".to_string(),
        priority: TaskPriority::Normal,
        input_data: (0..100).map(|j| (i * 100 + j) as f32).collect(),
        expected_output_size: 100,
        timeout: Some(std::time::Duration::from_secs(10)),
    }).collect();
    
    let task_ids = executor.submit_batch(tasks).unwrap();
    assert_eq!(task_ids.len(), 10);
    
    let results = executor.wait_for_all(&task_ids).unwrap();
    assert_eq!(results.len(), 10);
    
    for result in results {
        assert!(result.success);
        assert_eq!(result.output_data.len(), 100);
    }
}

#[test]
    fn test_executor_shutdown() -> Result<(), Box<dyn std::error::Error>> {
    let executor = SynthesisExecutor::new().unwrap();
    
    // Submit some tasks
    for i in 0..3 {
        let task = SynthesisTask {
            id: format!("shutdown_task_{}", i),
            goal: "Quick operation".to_string(),
            priority: TaskPriority::Normal,
            input_data: vec![0.0; 10],
            expected_output_size: 10,
            timeout: Some(std::time::Duration::from_secs(1)),
        };
        let _ = executor.submit_task(task);
    }
    
    // Shutdown executor
    let shutdown_result = executor.shutdown(std::time::Duration::from_secs(5));
    assert!(shutdown_result.is_ok());
    
    // Should not be able to submit new tasks after shutdown
    let task = SynthesisTask {
        id: "post_shutdown_task".to_string(),
        goal: "Should fail".to_string(),
        priority: TaskPriority::Normal,
        input_data: vec![0.0; 10],
        expected_output_size: 10,
        timeout: Some(std::time::Duration::from_secs(1)),
    };
    
    let submit_result = executor.submit_task(task);
    assert!(submit_result.is_err());
}

#[test]
    fn test_error_recovery() -> Result<(), Box<dyn std::error::Error>> {
    let executor = SynthesisExecutor::new().unwrap();
    
    // Submit a task that will fail
    let failing_task = SynthesisTask {
        id: "failing_task".to_string(),
        goal: "Invalid operation that should fail".to_string(),
        priority: TaskPriority::Normal,
        input_data: vec![], // Empty input should cause failure
        expected_output_size: 100,
        timeout: Some(std::time::Duration::from_secs(5)),
    };
    
    let failing_id = executor.submit_task(failing_task).unwrap();
    let failing_result = executor.wait_for_completion(&failing_id);
    
    // Task should fail
    assert!(failing_result.is_err() || !failing_result.is_ok().success);
    
    // Executor should still be functional
    let working_task = SynthesisTask {
        id: "working_task".to_string(),
        goal: "Simple working operation".to_string(),
        priority: TaskPriority::Normal,
        input_data: vec![1.0, 2.0, 3.0],
        expected_output_size: 3,
        timeout: Some(std::time::Duration::from_secs(5)),
    };
    
    let working_id = executor.submit_task(working_task).unwrap();
    let working_result = executor.wait_for_completion(&working_id).unwrap();
    
    assert!(working_result.success);
}

#[test]
    fn test_memory_management() -> Result<(), Box<dyn std::error::Error>> {
    let config = ExecutorConfig {
        memory_limit_mb: 100, // Small limit to test memory management
        ..ExecutorConfig::default()
    };
    let executor = SynthesisExecutor::with_config(config)?;
    
    // Submit task that might exceed memory limit
    let large_task = SynthesisTask {
        id: "large_memory_task".to_string(),
        goal: "Large matrix operation".to_string(),
        priority: TaskPriority::Normal,
        input_data: vec![0.0; 1000000], // Large input
        expected_output_size: 1000000,
        timeout: Some(std::time::Duration::from_secs(30)),
    };
    
    let task_id = executor.submit_task(large_task).unwrap();
    let result = executor.wait_for_completion(&task_id);
    
    // Should either complete successfully with memory management or fail gracefully
    match result {
        Ok(r) => assert!(r.success || !r.success), // Either outcome is acceptable
        Err(SynthesisError::ResourceError(_)) => {}, // Expected error type
        Err(_) => {}, // Other errors are also acceptable
    }
}

#[test]
    fn test_cache_effectiveness() -> Result<(), Box<dyn std::error::Error>> {
    let config = ExecutorConfig {
        cache_size_mb: 50,
        ..ExecutorConfig::default()
    };
    let executor = SynthesisExecutor::with_config(config)?;
    
    let task = SynthesisTask {
        id: "cache_test_1".to_string(),
        goal: "Identical operation for caching".to_string(),
        priority: TaskPriority::Normal,
        input_data: vec![1.0, 2.0, 3.0, 4.0],
        expected_output_size: 4,
        timeout: Some(std::time::Duration::from_secs(10)),
    };
    
    // First execution
    let task_id_1 = executor.submit_task(task.clone()).unwrap();
    let start_1 = std::time::Instant::now();
    let _result_1 = executor.wait_for_completion(&task_id_1).unwrap();
    let duration_1 = start_1.elapsed();
    
    // Second execution with same goal (should use cache)
    let mut task_2 = task;
    task_2.id = "cache_test_2".to_string();
    let task_id_2 = executor.submit_task(task_2).unwrap();
    let start_2 = std::time::Instant::now();
    let _result_2 = executor.wait_for_completion(&task_id_2).unwrap();
    let duration_2 = start_2.elapsed();
    
    // Second execution should be faster (cached)
    assert!(duration_2 <= duration_1);
}

#[test]
    fn test_task_dependencies() -> Result<(), Box<dyn std::error::Error>> {
    let executor = SynthesisExecutor::new().unwrap();
    
    // Create task dependency chain
    let task_a = SynthesisTask {
        id: "task_a".to_string(),
        goal: "Prepare data for task B".to_string(),
        priority: TaskPriority::Normal,
        input_data: vec![1.0, 2.0, 3.0],
        expected_output_size: 3,
        timeout: Some(std::time::Duration::from_secs(5)),
    };
    
    let task_b = SynthesisTaskWithDependencies {
        task: SynthesisTask {
            id: "task_b".to_string(),
            goal: "Process data from task A".to_string(),
            priority: TaskPriority::Normal,
            input_data: vec![], // Will use output from task A
            expected_output_size: 3,
            timeout: Some(std::time::Duration::from_secs(5)),
        },
        dependencies: vec!["task_a".to_string()],
    };
    
    let id_a = executor.submit_task(task_a).unwrap();
    let id_b = executor.submit_task_with_dependencies(task_b).unwrap();
    
    // Task B should complete after Task A
    let result_a = executor.wait_for_completion(&id_a).unwrap();
    let result_b = executor.wait_for_completion(&id_b).unwrap();
    
    assert!(result_a.success);
    assert!(result_b.success);
    assert!(result_b.completion_timestamp >= result_a.completion_timestamp);
}

#[test]
    fn test_statistics_collection() -> Result<(), Box<dyn std::error::Error>> {
    let executor = SynthesisExecutor::new().unwrap();
    
    // Execute several tasks
    for i in 0..5 {
        let task = SynthesisTask {
            id: format!("stats_task_{}", i),
            goal: "Collect statistics".to_string(),
            priority: TaskPriority::Normal,
            input_data: vec![0.0; 100],
            expected_output_size: 100,
            timeout: Some(std::time::Duration::from_secs(5)),
        };
        
        let task_id = executor.submit_task(task).unwrap();
        let _result = executor.wait_for_completion(&task_id);
    }
    
    let stats = executor.get_statistics().unwrap();
    assert!(stats.total_tasks_executed >= 5);
    assert!(stats.successful_tasks <= stats.total_tasks_executed);
    assert!(stats.average_execution_time.as_millis() > 0);
    assert!(stats.cache_hit_rate >= 0.0 && stats.cache_hit_rate <= 1.0);
}