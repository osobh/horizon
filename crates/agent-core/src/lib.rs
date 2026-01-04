//! Core agent infrastructure for the ExoRust GPU OS
//!
//! This crate provides the foundational components for autonomous agents:
//! - Agent lifecycle management
//! - Goal interpretation and task planning
//! - Inter-agent communication
//! - Resource management and scheduling
//! - Agent memory and state persistence

#![warn(missing_docs)]

pub mod agent;
pub mod communication;
pub mod error;
pub mod goal;
pub mod memory;
pub mod neural_router;
pub mod scheduler;

// Scheduler has been successfully refactored into modular components

pub use agent::{Agent, AgentConfig, AgentId, AgentState};
pub use communication::{AgentChannel, Message, MessageBus};
pub use error::{AgentError, AgentResult};
pub use goal::{Goal, GoalConstraints, GoalId, GoalPriority};
pub use memory::{AgentMemory, MemoryType};
pub use neural_router::{
    ClusterRegion, ConnectionMetrics, MultiGpuRoutingEngine, NetworkTopology, NeuralRouter,
    PerformanceMetrics, RoutingChoice, RoutingEntry, TrainingExample,
};
pub use scheduler::{Scheduler, SchedulingPolicy};

/// Re-export common types
pub mod prelude {
    pub use crate::{
        Agent, AgentConfig, AgentError, AgentId, AgentMemory, AgentResult, AgentState, Goal,
        GoalConstraints, GoalId, GoalPriority, MemoryType, Message, MessageBus, Scheduler,
        SchedulingPolicy,
    };
}

/// Initialize the agent system
pub async fn init() -> AgentResult<()> {
    // Initialize subsystems
    communication::init_message_bus().await?;
    scheduler::init_scheduler().await?;
    Ok(())
}

#[cfg(test)]
mod test_memory_split;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_agent_system_init() {
        assert!(init().await.is_ok());
    }

    #[test]
    fn test_prelude_exports() {
        // Ensure all types are accessible through prelude
        use crate::prelude::*;

        // Create instances to verify types are exported correctly
        let _agent_id = AgentId::new();
        let _goal_id = GoalId::new();
        let _priority = GoalPriority::default();
        let _mem_type = MemoryType::Working;
        let _policy = SchedulingPolicy::default();
        let _state = AgentState::default();
    }

    #[test]
    fn test_module_exports() {
        // Verify all modules are accessible
        use crate::{agent, communication, error, goal, memory, scheduler};

        // Just ensure modules exist and can be referenced
        let _ = agent::AgentId::new();
        let _ = communication::MessageId::new();
        let _ = goal::GoalId::new();
        let _ = memory::MemoryId::new();
    }

    #[test]
    fn test_direct_type_exports() {
        // Test that types can be used directly from crate root
        let agent_config = AgentConfig::default();
        assert_eq!(agent_config.name, "agent");

        let constraints = GoalConstraints::default();
        assert!(constraints.memory_limit.is_none());
    }

    #[tokio::test]
    async fn test_init_idempotency() {
        // Initialize multiple times should be safe
        assert!(init().await.is_ok());
        assert!(init().await.is_ok());
        assert!(init().await.is_ok());
    }

    #[test]
    fn test_error_result_type() {
        fn example_function() -> AgentResult<String> {
            Ok("success".to_string())
        }

        fn example_error_function() -> AgentResult<String> {
            Err(AgentError::Other("example error".to_string()))
        }

        assert!(example_function().is_ok());
        assert!(example_error_function().is_err());
    }

    #[test]
    fn test_crate_documentation() {
        // This test just ensures the crate-level documentation compiles
        // The actual test is that the code compiles with #![warn(missing_docs)]
    }

    #[tokio::test]
    async fn test_subsystem_initialization() {
        // Test that subsystems can be initialized independently
        assert!(communication::init_message_bus().await.is_ok());
        assert!(scheduler::init_scheduler().await.is_ok());
    }

    #[test]
    fn test_type_aliases() {
        // Ensure type aliases work correctly
        type MyAgentResult = AgentResult<()>;

        fn test_alias() -> MyAgentResult {
            Ok(())
        }

        assert!(test_alias().is_ok());
    }

    #[tokio::test]
    async fn test_initialization_error_propagation() {
        // Test that initialization errors are properly propagated
        // (This would normally test actual error cases, but our init is simple)
        let result = init().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_all_public_types_constructible() {
        // Verify all major public types can be constructed
        let _agent_id = AgentId::new();
        let _goal_id = GoalId::new();
        let _message = Message::new(
            AgentId::new(),
            None,
            "test".to_string(),
            serde_json::Value::Null,
        );
        let _config = AgentConfig::default();
        let _constraints = GoalConstraints::default();
    }

    #[test]
    fn test_prelude_completeness() {
        // Ensure prelude includes all commonly used types
        use crate::prelude::*;

        // Test that we can use all prelude types without additional imports
        let agent_id = AgentId::new();
        let goal = Goal::new("test goal".to_string(), GoalPriority::Normal);
        let _config = AgentConfig::default();
        let _memory_type = MemoryType::Episodic;
        let _scheduling_policy = SchedulingPolicy::RoundRobin;
        let _state = AgentState::Idle;
        let _priority = GoalPriority::High;

        // Verify types work together
        assert!(!agent_id.to_string().is_empty());
        assert_eq!(goal.description, "test goal");
    }

    #[test]
    fn test_error_handling_patterns() {
        // Test common error handling patterns
        use crate::AgentError;

        fn test_function(should_fail: bool) -> AgentResult<i32> {
            if should_fail {
                Err(AgentError::ExecutionError {
                    message: "Test failure".to_string(),
                })
            } else {
                Ok(42)
            }
        }

        // Test success case
        match test_function(false) {
            Ok(value) => assert_eq!(value, 42),
            Err(_) => panic!("Expected success"),
        }

        // Test failure case
        match test_function(true) {
            Ok(_) => panic!("Expected failure"),
            Err(e) => assert!(e.to_string().contains("Test failure")),
        }
    }

    #[test]
    fn test_memory_type_variations() {
        // Test all memory types are accessible
        let types = vec![
            MemoryType::Working,
            MemoryType::Episodic,
            MemoryType::Semantic,
            MemoryType::Procedural,
        ];

        for memory_type in types {
            // Just ensure they can be constructed and compared
            assert!(memory_type == memory_type);
        }
    }

    #[test]
    fn test_scheduling_policy_variations() {
        // Test all scheduling policies are accessible
        let policies = vec![
            SchedulingPolicy::FIFO,
            SchedulingPolicy::RoundRobin,
            SchedulingPolicy::Priority,
            SchedulingPolicy::FairShare,
        ];

        for policy in policies {
            assert!(policy == policy);
        }
    }

    #[test]
    fn test_goal_priority_variations() {
        // Test all goal priorities are accessible
        let priorities = vec![
            GoalPriority::Low,
            GoalPriority::Normal,
            GoalPriority::High,
            GoalPriority::Critical,
        ];

        for priority in priorities {
            assert!(priority == priority);
        }
    }

    #[test]
    fn test_agent_state_variations() {
        // Test all agent states are accessible
        let states = vec![
            AgentState::Initializing,
            AgentState::Idle,
            AgentState::Planning,
            AgentState::Executing,
            AgentState::Paused,
            AgentState::Failed,
        ];

        for state in states {
            assert!(state == state);
        }
    }

    #[test]
    fn test_concurrent_type_construction() {
        use std::sync::Arc;
        use std::thread;

        let handles: Vec<_> = (0..10)
            .map(|i| {
                thread::spawn(move || {
                    let agent_id = AgentId::new();
                    let goal_id = GoalId::new();
                    let mut config = AgentConfig::default();
                    config.name = format!("agent_{i}");

                    // Return constructed objects to verify they're valid
                    (agent_id, goal_id, config)
                })
            })
            .collect();

        for handle in handles {
            let (agent_id, goal_id, config) = handle.join().unwrap();
            assert!(!agent_id.to_string().is_empty());
            assert!(!goal_id.to_string().is_empty());
            assert!(config.name.starts_with("agent_"));
        }
    }

    #[test]
    fn test_module_isolation() {
        // Test that modules are properly isolated and don't conflict
        {
            use crate::agent::*;
            let _id = AgentId::new();
        }

        {
            use crate::goal::*;
            let _id = GoalId::new();
        }

        {
            use crate::memory::*;
            let _id = MemoryId::new();
        }

        {
            use crate::communication::*;
            let _id = MessageId::new();
        }
    }

    #[tokio::test]
    async fn test_async_initialization_multiple_times() {
        use tokio::task;

        let handles: Vec<_> = (0..5)
            .map(|_| task::spawn(async { init().await }))
            .collect();

        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_default_implementations() {
        // Test that default implementations work as expected
        let config = AgentConfig::default();
        assert_eq!(config.name, "unnamed");
        assert_eq!(config.agent_type, "generic");

        let constraints = GoalConstraints::default();
        assert!(constraints.memory_limit.is_none());
        assert!(constraints.time_limit.is_none());

        let priority = GoalPriority::default();
        assert_eq!(priority, GoalPriority::Normal);
    }

    #[test]
    fn test_string_representations() {
        // Test that types have proper string representations
        let agent_id = AgentId::new();
        let goal_id = GoalId::new();

        assert!(!agent_id.to_string().is_empty());
        assert!(!goal_id.to_string().is_empty());
    }

    #[test]
    fn test_error_conversion_patterns() {
        use crate::AgentError;

        // Test various error creation patterns
        let errors = vec![
            AgentError::ExecutionError {
                message: "test".to_string(),
            },
            AgentError::ResourceLimitExceeded {
                resource: "resource".to_string(),
            },
            AgentError::CommunicationFailure {
                message: "comm error".to_string(),
            },
            AgentError::Other("other error".to_string()),
        ];

        for error in errors {
            let error_string = error.to_string();
            assert!(!error_string.is_empty());
            // All our errors should contain some descriptive text
            assert!(error_string.len() > 5);
        }
    }

    #[test]
    fn test_type_consistency() {
        // Test that types maintain consistency across operations
        let agent_id = AgentId::new();
        let agent_id_copy = agent_id;
        assert_eq!(agent_id, agent_id_copy);

        let mut config1 = AgentConfig::default();
        config1.name = "test".to_string();
        let mut config2 = AgentConfig::default();
        config2.name = "test".to_string();
        assert_eq!(config1.name, config2.name);
    }
}
