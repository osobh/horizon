//! Template agent pattern for reducing boilerplate.
//!
//! This module provides a `BaseAgent<L>` wrapper that handles all the common
//! agent boilerplate (lifecycle management, health checks, autonomy level handling)
//! while delegating the actual processing logic to an `AgentLogic` implementation.
//!
//! # Example
//!
//! ```ignore
//! use horizon_agents_core::{
//!     AgentConfig, AgentLogic, AgentRequest, AgentResponse, BaseAgent, Result,
//! };
//! use async_trait::async_trait;
//!
//! pub struct MyLogic;
//!
//! #[async_trait]
//! impl AgentLogic for MyLogic {
//!     async fn process(&self, name: &str, request: AgentRequest) -> Result<AgentResponse> {
//!         Ok(AgentResponse::new(
//!             request.id,
//!             format!("{} processed: {}", name, request.content),
//!         ))
//!     }
//! }
//!
//! pub type MyAgent = BaseAgent<MyLogic>;
//! ```

use async_trait::async_trait;

use super::{
    Agent, AgentConfig, AgentRequest, AgentResponse, AgentState, AutonomyLevel, HealthStatus,
    Lifecycle,
};
use crate::error::{AgentError, Result};

/// Trait for implementing agent-specific processing logic.
///
/// Implement this trait to define what your agent does when it receives a request.
/// All the boilerplate (initialization, health checks, shutdown, autonomy management)
/// is handled by `BaseAgent`.
#[async_trait]
pub trait AgentLogic: Send + Sync {
    /// Process an incoming request and return a response.
    ///
    /// # Arguments
    /// * `name` - The agent's name (from config)
    /// * `request` - The incoming request to process
    ///
    /// # Returns
    /// The response to send back
    async fn process(&self, name: &str, request: AgentRequest) -> Result<AgentResponse>;

    /// Optional hook called during agent initialization.
    /// Override this to perform custom initialization.
    async fn on_init(&mut self) -> Result<()> {
        Ok(())
    }

    /// Optional hook called during agent shutdown.
    /// Override this to perform cleanup.
    async fn on_shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

/// A base agent that handles all common agent boilerplate.
///
/// This struct wraps an `AgentLogic` implementation and provides all the
/// standard agent functionality: lifecycle management, health checks,
/// autonomy level handling, etc.
///
/// # Type Parameters
/// * `L` - The `AgentLogic` implementation that provides the processing logic
pub struct BaseAgent<L: AgentLogic> {
    config: AgentConfig,
    lifecycle: Lifecycle,
    logic: L,
}

impl<L: AgentLogic> BaseAgent<L> {
    /// Create a new base agent with the given configuration and logic.
    ///
    /// # Arguments
    /// * `config` - Agent configuration
    /// * `logic` - The processing logic implementation
    ///
    /// # Returns
    /// A new `BaseAgent` instance, or an error if config validation fails
    pub fn new(config: AgentConfig, logic: L) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            lifecycle: Lifecycle::new(),
            logic,
        })
    }

    /// Get a reference to the underlying logic.
    pub fn logic(&self) -> &L {
        &self.logic
    }

    /// Get a mutable reference to the underlying logic.
    pub fn logic_mut(&mut self) -> &mut L {
        &mut self.logic
    }

    /// Get the agent's configuration.
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// Get the agent's lifecycle state.
    pub fn lifecycle(&self) -> &Lifecycle {
        &self.lifecycle
    }
}

#[async_trait]
impl<L: AgentLogic> Agent for BaseAgent<L> {
    fn name(&self) -> &str {
        &self.config.name
    }

    async fn init(&mut self) -> Result<()> {
        self.lifecycle.transition_to(AgentState::Initializing)?;
        self.logic.on_init().await?;
        self.lifecycle.transition_to(AgentState::Ready)?;
        Ok(())
    }

    async fn process(&self, request: AgentRequest) -> Result<AgentResponse> {
        self.lifecycle.require_operational()?;
        self.logic.process(self.name(), request).await
    }

    async fn health(&self) -> Result<HealthStatus> {
        if self.lifecycle.is_operational() {
            Ok(HealthStatus::Healthy)
        } else {
            Ok(HealthStatus::Unhealthy {
                reason: "Agent not operational".to_string(),
            })
        }
    }

    async fn shutdown(&mut self) -> Result<()> {
        self.lifecycle.transition_to(AgentState::ShuttingDown)?;
        self.logic.on_shutdown().await?;
        self.lifecycle.transition_to(AgentState::Shutdown)?;
        Ok(())
    }

    fn autonomy_level(&self) -> AutonomyLevel {
        self.config.autonomy_level
    }

    fn set_autonomy_level(&mut self, level: AutonomyLevel) -> Result<()> {
        if !self.config.autonomy_level.can_transition_to(level) {
            return Err(AgentError::InvalidAutonomyTransition {
                from: format!("{:?}", self.config.autonomy_level),
                to: format!("{:?}", level),
            });
        }
        self.config.autonomy_level = level;
        Ok(())
    }
}

/// A stub logic implementation that returns a simple acknowledgment.
///
/// Use this for placeholder agents or testing.
pub struct StubLogic;

#[async_trait]
impl AgentLogic for StubLogic {
    async fn process(&self, name: &str, request: AgentRequest) -> Result<AgentResponse> {
        Ok(
            AgentResponse::new(request.id, format!("Stub agent {} processed request", name))
                .with_recommendation(
                    "This is a stub agent with minimal implementation".to_string(),
                ),
        )
    }
}

/// A type alias for a stub agent.
pub type StubAgent = BaseAgent<StubLogic>;

impl StubAgent {
    /// Create a new stub agent with the given configuration.
    pub fn new_stub(config: AgentConfig) -> Result<Self> {
        Self::new(config, StubLogic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_stub_agent_creation() {
        let config = AgentConfig::new("test-agent".to_string());
        let agent = StubAgent::new_stub(config);
        assert!(agent.is_ok());
    }

    #[tokio::test]
    async fn test_stub_agent_init() {
        let config = AgentConfig::new("test-agent".to_string());
        let mut agent = StubAgent::new_stub(config).unwrap();
        assert!(agent.init().await.is_ok());
    }

    #[tokio::test]
    async fn test_stub_agent_process() {
        let config = AgentConfig::new("test-agent".to_string());
        let mut agent = StubAgent::new_stub(config).unwrap();
        agent.init().await.unwrap();

        let request = AgentRequest::new("test request".to_string());
        let response = agent.process(request).await;
        assert!(response.is_ok());
        assert!(response.unwrap().content.contains("Stub agent"));
    }

    #[tokio::test]
    async fn test_stub_agent_health() {
        let config = AgentConfig::new("test-agent".to_string());
        let mut agent = StubAgent::new_stub(config).unwrap();

        // Before init, agent is not healthy
        let health = agent.health().await.unwrap();
        assert!(!health.is_healthy());

        // After init, agent is healthy
        agent.init().await.unwrap();
        let health = agent.health().await.unwrap();
        assert!(health.is_healthy());
    }

    #[tokio::test]
    async fn test_stub_agent_shutdown() {
        let config = AgentConfig::new("test-agent".to_string());
        let mut agent = StubAgent::new_stub(config).unwrap();
        agent.init().await.unwrap();
        assert!(agent.shutdown().await.is_ok());
    }

    #[tokio::test]
    async fn test_stub_agent_autonomy_level() {
        let config = AgentConfig::new("test-agent".to_string());
        let agent = StubAgent::new_stub(config).unwrap();
        assert_eq!(agent.autonomy_level(), AutonomyLevel::Low);
    }

    #[tokio::test]
    async fn test_stub_agent_set_autonomy_level() {
        let config = AgentConfig::new("test-agent".to_string());
        let mut agent = StubAgent::new_stub(config).unwrap();

        assert!(agent.set_autonomy_level(AutonomyLevel::Medium).is_ok());
        assert_eq!(agent.autonomy_level(), AutonomyLevel::Medium);
    }

    #[tokio::test]
    async fn test_stub_agent_name() {
        let config = AgentConfig::new("my-agent".to_string());
        let agent = StubAgent::new_stub(config).unwrap();
        assert_eq!(agent.name(), "my-agent");
    }

    // Test custom logic
    struct CustomLogic {
        prefix: String,
    }

    #[async_trait]
    impl AgentLogic for CustomLogic {
        async fn process(&self, name: &str, request: AgentRequest) -> Result<AgentResponse> {
            Ok(AgentResponse::new(
                request.id,
                format!("{}: {} says {}", self.prefix, name, request.content),
            ))
        }
    }

    #[tokio::test]
    async fn test_custom_logic() {
        let config = AgentConfig::new("custom-agent".to_string());
        let logic = CustomLogic {
            prefix: "Custom".to_string(),
        };
        let mut agent = BaseAgent::new(config, logic).unwrap();
        agent.init().await.unwrap();

        let request = AgentRequest::new("hello".to_string());
        let response = agent.process(request).await.unwrap();
        assert!(response.content.contains("Custom"));
        assert!(response.content.contains("custom-agent"));
        assert!(response.content.contains("hello"));
    }
}
