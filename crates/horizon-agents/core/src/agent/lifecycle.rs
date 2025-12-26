use crate::error::{AgentError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentState {
    Uninitialized,
    Initializing,
    Ready,
    Running,
    ShuttingDown,
    Shutdown,
    Error,
}

impl AgentState {
    pub fn can_transition_to(&self, target: AgentState) -> bool {
        use AgentState::*;
        match (self, target) {
            // Can always stay in same state
            (a, b) if a == &b => true,
            // Normal lifecycle transitions
            (Uninitialized, Initializing) => true,
            (Initializing, Ready) => true,
            (Ready, Running) => true,
            (Running, Ready) => true,
            (Ready, ShuttingDown) => true,
            (Running, ShuttingDown) => true,
            (ShuttingDown, Shutdown) => true,
            // Error transitions
            (Initializing, Error) => true,
            (Running, Error) => true,
            // Can restart from error or shutdown
            (Error, Initializing) => true,
            (Shutdown, Initializing) => true,
            // All other transitions are invalid
            _ => false,
        }
    }

    pub fn is_operational(&self) -> bool {
        matches!(self, Self::Ready | Self::Running)
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Shutdown | Self::Error)
    }
}

pub struct Lifecycle {
    state: AgentState,
}

impl Lifecycle {
    pub fn new() -> Self {
        Self {
            state: AgentState::Uninitialized,
        }
    }

    pub fn state(&self) -> AgentState {
        self.state
    }

    pub fn transition_to(&mut self, target: AgentState) -> Result<()> {
        if !self.state.can_transition_to(target) {
            return Err(AgentError::InitializationFailed(format!(
                "Invalid state transition from {:?} to {:?}",
                self.state, target
            )));
        }

        self.state = target;
        Ok(())
    }

    pub fn is_operational(&self) -> bool {
        self.state.is_operational()
    }

    pub fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }

    pub fn require_operational(&self) -> Result<()> {
        if !self.is_operational() {
            return Err(AgentError::NotInitialized);
        }
        Ok(())
    }
}

impl Default for Lifecycle {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_state_transitions() {
        use AgentState::*;

        // Normal lifecycle
        assert!(Uninitialized.can_transition_to(Initializing));
        assert!(Initializing.can_transition_to(Ready));
        assert!(Ready.can_transition_to(Running));
        assert!(Running.can_transition_to(Ready));
        assert!(Ready.can_transition_to(ShuttingDown));
        assert!(ShuttingDown.can_transition_to(Shutdown));

        // Error transitions
        assert!(Initializing.can_transition_to(Error));
        assert!(Running.can_transition_to(Error));

        // Restart transitions
        assert!(Error.can_transition_to(Initializing));
        assert!(Shutdown.can_transition_to(Initializing));

        // Invalid transitions
        assert!(!Uninitialized.can_transition_to(Ready));
        assert!(!Running.can_transition_to(Shutdown));
        assert!(!ShuttingDown.can_transition_to(Running));
    }

    #[test]
    fn test_agent_state_is_operational() {
        use AgentState::*;

        assert!(!Uninitialized.is_operational());
        assert!(!Initializing.is_operational());
        assert!(Ready.is_operational());
        assert!(Running.is_operational());
        assert!(!ShuttingDown.is_operational());
        assert!(!Shutdown.is_operational());
        assert!(!Error.is_operational());
    }

    #[test]
    fn test_agent_state_is_terminal() {
        use AgentState::*;

        assert!(!Uninitialized.is_terminal());
        assert!(!Initializing.is_terminal());
        assert!(!Ready.is_terminal());
        assert!(!Running.is_terminal());
        assert!(!ShuttingDown.is_terminal());
        assert!(Shutdown.is_terminal());
        assert!(Error.is_terminal());
    }

    #[test]
    fn test_lifecycle_creation() {
        let lifecycle = Lifecycle::new();
        assert_eq!(lifecycle.state(), AgentState::Uninitialized);
    }

    #[test]
    fn test_lifecycle_transition_success() {
        let mut lifecycle = Lifecycle::new();

        assert!(lifecycle.transition_to(AgentState::Initializing).is_ok());
        assert_eq!(lifecycle.state(), AgentState::Initializing);

        assert!(lifecycle.transition_to(AgentState::Ready).is_ok());
        assert_eq!(lifecycle.state(), AgentState::Ready);
    }

    #[test]
    fn test_lifecycle_transition_failure() {
        let mut lifecycle = Lifecycle::new();

        let result = lifecycle.transition_to(AgentState::Ready);
        assert!(result.is_err());
        assert_eq!(lifecycle.state(), AgentState::Uninitialized);
    }

    #[test]
    fn test_lifecycle_is_operational() {
        let mut lifecycle = Lifecycle::new();
        assert!(!lifecycle.is_operational());

        lifecycle.transition_to(AgentState::Initializing).unwrap();
        assert!(!lifecycle.is_operational());

        lifecycle.transition_to(AgentState::Ready).unwrap();
        assert!(lifecycle.is_operational());

        lifecycle.transition_to(AgentState::Running).unwrap();
        assert!(lifecycle.is_operational());

        lifecycle.transition_to(AgentState::ShuttingDown).unwrap();
        assert!(!lifecycle.is_operational());
    }

    #[test]
    fn test_lifecycle_is_terminal() {
        let mut lifecycle = Lifecycle::new();
        assert!(!lifecycle.is_terminal());

        lifecycle.transition_to(AgentState::Initializing).unwrap();
        lifecycle.transition_to(AgentState::Ready).unwrap();
        assert!(!lifecycle.is_terminal());

        lifecycle.transition_to(AgentState::ShuttingDown).unwrap();
        lifecycle.transition_to(AgentState::Shutdown).unwrap();
        assert!(lifecycle.is_terminal());
    }

    #[test]
    fn test_lifecycle_require_operational() {
        let mut lifecycle = Lifecycle::new();

        let result = lifecycle.require_operational();
        assert!(result.is_err());

        lifecycle.transition_to(AgentState::Initializing).unwrap();
        lifecycle.transition_to(AgentState::Ready).unwrap();

        assert!(lifecycle.require_operational().is_ok());
    }

    #[test]
    fn test_lifecycle_full_cycle() {
        let mut lifecycle = Lifecycle::new();

        // Initialize
        lifecycle.transition_to(AgentState::Initializing).unwrap();
        lifecycle.transition_to(AgentState::Ready).unwrap();

        // Run
        lifecycle.transition_to(AgentState::Running).unwrap();
        lifecycle.transition_to(AgentState::Ready).unwrap();

        // Shutdown
        lifecycle.transition_to(AgentState::ShuttingDown).unwrap();
        lifecycle.transition_to(AgentState::Shutdown).unwrap();

        assert!(lifecycle.is_terminal());
    }

    #[test]
    fn test_lifecycle_error_recovery() {
        let mut lifecycle = Lifecycle::new();

        lifecycle.transition_to(AgentState::Initializing).unwrap();
        lifecycle.transition_to(AgentState::Error).unwrap();
        assert!(lifecycle.is_terminal());

        // Can restart from error
        lifecycle.transition_to(AgentState::Initializing).unwrap();
        lifecycle.transition_to(AgentState::Ready).unwrap();
        assert!(lifecycle.is_operational());
    }
}
