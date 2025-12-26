//! StratoSwarm Visual Editor
//!
//! A web-based visual topology editor for designing and managing
//! distributed GPU/CPU topologies in StratoSwarm.

pub mod api;
pub mod error;
pub mod rest_api_simple;
pub mod server;
pub mod topology;
pub mod websocket;
pub mod xp_api;

pub use error::{Result, VisualEditorError};

/// Re-export commonly used types
pub mod prelude {
    pub use crate::error::{Result, VisualEditorError};
    pub use crate::rest_api_simple::{create_rest_api_router, ApiResponse};
    pub use crate::server::create_app;
    pub use crate::topology::{TopologyValidator, ValidationResult};
    pub use crate::websocket::{WebSocketHandler, WebSocketMessage};
}
