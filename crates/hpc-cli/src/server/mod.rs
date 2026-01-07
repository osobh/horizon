//! Bootstrap HTTP server module
//!
//! Provides an embedded HTTP server that serves bootstrap scripts to remote nodes.
//! The server is started during the bootstrap process and serves platform-specific
//! installation scripts.

pub mod http;
pub mod scripts;

pub use http::{BootstrapCallback, BootstrapServer, ServerState};
pub use scripts::{BootstrapScript, ScriptGenerator};
