//! # Horizon MCP (Model Context Protocol)
//!
//! This crate provides a complete implementation of the Model Context Protocol (MCP)
//! for the Horizon platform. MCP enables AI agents to interact with tools and services
//! through a standardized JSON-RPC 2.0 interface.
//!
//! ## Features
//!
//! - **JSON-RPC 2.0**: Full JSON-RPC 2.0 protocol implementation
//! - **Tool Registry**: Register and manage tools with type-safe schemas
//! - **Server**: Build MCP servers that expose tools to AI agents
//! - **Client**: Call MCP tools from client applications
//! - **Async/Await**: Full async support with Tokio
//!
//! ## Server Example
//!
//! ```rust,ignore
//! use hpc_mcp::{McpServer, Tool, ToolCall, ToolResult, ToolHandler, Content};
//! use async_trait::async_trait;
//!
//! struct EchoHandler;
//!
//! #[async_trait]
//! impl ToolHandler for EchoHandler {
//!     async fn handle(&self, call: ToolCall) -> hpc_mcp::Result<ToolResult> {
//!         let msg = call.arguments
//!             .and_then(|v| v.get("message").and_then(|m| m.as_str()))
//!             .unwrap_or("no message");
//!         Ok(ToolResult::success(vec![Content::text(msg)]))
//!     }
//! }
//!
//! #[tokio::main]
//! async fn main() {
//!     let tool = Tool::new("echo", "Echo a message");
//!     let server = McpServer::builder()
//!         .register_tool(tool, EchoHandler)
//!         .build();
//!
//!     let request = r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#;
//!     let response = server.process_request(request).await;
//!     println!("{}", response);
//! }
//! ```
//!
//! ## Client Example
//!
//! ```rust
//! use hpc_mcp::{ToolClient, ToolCall};
//!
//! fn main() {
//!     let client = ToolClient::new();
//!     let request = client.list_tools_request();
//!     println!("{}", request);
//! }
//! ```

pub mod client;
pub mod error;
pub mod protocol;
pub mod server;

pub use client::{Connection, ToolClient};
pub use error::{McpError, Result};
pub use protocol::{
    Content, JsonRpcCodec, JsonRpcError, JsonRpcRequest, JsonRpcResponse, PropertySchema,
    RequestId, Tool, ToolCall, ToolResult, ToolSchema,
};
pub use server::{McpServer, Router, ServerBuilder, ToolHandler};
