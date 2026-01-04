use crate::protocol::{
    JsonRpcCodec, JsonRpcError, JsonRpcRequest, JsonRpcResponse, RequestId, Tool, ToolCall,
};
use crate::server::router::{Router, ToolHandler};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, warn};

/// MCP Server
pub struct McpServer {
    router: Arc<RwLock<Router>>,
    codec: JsonRpcCodec,
}

impl McpServer {
    /// Create a new server builder
    pub fn builder() -> ServerBuilder {
        ServerBuilder::new()
    }

    /// Process a JSON-RPC request
    pub async fn process_request(&self, request_str: &str) -> String {
        let request = match self.codec.decode_request(request_str) {
            Ok(req) => req,
            Err(e) => {
                error!("Failed to decode request: {}", e);
                let response = JsonRpcResponse::error(None, JsonRpcError::parse_error());
                return self.codec.encode_response(&response).unwrap_or_default();
            }
        };

        let response = self.handle_request(request).await;
        self.codec.encode_response(&response).unwrap_or_default()
    }

    async fn handle_request(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        debug!("Handling request: method={}", request.method);

        match request.method.as_str() {
            "tools/list" => self.handle_list_tools(request.id).await,
            "tools/call" => self.handle_tool_call(request).await,
            _ => {
                warn!("Method not found: {}", request.method);
                JsonRpcResponse::error(request.id, JsonRpcError::method_not_found())
            }
        }
    }

    async fn handle_list_tools(&self, id: Option<RequestId>) -> JsonRpcResponse {
        let router = self.router.read().await;
        let tools = router.list_tools();

        match serde_json::to_value(tools) {
            Ok(value) => JsonRpcResponse::success(
                id.unwrap_or(RequestId::Number(0)),
                serde_json::json!({"tools": value}),
            ),
            Err(e) => {
                error!("Failed to serialize tools: {}", e);
                JsonRpcResponse::error(id, JsonRpcError::internal_error())
            }
        }
    }

    async fn handle_tool_call(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        let params = match request.params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::error(
                    request.id,
                    JsonRpcError::invalid_params("Missing params"),
                )
            }
        };

        let tool_call: ToolCall = match serde_json::from_value(params) {
            Ok(call) => call,
            Err(e) => {
                error!("Failed to parse tool call: {}", e);
                return JsonRpcResponse::error(
                    request.id,
                    JsonRpcError::invalid_params("Invalid tool call format"),
                );
            }
        };

        let router = self.router.read().await;
        match router.route(tool_call).await {
            Ok(result) => {
                let value = serde_json::to_value(result).unwrap_or_default();
                JsonRpcResponse::success(request.id.unwrap_or(RequestId::Number(0)), value)
            }
            Err(e) => {
                error!("Tool execution failed: {}", e);
                JsonRpcResponse::error(request.id, JsonRpcError::custom(-32000, e.to_string()))
            }
        }
    }
}

/// Server builder
pub struct ServerBuilder {
    router: Router,
}

impl ServerBuilder {
    pub fn new() -> Self {
        Self {
            router: Router::new(),
        }
    }

    /// Register a tool with its handler
    pub fn register_tool<H: ToolHandler + 'static>(mut self, tool: Tool, handler: H) -> Self {
        self.router = self.router.register(tool, handler);
        self
    }

    /// Build the server
    pub fn build(self) -> McpServer {
        McpServer {
            router: Arc::new(RwLock::new(self.router)),
            codec: JsonRpcCodec::new(),
        }
    }
}

impl Default for ServerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{Content, ToolResult};
    use crate::server::router::ToolHandler;
    use async_trait::async_trait;

    struct EchoHandler;

    #[async_trait]
    impl ToolHandler for EchoHandler {
        async fn handle(&self, call: ToolCall) -> crate::error::Result<ToolResult> {
            let msg = call
                .arguments
                .as_ref()
                .and_then(|v| v.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("no message")
                .to_string();

            Ok(ToolResult::success(vec![Content::text(msg)]))
        }
    }

    #[tokio::test]
    async fn test_server_builder() {
        let tool = Tool::new("echo", "Echo tool");
        let server = McpServer::builder()
            .register_tool(tool, EchoHandler)
            .build();

        assert!(std::mem::size_of_val(&server) > 0);
    }

    #[tokio::test]
    async fn test_server_list_tools() {
        let tool = Tool::new("test", "Test tool");
        let server = McpServer::builder()
            .register_tool(tool, EchoHandler)
            .build();

        let request = r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#;
        let response_str = server.process_request(request).await;

        let response: JsonRpcResponse = serde_json::from_str(&response_str).unwrap();
        assert!(response.error.is_none());
        assert!(response.result.is_some());
    }

    #[tokio::test]
    async fn test_server_tool_call() {
        let tool = Tool::new("echo", "Echo tool");
        let server = McpServer::builder()
            .register_tool(tool, EchoHandler)
            .build();

        let request = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"echo","arguments":{"message":"hello"}}}"#;
        let response_str = server.process_request(request).await;

        let response: JsonRpcResponse = serde_json::from_str(&response_str).unwrap();
        assert!(response.error.is_none());
        assert!(response.result.is_some());
    }

    #[tokio::test]
    async fn test_server_method_not_found() {
        let server = McpServer::builder().build();

        let request = r#"{"jsonrpc":"2.0","id":1,"method":"unknown"}"#;
        let response_str = server.process_request(request).await;

        let response: JsonRpcResponse = serde_json::from_str(&response_str).unwrap();
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32601);
    }

    #[tokio::test]
    async fn test_server_parse_error() {
        let server = McpServer::builder().build();

        let request = r#"{invalid json}"#;
        let response_str = server.process_request(request).await;

        let response: JsonRpcResponse = serde_json::from_str(&response_str).unwrap();
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32700);
    }

    #[tokio::test]
    async fn test_server_tool_not_found() {
        let server = McpServer::builder().build();

        let request =
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"nonexistent"}}"#;
        let response_str = server.process_request(request).await;

        let response: JsonRpcResponse = serde_json::from_str(&response_str).unwrap();
        assert!(response.error.is_some());
    }

    #[tokio::test]
    async fn test_server_missing_params() {
        let server = McpServer::builder().build();

        let request = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call"}"#;
        let response_str = server.process_request(request).await;

        let response: JsonRpcResponse = serde_json::from_str(&response_str).unwrap();
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32602);
    }

    #[tokio::test]
    async fn test_server_builder_default() {
        let builder = ServerBuilder::default();
        let server = builder.build();
        assert!(std::mem::size_of_val(&server) > 0);
    }
}
