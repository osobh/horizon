use crate::error::{McpError, Result};
use crate::protocol::{Tool, ToolCall, ToolResult};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

/// Tool handler trait
#[async_trait]
pub trait ToolHandler: Send + Sync {
    async fn handle(&self, call: ToolCall) -> Result<ToolResult>;
}

/// Type alias for boxed tool handler
type BoxedHandler = Arc<dyn ToolHandler>;

/// Router for dispatching tool calls to handlers
pub struct Router {
    tools: HashMap<String, Tool>,
    handlers: HashMap<String, BoxedHandler>,
}

impl Router {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            handlers: HashMap::new(),
        }
    }

    /// Register a tool with its handler
    pub fn register<H: ToolHandler + 'static>(
        mut self,
        tool: Tool,
        handler: H,
    ) -> Self {
        let name = tool.name.clone();
        self.tools.insert(name.clone(), tool);
        self.handlers.insert(name, Arc::new(handler));
        self
    }

    /// Get all registered tools
    pub fn list_tools(&self) -> Vec<Tool> {
        self.tools.values().cloned().collect()
    }

    /// Get a specific tool by name
    pub fn get_tool(&self, name: &str) -> Option<&Tool> {
        self.tools.get(name)
    }

    /// Route a tool call to its handler
    pub async fn route(&self, call: ToolCall) -> Result<ToolResult> {
        let handler = self
            .handlers
            .get(&call.name)
            .ok_or_else(|| McpError::tool_not_found(&call.name))?;

        handler.handle(call).await
    }

    /// Check if a tool is registered
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Content;

    struct TestHandler {
        response: String,
    }

    #[async_trait]
    impl ToolHandler for TestHandler {
        async fn handle(&self, _call: ToolCall) -> Result<ToolResult> {
            Ok(ToolResult::success(vec![Content::text(&self.response)]))
        }
    }

    struct ErrorHandler;

    #[async_trait]
    impl ToolHandler for ErrorHandler {
        async fn handle(&self, _call: ToolCall) -> Result<ToolResult> {
            Err(McpError::internal("handler error"))
        }
    }

    #[tokio::test]
    async fn test_router_new() {
        let router = Router::new();
        assert_eq!(router.list_tools().len(), 0);
    }

    #[tokio::test]
    async fn test_router_register_tool() {
        let tool = Tool::new("test_tool", "A test tool");
        let handler = TestHandler {
            response: "ok".to_string(),
        };

        let router = Router::new().register(tool.clone(), handler);

        assert!(router.has_tool("test_tool"));
        assert_eq!(router.list_tools().len(), 1);
        assert_eq!(router.get_tool("test_tool").unwrap().name, "test_tool");
    }

    #[tokio::test]
    async fn test_router_route_success() {
        let tool = Tool::new("test_tool", "A test tool");
        let handler = TestHandler {
            response: "success".to_string(),
        };

        let router = Router::new().register(tool, handler);
        let call = ToolCall::new("test_tool");

        let result = router.route(call).await.unwrap();
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn test_router_route_not_found() {
        let router = Router::new();
        let call = ToolCall::new("unknown_tool");

        let result = router.route(call).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), McpError::ToolNotFound(_)));
    }

    #[tokio::test]
    async fn test_router_multiple_tools() {
        let tool1 = Tool::new("tool1", "First tool");
        let tool2 = Tool::new("tool2", "Second tool");

        let handler1 = TestHandler {
            response: "one".to_string(),
        };
        let handler2 = TestHandler {
            response: "two".to_string(),
        };

        let router = Router::new()
            .register(tool1, handler1)
            .register(tool2, handler2);

        assert_eq!(router.list_tools().len(), 2);
        assert!(router.has_tool("tool1"));
        assert!(router.has_tool("tool2"));
        assert!(!router.has_tool("tool3"));
    }

    #[tokio::test]
    async fn test_router_handler_error() {
        let tool = Tool::new("error_tool", "Error tool");
        let handler = ErrorHandler;

        let router = Router::new().register(tool, handler);
        let call = ToolCall::new("error_tool");

        let result = router.route(call).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), McpError::Internal(_)));
    }

    #[tokio::test]
    async fn test_router_get_tool() {
        let tool = Tool::new("test", "description");
        let handler = TestHandler {
            response: "ok".to_string(),
        };

        let router = Router::new().register(tool, handler);

        assert!(router.get_tool("test").is_some());
        assert!(router.get_tool("nonexistent").is_none());
    }

    #[tokio::test]
    async fn test_router_default() {
        let router = Router::default();
        assert_eq!(router.list_tools().len(), 0);
    }
}
