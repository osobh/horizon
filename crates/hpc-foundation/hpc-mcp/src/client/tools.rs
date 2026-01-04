use crate::client::connection::Connection;
use crate::error::Result;
use crate::protocol::{Tool, ToolCall, ToolResult};
use serde_json::Value;

/// Client for invoking MCP tools
pub struct ToolClient {
    connection: Connection,
}

impl ToolClient {
    pub fn new() -> Self {
        Self {
            connection: Connection::new(),
        }
    }

    /// List available tools
    pub fn list_tools_request(&self) -> String {
        let request = self.connection.create_request("tools/list");
        self.connection.encode_request(&request).unwrap_or_default()
    }

    /// Parse list tools response
    pub fn parse_tools_response(&self, response_data: &str) -> Result<Vec<Tool>> {
        let response = self.connection.decode_response(response_data)?;

        if let Some(error) = response.error {
            return Err(crate::error::McpError::protocol(format!(
                "Error listing tools: {}",
                error.message
            )));
        }

        let result = response.result.unwrap_or(Value::Null);
        let tools = result
            .get("tools")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        Ok(tools)
    }

    /// Create a tool call request
    pub fn call_tool_request(&self, call: &ToolCall) -> Result<String> {
        let request = self
            .connection
            .create_request("tools/call")
            .with_params(serde_json::to_value(call)?);

        self.connection.encode_request(&request)
    }

    /// Parse tool call response
    pub fn parse_tool_response(&self, response_data: &str) -> Result<ToolResult> {
        let response = self.connection.decode_response(response_data)?;

        if let Some(error) = response.error {
            return Err(crate::error::McpError::protocol(format!(
                "Tool call error: {}",
                error.message
            )));
        }

        let result = response.result.unwrap_or(Value::Null);
        serde_json::from_value(result).map_err(|e| e.into())
    }
}

impl Default for ToolClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Content;

    #[test]
    fn test_tool_client_new() {
        let client = ToolClient::new();
        assert!(std::mem::size_of_val(&client) > 0);
    }

    #[test]
    fn test_tool_client_default() {
        let client = ToolClient::default();
        assert!(std::mem::size_of_val(&client) > 0);
    }

    #[test]
    fn test_list_tools_request() {
        let client = ToolClient::new();
        let request = client.list_tools_request();

        assert!(request.contains("tools/list"));
        assert!(request.contains("2.0"));
    }

    #[test]
    fn test_parse_tools_response() {
        let client = ToolClient::new();
        let response = r#"{"jsonrpc":"2.0","id":1,"result":{"tools":[{"name":"test","description":"A test tool"}]}}"#;

        let tools = client.parse_tools_response(response).unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "test");
        assert_eq!(tools[0].description, "A test tool");
    }

    #[test]
    fn test_parse_tools_response_empty() {
        let client = ToolClient::new();
        let response = r#"{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}"#;

        let tools = client.parse_tools_response(response).unwrap();
        assert_eq!(tools.len(), 0);
    }

    #[test]
    fn test_parse_tools_response_error() {
        let client = ToolClient::new();
        let response =
            r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"Method not found"}}"#;

        let result = client.parse_tools_response(response);
        assert!(result.is_err());
    }

    #[test]
    fn test_call_tool_request() {
        let client = ToolClient::new();
        let call = ToolCall::new("test_tool").with_arguments(serde_json::json!({"arg": "value"}));

        let request = client.call_tool_request(&call).unwrap();
        assert!(request.contains("tools/call"));
        assert!(request.contains("test_tool"));
    }

    #[test]
    fn test_parse_tool_response_success() {
        let client = ToolClient::new();
        let response = r#"{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"success"}],"is_error":false}}"#;

        let result = client.parse_tool_response(response).unwrap();
        assert!(!result.is_error);
        assert!(result.content.is_some());
    }

    #[test]
    fn test_parse_tool_response_error() {
        let client = ToolClient::new();
        let response =
            r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32000,"message":"Tool failed"}}"#;

        let result = client.parse_tool_response(response);
        assert!(result.is_err());
    }

    #[test]
    fn test_call_tool_with_no_arguments() {
        let client = ToolClient::new();
        let call = ToolCall::new("simple_tool");

        let request = client.call_tool_request(&call).unwrap();
        assert!(request.contains("simple_tool"));
    }

    #[test]
    fn test_parse_tool_response_with_image() {
        let client = ToolClient::new();
        let response = r#"{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"image","data":"base64data","mime_type":"image/png"}],"is_error":false}}"#;

        let result = client.parse_tool_response(response).unwrap();
        assert!(!result.is_error);

        if let Some(content) = result.content {
            assert_eq!(content.len(), 1);
            match &content[0] {
                Content::Image { data, mime_type } => {
                    assert_eq!(data, "base64data");
                    assert_eq!(mime_type, "image/png");
                }
                _ => panic!("Expected image content"),
            }
        }
    }
}
