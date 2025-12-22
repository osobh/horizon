use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// JSON-RPC 2.0 Request
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<RequestId>,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl JsonRpcRequest {
    pub fn new(method: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: Some(RequestId::Number(1)),
            method: method.into(),
            params: None,
        }
    }

    pub fn with_id(mut self, id: RequestId) -> Self {
        self.id = Some(id);
        self
    }

    pub fn with_params(mut self, params: Value) -> Self {
        self.params = Some(params);
        self
    }

    pub fn notification(method: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: None,
            method: method.into(),
            params: None,
        }
    }
}

/// JSON-RPC 2.0 Response
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Option<RequestId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

impl JsonRpcResponse {
    pub fn success(id: RequestId, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: Some(id),
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Option<RequestId>, error: JsonRpcError) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(error),
        }
    }
}

/// JSON-RPC 2.0 Error
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcError {
    pub fn parse_error() -> Self {
        Self {
            code: -32700,
            message: "Parse error".to_string(),
            data: None,
        }
    }

    pub fn invalid_request() -> Self {
        Self {
            code: -32600,
            message: "Invalid Request".to_string(),
            data: None,
        }
    }

    pub fn method_not_found() -> Self {
        Self {
            code: -32601,
            message: "Method not found".to_string(),
            data: None,
        }
    }

    pub fn invalid_params(msg: impl Into<String>) -> Self {
        Self {
            code: -32602,
            message: msg.into(),
            data: None,
        }
    }

    pub fn internal_error() -> Self {
        Self {
            code: -32603,
            message: "Internal error".to_string(),
            data: None,
        }
    }

    pub fn custom(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }
}

/// Request/Response ID
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(untagged)]
pub enum RequestId {
    Number(u64),
    String(String),
}

/// MCP Tool Definition
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Tool {
    pub name: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<ToolSchema>,
}

impl Tool {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema: None,
        }
    }

    pub fn with_schema(mut self, schema: ToolSchema) -> Self {
        self.input_schema = Some(schema);
        self
    }
}

/// Tool Input Schema (JSON Schema)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolSchema {
    #[serde(rename = "type")]
    pub schema_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, PropertySchema>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
}

impl ToolSchema {
    pub fn object() -> Self {
        Self {
            schema_type: "object".to_string(),
            properties: Some(HashMap::new()),
            required: None,
        }
    }

    pub fn with_property(
        mut self,
        name: impl Into<String>,
        schema: PropertySchema,
    ) -> Self {
        self.properties
            .get_or_insert_with(HashMap::new)
            .insert(name.into(), schema);
        self
    }

    pub fn with_required(mut self, fields: Vec<String>) -> Self {
        self.required = Some(fields);
        self
    }
}

/// Property Schema
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PropertySchema {
    #[serde(rename = "type")]
    pub property_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl PropertySchema {
    pub fn string(description: impl Into<String>) -> Self {
        Self {
            property_type: "string".to_string(),
            description: Some(description.into()),
        }
    }

    pub fn number(description: impl Into<String>) -> Self {
        Self {
            property_type: "number".to_string(),
            description: Some(description.into()),
        }
    }

    pub fn boolean(description: impl Into<String>) -> Self {
        Self {
            property_type: "boolean".to_string(),
            description: Some(description.into()),
        }
    }
}

/// Tool Call
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Value>,
}

impl ToolCall {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            arguments: None,
        }
    }

    pub fn with_arguments(mut self, args: Value) -> Self {
        self.arguments = Some(args);
        self
    }
}

/// Tool Result
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<Content>>,
    #[serde(default)]
    pub is_error: bool,
}

impl ToolResult {
    pub fn success(content: Vec<Content>) -> Self {
        Self {
            content: Some(content),
            is_error: false,
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: Some(vec![Content::text(message)]),
            is_error: true,
        }
    }
}

/// Content Block
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum Content {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { data: String, mime_type: String },
}

impl Content {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    pub fn image(data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self::Image {
            data: data.into(),
            mime_type: mime_type.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jsonrpc_request_new() {
        let req = JsonRpcRequest::new("test_method");
        assert_eq!(req.jsonrpc, "2.0");
        assert_eq!(req.method, "test_method");
        assert!(req.id.is_some());
        assert!(req.params.is_none());
    }

    #[test]
    fn test_jsonrpc_request_with_params() {
        let req = JsonRpcRequest::new("test")
            .with_id(RequestId::String("123".to_string()))
            .with_params(serde_json::json!({"key": "value"}));

        assert_eq!(req.id, Some(RequestId::String("123".to_string())));
        assert!(req.params.is_some());
    }

    #[test]
    fn test_jsonrpc_notification() {
        let req = JsonRpcRequest::notification("notify");
        assert_eq!(req.jsonrpc, "2.0");
        assert_eq!(req.method, "notify");
        assert!(req.id.is_none());
    }

    #[test]
    fn test_jsonrpc_response_success() {
        let resp = JsonRpcResponse::success(
            RequestId::Number(1),
            serde_json::json!({"result": "ok"}),
        );

        assert_eq!(resp.jsonrpc, "2.0");
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_jsonrpc_response_error() {
        let resp = JsonRpcResponse::error(
            Some(RequestId::Number(1)),
            JsonRpcError::method_not_found(),
        );

        assert!(resp.result.is_none());
        assert!(resp.error.is_some());
    }

    #[test]
    fn test_jsonrpc_error_codes() {
        assert_eq!(JsonRpcError::parse_error().code, -32700);
        assert_eq!(JsonRpcError::invalid_request().code, -32600);
        assert_eq!(JsonRpcError::method_not_found().code, -32601);
        assert_eq!(JsonRpcError::invalid_params("test").code, -32602);
        assert_eq!(JsonRpcError::internal_error().code, -32603);
    }

    #[test]
    fn test_jsonrpc_error_custom() {
        let err = JsonRpcError::custom(100, "custom error");
        assert_eq!(err.code, 100);
        assert_eq!(err.message, "custom error");
    }

    #[test]
    fn test_tool_new() {
        let tool = Tool::new("my_tool", "A test tool");
        assert_eq!(tool.name, "my_tool");
        assert_eq!(tool.description, "A test tool");
        assert!(tool.input_schema.is_none());
    }

    #[test]
    fn test_tool_with_schema() {
        let schema = ToolSchema::object()
            .with_property("arg1", PropertySchema::string("First argument"))
            .with_required(vec!["arg1".to_string()]);

        let tool = Tool::new("my_tool", "A test tool").with_schema(schema);

        assert!(tool.input_schema.is_some());
        let schema = tool.input_schema.unwrap();
        assert_eq!(schema.schema_type, "object");
        assert!(schema.properties.is_some());
        assert_eq!(schema.required, Some(vec!["arg1".to_string()]));
    }

    #[test]
    fn test_property_schema_types() {
        let str_prop = PropertySchema::string("A string");
        assert_eq!(str_prop.property_type, "string");

        let num_prop = PropertySchema::number("A number");
        assert_eq!(num_prop.property_type, "number");

        let bool_prop = PropertySchema::boolean("A boolean");
        assert_eq!(bool_prop.property_type, "boolean");
    }

    #[test]
    fn test_tool_call() {
        let call = ToolCall::new("test")
            .with_arguments(serde_json::json!({"arg": "value"}));

        assert_eq!(call.name, "test");
        assert!(call.arguments.is_some());
    }

    #[test]
    fn test_tool_result_success() {
        let result = ToolResult::success(vec![Content::text("success")]);
        assert!(!result.is_error);
        assert!(result.content.is_some());
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResult::error("failed");
        assert!(result.is_error);
        assert!(result.content.is_some());
    }

    #[test]
    fn test_content_text() {
        let content = Content::text("hello");
        match content {
            Content::Text { text } => assert_eq!(text, "hello"),
            _ => panic!("Expected text content"),
        }
    }

    #[test]
    fn test_content_image() {
        let content = Content::image("base64data", "image/png");
        match content {
            Content::Image { data, mime_type } => {
                assert_eq!(data, "base64data");
                assert_eq!(mime_type, "image/png");
            }
            _ => panic!("Expected image content"),
        }
    }

    #[test]
    fn test_jsonrpc_request_serialization() {
        let req = JsonRpcRequest::new("test");
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: JsonRpcRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, deserialized);
    }

    #[test]
    fn test_jsonrpc_response_serialization() {
        let resp = JsonRpcResponse::success(
            RequestId::Number(1),
            serde_json::json!({"ok": true}),
        );
        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: JsonRpcResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp, deserialized);
    }

    #[test]
    fn test_tool_serialization() {
        let tool = Tool::new("test", "description");
        let json = serde_json::to_string(&tool).unwrap();
        let deserialized: Tool = serde_json::from_str(&json).unwrap();
        assert_eq!(tool, deserialized);
    }
}
