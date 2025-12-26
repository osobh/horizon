use crate::error::{McpError, Result};
use crate::protocol::{JsonRpcCodec, JsonRpcRequest, JsonRpcResponse, RequestId};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// MCP Client Connection
pub struct Connection {
    next_id: Arc<AtomicU64>,
    codec: JsonRpcCodec,
}

impl Connection {
    pub fn new() -> Self {
        Self {
            next_id: Arc::new(AtomicU64::new(1)),
            codec: JsonRpcCodec::new(),
        }
    }

    /// Generate a unique request ID
    pub fn next_id(&self) -> RequestId {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        RequestId::Number(id)
    }

    /// Create a request with auto-generated ID
    pub fn create_request(&self, method: impl Into<String>) -> JsonRpcRequest {
        JsonRpcRequest::new(method).with_id(self.next_id())
    }

    /// Encode a request to JSON string
    pub fn encode_request(&self, request: &JsonRpcRequest) -> Result<String> {
        self.codec.encode_request(request)
    }

    /// Decode a response from JSON string
    pub fn decode_response(&self, data: &str) -> Result<JsonRpcResponse> {
        self.codec.decode_response(data)
    }

    /// Send a request and parse response (for testing/sync scenarios)
    pub fn send_request(
        &self,
        _request: &JsonRpcRequest,
        response_data: &str,
    ) -> Result<JsonRpcResponse> {
        let response = self.decode_response(response_data)?;

        if let Some(ref error) = response.error {
            return Err(McpError::protocol(format!(
                "JSON-RPC error {}: {}",
                error.code, error.message
            )));
        }

        Ok(response)
    }
}

impl Default for Connection {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::JsonRpcError;
    use serde_json::json;

    #[test]
    fn test_connection_new() {
        let conn = Connection::new();
        assert!(std::mem::size_of_val(&conn) > 0);
    }

    #[test]
    fn test_connection_default() {
        let conn = Connection::default();
        assert!(std::mem::size_of_val(&conn) > 0);
    }

    #[test]
    fn test_next_id_increments() {
        let conn = Connection::new();
        let id1 = conn.next_id();
        let id2 = conn.next_id();
        let id3 = conn.next_id();

        match (id1, id2, id3) {
            (
                RequestId::Number(n1),
                RequestId::Number(n2),
                RequestId::Number(n3),
            ) => {
                assert_eq!(n1, 1);
                assert_eq!(n2, 2);
                assert_eq!(n3, 3);
            }
            _ => panic!("Expected number IDs"),
        }
    }

    #[test]
    fn test_create_request() {
        let conn = Connection::new();
        let req = conn.create_request("test_method");

        assert_eq!(req.method, "test_method");
        assert_eq!(req.jsonrpc, "2.0");
        assert!(req.id.is_some());
    }

    #[test]
    fn test_encode_request() {
        let conn = Connection::new();
        let req = conn.create_request("test");

        let encoded = conn.encode_request(&req).unwrap();
        assert!(encoded.contains("test"));
        assert!(encoded.contains("2.0"));
    }

    #[test]
    fn test_decode_response_success() {
        let conn = Connection::new();
        let json = r#"{"jsonrpc":"2.0","id":1,"result":{"value":"ok"}}"#;

        let response = conn.decode_response(json).unwrap();
        assert!(response.result.is_some());
        assert!(response.error.is_none());
    }

    #[test]
    fn test_decode_response_error() {
        let conn = Connection::new();
        let json = r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"Method not found"}}"#;

        let response = conn.decode_response(json).unwrap();
        assert!(response.result.is_none());
        assert!(response.error.is_some());
    }

    #[test]
    fn test_send_request_success() {
        let conn = Connection::new();
        let req = conn.create_request("test");
        let response_json = r#"{"jsonrpc":"2.0","id":1,"result":{"ok":true}}"#;

        let response = conn.send_request(&req, response_json).unwrap();
        assert!(response.result.is_some());
    }

    #[test]
    fn test_send_request_error() {
        let conn = Connection::new();
        let req = conn.create_request("test");
        let response_json = r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"Not found"}}"#;

        let result = conn.send_request(&req, response_json);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), McpError::Protocol(_)));
    }

    #[test]
    fn test_send_request_malformed_json() {
        let conn = Connection::new();
        let req = conn.create_request("test");
        let bad_json = "{invalid}";

        let result = conn.send_request(&req, bad_json);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_requests_different_ids() {
        let conn = Connection::new();
        let req1 = conn.create_request("method1");
        let req2 = conn.create_request("method2");

        assert_ne!(req1.id, req2.id);
    }
}
