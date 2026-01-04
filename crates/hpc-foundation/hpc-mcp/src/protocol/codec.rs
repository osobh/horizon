use crate::error::{McpError, Result};
use crate::protocol::types::{JsonRpcRequest, JsonRpcResponse};
use serde::{Deserialize, Serialize};

/// JSON-RPC Codec for encoding/decoding messages
pub struct JsonRpcCodec;

impl JsonRpcCodec {
    pub fn new() -> Self {
        Self
    }

    /// Encode a request to JSON
    pub fn encode_request(&self, request: &JsonRpcRequest) -> Result<String> {
        serde_json::to_string(request).map_err(McpError::from)
    }

    /// Decode a request from JSON
    pub fn decode_request(&self, data: &str) -> Result<JsonRpcRequest> {
        serde_json::from_str(data).map_err(McpError::from)
    }

    /// Encode a response to JSON
    pub fn encode_response(&self, response: &JsonRpcResponse) -> Result<String> {
        serde_json::to_string(response).map_err(McpError::from)
    }

    /// Decode a response from JSON
    pub fn decode_response(&self, data: &str) -> Result<JsonRpcResponse> {
        serde_json::from_str(data).map_err(McpError::from)
    }

    /// Encode any serializable value to JSON
    pub fn encode<T: Serialize>(&self, value: &T) -> Result<String> {
        serde_json::to_string(value).map_err(McpError::from)
    }

    /// Decode JSON to any deserializable type
    pub fn decode<'a, T: Deserialize<'a>>(&self, data: &'a str) -> Result<T> {
        serde_json::from_str(data).map_err(McpError::from)
    }
}

impl Default for JsonRpcCodec {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::types::{JsonRpcError, RequestId};
    use serde_json::{json, Value};

    #[test]
    fn test_codec_new() {
        let codec = JsonRpcCodec::new();
        assert!(std::mem::size_of_val(&codec) == 0);
    }

    #[test]
    fn test_codec_default() {
        let codec = JsonRpcCodec::default();
        assert!(std::mem::size_of_val(&codec) == 0);
    }

    #[test]
    fn test_encode_decode_request() {
        let codec = JsonRpcCodec::new();
        let request = JsonRpcRequest::new("test_method")
            .with_id(RequestId::Number(1))
            .with_params(json!({"key": "value"}));

        let encoded = codec.encode_request(&request).unwrap();
        let decoded = codec.decode_request(&encoded).unwrap();

        assert_eq!(request, decoded);
    }

    #[test]
    fn test_encode_decode_response_success() {
        let codec = JsonRpcCodec::new();
        let response = JsonRpcResponse::success(RequestId::Number(1), json!({"result": "ok"}));

        let encoded = codec.encode_response(&response).unwrap();
        let decoded = codec.decode_response(&encoded).unwrap();

        assert_eq!(response, decoded);
    }

    #[test]
    fn test_encode_decode_response_error() {
        let codec = JsonRpcCodec::new();
        let response =
            JsonRpcResponse::error(Some(RequestId::Number(1)), JsonRpcError::method_not_found());

        let encoded = codec.encode_response(&response).unwrap();
        let decoded = codec.decode_response(&encoded).unwrap();

        assert_eq!(response, decoded);
    }

    #[test]
    fn test_decode_invalid_json() {
        let codec = JsonRpcCodec::new();
        let result = codec.decode_request("{invalid json");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), McpError::Serialization(_)));
    }

    #[test]
    fn test_encode_generic() {
        let codec = JsonRpcCodec::new();
        let data = json!({"test": "value"});
        let encoded = codec.encode(&data).unwrap();
        assert!(encoded.contains("test"));
        assert!(encoded.contains("value"));
    }

    #[test]
    fn test_decode_generic() {
        let codec = JsonRpcCodec::new();
        let json_str = r#"{"test": "value"}"#;
        let decoded: Value = codec.decode(json_str).unwrap();
        assert_eq!(decoded["test"], "value");
    }

    #[test]
    fn test_request_roundtrip_with_null_params() {
        let codec = JsonRpcCodec::new();
        let request = JsonRpcRequest::new("test");

        let encoded = codec.encode_request(&request).unwrap();
        let decoded = codec.decode_request(&encoded).unwrap();

        assert_eq!(request.method, decoded.method);
        assert_eq!(request.jsonrpc, decoded.jsonrpc);
    }

    #[test]
    fn test_response_roundtrip_with_string_id() {
        let codec = JsonRpcCodec::new();
        let response = JsonRpcResponse::success(
            RequestId::String("abc".to_string()),
            json!({"status": "ok"}),
        );

        let encoded = codec.encode_response(&response).unwrap();
        let decoded = codec.decode_response(&encoded).unwrap();

        assert!(decoded.error.is_none());
        assert!(decoded.result.is_some());
        assert_eq!(decoded.id, Some(RequestId::String("abc".to_string())));
    }
}
