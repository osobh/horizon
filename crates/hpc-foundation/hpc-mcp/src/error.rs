use thiserror::Error;

pub type Result<T> = std::result::Result<T, McpError>;

#[derive(Debug, Error)]
pub enum McpError {
    #[error("Protocol error: {0}")]
    Protocol(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Transport error: {0}")]
    Transport(String),

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    #[error("Method not found: {0}")]
    MethodNotFound(String),

    #[error("Invalid params: {0}")]
    InvalidParams(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl McpError {
    pub fn protocol(msg: impl Into<String>) -> Self {
        Self::Protocol(msg.into())
    }

    pub fn transport(msg: impl Into<String>) -> Self {
        Self::Transport(msg.into())
    }

    pub fn tool_not_found(name: impl Into<String>) -> Self {
        Self::ToolNotFound(name.into())
    }

    pub fn invalid_request(msg: impl Into<String>) -> Self {
        Self::InvalidRequest(msg.into())
    }

    pub fn invalid_response(msg: impl Into<String>) -> Self {
        Self::InvalidResponse(msg.into())
    }

    pub fn method_not_found(method: impl Into<String>) -> Self {
        Self::MethodNotFound(method.into())
    }

    pub fn invalid_params(msg: impl Into<String>) -> Self {
        Self::InvalidParams(msg.into())
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_error() {
        let err = McpError::protocol("test error");
        assert!(err.to_string().contains("Protocol error"));
        assert!(err.to_string().contains("test error"));
    }

    #[test]
    fn test_transport_error() {
        let err = McpError::transport("connection failed");
        assert!(err.to_string().contains("Transport error"));
        assert!(err.to_string().contains("connection failed"));
    }

    #[test]
    fn test_tool_not_found() {
        let err = McpError::tool_not_found("my_tool");
        assert!(err.to_string().contains("Tool not found"));
        assert!(err.to_string().contains("my_tool"));
    }

    #[test]
    fn test_invalid_request() {
        let err = McpError::invalid_request("bad format");
        assert!(err.to_string().contains("Invalid request"));
    }

    #[test]
    fn test_method_not_found() {
        let err = McpError::method_not_found("unknown_method");
        assert!(err.to_string().contains("Method not found"));
        assert!(err.to_string().contains("unknown_method"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: McpError = io_err.into();
        assert!(matches!(err, McpError::Io(_)));
    }

    #[test]
    fn test_serialization_error_conversion() {
        let json_err = serde_json::from_str::<serde_json::Value>("{invalid")
            .expect_err("should fail");
        let err: McpError = json_err.into();
        assert!(matches!(err, McpError::Serialization(_)));
    }
}
