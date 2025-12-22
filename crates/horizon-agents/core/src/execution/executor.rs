use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::error::{AgentError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: HashMap<String, String>,
}

impl Tool {
    pub fn new(name: String, description: String) -> Self {
        Self {
            name,
            description,
            parameters: HashMap::new(),
        }
    }

    pub fn with_parameter(mut self, key: String, value: String) -> Self {
        self.parameters.insert(key, value);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRequest {
    pub id: Uuid,
    pub tool: Tool,
    pub context: HashMap<String, String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl ExecutionRequest {
    pub fn new(tool: Tool) -> Self {
        Self {
            id: Uuid::new_v4(),
            tool,
            context: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn with_context(mut self, key: String, value: String) -> Self {
        self.context.insert(key, value);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub request_id: Uuid,
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
    pub metadata: HashMap<String, String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl ExecutionResult {
    pub fn success(request_id: Uuid, output: String) -> Self {
        Self {
            request_id,
            success: true,
            output,
            error: None,
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn failure(request_id: Uuid, error: String) -> Self {
        Self {
            request_id,
            success: false,
            output: String::new(),
            error: Some(error),
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

pub struct Executor {
    tools: HashMap<String, Tool>,
}

impl Executor {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register_tool(&mut self, tool: Tool) -> Result<()> {
        if self.tools.contains_key(&tool.name) {
            return Err(AgentError::ExecutionFailed(format!(
                "Tool {} already registered",
                tool.name
            )));
        }
        self.tools.insert(tool.name.clone(), tool);
        Ok(())
    }

    pub fn get_tool(&self, name: &str) -> Result<&Tool> {
        self.tools
            .get(name)
            .ok_or_else(|| AgentError::ToolNotFound(name.to_string()))
    }

    pub fn list_tools(&self) -> Vec<&Tool> {
        self.tools.values().collect()
    }

    pub async fn execute(&self, request: ExecutionRequest) -> Result<ExecutionResult> {
        // Verify tool exists
        self.get_tool(&request.tool.name)?;

        // In a real implementation, this would execute the tool
        // For now, we return a success result
        Ok(ExecutionResult::success(
            request.id,
            format!("Executed tool: {}", request.tool.name),
        ))
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_creation() {
        let tool = Tool::new("test-tool".to_string(), "A test tool".to_string());
        assert_eq!(tool.name, "test-tool");
        assert_eq!(tool.description, "A test tool");
        assert!(tool.parameters.is_empty());
    }

    #[test]
    fn test_tool_with_parameters() {
        let tool = Tool::new("test-tool".to_string(), "A test tool".to_string())
            .with_parameter("param1".to_string(), "value1".to_string())
            .with_parameter("param2".to_string(), "value2".to_string());

        assert_eq!(tool.parameters.len(), 2);
        assert_eq!(tool.parameters.get("param1").unwrap(), "value1");
    }

    #[test]
    fn test_execution_request_creation() {
        let tool = Tool::new("test-tool".to_string(), "A test tool".to_string());
        let req = ExecutionRequest::new(tool);
        assert_eq!(req.tool.name, "test-tool");
        assert!(req.context.is_empty());
    }

    #[test]
    fn test_execution_request_with_context() {
        let tool = Tool::new("test-tool".to_string(), "A test tool".to_string());
        let req = ExecutionRequest::new(tool)
            .with_context("key1".to_string(), "value1".to_string());

        assert_eq!(req.context.len(), 1);
        assert_eq!(req.context.get("key1").unwrap(), "value1");
    }

    #[test]
    fn test_execution_result_success() {
        let req_id = Uuid::new_v4();
        let result = ExecutionResult::success(req_id, "output".to_string());
        assert_eq!(result.request_id, req_id);
        assert!(result.success);
        assert_eq!(result.output, "output");
        assert!(result.error.is_none());
    }

    #[test]
    fn test_execution_result_failure() {
        let req_id = Uuid::new_v4();
        let result = ExecutionResult::failure(req_id, "error message".to_string());
        assert_eq!(result.request_id, req_id);
        assert!(!result.success);
        assert_eq!(result.error.as_ref().unwrap(), "error message");
    }

    #[test]
    fn test_execution_result_with_metadata() {
        let req_id = Uuid::new_v4();
        let result = ExecutionResult::success(req_id, "output".to_string())
            .with_metadata("key1".to_string(), "value1".to_string());

        assert_eq!(result.metadata.len(), 1);
        assert_eq!(result.metadata.get("key1").unwrap(), "value1");
    }

    #[test]
    fn test_executor_creation() {
        let executor = Executor::new();
        assert_eq!(executor.list_tools().len(), 0);
    }

    #[test]
    fn test_executor_register_tool() {
        let mut executor = Executor::new();
        let tool = Tool::new("test-tool".to_string(), "A test tool".to_string());

        assert!(executor.register_tool(tool).is_ok());
        assert_eq!(executor.list_tools().len(), 1);
    }

    #[test]
    fn test_executor_register_duplicate_tool() {
        let mut executor = Executor::new();
        let tool1 = Tool::new("test-tool".to_string(), "A test tool".to_string());
        let tool2 = Tool::new("test-tool".to_string(), "Another test tool".to_string());

        executor.register_tool(tool1).unwrap();
        let result = executor.register_tool(tool2);
        assert!(result.is_err());
    }

    #[test]
    fn test_executor_get_tool() {
        let mut executor = Executor::new();
        let tool = Tool::new("test-tool".to_string(), "A test tool".to_string());

        executor.register_tool(tool).unwrap();
        let retrieved = executor.get_tool("test-tool");
        assert!(retrieved.is_ok());
        assert_eq!(retrieved.unwrap().name, "test-tool");
    }

    #[test]
    fn test_executor_get_nonexistent_tool() {
        let executor = Executor::new();
        let result = executor.get_tool("nonexistent");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AgentError::ToolNotFound(_)));
    }

    #[tokio::test]
    async fn test_executor_execute_success() {
        let mut executor = Executor::new();
        let tool = Tool::new("test-tool".to_string(), "A test tool".to_string());
        executor.register_tool(tool.clone()).unwrap();

        let request = ExecutionRequest::new(tool);
        let result = executor.execute(request).await;
        assert!(result.is_ok());

        let execution_result = result.unwrap();
        assert!(execution_result.success);
    }

    #[tokio::test]
    async fn test_executor_execute_nonexistent_tool() {
        let executor = Executor::new();
        let tool = Tool::new("nonexistent".to_string(), "A test tool".to_string());
        let request = ExecutionRequest::new(tool);

        let result = executor.execute(request).await;
        assert!(result.is_err());
    }
}
