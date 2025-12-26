use crate::schema::create_tool_schemas;
use crate::tools;
use anyhow::Result;
use serde_json::Value;

pub struct SchedulerMcpServer {
    tools: Vec<Value>,
}

impl SchedulerMcpServer {
    pub fn new() -> Self {
        Self {
            tools: create_tool_schemas(),
        }
    }

    pub fn list_tools(&self) -> &[Value] {
        &self.tools
    }

    pub async fn call_tool(&self, name: &str, params: Value) -> Result<Value> {
        match name {
            "submit_job" => tools::submit_job(params).await,
            "list_queue" => tools::list_queue(params).await,
            "get_job" => tools::get_job(params).await,
            "cancel_job" => tools::cancel_job(params).await,
            _ => Err(anyhow::anyhow!("Unknown tool: {}", name)),
        }
    }
}

impl Default for SchedulerMcpServer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_server_new() {
        let server = SchedulerMcpServer::new();
        assert_eq!(server.list_tools().len(), 4);
    }

    #[test]
    fn test_server_default() {
        let server = SchedulerMcpServer::default();
        assert_eq!(server.list_tools().len(), 4);
    }

    #[tokio::test]
    async fn test_call_submit_job() {
        let server = SchedulerMcpServer::new();
        let params = json!({"job_id": "test", "priority": 100});

        let result = server.call_tool("submit_job", params).await.unwrap();
        assert_eq!(result["job_id"], "test");
    }

    #[tokio::test]
    async fn test_call_list_queue() {
        let server = SchedulerMcpServer::new();
        let params = json!({"limit": 2});

        let result = server.call_tool("list_queue", params).await.unwrap();
        assert!(result["jobs"].is_array());
    }

    #[tokio::test]
    async fn test_call_unknown_tool() {
        let server = SchedulerMcpServer::new();
        let params = json!({});

        let result = server.call_tool("unknown_tool", params).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_all_tools_callable() {
        let server = SchedulerMcpServer::new();

        // Submit job
        let result = server
            .call_tool("submit_job", json!({"job_id": "j1", "priority": 50}))
            .await;
        assert!(result.is_ok());

        // List queue
        let result = server.call_tool("list_queue", json!({})).await;
        assert!(result.is_ok());

        // Get job
        let result = server.call_tool("get_job", json!({"job_id": "j1"})).await;
        assert!(result.is_ok());

        // Cancel job
        let result = server.call_tool("cancel_job", json!({"job_id": "j1"})).await;
        assert!(result.is_ok());
    }
}
