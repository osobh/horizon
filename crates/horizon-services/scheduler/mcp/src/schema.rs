use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolName {
    SubmitJob,
    ListQueue,
    GetJob,
    CancelJob,
}

pub fn create_tool_schemas() -> Vec<serde_json::Value> {
    vec![
        json!({
            "name": "submit_job",
            "description": "Submit a new job to the scheduler",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Unique job identifier"},
                    "priority": {"type": "integer", "description": "Job priority (0-100)"},
                    "resources": {
                        "type": "object",
                        "properties": {
                            "cpu": {"type": "integer"},
                            "memory": {"type": "integer"}
                        }
                    }
                },
                "required": ["job_id", "priority"]
            }
        }),
        json!({
            "name": "list_queue",
            "description": "List jobs in the queue",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max number of jobs to return"}
                }
            }
        }),
        json!({
            "name": "get_job",
            "description": "Get details of a specific job",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job identifier"}
                },
                "required": ["job_id"]
            }
        }),
        json!({
            "name": "cancel_job",
            "description": "Cancel a running or queued job",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job identifier"}
                },
                "required": ["job_id"]
            }
        }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_tool_schemas() {
        let schemas = create_tool_schemas();
        assert_eq!(schemas.len(), 4);
    }

    #[test]
    fn test_tool_name_serialization() {
        let name = ToolName::SubmitJob;
        let json = serde_json::to_string(&name).unwrap();
        assert_eq!(json, "\"submit_job\"");
    }

    #[test]
    fn test_schema_structure() {
        let schemas = create_tool_schemas();
        for schema in schemas {
            assert!(schema["name"].is_string());
            assert!(schema["description"].is_string());
            assert!(schema["parameters"].is_object());
        }
    }
}
