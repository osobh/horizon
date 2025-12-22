use anyhow::Result;
use serde_json::{json, Value};

// Mock job operations for testing
// In production, these would call actual scheduler APIs

pub async fn submit_job(params: Value) -> Result<Value> {
    let job_id = params["job_id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing job_id"))?;

    let priority = params["priority"]
        .as_i64()
        .ok_or_else(|| anyhow::anyhow!("Missing priority"))?;

    Ok(json!({
        "success": true,
        "job_id": job_id,
        "priority": priority,
        "status": "submitted"
    }))
}

pub async fn list_queue(params: Value) -> Result<Value> {
    let limit = params["limit"].as_i64().unwrap_or(10);

    // Mock queue data
    let jobs = (0..limit.min(5))
        .map(|i| {
            json!({
                "job_id": format!("job_{}", i),
                "priority": i * 10,
                "status": "queued"
            })
        })
        .collect::<Vec<_>>();

    Ok(json!({
        "jobs": jobs,
        "count": jobs.len()
    }))
}

pub async fn get_job(params: Value) -> Result<Value> {
    let job_id = params["job_id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing job_id"))?;

    Ok(json!({
        "job_id": job_id,
        "priority": 50,
        "status": "running",
        "progress": 0.75
    }))
}

pub async fn cancel_job(params: Value) -> Result<Value> {
    let job_id = params["job_id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing job_id"))?;

    Ok(json!({
        "success": true,
        "job_id": job_id,
        "status": "cancelled"
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_submit_job() {
        let params = json!({
            "job_id": "test_job_1",
            "priority": 100
        });

        let result = submit_job(params).await.unwrap();
        assert_eq!(result["job_id"], "test_job_1");
        assert_eq!(result["priority"], 100);
        assert_eq!(result["status"], "submitted");
    }

    #[tokio::test]
    async fn test_submit_job_missing_params() {
        let params = json!({});
        let result = submit_job(params).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_list_queue() {
        let params = json!({"limit": 3});
        let result = list_queue(params).await.unwrap();

        assert!(result["jobs"].is_array());
        assert_eq!(result["jobs"].as_array().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn test_list_queue_default_limit() {
        let params = json!({});
        let result = list_queue(params).await.unwrap();

        assert!(result["jobs"].is_array());
    }

    #[tokio::test]
    async fn test_get_job() {
        let params = json!({"job_id": "job_123"});
        let result = get_job(params).await.unwrap();

        assert_eq!(result["job_id"], "job_123");
        assert!(result["status"].is_string());
    }

    #[tokio::test]
    async fn test_get_job_missing_id() {
        let params = json!({});
        let result = get_job(params).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_cancel_job() {
        let params = json!({"job_id": "job_456"});
        let result = cancel_job(params).await.unwrap();

        assert_eq!(result["job_id"], "job_456");
        assert_eq!(result["status"], "cancelled");
    }

    #[tokio::test]
    async fn test_cancel_job_missing_id() {
        let params = json!({});
        let result = cancel_job(params).await;
        assert!(result.is_err());
    }
}
