//! RustySpark Bridge for Horizon Integration
//!
//! Provides data processing job management via the RustySpark WebUI API:
//! - Job listing, details, and status management
//! - Stage and task monitoring
//! - Job metrics and progress tracking
//!
//! # Features
//!
//! - REST API integration with RustySpark WebUI
//! - Job lifecycle management (Spark-style)
//! - Stage/task hierarchy tracking
//! - Mock mode for development without live RustySpark server

use std::sync::Arc;
use tokio::sync::RwLock;

/// Job status (Spark-compatible)
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SparkJobStatus {
    /// Job is pending
    Pending,
    /// Job is running
    Running,
    /// Job completed successfully
    Completed,
    /// Job failed
    Failed,
    /// Job was cancelled
    Cancelled,
}

/// Stage status
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum StageStatus {
    /// Stage is pending
    Pending,
    /// Stage is running
    Running,
    /// Stage completed successfully
    Completed,
    /// Stage failed
    Failed,
    /// Stage was skipped
    Skipped,
}

/// Task status
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TaskStatus {
    /// Task is pending
    Pending,
    /// Task is running
    Running,
    /// Task completed successfully
    Completed,
    /// Task failed
    Failed,
    /// Task was killed
    Killed,
}

/// Spark job information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SparkJob {
    /// Job ID
    pub id: String,
    /// Job name
    pub name: String,
    /// Job status
    pub status: SparkJobStatus,
    /// Submission timestamp (ms since epoch)
    pub submitted_at_ms: u64,
    /// Start timestamp (ms since epoch)
    pub started_at_ms: Option<u64>,
    /// Completion timestamp (ms since epoch)
    pub completed_at_ms: Option<u64>,
    /// Duration in milliseconds
    pub duration_ms: Option<i64>,
    /// Total stages
    pub total_stages: i32,
    /// Total tasks
    pub total_tasks: i32,
    /// Failed tasks
    pub failed_tasks: i32,
}

/// Spark stage information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SparkStage {
    /// Stage ID
    pub id: i64,
    /// Job ID
    pub job_id: String,
    /// Stage name
    pub name: String,
    /// Stage status
    pub status: StageStatus,
    /// Total tasks in this stage
    pub total_tasks: i32,
    /// Completed tasks
    pub completed_tasks: i32,
    /// Failed tasks
    pub failed_tasks: i32,
    /// Shuffle read bytes
    pub shuffle_read_bytes: i64,
    /// Shuffle write bytes
    pub shuffle_write_bytes: i64,
    /// Input bytes
    pub input_bytes: i64,
    /// Output bytes
    pub output_bytes: i64,
    /// Duration in milliseconds
    pub duration_ms: Option<i64>,
}

/// Spark task information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SparkTask {
    /// Task ID
    pub id: i64,
    /// Stage ID
    pub stage_id: i64,
    /// Job ID
    pub job_id: String,
    /// Partition index
    pub partition_id: i32,
    /// Executor ID
    pub executor_id: String,
    /// Host where task ran
    pub host: String,
    /// Task status
    pub status: TaskStatus,
    /// Duration in milliseconds
    pub duration_ms: Option<i64>,
    /// GC time in milliseconds
    pub gc_time_ms: i64,
    /// Input bytes
    pub input_bytes: i64,
    /// Output bytes
    pub output_bytes: i64,
    /// Shuffle read bytes
    pub shuffle_read_bytes: i64,
    /// Shuffle write bytes
    pub shuffle_write_bytes: i64,
}

/// RustySpark summary statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SparkSummary {
    /// Total jobs
    pub total_jobs: i64,
    /// Running jobs
    pub running_jobs: i64,
    /// Completed jobs
    pub completed_jobs: i64,
    /// Failed jobs
    pub failed_jobs: i64,
    /// Total stages
    pub total_stages: i64,
    /// Total tasks
    pub total_tasks: i64,
    /// Total shuffle read bytes
    pub total_shuffle_read_bytes: i64,
    /// Total shuffle write bytes
    pub total_shuffle_write_bytes: i64,
}

/// RustySpark server status
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct RustySparkStatus {
    /// Whether connected to RustySpark WebUI
    pub connected: bool,
    /// Server URL
    pub server_url: String,
    /// Server version
    pub version: String,
    /// Summary statistics
    pub summary: Option<SparkSummary>,
}

/// RustySpark Bridge for Horizon integration.
pub struct RustySparkBridge {
    /// Server URL (configurable)
    server_url: Arc<RwLock<String>>,
    /// HTTP client for API requests
    #[cfg(feature = "rustyspark-live")]
    client: reqwest::Client,
    /// Mock state for development
    #[cfg(not(feature = "rustyspark-live"))]
    mock_state: Arc<RwLock<MockSparkState>>,
}

#[cfg(not(feature = "rustyspark-live"))]
struct MockSparkState {
    jobs: Vec<SparkJob>,
    stages: Vec<SparkStage>,
    tasks: Vec<SparkTask>,
    next_job_id: u64,
}

#[cfg(not(feature = "rustyspark-live"))]
impl Default for MockSparkState {
    fn default() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let jobs = vec![
            SparkJob {
                id: "job-001".to_string(),
                name: "ETL Pipeline - Sales Data".to_string(),
                status: SparkJobStatus::Running,
                submitted_at_ms: now - 60000,
                started_at_ms: Some(now - 55000),
                completed_at_ms: None,
                duration_ms: Some(55000),
                total_stages: 5,
                total_tasks: 200,
                failed_tasks: 0,
            },
            SparkJob {
                id: "job-002".to_string(),
                name: "Feature Engineering".to_string(),
                status: SparkJobStatus::Completed,
                submitted_at_ms: now - 300000,
                started_at_ms: Some(now - 295000),
                completed_at_ms: Some(now - 120000),
                duration_ms: Some(175000),
                total_stages: 8,
                total_tasks: 500,
                failed_tasks: 2,
            },
            SparkJob {
                id: "job-003".to_string(),
                name: "Model Training Data Prep".to_string(),
                status: SparkJobStatus::Failed,
                submitted_at_ms: now - 600000,
                started_at_ms: Some(now - 595000),
                completed_at_ms: Some(now - 500000),
                duration_ms: Some(95000),
                total_stages: 4,
                total_tasks: 150,
                failed_tasks: 45,
            },
            SparkJob {
                id: "job-004".to_string(),
                name: "Daily Aggregation".to_string(),
                status: SparkJobStatus::Pending,
                submitted_at_ms: now - 5000,
                started_at_ms: None,
                completed_at_ms: None,
                duration_ms: None,
                total_stages: 3,
                total_tasks: 0,
                failed_tasks: 0,
            },
        ];

        let stages = vec![
            SparkStage {
                id: 1,
                job_id: "job-001".to_string(),
                name: "Scan parquet files".to_string(),
                status: StageStatus::Completed,
                total_tasks: 50,
                completed_tasks: 50,
                failed_tasks: 0,
                shuffle_read_bytes: 0,
                shuffle_write_bytes: 1024 * 1024 * 512,
                input_bytes: 1024 * 1024 * 1024 * 2,
                output_bytes: 0,
                duration_ms: Some(15000),
            },
            SparkStage {
                id: 2,
                job_id: "job-001".to_string(),
                name: "Filter and transform".to_string(),
                status: StageStatus::Running,
                total_tasks: 50,
                completed_tasks: 35,
                failed_tasks: 0,
                shuffle_read_bytes: 1024 * 1024 * 512,
                shuffle_write_bytes: 1024 * 1024 * 256,
                input_bytes: 0,
                output_bytes: 0,
                duration_ms: Some(20000),
            },
            SparkStage {
                id: 3,
                job_id: "job-001".to_string(),
                name: "Aggregate".to_string(),
                status: StageStatus::Pending,
                total_tasks: 50,
                completed_tasks: 0,
                failed_tasks: 0,
                shuffle_read_bytes: 0,
                shuffle_write_bytes: 0,
                input_bytes: 0,
                output_bytes: 0,
                duration_ms: None,
            },
        ];

        let tasks = vec![
            SparkTask {
                id: 1,
                stage_id: 2,
                job_id: "job-001".to_string(),
                partition_id: 0,
                executor_id: "executor-1".to_string(),
                host: "worker-01".to_string(),
                status: TaskStatus::Running,
                duration_ms: Some(5000),
                gc_time_ms: 120,
                input_bytes: 1024 * 1024 * 10,
                output_bytes: 1024 * 1024 * 5,
                shuffle_read_bytes: 1024 * 1024 * 10,
                shuffle_write_bytes: 1024 * 1024 * 5,
            },
        ];

        Self {
            jobs,
            stages,
            tasks,
            next_job_id: 5,
        }
    }
}

impl RustySparkBridge {
    /// Create a new RustySpark bridge.
    pub fn new() -> Self {
        #[cfg(feature = "rustyspark-live")]
        {
            Self {
                server_url: Arc::new(RwLock::new("http://localhost:4040".to_string())),
                client: reqwest::Client::new(),
            }
        }
        #[cfg(not(feature = "rustyspark-live"))]
        {
            Self {
                server_url: Arc::new(RwLock::new("http://localhost:4040".to_string())),
                mock_state: Arc::new(RwLock::new(MockSparkState::default())),
            }
        }
    }

    /// Initialize the RustySpark bridge.
    pub async fn initialize(&self) -> Result<(), String> {
        #[cfg(feature = "rustyspark-live")]
        {
            let url = self.server_url.read().await.clone();
            match self
                .client
                .get(format!("{}/api/v1/health", url))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    tracing::info!("Connected to RustySpark WebUI at {}", url);
                    Ok(())
                }
                Ok(resp) => {
                    tracing::warn!("RustySpark WebUI returned status: {}", resp.status());
                    Ok(())
                }
                Err(e) => {
                    tracing::warn!("Could not connect to RustySpark WebUI: {}", e);
                    Ok(())
                }
            }
        }
        #[cfg(not(feature = "rustyspark-live"))]
        {
            tracing::info!("RustySpark bridge initialized (mock mode)");
            Ok(())
        }
    }

    /// Set the RustySpark server URL.
    pub async fn set_server_url(&self, url: String) {
        *self.server_url.write().await = url;
    }

    /// Get the current RustySpark server URL.
    pub async fn get_server_url(&self) -> String {
        self.server_url.read().await.clone()
    }

    /// Get RustySpark server status.
    pub async fn get_status(&self) -> RustySparkStatus {
        #[cfg(feature = "rustyspark-live")]
        {
            let url = self.server_url.read().await.clone();
            let connected = self
                .client
                .get(format!("{}/api/v1/health", url))
                .send()
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false);

            let summary = self.get_summary().await.ok();

            RustySparkStatus {
                connected,
                server_url: url,
                version: "unknown".to_string(),
                summary,
            }
        }
        #[cfg(not(feature = "rustyspark-live"))]
        {
            let summary = self.get_summary().await.ok();
            RustySparkStatus {
                connected: true,
                server_url: self.server_url.read().await.clone(),
                version: "mock-1.0.0".to_string(),
                summary,
            }
        }
    }

    /// Get summary statistics.
    pub async fn get_summary(&self) -> Result<SparkSummary, String> {
        #[cfg(feature = "rustyspark-live")]
        {
            // In live mode, aggregate from jobs endpoint
            let jobs = self.list_jobs(None, None).await?;
            let running = jobs.iter().filter(|j| j.status == SparkJobStatus::Running).count() as i64;
            let completed = jobs.iter().filter(|j| j.status == SparkJobStatus::Completed).count() as i64;
            let failed = jobs.iter().filter(|j| j.status == SparkJobStatus::Failed).count() as i64;

            Ok(SparkSummary {
                total_jobs: jobs.len() as i64,
                running_jobs: running,
                completed_jobs: completed,
                failed_jobs: failed,
                total_stages: jobs.iter().map(|j| j.total_stages as i64).sum(),
                total_tasks: jobs.iter().map(|j| j.total_tasks as i64).sum(),
                total_shuffle_read_bytes: 0,
                total_shuffle_write_bytes: 0,
            })
        }
        #[cfg(not(feature = "rustyspark-live"))]
        {
            let state = self.mock_state.read().await;
            let running = state.jobs.iter().filter(|j| j.status == SparkJobStatus::Running).count() as i64;
            let completed = state.jobs.iter().filter(|j| j.status == SparkJobStatus::Completed).count() as i64;
            let failed = state.jobs.iter().filter(|j| j.status == SparkJobStatus::Failed).count() as i64;

            Ok(SparkSummary {
                total_jobs: state.jobs.len() as i64,
                running_jobs: running,
                completed_jobs: completed,
                failed_jobs: failed,
                total_stages: state.stages.len() as i64,
                total_tasks: state.tasks.len() as i64,
                total_shuffle_read_bytes: state.stages.iter().map(|s| s.shuffle_read_bytes).sum(),
                total_shuffle_write_bytes: state.stages.iter().map(|s| s.shuffle_write_bytes).sum(),
            })
        }
    }

    /// List jobs.
    pub async fn list_jobs(
        &self,
        status: Option<SparkJobStatus>,
        limit: Option<i64>,
    ) -> Result<Vec<SparkJob>, String> {
        #[cfg(feature = "rustyspark-live")]
        {
            let url = self.server_url.read().await.clone();
            let mut req_url = format!("{}/api/v1/jobs", url);
            if let Some(s) = status {
                req_url = format!("{}?status={:?}", req_url, s);
            }

            let resp = self
                .client
                .get(&req_url)
                .send()
                .await
                .map_err(|e| format!("Failed to fetch jobs: {}", e))?;

            if !resp.status().is_success() {
                return Err(format!("Request returned status: {}", resp.status()));
            }

            resp.json()
                .await
                .map_err(|e| format!("Failed to parse response: {}", e))
        }
        #[cfg(not(feature = "rustyspark-live"))]
        {
            let state = self.mock_state.read().await;
            let mut jobs = state.jobs.clone();

            if let Some(s) = status {
                jobs.retain(|j| j.status == s);
            }

            let limit = limit.unwrap_or(100) as usize;
            jobs.truncate(limit);

            Ok(jobs)
        }
    }

    /// Get job by ID.
    pub async fn get_job(&self, id: &str) -> Result<SparkJob, String> {
        #[cfg(feature = "rustyspark-live")]
        {
            let url = self.server_url.read().await.clone();
            let resp = self
                .client
                .get(format!("{}/api/v1/jobs/{}", url, id))
                .send()
                .await
                .map_err(|e| format!("Failed to fetch job: {}", e))?;

            if resp.status().as_u16() == 404 {
                return Err(format!("Job {} not found", id));
            }
            if !resp.status().is_success() {
                return Err(format!("Request returned status: {}", resp.status()));
            }

            resp.json()
                .await
                .map_err(|e| format!("Failed to parse response: {}", e))
        }
        #[cfg(not(feature = "rustyspark-live"))]
        {
            let state = self.mock_state.read().await;
            state
                .jobs
                .iter()
                .find(|j| j.id == id)
                .cloned()
                .ok_or_else(|| format!("Job {} not found", id))
        }
    }

    /// Get stages for a job.
    pub async fn get_job_stages(&self, job_id: &str) -> Result<Vec<SparkStage>, String> {
        #[cfg(feature = "rustyspark-live")]
        {
            let url = self.server_url.read().await.clone();
            let resp = self
                .client
                .get(format!("{}/api/v1/jobs/{}/stages", url, job_id))
                .send()
                .await
                .map_err(|e| format!("Failed to fetch stages: {}", e))?;

            if !resp.status().is_success() {
                return Err(format!("Request returned status: {}", resp.status()));
            }

            resp.json()
                .await
                .map_err(|e| format!("Failed to parse response: {}", e))
        }
        #[cfg(not(feature = "rustyspark-live"))]
        {
            let state = self.mock_state.read().await;
            Ok(state
                .stages
                .iter()
                .filter(|s| s.job_id == job_id)
                .cloned()
                .collect())
        }
    }

    /// Get tasks for a stage.
    pub async fn get_stage_tasks(&self, stage_id: i64) -> Result<Vec<SparkTask>, String> {
        #[cfg(feature = "rustyspark-live")]
        {
            let url = self.server_url.read().await.clone();
            let resp = self
                .client
                .get(format!("{}/api/v1/stages/{}/tasks", url, stage_id))
                .send()
                .await
                .map_err(|e| format!("Failed to fetch tasks: {}", e))?;

            if !resp.status().is_success() {
                return Err(format!("Request returned status: {}", resp.status()));
            }

            resp.json()
                .await
                .map_err(|e| format!("Failed to parse response: {}", e))
        }
        #[cfg(not(feature = "rustyspark-live"))]
        {
            let state = self.mock_state.read().await;
            Ok(state
                .tasks
                .iter()
                .filter(|t| t.stage_id == stage_id)
                .cloned()
                .collect())
        }
    }

    /// Cancel a job.
    pub async fn cancel_job(&self, id: &str) -> Result<(), String> {
        #[cfg(feature = "rustyspark-live")]
        {
            let url = self.server_url.read().await.clone();
            let resp = self
                .client
                .delete(format!("{}/api/v1/jobs/{}", url, id))
                .send()
                .await
                .map_err(|e| format!("Failed to cancel job: {}", e))?;

            if !resp.status().is_success() {
                return Err(format!("Request returned status: {}", resp.status()));
            }
            Ok(())
        }
        #[cfg(not(feature = "rustyspark-live"))]
        {
            let mut state = self.mock_state.write().await;
            if let Some(job) = state.jobs.iter_mut().find(|j| j.id == id) {
                job.status = SparkJobStatus::Cancelled;
                Ok(())
            } else {
                Err(format!("Job {} not found", id))
            }
        }
    }
}

impl Default for RustySparkBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bridge_creation() {
        let bridge = RustySparkBridge::new();
        let _ = bridge.initialize().await;
        let status = bridge.get_status().await;
        assert!(status.connected || !cfg!(feature = "rustyspark-live"));
    }

    #[tokio::test]
    async fn test_list_jobs_mock() {
        let bridge = RustySparkBridge::new();
        let _ = bridge.initialize().await;

        #[cfg(not(feature = "rustyspark-live"))]
        {
            let jobs = bridge.list_jobs(None, None).await.expect("Should list jobs");
            assert!(!jobs.is_empty());
        }
    }

    #[tokio::test]
    async fn test_get_summary_mock() {
        let bridge = RustySparkBridge::new();
        let _ = bridge.initialize().await;

        #[cfg(not(feature = "rustyspark-live"))]
        {
            let summary = bridge.get_summary().await.expect("Should get summary");
            assert!(summary.total_jobs > 0);
        }
    }
}
