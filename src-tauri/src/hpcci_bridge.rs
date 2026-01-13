//! HPC-CI Bridge for Horizon Integration
//!
//! Provides CI/CD pipeline management via the HPC-CI controller API:
//! - Pipeline listing, details, and triggering
//! - Agent status and management
//! - Approval workflow
//! - Dashboard summary statistics
//!
//! # Features
//!
//! - REST API integration with HPC-CI controller
//! - Pipeline lifecycle management
//! - Agent health monitoring
//! - Mock mode for development without live HPC-CI server

use std::sync::Arc;
use tokio::sync::RwLock;

/// Pipeline status
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PipelineStatus {
    /// Pipeline is queued
    Queued,
    /// Pipeline is running
    Running,
    /// Pipeline completed successfully
    Success,
    /// Pipeline failed
    Failed,
    /// Pipeline was cancelled
    Cancelled,
    /// Pipeline timed out
    Timeout,
    /// Pipeline was skipped
    Skipped,
}

/// Pipeline summary for list view
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PipelineSummary {
    /// Pipeline ID
    pub id: String,
    /// Repository name
    pub repo: String,
    /// Branch name
    pub branch: String,
    /// Short commit SHA
    pub sha_short: String,
    /// Pipeline status
    pub status: PipelineStatus,
    /// Duration in milliseconds
    pub duration_ms: Option<u64>,
    /// Start timestamp (ms since epoch)
    pub started_at_ms: u64,
    /// Stages progress (e.g., "2/4")
    pub stages_progress: String,
}

/// Pipeline details
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PipelineDetail {
    /// Pipeline ID
    pub id: String,
    /// Repository name
    pub repo: String,
    /// Branch name
    pub branch: String,
    /// Full commit SHA
    pub sha: String,
    /// Pipeline status
    pub status: PipelineStatus,
    /// Duration in milliseconds
    pub duration_ms: Option<u64>,
    /// Start timestamp (ms since epoch)
    pub started_at_ms: u64,
    /// Finish timestamp (ms since epoch)
    pub finished_at_ms: Option<u64>,
    /// Pipeline stages
    pub stages: Vec<StageInfo>,
    /// Trigger information
    pub trigger: TriggerInfo,
}

/// Stage information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StageInfo {
    /// Stage name
    pub name: String,
    /// Stage status
    pub status: PipelineStatus,
    /// Duration in milliseconds
    pub duration_ms: Option<u64>,
    /// Jobs in this stage
    pub jobs: Vec<JobInfo>,
}

/// Job information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct JobInfo {
    /// Job name
    pub name: String,
    /// Job status
    pub status: PipelineStatus,
    /// Duration in milliseconds
    pub duration_ms: Option<u64>,
    /// Agent ID running this job
    pub agent_id: Option<String>,
}

/// Trigger information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TriggerInfo {
    /// Trigger type (push, pr, manual, schedule)
    pub trigger_type: String,
    /// User who triggered (if applicable)
    pub user: Option<String>,
    /// Commit message
    pub commit_message: Option<String>,
}

/// Agent status
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AgentStatus {
    /// Agent is online and accepting jobs
    Online,
    /// Agent is offline
    Offline,
    /// Agent is draining (finishing current jobs)
    Draining,
    /// Agent is in maintenance mode
    Maintenance,
}

/// Agent summary for list view
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentSummary {
    /// Agent ID
    pub id: String,
    /// Agent status
    pub status: AgentStatus,
    /// Current number of running jobs
    pub current_jobs: u32,
    /// Maximum concurrent jobs
    pub max_jobs: u32,
    /// Agent capabilities (e.g., ["docker", "gpu"])
    pub capabilities: Vec<String>,
    /// Number of GPUs (if any)
    pub gpu_count: Option<u32>,
}

/// Dashboard summary statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DashboardSummary {
    /// Number of running pipelines
    pub pipelines_running: u32,
    /// Number of queued pipelines
    pub pipelines_queued: u32,
    /// Pipelines succeeded in last 24h
    pub pipelines_succeeded_24h: u32,
    /// Pipelines failed in last 24h
    pub pipelines_failed_24h: u32,
    /// Number of online agents
    pub agents_online: u32,
    /// Total number of agents
    pub agents_total: u32,
    /// Recent pipeline summaries
    pub recent_pipelines: Vec<PipelineSummary>,
}

/// Approval request
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ApprovalRequest {
    /// Approval ID
    pub id: String,
    /// Pipeline ID
    pub pipeline_id: String,
    /// Environment name
    pub environment: String,
    /// User who requested
    pub requested_by: String,
    /// Request timestamp (ms since epoch)
    pub requested_at_ms: u64,
    /// Approval status
    pub status: ApprovalStatus,
}

/// Approval status
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ApprovalStatus {
    /// Pending approval
    Pending,
    /// Approved
    Approved,
    /// Rejected
    Rejected,
    /// Expired
    Expired,
}

/// Pipeline filter for list queries
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PipelineFilter {
    /// Filter by status
    pub status: Option<PipelineStatus>,
    /// Filter by repository
    pub repo: Option<String>,
    /// Filter by branch
    pub branch: Option<String>,
    /// Maximum results
    pub limit: Option<u32>,
    /// Offset for pagination
    pub offset: Option<u32>,
}

/// Trigger parameters for starting a pipeline
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TriggerParams {
    /// Repository name
    pub repo: String,
    /// Branch name
    pub branch: String,
    /// Specific commit SHA (optional)
    pub sha: Option<String>,
    /// Input parameters (optional)
    pub inputs: Option<serde_json::Value>,
}

/// Log entry
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LogEntry {
    /// Timestamp (ms since epoch)
    pub timestamp_ms: u64,
    /// Stage name
    pub stage: String,
    /// Job name
    pub job: String,
    /// Log level
    pub level: String,
    /// Log content
    pub content: String,
}

/// Logs response
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LogsResponse {
    /// Pipeline ID
    pub pipeline_id: String,
    /// Log entries
    pub entries: Vec<LogEntry>,
    /// Whether more logs are available
    pub has_more: bool,
    /// Next offset for pagination
    pub next_offset: Option<u64>,
}

/// HPC-CI server status
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct HpcCiStatus {
    /// Whether connected to HPC-CI controller
    pub connected: bool,
    /// Server URL
    pub server_url: String,
    /// Server version
    pub version: String,
    /// Summary statistics
    pub summary: Option<DashboardSummary>,
}

/// HPC-CI Bridge for Horizon integration.
pub struct HpcCiBridge {
    /// Server URL (configurable)
    server_url: Arc<RwLock<String>>,
    /// HTTP client for API requests
    #[cfg(feature = "hpcci-live")]
    client: reqwest::Client,
    /// Mock state for development
    #[cfg(not(feature = "hpcci-live"))]
    mock_state: Arc<RwLock<MockHpcCiState>>,
}

#[cfg(not(feature = "hpcci-live"))]
struct MockHpcCiState {
    pipelines: Vec<PipelineSummary>,
    pipeline_details: std::collections::HashMap<String, PipelineDetail>,
    agents: Vec<AgentSummary>,
    approvals: Vec<ApprovalRequest>,
    next_pipeline_id: u64,
}

#[cfg(not(feature = "hpcci-live"))]
impl Default for MockHpcCiState {
    fn default() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let pipelines = vec![
            PipelineSummary {
                id: "PL-001".to_string(),
                repo: "rustystack/horizon".to_string(),
                branch: "main".to_string(),
                sha_short: "a1b2c3d".to_string(),
                status: PipelineStatus::Running,
                duration_ms: Some(45000),
                started_at_ms: now - 45000,
                stages_progress: "2/4".to_string(),
            },
            PipelineSummary {
                id: "PL-002".to_string(),
                repo: "rustystack/slai".to_string(),
                branch: "feature/scheduler".to_string(),
                sha_short: "e5f6g7h".to_string(),
                status: PipelineStatus::Success,
                duration_ms: Some(120000),
                started_at_ms: now - 300000,
                stages_progress: "3/3".to_string(),
            },
            PipelineSummary {
                id: "PL-003".to_string(),
                repo: "rustystack/argus".to_string(),
                branch: "main".to_string(),
                sha_short: "i8j9k0l".to_string(),
                status: PipelineStatus::Failed,
                duration_ms: Some(60000),
                started_at_ms: now - 600000,
                stages_progress: "2/4".to_string(),
            },
            PipelineSummary {
                id: "PL-004".to_string(),
                repo: "rustystack/warp".to_string(),
                branch: "develop".to_string(),
                sha_short: "m1n2o3p".to_string(),
                status: PipelineStatus::Queued,
                duration_ms: None,
                started_at_ms: now - 5000,
                stages_progress: "0/5".to_string(),
            },
        ];

        let mut pipeline_details = std::collections::HashMap::new();
        pipeline_details.insert(
            "PL-001".to_string(),
            PipelineDetail {
                id: "PL-001".to_string(),
                repo: "rustystack/horizon".to_string(),
                branch: "main".to_string(),
                sha: "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6".to_string(),
                status: PipelineStatus::Running,
                duration_ms: Some(45000),
                started_at_ms: now - 45000,
                finished_at_ms: None,
                stages: vec![
                    StageInfo {
                        name: "validate".to_string(),
                        status: PipelineStatus::Success,
                        duration_ms: Some(15000),
                        jobs: vec![
                            JobInfo {
                                name: "lint".to_string(),
                                status: PipelineStatus::Success,
                                duration_ms: Some(8000),
                                agent_id: Some("agent-01".to_string()),
                            },
                            JobInfo {
                                name: "format".to_string(),
                                status: PipelineStatus::Success,
                                duration_ms: Some(5000),
                                agent_id: Some("agent-02".to_string()),
                            },
                        ],
                    },
                    StageInfo {
                        name: "build".to_string(),
                        status: PipelineStatus::Running,
                        duration_ms: Some(30000),
                        jobs: vec![JobInfo {
                            name: "compile".to_string(),
                            status: PipelineStatus::Running,
                            duration_ms: None,
                            agent_id: Some("agent-01".to_string()),
                        }],
                    },
                    StageInfo {
                        name: "test".to_string(),
                        status: PipelineStatus::Queued,
                        duration_ms: None,
                        jobs: vec![
                            JobInfo {
                                name: "unit-tests".to_string(),
                                status: PipelineStatus::Queued,
                                duration_ms: None,
                                agent_id: None,
                            },
                            JobInfo {
                                name: "integration-tests".to_string(),
                                status: PipelineStatus::Queued,
                                duration_ms: None,
                                agent_id: None,
                            },
                        ],
                    },
                    StageInfo {
                        name: "deploy".to_string(),
                        status: PipelineStatus::Queued,
                        duration_ms: None,
                        jobs: vec![JobInfo {
                            name: "deploy-staging".to_string(),
                            status: PipelineStatus::Queued,
                            duration_ms: None,
                            agent_id: None,
                        }],
                    },
                ],
                trigger: TriggerInfo {
                    trigger_type: "push".to_string(),
                    user: Some("developer".to_string()),
                    commit_message: Some("feat: add dashboard components".to_string()),
                },
            },
        );

        let agents = vec![
            AgentSummary {
                id: "agent-01".to_string(),
                status: AgentStatus::Online,
                current_jobs: 2,
                max_jobs: 4,
                capabilities: vec!["docker".to_string(), "gpu".to_string(), "linux".to_string()],
                gpu_count: Some(4),
            },
            AgentSummary {
                id: "agent-02".to_string(),
                status: AgentStatus::Online,
                current_jobs: 1,
                max_jobs: 4,
                capabilities: vec!["docker".to_string(), "linux".to_string()],
                gpu_count: None,
            },
            AgentSummary {
                id: "agent-03".to_string(),
                status: AgentStatus::Draining,
                current_jobs: 1,
                max_jobs: 2,
                capabilities: vec!["docker".to_string(), "macos".to_string()],
                gpu_count: None,
            },
            AgentSummary {
                id: "agent-04".to_string(),
                status: AgentStatus::Offline,
                current_jobs: 0,
                max_jobs: 4,
                capabilities: vec!["docker".to_string(), "gpu".to_string(), "linux".to_string()],
                gpu_count: Some(8),
            },
        ];

        let approvals = vec![ApprovalRequest {
            id: "APR-001".to_string(),
            pipeline_id: "PL-001".to_string(),
            environment: "production".to_string(),
            requested_by: "developer".to_string(),
            requested_at_ms: now - 30000,
            status: ApprovalStatus::Pending,
        }];

        Self {
            pipelines,
            pipeline_details,
            agents,
            approvals,
            next_pipeline_id: 5,
        }
    }
}

impl HpcCiBridge {
    /// Create a new HPC-CI bridge.
    pub fn new() -> Self {
        #[cfg(feature = "hpcci-live")]
        {
            Self {
                server_url: Arc::new(RwLock::new("http://localhost:9000".to_string())),
                client: reqwest::Client::new(),
            }
        }
        #[cfg(not(feature = "hpcci-live"))]
        {
            Self {
                server_url: Arc::new(RwLock::new("http://localhost:9000".to_string())),
                mock_state: Arc::new(RwLock::new(MockHpcCiState::default())),
            }
        }
    }

    /// Initialize the HPC-CI bridge.
    pub async fn initialize(&self) -> Result<(), String> {
        #[cfg(feature = "hpcci-live")]
        {
            let url = self.server_url.read().await.clone();
            match self
                .client
                .get(format!("{}/api/v1/status", url))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    tracing::info!("Connected to HPC-CI controller at {}", url);
                    Ok(())
                }
                Ok(resp) => {
                    tracing::warn!("HPC-CI controller returned status: {}", resp.status());
                    Ok(())
                }
                Err(e) => {
                    tracing::warn!("Could not connect to HPC-CI controller: {}", e);
                    Ok(())
                }
            }
        }
        #[cfg(not(feature = "hpcci-live"))]
        {
            tracing::info!("HPC-CI bridge initialized (mock mode)");
            Ok(())
        }
    }

    /// Set the HPC-CI server URL.
    pub async fn set_server_url(&self, url: String) {
        *self.server_url.write().await = url;
    }

    /// Get the current HPC-CI server URL.
    pub async fn get_server_url(&self) -> String {
        self.server_url.read().await.clone()
    }

    /// Get HPC-CI server status.
    pub async fn get_status(&self) -> HpcCiStatus {
        #[cfg(feature = "hpcci-live")]
        {
            let url = self.server_url.read().await.clone();
            let connected = self
                .client
                .get(format!("{}/api/v1/status", url))
                .send()
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false);

            let summary = self.get_dashboard_summary().await.ok();

            HpcCiStatus {
                connected,
                server_url: url,
                version: "unknown".to_string(),
                summary,
            }
        }
        #[cfg(not(feature = "hpcci-live"))]
        {
            let summary = self.get_dashboard_summary().await.ok();
            HpcCiStatus {
                connected: true,
                server_url: self.server_url.read().await.clone(),
                version: "mock-1.0.0".to_string(),
                summary,
            }
        }
    }

    /// List pipelines with optional filters.
    pub async fn list_pipelines(
        &self,
        filter: PipelineFilter,
    ) -> Result<Vec<PipelineSummary>, String> {
        #[cfg(feature = "hpcci-live")]
        {
            let url = self.server_url.read().await.clone();
            let resp = self
                .client
                .get(format!("{}/api/v1/pipelines", url))
                .query(&filter)
                .send()
                .await
                .map_err(|e| format!("Failed to fetch pipelines: {}", e))?;

            if !resp.status().is_success() {
                return Err(format!("Request returned status: {}", resp.status()));
            }

            resp.json()
                .await
                .map_err(|e| format!("Failed to parse response: {}", e))
        }
        #[cfg(not(feature = "hpcci-live"))]
        {
            let state = self.mock_state.read().await;
            let mut pipelines = state.pipelines.clone();

            // Apply filters
            if let Some(status) = filter.status {
                pipelines.retain(|p| p.status == status);
            }
            if let Some(ref repo) = filter.repo {
                pipelines.retain(|p| p.repo.contains(repo));
            }
            if let Some(ref branch) = filter.branch {
                pipelines.retain(|p| p.branch == *branch);
            }

            // Apply pagination
            let offset = filter.offset.unwrap_or(0) as usize;
            let limit = filter.limit.unwrap_or(50) as usize;
            let pipelines: Vec<_> = pipelines.into_iter().skip(offset).take(limit).collect();

            Ok(pipelines)
        }
    }

    /// Get pipeline details.
    pub async fn get_pipeline(&self, id: &str) -> Result<PipelineDetail, String> {
        #[cfg(feature = "hpcci-live")]
        {
            let url = self.server_url.read().await.clone();
            let resp = self
                .client
                .get(format!("{}/api/v1/pipelines/{}", url, id))
                .send()
                .await
                .map_err(|e| format!("Failed to fetch pipeline: {}", e))?;

            if resp.status().as_u16() == 404 {
                return Err(format!("Pipeline {} not found", id));
            }
            if !resp.status().is_success() {
                return Err(format!("Request returned status: {}", resp.status()));
            }

            resp.json()
                .await
                .map_err(|e| format!("Failed to parse response: {}", e))
        }
        #[cfg(not(feature = "hpcci-live"))]
        {
            let state = self.mock_state.read().await;
            state
                .pipeline_details
                .get(id)
                .cloned()
                .ok_or_else(|| format!("Pipeline {} not found", id))
        }
    }

    /// Trigger a new pipeline.
    pub async fn trigger_pipeline(&self, params: TriggerParams) -> Result<String, String> {
        #[cfg(feature = "hpcci-live")]
        {
            let url = self.server_url.read().await.clone();
            let resp = self
                .client
                .post(format!("{}/api/v1/pipelines/trigger", url))
                .json(&params)
                .send()
                .await
                .map_err(|e| format!("Failed to trigger pipeline: {}", e))?;

            if !resp.status().is_success() {
                return Err(format!("Request returned status: {}", resp.status()));
            }

            #[derive(serde::Deserialize)]
            struct TriggerResponse {
                pipeline_id: String,
            }
            let result: TriggerResponse = resp
                .json()
                .await
                .map_err(|e| format!("Failed to parse response: {}", e))?;
            Ok(result.pipeline_id)
        }
        #[cfg(not(feature = "hpcci-live"))]
        {
            let mut state = self.mock_state.write().await;
            let pipeline_id = format!("PL-{:03}", state.next_pipeline_id);
            state.next_pipeline_id += 1;

            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;

            let summary = PipelineSummary {
                id: pipeline_id.clone(),
                repo: params.repo.clone(),
                branch: params.branch.clone(),
                sha_short: params
                    .sha
                    .as_ref()
                    .map(|s| s[..7].to_string())
                    .unwrap_or_else(|| "abc1234".to_string()),
                status: PipelineStatus::Queued,
                duration_ms: None,
                started_at_ms: now,
                stages_progress: "0/3".to_string(),
            };

            state.pipelines.insert(0, summary);
            Ok(pipeline_id)
        }
    }

    /// Cancel a pipeline.
    pub async fn cancel_pipeline(&self, id: &str) -> Result<(), String> {
        #[cfg(feature = "hpcci-live")]
        {
            let url = self.server_url.read().await.clone();
            let resp = self
                .client
                .post(format!("{}/api/v1/pipelines/{}/cancel", url, id))
                .send()
                .await
                .map_err(|e| format!("Failed to cancel pipeline: {}", e))?;

            if !resp.status().is_success() {
                return Err(format!("Request returned status: {}", resp.status()));
            }
            Ok(())
        }
        #[cfg(not(feature = "hpcci-live"))]
        {
            let mut state = self.mock_state.write().await;
            if let Some(pipeline) = state.pipelines.iter_mut().find(|p| p.id == id) {
                pipeline.status = PipelineStatus::Cancelled;
                Ok(())
            } else {
                Err(format!("Pipeline {} not found", id))
            }
        }
    }

    /// Retry a failed pipeline.
    pub async fn retry_pipeline(&self, id: &str) -> Result<String, String> {
        #[cfg(feature = "hpcci-live")]
        {
            let url = self.server_url.read().await.clone();
            let resp = self
                .client
                .post(format!("{}/api/v1/pipelines/{}/retry", url, id))
                .send()
                .await
                .map_err(|e| format!("Failed to retry pipeline: {}", e))?;

            if !resp.status().is_success() {
                return Err(format!("Request returned status: {}", resp.status()));
            }

            #[derive(serde::Deserialize)]
            struct RetryResponse {
                pipeline_id: String,
            }
            let result: RetryResponse = resp
                .json()
                .await
                .map_err(|e| format!("Failed to parse response: {}", e))?;
            Ok(result.pipeline_id)
        }
        #[cfg(not(feature = "hpcci-live"))]
        {
            // In mock mode, create a new pipeline based on the old one
            let mut state = self.mock_state.write().await;
            if let Some(old_pipeline) = state.pipelines.iter().find(|p| p.id == id).cloned() {
                let new_id = format!("PL-{:03}", state.next_pipeline_id);
                state.next_pipeline_id += 1;

                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64;

                let new_pipeline = PipelineSummary {
                    id: new_id.clone(),
                    repo: old_pipeline.repo,
                    branch: old_pipeline.branch,
                    sha_short: old_pipeline.sha_short,
                    status: PipelineStatus::Queued,
                    duration_ms: None,
                    started_at_ms: now,
                    stages_progress: "0/3".to_string(),
                };

                state.pipelines.insert(0, new_pipeline);
                Ok(new_id)
            } else {
                Err(format!("Pipeline {} not found", id))
            }
        }
    }

    /// List all agents.
    pub async fn list_agents(&self) -> Result<Vec<AgentSummary>, String> {
        #[cfg(feature = "hpcci-live")]
        {
            let url = self.server_url.read().await.clone();
            let resp = self
                .client
                .get(format!("{}/api/v1/agents", url))
                .send()
                .await
                .map_err(|e| format!("Failed to fetch agents: {}", e))?;

            if !resp.status().is_success() {
                return Err(format!("Request returned status: {}", resp.status()));
            }

            resp.json()
                .await
                .map_err(|e| format!("Failed to parse response: {}", e))
        }
        #[cfg(not(feature = "hpcci-live"))]
        {
            Ok(self.mock_state.read().await.agents.clone())
        }
    }

    /// Drain an agent.
    pub async fn drain_agent(&self, id: &str) -> Result<(), String> {
        #[cfg(feature = "hpcci-live")]
        {
            let url = self.server_url.read().await.clone();
            let resp = self
                .client
                .post(format!("{}/api/v1/agents/{}/drain", url, id))
                .send()
                .await
                .map_err(|e| format!("Failed to drain agent: {}", e))?;

            if !resp.status().is_success() {
                return Err(format!("Request returned status: {}", resp.status()));
            }
            Ok(())
        }
        #[cfg(not(feature = "hpcci-live"))]
        {
            let mut state = self.mock_state.write().await;
            if let Some(agent) = state.agents.iter_mut().find(|a| a.id == id) {
                agent.status = AgentStatus::Draining;
                Ok(())
            } else {
                Err(format!("Agent {} not found", id))
            }
        }
    }

    /// Enable an agent.
    pub async fn enable_agent(&self, id: &str) -> Result<(), String> {
        #[cfg(feature = "hpcci-live")]
        {
            let url = self.server_url.read().await.clone();
            let resp = self
                .client
                .post(format!("{}/api/v1/agents/{}/enable", url, id))
                .send()
                .await
                .map_err(|e| format!("Failed to enable agent: {}", e))?;

            if !resp.status().is_success() {
                return Err(format!("Request returned status: {}", resp.status()));
            }
            Ok(())
        }
        #[cfg(not(feature = "hpcci-live"))]
        {
            let mut state = self.mock_state.write().await;
            if let Some(agent) = state.agents.iter_mut().find(|a| a.id == id) {
                agent.status = AgentStatus::Online;
                Ok(())
            } else {
                Err(format!("Agent {} not found", id))
            }
        }
    }

    /// Get pending approvals.
    pub async fn get_approvals(&self) -> Result<Vec<ApprovalRequest>, String> {
        #[cfg(feature = "hpcci-live")]
        {
            let url = self.server_url.read().await.clone();
            let resp = self
                .client
                .get(format!("{}/api/v1/approvals", url))
                .send()
                .await
                .map_err(|e| format!("Failed to fetch approvals: {}", e))?;

            if !resp.status().is_success() {
                return Err(format!("Request returned status: {}", resp.status()));
            }

            resp.json()
                .await
                .map_err(|e| format!("Failed to parse response: {}", e))
        }
        #[cfg(not(feature = "hpcci-live"))]
        {
            Ok(self
                .mock_state
                .read()
                .await
                .approvals
                .iter()
                .filter(|a| a.status == ApprovalStatus::Pending)
                .cloned()
                .collect())
        }
    }

    /// Submit an approval decision.
    pub async fn submit_approval(
        &self,
        id: &str,
        approved: bool,
        comment: Option<String>,
    ) -> Result<(), String> {
        #[cfg(feature = "hpcci-live")]
        {
            let url = self.server_url.read().await.clone();

            #[derive(serde::Serialize)]
            struct ApprovalSubmission {
                approved: bool,
                comment: Option<String>,
            }

            let resp = self
                .client
                .post(format!("{}/api/v1/approvals/{}", url, id))
                .json(&ApprovalSubmission { approved, comment })
                .send()
                .await
                .map_err(|e| format!("Failed to submit approval: {}", e))?;

            if !resp.status().is_success() {
                return Err(format!("Request returned status: {}", resp.status()));
            }
            Ok(())
        }
        #[cfg(not(feature = "hpcci-live"))]
        {
            let mut state = self.mock_state.write().await;
            if let Some(approval) = state.approvals.iter_mut().find(|a| a.id == id) {
                approval.status = if approved {
                    ApprovalStatus::Approved
                } else {
                    ApprovalStatus::Rejected
                };
                Ok(())
            } else {
                Err(format!("Approval {} not found", id))
            }
        }
    }

    /// Get dashboard summary statistics.
    pub async fn get_dashboard_summary(&self) -> Result<DashboardSummary, String> {
        #[cfg(feature = "hpcci-live")]
        {
            let url = self.server_url.read().await.clone();
            let resp = self
                .client
                .get(format!("{}/api/v1/dashboard/summary", url))
                .send()
                .await
                .map_err(|e| format!("Failed to fetch dashboard summary: {}", e))?;

            if !resp.status().is_success() {
                return Err(format!("Request returned status: {}", resp.status()));
            }

            resp.json()
                .await
                .map_err(|e| format!("Failed to parse response: {}", e))
        }
        #[cfg(not(feature = "hpcci-live"))]
        {
            let state = self.mock_state.read().await;
            let running = state
                .pipelines
                .iter()
                .filter(|p| p.status == PipelineStatus::Running)
                .count() as u32;
            let queued = state
                .pipelines
                .iter()
                .filter(|p| p.status == PipelineStatus::Queued)
                .count() as u32;
            let succeeded = state
                .pipelines
                .iter()
                .filter(|p| p.status == PipelineStatus::Success)
                .count() as u32;
            let failed = state
                .pipelines
                .iter()
                .filter(|p| p.status == PipelineStatus::Failed)
                .count() as u32;
            let online = state
                .agents
                .iter()
                .filter(|a| a.status == AgentStatus::Online)
                .count() as u32;

            Ok(DashboardSummary {
                pipelines_running: running,
                pipelines_queued: queued,
                pipelines_succeeded_24h: succeeded,
                pipelines_failed_24h: failed,
                agents_online: online,
                agents_total: state.agents.len() as u32,
                recent_pipelines: state.pipelines.clone(),
            })
        }
    }

    /// Get pipeline logs.
    pub async fn get_pipeline_logs(&self, id: &str, offset: Option<u64>) -> Result<LogsResponse, String> {
        #[cfg(feature = "hpcci-live")]
        {
            let url = self.server_url.read().await.clone();
            let mut req = self.client.get(format!("{}/api/v1/pipelines/{}/logs", url, id));
            if let Some(off) = offset {
                req = req.query(&[("offset", off)]);
            }
            let resp = req
                .send()
                .await
                .map_err(|e| format!("Failed to fetch logs: {}", e))?;

            if resp.status().as_u16() == 404 {
                return Err(format!("Pipeline {} not found", id));
            }
            if !resp.status().is_success() {
                return Err(format!("Request returned status: {}", resp.status()));
            }

            resp.json()
                .await
                .map_err(|e| format!("Failed to parse response: {}", e))
        }
        #[cfg(not(feature = "hpcci-live"))]
        {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;

            // Generate mock log entries
            let entries = vec![
                LogEntry {
                    timestamp_ms: now - 45000,
                    stage: "validate".to_string(),
                    job: "lint".to_string(),
                    level: "info".to_string(),
                    content: "Running cargo clippy...".to_string(),
                },
                LogEntry {
                    timestamp_ms: now - 44000,
                    stage: "validate".to_string(),
                    job: "lint".to_string(),
                    level: "info".to_string(),
                    content: "Checking rustystack/horizon v0.1.0".to_string(),
                },
                LogEntry {
                    timestamp_ms: now - 43000,
                    stage: "validate".to_string(),
                    job: "lint".to_string(),
                    level: "warn".to_string(),
                    content: "warning: unused variable `x`".to_string(),
                },
                LogEntry {
                    timestamp_ms: now - 40000,
                    stage: "build".to_string(),
                    job: "compile".to_string(),
                    level: "info".to_string(),
                    content: "Compiling horizon v0.1.0 (/workspace)".to_string(),
                },
                LogEntry {
                    timestamp_ms: now - 35000,
                    stage: "build".to_string(),
                    job: "compile".to_string(),
                    level: "info".to_string(),
                    content: "Compiling 142 crates...".to_string(),
                },
            ];

            Ok(LogsResponse {
                pipeline_id: id.to_string(),
                entries,
                has_more: false,
                next_offset: None,
            })
        }
    }
}

impl Default for HpcCiBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bridge_creation() {
        let bridge = HpcCiBridge::new();
        let _ = bridge.initialize().await;
        let status = bridge.get_status().await;
        assert!(status.connected || !cfg!(feature = "hpcci-live"));
    }

    #[tokio::test]
    async fn test_list_pipelines_mock() {
        let bridge = HpcCiBridge::new();
        let _ = bridge.initialize().await;

        #[cfg(not(feature = "hpcci-live"))]
        {
            let pipelines = bridge
                .list_pipelines(PipelineFilter::default())
                .await
                .expect("Should list pipelines");
            assert!(!pipelines.is_empty());
        }
    }

    #[tokio::test]
    async fn test_list_agents_mock() {
        let bridge = HpcCiBridge::new();
        let _ = bridge.initialize().await;

        #[cfg(not(feature = "hpcci-live"))]
        {
            let agents = bridge.list_agents().await.expect("Should list agents");
            assert!(!agents.is_empty());
            assert!(agents.iter().any(|a| a.status == AgentStatus::Online));
        }
    }

    #[tokio::test]
    async fn test_trigger_pipeline_mock() {
        let bridge = HpcCiBridge::new();
        let _ = bridge.initialize().await;

        #[cfg(not(feature = "hpcci-live"))]
        {
            let params = TriggerParams {
                repo: "rustystack/test".to_string(),
                branch: "main".to_string(),
                sha: None,
                inputs: None,
            };

            let id = bridge
                .trigger_pipeline(params)
                .await
                .expect("Should trigger pipeline");
            assert!(id.starts_with("PL-"));
        }
    }

    #[tokio::test]
    async fn test_dashboard_summary_mock() {
        let bridge = HpcCiBridge::new();
        let _ = bridge.initialize().await;

        #[cfg(not(feature = "hpcci-live"))]
        {
            let summary = bridge
                .get_dashboard_summary()
                .await
                .expect("Should get summary");
            assert!(summary.agents_total > 0);
        }
    }
}
