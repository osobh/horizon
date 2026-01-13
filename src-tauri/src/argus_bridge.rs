//! Argus Bridge for Horizon Integration
//!
//! Provides observability features via the Argus Prometheus-compatible API:
//! - Metrics querying (PromQL instant and range queries)
//! - Alert management (listing active/pending/resolved alerts)
//! - Target status (scrape target health monitoring)
//!
//! # Features
//!
//! - Prometheus-compatible query API
//! - Alert severity classification
//! - Target health monitoring
//! - Mock mode for development without live Argus server

use std::sync::Arc;
use tokio::sync::RwLock;

/// Query result from Prometheus-compatible API.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QueryResult {
    /// Result type: "vector", "matrix", "scalar", "string"
    pub result_type: String,
    /// Result data
    pub data: Vec<MetricSample>,
}

/// A single metric sample (instant or range).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MetricSample {
    /// Metric labels
    pub metric: std::collections::HashMap<String, String>,
    /// Value for instant queries (timestamp, value)
    #[serde(default)]
    pub value: Option<(f64, String)>,
    /// Values for range queries (timestamp, value) pairs
    #[serde(default)]
    pub values: Vec<(f64, String)>,
}

/// Alert information.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Alert {
    /// Alert name
    pub name: String,
    /// Alert state: firing, pending, resolved
    pub state: String,
    /// Severity: critical, warning, info
    pub severity: String,
    /// Alert summary
    pub summary: String,
    /// Alert description
    pub description: String,
    /// Labels associated with the alert
    pub labels: std::collections::HashMap<String, String>,
    /// Annotations (summary, description, etc.)
    pub annotations: std::collections::HashMap<String, String>,
    /// Active since (Unix timestamp)
    pub active_at: f64,
    /// Resolved at (Unix timestamp, if resolved)
    pub resolved_at: Option<f64>,
    /// Alert fingerprint (unique identifier)
    pub fingerprint: String,
}

/// Scrape target information.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Target {
    /// Target labels
    pub labels: std::collections::HashMap<String, String>,
    /// Target scrape URL
    pub scrape_url: String,
    /// Target health: up, down, unknown
    pub health: String,
    /// Last scrape timestamp (Unix)
    pub last_scrape: f64,
    /// Last scrape duration in seconds
    pub last_scrape_duration: f64,
    /// Last error (if any)
    pub last_error: Option<String>,
    /// Job name
    pub job: String,
    /// Instance identifier
    pub instance: String,
}

/// Argus server status.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ArgusStatus {
    /// Whether connected to Argus server
    pub connected: bool,
    /// Server URL
    pub server_url: String,
    /// Server version
    pub version: String,
    /// Total active alerts
    pub active_alerts: usize,
    /// Total targets
    pub total_targets: usize,
    /// Healthy targets
    pub healthy_targets: usize,
    /// Last successful query timestamp
    pub last_query: Option<f64>,
}

/// Argus Bridge for Horizon integration.
pub struct ArgusBridge {
    /// Server URL (configurable)
    server_url: Arc<RwLock<String>>,
    /// HTTP client for API requests
    #[cfg(feature = "argus-live")]
    client: reqwest::Client,
    /// Mock state for development
    #[cfg(not(feature = "argus-live"))]
    mock_state: Arc<RwLock<MockArgusState>>,
}

#[cfg(not(feature = "argus-live"))]
struct MockArgusState {
    alerts: Vec<Alert>,
    targets: Vec<Target>,
    metrics: std::collections::HashMap<String, Vec<MetricSample>>,
}

#[cfg(not(feature = "argus-live"))]
impl Default for MockArgusState {
    fn default() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        Self {
            alerts: vec![
                Alert {
                    name: "HighGPUUtilization".to_string(),
                    state: "firing".to_string(),
                    severity: "warning".to_string(),
                    summary: "GPU utilization above 90%".to_string(),
                    description: "GPU 0 has been running at >90% utilization for 5 minutes".to_string(),
                    labels: [
                        ("alertname".to_string(), "HighGPUUtilization".to_string()),
                        ("instance".to_string(), "gpu-node-01:9090".to_string()),
                        ("gpu".to_string(), "0".to_string()),
                    ].into_iter().collect(),
                    annotations: [
                        ("summary".to_string(), "GPU utilization above 90%".to_string()),
                    ].into_iter().collect(),
                    active_at: now - 300.0,
                    resolved_at: None,
                    fingerprint: "abc123".to_string(),
                },
                Alert {
                    name: "HighMemoryPressure".to_string(),
                    state: "pending".to_string(),
                    severity: "critical".to_string(),
                    summary: "Memory usage above 95%".to_string(),
                    description: "Node memory usage is critically high".to_string(),
                    labels: [
                        ("alertname".to_string(), "HighMemoryPressure".to_string()),
                        ("instance".to_string(), "worker-02:9090".to_string()),
                    ].into_iter().collect(),
                    annotations: [
                        ("summary".to_string(), "Memory usage above 95%".to_string()),
                    ].into_iter().collect(),
                    active_at: now - 60.0,
                    resolved_at: None,
                    fingerprint: "def456".to_string(),
                },
            ],
            targets: vec![
                Target {
                    labels: [
                        ("job".to_string(), "node-exporter".to_string()),
                        ("instance".to_string(), "gpu-node-01:9100".to_string()),
                    ].into_iter().collect(),
                    scrape_url: "http://gpu-node-01:9100/metrics".to_string(),
                    health: "up".to_string(),
                    last_scrape: now - 2.0,
                    last_scrape_duration: 0.045,
                    last_error: None,
                    job: "node-exporter".to_string(),
                    instance: "gpu-node-01:9100".to_string(),
                },
                Target {
                    labels: [
                        ("job".to_string(), "dcgm-exporter".to_string()),
                        ("instance".to_string(), "gpu-node-01:9400".to_string()),
                    ].into_iter().collect(),
                    scrape_url: "http://gpu-node-01:9400/metrics".to_string(),
                    health: "up".to_string(),
                    last_scrape: now - 1.5,
                    last_scrape_duration: 0.032,
                    last_error: None,
                    job: "dcgm-exporter".to_string(),
                    instance: "gpu-node-01:9400".to_string(),
                },
                Target {
                    labels: [
                        ("job".to_string(), "node-exporter".to_string()),
                        ("instance".to_string(), "worker-02:9100".to_string()),
                    ].into_iter().collect(),
                    scrape_url: "http://worker-02:9100/metrics".to_string(),
                    health: "down".to_string(),
                    last_scrape: now - 30.0,
                    last_scrape_duration: 0.0,
                    last_error: Some("connection refused".to_string()),
                    job: "node-exporter".to_string(),
                    instance: "worker-02:9100".to_string(),
                },
            ],
            metrics: std::collections::HashMap::new(),
        }
    }
}

impl ArgusBridge {
    /// Create a new Argus bridge.
    pub fn new() -> Self {
        #[cfg(feature = "argus-live")]
        {
            Self {
                server_url: Arc::new(RwLock::new("http://localhost:9090".to_string())),
                client: reqwest::Client::new(),
            }
        }
        #[cfg(not(feature = "argus-live"))]
        {
            Self {
                server_url: Arc::new(RwLock::new("http://localhost:9090".to_string())),
                mock_state: Arc::new(RwLock::new(MockArgusState::default())),
            }
        }
    }

    /// Initialize the Argus bridge.
    pub async fn initialize(&self) -> Result<(), String> {
        #[cfg(feature = "argus-live")]
        {
            // Test connection to Argus server
            let url = self.server_url.read().await.clone();
            match self.client.get(format!("{}/api/v1/status/buildinfo", url)).send().await {
                Ok(resp) if resp.status().is_success() => {
                    tracing::info!("Connected to Argus server at {}", url);
                    Ok(())
                }
                Ok(resp) => {
                    tracing::warn!("Argus server returned status: {}", resp.status());
                    Ok(()) // Still allow initialization
                }
                Err(e) => {
                    tracing::warn!("Could not connect to Argus server: {}", e);
                    Ok(()) // Allow offline mode
                }
            }
        }
        #[cfg(not(feature = "argus-live"))]
        {
            tracing::info!("Argus bridge initialized (mock mode)");
            Ok(())
        }
    }

    /// Set the Argus server URL.
    pub async fn set_server_url(&self, url: String) {
        *self.server_url.write().await = url;
    }

    /// Get the current Argus server URL.
    pub async fn get_server_url(&self) -> String {
        self.server_url.read().await.clone()
    }

    /// Get Argus server status.
    pub async fn get_status(&self) -> ArgusStatus {
        #[cfg(feature = "argus-live")]
        {
            let url = self.server_url.read().await.clone();
            let connected = self.client
                .get(format!("{}/api/v1/status/buildinfo", url))
                .send()
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false);

            let alerts = self.get_alerts().await.unwrap_or_default();
            let targets = self.get_targets().await.unwrap_or_default();
            let healthy = targets.iter().filter(|t| t.health == "up").count();

            ArgusStatus {
                connected,
                server_url: url,
                version: "unknown".to_string(),
                active_alerts: alerts.iter().filter(|a| a.state == "firing").count(),
                total_targets: targets.len(),
                healthy_targets: healthy,
                last_query: Some(std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64()),
            }
        }
        #[cfg(not(feature = "argus-live"))]
        {
            let state = self.mock_state.read().await;
            let healthy = state.targets.iter().filter(|t| t.health == "up").count();
            ArgusStatus {
                connected: true,
                server_url: self.server_url.read().await.clone(),
                version: "mock-1.0.0".to_string(),
                active_alerts: state.alerts.iter().filter(|a| a.state == "firing").count(),
                total_targets: state.targets.len(),
                healthy_targets: healthy,
                last_query: Some(std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64()),
            }
        }
    }

    /// Execute an instant query (PromQL).
    pub async fn query_instant(&self, query: &str) -> Result<QueryResult, String> {
        #[cfg(feature = "argus-live")]
        {
            let url = self.server_url.read().await.clone();
            let resp = self.client
                .get(format!("{}/api/v1/query", url))
                .query(&[("query", query)])
                .send()
                .await
                .map_err(|e| format!("Query failed: {}", e))?;

            if !resp.status().is_success() {
                return Err(format!("Query returned status: {}", resp.status()));
            }

            let body: serde_json::Value = resp.json().await
                .map_err(|e| format!("Failed to parse response: {}", e))?;

            // Parse Prometheus response format
            self.parse_query_response(&body)
        }
        #[cfg(not(feature = "argus-live"))]
        {
            // Generate mock data based on query
            Ok(self.generate_mock_instant_query(query).await)
        }
    }

    /// Execute a range query (PromQL).
    pub async fn query_range(
        &self,
        query: &str,
        start: i64,
        end: i64,
        step: u64,
    ) -> Result<QueryResult, String> {
        #[cfg(feature = "argus-live")]
        {
            let url = self.server_url.read().await.clone();
            let resp = self.client
                .get(format!("{}/api/v1/query_range", url))
                .query(&[
                    ("query", query),
                    ("start", &start.to_string()),
                    ("end", &end.to_string()),
                    ("step", &format!("{}s", step)),
                ])
                .send()
                .await
                .map_err(|e| format!("Range query failed: {}", e))?;

            if !resp.status().is_success() {
                return Err(format!("Range query returned status: {}", resp.status()));
            }

            let body: serde_json::Value = resp.json().await
                .map_err(|e| format!("Failed to parse response: {}", e))?;

            self.parse_query_response(&body)
        }
        #[cfg(not(feature = "argus-live"))]
        {
            Ok(self.generate_mock_range_query(query, start, end, step).await)
        }
    }

    #[cfg(feature = "argus-live")]
    fn parse_query_response(&self, body: &serde_json::Value) -> Result<QueryResult, String> {
        let status = body.get("status").and_then(|s| s.as_str()).unwrap_or("error");
        if status != "success" {
            let error = body.get("error").and_then(|e| e.as_str()).unwrap_or("Unknown error");
            return Err(format!("Query error: {}", error));
        }

        let data = body.get("data").ok_or("Missing data field")?;
        let result_type = data.get("resultType")
            .and_then(|t| t.as_str())
            .unwrap_or("vector")
            .to_string();

        let result = data.get("result").and_then(|r| r.as_array()).unwrap_or(&vec![]);
        let samples: Vec<MetricSample> = result.iter().filter_map(|item| {
            let metric: std::collections::HashMap<String, String> = item.get("metric")
                .and_then(|m| serde_json::from_value(m.clone()).ok())
                .unwrap_or_default();

            let value = item.get("value").and_then(|v| {
                let arr = v.as_array()?;
                let ts = arr.first()?.as_f64()?;
                let val = arr.get(1)?.as_str()?.to_string();
                Some((ts, val))
            });

            let values: Vec<(f64, String)> = item.get("values")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter().filter_map(|v| {
                        let inner = v.as_array()?;
                        let ts = inner.first()?.as_f64()?;
                        let val = inner.get(1)?.as_str()?.to_string();
                        Some((ts, val))
                    }).collect()
                })
                .unwrap_or_default();

            Some(MetricSample { metric, value, values })
        }).collect();

        Ok(QueryResult { result_type, data: samples })
    }

    #[cfg(not(feature = "argus-live"))]
    async fn generate_mock_instant_query(&self, query: &str) -> QueryResult {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        // Generate mock data based on common queries
        let samples = if query.contains("up") {
            vec![
                MetricSample {
                    metric: [("job".to_string(), "node-exporter".to_string())].into_iter().collect(),
                    value: Some((now, "1".to_string())),
                    values: vec![],
                },
            ]
        } else if query.contains("cpu") || query.contains("CPU") {
            vec![
                MetricSample {
                    metric: [("cpu".to_string(), "0".to_string())].into_iter().collect(),
                    value: Some((now, "45.2".to_string())),
                    values: vec![],
                },
            ]
        } else if query.contains("memory") || query.contains("mem") {
            vec![
                MetricSample {
                    metric: [("instance".to_string(), "localhost:9090".to_string())].into_iter().collect(),
                    value: Some((now, "8589934592".to_string())), // 8GB
                    values: vec![],
                },
            ]
        } else if query.contains("gpu") || query.contains("GPU") {
            vec![
                MetricSample {
                    metric: [("gpu".to_string(), "0".to_string())].into_iter().collect(),
                    value: Some((now, "78.5".to_string())),
                    values: vec![],
                },
            ]
        } else {
            vec![
                MetricSample {
                    metric: [("__name__".to_string(), "mock_metric".to_string())].into_iter().collect(),
                    value: Some((now, "42".to_string())),
                    values: vec![],
                },
            ]
        };

        QueryResult {
            result_type: "vector".to_string(),
            data: samples,
        }
    }

    #[cfg(not(feature = "argus-live"))]
    async fn generate_mock_range_query(&self, query: &str, start: i64, end: i64, step: u64) -> QueryResult {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let base_value: f64 = if query.contains("cpu") {
            45.0
        } else if query.contains("memory") || query.contains("mem") {
            70.0
        } else if query.contains("gpu") {
            65.0
        } else {
            50.0
        };

        let mut values = Vec::new();
        let mut ts = start as f64;
        while ts <= end as f64 {
            let noise: f64 = rng.gen_range(-10.0..10.0);
            let val = (base_value + noise).max(0.0).min(100.0);
            values.push((ts, format!("{:.2}", val)));
            ts += step as f64;
        }

        let samples = vec![
            MetricSample {
                metric: [("instance".to_string(), "localhost:9090".to_string())].into_iter().collect(),
                value: None,
                values,
            },
        ];

        QueryResult {
            result_type: "matrix".to_string(),
            data: samples,
        }
    }

    /// Get all alerts.
    pub async fn get_alerts(&self) -> Result<Vec<Alert>, String> {
        #[cfg(feature = "argus-live")]
        {
            let url = self.server_url.read().await.clone();
            let resp = self.client
                .get(format!("{}/api/v1/alerts", url))
                .send()
                .await
                .map_err(|e| format!("Failed to fetch alerts: {}", e))?;

            if !resp.status().is_success() {
                return Err(format!("Alerts request returned status: {}", resp.status()));
            }

            let body: serde_json::Value = resp.json().await
                .map_err(|e| format!("Failed to parse alerts: {}", e))?;

            // Parse alerts from response
            let alerts_data = body.get("data")
                .and_then(|d| d.get("alerts"))
                .and_then(|a| a.as_array())
                .unwrap_or(&vec![]);

            let alerts: Vec<Alert> = alerts_data.iter().filter_map(|a| {
                serde_json::from_value(a.clone()).ok()
            }).collect();

            Ok(alerts)
        }
        #[cfg(not(feature = "argus-live"))]
        {
            Ok(self.mock_state.read().await.alerts.clone())
        }
    }

    /// Get all scrape targets.
    pub async fn get_targets(&self) -> Result<Vec<Target>, String> {
        #[cfg(feature = "argus-live")]
        {
            let url = self.server_url.read().await.clone();
            let resp = self.client
                .get(format!("{}/api/v1/targets", url))
                .send()
                .await
                .map_err(|e| format!("Failed to fetch targets: {}", e))?;

            if !resp.status().is_success() {
                return Err(format!("Targets request returned status: {}", resp.status()));
            }

            let body: serde_json::Value = resp.json().await
                .map_err(|e| format!("Failed to parse targets: {}", e))?;

            // Parse targets from response
            let active = body.get("data")
                .and_then(|d| d.get("activeTargets"))
                .and_then(|t| t.as_array())
                .unwrap_or(&vec![]);

            let targets: Vec<Target> = active.iter().filter_map(|t| {
                let labels: std::collections::HashMap<String, String> = t.get("labels")
                    .and_then(|l| serde_json::from_value(l.clone()).ok())
                    .unwrap_or_default();

                Some(Target {
                    labels: labels.clone(),
                    scrape_url: t.get("scrapeUrl").and_then(|s| s.as_str())?.to_string(),
                    health: t.get("health").and_then(|h| h.as_str()).unwrap_or("unknown").to_string(),
                    last_scrape: t.get("lastScrape").and_then(|l| l.as_f64()).unwrap_or(0.0),
                    last_scrape_duration: t.get("lastScrapeDuration").and_then(|d| d.as_f64()).unwrap_or(0.0),
                    last_error: t.get("lastError").and_then(|e| e.as_str()).map(String::from),
                    job: labels.get("job").cloned().unwrap_or_default(),
                    instance: labels.get("instance").cloned().unwrap_or_default(),
                })
            }).collect();

            Ok(targets)
        }
        #[cfg(not(feature = "argus-live"))]
        {
            Ok(self.mock_state.read().await.targets.clone())
        }
    }
}

impl Default for ArgusBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bridge_creation() {
        let bridge = ArgusBridge::new();
        let _ = bridge.initialize().await;
        let status = bridge.get_status().await;
        assert!(status.connected || !cfg!(feature = "argus-live"));
    }

    #[tokio::test]
    async fn test_get_alerts_mock() {
        let bridge = ArgusBridge::new();
        let _ = bridge.initialize().await;

        #[cfg(not(feature = "argus-live"))]
        {
            let alerts = bridge.get_alerts().await.expect("Should get alerts");
            assert!(!alerts.is_empty());
            assert!(alerts.iter().any(|a| a.state == "firing"));
        }
    }

    #[tokio::test]
    async fn test_get_targets_mock() {
        let bridge = ArgusBridge::new();
        let _ = bridge.initialize().await;

        #[cfg(not(feature = "argus-live"))]
        {
            let targets = bridge.get_targets().await.expect("Should get targets");
            assert!(!targets.is_empty());
            assert!(targets.iter().any(|t| t.health == "up"));
        }
    }

    #[tokio::test]
    async fn test_query_instant_mock() {
        let bridge = ArgusBridge::new();
        let _ = bridge.initialize().await;

        #[cfg(not(feature = "argus-live"))]
        {
            let result = bridge.query_instant("up").await.expect("Should query");
            assert_eq!(result.result_type, "vector");
            assert!(!result.data.is_empty());
        }
    }

    #[tokio::test]
    async fn test_query_range_mock() {
        let bridge = ArgusBridge::new();
        let _ = bridge.initialize().await;

        #[cfg(not(feature = "argus-live"))]
        {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64;

            let result = bridge.query_range("cpu_usage", now - 3600, now, 60).await
                .expect("Should query range");
            assert_eq!(result.result_type, "matrix");
            assert!(!result.data.is_empty());
            assert!(!result.data[0].values.is_empty());
        }
    }
}
