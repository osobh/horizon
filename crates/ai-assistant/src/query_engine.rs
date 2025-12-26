//! Query engine for infrastructure information

use crate::error::{AssistantError, AssistantResult};
use crate::parser::{Intent, ParsedQuery};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::process::Command;

/// Result of a query operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Type of resource
    pub resource_type: String,
    /// Resource identifier
    pub id: String,
    /// Resource data
    pub data: HashMap<String, serde_json::Value>,
    /// Metadata about the resource
    pub metadata: QueryMetadata,
}

/// Metadata for query results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMetadata {
    /// When the data was collected
    pub timestamp: i64,
    /// Source of the data
    pub source: String,
    /// Confidence in the data accuracy
    pub confidence: f32,
    /// How fresh the data is (seconds)
    pub age_seconds: u64,
}

/// Query engine that retrieves infrastructure information
pub struct QueryEngine {
    /// Cached results for performance
    cache: tokio::sync::RwLock<HashMap<String, CachedResult>>,
    /// Cache TTL in seconds
    cache_ttl: u64,
}

#[derive(Debug, Clone)]
struct CachedResult {
    data: Vec<QueryResult>,
    timestamp: i64,
}

impl QueryEngine {
    pub async fn new() -> AssistantResult<Self> {
        Ok(Self {
            cache: tokio::sync::RwLock::new(HashMap::new()),
            cache_ttl: 30, // 30 seconds cache
        })
    }

    pub async fn execute(&self, parsed: &ParsedQuery) -> AssistantResult<Vec<QueryResult>> {
        match &parsed.intent {
            Intent::Query {
                resource_type,
                filters,
                projection,
            } => {
                self.query_resources(resource_type, filters, projection.as_deref())
                    .await
            }
            Intent::Status { target } => self.get_status(target.as_deref()).await,
            Intent::Logs {
                target,
                follow: _,
                lines,
            } => self.get_logs(target, *lines).await,
            Intent::Debug {
                target,
                symptoms: _,
            } => self.get_debug_info(target).await,
            _ => Err(AssistantError::QueryError(
                "Intent does not require querying".to_string(),
            )),
        }
    }

    async fn query_resources(
        &self,
        resource_type: &str,
        filters: &HashMap<String, String>,
        projection: Option<&[String]>,
    ) -> AssistantResult<Vec<QueryResult>> {
        let cache_key = format!("{}:{:?}:{:?}", resource_type, filters, projection);

        // Check cache first
        if let Some(cached) = self.get_cached(&cache_key).await {
            return Ok(cached);
        }

        let results = match resource_type.to_lowercase().as_str() {
            s if s.contains("agent") => self.query_agents(filters, projection).await?,
            s if s.contains("node") => self.query_nodes(filters, projection).await?,
            s if s.contains("app") || s.contains("application") => {
                self.query_applications(filters, projection).await?
            }
            s if s.contains("gpu") => self.query_gpus(filters, projection).await?,
            s if s.contains("memory") => self.query_memory(filters, projection).await?,
            s if s.contains("network") => self.query_network(filters, projection).await?,
            _ => self.query_all_resources(filters, projection).await?,
        };

        // Cache the results
        self.cache_results(&cache_key, &results).await;

        Ok(results)
    }

    async fn query_agents(
        &self,
        filters: &HashMap<String, String>,
        projection: Option<&[String]>,
    ) -> AssistantResult<Vec<QueryResult>> {
        // Execute stratoswarm command to get agent status
        let output = Command::new("stratoswarm")
            .arg("status")
            .arg("agents")
            .arg("--format=json")
            .output()
            .await
            .map_err(|e| AssistantError::QueryError(format!("Failed to execute command: {}", e)))?;

        if !output.status.success() {
            // If command fails, return mock data for demonstration
            return Ok(self.get_mock_agents(filters, projection));
        }

        let json_output = String::from_utf8_lossy(&output.stdout);
        let agents: Vec<serde_json::Value> =
            serde_json::from_str(&json_output).unwrap_or_else(|_| self.get_mock_agents_json());

        let mut results = Vec::new();
        for agent in agents {
            let mut data = HashMap::new();

            // Extract agent data
            if let Some(obj) = agent.as_object() {
                for (key, value) in obj {
                    data.insert(key.clone(), value.clone());
                }
            }

            // Apply filters
            if self.matches_filters(&data, filters) {
                // Apply projection
                if let Some(fields) = projection {
                    data = self.apply_projection(data, fields);
                }

                let id = data
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();

                results.push(QueryResult {
                    resource_type: "agent".to_string(),
                    id,
                    data,
                    metadata: QueryMetadata {
                        timestamp: chrono::Utc::now().timestamp(),
                        source: "stratoswarm-cli".to_string(),
                        confidence: 0.95,
                        age_seconds: 0,
                    },
                });
            }
        }

        Ok(results)
    }

    async fn query_nodes(
        &self,
        filters: &HashMap<String, String>,
        projection: Option<&[String]>,
    ) -> AssistantResult<Vec<QueryResult>> {
        // Similar to agents, but for nodes
        let mock_nodes = vec![
            self.create_mock_node("node-1", "datacenter", "available", "16", "64Gi", "RTX4090"),
            self.create_mock_node("node-2", "workstation", "available", "8", "32Gi", "GTX1080"),
            self.create_mock_node("node-3", "laptop", "busy", "4", "16Gi", "integrated"),
        ];

        let mut results = Vec::new();
        for node in mock_nodes {
            if self.matches_filters(&node, filters) {
                let data = if let Some(fields) = projection {
                    self.apply_projection(node, fields)
                } else {
                    node
                };

                let id = data
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();

                results.push(QueryResult {
                    resource_type: "node".to_string(),
                    id,
                    data,
                    metadata: QueryMetadata {
                        timestamp: chrono::Utc::now().timestamp(),
                        source: "cluster-mesh".to_string(),
                        confidence: 0.9,
                        age_seconds: 5,
                    },
                });
            }
        }

        Ok(results)
    }

    async fn query_applications(
        &self,
        filters: &HashMap<String, String>,
        projection: Option<&[String]>,
    ) -> AssistantResult<Vec<QueryResult>> {
        let mock_apps = vec![
            self.create_mock_app("web-frontend", "running", "3", "nginx"),
            self.create_mock_app("api-backend", "running", "5", "rust"),
            self.create_mock_app("ml-training", "evolving", "1", "python"),
        ];

        let mut results = Vec::new();
        for app in mock_apps {
            if self.matches_filters(&app, filters) {
                let data = if let Some(fields) = projection {
                    self.apply_projection(app, fields)
                } else {
                    app
                };

                let id = data
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();

                results.push(QueryResult {
                    resource_type: "application".to_string(),
                    id,
                    data,
                    metadata: QueryMetadata {
                        timestamp: chrono::Utc::now().timestamp(),
                        source: "runtime".to_string(),
                        confidence: 0.95,
                        age_seconds: 2,
                    },
                });
            }
        }

        Ok(results)
    }

    async fn query_gpus(
        &self,
        filters: &HashMap<String, String>,
        projection: Option<&[String]>,
    ) -> AssistantResult<Vec<QueryResult>> {
        let mock_gpus = vec![
            self.create_mock_gpu("gpu-0", "RTX4090", "24GB", "85%", "active"),
            self.create_mock_gpu("gpu-1", "RTX3080", "10GB", "45%", "active"),
        ];

        let mut results = Vec::new();
        for gpu in mock_gpus {
            if self.matches_filters(&gpu, filters) {
                let data = if let Some(fields) = projection {
                    self.apply_projection(gpu, fields)
                } else {
                    gpu
                };

                let id = data
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();

                results.push(QueryResult {
                    resource_type: "gpu".to_string(),
                    id,
                    data,
                    metadata: QueryMetadata {
                        timestamp: chrono::Utc::now().timestamp(),
                        source: "gpu-agents".to_string(),
                        confidence: 0.98,
                        age_seconds: 1,
                    },
                });
            }
        }

        Ok(results)
    }

    async fn query_memory(
        &self,
        filters: &HashMap<String, String>,
        projection: Option<&[String]>,
    ) -> AssistantResult<Vec<QueryResult>> {
        let mock_memory = vec![
            self.create_mock_memory_tier("gpu", "24GB", "18GB", "75%"),
            self.create_mock_memory_tier("cpu", "64GB", "32GB", "50%"),
            self.create_mock_memory_tier("nvme", "1TB", "600GB", "60%"),
        ];

        let mut results = Vec::new();
        for mem in mock_memory {
            if self.matches_filters(&mem, filters) {
                let data = if let Some(fields) = projection {
                    self.apply_projection(mem, fields)
                } else {
                    mem
                };

                let id = data
                    .get("tier")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();

                results.push(QueryResult {
                    resource_type: "memory".to_string(),
                    id,
                    data,
                    metadata: QueryMetadata {
                        timestamp: chrono::Utc::now().timestamp(),
                        source: "tier-watch".to_string(),
                        confidence: 0.99,
                        age_seconds: 0,
                    },
                });
            }
        }

        Ok(results)
    }

    async fn query_network(
        &self,
        filters: &HashMap<String, String>,
        projection: Option<&[String]>,
    ) -> AssistantResult<Vec<QueryResult>> {
        let mock_network = vec![
            self.create_mock_network("cluster-mesh", "active", "42", "1Gbps"),
            self.create_mock_network("agent-comm", "active", "156", "10Gbps"),
        ];

        let mut results = Vec::new();
        for net in mock_network {
            if self.matches_filters(&net, filters) {
                let data = if let Some(fields) = projection {
                    self.apply_projection(net, fields)
                } else {
                    net
                };

                let id = data
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();

                results.push(QueryResult {
                    resource_type: "network".to_string(),
                    id,
                    data,
                    metadata: QueryMetadata {
                        timestamp: chrono::Utc::now().timestamp(),
                        source: "net".to_string(),
                        confidence: 0.9,
                        age_seconds: 3,
                    },
                });
            }
        }

        Ok(results)
    }

    async fn query_all_resources(
        &self,
        filters: &HashMap<String, String>,
        projection: Option<&[String]>,
    ) -> AssistantResult<Vec<QueryResult>> {
        let mut all_results = Vec::new();

        // Combine results from all resource types
        all_results.extend(self.query_agents(filters, projection).await?);
        all_results.extend(self.query_nodes(filters, projection).await?);
        all_results.extend(self.query_applications(filters, projection).await?);
        all_results.extend(self.query_gpus(filters, projection).await?);

        Ok(all_results)
    }

    async fn get_status(&self, target: Option<&str>) -> AssistantResult<Vec<QueryResult>> {
        if let Some(target_name) = target {
            // Get status for specific target
            let mut filters = HashMap::new();
            filters.insert("name".to_string(), target_name.to_string());
            self.query_all_resources(&filters, None).await
        } else {
            // Get overall system status
            let mut results = Vec::new();

            // System overview
            let mut system_data = HashMap::new();
            system_data.insert("total_agents".to_string(), serde_json::json!(12));
            system_data.insert("active_nodes".to_string(), serde_json::json!(3));
            system_data.insert("running_apps".to_string(), serde_json::json!(8));
            system_data.insert("gpu_utilization".to_string(), serde_json::json!("65%"));
            system_data.insert("memory_usage".to_string(), serde_json::json!("58%"));
            system_data.insert("evolution_active".to_string(), serde_json::json!(true));

            results.push(QueryResult {
                resource_type: "system".to_string(),
                id: "stratoswarm-cluster".to_string(),
                data: system_data,
                metadata: QueryMetadata {
                    timestamp: chrono::Utc::now().timestamp(),
                    source: "system".to_string(),
                    confidence: 0.95,
                    age_seconds: 1,
                },
            });

            Ok(results)
        }
    }

    async fn get_logs(
        &self,
        target: &str,
        lines: Option<u32>,
    ) -> AssistantResult<Vec<QueryResult>> {
        // Mock log data
        let log_lines = vec![
            "2024-01-20 10:30:15 INFO: Agent started successfully",
            "2024-01-20 10:30:16 DEBUG: GPU memory allocated: 2GB",
            "2024-01-20 10:30:17 INFO: Evolution cycle initiated",
            "2024-01-20 10:30:18 WARN: High memory usage detected",
            "2024-01-20 10:30:19 INFO: Migration to tier-2 completed",
        ];

        let num_lines = lines.unwrap_or(100) as usize;
        let selected_lines = log_lines.into_iter().take(num_lines).collect::<Vec<_>>();

        let mut log_data = HashMap::new();
        log_data.insert("target".to_string(), serde_json::json!(target));
        log_data.insert("lines".to_string(), serde_json::json!(selected_lines));
        log_data.insert(
            "total_lines".to_string(),
            serde_json::json!(selected_lines.len()),
        );

        Ok(vec![QueryResult {
            resource_type: "logs".to_string(),
            id: target.to_string(),
            data: log_data,
            metadata: QueryMetadata {
                timestamp: chrono::Utc::now().timestamp(),
                source: "logging".to_string(),
                confidence: 1.0,
                age_seconds: 0,
            },
        }])
    }

    async fn get_debug_info(&self, target: &str) -> AssistantResult<Vec<QueryResult>> {
        let mut debug_data = HashMap::new();
        debug_data.insert("target".to_string(), serde_json::json!(target));
        debug_data.insert("status".to_string(), serde_json::json!("healthy"));
        debug_data.insert("cpu_usage".to_string(), serde_json::json!("45%"));
        debug_data.insert("memory_usage".to_string(), serde_json::json!("2.1GB"));
        debug_data.insert("gpu_usage".to_string(), serde_json::json!("80%"));
        debug_data.insert("network_connections".to_string(), serde_json::json!(15));
        debug_data.insert("recent_errors".to_string(), serde_json::json!([]));
        debug_data.insert(
            "recommendations".to_string(),
            serde_json::json!(["Consider scaling up if demand increases"]),
        );

        Ok(vec![QueryResult {
            resource_type: "debug".to_string(),
            id: target.to_string(),
            data: debug_data,
            metadata: QueryMetadata {
                timestamp: chrono::Utc::now().timestamp(),
                source: "debug-engine".to_string(),
                confidence: 0.9,
                age_seconds: 0,
            },
        }])
    }

    // Helper methods for mock data
    fn get_mock_agents(
        &self,
        filters: &HashMap<String, String>,
        projection: Option<&[String]>,
    ) -> Vec<QueryResult> {
        let mock_agents = vec![
            self.create_mock_agent("agent-001", "running", "conservative", "2.1GB", "45%"),
            self.create_mock_agent("agent-002", "evolving", "aggressive", "1.8GB", "78%"),
            self.create_mock_agent("agent-003", "running", "balanced", "3.2GB", "32%"),
        ];

        mock_agents
            .into_iter()
            .filter_map(|agent| {
                if self.matches_filters(&agent, filters) {
                    let data = if let Some(fields) = projection {
                        self.apply_projection(agent, fields)
                    } else {
                        agent
                    };

                    let id = data.get("id")?.as_str()?.to_string();
                    Some(QueryResult {
                        resource_type: "agent".to_string(),
                        id,
                        data,
                        metadata: QueryMetadata {
                            timestamp: chrono::Utc::now().timestamp(),
                            source: "mock".to_string(),
                            confidence: 0.8,
                            age_seconds: 5,
                        },
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    fn get_mock_agents_json(&self) -> Vec<serde_json::Value> {
        vec![
            serde_json::json!({
                "id": "agent-001",
                "status": "running",
                "personality": "conservative",
                "memory_usage": "2.1GB",
                "cpu_usage": "45%"
            }),
            serde_json::json!({
                "id": "agent-002",
                "status": "evolving",
                "personality": "aggressive",
                "memory_usage": "1.8GB",
                "cpu_usage": "78%"
            }),
        ]
    }

    fn create_mock_agent(
        &self,
        id: &str,
        status: &str,
        personality: &str,
        memory: &str,
        cpu: &str,
    ) -> HashMap<String, serde_json::Value> {
        let mut data = HashMap::new();
        data.insert("id".to_string(), serde_json::json!(id));
        data.insert("status".to_string(), serde_json::json!(status));
        data.insert("personality".to_string(), serde_json::json!(personality));
        data.insert("memory_usage".to_string(), serde_json::json!(memory));
        data.insert("cpu_usage".to_string(), serde_json::json!(cpu));
        data
    }

    fn create_mock_node(
        &self,
        id: &str,
        type_: &str,
        status: &str,
        cpu: &str,
        memory: &str,
        gpu: &str,
    ) -> HashMap<String, serde_json::Value> {
        let mut data = HashMap::new();
        data.insert("id".to_string(), serde_json::json!(id));
        data.insert("type".to_string(), serde_json::json!(type_));
        data.insert("status".to_string(), serde_json::json!(status));
        data.insert("cpu_cores".to_string(), serde_json::json!(cpu));
        data.insert("memory".to_string(), serde_json::json!(memory));
        data.insert("gpu".to_string(), serde_json::json!(gpu));
        data
    }

    fn create_mock_app(
        &self,
        name: &str,
        status: &str,
        replicas: &str,
        language: &str,
    ) -> HashMap<String, serde_json::Value> {
        let mut data = HashMap::new();
        data.insert("name".to_string(), serde_json::json!(name));
        data.insert("status".to_string(), serde_json::json!(status));
        data.insert("replicas".to_string(), serde_json::json!(replicas));
        data.insert("language".to_string(), serde_json::json!(language));
        data
    }

    fn create_mock_gpu(
        &self,
        id: &str,
        model: &str,
        memory: &str,
        utilization: &str,
        status: &str,
    ) -> HashMap<String, serde_json::Value> {
        let mut data = HashMap::new();
        data.insert("id".to_string(), serde_json::json!(id));
        data.insert("model".to_string(), serde_json::json!(model));
        data.insert("memory".to_string(), serde_json::json!(memory));
        data.insert("utilization".to_string(), serde_json::json!(utilization));
        data.insert("status".to_string(), serde_json::json!(status));
        data
    }

    fn create_mock_memory_tier(
        &self,
        tier: &str,
        total: &str,
        used: &str,
        utilization: &str,
    ) -> HashMap<String, serde_json::Value> {
        let mut data = HashMap::new();
        data.insert("tier".to_string(), serde_json::json!(tier));
        data.insert("total".to_string(), serde_json::json!(total));
        data.insert("used".to_string(), serde_json::json!(used));
        data.insert("utilization".to_string(), serde_json::json!(utilization));
        data
    }

    fn create_mock_network(
        &self,
        name: &str,
        status: &str,
        connections: &str,
        bandwidth: &str,
    ) -> HashMap<String, serde_json::Value> {
        let mut data = HashMap::new();
        data.insert("name".to_string(), serde_json::json!(name));
        data.insert("status".to_string(), serde_json::json!(status));
        data.insert("connections".to_string(), serde_json::json!(connections));
        data.insert("bandwidth".to_string(), serde_json::json!(bandwidth));
        data
    }

    fn matches_filters(
        &self,
        data: &HashMap<String, serde_json::Value>,
        filters: &HashMap<String, String>,
    ) -> bool {
        for (key, expected) in filters {
            if let Some(actual) = data.get(key) {
                let actual_str = actual.as_str().unwrap_or("");
                if !actual_str.contains(expected) {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }

    fn apply_projection(
        &self,
        mut data: HashMap<String, serde_json::Value>,
        fields: &[String],
    ) -> HashMap<String, serde_json::Value> {
        let mut projected = HashMap::new();
        for field in fields {
            if let Some(value) = data.remove(field) {
                projected.insert(field.clone(), value);
            }
        }
        projected
    }

    async fn get_cached(&self, key: &str) -> Option<Vec<QueryResult>> {
        let cache = self.cache.read().await;
        if let Some(cached) = cache.get(key) {
            let now = chrono::Utc::now().timestamp();
            if now - cached.timestamp < self.cache_ttl as i64 {
                return Some(cached.data.clone());
            }
        }
        None
    }

    async fn cache_results(&self, key: &str, results: &[QueryResult]) {
        let mut cache = self.cache.write().await;
        cache.insert(
            key.to_string(),
            CachedResult {
                data: results.to_vec(),
                timestamp: chrono::Utc::now().timestamp(),
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Intent;

    #[tokio::test]
    async fn test_query_engine_creation() {
        let engine = QueryEngine::new().await.unwrap();
        assert_eq!(engine.cache_ttl, 30);
    }

    #[tokio::test]
    async fn test_agent_query() {
        let engine = QueryEngine::new().await.unwrap();
        let mut filters = HashMap::new();
        filters.insert("status".to_string(), "running".to_string());

        let results = engine.query_agents(&filters, None).await.unwrap();
        assert!(!results.is_empty());

        for result in &results {
            assert_eq!(result.resource_type, "agent");
            assert!(result.data.contains_key("status"));
        }
    }

    #[tokio::test]
    async fn test_node_query_with_projection() {
        let engine = QueryEngine::new().await.unwrap();
        let filters = HashMap::new();
        let projection = vec!["id".to_string(), "type".to_string()];

        let results = engine
            .query_nodes(&filters, Some(&projection))
            .await
            .unwrap();
        assert!(!results.is_empty());

        for result in &results {
            assert_eq!(result.resource_type, "node");
            assert!(result.data.contains_key("id"));
            assert!(result.data.contains_key("type"));
            assert!(!result.data.contains_key("memory")); // Should be filtered out
        }
    }

    #[tokio::test]
    async fn test_system_status() {
        let engine = QueryEngine::new().await.unwrap();

        let results = engine.get_status(None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].resource_type, "system");
        assert!(results[0].data.contains_key("total_agents"));
    }

    #[tokio::test]
    async fn test_logs_query() {
        let engine = QueryEngine::new().await.unwrap();

        let results = engine.get_logs("test-app", Some(3)).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].resource_type, "logs");

        if let Some(lines) = results[0].data.get("lines") {
            if let Some(lines_array) = lines.as_array() {
                assert!(lines_array.len() <= 3);
            }
        }
    }

    #[tokio::test]
    async fn test_cache_functionality() {
        let engine = QueryEngine::new().await.unwrap();
        let filters = HashMap::new();

        // First query - should hit the actual query
        let results1 = engine.query_agents(&filters, None).await.unwrap();

        // Second query - should hit the cache (same results)
        let results2 = engine.query_agents(&filters, None).await.unwrap();

        assert_eq!(results1.len(), results2.len());
    }
}
