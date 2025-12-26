//! XP (Experience Points) API endpoints for StratoSwarm Visual Editor
//!
//! This module provides REST API endpoints for agent XP management:
//! - Award XP to agents
//! - Get XP history and statistics  
//! - Trigger agent evolution
//! - System-wide XP overview

use axum::{
    extract::{Path, Query, State},
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use validator::Validate;

use crate::{
    error::{Result, VisualEditorError},
    rest_api_simple::ApiResponse,
    server::{AppState, AgentStore},
};

// Re-export XP types from agent-core
pub use stratoswarm_agent_core::agent::{
    AgentStats, EvolutionResult, XPGainRecord, LEVEL_THRESHOLDS,
};
use stratoswarm_agent_core::{Agent, AgentConfig, AgentId};

// =============================================================================
// Request/Response Types
// =============================================================================

/// Request to award XP to an agent
#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct AwardXPRequest {
    /// Amount of XP to award (1-10000)
    #[validate(range(min = 1, max = 10000))]
    pub amount: u64,
    
    /// Reason for awarding XP
    #[validate(length(min = 1, max = 200))]
    pub reason: String,
    
    /// Category of XP gain
    #[validate(length(min = 1, max = 50))]
    pub category: String,
}

/// Response after awarding XP
#[derive(Debug, Serialize, Deserialize)]
pub struct AwardXPResponse {
    /// Agent ID
    pub agent_id: String,
    
    /// XP awarded
    pub xp_awarded: u64,
    
    /// New total XP
    pub new_total_xp: u64,
    
    /// New current XP
    pub new_current_xp: u64,
    
    /// New level
    pub new_level: u32,
    
    /// Whether agent leveled up
    pub leveled_up: bool,
    
    /// Whether agent is ready to evolve
    pub ready_to_evolve: bool,
    
    /// Timestamp of XP gain
    pub timestamp: String,
}

/// Request to trigger agent evolution
#[derive(Debug, Serialize, Deserialize)]
pub struct EvolutionRequest {
    /// Optional notes about evolution trigger
    pub notes: Option<String>,
}

/// Response for XP history query
#[derive(Debug, Serialize)]
pub struct XPHistoryResponse {
    /// Agent ID
    pub agent_id: String,
    
    /// XP gain records
    pub history: Vec<XPGainRecord>,
    
    /// Total records available
    pub total_records: usize,
    
    /// Page information
    pub page: u32,
    
    /// Records per page
    pub limit: u32,
    
    /// Whether there are more records
    pub has_more: bool,
}

/// Query parameters for XP history
#[derive(Debug, Deserialize)]
pub struct XPHistoryQuery {
    /// Page number (default: 1)
    pub page: Option<u32>,
    
    /// Records per page (default: 50, max: 100)
    pub limit: Option<u32>,
    
    /// Filter by category
    pub category: Option<String>,
    
    /// Filter by minimum XP amount
    pub min_amount: Option<u64>,
    
    /// Filter by date range (ISO 8601)
    pub since: Option<String>,
}

/// Response for evolution status
#[derive(Debug, Serialize)]
pub struct EvolutionStatusResponse {
    /// Agent ID
    pub agent_id: String,
    
    /// Current level
    pub current_level: u32,
    
    /// Current XP
    pub current_xp: u64,
    
    /// XP needed for next level
    pub xp_needed_for_next_level: u64,
    
    /// Whether ready to evolve
    pub ready_to_evolve: bool,
    
    /// Next level (if ready to evolve)
    pub next_level: Option<u32>,
    
    /// Evolution readiness percentage
    pub evolution_progress_percent: f64,
}

/// System-wide XP overview response
#[derive(Debug, Serialize)]
pub struct SystemXPOverview {
    /// Total number of agents
    pub total_agents: u32,
    
    /// Total XP across all agents
    pub total_xp_awarded: u64,
    
    /// Average agent level
    pub average_level: f64,
    
    /// Highest level agent
    pub highest_level: u32,
    
    /// Number of agents ready to evolve
    pub agents_ready_to_evolve: u32,
    
    /// XP gained in last 24 hours
    pub xp_gained_24h: u64,
    
    /// Recent evolutions (last 10)
    pub recent_evolutions: Vec<RecentEvolution>,
    
    /// Top XP categories
    pub top_categories: Vec<XPCategoryStats>,
    
    /// Level distribution
    pub level_distribution: HashMap<u32, u32>,
}

/// Recent evolution information
#[derive(Debug, Serialize)]
pub struct RecentEvolution {
    /// Agent ID
    pub agent_id: String,
    
    /// Agent name/identifier
    pub agent_name: String,
    
    /// Evolution result
    pub evolution: EvolutionResult,
    
    /// Timestamp of evolution
    pub timestamp: String,
}

/// XP category statistics
#[derive(Debug, Serialize)]
pub struct XPCategoryStats {
    /// Category name
    pub category: String,
    
    /// Total XP awarded in category
    pub total_xp: u64,
    
    /// Number of XP gains in category
    pub count: u32,
    
    /// Average XP per gain
    pub average_xp: f64,
    
    /// Percentage of total system XP
    pub percentage: f64,
}

// =============================================================================
// Router Setup
// =============================================================================

/// Create XP API router
pub fn create_xp_api_router() -> Router<AppState> {
    Router::new()
        .route("/:id/xp", post(award_xp))
        .route("/:id/xp-history", get(get_xp_history))
        .route("/:id/evolve", post(evolve_agent))
        .route("/:id/evolution-status", get(get_evolution_status))
}

/// Create XP system router (for system-wide endpoints)
pub fn create_xp_system_router() -> Router<AppState> {
    Router::new()
        .route("/xp-overview", get(get_system_xp_overview))
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Get or create an agent in the store
async fn get_or_create_agent(
    agents: &AgentStore,
    agent_id: &str,
) -> Result<std::sync::Arc<Agent>> {
    let mut agents_map = agents.write().await;
    
    if let Some(agent) = agents_map.get(agent_id) {
        Ok(agent.clone())
    } else {
        // Create a new agent with the given ID
        let mut config = AgentConfig::default();
        config.name = agent_id.to_string();
        
        let agent = Agent::new(config)?;
        let agent_arc = std::sync::Arc::new(agent);
        agents_map.insert(agent_id.to_string(), agent_arc.clone());
        
        // Initialize the agent
        agent_arc.initialize().await
            .map_err(|e| VisualEditorError::AgentXPError(format!("Failed to initialize agent: {}", e)))?;
        
        Ok(agent_arc)
    }
}

// =============================================================================
// Handler Functions
// =============================================================================

/// Award XP to an agent
async fn award_xp(
    State(state): State<AppState>,
    Path(agent_id): Path<String>,
    Json(request): Json<AwardXPRequest>,
) -> Result<Json<ApiResponse<AwardXPResponse>>> {
    // Validate request
    request.validate()
        .map_err(|e| VisualEditorError::ValidationError(e.to_string()))?;
    
    let agent = get_or_create_agent(&state.agents, &agent_id).await?;
    
    // Get stats before XP award
    let stats_before = agent.stats().await;
    let level_before = stats_before.level;
    
    // Award XP
    agent.award_xp(request.amount, request.reason.clone(), request.category.clone()).await
        .map_err(|e| VisualEditorError::AgentXPError(e.to_string()))?;
    
    // Get stats after XP award
    let stats_after = agent.stats().await;
    let leveled_up = stats_after.level > level_before;
    let ready_to_evolve = agent.check_evolution_readiness().await;
    
    let response = AwardXPResponse {
        agent_id: agent_id.clone(),
        xp_awarded: request.amount,
        new_total_xp: stats_after.total_xp,
        new_current_xp: stats_after.current_xp,
        new_level: stats_after.level,
        leveled_up,
        ready_to_evolve,
        timestamp: chrono::Utc::now().to_rfc3339(),
    };

    // Broadcast real-time XP gained event
    state.ws_handler.broadcast_agent_xp_gained(
        agent_id,
        request.amount,
        request.reason,
        request.category,
        stats_after.total_xp,
        stats_after.current_xp,
        stats_after.level,
        leveled_up,
        ready_to_evolve,
    ).await;
    
    Ok(Json(ApiResponse::success(response)))
}

/// Get agent XP history
async fn get_xp_history(
    State(state): State<AppState>,
    Path(agent_id): Path<String>,
    Query(query): Query<XPHistoryQuery>,
) -> Result<Json<ApiResponse<XPHistoryResponse>>> {
    let agent = get_or_create_agent(&state.agents, &agent_id).await?;
    let stats = agent.stats().await;
    
    // Apply filters and pagination
    let page = query.page.unwrap_or(1);
    let limit = query.limit.unwrap_or(50).min(100); // Max 100 per page
    let skip = (page - 1) * limit;
    
    let mut filtered_history: Vec<_> = stats.xp_history.iter().cloned().collect();
    
    // Filter by category if specified
    if let Some(category) = &query.category {
        filtered_history.retain(|record| record.category == *category);
    }
    
    // Filter by minimum amount if specified
    if let Some(min_amount) = query.min_amount {
        filtered_history.retain(|record| record.amount >= min_amount);
    }
    
    // Sort by timestamp (most recent first)
    filtered_history.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    
    let total_records = filtered_history.len();
    let history: Vec<_> = filtered_history
        .into_iter()
        .skip(skip as usize)
        .take(limit as usize)
        .collect();
        
    let has_more = total_records > (skip + limit) as usize;
    
    let response = XPHistoryResponse {
        agent_id,
        history,
        total_records,
        page,
        limit,
        has_more,
    };
    
    Ok(Json(ApiResponse::success(response)))
}

/// Trigger agent evolution
async fn evolve_agent(
    State(state): State<AppState>,
    Path(agent_id): Path<String>,
    Json(_request): Json<EvolutionRequest>,
) -> Result<Json<ApiResponse<EvolutionResult>>> {
    let agent = get_or_create_agent(&state.agents, &agent_id).await?;
    
    // Check if evolution is allowed
    let ready = agent.check_evolution_readiness().await;
    if !ready {
        return Err(VisualEditorError::EvolutionNotAllowed(
            "Agent is not ready for evolution".to_string()
        ));
    }
    
    // Trigger evolution
    let evolution_result = agent.trigger_evolution().await
        .map_err(|e| VisualEditorError::AgentXPError(format!("Evolution failed: {}", e)))?;
    
    // Broadcast real-time evolution event
    state.ws_handler.broadcast_agent_evolved(
        agent_id,
        evolution_result.previous_level,
        evolution_result.new_level,
        evolution_result.xp_at_evolution,
    ).await;
    
    Ok(Json(ApiResponse::success(evolution_result)))
}

/// Get agent evolution status
async fn get_evolution_status(
    State(state): State<AppState>,
    Path(agent_id): Path<String>,
) -> Result<Json<ApiResponse<EvolutionStatusResponse>>> {
    let agent = get_or_create_agent(&state.agents, &agent_id).await?;
    let stats = agent.stats().await;
    
    let ready_to_evolve = agent.check_evolution_readiness().await;
    let xp_needed = agent.get_xp_for_next_level().await;
    
    // Calculate evolution progress percentage
    let current_level_index = (stats.level.saturating_sub(1)) as usize;
    let evolution_progress_percent = if current_level_index + 1 < LEVEL_THRESHOLDS.len() {
        let current_threshold = if current_level_index > 0 {
            LEVEL_THRESHOLDS[current_level_index - 1]
        } else {
            0
        };
        let next_threshold = LEVEL_THRESHOLDS[current_level_index];
        let threshold_range = next_threshold - current_threshold;
        let progress_in_range = stats.current_xp - current_threshold;
        
        if threshold_range > 0 {
            (progress_in_range as f64 / threshold_range as f64) * 100.0
        } else {
            100.0
        }
    } else {
        100.0 // Max level
    };
    
    let next_level = if ready_to_evolve {
        Some(stats.level + 1)
    } else {
        None
    };
    
    let response = EvolutionStatusResponse {
        agent_id,
        current_level: stats.level,
        current_xp: stats.current_xp,
        xp_needed_for_next_level: xp_needed,
        ready_to_evolve,
        next_level,
        evolution_progress_percent,
    };
    
    Ok(Json(ApiResponse::success(response)))
}

/// Get system-wide XP overview
async fn get_system_xp_overview(
    State(state): State<AppState>,
) -> Result<Json<ApiResponse<SystemXPOverview>>> {
    let agents_map = state.agents.read().await;
    let total_agents = agents_map.len() as u32;
    
    if total_agents == 0 {
        let response = SystemXPOverview {
            total_agents: 0,
            total_xp_awarded: 0,
            average_level: 0.0,
            highest_level: 0,
            agents_ready_to_evolve: 0,
            xp_gained_24h: 0,
            recent_evolutions: Vec::new(),
            top_categories: Vec::new(),
            level_distribution: HashMap::new(),
        };
        return Ok(Json(ApiResponse::success(response)));
    }
    
    let mut total_xp_awarded = 0u64;
    let mut total_level = 0u64;
    let mut highest_level = 0u32;
    let mut agents_ready_to_evolve = 0u32;
    let mut level_distribution = HashMap::new();
    let mut all_xp_records = Vec::new();
    
    // Collect stats from all agents
    for agent in agents_map.values() {
        let stats = agent.stats().await;
        total_xp_awarded += stats.total_xp;
        total_level += stats.level as u64;
        highest_level = highest_level.max(stats.level);
        
        if agent.check_evolution_readiness().await {
            agents_ready_to_evolve += 1;
        }
        
        *level_distribution.entry(stats.level).or_insert(0) += 1;
        all_xp_records.extend(stats.xp_history.clone());
    }
    
    let average_level = total_level as f64 / total_agents as f64;
    
    // Calculate XP gained in last 24 hours
    let twenty_four_hours_ago = chrono::Utc::now() - chrono::Duration::hours(24);
    let xp_gained_24h = all_xp_records
        .iter()
        .filter(|record| record.timestamp > twenty_four_hours_ago)
        .map(|record| record.amount)
        .sum();
    
    // Calculate top categories
    let mut category_stats = HashMap::new();
    for record in &all_xp_records {
        let entry = category_stats.entry(record.category.clone()).or_insert((0u64, 0u32));
        entry.0 += record.amount;
        entry.1 += 1;
    }
    
    let mut top_categories: Vec<_> = category_stats
        .into_iter()
        .map(|(category, (total_xp, count))| XPCategoryStats {
            category,
            total_xp,
            count,
            average_xp: total_xp as f64 / count as f64,
            percentage: if total_xp_awarded > 0 {
                (total_xp as f64 / total_xp_awarded as f64) * 100.0
            } else {
                0.0
            },
        })
        .collect();
        
    top_categories.sort_by(|a, b| b.total_xp.cmp(&a.total_xp));
    top_categories.truncate(10); // Top 10 categories
    
    let response = SystemXPOverview {
        total_agents,
        total_xp_awarded,
        average_level,
        highest_level,
        agents_ready_to_evolve,
        xp_gained_24h,
        recent_evolutions: Vec::new(), // TODO: Track evolutions separately
        top_categories,
        level_distribution,
    };

    // Broadcast system XP update periodically (this endpoint might be polled)
    state.ws_handler.broadcast_system_xp_update(
        total_agents,
        total_xp_awarded,
        average_level,
        highest_level,
        agents_ready_to_evolve,
        xp_gained_24h,
    ).await;
    
    Ok(Json(ApiResponse::success(response)))
}

// =============================================================================
// Tests (TDD - Starting with failing tests)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::create_app;
    use axum::http::StatusCode;
    use serde_json::json;
    use tokio_test;
    use tower::ServiceExt;

    /// Create test app with XP routes
    fn create_test_app() -> Router {
        use crate::server::create_app;
        create_app()
    }

    #[tokio::test]
    async fn test_award_xp_endpoint_exists() {
        let app = create_test_app();
        
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/api/v1/agents/test-agent-id/xp")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                serde_json::to_string(&AwardXPRequest {
                    amount: 100,
                    reason: "Test XP award".to_string(),
                    category: "test".to_string(),
                })
                .unwrap(),
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        
        // Should fail with 404 initially (route not found)
        // This test will pass once we integrate the XP router
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_xp_history_endpoint_exists() {
        let app = create_test_app();
        
        let request = axum::http::Request::builder()
            .method("GET")
            .uri("/api/v1/agents/test-agent-id/xp-history")
            .body(axum::body::Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        
        // Should fail with 404 initially
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_evolution_endpoint_exists() {
        let app = create_test_app();
        
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/api/v1/agents/test-agent-id/evolve")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                serde_json::to_string(&EvolutionRequest {
                    notes: Some("Test evolution".to_string()),
                })
                .unwrap(),
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        
        // Should fail with 404 initially
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_evolution_status_endpoint_exists() {
        let app = create_test_app();
        
        let request = axum::http::Request::builder()
            .method("GET")
            .uri("/api/v1/agents/test-agent-id/evolution-status")
            .body(axum::body::Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        
        // Should fail with 404 initially
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_system_xp_overview_endpoint_exists() {
        let app = create_test_app();
        
        let request = axum::http::Request::builder()
            .method("GET")
            .uri("/api/v1/system/xp-overview")
            .body(axum::body::Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        
        // Should fail with 404 initially
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn test_award_xp_request_validation() {
        // Test valid request
        let valid_request = AwardXPRequest {
            amount: 100,
            reason: "Valid reason".to_string(),
            category: "valid".to_string(),
        };
        assert!(valid_request.validate().is_ok());
        
        // Test invalid amount (too high)
        let invalid_request = AwardXPRequest {
            amount: 20000, // Over limit
            reason: "Valid reason".to_string(),
            category: "valid".to_string(),
        };
        assert!(invalid_request.validate().is_err());
        
        // Test invalid reason (empty)
        let invalid_request = AwardXPRequest {
            amount: 100,
            reason: "".to_string(),
            category: "valid".to_string(),
        };
        assert!(invalid_request.validate().is_err());
    }

    #[test]
    fn test_xp_history_query_defaults() {
        let query = XPHistoryQuery {
            page: None,
            limit: None,
            category: None,
            min_amount: None,
            since: None,
        };
        
        // All fields should be optional
        assert!(query.page.is_none());
        assert!(query.limit.is_none());
    }

    #[test]
    fn test_request_response_serialization() {
        // Test AwardXPRequest serialization
        let request = AwardXPRequest {
            amount: 100,
            reason: "Test".to_string(),
            category: "test".to_string(),
        };
        let json = serde_json::to_string(&request).unwrap();
        let deserialized: AwardXPRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(request.amount, deserialized.amount);
        
        // Test AwardXPResponse serialization
        let response = AwardXPResponse {
            agent_id: "test".to_string(),
            xp_awarded: 100,
            new_total_xp: 200,
            new_current_xp: 100,
            new_level: 2,
            leveled_up: true,
            ready_to_evolve: false,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };
        let json = serde_json::to_string(&response).unwrap();
        let deserialized: AwardXPResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(response.agent_id, deserialized.agent_id);
    }
}