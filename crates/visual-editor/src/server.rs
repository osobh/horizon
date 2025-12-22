use axum::{
    extract::{ws::WebSocketUpgrade, State},
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use serde_json::json;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};

// Import from agent-core
use exorust_agent_core::{Agent, AgentConfig, AgentId};

use crate::{
    api::{create_graphql_schema, GraphQLSchema},
    error::Result,
    rest_api_simple::create_rest_api_router,
    websocket::WebSocketHandler,
};

/// Simple agent store for development/testing
pub type AgentStore = Arc<RwLock<HashMap<String, Arc<Agent>>>>;

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    pub ws_handler: Arc<WebSocketHandler>,
    pub graphql_schema: GraphQLSchema,
    pub agents: AgentStore,
}

/// Create the main application router
pub fn create_app() -> Router {
    let ws_handler = Arc::new(WebSocketHandler::new());
    let graphql_schema = create_graphql_schema();
    let agents = Arc::new(RwLock::new(HashMap::new()));

    let state = AppState {
        ws_handler,
        graphql_schema,
        agents,
    };

    Router::new()
        .route("/", get(index_handler))
        .route("/health", get(health_handler))
        .route("/ws", get(websocket_handler))
        .route("/graphql", post(graphql_handler))
        .route("/graphql/playground", get(graphql_playground))
        // Merge REST API routes
        .merge(create_rest_api_router())
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state)
}

/// Health check endpoint
async fn health_handler() -> impl IntoResponse {
    Json(json!({
        "status": "healthy",
        "service": "stratoswarm-visual-editor",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

/// Index page handler
async fn index_handler() -> impl IntoResponse {
    Html(
        r#"
        <!DOCTYPE html>
        <html>
        <head>
            <title>StratoSwarm Infrastructure Intelligence Dashboard Backend</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .section { margin: 30px 0; }
                .endpoint { margin: 10px 0; }
                .method { font-weight: bold; color: #0066cc; }
                ul { list-style-type: none; padding: 0; }
                li { margin: 5px 0; }
                a { color: #0066cc; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <h1>StratoSwarm Infrastructure Intelligence Dashboard Backend</h1>
            <p>Comprehensive backend APIs for container management, infrastructure intelligence, and advanced features.</p>
            
            <div class="section">
                <h2>Core Endpoints</h2>
                <ul>
                    <li><a href="/health">Health Check</a></li>
                    <li><a href="/graphql/playground">GraphQL Playground</a></li>
                    <li>WebSocket: <code>/ws</code></li>
                </ul>
            </div>

            <div class="section">
                <h2>Container Management APIs</h2>
                <div class="endpoint"><span class="method">GET/POST</span> <code>/api/containers/templates</code> - Container templates CRUD</div>
                <div class="endpoint"><span class="method">POST</span> <code>/api/containers/validate</code> - Validate container configuration</div>
                <div class="endpoint"><span class="method">GET</span> <code>/api/containers/resources</code> - Available cluster resources</div>
                <div class="endpoint"><span class="method">POST</span> <code>/api/containers/estimate-cost</code> - Cost estimation</div>
                <div class="endpoint"><span class="method">GET</span> <code>/api/library/templates</code> - Browse template library</div>
                <div class="endpoint"><span class="method">GET/POST</span> <code>/api/pipelines</code> - Pipeline management</div>
                <div class="endpoint"><span class="method">GET</span> <code>/api/deployments</code> - Active deployments</div>
            </div>

            <div class="section">
                <h2>Infrastructure Intelligence APIs</h2>
                <div class="endpoint"><span class="method">GET</span> <code>/api/topology/network</code> - Network topology</div>
                <div class="endpoint"><span class="method">GET</span> <code>/api/topology/physical</code> - Physical infrastructure</div>
                <div class="endpoint"><span class="method">GET</span> <code>/api/swarmlets</code> - Swarmlet nodes status</div>
                <div class="endpoint"><span class="method">GET</span> <code>/api/system/overview</code> - System overview metrics</div>
                <div class="endpoint"><span class="method">GET</span> <code>/api/gpu/utilization</code> - GPU utilization metrics</div>
            </div>

            <div class="section">
                <h2>Advanced Features APIs</h2>
                <div class="endpoint"><span class="method">GET/PUT</span> <code>/api/config/system</code> - System configuration</div>
                <div class="endpoint"><span class="method">GET</span> <code>/api/security/compliance</code> - Compliance status</div>
                <div class="endpoint"><span class="method">GET</span> <code>/api/cost/current</code> - Current costs</div>
                <div class="endpoint"><span class="method">GET</span> <code>/api/cost/optimization</code> - Cost optimization recommendations</div>
            </div>

            <div class="section">
                <h2>Real-time Features (WebSocket)</h2>
                <div class="endpoint"><code>/ws/gpu-metrics</code> - Real-time GPU utilization</div>
                <div class="endpoint"><code>/ws/swarmlet-status</code> - Live swarmlet health updates</div>
                <div class="endpoint"><code>/ws/deployment-logs</code> - Live deployment log streaming</div>
                <div class="endpoint"><code>/ws/system-alerts</code> - Real-time alert notifications</div>
            </div>

            <div class="section">
                <h2>Integration Status</h2>
                <ul>
                    <li>✓ REST API framework with Axum</li>
                    <li>✓ GraphQL API with async-graphql</li>
                    <li>✓ WebSocket support for real-time features</li>
                    <li>✓ Comprehensive data models</li>
                    <li>⚠ Container Management APIs (sample data)</li>
                    <li>⚠ Infrastructure Intelligence integration (pending)</li>
                    <li>⚠ GPU Agents integration (pending)</li>
                    <li>⚠ Cost Optimization integration (pending)</li>
                    <li>⚠ Swarmlet integration (pending)</li>
                </ul>
            </div>
        </body>
        </html>
    "#,
    )
}

/// WebSocket upgrade handler
async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| async move {
        state.ws_handler.handle_connection(socket).await;
    })
}

/// GraphQL handler
async fn graphql_handler(
    State(state): State<AppState>,
    req: axum::extract::Json<async_graphql::Request>,
) -> impl IntoResponse {
    let response = state.graphql_schema.execute(req.0).await;
    Json(response)
}

/// GraphQL playground handler
async fn graphql_playground() -> impl IntoResponse {
    Html(async_graphql::http::playground_source(
        async_graphql::http::GraphQLPlaygroundConfig::new("/graphql"),
    ))
}

/// Run the server
pub async fn run_server(addr: &str) -> Result<()> {
    let app = create_app();

    tracing::info!("Starting Visual Editor server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| {
            tracing::error!("Failed to bind to address {}: {}", addr, e);
            crate::error::VisualEditorError::IoError(e)
        })?;

    tracing::info!("Successfully bound to {}", addr);
    tracing::info!("Visual Editor server is ready to accept connections");

    axum::serve(listener, app)
        .await
        .map_err(|e| {
            tracing::error!("Server error: {}", e);
            crate::error::VisualEditorError::IoError(e)
        })?;

    Ok(())
}
