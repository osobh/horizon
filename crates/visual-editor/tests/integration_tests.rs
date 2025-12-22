use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::json;
use stratoswarm_visual_editor::{
    server::create_app,
    topology::TopologyValidator,
    websocket::{WebSocketHandler, WebSocketMessage},
};
use tower::ServiceExt;

#[tokio::test]
async fn test_server_startup() {
    let app = create_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_graphql_endpoint() {
    let app = create_app();

    let query = json!({
        "query": "{ topologies { id name nodes { id type } } }"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/graphql")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&query).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_websocket_message_serialization() {
    let msg = WebSocketMessage::TopologyUpdate {
        topology_id: "test-123".to_string(),
        data: json!({
            "nodes": [],
            "edges": []
        }),
    };

    let serialized = serde_json::to_string(&msg).unwrap();
    let deserialized: WebSocketMessage = serde_json::from_str(&serialized).unwrap();

    match deserialized {
        WebSocketMessage::TopologyUpdate { topology_id, .. } => {
            assert_eq!(topology_id, "test-123");
        }
        _ => panic!("Wrong message type"),
    }
}

#[tokio::test]
async fn test_topology_validation() {
    let validator = TopologyValidator::new();

    let valid_topology = json!({
        "id": "test-topology",
        "name": "Test Topology",
        "nodes": [
            {
                "id": "node1",
                "type": "gpu",
                "position": { "x": 0, "y": 0 }
            },
            {
                "id": "node2",
                "type": "cpu",
                "position": { "x": 100, "y": 100 }
            }
        ],
        "edges": [
            {
                "id": "edge1",
                "source": "node1",
                "target": "node2",
                "bandwidth": 10000
            }
        ]
    });

    let valid_result = validator.validate(&valid_topology).unwrap();
    assert!(valid_result.is_valid);

    let invalid_topology = json!({
        "id": "test-topology",
        "name": "Test Topology",
        "nodes": [],
        "edges": [
            {
                "id": "edge1",
                "source": "nonexistent",
                "target": "alsonothere",
                "bandwidth": 10000
            }
        ]
    });

    let invalid_result = validator.validate(&invalid_topology).unwrap();
    assert!(!invalid_result.is_valid);
    assert!(!invalid_result.errors.is_empty());
}

#[tokio::test]
async fn test_websocket_handler() {
    let handler = WebSocketHandler::new();

    // Test connection handling
    let client_id = handler.add_client().await;
    assert!(!client_id.is_empty());

    // Test message broadcasting
    let msg = WebSocketMessage::TopologyUpdate {
        topology_id: "test-123".to_string(),
        data: json!({"test": true}),
    };

    handler.broadcast(msg).await;

    // Test client removal
    handler.remove_client(&client_id).await;
}

#[tokio::test]
async fn test_api_topology_crud() {
    let app = create_app();

    // Create topology
    let create_mutation = json!({
        "query": r#"
            mutation CreateTopology($input: CreateTopologyInput!) {
                createTopology(input: $input) {
                    id
                    name
                }
            }
        "#,
        "variables": {
            "input": {
                "name": "New Topology",
                "description": "Test topology"
            }
        }
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/graphql")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&create_mutation).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_error_handling() {
    use stratoswarm_visual_editor::error::{Result, VisualEditorError};

    fn test_function() -> Result<()> {
        Err(VisualEditorError::ValidationError(
            "Invalid topology".to_string(),
        ))
    }

    match test_function() {
        Err(VisualEditorError::ValidationError(msg)) => {
            assert_eq!(msg, "Invalid topology");
        }
        _ => panic!("Expected ValidationError"),
    }
}
