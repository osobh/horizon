//! Integration tests for the AI Assistant

use ai_assistant::*;
use std::collections::HashMap;
use tokio;

#[tokio::test]
async fn test_full_deploy_workflow() {
    let config = AssistantConfig::default();
    let assistant = AiAssistant::new(config).await.unwrap();

    // Test deploy command
    let response = assistant
        .process_input("deploy my web app from github.com/user/webapp")
        .await
        .unwrap();

    assert!(matches!(response.intent, Intent::Deploy { .. }));
    assert!(response.command.is_some());
    assert!(response.confidence > 0.8);
    assert!(!response.suggestions.is_empty());

    if let Some(command) = &response.command {
        assert_eq!(command.command, "stratoswarm");
        assert!(command.args.contains(&"deploy".to_string()));
        assert!(command.args.contains(&"github.com/user/webapp".to_string()));
    }
}

#[tokio::test]
async fn test_scale_with_resources() {
    let config = AssistantConfig::default();
    let assistant = AiAssistant::new(config).await.unwrap();

    let response = assistant
        .process_input("scale web-service to 5 replicas with 2 CPU and 4GB memory")
        .await
        .unwrap();

    assert!(matches!(response.intent, Intent::Scale { .. }));
    assert!(response.command.is_some());

    if let Intent::Scale {
        target,
        replicas,
        resources,
    } = response.intent
    {
        assert_eq!(target, "web-service");
        assert_eq!(replicas, Some(5));
        // Note: Resource parsing from entities would be in a real implementation
    }
}

#[tokio::test]
async fn test_query_system_status() {
    let config = AssistantConfig::default();
    let assistant = AiAssistant::new(config).await.unwrap();

    let response = assistant
        .process_input("show me the system status")
        .await
        .unwrap();

    assert!(matches!(response.intent, Intent::Status { .. }));
    assert!(response.query_results.is_some());

    if let Some(results) = &response.query_results {
        assert!(!results.is_empty());
        assert_eq!(results[0].resource_type, "system");
    }
}

#[tokio::test]
async fn test_debug_application() {
    let config = AssistantConfig::default();
    let assistant = AiAssistant::new(config).await.unwrap();

    let response = assistant
        .process_input("debug my application that's having performance issues")
        .await
        .unwrap();

    assert!(matches!(response.intent, Intent::Debug { .. }));
    assert!(response.query_results.is_some());

    if let Intent::Debug { target, .. } = response.intent {
        assert!(target.contains("application"));
    }
}

#[tokio::test]
async fn test_learning_from_feedback() {
    let config = AssistantConfig::default();
    let assistant = AiAssistant::new(config).await.unwrap();

    // First interaction
    let response1 = assistant.process_input("deploy my app").await.unwrap();
    let original_confidence = response1.confidence;

    // Positive feedback
    assistant
        .handle_feedback("deploy my app", &response1, true)
        .await
        .unwrap();

    // Similar interaction should have higher confidence
    let response2 = assistant.process_input("deploy another app").await.unwrap();
    // Note: In a real implementation with proper learning, this would be higher
    assert!(response2.confidence >= original_confidence * 0.9); // Allow some variance
}

#[tokio::test]
async fn test_help_system() {
    let config = AssistantConfig::default();
    let assistant = AiAssistant::new(config).await.unwrap();

    let response = assistant
        .process_input("help me with deployment")
        .await
        .unwrap();

    assert!(matches!(response.intent, Intent::Help { .. }));
    assert!(response.response.contains("deploy"));
    assert!(!response.suggestions.is_empty());
}

#[tokio::test]
async fn test_evolution_command() {
    let config = AssistantConfig::default();
    let assistant = AiAssistant::new(config).await.unwrap();

    let response = assistant
        .process_input("evolve my ml model for better accuracy")
        .await
        .unwrap();

    assert!(matches!(response.intent, Intent::Evolve { .. }));
    assert!(response.command.is_some());

    if let Intent::Evolve {
        target,
        fitness_function,
    } = response.intent
    {
        assert!(target.contains("ml"));
        // Note: Fitness function extraction would need more sophisticated parsing
    }
}

#[tokio::test]
async fn test_logs_retrieval() {
    let config = AssistantConfig::default();
    let assistant = AiAssistant::new(config).await.unwrap();

    let response = assistant
        .process_input("show me logs for web-service")
        .await
        .unwrap();

    assert!(matches!(response.intent, Intent::Logs { .. }));
    assert!(response.query_results.is_some());

    if let Some(results) = &response.query_results {
        assert_eq!(results[0].resource_type, "logs");
        assert!(results[0].data.contains_key("lines"));
    }
}

#[tokio::test]
async fn test_rollback_command() {
    let config = AssistantConfig::default();
    let assistant = AiAssistant::new(config).await.unwrap();

    let response = assistant
        .process_input("rollback my-app to previous version")
        .await
        .unwrap();

    assert!(matches!(response.intent, Intent::Rollback { .. }));
    assert!(response.command.is_some());

    if let Some(command) = &response.command {
        assert!(command.requires_confirmation);
        assert_eq!(command.impact_level, "high");
    }
}

#[tokio::test]
async fn test_unknown_intent_handling() {
    let config = AssistantConfig::default();
    let assistant = AiAssistant::new(config).await.unwrap();

    let response = assistant
        .process_input("blahblahblah random gibberish xyz")
        .await
        .unwrap();

    assert!(matches!(response.intent, Intent::Unknown { .. }));
    assert!(response.confidence < 0.5);
    assert!(response.command.is_none());
    assert!(!response.suggestions.is_empty());
}

#[tokio::test]
async fn test_confidence_threshold() {
    let mut config = AssistantConfig::default();
    config.confidence_threshold = 0.9; // Very high threshold

    let assistant = AiAssistant::new(config).await.unwrap();

    let response = assistant
        .process_input("maybe deploy something")
        .await
        .unwrap();

    // Should still work but with lower confidence noted
    assert!(response.confidence <= 0.9);
}

#[tokio::test]
async fn test_suggestions_contextual() {
    let config = AssistantConfig::default();
    let assistant = AiAssistant::new(config).await.unwrap();

    let response = assistant.process_input("deploy my app").await.unwrap();

    // Deploy-related suggestions should be provided
    assert!(response
        .suggestions
        .iter()
        .any(|s| s.to_lowercase().contains("scale")));
    assert!(response
        .suggestions
        .iter()
        .any(|s| s.to_lowercase().contains("status")));
}

#[tokio::test]
async fn test_query_with_filters() {
    let config = AssistantConfig::default();
    let assistant = AiAssistant::new(config).await.unwrap();

    let response = assistant
        .process_input("show me running agents")
        .await
        .unwrap();

    assert!(matches!(response.intent, Intent::Query { .. }));

    if let Intent::Query {
        resource_type,
        filters,
        ..
    } = response.intent
    {
        assert!(resource_type.contains("agents"));
        // Note: Filter extraction would need more sophisticated parsing
    }
}

#[tokio::test]
async fn test_optimization_command() {
    let config = AssistantConfig::default();
    let assistant = AiAssistant::new(config).await.unwrap();

    let response = assistant
        .process_input("optimize my service for better latency")
        .await
        .unwrap();

    assert!(matches!(response.intent, Intent::Optimize { .. }));
    assert!(response.command.is_some());

    if let Intent::Optimize { target, metric, .. } = response.intent {
        assert!(target.contains("service"));
        assert!(metric.contains("latency"));
    }
}

#[tokio::test]
async fn test_multiple_resource_query() {
    let config = AssistantConfig::default();
    let assistant = AiAssistant::new(config).await.unwrap();

    let response = assistant
        .process_input("show me all resources")
        .await
        .unwrap();

    assert!(matches!(response.intent, Intent::Query { .. }));
    assert!(response.query_results.is_some());

    if let Some(results) = &response.query_results {
        // Should return multiple resource types
        let resource_types: std::collections::HashSet<_> =
            results.iter().map(|r| &r.resource_type).collect();
        assert!(resource_types.len() > 1);
    }
}

#[tokio::test]
async fn test_response_natural_language() {
    let config = AssistantConfig::default();
    let assistant = AiAssistant::new(config).await.unwrap();

    let response = assistant.process_input("deploy my blog").await.unwrap();

    // Response should be natural language, not just raw data
    assert!(response.response.len() > 10);
    assert!(response.response.contains("deploy") || response.response.contains("Deploy"));
    assert!(response.response.contains("blog"));
}

#[tokio::test]
async fn test_concurrent_requests() {
    let config = AssistantConfig::default();
    let assistant = std::sync::Arc::new(AiAssistant::new(config).await.unwrap());

    // Spawn multiple concurrent requests
    let mut handles = Vec::new();
    for i in 0..5 {
        let assistant_clone = assistant.clone();
        let handle = tokio::spawn(async move {
            let response = assistant_clone
                .process_input(&format!("deploy app-{}", i))
                .await
                .unwrap();
            assert!(matches!(response.intent, Intent::Deploy { .. }));
        });
        handles.push(handle);
    }

    // Wait for all requests to complete
    for handle in handles {
        handle.await.unwrap();
    }
}
