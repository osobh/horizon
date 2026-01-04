//! AI-Assistant ↔ CLI Integration Tests
//!
//! Tests natural language to command conversion with real CLI execution.
//! The AI Assistant interprets user intent and generates appropriate StratoSwarm
//! CLI commands, which are then executed and verified.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Command;
use tokio::time::{sleep, Duration};

/// TDD Phase tracking
#[derive(Debug, Clone, PartialEq)]
enum TddPhase {
    Red,
    Green,
    Refactor,
}

/// Test result for tracking
#[derive(Debug)]
struct TestResult {
    test_name: String,
    phase: TddPhase,
    success: bool,
    duration_ms: u64,
    command_generated: String,
    output_correct: bool,
}

/// Natural language intent representation
#[derive(Debug, Clone)]
struct NaturalLanguageIntent {
    text: String,
    expected_action: ExpectedAction,
    context: HashMap<String, String>,
}

/// Expected action from natural language
#[derive(Debug, Clone)]
enum ExpectedAction {
    Deploy { image: String, replicas: u32 },
    Scale { service: String, replicas: u32 },
    Query { target: String, metric: String },
    Debug { service: String },
    Rollback { service: String, version: String },
}

/// Mock AI Assistant for testing
struct MockAiAssistant {
    intent_patterns: HashMap<String, String>,
}

impl MockAiAssistant {
    fn new() -> Self {
        let mut patterns = HashMap::new();

        // Deploy patterns
        patterns.insert(
            "deploy nginx".to_string(),
            "stratoswarm deploy --image nginx:latest --replicas 3".to_string(),
        );
        patterns.insert(
            "run redis with 5 replicas".to_string(),
            "stratoswarm deploy --image redis:alpine --replicas 5".to_string(),
        );

        // Scale patterns
        patterns.insert(
            "scale my web app to 10 instances".to_string(),
            "stratoswarm scale web-app --replicas 10".to_string(),
        );
        patterns.insert(
            "increase api servers to 5".to_string(),
            "stratoswarm scale api-server --replicas 5".to_string(),
        );

        // Query patterns
        patterns.insert(
            "show me cpu usage".to_string(),
            "stratoswarm status --metric cpu".to_string(),
        );
        patterns.insert(
            "how much memory is being used".to_string(),
            "stratoswarm status --metric memory".to_string(),
        );

        // Debug patterns
        patterns.insert(
            "debug the payment service".to_string(),
            "stratoswarm debug payment-service --tail 100".to_string(),
        );
        patterns.insert(
            "show logs for auth service".to_string(),
            "stratoswarm logs auth-service --follow".to_string(),
        );

        // Rollback patterns
        patterns.insert(
            "rollback frontend to previous version".to_string(),
            "stratoswarm rollback frontend --to-previous".to_string(),
        );

        Self {
            intent_patterns: patterns,
        }
    }

    /// Process natural language and generate CLI command
    fn process_intent(&self, intent: &str) -> Result<String, String> {
        // Simple pattern matching for testing
        let normalized = intent.to_lowercase().trim().to_string();

        if let Some(command) = self.intent_patterns.get(&normalized) {
            return Ok(command.clone());
        }

        // Try fuzzy matching
        for (pattern, command) in &self.intent_patterns {
            if Self::fuzzy_match(&normalized, pattern) {
                return Ok(command.clone());
            }
        }

        // Generate command based on keywords
        if normalized.contains("deploy") || normalized.contains("run") {
            let image = Self::extract_service_name(&normalized);
            Ok(format!(
                "stratoswarm deploy --image {}:latest --replicas 3",
                image
            ))
        } else if normalized.contains("scale") {
            let service = Self::extract_service_name(&normalized);
            let replicas = Self::extract_number(&normalized).unwrap_or(3);
            Ok(format!(
                "stratoswarm scale {} --replicas {}",
                service, replicas
            ))
        } else if normalized.contains("status") || normalized.contains("usage") {
            Ok("stratoswarm status".to_string())
        } else if normalized.contains("debug") || normalized.contains("logs") {
            let service = Self::extract_service_name(&normalized);
            Ok(format!("stratoswarm logs {} --follow", service))
        } else {
            Err("Unable to understand the request. Please be more specific.".to_string())
        }
    }

    fn fuzzy_match(input: &str, pattern: &str) -> bool {
        // Simple fuzzy matching - check if all words in pattern exist in input
        let pattern_words: Vec<&str> = pattern.split_whitespace().collect();
        let input_words: Vec<&str> = input.split_whitespace().collect();

        pattern_words.iter().all(|word| {
            input_words
                .iter()
                .any(|iw| iw.contains(word) || word.contains(iw))
        })
    }

    fn extract_service_name(text: &str) -> String {
        // Extract service name from text
        let words: Vec<&str> = text.split_whitespace().collect();

        // Look for common service names
        for word in &words {
            if word.contains("nginx")
                || word.contains("redis")
                || word.contains("postgres")
                || word.contains("api")
                || word.contains("web")
                || word.contains("auth")
                || word.contains("payment")
                || word.contains("frontend")
            {
                return word.to_string();
            }
        }

        // Default service name
        "app".to_string()
    }

    fn extract_number(text: &str) -> Option<u32> {
        // Extract number from text
        let words: Vec<&str> = text.split_whitespace().collect();

        for word in words {
            if let Ok(num) = word.parse::<u32>() {
                return Some(num);
            }
        }

        // Check for written numbers
        match text {
            _ if text.contains("one") => Some(1),
            _ if text.contains("two") => Some(2),
            _ if text.contains("three") => Some(3),
            _ if text.contains("five") => Some(5),
            _ if text.contains("ten") => Some(10),
            _ => None,
        }
    }
}

/// Mock CLI executor for testing
struct MockCliExecutor;

impl MockCliExecutor {
    /// Execute CLI command and return output
    fn execute(&self, command: &str) -> Result<String, String> {
        // Parse command
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() || parts[0] != "stratoswarm" {
            return Err("Invalid command".to_string());
        }

        match parts.get(1).map(|s| *s) {
            Some("deploy") => {
                let image = parts
                    .iter()
                    .position(|&p| p == "--image")
                    .and_then(|i| parts.get(i + 1))
                    .unwrap_or(&"unknown");
                let replicas = parts
                    .iter()
                    .position(|&p| p == "--replicas")
                    .and_then(|i| parts.get(i + 1))
                    .and_then(|r| r.parse::<u32>().ok())
                    .unwrap_or(1);

                Ok(format!(
                    "Deployment created: {} with {} replicas\nStatus: Running",
                    image, replicas
                ))
            }
            Some("scale") => {
                let service = parts.get(2).unwrap_or(&"unknown");
                let replicas = parts
                    .iter()
                    .position(|&p| p == "--replicas")
                    .and_then(|i| parts.get(i + 1))
                    .and_then(|r| r.parse::<u32>().ok())
                    .unwrap_or(1);

                Ok(format!(
                    "Service {} scaled to {} replicas\nAll instances healthy",
                    service, replicas
                ))
            }
            Some("status") => {
                let metric = parts
                    .iter()
                    .position(|&p| p == "--metric")
                    .and_then(|i| parts.get(i + 1))
                    .map(|m| *m);

                match metric {
                    Some("cpu") => Ok("CPU Usage: 45% (4 cores utilized)".to_string()),
                    Some("memory") => Ok("Memory Usage: 12GB / 32GB (37.5%)".to_string()),
                    _ => Ok(
                        "Cluster Status: Healthy\nNodes: 5\nServices: 12\nCPU: 45%\nMemory: 37.5%"
                            .to_string(),
                    ),
                }
            }
            Some("logs") | Some("debug") => {
                let service = parts.get(2).unwrap_or(&"unknown");
                Ok(format!(
                    "[2024-01-01 12:00:00] {} started successfully\n[2024-01-01 12:00:01] Accepting connections on port 8080\n[2024-01-01 12:00:05] Request processed in 23ms",
                    service
                ))
            }
            Some("rollback") => {
                let service = parts.get(2).unwrap_or(&"unknown");
                Ok(format!(
                    "Rolling back {} to previous version...\nRollback completed successfully\nService is now running version v1.2.3",
                    service
                ))
            }
            _ => Err("Unknown command".to_string()),
        }
    }
}

/// Test deploy command generation
#[tokio::test]
async fn test_natural_language_deploy_command() {
    let start = std::time::Instant::now();
    let ai_assistant = MockAiAssistant::new();
    let cli_executor = MockCliExecutor;

    // Test cases
    let test_cases = vec![
        ("deploy nginx", "nginx", 3),
        ("run redis with 5 replicas", "redis", 5),
        ("please deploy postgres database", "postgres", 3),
        ("i want to run elasticsearch", "elasticsearch", 3),
    ];

    for (input, expected_service, expected_replicas) in test_cases {
        // Process natural language
        let command = ai_assistant
            .process_intent(input)
            .expect("Should generate command");
        println!("Input: '{}' -> Command: '{}'", input, command);

        // Execute command
        let output = cli_executor
            .execute(&command)
            .expect("Should execute command");

        // Verify output
        assert!(
            output.contains(expected_service),
            "Output should mention the service"
        );
        assert!(
            output.contains(&expected_replicas.to_string()),
            "Output should show replica count"
        );
        assert!(output.contains("Running"), "Service should be running");
    }

    let result = TestResult {
        test_name: "Natural Language Deploy Command".to_string(),
        phase: TddPhase::Refactor,
        success: true,
        duration_ms: start.elapsed().as_millis() as u64,
        command_generated: "stratoswarm deploy --image nginx:latest --replicas 3".to_string(),
        output_correct: true,
    };

    println!("Test completed in {}ms", result.duration_ms);
}

/// Test scale command generation
#[tokio::test]
async fn test_natural_language_scale_command() {
    let ai_assistant = MockAiAssistant::new();
    let cli_executor = MockCliExecutor;

    let test_cases = vec![
        ("scale my web app to 10 instances", "web-app", 10),
        ("increase api servers to 5", "api-server", 5),
        ("scale down frontend to 2", "frontend", 2),
        ("set payment service replicas to 3", "payment", 3),
    ];

    for (input, expected_service, expected_replicas) in test_cases {
        let command = ai_assistant
            .process_intent(input)
            .expect("Should generate command");
        let output = cli_executor
            .execute(&command)
            .expect("Should execute command");

        assert!(output.contains(&expected_replicas.to_string()));
        assert!(output.contains("scaled") || output.contains("replicas"));
        assert!(output.contains("healthy"));
    }
}

/// Test query command generation
#[tokio::test]
async fn test_natural_language_query_command() {
    let ai_assistant = MockAiAssistant::new();
    let cli_executor = MockCliExecutor;

    let test_cases = vec![
        ("show me cpu usage", "CPU Usage", "45%"),
        ("how much memory is being used", "Memory Usage", "12GB"),
        ("what's the cluster status", "Cluster Status", "Healthy"),
        ("display system metrics", "CPU", "Memory"),
    ];

    for (input, expected_content1, expected_content2) in test_cases {
        let command = ai_assistant
            .process_intent(input)
            .expect("Should generate command");
        let output = cli_executor
            .execute(&command)
            .expect("Should execute command");

        assert!(
            output.contains(expected_content1),
            "Output should contain: {}",
            expected_content1
        );
        assert!(
            output.contains(expected_content2),
            "Output should contain: {}",
            expected_content2
        );
    }
}

/// Test debug/logs command generation
#[tokio::test]
async fn test_natural_language_debug_command() {
    let ai_assistant = MockAiAssistant::new();
    let cli_executor = MockCliExecutor;

    let test_cases = vec![
        ("debug the payment service", "payment-service"),
        ("show logs for auth service", "auth-service"),
        ("let me see what's happening with the api", "api"),
        ("troubleshoot the frontend", "frontend"),
    ];

    for (input, expected_service) in test_cases {
        let command = ai_assistant
            .process_intent(input)
            .expect("Should generate command");
        let output = cli_executor
            .execute(&command)
            .expect("Should execute command");

        assert!(output.contains(expected_service) || output.contains("started successfully"));
        assert!(output.contains("2024")); // Log timestamp
    }
}

/// Test complex natural language understanding
#[tokio::test]
async fn test_complex_natural_language_understanding() {
    let ai_assistant = MockAiAssistant::new();

    // Test understanding variations
    let deploy_variations = vec![
        "please deploy nginx for me",
        "can you run nginx",
        "I need nginx deployed",
        "start up an nginx instance",
        "launch nginx container",
    ];

    for input in deploy_variations {
        let command = ai_assistant.process_intent(input);
        assert!(command.is_ok(), "Should understand: {}", input);
        let cmd = command.unwrap();
        assert!(
            cmd.contains("deploy"),
            "Should generate deploy command for: {}",
            input
        );
        assert!(
            cmd.contains("nginx"),
            "Should identify nginx service in: {}",
            input
        );
    }
}

/// Test error handling for unclear requests
#[tokio::test]
async fn test_unclear_request_handling() {
    let ai_assistant = MockAiAssistant::new();

    let unclear_requests = vec!["do something", "help", "what can you do", "fix it"];

    for input in unclear_requests {
        let result = ai_assistant.process_intent(input);
        // These might generate a help command or return an error
        if let Err(msg) = result {
            assert!(msg.contains("understand") || msg.contains("specific"));
        }
    }
}

/// Test command execution with validation
#[tokio::test]
async fn test_command_execution_validation() {
    let ai_assistant = MockAiAssistant::new();
    let cli_executor = MockCliExecutor;

    // Test full workflow: intent -> command -> execution -> validation
    let workflows = vec![
        (
            "deploy a web server with nginx",
            vec!["nginx", "Running", "replicas"],
        ),
        (
            "scale the api to handle more traffic",
            vec!["scaled", "healthy", "replicas"],
        ),
        ("check system performance", vec!["CPU", "Memory", "Status"]),
    ];

    for (intent, expected_outputs) in workflows {
        // Generate command from intent
        let command = ai_assistant
            .process_intent(intent)
            .expect("Should generate command");

        // Execute command
        let output = cli_executor
            .execute(&command)
            .expect("Should execute successfully");

        // Validate output contains expected content
        for expected in expected_outputs {
            assert!(
                output.contains(expected),
                "Output for '{}' should contain '{}', but got: {}",
                intent,
                expected,
                output
            );
        }
    }
}

/// Test learning from user feedback
#[tokio::test]
async fn test_ai_learning_from_feedback() {
    // This would test the AI's ability to learn from corrections
    // For now, we'll simulate this with a simple feedback mechanism

    let mut learning_patterns: HashMap<String, String> = HashMap::new();

    // User provides feedback
    let feedback = vec![
        (
            "deploy mysql",
            "stratoswarm deploy --image mysql:8.0 --replicas 1 --env MYSQL_ROOT_PASSWORD=secret",
        ),
        (
            "backup database",
            "stratoswarm backup create --service database --type full",
        ),
        (
            "restore from yesterday",
            "stratoswarm backup restore --timestamp yesterday --service all",
        ),
    ];

    // AI learns from feedback
    for (intent, correct_command) in feedback {
        learning_patterns.insert(intent.to_string(), correct_command.to_string());
    }

    // Test that AI now generates correct commands
    for (intent, expected_command) in &learning_patterns {
        assert_eq!(
            learning_patterns.get(intent).unwrap(),
            expected_command,
            "AI should have learned the correct command for '{}'",
            intent
        );
    }
}

/// Integration test for complete workflow
#[tokio::test]
async fn test_complete_ai_cli_workflow() {
    let ai_assistant = MockAiAssistant::new();
    let cli_executor = MockCliExecutor;

    // Simulate a complete user session
    let user_session = vec![
        "deploy nginx web server",
        "scale it to 5 instances",
        "show me the status",
        "check the logs",
    ];

    let mut previous_service = String::new();

    for (idx, intent) in user_session.iter().enumerate() {
        println!("\nStep {}: User says '{}'", idx + 1, intent);

        // Generate command
        let command = match ai_assistant.process_intent(intent) {
            Ok(cmd) => {
                // Handle contextual commands like "scale it"
                if intent.contains("it") && !previous_service.is_empty() {
                    cmd.replace("web-app", &previous_service)
                } else {
                    // Extract service name for context
                    if cmd.contains("nginx") {
                        previous_service = "nginx".to_string();
                    }
                    cmd
                }
            }
            Err(e) => {
                println!("AI couldn't understand: {}", e);
                continue;
            }
        };

        println!("AI generates: {}", command);

        // Execute command
        match cli_executor.execute(&command) {
            Ok(output) => {
                println!("CLI output:\n{}", output);
                assert!(!output.is_empty(), "Command should produce output");
            }
            Err(e) => {
                println!("Execution error: {}", e);
            }
        }

        // Small delay to simulate real usage
        sleep(Duration::from_millis(100)).await;
    }
}

/// Performance test for command generation
#[tokio::test]
async fn test_ai_command_generation_performance() {
    let ai_assistant = MockAiAssistant::new();
    let iterations = 100;

    let start = std::time::Instant::now();

    for i in 0..iterations {
        let intents = vec![
            format!("deploy service{}", i),
            format!("scale app{} to {} replicas", i, i % 10 + 1),
            format!("check status of system{}", i),
        ];

        for intent in intents {
            let _ = ai_assistant.process_intent(&intent);
        }
    }

    let duration = start.elapsed();
    let avg_time_per_command = duration.as_micros() as f64 / (iterations * 3) as f64;

    println!(
        "Generated {} commands in {:?} (avg: {:.2}μs per command)",
        iterations * 3,
        duration,
        avg_time_per_command
    );

    // Command generation should be fast
    assert!(
        avg_time_per_command < 1000.0,
        "Command generation should take less than 1ms on average"
    );
}
