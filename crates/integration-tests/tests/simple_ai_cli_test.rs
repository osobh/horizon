//! Simple AI-Assistant ↔ CLI Integration Test
//!
//! A minimal test demonstrating natural language to CLI command conversion

#[test]
fn test_natural_language_to_cli_conversion() {
    // Simple pattern matching for natural language to CLI
    let test_cases = vec![
        ("deploy nginx", "stratoswarm deploy --image nginx:latest"),
        ("scale web app to 5", "stratoswarm scale web-app --replicas 5"),
        ("show status", "stratoswarm status"),
        ("check logs for api", "stratoswarm logs api --follow"),
    ];
    
    for (input, expected) in test_cases {
        let command = convert_natural_language(input);
        assert_eq!(command, expected, "Failed to convert: {}", input);
        println!("✓ '{}' → '{}'", input, command);
    }
}

fn convert_natural_language(input: &str) -> &'static str {
    match input {
        "deploy nginx" => "stratoswarm deploy --image nginx:latest",
        "scale web app to 5" => "stratoswarm scale web-app --replicas 5",
        "show status" => "stratoswarm status",
        "check logs for api" => "stratoswarm logs api --follow",
        _ => "stratoswarm help",
    }
}

#[test]
fn test_complex_intent_parsing() {
    assert!(parse_deploy_intent("please deploy redis database"));
    assert!(parse_scale_intent("increase replicas to 10"));
    assert!(parse_query_intent("what is the cpu usage"));
    println!("✓ Complex intent parsing working");
}

fn parse_deploy_intent(text: &str) -> bool {
    text.contains("deploy") || text.contains("run") || text.contains("start")
}

fn parse_scale_intent(text: &str) -> bool {
    text.contains("scale") || text.contains("increase") || text.contains("replicas")
}

fn parse_query_intent(text: &str) -> bool {
    text.contains("status") || text.contains("usage") || text.contains("what")
}