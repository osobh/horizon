//! Integration tests for the .swarm DSL parser and compiler

use insta::assert_snapshot;
use pretty_assertions::assert_eq;
use stratoswarm_dsl::{compile, parse, parse_and_compile, DslError};

#[test]
#[ignore = "Templates not fully implemented yet"]
fn test_parse_complete_swarm_file() {
    let input = r#"
        // Import external swarm definitions
        import "github.com/stratoswarm/stdlib" as stdlib;
        import "./local/templates.swarm";

        // Template for microservice pattern
        template Microservice(name: String, port: Int) {
            swarm ${name}_service {
                agents {
                    api: WebAgent {
                        replicas: 3..10,
                        tier_preference: ["GPU", "CPU", "NVMe"],
                        
                        resources {
                            cpu: 2.0,
                            memory: "4Gi",
                            gpu: optional(0.5),
                        }
                        
                        network {
                            expose: ${port},
                            mesh: true,
                            load_balance: "least_connections",
                        }
                        
                        evolution {
                            strategy: "conservative",
                            fitness: "latency < 100ms && error_rate < 0.01",
                        }
                    }
                    
                    worker: ComputeAgent {
                        replicas: 5,
                        requires_gpu: true,
                        
                        affinity {
                            prefer_same_node: api,
                            avoid_nodes_with: "tier2_pressure > 80%",
                        }
                    }
                }
                
                connections {
                    api -> worker: {
                        protocol: "grpc",
                        retry: exponential_backoff(1, 30),
                        circuit_breaker: true,
                    }
                }
            }
        }

        // Main application swarm
        swarm myapp {
            agents {
                frontend: WebAgent {
                    replicas: 3..10,
                    tier_preference: ["GPU", "CPU", "NVMe"],
                    
                    resources {
                        cpu: 2.0,
                        memory: "4Gi",
                        gpu: optional(0.5),
                    }
                    
                    network {
                        expose: 80,
                        mesh: true,
                        load_balance: "least_connections",
                    }
                    
                    evolution {
                        strategy: "conservative",
                        fitness: "latency < 100ms && error_rate < 0.01",
                    }
                    
                    personality {
                        risk_tolerance: 0.7,
                        cooperation: 0.9,
                        exploration: 0.5,
                        efficiency_focus: 0.8,
                        stability_preference: 0.6,
                    }
                }
                
                backend: ComputeAgent {
                    replicas: 5,
                    requires_gpu: true,
                    
                    affinity {
                        prefer_same_node: frontend,
                        avoid_nodes_with: "tier2_pressure > 80%",
                    }
                    
                    code {
                        source: "github.com/myorg/backend",
                        auto_evolve: true,
                    }
                }
                
                database: StorageAgent {
                    replicas: 3,
                    
                    resources {
                        cpu: 4.0,
                        memory: "16Gi",
                    }
                    
                    storage {
                        size: "100Gi",
                        type: "nvme",
                        replication: 3,
                    }
                }
            }
            
            connections {
                frontend -> backend: {
                    protocol: "grpc",
                    retry: exponential_backoff(1, 30),
                    circuit_breaker: true,
                }
                
                backend -> database: {
                    protocol: "tcp",
                    port: 5432,
                    pool_size: 20,
                }
            }
            
            policies {
                zero_downtime_updates: true,
                canary_rollout: "10% -> 50% -> 100%",
                rollback_on: "error_rate > 5%",
                max_surge: "25%",
                max_unavailable: "0%",
            }
            
            functions {
                fn health_check(agent: Agent) -> Bool {
                    response = http_get("http://${agent.ip}:${agent.port}/health");
                    result = response.status == 200;
                }
                
                fn scale_decision(metrics: Metrics) -> Int {
                    target = metrics.request_rate / 100;
                    result = max(3, min(10, target));
                }
            }
            
            evolution {
                enabled: true,
                population_size: 100,
                mutation_rate: 0.1,
                crossover_rate: 0.7,
                selection_pressure: 2.0,
            }
            
            affinity {
                node_selector: {
                    tier: "production",
                    region: "us-west-2",
                }
                tolerations: [
                    {
                        key: "gpu",
                        operator: "Equal",
                        value: "nvidia",
                        effect: "NoSchedule",
                    }
                ]
            }
        }
    "#;

    let result = parse(input);
    assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

    let file = result.unwrap();
    assert_eq!(file.imports.len(), 2);
    assert_eq!(file.templates.len(), 1);
    assert_eq!(file.swarms.len(), 1);

    let swarm = &file.swarms[0];
    assert_eq!(swarm.name, "myapp");
    assert_eq!(swarm.agents.len(), 3);
    assert_eq!(swarm.connections.len(), 2);
    assert!(swarm.evolution.is_some());
    assert!(swarm.affinity.is_some());
}

#[test]
fn test_compile_to_agent_specs() {
    let input = r#"
        swarm simple_app {
            agents {
                web: WebAgent {
                    replicas: 3,
                    resources {
                        cpu: 2.0,
                        memory: "4Gi",
                    }
                    network {
                        expose: 8080,
                    }
                }
                
                api: ComputeAgent {
                    replicas: 2,
                    resources {
                        cpu: 4.0,
                        memory: "8Gi",
                        gpu: 1.0,
                    }
                }
            }
            
            connections {
                web -> api: {
                    protocol: "http",
                }
            }
        }
    "#;

    let result = parse_and_compile(input);
    assert!(result.is_ok(), "Failed to compile: {:?}", result.err());

    let specs = result.unwrap();
    assert_eq!(specs.len(), 2);

    // Check web agent spec
    let web_spec = specs.iter().find(|s| s.name == "web").unwrap();
    assert_eq!(web_spec.agent_type, "WebAgent");
    assert_eq!(web_spec.replicas, (3, None));
    assert_eq!(web_spec.config.resources.cpu, 2.0);
    assert_eq!(web_spec.config.resources.memory, 4 * 1024 * 1024 * 1024);
    assert!(web_spec.config.resources.gpu.is_none());
    assert_eq!(web_spec.config.network.expose_ports, vec![8080]);
    assert_eq!(web_spec.connections, vec!["api"]);

    // Check api agent spec
    let api_spec = specs.iter().find(|s| s.name == "api").unwrap();
    assert_eq!(api_spec.agent_type, "ComputeAgent");
    assert_eq!(api_spec.replicas, (2, None));
    assert_eq!(api_spec.config.resources.cpu, 4.0);
    assert_eq!(api_spec.config.resources.gpu, Some(1.0));
    assert!(api_spec.connections.is_empty());
}

#[test]
fn test_tier_preference_parsing() {
    let input = r#"
        swarm test {
            agents {
                compute: ComputeAgent {
                    tier_preference: ["GPU", "CPU", "NVMe", "Memory"],
                }
            }
        }
    "#;

    let result = parse_and_compile(input);
    assert!(result.is_ok());

    let specs = result.unwrap();
    let spec = &specs[0];
    assert_eq!(
        spec.config.tier_preferences,
        vec!["GPU", "CPU", "NVMe", "Memory"]
    );
}

#[test]
fn test_personality_traits() {
    let input = r#"
        swarm test {
            agents {
                adaptive: WebAgent {
                    personality {
                        risk_tolerance: 0.8,
                        cooperation: 0.9,
                        exploration: 0.7,
                        efficiency_focus: 0.6,
                        stability_preference: 0.5,
                    }
                }
            }
        }
    "#;

    let result = parse_and_compile(input);
    assert!(result.is_ok());

    let specs = result.unwrap();
    let spec = &specs[0];

    assert_eq!(spec.config.personality.risk_tolerance, 0.8);
    assert_eq!(spec.config.personality.cooperation, 0.9);
    assert_eq!(spec.config.personality.exploration, 0.7);
    assert_eq!(spec.config.personality.efficiency_focus, 0.6);
    assert_eq!(spec.config.personality.stability_preference, 0.5);
}

#[test]
fn test_evolution_configuration() {
    let input = r#"
        swarm test {
            agents {
                evolving: ComputeAgent {
                    evolution {
                        strategy: "aggressive",
                        fitness: "throughput > 1000",
                    }
                }
            }
            
            evolution {
                enabled: true,
                population_size: 200,
                mutation_rate: 0.2,
            }
        }
    "#;

    let result = parse_and_compile(input);
    assert!(result.is_ok());

    let specs = result.unwrap();
    let spec = &specs[0];
    assert!(spec.config.evolution_enabled);
}

#[test]
fn test_range_replicas() {
    let input = r#"
        swarm test {
            agents {
                scalable: WebAgent {
                    replicas: 2..20,
                }
            }
        }
    "#;

    let result = parse_and_compile(input);
    assert!(result.is_ok());

    let specs = result.unwrap();
    let spec = &specs[0];
    assert_eq!(spec.replicas, (2, Some(20)));
}

#[test]
fn test_optional_gpu() {
    let input = r#"
        swarm test {
            agents {
                ml_service: ComputeAgent {
                    resources {
                        cpu: 4.0,
                        memory: "8Gi",
                        gpu: optional(2.0),
                    }
                }
            }
        }
    "#;

    let result = parse_and_compile(input);
    if let Err(e) = &result {
        println!("Parse error in test_optional_gpu: {:?}", e);
    }
    assert!(result.is_ok());

    let specs = result.unwrap();
    let spec = &specs[0];
    assert_eq!(spec.config.resources.gpu, Some(2.0));
    assert!(spec.config.resources.gpu_optional);
}

#[test]
fn test_complex_object_nesting() {
    let input = r#"
        swarm test {
            agents {
                complex: WebAgent {
                    config: {
                        server: {
                            host: "0.0.0.0",
                            port: 8080,
                            workers: 4
                        },
                        database: {
                            url: "postgres://localhost/mydb",
                            pool_size: 10
                        },
                        features: ["auth", "logging", "metrics"]
                    }
                }
            }
        }
    "#;

    let result = parse(input);
    if let Err(e) = &result {
        println!("Parse error in test_complex_object_nesting: {:?}", e);
    }
    assert!(result.is_ok());

    let file = result.unwrap();
    let agent = &file.swarms[0].agents["complex"];
    assert!(agent.properties.contains_key("config"));
}

#[test]
fn test_error_unknown_agent_type() {
    let input = r#"
        swarm test {
            agents {
                unknown: UnknownAgentType {
                    replicas: 1,
                }
            }
        }
    "#;

    let result = parse_and_compile(input);
    assert!(result.is_err());

    match result.err().unwrap() {
        DslError::ValidationError(msg) => {
            assert!(msg.contains("Unknown agent type"));
        }
        _ => panic!("Expected validation error"),
    }
}

#[test]
fn test_error_missing_connection_target() {
    let input = r#"
        swarm test {
            agents {
                source: WebAgent {}
            }
            
            connections {
                source -> nonexistent: {
                    protocol: "http",
                }
            }
        }
    "#;

    let result = parse_and_compile(input);
    assert!(result.is_err());

    match result.err().unwrap() {
        DslError::ValidationError(msg) => {
            assert!(msg.contains("not found"));
        }
        _ => panic!("Expected validation error"),
    }
}

#[test]
fn test_parse_syntax_error() {
    let input = r#"
        swarm test {
            this is not valid syntax
        }
    "#;

    let result = parse(input);
    assert!(result.is_err());

    match result.err().unwrap() {
        DslError::ParseError { .. } => {
            // Expected parse error
        }
        _ => panic!("Expected parse error"),
    }
}

#[test]
fn test_snapshot_complex_swarm() {
    let input = r#"
        swarm production_app {
            agents {
                frontend: WebAgent {
                    replicas: 5..20,
                    tier_preference: ["GPU", "CPU"],
                    
                    resources {
                        cpu: 2.0,
                        memory: "4Gi",
                    }
                    
                    evolution {
                        strategy: "conservative",
                    }
                }
                
                backend: ComputeAgent {
                    replicas: 10,
                    requires_gpu: true,
                }
            }
            
            connections {
                frontend -> backend: {
                    protocol: "grpc",
                }
            }
            
            policies {
                zero_downtime_updates: true,
            }
        }
    "#;

    let result = parse(input);
    assert!(result.is_ok());

    let file = result.unwrap();
    let serialized = serde_json::to_string_pretty(&file).unwrap();

    // This will create a snapshot file for visual inspection
    assert_snapshot!(serialized);
}

#[test]
fn test_memory_parsing_units() {
    let input = r#"
        swarm test {
            agents {
                mem_test: WebAgent {
                    resources {
                        memory: "512Mi",
                    }
                }
                
                mem_test2: WebAgent {
                    resources {
                        memory: "2Gi",
                    }
                }
                
                mem_test3: WebAgent {
                    resources {
                        memory: "1024Ki",
                    }
                }
            }
        }
    "#;

    let result = parse_and_compile(input);
    assert!(result.is_ok());

    let specs = result.unwrap();

    let spec1 = specs.iter().find(|s| s.name == "mem_test").unwrap();
    assert_eq!(spec1.config.resources.memory, 512 * 1024 * 1024);

    let spec2 = specs.iter().find(|s| s.name == "mem_test2").unwrap();
    assert_eq!(spec2.config.resources.memory, 2 * 1024 * 1024 * 1024);

    let spec3 = specs.iter().find(|s| s.name == "mem_test3").unwrap();
    assert_eq!(spec3.config.resources.memory, 1024 * 1024);
}

#[test]
fn test_array_types() {
    let input = r#"
        swarm test {
            agents {
                array_test: WebAgent {
                    ports: [80, 443, 8080],
                    domains: ["example.com", "api.example.com"],
                    flags: [true, false, true],
                    mixed: [1, "two", true],
                }
            }
        }
    "#;

    let result = parse(input);
    assert!(result.is_ok());

    let file = result.unwrap();
    let agent = &file.swarms[0].agents["array_test"];

    // Check ports array
    if let Some(stratoswarm_dsl::ast::Value::Array(ports)) = agent.properties.get("ports") {
        assert_eq!(ports.len(), 3);
        assert!(matches!(
            ports[0],
            stratoswarm_dsl::ast::Value::Number(80.0)
        ));
    } else {
        panic!("Expected ports array");
    }

    // Check domains array
    if let Some(stratoswarm_dsl::ast::Value::Array(domains)) = agent.properties.get("domains") {
        assert_eq!(domains.len(), 2);
        assert!(
            matches!(&domains[0], stratoswarm_dsl::ast::Value::String(s) if s == "example.com")
        );
    } else {
        panic!("Expected domains array");
    }
}
