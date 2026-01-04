//! Quickstart command implementation

use crate::{output, CliError, Result};
use clap::Args;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Args)]
pub struct QuickstartArgs {
    /// Template to use
    #[arg(short, long, value_enum)]
    pub template: Template,

    /// Enable GPU support
    #[arg(long)]
    pub gpu: bool,

    /// Output directory
    #[arg(short, long, default_value = ".")]
    pub output: PathBuf,

    /// Project name
    #[arg(short, long)]
    pub name: Option<String>,

    /// Number of agent replicas
    #[arg(short, long, default_value = "3")]
    pub replicas: u32,

    /// Enable evolution
    #[arg(short, long)]
    pub evolution: bool,

    /// Skip confirmation
    #[arg(short = 'y', long)]
    pub yes: bool,
}

#[derive(Debug, Clone, PartialEq, clap::ValueEnum)]
pub enum Template {
    /// Basic web API template
    WebApi,
    /// Microservices template
    Microservices,
    /// Machine learning pipeline
    MlPipeline,
    /// Data processing pipeline
    DataPipeline,
    /// Static website
    StaticSite,
    /// Full-stack application
    FullStack,
}

pub async fn execute(args: QuickstartArgs) -> Result<()> {
    let project_name = args
        .name
        .clone()
        .unwrap_or_else(|| format!("{}-app", args.template));

    let project_dir = args.output.join(&project_name);

    // Check if directory already exists
    if project_dir.exists() && !args.yes {
        return Err(CliError::Command(format!(
            "Directory {:?} already exists. Use --yes to overwrite.",
            project_dir
        )));
    }

    output::info(&format!(
        "Creating {} project '{}'",
        args.template, project_name
    ));

    // Create project directory
    fs::create_dir_all(&project_dir)?;

    // Generate files based on template
    match args.template {
        Template::WebApi => create_web_api_template(&project_dir, &args)?,
        Template::Microservices => create_microservices_template(&project_dir, &args)?,
        Template::MlPipeline => create_ml_pipeline_template(&project_dir, &args)?,
        Template::DataPipeline => create_data_pipeline_template(&project_dir, &args)?,
        Template::StaticSite => create_static_site_template(&project_dir, &args)?,
        Template::FullStack => create_fullstack_template(&project_dir, &args)?,
    }

    // Create common files
    create_readme(&project_dir, &project_name, &args)?;
    create_gitignore(&project_dir)?;

    output::success(&format!(
        "✓ Created {} project in {:?}",
        args.template, project_dir
    ));
    output::info("\nNext steps:");
    output::info(&format!("  cd {}", project_dir.display()));
    output::info("  stratoswarm deploy app.swarm");

    Ok(())
}

fn create_web_api_template(dir: &PathBuf, args: &QuickstartArgs) -> Result<()> {
    let swarm_content = format!(
        r#"// Web API Application
swarm web_api {{
    agents {{
        api: WebAgent {{
            replicas: {replicas},
            tier_preference: [{gpu_tier}"CPU", "Memory"],
            
            resources {{
                cpu: 2.0,
                memory: "2Gi",{gpu_resource}
            }}
            
            network {{
                expose: 8080,
                mesh: true,
                load_balance: "least_connections",
            }}{evolution_config}
        }}
        
        cache: StorageAgent {{
            replicas: 1,
            
            resources {{
                cpu: 1.0,
                memory: "1Gi",
            }}
            
            storage {{
                type: "memory",
                size: "1Gi",
            }}
        }}
    }}
    
    connections {{
        api -> cache: {{
            protocol: "redis",
            retry: exponential_backoff(1, 30),
        }}
    }}
    
    policies {{
        zero_downtime_updates: true,
        canary_rollout: "10% -> 50% -> 100%",
        rollback_on: "error_rate > 5%",
    }}
}}
"#,
        replicas = args.replicas,
        gpu_tier = if args.gpu { "\"GPU\", " } else { "" },
        gpu_resource = if args.gpu {
            "\n                gpu: optional(0.5),"
        } else {
            ""
        },
        evolution_config = if args.evolution {
            r#"
            
            evolution {
                strategy: "conservative",
                fitness: "latency < 100ms && error_rate < 0.01",
            }"#
        } else {
            ""
        }
    );

    fs::write(dir.join("app.swarm"), swarm_content)?;

    // Create sample API code
    let api_code = r#"from fastapi import FastAPI
import redis
import os

app = FastAPI()
redis_client = redis.Redis(host='cache', port=6379)

@app.get("/")
async def root():
    return {"message": "Hello from StratoSwarm!"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/cache/{key}")
async def get_cache(key: str):
    value = redis_client.get(key)
    return {"key": key, "value": value.decode() if value else None}

@app.post("/cache/{key}")
async def set_cache(key: str, value: str):
    redis_client.set(key, value)
    return {"status": "stored"}
"#;

    fs::write(dir.join("api.py"), api_code)?;

    // Create requirements.txt
    fs::write(dir.join("requirements.txt"), "fastapi\nuvicorn\nredis\n")?;

    Ok(())
}

fn create_microservices_template(dir: &PathBuf, args: &QuickstartArgs) -> Result<()> {
    let swarm_content = format!(
        r#"// Microservices Application
swarm microservices {{
    agents {{
        gateway: WebAgent {{
            replicas: {replicas},
            
            resources {{
                cpu: 2.0,
                memory: "2Gi",
            }}
            
            network {{
                expose: 80,
                mesh: true,
            }}
        }}
        
        auth_service: ComputeAgent {{
            replicas: 2,
            
            resources {{
                cpu: 1.0,
                memory: "1Gi",
            }}
        }}
        
        user_service: ComputeAgent {{
            replicas: 3,
            
            resources {{
                cpu: 1.5,
                memory: "2Gi",
            }}
        }}
        
        order_service: ComputeAgent {{
            replicas: 3,
            
            resources {{
                cpu: 2.0,
                memory: "4Gi",{gpu_resource}
            }}{evolution_config}
        }}
        
        database: StorageAgent {{
            replicas: 3,
            
            resources {{
                cpu: 4.0,
                memory: "8Gi",
            }}
            
            storage {{
                type: "persistent",
                size: "100Gi",
                replication: 3,
            }}
        }}
    }}
    
    connections {{
        gateway -> auth_service: {{
            protocol: "grpc",
        }}
        gateway -> user_service: {{
            protocol: "grpc",
        }}
        gateway -> order_service: {{
            protocol: "grpc",
        }}
        auth_service -> database: {{
            protocol: "postgresql",
        }}
        user_service -> database: {{
            protocol: "postgresql",
        }}
        order_service -> database: {{
            protocol: "postgresql",
        }}
    }}
    
    policies {{
        zero_downtime_updates: true,
        circuit_breaker: true,
        distributed_tracing: true,
    }}
}}
"#,
        replicas = args.replicas,
        gpu_resource = if args.gpu {
            "\n                gpu: optional(1.0),"
        } else {
            ""
        },
        evolution_config = if args.evolution {
            r#"
            
            evolution {
                strategy: "balanced",
                fitness: "throughput > 1000 && latency < 50ms",
            }"#
        } else {
            ""
        }
    );

    fs::write(dir.join("app.swarm"), swarm_content)?;

    // Create service directories
    for service in ["gateway", "auth-service", "user-service", "order-service"] {
        fs::create_dir_all(dir.join(service))?;
    }

    Ok(())
}

fn create_ml_pipeline_template(dir: &PathBuf, args: &QuickstartArgs) -> Result<()> {
    if !args.gpu {
        output::warn("ML Pipeline template typically requires GPU. Consider using --gpu flag.");
    }

    let swarm_content = format!(
        r#"// Machine Learning Pipeline
swarm ml_pipeline {{
    agents {{
        data_ingestion: ComputeAgent {{
            replicas: 2,
            
            resources {{
                cpu: 2.0,
                memory: "4Gi",
            }}
        }}
        
        preprocessing: ComputeAgent {{
            replicas: 4,
            
            resources {{
                cpu: 4.0,
                memory: "8Gi",
                gpu: optional(0.5),
            }}
        }}
        
        training: GPUAgent {{
            replicas: {gpu_replicas},
            requires_gpu: true,
            
            resources {{
                cpu: 8.0,
                memory: "32Gi",
                gpu: 2.0,
            }}
            
            evolution {{
                strategy: "aggressive",
                fitness: "model_accuracy > 0.95",
            }}
        }}
        
        inference: ComputeAgent {{
            replicas: {replicas},
            
            resources {{
                cpu: 2.0,
                memory: "4Gi",
                gpu: optional(1.0),
            }}
        }}
        
        model_store: StorageAgent {{
            replicas: 1,
            
            resources {{
                cpu: 1.0,
                memory: "2Gi",
            }}
            
            storage {{
                type: "persistent",
                size: "500Gi",
            }}
        }}
    }}
    
    connections {{
        data_ingestion -> preprocessing: {{
            protocol: "streaming",
        }}
        preprocessing -> training: {{
            protocol: "batch",
        }}
        training -> model_store: {{
            protocol: "s3",
        }}
        model_store -> inference: {{
            protocol: "http",
        }}
    }}
    
    policies {{
        gpu_affinity: true,
        data_locality: true,
        auto_scaling: "gpu_utilization > 80%",
    }}
}}
"#,
        replicas = args.replicas,
        gpu_replicas = if args.gpu { "2" } else { "1" }
    );

    fs::write(dir.join("app.swarm"), swarm_content)?;
    Ok(())
}

fn create_data_pipeline_template(dir: &PathBuf, args: &QuickstartArgs) -> Result<()> {
    let swarm_content = format!(
        r#"// Data Processing Pipeline
swarm data_pipeline {{
    agents {{
        collector: ComputeAgent {{
            replicas: {replicas},
            
            resources {{
                cpu: 2.0,
                memory: "4Gi",
            }}
        }}
        
        transformer: ComputeAgent {{
            replicas: 5,
            
            resources {{
                cpu: 4.0,
                memory: "8Gi",{gpu_resource}
            }}{evolution_config}
        }}
        
        aggregator: ComputeAgent {{
            replicas: 3,
            
            resources {{
                cpu: 4.0,
                memory: "16Gi",
            }}
        }}
        
        storage: StorageAgent {{
            replicas: 3,
            
            resources {{
                cpu: 2.0,
                memory: "4Gi",
            }}
            
            storage {{
                type: "distributed",
                size: "1Ti",
                replication: 2,
            }}
        }}
    }}
    
    connections {{
        collector -> transformer: {{
            protocol: "kafka",
            partitions: 10,
        }}
        transformer -> aggregator: {{
            protocol: "kafka",
            partitions: 5,
        }}
        aggregator -> storage: {{
            protocol: "parquet",
        }}
    }}
    
    policies {{
        data_retention: "30d",
        compression: "snappy",
        exactly_once_processing: true,
    }}
}}
"#,
        replicas = args.replicas,
        gpu_resource = if args.gpu {
            "\n                gpu: optional(1.0),"
        } else {
            ""
        },
        evolution_config = if args.evolution {
            r#"
            
            evolution {
                strategy: "balanced",
                fitness: "throughput > 10000 && latency < 100ms",
            }"#
        } else {
            ""
        }
    );

    fs::write(dir.join("app.swarm"), swarm_content)?;
    Ok(())
}

fn create_static_site_template(dir: &PathBuf, args: &QuickstartArgs) -> Result<()> {
    let swarm_content = format!(
        r#"// Static Website
swarm static_site {{
    agents {{
        web: WebAgent {{
            replicas: {replicas},
            
            resources {{
                cpu: 0.5,
                memory: "512Mi",
            }}
            
            network {{
                expose: 80,
                cdn: true,
            }}
        }}
    }}
    
    policies {{
        cache_control: "public, max-age=3600",
        compression: true,
        ssl: true,
    }}
}}
"#,
        replicas = args.replicas
    );

    fs::write(dir.join("app.swarm"), swarm_content)?;

    // Create sample HTML
    fs::create_dir_all(dir.join("public"))?;
    fs::write(
        dir.join("public/index.html"),
        r#"<!DOCTYPE html>
<html>
<head>
    <title>StratoSwarm Static Site</title>
</head>
<body>
    <h1>Welcome to StratoSwarm!</h1>
    <p>This static site is running on the StratoSwarm platform.</p>
</body>
</html>
"#,
    )?;

    Ok(())
}

fn create_fullstack_template(dir: &PathBuf, args: &QuickstartArgs) -> Result<()> {
    let swarm_content = format!(
        r#"// Full-Stack Application
swarm fullstack {{
    agents {{
        frontend: WebAgent {{
            replicas: {replicas},
            
            resources {{
                cpu: 1.0,
                memory: "1Gi",
            }}
            
            network {{
                expose: 3000,
                mesh: true,
            }}
        }}
        
        backend: ComputeAgent {{
            replicas: {replicas},
            
            resources {{
                cpu: 2.0,
                memory: "4Gi",{gpu_resource}
            }}{evolution_config}
            
            network {{
                expose: 8080,
            }}
        }}
        
        database: StorageAgent {{
            replicas: 1,
            
            resources {{
                cpu: 2.0,
                memory: "4Gi",
            }}
            
            storage {{
                type: "persistent",
                size: "50Gi",
            }}
        }}
        
        cache: StorageAgent {{
            replicas: 1,
            
            resources {{
                cpu: 1.0,
                memory: "2Gi",
            }}
            
            storage {{
                type: "memory",
                size: "2Gi",
            }}
        }}
    }}
    
    connections {{
        frontend -> backend: {{
            protocol: "http",
            path: "/api",
        }}
        backend -> database: {{
            protocol: "postgresql",
        }}
        backend -> cache: {{
            protocol: "redis",
        }}
    }}
    
    policies {{
        zero_downtime_updates: true,
        session_affinity: true,
        rate_limiting: "1000/hour",
    }}
}}
"#,
        replicas = args.replicas,
        gpu_resource = if args.gpu {
            "\n                gpu: optional(0.5),"
        } else {
            ""
        },
        evolution_config = if args.evolution {
            r#"
            
            evolution {
                strategy: "conservative",
                fitness: "response_time < 200ms && error_rate < 0.01",
            }"#
        } else {
            ""
        }
    );

    fs::write(dir.join("app.swarm"), swarm_content)?;

    // Create frontend and backend directories
    fs::create_dir_all(dir.join("frontend"))?;
    fs::create_dir_all(dir.join("backend"))?;

    Ok(())
}

fn create_readme(dir: &PathBuf, name: &str, args: &QuickstartArgs) -> Result<()> {
    let readme = format!(
        r#"# {}

A {} application running on StratoSwarm.

## Quick Start

1. Deploy the application:
   ```bash
   stratoswarm deploy app.swarm
   ```

2. Check status:
   ```bash
   stratoswarm status
   ```

3. View logs:
   ```bash
   stratoswarm logs <agent-name>
   ```

## Configuration

- **Template**: {}
- **GPU Support**: {}
- **Evolution**: {}
- **Initial Replicas**: {}

## Architecture

See `app.swarm` for the complete application architecture.

## Scaling

Scale agents as needed:
```bash
stratoswarm scale <agent>=<replicas>
```

## Evolution

{}

---
Generated by StratoSwarm CLI
"#,
        name,
        args.template,
        args.template,
        if args.gpu { "Enabled" } else { "Disabled" },
        if args.evolution {
            "Enabled"
        } else {
            "Disabled"
        },
        args.replicas,
        if args.evolution {
            "This application has evolution enabled. Run evolution cycles with:\n```bash\nstratoswarm evolve <agent> --generations 100\n```"
        } else {
            "Evolution is not enabled for this application."
        }
    );

    fs::write(dir.join("README.md"), readme)?;
    Ok(())
}

fn create_gitignore(dir: &PathBuf) -> Result<()> {
    let gitignore = r#"# StratoSwarm
.stratoswarm/
*.swarm.lock

# Python
__pycache__/
*.py[cod]
*$py.class
.env
venv/

# Node
node_modules/
dist/
build/

# General
.DS_Store
*.log
"#;

    fs::write(dir.join(".gitignore"), gitignore)?;
    Ok(())
}

impl std::fmt::Display for Template {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Template::WebApi => write!(f, "web-api"),
            Template::Microservices => write!(f, "microservices"),
            Template::MlPipeline => write!(f, "ml-pipeline"),
            Template::DataPipeline => write!(f, "data-pipeline"),
            Template::StaticSite => write!(f, "static-site"),
            Template::FullStack => write!(f, "full-stack"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_template_display() {
        assert_eq!(Template::WebApi.to_string(), "web-api");
        assert_eq!(Template::Microservices.to_string(), "microservices");
        assert_eq!(Template::MlPipeline.to_string(), "ml-pipeline");
    }

    #[tokio::test]
    async fn test_create_web_api_template() {
        let temp_dir = TempDir::new().unwrap();

        let args = QuickstartArgs {
            template: Template::WebApi,
            gpu: false,
            output: temp_dir.path().to_path_buf(),
            name: Some("test-app".to_string()),
            replicas: 3,
            evolution: true,
            yes: true,
        };

        let result = execute(args).await;
        assert!(result.is_ok());

        // Check files were created
        let project_dir = temp_dir.path().join("test-app");
        assert!(project_dir.join("app.swarm").exists());
        assert!(project_dir.join("api.py").exists());
        assert!(project_dir.join("requirements.txt").exists());
        assert!(project_dir.join("README.md").exists());
        assert!(project_dir.join(".gitignore").exists());
    }

    #[tokio::test]
    async fn test_create_all_templates() {
        let templates = vec![
            Template::WebApi,
            Template::Microservices,
            Template::MlPipeline,
            Template::DataPipeline,
            Template::StaticSite,
            Template::FullStack,
        ];

        for template in templates {
            let temp_dir = TempDir::new().unwrap();

            let args = QuickstartArgs {
                template: template.clone(),
                gpu: true,
                output: temp_dir.path().to_path_buf(),
                name: None,
                replicas: 2,
                evolution: false,
                yes: true,
            };

            let result = execute(args).await;
            assert!(
                result.is_ok(),
                "Failed to create {} template",
                template.to_string()
            );
        }
    }

    #[tokio::test]
    async fn test_directory_exists_error() {
        let temp_dir = TempDir::new().unwrap();
        let project_name = "existing-app";

        // Create existing directory
        fs::create_dir_all(temp_dir.path().join(project_name)).unwrap();

        let args = QuickstartArgs {
            template: Template::WebApi,
            gpu: false,
            output: temp_dir.path().to_path_buf(),
            name: Some(project_name.to_string()),
            replicas: 1,
            evolution: false,
            yes: false,
        };

        let result = execute(args).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }
}
