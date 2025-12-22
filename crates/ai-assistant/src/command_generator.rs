//! Command generation from natural language intents

use crate::error::{AssistantError, AssistantResult};
use crate::parser::{Intent, ParsedQuery, ResourceSpec};
use handlebars::Handlebars;
use serde::{Deserialize, Serialize};
// use serde_json::json;  // Not used in this file
use std::collections::HashMap;

/// Generated command ready for execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedCommand {
    /// The actual command to execute
    pub command: String,
    /// Arguments for the command
    pub args: Vec<String>,
    /// Environment variables if needed
    pub env: HashMap<String, String>,
    /// Description of what the command does
    pub description: String,
    /// Whether user confirmation is required
    pub requires_confirmation: bool,
    /// Estimated impact level (low, medium, high)
    pub impact_level: String,
    /// Rollback command if applicable
    pub rollback_command: Option<Box<GeneratedCommand>>,
}

/// Command generator that creates executable commands from intents
pub struct CommandGenerator {
    templates: Handlebars<'static>,
    command_mappings: HashMap<String, CommandTemplate>,
}

/// Template for command generation
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CommandTemplate {
    pub command: String,
    pub args_template: Vec<String>,
    pub requires_confirmation: bool,
    pub impact_level: String,
}

impl CommandGenerator {
    pub fn new() -> AssistantResult<Self> {
        let mut templates = Handlebars::new();
        templates.set_strict_mode(false);

        // Register command templates
        let command_mappings = Self::initialize_command_mappings();

        Ok(Self {
            templates,
            command_mappings,
        })
    }

    pub async fn generate(&self, parsed: &ParsedQuery) -> AssistantResult<GeneratedCommand> {
        match &parsed.intent {
            Intent::Deploy {
                target,
                source,
                config,
            } => {
                self.generate_deploy_command(target, source.as_deref(), config)
                    .await
            }
            Intent::Scale {
                target,
                replicas,
                resources,
            } => {
                self.generate_scale_command(target, *replicas, resources)
                    .await
            }
            Intent::Query {
                resource_type,
                filters,
                projection,
            } => {
                self.generate_query_command(resource_type, filters, projection.as_deref())
                    .await
            }
            Intent::Debug { target, symptoms } => {
                self.generate_debug_command(target, symptoms).await
            }
            Intent::Optimize {
                target,
                metric,
                constraints,
            } => {
                self.generate_optimize_command(target, metric, constraints)
                    .await
            }
            Intent::Status { target } => self.generate_status_command(target.as_deref()).await,
            Intent::Logs {
                target,
                follow,
                lines,
            } => self.generate_logs_command(target, *follow, *lines).await,
            Intent::Rollback { target, version } => {
                self.generate_rollback_command(target, version.as_deref())
                    .await
            }
            Intent::Evolve {
                target,
                fitness_function,
            } => {
                self.generate_evolve_command(target, fitness_function.as_deref())
                    .await
            }
            Intent::Help { topic } => self.generate_help_command(topic.as_deref()).await,
            Intent::Unknown { raw_input } => Err(AssistantError::CommandGenerationError(format!(
                "Cannot generate command for unknown intent: {}",
                raw_input
            ))),
        }
    }

    fn initialize_command_mappings() -> HashMap<String, CommandTemplate> {
        let mut mappings = HashMap::new();

        mappings.insert(
            "deploy".to_string(),
            CommandTemplate {
                command: "stratoswarm".to_string(),
                args_template: vec!["deploy".to_string()],
                requires_confirmation: false,
                impact_level: "medium".to_string(),
            },
        );

        mappings.insert(
            "scale".to_string(),
            CommandTemplate {
                command: "stratoswarm".to_string(),
                args_template: vec!["scale".to_string()],
                requires_confirmation: true,
                impact_level: "medium".to_string(),
            },
        );

        mappings.insert(
            "query".to_string(),
            CommandTemplate {
                command: "stratoswarm".to_string(),
                args_template: vec!["status".to_string()],
                requires_confirmation: false,
                impact_level: "low".to_string(),
            },
        );

        mappings
    }

    async fn generate_deploy_command(
        &self,
        target: &str,
        source: Option<&str>,
        config: &HashMap<String, String>,
    ) -> AssistantResult<GeneratedCommand> {
        let mut args = vec!["deploy".to_string()];

        // Add source if provided
        if let Some(src) = source {
            args.push("--source".to_string());
            args.push(src.to_string());
        } else {
            args.push(".".to_string()); // Current directory
        }

        // Add name
        args.push("--name".to_string());
        args.push(target.to_string());

        // Add any config options
        for (key, value) in config {
            args.push(format!("--{}", key));
            args.push(value.clone());
        }

        // Generate rollback command
        let rollback = Some(Box::new(GeneratedCommand {
            command: "stratoswarm".to_string(),
            args: vec![
                "rollback".to_string(),
                target.to_string(),
                "--to-previous".to_string(),
            ],
            env: HashMap::new(),
            description: format!("Rollback {} to previous version", target),
            requires_confirmation: true,
            impact_level: "high".to_string(),
            rollback_command: None,
        }));

        Ok(GeneratedCommand {
            command: "stratoswarm".to_string(),
            args,
            env: HashMap::new(),
            description: format!(
                "Deploy {} from {}",
                target,
                source.unwrap_or("current directory")
            ),
            requires_confirmation: false,
            impact_level: "medium".to_string(),
            rollback_command: rollback,
        })
    }

    async fn generate_scale_command(
        &self,
        target: &str,
        replicas: Option<u32>,
        resources: &Option<ResourceSpec>,
    ) -> AssistantResult<GeneratedCommand> {
        let mut args = vec!["scale".to_string(), target.to_string()];

        if let Some(count) = replicas {
            args.push("--replicas".to_string());
            args.push(count.to_string());
        }

        if let Some(res) = resources {
            if let Some(cpu) = &res.cpu {
                args.push("--cpu".to_string());
                args.push(cpu.clone());
            }
            if let Some(memory) = &res.memory {
                args.push("--memory".to_string());
                args.push(memory.clone());
            }
            if let Some(gpu) = &res.gpu {
                args.push("--gpu".to_string());
                args.push(gpu.clone());
            }
        }

        Ok(GeneratedCommand {
            command: "stratoswarm".to_string(),
            args,
            env: HashMap::new(),
            description: format!("Scale {} to {} replicas", target, replicas.unwrap_or(0)),
            requires_confirmation: true,
            impact_level: "medium".to_string(),
            rollback_command: None,
        })
    }

    async fn generate_query_command(
        &self,
        resource_type: &str,
        filters: &HashMap<String, String>,
        projection: Option<&[String]>,
    ) -> AssistantResult<GeneratedCommand> {
        let mut args = vec!["status".to_string()];

        // Parse resource type
        if resource_type.contains("agent") {
            args.push("agents".to_string());
        } else if resource_type.contains("node") {
            args.push("nodes".to_string());
        } else if resource_type.contains("app") || resource_type.contains("application") {
            args.push("apps".to_string());
        } else {
            args.push("all".to_string());
        }

        // Add filters
        for (key, value) in filters {
            args.push(format!("--{}", key));
            args.push(value.clone());
        }

        // Add projection if specified
        if let Some(fields) = projection {
            if !fields.is_empty() {
                args.push("--fields".to_string());
                args.push(fields.join(","));
            }
        }

        Ok(GeneratedCommand {
            command: "stratoswarm".to_string(),
            args,
            env: HashMap::new(),
            description: format!("Query {} resources", resource_type),
            requires_confirmation: false,
            impact_level: "low".to_string(),
            rollback_command: None,
        })
    }

    async fn generate_debug_command(
        &self,
        target: &str,
        symptoms: &[String],
    ) -> AssistantResult<GeneratedCommand> {
        let mut args = vec!["debug".to_string(), target.to_string()];

        if !symptoms.is_empty() {
            args.push("--symptoms".to_string());
            args.push(symptoms.join(","));
        }

        args.push("--interactive".to_string());

        Ok(GeneratedCommand {
            command: "stratoswarm".to_string(),
            args,
            env: HashMap::new(),
            description: format!("Debug issues with {}", target),
            requires_confirmation: false,
            impact_level: "low".to_string(),
            rollback_command: None,
        })
    }

    async fn generate_optimize_command(
        &self,
        target: &str,
        metric: &str,
        constraints: &[String],
    ) -> AssistantResult<GeneratedCommand> {
        let mut args = vec!["optimize".to_string(), target.to_string()];

        args.push("--metric".to_string());
        args.push(metric.to_string());

        if !constraints.is_empty() {
            args.push("--constraints".to_string());
            args.push(constraints.join(","));
        }

        Ok(GeneratedCommand {
            command: "stratoswarm".to_string(),
            args,
            env: HashMap::new(),
            description: format!("Optimize {} for {}", target, metric),
            requires_confirmation: true,
            impact_level: "medium".to_string(),
            rollback_command: None,
        })
    }

    async fn generate_status_command(
        &self,
        target: Option<&str>,
    ) -> AssistantResult<GeneratedCommand> {
        let mut args = vec!["status".to_string()];

        if let Some(t) = target {
            args.push(t.to_string());
        }

        Ok(GeneratedCommand {
            command: "stratoswarm".to_string(),
            args,
            env: HashMap::new(),
            description: format!("Get status of {}", target.unwrap_or("all resources")),
            requires_confirmation: false,
            impact_level: "low".to_string(),
            rollback_command: None,
        })
    }

    async fn generate_logs_command(
        &self,
        target: &str,
        follow: bool,
        lines: Option<u32>,
    ) -> AssistantResult<GeneratedCommand> {
        let mut args = vec!["logs".to_string(), target.to_string()];

        if follow {
            args.push("--follow".to_string());
        }

        if let Some(n) = lines {
            args.push("--lines".to_string());
            args.push(n.to_string());
        }

        Ok(GeneratedCommand {
            command: "stratoswarm".to_string(),
            args,
            env: HashMap::new(),
            description: format!("View logs for {}", target),
            requires_confirmation: false,
            impact_level: "low".to_string(),
            rollback_command: None,
        })
    }

    async fn generate_rollback_command(
        &self,
        target: &str,
        version: Option<&str>,
    ) -> AssistantResult<GeneratedCommand> {
        let mut args = vec!["rollback".to_string(), target.to_string()];

        if let Some(v) = version {
            args.push("--to".to_string());
            args.push(v.to_string());
        } else {
            args.push("--to-previous".to_string());
        }

        Ok(GeneratedCommand {
            command: "stratoswarm".to_string(),
            args,
            env: HashMap::new(),
            description: format!(
                "Rollback {} to {}",
                target,
                version.unwrap_or("previous version")
            ),
            requires_confirmation: true,
            impact_level: "high".to_string(),
            rollback_command: None,
        })
    }

    async fn generate_evolve_command(
        &self,
        target: &str,
        fitness_function: Option<&str>,
    ) -> AssistantResult<GeneratedCommand> {
        let mut args = vec!["evolve".to_string(), target.to_string()];

        if let Some(fitness) = fitness_function {
            args.push("--fitness".to_string());
            args.push(fitness.to_string());
        }

        args.push("--generations".to_string());
        args.push("10".to_string()); // Default generations

        Ok(GeneratedCommand {
            command: "stratoswarm".to_string(),
            args,
            env: HashMap::new(),
            description: format!("Evolve {} using genetic algorithms", target),
            requires_confirmation: true,
            impact_level: "high".to_string(),
            rollback_command: None,
        })
    }

    async fn generate_help_command(
        &self,
        topic: Option<&str>,
    ) -> AssistantResult<GeneratedCommand> {
        let mut args = vec!["help".to_string()];

        if let Some(t) = topic {
            args.push(t.to_string());
        }

        Ok(GeneratedCommand {
            command: "stratoswarm".to_string(),
            args,
            env: HashMap::new(),
            description: format!("Get help for {}", topic.unwrap_or("general usage")),
            requires_confirmation: false,
            impact_level: "low".to_string(),
            rollback_command: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Intent;

    #[tokio::test]
    async fn test_deploy_command_generation() {
        let generator = CommandGenerator::new().unwrap();
        let parsed = ParsedQuery {
            intent: Intent::Deploy {
                target: "myapp".to_string(),
                source: Some("https://github.com/user/repo".to_string()),
                config: HashMap::new(),
            },
            confidence: 0.9,
            entities: HashMap::new(),
            context: Default::default(),
            raw_input: "deploy myapp from https://github.com/user/repo".to_string(),
        };

        let command = generator.generate(&parsed).await.unwrap();
        assert_eq!(command.command, "stratoswarm");
        assert_eq!(command.args[0], "deploy");
        assert!(command.args.contains(&"--source".to_string()));
        assert!(command
            .args
            .contains(&"https://github.com/user/repo".to_string()));
        assert_eq!(command.impact_level, "medium");
        assert!(command.rollback_command.is_some());
    }

    #[tokio::test]
    async fn test_scale_command_generation() {
        let generator = CommandGenerator::new().unwrap();
        let parsed = ParsedQuery {
            intent: Intent::Scale {
                target: "web-service".to_string(),
                replicas: Some(5),
                resources: Some(ResourceSpec {
                    cpu: Some("2".to_string()),
                    memory: Some("4Gi".to_string()),
                    gpu: None,
                }),
            },
            confidence: 0.85,
            entities: HashMap::new(),
            context: Default::default(),
            raw_input: "scale web-service to 5 with 2 cpu and 4Gi memory".to_string(),
        };

        let command = generator.generate(&parsed).await.unwrap();
        assert_eq!(command.command, "stratoswarm");
        assert_eq!(command.args[0], "scale");
        assert!(command.args.contains(&"--replicas".to_string()));
        assert!(command.args.contains(&"5".to_string()));
        assert!(command.args.contains(&"--cpu".to_string()));
        assert!(command.args.contains(&"--memory".to_string()));
        assert!(command.requires_confirmation);
    }

    #[tokio::test]
    async fn test_query_command_generation() {
        let generator = CommandGenerator::new().unwrap();
        let mut filters = HashMap::new();
        filters.insert("status".to_string(), "running".to_string());

        let parsed = ParsedQuery {
            intent: Intent::Query {
                resource_type: "agents".to_string(),
                filters,
                projection: Some(vec!["name".to_string(), "cpu".to_string()]),
            },
            confidence: 0.85,
            entities: HashMap::new(),
            context: Default::default(),
            raw_input: "show running agents".to_string(),
        };

        let command = generator.generate(&parsed).await.unwrap();
        assert_eq!(command.command, "stratoswarm");
        assert_eq!(command.args[0], "status");
        assert!(command.args.contains(&"agents".to_string()));
        assert!(command.args.contains(&"--status".to_string()));
        assert!(command.args.contains(&"--fields".to_string()));
        assert!(!command.requires_confirmation);
        assert_eq!(command.impact_level, "low");
    }

    #[tokio::test]
    async fn test_evolve_command_generation() {
        let generator = CommandGenerator::new().unwrap();
        let parsed = ParsedQuery {
            intent: Intent::Evolve {
                target: "ml-model".to_string(),
                fitness_function: Some("accuracy > 0.95".to_string()),
            },
            confidence: 0.8,
            entities: HashMap::new(),
            context: Default::default(),
            raw_input: "evolve ml-model for accuracy > 0.95".to_string(),
        };

        let command = generator.generate(&parsed).await.unwrap();
        assert_eq!(command.command, "stratoswarm");
        assert_eq!(command.args[0], "evolve");
        assert!(command.args.contains(&"--fitness".to_string()));
        assert!(command.args.contains(&"accuracy > 0.95".to_string()));
        assert!(command.requires_confirmation);
        assert_eq!(command.impact_level, "high");
    }
}
