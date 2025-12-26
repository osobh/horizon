//! Response templates for natural language generation

use crate::command_generator::GeneratedCommand;
use crate::error::{AssistantError, AssistantResult};
use crate::parser::{Intent, ParsedQuery};
use crate::query_engine::QueryResult;
use handlebars::{Context, Handlebars, Helper, HelperResult, Output, RenderContext};
use serde_json::{json, Value};
use std::collections::HashMap;

/// Template engine for generating natural language responses
pub struct TemplateEngine {
    handlebars: Handlebars<'static>,
}

impl TemplateEngine {
    pub fn new() -> AssistantResult<Self> {
        let mut handlebars = Handlebars::new();
        handlebars.set_strict_mode(false);

        // Register custom helpers
        handlebars.register_helper("pluralize", Box::new(pluralize_helper));
        handlebars.register_helper("format_duration", Box::new(format_duration_helper));
        handlebars.register_helper("format_bytes", Box::new(format_bytes_helper));
        handlebars.register_helper("join", Box::new(join_helper));

        // Register templates
        Self::register_templates(&mut handlebars)?;

        Ok(Self { handlebars })
    }

    /// Render a response template
    pub fn render_template(&self, template_name: &str, data: &Value) -> AssistantResult<String> {
        self.handlebars
            .render(template_name, data)
            .map_err(|e| AssistantError::TemplateError(e.to_string()))
    }

    /// Register all response templates
    fn register_templates(handlebars: &mut Handlebars) -> AssistantResult<()> {
        // Deploy templates
        handlebars.register_template_string("deploy_success", 
            "✅ Successfully deployed **{{target}}**{{#if source}} from {{source}}{{/if}}.\n\
             {{#if command}}Command: `{{command.command}} {{join command.args ' '}}`{{/if}}\n\
             {{#if rollback_available}}💡 Use `rollback {{target}}` if you need to undo this deployment.{{/if}}"
        ).map_err(|e| AssistantError::TemplateError(e.to_string()))?;

        handlebars
            .register_template_string(
                "deploy_confirm",
                "🚀 Ready to deploy **{{target}}**{{#if source}} from {{source}}{{/if}}.\n\
             This will:\n\
             {{#each actions}}- {{this}}\n{{/each}}\
             \nProceed? (y/n)",
            )
            .map_err(|e| AssistantError::TemplateError(e.to_string()))?;

        // Scale templates
        handlebars.register_template_string("scale_success",
            "📊 Successfully scaled **{{target}}**{{#if replicas}} to {{replicas}} {{pluralize replicas 'replica' 'replicas'}}{{/if}}.\n\
             {{#if resources}}Resources: {{#if resources.cpu}}{{resources.cpu}} CPU{{/if}}{{#if resources.memory}}, {{resources.memory}} memory{{/if}}{{#if resources.gpu}}, {{resources.gpu}} GPU{{/if}}{{/if}}"
        ).map_err(|e| AssistantError::TemplateError(e.to_string()))?;

        // Query templates
        handlebars.register_template_string("query_results",
            "📋 Found {{result_count}} {{pluralize result_count 'result' 'results'}} for **{{resource_type}}**:\n\n\
             {{#each results}}\
             **{{this.id}}** ({{this.resource_type}})\n\
             {{#each this.data}}  {{@key}}: {{this}}\n{{/each}}\
             {{#unless @last}}\n{{/unless}}\
             {{/each}}"
        ).map_err(|e| AssistantError::TemplateError(e.to_string()))?;

        handlebars
            .register_template_string(
                "query_empty",
                "🔍 No {{resource_type}} found matching your criteria.\n\
             Try:\n\
             - Broadening your search terms\n\
             - Checking different resource types\n\
             - Using `status all` for a complete overview",
            )
            .map_err(|e| AssistantError::TemplateError(e.to_string()))?;

        // Status templates
        handlebars
            .register_template_string(
                "status_overview",
                "🌐 **StratoSwarm Cluster Status**\n\n\
             {{#with system_data}}\
             **Agents**: {{total_agents}} active\n\
             **Nodes**: {{active_nodes}} online\n\
             **Applications**: {{running_apps}} running\n\
             **GPU Utilization**: {{gpu_utilization}}\n\
             **Memory Usage**: {{memory_usage}}\n\
             **Evolution**: {{#if evolution_active}}✅ Active{{else}}⏸️ Paused{{/if}}\n\
             {{/with}}\
             \n💡 Use `show agents` or `show nodes` for detailed information.",
            )
            .map_err(|e| AssistantError::TemplateError(e.to_string()))?;

        // Debug templates
        handlebars
            .register_template_string(
                "debug_report",
                "🔧 **Debug Report for {{target}}**\n\n\
             **Status**: {{status}}\n\
             **Resource Usage**:\n\
             - CPU: {{cpu_usage}}\n\
             - Memory: {{memory_usage}}\n\
             - GPU: {{gpu_usage}}\n\
             - Network: {{network_connections}} connections\n\n\
             {{#if recent_errors.length}}\
             **Recent Errors**:\n\
             {{#each recent_errors}}- {{this}}\n{{/each}}\n\
             {{else}}\
             ✅ No recent errors detected.\n\
             {{/if}}\
             {{#if recommendations.length}}\
             **Recommendations**:\n\
             {{#each recommendations}}- {{this}}\n{{/each}}\
             {{/if}}",
            )
            .map_err(|e| AssistantError::TemplateError(e.to_string()))?;

        // Logs templates
        handlebars
            .register_template_string(
                "logs_display",
                "📜 **Logs for {{target}}** ({{total_lines}} lines):\n\n\
             ```\n\
             {{#each lines}}{{this}}\n{{/each}}\
             ```\n\n\
             💡 Use `logs {{target}} --follow` for live updates.",
            )
            .map_err(|e| AssistantError::TemplateError(e.to_string()))?;

        // Evolution templates
        handlebars.register_template_string("evolution_started",
            "🧬 **Evolution started for {{target}}**\n\n\
             {{#if fitness_function}}Fitness Function: `{{fitness_function}}`\n{{/if}}\
             Generations: {{generations}}\n\
             Population Size: {{population_size}}\n\n\
             🔄 Evolution is running in the background. Use `status {{target}}` to monitor progress."
        ).map_err(|e| AssistantError::TemplateError(e.to_string()))?;

        // Help templates
        handlebars.register_template_string("help_general",
            "🆘 **StratoSwarm AI Assistant Help**\n\n\
             I can help you with:\n\
             - **Deploy**: `deploy myapp` or `deploy from github.com/user/repo`\n\
             - **Scale**: `scale myapp to 5` or `scale with 4 CPU and 8GB memory`\n\
             - **Query**: `show running agents` or `list all nodes`\n\
             - **Debug**: `debug myapp` or `troubleshoot performance issues`\n\
             - **Status**: `status` or `status myapp`\n\
             - **Logs**: `logs myapp` or `tail logs for service`\n\
             - **Evolve**: `evolve myapp for better performance`\n\
             - **Optimize**: `optimize myapp for latency`\n\n\
             💡 Try asking in natural language! I understand context and can help with follow-up questions."
        ).map_err(|e| AssistantError::TemplateError(e.to_string()))?;

        handlebars
            .register_template_string(
                "help_topic",
                "🔍 **Help: {{topic}}**\n\n\
             {{#if content}}{{content}}{{else}}\
             I don't have specific help for '{{topic}}' yet.\n\
             Try:\n\
             - `help` for general assistance\n\
             - Being more specific about what you need\n\
             - Checking the documentation\n\
             {{/if}}",
            )
            .map_err(|e| AssistantError::TemplateError(e.to_string()))?;

        // Error templates
        handlebars
            .register_template_string(
                "error_general",
                "❌ **Error**: {{error_message}}\n\n\
             {{#if suggestions.length}}\
             Try:\n\
             {{#each suggestions}}- {{this}}\n{{/each}}\
             {{else}}\
             💡 Use `help` for general assistance or try rephrasing your request.\n\
             {{/if}}",
            )
            .map_err(|e| AssistantError::TemplateError(e.to_string()))?;

        handlebars.register_template_string("confidence_low",
            "🤔 I'm not completely sure about your request ({{confidence}}% confidence).\n\n\
             I think you want to: **{{interpreted_intent}}**\n\n\
             {{#if command}}Generated command: `{{command.command}} {{join command.args ' '}}`\n\n{{/if}}\
             Is this correct? If not, try being more specific or use `help` for guidance."
        ).map_err(|e| AssistantError::TemplateError(e.to_string()))?;

        Ok(())
    }
}

/// Get the appropriate template name for an intent
pub fn get_template_for_intent(intent: &Intent) -> AssistantResult<&'static str> {
    match intent {
        Intent::Deploy { .. } => Ok("deploy_success"),
        Intent::Scale { .. } => Ok("scale_success"),
        Intent::Query { .. } => Ok("query_results"),
        Intent::Status { .. } => Ok("status_overview"),
        Intent::Debug { .. } => Ok("debug_report"),
        Intent::Logs { .. } => Ok("logs_display"),
        Intent::Evolve { .. } => Ok("evolution_started"),
        Intent::Help { .. } => Ok("help_general"),
        Intent::Unknown { .. } => Ok("error_general"),
        _ => Ok("error_general"),
    }
}

/// Render a template with the given data
pub fn render_template(
    template_name: &str,
    parsed: &ParsedQuery,
    command: &Option<GeneratedCommand>,
    query_results: &Option<Vec<QueryResult>>,
) -> AssistantResult<String> {
    let engine = TemplateEngine::new()?;

    let data = prepare_template_data(parsed, command, query_results)?;
    engine.render_template(template_name, &data)
}

/// Prepare data for template rendering
fn prepare_template_data(
    parsed: &ParsedQuery,
    command: &Option<GeneratedCommand>,
    query_results: &Option<Vec<QueryResult>>,
) -> AssistantResult<Value> {
    let mut data = json!({
        "confidence": (parsed.confidence * 100.0) as u32,
        "interpreted_intent": format!("{:?}", parsed.intent),
    });

    // Add intent-specific data
    match &parsed.intent {
        Intent::Deploy {
            target,
            source,
            config,
        } => {
            data["target"] = json!(target);
            data["source"] = json!(source);
            data["config"] = json!(config);
            data["rollback_available"] = json!(true);
        }
        Intent::Scale {
            target,
            replicas,
            resources,
        } => {
            data["target"] = json!(target);
            data["replicas"] = json!(replicas);
            data["resources"] = json!(resources);
        }
        Intent::Query { resource_type, .. } => {
            data["resource_type"] = json!(resource_type);
            if let Some(results) = query_results {
                data["results"] = json!(results);
                data["result_count"] = json!(results.len());
            } else {
                data["result_count"] = json!(0);
            }
        }
        Intent::Status { target } => {
            data["target"] = json!(target);
            if let Some(results) = query_results {
                if let Some(system_result) = results.first() {
                    if system_result.resource_type == "system" {
                        data["system_data"] = json!(system_result.data);
                    }
                }
            }
        }
        Intent::Debug { target, .. } => {
            data["target"] = json!(target);
            if let Some(results) = query_results {
                if let Some(debug_result) = results.first() {
                    // Merge debug data into template data
                    for (key, value) in &debug_result.data {
                        data[key] = value.clone();
                    }
                }
            }
        }
        Intent::Logs { target, .. } => {
            data["target"] = json!(target);
            if let Some(results) = query_results {
                if let Some(log_result) = results.first() {
                    for (key, value) in &log_result.data {
                        data[key] = value.clone();
                    }
                }
            }
        }
        Intent::Evolve {
            target,
            fitness_function,
        } => {
            data["target"] = json!(target);
            data["fitness_function"] = json!(fitness_function);
            data["generations"] = json!(10); // Default
            data["population_size"] = json!(100); // Default
        }
        Intent::Help { topic } => {
            data["topic"] = json!(topic);
        }
        Intent::Unknown { raw_input } => {
            data["error_message"] = json!(format!("I don't understand: {}", raw_input));
            data["suggestions"] = json!(vec![
                "Try rephrasing your request",
                "Use 'help' for available commands",
                "Be more specific about what you want to do"
            ]);
        }
        _ => {}
    }

    // Add command data if available
    if let Some(cmd) = command {
        data["command"] = json!(cmd);

        // Add action descriptions for confirmation
        let actions = generate_action_descriptions(cmd);
        data["actions"] = json!(actions);
    }

    Ok(data)
}

/// Generate human-readable action descriptions for commands
fn generate_action_descriptions(command: &GeneratedCommand) -> Vec<String> {
    let mut actions = Vec::new();

    match command.command.as_str() {
        "stratoswarm" => {
            if let Some(subcommand) = command.args.get(0) {
                match subcommand.as_str() {
                    "deploy" => {
                        actions.push("Create new deployment".to_string());
                        if command.args.contains(&"--source".to_string()) {
                            actions.push("Pull source code from repository".to_string());
                        }
                        actions.push("Allocate resources for the application".to_string());
                        actions.push("Start application containers".to_string());
                    }
                    "scale" => {
                        actions.push("Adjust resource allocation".to_string());
                        if command.args.contains(&"--replicas".to_string()) {
                            actions.push("Change number of running instances".to_string());
                        }
                        actions.push("Rebalance workload distribution".to_string());
                    }
                    "rollback" => {
                        actions.push("Stop current version".to_string());
                        actions.push("Restore previous version".to_string());
                        actions.push("Update routing configuration".to_string());
                    }
                    "evolve" => {
                        actions.push("Start genetic algorithm optimization".to_string());
                        actions.push("Generate population variations".to_string());
                        actions.push("Run fitness evaluations".to_string());
                    }
                    _ => {
                        actions.push(format!("Execute {} operation", subcommand));
                    }
                }
            }
        }
        _ => {
            actions.push(format!("Execute command: {}", command.command));
        }
    }

    if command.requires_confirmation {
        actions.push("⚠️ This operation requires confirmation".to_string());
    }

    actions
}

// Helper functions for Handlebars
fn pluralize_helper(
    h: &Helper,
    _: &Handlebars,
    _: &Context,
    _: &mut RenderContext,
    out: &mut dyn Output,
) -> HelperResult {
    let count = h.param(0).and_then(|v| v.value().as_u64()).unwrap_or(0);

    let singular = h.param(1).and_then(|v| v.value().as_str()).unwrap_or("");

    let plural = h.param(2).and_then(|v| v.value().as_str()).unwrap_or("");

    let result = if count == 1 { singular } else { plural };
    out.write(result)?;
    Ok(())
}

fn format_duration_helper(
    h: &Helper,
    _: &Handlebars,
    _: &Context,
    _: &mut RenderContext,
    out: &mut dyn Output,
) -> HelperResult {
    let seconds = h.param(0).and_then(|v| v.value().as_u64()).unwrap_or(0);

    let result = if seconds < 60 {
        format!("{}s", seconds)
    } else if seconds < 3600 {
        format!("{}m {}s", seconds / 60, seconds % 60)
    } else {
        format!("{}h {}m", seconds / 3600, (seconds % 3600) / 60)
    };

    out.write(&result)?;
    Ok(())
}

fn format_bytes_helper(
    h: &Helper,
    _: &Handlebars,
    _: &Context,
    _: &mut RenderContext,
    out: &mut dyn Output,
) -> HelperResult {
    let bytes = h.param(0).and_then(|v| v.value().as_u64()).unwrap_or(0);

    let result = if bytes < 1024 {
        format!("{}B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1}GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    };

    out.write(&result)?;
    Ok(())
}

fn join_helper(
    h: &Helper,
    _: &Handlebars,
    _: &Context,
    _: &mut RenderContext,
    out: &mut dyn Output,
) -> HelperResult {
    let empty_vec = Vec::new();
    let array = h
        .param(0)
        .and_then(|v| v.value().as_array())
        .unwrap_or(&empty_vec);

    let separator = h.param(1).and_then(|v| v.value().as_str()).unwrap_or(" ");

    let result: Vec<String> = array
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();

    let joined = result.join(separator);
    out.write(&joined)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Intent;
    use std::collections::HashMap;

    #[test]
    fn test_template_engine_creation() {
        let engine = TemplateEngine::new().unwrap();
        // Just test that it creates without errors
        assert!(true);
    }

    #[test]
    fn test_deploy_template_rendering() {
        let engine = TemplateEngine::new().unwrap();

        let data = json!({
            "target": "myapp",
            "source": "github.com/user/repo",
            "rollback_available": true
        });

        let result = engine.render_template("deploy_success", &data).unwrap();
        assert!(result.contains("myapp"));
        assert!(result.contains("github.com/user/repo"));
        assert!(result.contains("rollback"));
    }

    #[test]
    fn test_query_results_template() {
        let engine = TemplateEngine::new().unwrap();

        let data = json!({
            "resource_type": "agents",
            "results": [
                {
                    "id": "agent-001",
                    "resource_type": "agent",
                    "data": {
                        "status": "running",
                        "cpu": "45%"
                    }
                }
            ]
        });

        let result = engine.render_template("query_results", &data).unwrap();
        assert!(result.contains("agent-001"));
        assert!(result.contains("running"));
    }

    #[test]
    fn test_pluralization_helper() {
        let mut engine = TemplateEngine::new().unwrap();

        let data = json!({
            "count": 1,
            "items": [{"name": "test"}]
        });

        // Test with a simple template that uses pluralize
        let template = "{{pluralize count 'item' 'items'}}";
        engine
            .handlebars
            .register_template_string("test_plural", template)
            .unwrap();

        let result = engine.render_template("test_plural", &data).unwrap();
        assert_eq!(result, "item");
    }

    #[test]
    fn test_get_template_for_intent() {
        use std::collections::HashMap;
        let intent = Intent::Deploy {
            target: "test".to_string(),
            source: None,
            config: HashMap::new(),
        };

        let template_name = get_template_for_intent(&intent).unwrap();
        assert_eq!(template_name, "deploy_success");
    }

    #[test]
    fn test_action_descriptions() {
        use std::collections::HashMap;
        let command = GeneratedCommand {
            command: "stratoswarm".to_string(),
            args: vec!["deploy".to_string(), "myapp".to_string()],
            env: HashMap::new(),
            description: "Deploy myapp".to_string(),
            requires_confirmation: false,
            impact_level: "medium".to_string(),
            rollback_command: None,
        };

        let actions = generate_action_descriptions(&command);
        assert!(!actions.is_empty());
        assert!(actions.iter().any(|a| a.contains("deployment")));
    }
}
