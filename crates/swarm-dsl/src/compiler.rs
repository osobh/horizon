//! Compiler for .swarm DSL to agent specifications

use crate::ast::*;
use crate::types::{
    AgentConfig, AgentSpec, NetworkConfig, PersonalityTraits, ResourceRequirements,
};
use crate::validator::SwarmValidator;
use crate::DslError;
use anyhow::Result;
use std::collections::HashMap;

/// Compiler that transforms SwarmFile AST into executable agent specifications
pub struct SwarmCompiler {
    validator: SwarmValidator,
    template_engine: tera::Tera,
}

impl SwarmCompiler {
    /// Create a new compiler instance
    pub fn new() -> Self {
        Self {
            validator: SwarmValidator::new(),
            template_engine: tera::Tera::default(),
        }
    }

    /// Compile a SwarmFile into agent specifications
    pub fn compile(&self, file: SwarmFile) -> Result<Vec<AgentSpec>, DslError> {
        // Validate first
        self.validator.validate(&file)?;

        let mut all_specs = Vec::new();

        // Process each swarm definition
        for swarm in file.swarms {
            let specs = self.compile_swarm(swarm)?;
            all_specs.extend(specs);
        }

        Ok(all_specs)
    }

    /// Compile a single swarm definition
    fn compile_swarm(&self, swarm: SwarmDefinition) -> Result<Vec<AgentSpec>, DslError> {
        let mut specs = Vec::new();

        // Compile each agent
        for (name, agent) in &swarm.agents {
            let spec = self.compile_agent(name.clone(), agent.clone(), &swarm)?;
            specs.push(spec);
        }

        Ok(specs)
    }

    /// Compile an agent definition into an AgentSpec
    fn compile_agent(
        &self,
        name: String,
        agent: Agent,
        swarm: &SwarmDefinition,
    ) -> Result<AgentSpec, DslError> {
        // Extract resources
        let resources = self.extract_resources(&agent.properties)?;

        // Extract network configuration
        let network = self.extract_network_config(&agent.properties)?;

        // Extract personality traits
        let personality = self.extract_personality(&agent.properties)?;

        // Extract replicas
        let replicas = self.extract_replicas(&agent.properties)?;

        // Find connections for this agent
        let connections = self.find_agent_connections(&name, &swarm.connections);

        // Create agent configuration
        let config = AgentConfig {
            resources,
            network,
            personality,
            evolution_enabled: self.is_evolution_enabled(&agent.properties, &swarm.evolution),
            tier_preferences: self.extract_tier_preferences(&agent.properties)?,
        };

        Ok(AgentSpec {
            name,
            agent_type: agent.agent_type,
            replicas,
            config,
            connections,
            metadata: self.extract_metadata(&agent.properties)?,
        })
    }

    /// Extract resource requirements from agent properties
    fn extract_resources(
        &self,
        properties: &indexmap::IndexMap<String, Value>,
    ) -> Result<ResourceRequirements, DslError> {
        let mut resources = ResourceRequirements::default();

        // Check for direct resource properties
        if let Some(Value::Number(cpu)) = properties.get("cpu") {
            resources.cpu = *cpu;
        }

        if let Some(Value::String(memory)) = properties.get("memory") {
            resources.memory = self.parse_memory_string(memory)?;
        }

        if let Some(Value::Number(gpu)) = properties.get("gpu") {
            resources.gpu = Some(*gpu);
        }

        // Check for nested resources object
        if let Some(Value::Object(res_obj)) = properties.get("resources") {
            if let Some(Value::Number(cpu)) = res_obj.get("cpu") {
                resources.cpu = *cpu;
            }

            if let Some(Value::String(memory)) = res_obj.get("memory") {
                resources.memory = self.parse_memory_string(memory)?;
            }

            if let Some(Value::Number(gpu)) = res_obj.get("gpu") {
                resources.gpu = Some(*gpu);
            }

            // Handle optional GPU
            if let Some(Value::FunctionCall { name, args }) = res_obj.get("gpu") {
                if name == "optional" && !args.is_empty() {
                    if let Value::Number(gpu) = &args[0] {
                        resources.gpu = Some(*gpu);
                        resources.gpu_optional = true;
                    }
                }
            }
        }

        // Check for requires_gpu flag
        if let Some(Value::Boolean(true)) = properties.get("requires_gpu") {
            if resources.gpu.is_none() {
                resources.gpu = Some(1.0);
            }
        }

        Ok(resources)
    }

    /// Extract network configuration
    fn extract_network_config(
        &self,
        properties: &indexmap::IndexMap<String, Value>,
    ) -> Result<NetworkConfig, DslError> {
        let mut network = NetworkConfig::default();

        // Check for direct network properties
        if let Some(Value::Number(port)) = properties.get("expose") {
            network.expose_ports = vec![*port as u16];
        }

        // Check for nested network object
        if let Some(Value::Object(net_obj)) = properties.get("network") {
            if let Some(Value::Number(port)) = net_obj.get("expose") {
                network.expose_ports = vec![*port as u16];
            }

            if let Some(Value::Boolean(mesh)) = net_obj.get("mesh") {
                network.enable_mesh = *mesh;
            }

            if let Some(Value::String(lb)) = net_obj.get("load_balance") {
                network.load_balance_strategy = lb.clone();
            }
        }

        Ok(network)
    }

    /// Extract personality traits
    fn extract_personality(
        &self,
        properties: &indexmap::IndexMap<String, Value>,
    ) -> Result<PersonalityTraits, DslError> {
        let mut personality = PersonalityTraits::default();

        if let Some(Value::Object(pers_obj)) = properties.get("personality") {
            if let Some(Value::Number(risk)) = pers_obj.get("risk_tolerance") {
                personality.risk_tolerance = *risk as f32;
            }

            if let Some(Value::Number(coop)) = pers_obj.get("cooperation") {
                personality.cooperation = *coop as f32;
            }

            if let Some(Value::Number(explore)) = pers_obj.get("exploration") {
                personality.exploration = *explore as f32;
            }

            if let Some(Value::Number(efficiency)) = pers_obj.get("efficiency_focus") {
                personality.efficiency_focus = *efficiency as f32;
            }

            if let Some(Value::Number(stability)) = pers_obj.get("stability_preference") {
                personality.stability_preference = *stability as f32;
            }
        }

        Ok(personality)
    }

    /// Extract replica count or range
    fn extract_replicas(
        &self,
        properties: &indexmap::IndexMap<String, Value>,
    ) -> Result<(u32, Option<u32>), DslError> {
        if let Some(value) = properties.get("replicas") {
            match value {
                Value::Number(n) => Ok((*n as u32, None)),
                Value::Range { start, end } => Ok((*start as u32, Some(*end as u32))),
                _ => Err(DslError::CompilationError(
                    "Invalid replicas value type".to_string(),
                )),
            }
        } else {
            // Default to 1 replica
            Ok((1, None))
        }
    }

    /// Extract tier preferences
    fn extract_tier_preferences(
        &self,
        properties: &indexmap::IndexMap<String, Value>,
    ) -> Result<Vec<String>, DslError> {
        if let Some(Value::Array(tiers)) = properties.get("tier_preference") {
            tiers
                .iter()
                .map(|v| match v {
                    Value::String(s) => Ok(s.clone()),
                    Value::TierType(t) => Ok(format!("{:?}", t)),
                    _ => Err(DslError::CompilationError(
                        "Invalid tier preference type".to_string(),
                    )),
                })
                .collect()
        } else {
            Ok(vec![])
        }
    }

    /// Check if evolution is enabled for the agent
    fn is_evolution_enabled(
        &self,
        properties: &indexmap::IndexMap<String, Value>,
        swarm_evolution: &Option<Evolution>,
    ) -> bool {
        // Check agent-level evolution
        if let Some(Value::Object(evo_obj)) = properties.get("evolution") {
            if let Some(Value::Boolean(enabled)) = evo_obj.get("enabled") {
                return *enabled;
            }
            // If evolution object exists but no explicit enabled flag, assume true
            return true;
        }

        // Check swarm-level evolution
        if let Some(evolution) = swarm_evolution {
            if let Some(Value::Boolean(enabled)) = evolution.properties.get("enabled") {
                return *enabled;
            }
            // If evolution block exists, assume enabled
            return true;
        }

        false
    }

    /// Find connections for a specific agent
    fn find_agent_connections(&self, agent_name: &str, connections: &[Connection]) -> Vec<String> {
        connections
            .iter()
            .filter_map(|conn| {
                if conn.from == agent_name {
                    Some(conn.to.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Extract metadata from properties
    fn extract_metadata(
        &self,
        properties: &indexmap::IndexMap<String, Value>,
    ) -> Result<HashMap<String, String>, DslError> {
        let mut metadata = HashMap::new();

        // Add string properties as metadata
        for (key, value) in properties {
            // Skip known properties
            if matches!(
                key.as_str(),
                "replicas"
                    | "cpu"
                    | "memory"
                    | "gpu"
                    | "resources"
                    | "network"
                    | "personality"
                    | "evolution"
                    | "tier_preference"
            ) {
                continue;
            }

            match value {
                Value::String(s) => {
                    metadata.insert(key.clone(), s.clone());
                }
                Value::Number(n) => {
                    metadata.insert(key.clone(), n.to_string());
                }
                Value::Boolean(b) => {
                    metadata.insert(key.clone(), b.to_string());
                }
                _ => {}
            }
        }

        Ok(metadata)
    }

    /// Parse memory string (e.g., "4Gi", "512Mi") to bytes
    fn parse_memory_string(&self, memory: &str) -> Result<u64, DslError> {
        let memory = memory.trim();

        if memory.ends_with("Gi") {
            let value = memory.trim_end_matches("Gi").parse::<f64>().map_err(|_| {
                DslError::CompilationError(format!("Invalid memory value: {}", memory))
            })?;
            Ok((value * 1024.0 * 1024.0 * 1024.0) as u64)
        } else if memory.ends_with("Mi") {
            let value = memory.trim_end_matches("Mi").parse::<f64>().map_err(|_| {
                DslError::CompilationError(format!("Invalid memory value: {}", memory))
            })?;
            Ok((value * 1024.0 * 1024.0) as u64)
        } else if memory.ends_with("Ki") {
            let value = memory.trim_end_matches("Ki").parse::<f64>().map_err(|_| {
                DslError::CompilationError(format!("Invalid memory value: {}", memory))
            })?;
            Ok((value * 1024.0) as u64)
        } else {
            // Assume bytes if no unit
            memory.parse::<u64>().map_err(|_| {
                DslError::CompilationError(format!("Invalid memory value: {}", memory))
            })
        }
    }
}

impl Default for SwarmCompiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_simple_agent() {
        let compiler = SwarmCompiler::new();

        let mut swarm = SwarmDefinition::new("test".to_string());
        let mut agent = Agent {
            agent_type: "WebAgent".to_string(),
            type_params: vec![],
            properties: indexmap::IndexMap::new(),
        };

        agent
            .properties
            .insert("replicas".to_string(), Value::Number(3.0));
        agent
            .properties
            .insert("cpu".to_string(), Value::Number(2.0));
        agent
            .properties
            .insert("memory".to_string(), Value::String("4Gi".to_string()));

        swarm.agents.insert("frontend".to_string(), agent);

        let file = SwarmFile {
            imports: vec![],
            templates: vec![],
            swarms: vec![swarm],
        };

        let result = compiler.compile(file);
        assert!(result.is_ok());

        let specs = result.unwrap();
        assert_eq!(specs.len(), 1);

        let spec = &specs[0];
        assert_eq!(spec.name, "frontend");
        assert_eq!(spec.agent_type, "WebAgent");
        assert_eq!(spec.replicas, (3, None));
        assert_eq!(spec.config.resources.cpu, 2.0);
        assert_eq!(spec.config.resources.memory, 4 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_compile_with_range_replicas() {
        let compiler = SwarmCompiler::new();

        let mut swarm = SwarmDefinition::new("test".to_string());
        let mut agent = Agent {
            agent_type: "WebAgent".to_string(),
            type_params: vec![],
            properties: indexmap::IndexMap::new(),
        };

        agent.properties.insert(
            "replicas".to_string(),
            Value::Range {
                start: 3.0,
                end: 10.0,
            },
        );
        swarm.agents.insert("frontend".to_string(), agent);

        let file = SwarmFile {
            imports: vec![],
            templates: vec![],
            swarms: vec![swarm],
        };

        let result = compiler.compile(file);
        assert!(result.is_ok());

        let specs = result.unwrap();
        let spec = &specs[0];
        assert_eq!(spec.replicas, (3, Some(10)));
    }

    #[test]
    fn test_compile_with_nested_resources() {
        let compiler = SwarmCompiler::new();

        let mut swarm = SwarmDefinition::new("test".to_string());
        let mut agent = Agent {
            agent_type: "ComputeAgent".to_string(),
            type_params: vec![],
            properties: indexmap::IndexMap::new(),
        };

        let mut resources = indexmap::IndexMap::new();
        resources.insert("cpu".to_string(), Value::Number(4.0));
        resources.insert("memory".to_string(), Value::String("8Gi".to_string()));
        resources.insert("gpu".to_string(), Value::Number(1.0));

        agent
            .properties
            .insert("resources".to_string(), Value::Object(resources));
        swarm.agents.insert("compute".to_string(), agent);

        let file = SwarmFile {
            imports: vec![],
            templates: vec![],
            swarms: vec![swarm],
        };

        let result = compiler.compile(file);
        assert!(result.is_ok());

        let specs = result.unwrap();
        let spec = &specs[0];
        assert_eq!(spec.config.resources.cpu, 4.0);
        assert_eq!(spec.config.resources.memory, 8 * 1024 * 1024 * 1024);
        assert_eq!(spec.config.resources.gpu, Some(1.0));
    }

    #[test]
    fn test_compile_with_connections() {
        let compiler = SwarmCompiler::new();

        let mut swarm = SwarmDefinition::new("test".to_string());

        // Add two agents
        swarm.agents.insert(
            "frontend".to_string(),
            Agent {
                agent_type: "WebAgent".to_string(),
                type_params: vec![],
                properties: indexmap::IndexMap::new(),
            },
        );

        swarm.agents.insert(
            "backend".to_string(),
            Agent {
                agent_type: "ComputeAgent".to_string(),
                type_params: vec![],
                properties: indexmap::IndexMap::new(),
            },
        );

        // Add connection
        swarm.connections.push(Connection {
            from: "frontend".to_string(),
            to: "backend".to_string(),
            properties: indexmap::IndexMap::new(),
        });

        let file = SwarmFile {
            imports: vec![],
            templates: vec![],
            swarms: vec![swarm],
        };

        let result = compiler.compile(file);
        assert!(result.is_ok());

        let specs = result.unwrap();
        assert_eq!(specs.len(), 2);

        // Find frontend spec
        let frontend_spec = specs.iter().find(|s| s.name == "frontend").unwrap();
        assert_eq!(frontend_spec.connections, vec!["backend"]);

        // Backend should have no outgoing connections
        let backend_spec = specs.iter().find(|s| s.name == "backend").unwrap();
        assert!(backend_spec.connections.is_empty());
    }

    #[test]
    fn test_parse_memory_values() {
        let compiler = SwarmCompiler::new();

        assert_eq!(
            compiler.parse_memory_string("4Gi").unwrap(),
            4 * 1024 * 1024 * 1024
        );
        assert_eq!(
            compiler.parse_memory_string("512Mi").unwrap(),
            512 * 1024 * 1024
        );
        assert_eq!(compiler.parse_memory_string("1024Ki").unwrap(), 1024 * 1024);
        assert_eq!(compiler.parse_memory_string("1000").unwrap(), 1000);
    }

    #[test]
    fn test_compile_with_evolution() {
        let compiler = SwarmCompiler::new();

        let mut swarm = SwarmDefinition::new("test".to_string());
        let mut agent = Agent {
            agent_type: "ComputeAgent".to_string(),
            type_params: vec![],
            properties: indexmap::IndexMap::new(),
        };

        let mut evolution = indexmap::IndexMap::new();
        evolution.insert("enabled".to_string(), Value::Boolean(true));
        evolution.insert(
            "strategy".to_string(),
            Value::String("conservative".to_string()),
        );

        agent
            .properties
            .insert("evolution".to_string(), Value::Object(evolution));
        swarm.agents.insert("evolving".to_string(), agent);

        let file = SwarmFile {
            imports: vec![],
            templates: vec![],
            swarms: vec![swarm],
        };

        let result = compiler.compile(file);
        assert!(result.is_ok());

        let specs = result.unwrap();
        let spec = &specs[0];
        assert!(spec.config.evolution_enabled);
    }
}
