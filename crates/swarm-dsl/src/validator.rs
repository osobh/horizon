//! Validator for .swarm DSL AST

use crate::ast::*;
use crate::DslError;
use anyhow::Result;
use std::collections::HashSet;

/// Validator for SwarmFile AST
pub struct SwarmValidator {
    /// Known agent types
    known_agent_types: HashSet<String>,
    /// Known function names
    known_functions: HashSet<String>,
}

impl SwarmValidator {
    /// Create a new validator with default known types
    pub fn new() -> Self {
        let mut known_agent_types = HashSet::new();
        known_agent_types.insert("WebAgent".to_string());
        known_agent_types.insert("ComputeAgent".to_string());
        known_agent_types.insert("StorageAgent".to_string());
        known_agent_types.insert("NetworkAgent".to_string());
        known_agent_types.insert("GPUAgent".to_string());

        let mut known_functions = HashSet::new();
        known_functions.insert("exponential_backoff".to_string());
        known_functions.insert("linear_backoff".to_string());
        known_functions.insert("optional".to_string());

        Self {
            known_agent_types,
            known_functions,
        }
    }

    /// Validate a SwarmFile
    pub fn validate(&self, file: &SwarmFile) -> Result<(), DslError> {
        // Validate imports
        for import in &file.imports {
            self.validate_import(import)?;
        }

        // Validate templates
        for template in &file.templates {
            self.validate_template(template)?;
        }

        // Validate swarms
        for swarm in &file.swarms {
            self.validate_swarm(swarm)?;
        }

        Ok(())
    }

    fn validate_import(&self, import: &Import) -> Result<(), DslError> {
        if import.path.is_empty() {
            return Err(DslError::ValidationError(
                "Import path cannot be empty".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_template(&self, template: &Template) -> Result<(), DslError> {
        if template.name.is_empty() {
            return Err(DslError::ValidationError(
                "Template name cannot be empty".to_string(),
            ));
        }

        // Validate template body
        self.validate_swarm(&template.body)?;

        Ok(())
    }

    fn validate_swarm(&self, swarm: &SwarmDefinition) -> Result<(), DslError> {
        if swarm.name.is_empty() {
            return Err(DslError::ValidationError(
                "Swarm name cannot be empty".to_string(),
            ));
        }

        // Validate agents
        for (name, agent) in &swarm.agents {
            self.validate_agent(name, agent)?;
        }

        // Validate connections
        for conn in &swarm.connections {
            self.validate_connection(conn, swarm)?;
        }

        // Validate policies
        for (name, value) in &swarm.policies {
            self.validate_policy(name, value)?;
        }

        // Validate functions
        for func in &swarm.functions {
            self.validate_function(func)?;
        }

        // Validate evolution if present
        if let Some(evolution) = &swarm.evolution {
            self.validate_evolution(evolution)?;
        }

        // Validate affinity if present
        if let Some(affinity) = &swarm.affinity {
            self.validate_affinity(affinity)?;
        }

        Ok(())
    }

    fn validate_agent(&self, name: &str, agent: &Agent) -> Result<(), DslError> {
        if name.is_empty() {
            return Err(DslError::ValidationError(
                "Agent name cannot be empty".to_string(),
            ));
        }

        // Check if agent type is known
        if !self.known_agent_types.contains(&agent.agent_type) {
            return Err(DslError::ValidationError(format!(
                "Unknown agent type: {}",
                agent.agent_type
            )));
        }

        // Validate properties
        for (prop_name, value) in &agent.properties {
            self.validate_property(prop_name, value)?;
        }

        Ok(())
    }

    fn validate_connection(
        &self,
        conn: &Connection,
        swarm: &SwarmDefinition,
    ) -> Result<(), DslError> {
        // Check if source agent exists
        if !swarm.agents.contains_key(&conn.from) {
            return Err(DslError::ValidationError(format!(
                "Connection source agent '{}' not found",
                conn.from
            )));
        }

        // Check if target agent exists
        if !swarm.agents.contains_key(&conn.to) {
            return Err(DslError::ValidationError(format!(
                "Connection target agent '{}' not found",
                conn.to
            )));
        }

        // Validate connection properties
        for (prop_name, value) in &conn.properties {
            self.validate_property(prop_name, value)?;
        }

        Ok(())
    }

    fn validate_policy(&self, name: &str, value: &Value) -> Result<(), DslError> {
        if name.is_empty() {
            return Err(DslError::ValidationError(
                "Policy name cannot be empty".to_string(),
            ));
        }

        self.validate_value(value)?;
        Ok(())
    }

    fn validate_function(&self, func: &Function) -> Result<(), DslError> {
        if func.name.is_empty() {
            return Err(DslError::ValidationError(
                "Function name cannot be empty".to_string(),
            ));
        }

        if func.return_type.is_empty() {
            return Err(DslError::ValidationError(
                "Function return type cannot be empty".to_string(),
            ));
        }

        // Validate statements
        for stmt in &func.body {
            self.validate_statement(stmt)?;
        }

        Ok(())
    }

    fn validate_statement(&self, stmt: &Statement) -> Result<(), DslError> {
        if stmt.var_name.is_empty() {
            return Err(DslError::ValidationError(
                "Statement variable name cannot be empty".to_string(),
            ));
        }

        self.validate_expression(&stmt.expression)?;
        Ok(())
    }

    fn validate_expression(&self, expr: &Expression) -> Result<(), DslError> {
        match expr {
            Expression::Value(val) => self.validate_value(val)?,
            Expression::FunctionCall { name, args } => {
                if !self.known_functions.contains(name) {
                    // Allow unknown functions but warn (could be user-defined)
                    tracing::warn!("Unknown function: {}", name);
                }
                for arg in args {
                    self.validate_value(arg)?;
                }
            }
            Expression::Interpolation(inner) => {
                self.validate_expression(inner)?;
            }
        }
        Ok(())
    }

    fn validate_evolution(&self, evolution: &Evolution) -> Result<(), DslError> {
        for (prop_name, value) in &evolution.properties {
            self.validate_property(prop_name, value)?;
        }
        Ok(())
    }

    fn validate_affinity(&self, affinity: &Affinity) -> Result<(), DslError> {
        for (rule_name, value) in &affinity.rules {
            self.validate_property(rule_name, value)?;
        }
        Ok(())
    }

    fn validate_property(&self, name: &str, value: &Value) -> Result<(), DslError> {
        if name.is_empty() {
            return Err(DslError::ValidationError(
                "Property name cannot be empty".to_string(),
            ));
        }

        self.validate_value(value)?;
        Ok(())
    }

    fn validate_value(&self, value: &Value) -> Result<(), DslError> {
        match value {
            Value::String(s) => {
                // Strings are always valid, but could add length checks
                if s.len() > 10000 {
                    return Err(DslError::ValidationError(
                        "String value too long (max 10000 chars)".to_string(),
                    ));
                }
            }
            Value::Number(n) => {
                // Check for invalid numbers
                if n.is_nan() || n.is_infinite() {
                    return Err(DslError::ValidationError(
                        "Invalid number value (NaN or infinite)".to_string(),
                    ));
                }
            }
            Value::Array(arr) => {
                // Validate each array element
                for elem in arr {
                    self.validate_value(elem)?;
                }
            }
            Value::Object(obj) => {
                // Validate each object field
                for (key, val) in obj {
                    if key.is_empty() {
                        return Err(DslError::ValidationError(
                            "Object key cannot be empty".to_string(),
                        ));
                    }
                    self.validate_value(val)?;
                }
            }
            Value::Range { start, end } => {
                // Validate range
                if start > end {
                    return Err(DslError::ValidationError(format!(
                        "Invalid range: start ({}) > end ({})",
                        start, end
                    )));
                }
            }
            Value::FunctionCall { name, args } => {
                if !self.known_functions.contains(name) {
                    tracing::warn!("Unknown function in value: {}", name);
                }
                for arg in args {
                    self.validate_value(arg)?;
                }
            }
            Value::Boolean(_) | Value::Identifier(_) | Value::TierType(_) => {
                // These are always valid
            }
        }

        Ok(())
    }
}

impl Default for SwarmValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_empty_swarm() {
        let validator = SwarmValidator::new();
        let swarm = SwarmDefinition::new("test".to_string());
        let file = SwarmFile {
            imports: vec![],
            templates: vec![],
            swarms: vec![swarm],
        };

        assert!(validator.validate(&file).is_ok());
    }

    #[test]
    fn test_validate_unknown_agent_type() {
        let validator = SwarmValidator::new();
        let mut swarm = SwarmDefinition::new("test".to_string());
        swarm.agents.insert(
            "unknown".to_string(),
            Agent {
                agent_type: "UnknownAgent".to_string(),
                type_params: vec![],
                properties: Default::default(),
            },
        );

        let file = SwarmFile {
            imports: vec![],
            templates: vec![],
            swarms: vec![swarm],
        };

        let result = validator.validate(&file);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unknown agent type"));
    }

    #[test]
    fn test_validate_connection_missing_agent() {
        let validator = SwarmValidator::new();
        let mut swarm = SwarmDefinition::new("test".to_string());

        // Add connection without agents
        swarm.connections.push(Connection {
            from: "frontend".to_string(),
            to: "backend".to_string(),
            properties: Default::default(),
        });

        let file = SwarmFile {
            imports: vec![],
            templates: vec![],
            swarms: vec![swarm],
        };

        let result = validator.validate(&file);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_validate_valid_connection() {
        let validator = SwarmValidator::new();
        let mut swarm = SwarmDefinition::new("test".to_string());

        // Add agents
        swarm.agents.insert(
            "frontend".to_string(),
            Agent {
                agent_type: "WebAgent".to_string(),
                type_params: vec![],
                properties: Default::default(),
            },
        );

        swarm.agents.insert(
            "backend".to_string(),
            Agent {
                agent_type: "ComputeAgent".to_string(),
                type_params: vec![],
                properties: Default::default(),
            },
        );

        // Add connection
        swarm.connections.push(Connection {
            from: "frontend".to_string(),
            to: "backend".to_string(),
            properties: Default::default(),
        });

        let file = SwarmFile {
            imports: vec![],
            templates: vec![],
            swarms: vec![swarm],
        };

        assert!(validator.validate(&file).is_ok());
    }

    #[test]
    fn test_validate_invalid_range() {
        let validator = SwarmValidator::new();
        let value = Value::Range {
            start: 10.0,
            end: 5.0,
        };

        let result = validator.validate_value(&value);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid range"));
    }

    #[test]
    fn test_validate_nan_number() {
        let validator = SwarmValidator::new();
        let value = Value::Number(f64::NAN);

        let result = validator.validate_value(&value);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid number"));
    }

    #[test]
    fn test_validate_long_string() {
        let validator = SwarmValidator::new();
        let long_string = "a".repeat(10001);
        let value = Value::String(long_string);

        let result = validator.validate_value(&value);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too long"));
    }
}
