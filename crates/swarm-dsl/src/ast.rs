//! Abstract Syntax Tree for the .swarm DSL

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// A complete .swarm file
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SwarmFile {
    /// Import statements
    pub imports: Vec<Import>,
    /// Template definitions
    pub templates: Vec<Template>,
    /// Swarm definitions
    pub swarms: Vec<SwarmDefinition>,
}

/// Import statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Import {
    /// Path to import from
    pub path: String,
    /// Optional alias
    pub alias: Option<String>,
}

/// Template definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Template {
    /// Template name
    pub name: String,
    /// Template parameters
    pub parameters: Vec<Parameter>,
    /// Template body (swarm definition)
    pub body: SwarmDefinition,
}

/// Template or function parameter
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Parameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: String,
}

/// Swarm definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SwarmDefinition {
    /// Swarm name
    pub name: String,
    /// Agent definitions
    pub agents: IndexMap<String, Agent>,
    /// Connection definitions
    pub connections: Vec<Connection>,
    /// Policy definitions
    pub policies: IndexMap<String, Value>,
    /// Function definitions
    pub functions: Vec<Function>,
    /// Evolution configuration
    pub evolution: Option<Evolution>,
    /// Affinity rules
    pub affinity: Option<Affinity>,
}

/// Agent definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Agent {
    /// Agent type (e.g., WebAgent, ComputeAgent)
    pub agent_type: String,
    /// Type parameters (for generic agents)
    pub type_params: Vec<String>,
    /// Agent properties
    pub properties: IndexMap<String, Value>,
}

/// Connection between agents
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Connection {
    /// Source agent
    pub from: String,
    /// Target agent
    pub to: String,
    /// Connection properties
    pub properties: IndexMap<String, Value>,
}

/// Evolution configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Evolution {
    /// Evolution properties
    pub properties: IndexMap<String, Value>,
}

/// Affinity rules
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Affinity {
    /// Affinity rules
    pub rules: IndexMap<String, Value>,
}

/// Function definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Function {
    /// Function name
    pub name: String,
    /// Function parameters
    pub parameters: Vec<Parameter>,
    /// Return type
    pub return_type: String,
    /// Function body (statements)
    pub body: Vec<Statement>,
}

/// Statement in a function
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Statement {
    /// Variable name
    pub var_name: String,
    /// Expression to assign
    pub expression: Expression,
}

/// Expression
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expression {
    /// Value expression
    Value(Value),
    /// Function call
    FunctionCall {
        /// Function name
        name: String,
        /// Arguments
        args: Vec<Value>,
    },
    /// Interpolation
    Interpolation(Box<Expression>),
}

/// Value types in the DSL
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    /// String value
    String(String),
    /// Number value (integer or float)
    Number(f64),
    /// Boolean value
    Boolean(bool),
    /// Identifier reference
    Identifier(String),
    /// Array of values
    Array(Vec<Value>),
    /// Object with key-value pairs
    Object(IndexMap<String, Value>),
    /// Range (e.g., 3..10)
    Range {
        /// Start value
        start: f64,
        /// End value
        end: f64,
    },
    /// Function call
    FunctionCall {
        /// Function name
        name: String,
        /// Arguments
        args: Vec<Value>,
    },
    /// Tier type
    TierType(TierType),
}

/// Tier types for resource allocation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TierType {
    /// GPU tier
    GPU,
    /// CPU tier
    CPU,
    /// NVMe tier
    NVMe,
    /// Memory tier
    Memory,
}

impl SwarmFile {
    /// Create a new empty SwarmFile
    pub fn new() -> Self {
        Self {
            imports: Vec::new(),
            templates: Vec::new(),
            swarms: Vec::new(),
        }
    }
}

impl Default for SwarmFile {
    fn default() -> Self {
        Self::new()
    }
}

impl SwarmDefinition {
    /// Create a new SwarmDefinition with the given name
    pub fn new(name: String) -> Self {
        Self {
            name,
            agents: IndexMap::new(),
            connections: Vec::new(),
            policies: IndexMap::new(),
            functions: Vec::new(),
            evolution: None,
            affinity: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swarm_file_creation() {
        let mut file = SwarmFile::new();
        assert!(file.imports.is_empty());
        assert!(file.templates.is_empty());
        assert!(file.swarms.is_empty());

        let swarm = SwarmDefinition::new("test".to_string());
        file.swarms.push(swarm);
        assert_eq!(file.swarms.len(), 1);
    }

    #[test]
    fn test_agent_creation() {
        let mut agent = Agent {
            agent_type: "WebAgent".to_string(),
            type_params: vec![],
            properties: IndexMap::new(),
        };

        agent
            .properties
            .insert("replicas".to_string(), Value::Number(3.0));

        assert_eq!(agent.agent_type, "WebAgent");
        assert_eq!(agent.properties.len(), 1);
    }

    #[test]
    fn test_value_types() {
        let string_val = Value::String("test".to_string());
        let num_val = Value::Number(42.0);
        let bool_val = Value::Boolean(true);
        let range_val = Value::Range {
            start: 1.0,
            end: 10.0,
        };

        assert!(matches!(string_val, Value::String(_)));
        assert!(matches!(num_val, Value::Number(_)));
        assert!(matches!(bool_val, Value::Boolean(_)));
        assert!(matches!(range_val, Value::Range { .. }));
    }

    #[test]
    fn test_connection_creation() {
        let mut conn = Connection {
            from: "frontend".to_string(),
            to: "backend".to_string(),
            properties: IndexMap::new(),
        };

        conn.properties
            .insert("protocol".to_string(), Value::String("grpc".to_string()));

        assert_eq!(conn.from, "frontend");
        assert_eq!(conn.to, "backend");
        assert_eq!(conn.properties.len(), 1);
    }
}
