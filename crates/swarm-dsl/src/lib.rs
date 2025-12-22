//! StratoSwarm DSL Parser and Compiler
//!
//! This crate provides parsing and compilation for the .swarm domain-specific language,
//! enabling declarative infrastructure definitions without YAML.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use anyhow::Result;
use thiserror::Error;

pub mod ast;
pub mod compiler;
pub mod parser;
pub mod types;
pub mod validator;

pub use ast::{SwarmDefinition, SwarmFile};
pub use compiler::SwarmCompiler;
pub use parser::SwarmParser;
pub use types::{AgentConfig, AgentSpec, NetworkConfig, PersonalityTraits, ResourceRequirements};
pub use validator::SwarmValidator;

/// Errors that can occur during DSL processing
#[derive(Error, Debug)]
pub enum DslError {
    /// Parse error with location information
    #[error("Parse error at line {line}, column {column}: {message}")]
    ParseError {
        /// Line number where error occurred
        line: usize,
        /// Column number where error occurred
        column: usize,
        /// Error message
        message: String,
    },

    /// Validation error
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// Compilation error
    #[error("Compilation error: {0}")]
    CompilationError(String),

    /// Template error
    #[error("Template error: {0}")]
    TemplateError(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Parse a .swarm file from a string
pub fn parse(input: &str) -> Result<SwarmFile, DslError> {
    let parser = SwarmParser::new();
    parser.parse(input)
}

/// Compile a parsed SwarmFile into agent specifications
pub fn compile(swarm_file: SwarmFile) -> Result<Vec<AgentSpec>, DslError> {
    let compiler = SwarmCompiler::new();
    compiler.compile(swarm_file)
}

/// Parse and compile a .swarm file in one step
pub fn parse_and_compile(input: &str) -> Result<Vec<AgentSpec>, DslError> {
    let swarm_file = parse(input)?;
    compile(swarm_file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_swarm() {
        let input = r#"
            swarm myapp {
                agents {
                    frontend: WebAgent {
                        replicas: 3,
                        tier_preference: ["GPU", "CPU"],
                    }
                }
            }
        "#;

        let result = parse(input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_with_connections() {
        let input = r#"
            swarm myapp {
                agents {
                    frontend: WebAgent {
                        replicas: 3,
                    }
                    backend: ComputeAgent {
                        replicas: 5,
                    }
                }
                
                connections {
                    frontend -> backend: {
                        protocol: "grpc",
                    }
                }
            }
        "#;

        let result = parse(input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_with_policies() {
        let input = r#"
            swarm myapp {
                policies {
                    zero_downtime_updates: true,
                    rollback_on: "error_rate > 5%",
                }
            }
        "#;

        let result = parse(input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_error_invalid_syntax() {
        let input = r#"
            swarm myapp {
                invalid syntax here
            }
        "#;

        let result = parse(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_compile_simple_swarm() {
        let input = r#"
            swarm myapp {
                agents {
                    frontend: WebAgent {
                        replicas: 3,
                    }
                }
            }
        "#;

        let result = parse_and_compile(input);
        assert!(result.is_ok());

        let specs = result.unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "frontend");
    }
}
