//! Validation functionality for code modifications

use super::types::*;
use crate::error::{EvolutionEngineError, EvolutionEngineResult};

/// Validates code modifications for safety and correctness
pub struct ModificationValidator {
    /// Dangerous patterns to check for
    dangerous_patterns: Vec<&'static str>,
}

impl ModificationValidator {
    /// Create new modification validator
    pub fn new() -> Self {
        Self {
            dangerous_patterns: vec![
                "os.system",
                "subprocess.Popen",
                "eval(",
                "exec(",
                "__import__",
                "open('/etc",
                "open('/sys",
                "rm -rf /",
                "dd if=",
                "mkfs.",
            ],
        }
    }

    /// Validate syntax of modified code
    pub fn validate_syntax(&self, code: &str, language: &str) -> EvolutionEngineResult<bool> {
        match language {
            "python" => self.validate_python_syntax(code),
            "rust" => Ok(true), // Simplified for now
            _ => Ok(true),      // Accept other languages for now
        }
    }

    /// Validate safety of code modifications
    pub fn validate_safety(&self, code: &str) -> EvolutionEngineResult<bool> {
        // Check for dangerous patterns
        for pattern in &self.dangerous_patterns {
            if code.contains(pattern) {
                return Ok(false);
            }
        }

        // Check for file system operations outside safe directories
        if code.contains("open(") {
            // Simple check - in real implementation would parse properly
            if code.contains("open(\"/") && !code.contains("open(\"/tmp") {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Check compatibility with existing code
    pub fn check_compatibility(
        &self,
        modification: &CodeModification,
        existing_code: &str,
    ) -> EvolutionEngineResult<bool> {
        // Check if dependencies are available
        for dep in &modification.dependencies {
            let import_pattern = format!("import {}", dep);
            let from_import_pattern = format!("from {} import", dep);

            if !existing_code.contains(&import_pattern)
                && !existing_code.contains(&from_import_pattern)
            {
                // Would need to be imported - check if it's a standard library
                if !self.is_standard_library(dep) {
                    return Ok(false);
                }
            }
        }

        // Check if modification targets valid location
        if !modification.target_file.is_empty() {
            // In real implementation, would check if file exists
            return Ok(true);
        }

        Ok(true)
    }

    // Helper methods

    fn validate_python_syntax(&self, code: &str) -> EvolutionEngineResult<bool> {
        // Simple validation - check for basic syntax elements
        let mut indent_level = 0;
        let mut in_string = false;
        let mut string_char = ' ';

        for line in code.lines() {
            let trimmed = line.trim();

            // Skip empty lines and comments
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            // Check for unclosed strings (simplified)
            let mut chars = line.chars();
            while let Some(ch) = chars.next() {
                if !in_string && (ch == '"' || ch == '\'') {
                    in_string = true;
                    string_char = ch;
                } else if in_string && ch == string_char {
                    // Check if escaped
                    let prev_char = line
                        .chars()
                        .nth(line.find(ch).unwrap_or(0).saturating_sub(1));
                    if prev_char != Some('\\') {
                        in_string = false;
                    }
                }
            }

            // Check for missing colons
            if trimmed.starts_with("def ") && !trimmed.ends_with(':') && !trimmed.contains("->") {
                return Ok(false);
            }
            if trimmed.starts_with("class ") && !trimmed.ends_with(':') {
                return Ok(false);
            }
            if (trimmed.starts_with("if ")
                || trimmed.starts_with("elif ")
                || trimmed.starts_with("else")
                || trimmed.starts_with("for ")
                || trimmed.starts_with("while "))
                && !trimmed.ends_with(':')
            {
                return Ok(false);
            }
        }

        Ok(!in_string) // No unclosed strings
    }

    fn is_standard_library(&self, module: &str) -> bool {
        // Common Python standard library modules
        matches!(
            module,
            "os" | "sys"
                | "re"
                | "json"
                | "time"
                | "datetime"
                | "collections"
                | "itertools"
                | "functools"
                | "pathlib"
                | "typing"
                | "enum"
                | "dataclasses"
                | "abc"
                | "io"
        )
    }
}
