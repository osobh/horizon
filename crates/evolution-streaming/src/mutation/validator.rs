//! Safety validation for mutated agents

use crate::AgentGenome;
use std::collections::HashSet;
use thiserror::Error;

/// Safety validation errors
#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Code contains forbidden keyword: {0}")]
    ForbiddenKeyword(String),

    #[error("Code too long: {actual} > {max}")]
    CodeTooLong { actual: usize, max: usize },

    #[error("Too many parameters: {actual} > {max}")]
    TooManyParameters { actual: usize, max: usize },

    #[error("Parameter out of bounds: {value} not in [{min}, {max}]")]
    ParameterOutOfBounds { value: f32, min: f32, max: f32 },

    #[error("Invalid syntax: {0}")]
    InvalidSyntax(String),

    #[error("Infinite loop detected")]
    InfiniteLoopDetected,
}

/// Safety validator for agent genomes
#[derive(Debug, Clone)]
pub struct SafetyValidator {
    forbidden_keywords: HashSet<String>,
    max_code_length: usize,
    max_parameters: usize,
    parameter_bounds: (f32, f32),
    enable_syntax_check: bool,
}

impl SafetyValidator {
    /// Create a new safety validator
    pub fn new() -> Self {
        let mut forbidden_keywords = HashSet::new();

        // Common dangerous keywords
        forbidden_keywords.insert("exec".to_string());
        forbidden_keywords.insert("eval".to_string());
        forbidden_keywords.insert("system".to_string());
        forbidden_keywords.insert("unsafe".to_string());
        forbidden_keywords.insert("transmute".to_string());
        forbidden_keywords.insert("__import__".to_string());
        forbidden_keywords.insert("import".to_string());
        forbidden_keywords.insert("include".to_string());
        forbidden_keywords.insert("require".to_string());

        Self {
            forbidden_keywords,
            max_code_length: 10_000,
            max_parameters: 1000,
            parameter_bounds: (-1000.0, 1000.0),
            enable_syntax_check: true,
        }
    }

    /// Configure maximum code length
    pub fn with_max_code_length(mut self, max_length: usize) -> Self {
        self.max_code_length = max_length;
        self
    }

    /// Configure maximum parameters
    pub fn with_max_parameters(mut self, max_params: usize) -> Self {
        self.max_parameters = max_params;
        self
    }

    /// Configure parameter bounds
    pub fn with_parameter_bounds(mut self, min: f32, max: f32) -> Self {
        self.parameter_bounds = (min, max);
        self
    }

    /// Add forbidden keyword
    pub fn add_forbidden_keyword(mut self, keyword: String) -> Self {
        self.forbidden_keywords.insert(keyword);
        self
    }

    /// Enable or disable syntax checking
    pub fn with_syntax_check(mut self, enabled: bool) -> Self {
        self.enable_syntax_check = enabled;
        self
    }

    /// Validate agent genome for safety
    pub async fn validate(&self, genome: &AgentGenome) -> Result<(), ValidationError> {
        // Check code length
        if genome.code.len() > self.max_code_length {
            return Err(ValidationError::CodeTooLong {
                actual: genome.code.len(),
                max: self.max_code_length,
            });
        }

        // Check parameter count
        if genome.parameters.len() > self.max_parameters {
            return Err(ValidationError::TooManyParameters {
                actual: genome.parameters.len(),
                max: self.max_parameters,
            });
        }

        // Check forbidden keywords
        let code_lower = genome.code.to_lowercase();
        for keyword in &self.forbidden_keywords {
            if code_lower.contains(&keyword.to_lowercase()) {
                return Err(ValidationError::ForbiddenKeyword(keyword.clone()));
            }
        }

        // Check parameter bounds
        let (min_param, max_param) = self.parameter_bounds;
        for &param in &genome.parameters {
            if param < min_param || param > max_param {
                return Err(ValidationError::ParameterOutOfBounds {
                    value: param,
                    min: min_param,
                    max: max_param,
                });
            }
        }

        // Basic syntax validation
        if self.enable_syntax_check {
            self.validate_syntax(&genome.code)?;
        }

        // Check for infinite loops
        self.check_infinite_loops(&genome.code)?;

        Ok(())
    }

    /// Basic syntax validation
    fn validate_syntax(&self, code: &str) -> Result<(), ValidationError> {
        // Simple bracket matching
        let mut bracket_count = 0;
        let mut brace_count = 0;
        let mut paren_count = 0;

        for ch in code.chars() {
            match ch {
                '[' => bracket_count += 1,
                ']' => {
                    bracket_count -= 1;
                    if bracket_count < 0 {
                        return Err(ValidationError::InvalidSyntax("Unmatched ']'".to_string()));
                    }
                }
                '{' => brace_count += 1,
                '}' => {
                    brace_count -= 1;
                    if brace_count < 0 {
                        return Err(ValidationError::InvalidSyntax("Unmatched '}'".to_string()));
                    }
                }
                '(' => paren_count += 1,
                ')' => {
                    paren_count -= 1;
                    if paren_count < 0 {
                        return Err(ValidationError::InvalidSyntax("Unmatched ')'".to_string()));
                    }
                }
                _ => {}
            }
        }

        if bracket_count != 0 {
            return Err(ValidationError::InvalidSyntax(
                "Unmatched '[' or ']'".to_string(),
            ));
        }
        if brace_count != 0 {
            return Err(ValidationError::InvalidSyntax(
                "Unmatched '{' or '}'".to_string(),
            ));
        }
        if paren_count != 0 {
            return Err(ValidationError::InvalidSyntax(
                "Unmatched '(' or ')'".to_string(),
            ));
        }

        Ok(())
    }

    /// Check for potential infinite loops
    fn check_infinite_loops(&self, code: &str) -> Result<(), ValidationError> {
        let code_lower = code.to_lowercase();

        // Simple heuristics for infinite loop detection
        let dangerous_patterns = [
            "while true",
            "while(true)",
            "while 1",
            "while(1)",
            "for(;;)",
            "loop {",
        ];

        for pattern in &dangerous_patterns {
            if code_lower.contains(pattern) {
                // Check if there's a break statement
                if !code_lower.contains("break") && !code_lower.contains("return") {
                    return Err(ValidationError::InfiniteLoopDetected);
                }
            }
        }

        Ok(())
    }

    /// Quick validation without async overhead
    pub fn validate_sync(&self, genome: &AgentGenome) -> Result<(), ValidationError> {
        // Synchronous version for performance-critical paths
        if genome.code.len() > self.max_code_length {
            return Err(ValidationError::CodeTooLong {
                actual: genome.code.len(),
                max: self.max_code_length,
            });
        }

        let (min_param, max_param) = self.parameter_bounds;
        for &param in &genome.parameters {
            if param < min_param || param > max_param {
                return Err(ValidationError::ParameterOutOfBounds {
                    value: param,
                    min: min_param,
                    max: max_param,
                });
            }
        }

        Ok(())
    }
}

impl Default for SafetyValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for FastValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Fast validator for high-throughput scenarios
#[derive(Debug, Clone)]
pub struct FastValidator {
    max_code_length: usize,
    max_parameters: usize,
    parameter_bounds: (f32, f32),
}

impl FastValidator {
    /// Create a new fast validator
    pub fn new() -> Self {
        Self {
            max_code_length: 10_000,
            max_parameters: 1000,
            parameter_bounds: (-1000.0, 1000.0),
        }
    }

    /// Validate with minimal checks for performance
    pub fn validate_fast(&self, genome: &AgentGenome) -> Result<(), ValidationError> {
        if genome.code.len() > self.max_code_length {
            return Err(ValidationError::CodeTooLong {
                actual: genome.code.len(),
                max: self.max_code_length,
            });
        }

        if genome.parameters.len() > self.max_parameters {
            return Err(ValidationError::TooManyParameters {
                actual: genome.parameters.len(),
                max: self.max_parameters,
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_safety_validator_creation() {
        let validator = SafetyValidator::new();
        assert!(!validator.forbidden_keywords.is_empty());
        assert_eq!(validator.max_code_length, 10_000);
        assert_eq!(validator.max_parameters, 1000);
    }

    #[tokio::test]
    async fn test_valid_genome() {
        let validator = SafetyValidator::new();
        let genome = AgentGenome::new("fn test() { return 42; }".to_string(), vec![1.0, 2.0]);

        let result = validator.validate(&genome).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_forbidden_keyword() {
        let validator = SafetyValidator::new();
        let genome = AgentGenome::new("exec('rm -rf /')".to_string(), vec![]);

        let result = validator.validate(&genome).await;
        assert!(matches!(result, Err(ValidationError::ForbiddenKeyword(_))));
    }

    #[tokio::test]
    async fn test_code_too_long() {
        let validator = SafetyValidator::new().with_max_code_length(10);
        let genome = AgentGenome::new("this is a very long code string".to_string(), vec![]);

        let result = validator.validate(&genome).await;
        assert!(matches!(result, Err(ValidationError::CodeTooLong { .. })));
    }

    #[tokio::test]
    async fn test_too_many_parameters() {
        let validator = SafetyValidator::new().with_max_parameters(2);
        let genome = AgentGenome::new("test".to_string(), vec![1.0, 2.0, 3.0]);

        let result = validator.validate(&genome).await;
        assert!(matches!(
            result,
            Err(ValidationError::TooManyParameters { .. })
        ));
    }

    #[tokio::test]
    async fn test_parameter_out_of_bounds() {
        let validator = SafetyValidator::new().with_parameter_bounds(-10.0, 10.0);
        let genome = AgentGenome::new("test".to_string(), vec![100.0]);

        let result = validator.validate(&genome).await;
        assert!(matches!(
            result,
            Err(ValidationError::ParameterOutOfBounds { .. })
        ));
    }

    #[tokio::test]
    async fn test_invalid_syntax() {
        let validator = SafetyValidator::new();
        let genome = AgentGenome::new("fn test() { return 42; }]".to_string(), vec![]);

        let result = validator.validate(&genome).await;
        assert!(matches!(result, Err(ValidationError::InvalidSyntax(_))));
    }

    #[tokio::test]
    async fn test_infinite_loop_detection() {
        let validator = SafetyValidator::new();
        let genome = AgentGenome::new("while true { }".to_string(), vec![]);

        let result = validator.validate(&genome).await;
        assert!(matches!(result, Err(ValidationError::InfiniteLoopDetected)));
    }

    #[tokio::test]
    async fn test_infinite_loop_with_break() {
        let validator = SafetyValidator::new();
        let genome = AgentGenome::new("while true { break; }".to_string(), vec![]);

        let result = validator.validate(&genome).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_custom_forbidden_keyword() {
        let validator = SafetyValidator::new().add_forbidden_keyword("dangerous".to_string());
        let genome = AgentGenome::new("dangerous operation".to_string(), vec![]);

        let result = validator.validate(&genome).await;
        assert!(matches!(result, Err(ValidationError::ForbiddenKeyword(_))));
    }

    #[test]
    fn test_validator_configuration() {
        let validator = SafetyValidator::new()
            .with_max_code_length(5000)
            .with_max_parameters(500)
            .with_parameter_bounds(-100.0, 100.0)
            .with_syntax_check(false);

        assert_eq!(validator.max_code_length, 5000);
        assert_eq!(validator.max_parameters, 500);
        assert_eq!(validator.parameter_bounds, (-100.0, 100.0));
        assert!(!validator.enable_syntax_check);
    }

    #[test]
    fn test_sync_validation() {
        let validator = SafetyValidator::new();
        let genome = AgentGenome::new("test".to_string(), vec![1.0]);

        let result = validator.validate_sync(&genome);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fast_validator() {
        let validator = FastValidator::new();
        let genome = AgentGenome::new("test".to_string(), vec![1.0]);

        let result = validator.validate_fast(&genome);
        assert!(result.is_ok());
    }

    #[test]
    fn test_syntax_validation_brackets() {
        let validator = SafetyValidator::new();

        // Valid syntax
        assert!(validator.validate_syntax("fn test() { [1, 2, 3] }").is_ok());

        // Invalid syntax
        assert!(validator.validate_syntax("fn test() { [1, 2, 3 }").is_err());
        assert!(validator.validate_syntax("fn test() ] { }").is_err());
        assert!(validator.validate_syntax("fn test( { }").is_err());
    }

    #[tokio::test]
    async fn test_case_insensitive_forbidden_keywords() {
        let validator = SafetyValidator::new();
        let genome = AgentGenome::new("EXEC('command')".to_string(), vec![]);

        let result = validator.validate(&genome).await;
        assert!(matches!(result, Err(ValidationError::ForbiddenKeyword(_))));
    }

    #[tokio::test]
    async fn test_all_forbidden_keywords() {
        let validator = SafetyValidator::new();
        let keywords = vec![
            "exec",
            "eval",
            "system",
            "unsafe",
            "transmute",
            "__import__",
            "import",
            "include",
            "require",
        ];

        for keyword in keywords {
            let code = format!("{}('test')", keyword);
            let genome = AgentGenome::new(code, vec![]);
            let result = validator.validate(&genome).await;
            assert!(
                matches!(result, Err(ValidationError::ForbiddenKeyword(_))),
                "Keyword '{}' should be forbidden",
                keyword
            );
        }
    }

    #[tokio::test]
    async fn test_parameter_bounds_edge_cases() {
        let validator = SafetyValidator::new().with_parameter_bounds(-5.0, 5.0);

        // Exactly at bounds should be OK
        let genome_min = AgentGenome::new("test".to_string(), vec![-5.0]);
        assert!(validator.validate(&genome_min).await.is_ok());

        let genome_max = AgentGenome::new("test".to_string(), vec![5.0]);
        assert!(validator.validate(&genome_max).await.is_ok());

        // Just outside bounds should fail
        let genome_under = AgentGenome::new("test".to_string(), vec![-5.1]);
        assert!(matches!(
            validator.validate(&genome_under).await,
            Err(ValidationError::ParameterOutOfBounds { .. })
        ));

        let genome_over = AgentGenome::new("test".to_string(), vec![5.1]);
        assert!(matches!(
            validator.validate(&genome_over).await,
            Err(ValidationError::ParameterOutOfBounds { .. })
        ));
    }

    #[tokio::test]
    async fn test_empty_code_validation() {
        let validator = SafetyValidator::new();
        let genome = AgentGenome::new("".to_string(), vec![]);

        let result = validator.validate(&genome).await;
        assert!(result.is_ok()); // Empty code should be valid
    }

    #[tokio::test]
    async fn test_empty_parameters_validation() {
        let validator = SafetyValidator::new();
        let genome = AgentGenome::new("fn test() {}".to_string(), vec![]);

        let result = validator.validate(&genome).await;
        assert!(result.is_ok()); // Empty parameters should be valid
    }

    #[tokio::test]
    async fn test_syntax_validation_disabled() {
        let validator = SafetyValidator::new().with_syntax_check(false);
        let genome = AgentGenome::new("fn test() { unclosed brace".to_string(), vec![]);

        let result = validator.validate(&genome).await;
        // Should not fail on syntax when disabled
        assert!(
            result.is_ok() || !matches!(result.unwrap_err(), ValidationError::InvalidSyntax(_))
        );
    }

    #[tokio::test]
    async fn test_complex_syntax_validation() {
        let validator = SafetyValidator::new();

        // Nested structures should work
        let complex_code = "fn test() { if (true) { let arr = [1, 2, {a: 3}]; } }";
        let genome = AgentGenome::new(complex_code.to_string(), vec![]);
        assert!(validator.validate(&genome).await.is_ok());

        // Mixed unmatched brackets
        let invalid_code = "fn test() { if (true { let arr = [1, 2, 3}]; } )";
        let genome = AgentGenome::new(invalid_code.to_string(), vec![]);
        assert!(matches!(
            validator.validate(&genome).await,
            Err(ValidationError::InvalidSyntax(_))
        ));
    }

    #[tokio::test]
    async fn test_infinite_loop_patterns() {
        let validator = SafetyValidator::new();

        let dangerous_patterns = vec![
            "while true { compute(); }",
            "while(true) { process(); }",
            "while 1 { work(); }",
            "while(1) { execute(); }",
            "for(;;) { run(); }",
            "loop { calculate(); }",
        ];

        for pattern in dangerous_patterns {
            let genome = AgentGenome::new(pattern.to_string(), vec![]);
            let result = validator.validate(&genome).await;
            assert!(
                matches!(result, Err(ValidationError::InfiniteLoopDetected)),
                "Pattern '{}' should be detected as infinite loop",
                pattern
            );
        }
    }

    #[tokio::test]
    async fn test_infinite_loop_with_return() {
        let validator = SafetyValidator::new();
        let genome = AgentGenome::new(
            "while true { if (condition) return 5; }".to_string(),
            vec![],
        );

        let result = validator.validate(&genome).await;
        assert!(result.is_ok()); // Should be OK with return statement
    }

    #[test]
    fn test_validation_error_display() {
        let errors = vec![
            ValidationError::ForbiddenKeyword("exec".to_string()),
            ValidationError::CodeTooLong {
                actual: 1000,
                max: 500,
            },
            ValidationError::TooManyParameters {
                actual: 100,
                max: 50,
            },
            ValidationError::ParameterOutOfBounds {
                value: 15.0,
                min: -10.0,
                max: 10.0,
            },
            ValidationError::InvalidSyntax("unmatched brace".to_string()),
            ValidationError::InfiniteLoopDetected,
        ];

        for error in errors {
            let message = error.to_string();
            assert!(!message.is_empty());
        }
    }

    #[test]
    fn test_default_validator() {
        let validator1 = SafetyValidator::new();
        let validator2 = SafetyValidator::default();

        assert_eq!(validator1.max_code_length, validator2.max_code_length);
        assert_eq!(validator1.max_parameters, validator2.max_parameters);
        assert_eq!(validator1.parameter_bounds, validator2.parameter_bounds);
    }

    #[tokio::test]
    async fn test_large_parameter_array() {
        let validator = SafetyValidator::new();
        let large_params: Vec<f32> = (0..500).map(|i| i as f32).collect();
        let genome = AgentGenome::new("test".to_string(), large_params);

        let result = validator.validate(&genome).await;
        assert!(result.is_ok()); // Should be within default limit of 1000
    }

    #[tokio::test]
    async fn test_unicode_in_code() {
        let validator = SafetyValidator::new();
        let genome = AgentGenome::new("fn test() { let 测试 = 42; }".to_string(), vec![]);

        let result = validator.validate(&genome).await;
        assert!(result.is_ok()); // Unicode should be allowed
    }

    #[tokio::test]
    async fn test_special_characters_in_code() {
        let validator = SafetyValidator::new();
        let genome = AgentGenome::new(
            "fn test() { let x = \"hello\nworld\"; }".to_string(),
            vec![],
        );

        let result = validator.validate(&genome).await;
        assert!(result.is_ok()); // Special chars should be OK
    }

    #[test]
    fn test_fast_validator_configuration() {
        let validator = FastValidator {
            max_code_length: 5000,
            max_parameters: 100,
            parameter_bounds: (-50.0, 50.0),
        };

        assert_eq!(validator.max_code_length, 5000);
        assert_eq!(validator.max_parameters, 100);
        assert_eq!(validator.parameter_bounds, (-50.0, 50.0));
    }

    #[test]
    fn test_fast_validator_code_length() {
        let validator = FastValidator::new();
        let long_code = "a".repeat(15000);
        let genome = AgentGenome::new(long_code, vec![]);

        let result = validator.validate_fast(&genome);
        assert!(matches!(result, Err(ValidationError::CodeTooLong { .. })));
    }

    #[test]
    fn test_fast_validator_parameter_count() {
        let validator = FastValidator::new();
        let many_params: Vec<f32> = (0..1500).map(|i| i as f32).collect();
        let genome = AgentGenome::new("test".to_string(), many_params);

        let result = validator.validate_fast(&genome);
        assert!(matches!(
            result,
            Err(ValidationError::TooManyParameters { .. })
        ));
    }

    #[tokio::test]
    async fn test_concurrent_validation() {
        use std::sync::Arc;
        use tokio::task;

        let validator = Arc::new(SafetyValidator::new());
        let mut handles = vec![];

        for i in 0..10 {
            let validator_clone = validator.clone();
            let handle = task::spawn(async move {
                let genome =
                    AgentGenome::new(format!("fn test_{} {{ return {}; }}", i, i), vec![i as f32]);
                validator_clone.validate(&genome).await
            });
            handles.push(handle);
        }

        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_sync_vs_async_validation() {
        let validator = SafetyValidator::new();
        let genome = AgentGenome::new("fn test() { return 42; }".to_string(), vec![1.0, 2.0]);

        let async_result = validator.validate(&genome).await;
        let sync_result = validator.validate_sync(&genome);

        assert!(async_result.is_ok());
        assert!(sync_result.is_ok());
    }

    #[tokio::test]
    async fn test_validator_with_extreme_bounds() {
        let validator =
            SafetyValidator::new().with_parameter_bounds(f32::NEG_INFINITY, f32::INFINITY);

        let genome = AgentGenome::new("test".to_string(), vec![f32::MAX, f32::MIN]);
        let result = validator.validate(&genome).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_syntax_validation_performance() {
        let validator = SafetyValidator::new();
        let large_code = "{".repeat(1000) + &"}".repeat(1000);

        let start = std::time::Instant::now();
        let result = validator.validate_syntax(&large_code);
        let duration = start.elapsed();

        assert!(result.is_ok());
        assert!(duration.as_millis() < 100); // Should be fast
    }

    #[tokio::test]
    async fn test_forbidden_keyword_substring_matching() {
        let validator = SafetyValidator::new();

        // Should match substrings
        let genome1 = AgentGenome::new("execute_command".to_string(), vec![]);
        let result1 = validator.validate(&genome1).await;
        assert!(matches!(result1, Err(ValidationError::ForbiddenKeyword(_))));

        // Should not match partial words that aren't dangerous
        let genome2 = AgentGenome::new("excellent_function".to_string(), vec![]);
        let result2 = validator.validate(&genome2).await;
        assert!(result2.is_ok()); // "exec" in "excellent" should be OK
    }

    #[tokio::test]
    async fn test_parameter_bounds_with_infinity() {
        let validator = SafetyValidator::new();
        let genome = AgentGenome::new("test".to_string(), vec![f32::INFINITY]);

        let result = validator.validate(&genome).await;
        assert!(matches!(
            result,
            Err(ValidationError::ParameterOutOfBounds { .. })
        ));
    }

    #[tokio::test]
    async fn test_parameter_bounds_with_nan() {
        let validator = SafetyValidator::new();
        let genome = AgentGenome::new("test".to_string(), vec![f32::NAN]);

        let result = validator.validate(&genome).await;
        // NaN comparisons always return false, so this should fail bounds check
        assert!(matches!(
            result,
            Err(ValidationError::ParameterOutOfBounds { .. })
        ));
    }

    #[test]
    fn test_validator_cloning() {
        let validator1 = SafetyValidator::new().with_max_code_length(5000);
        let validator2 = validator1.clone();

        assert_eq!(validator1.max_code_length, validator2.max_code_length);
    }

    #[tokio::test]
    async fn test_multiple_validation_errors() {
        let validator = SafetyValidator::new()
            .with_max_code_length(10)
            .add_forbidden_keyword("test".to_string());

        let genome = AgentGenome::new("this is a very long test code".to_string(), vec![]);

        // Should fail on first error encountered (code length)
        let result = validator.validate(&genome).await;
        assert!(matches!(result, Err(ValidationError::CodeTooLong { .. })));
    }

    #[test]
    fn test_fast_validator_default() {
        let validator = FastValidator::new();
        assert_eq!(validator.max_code_length, 10_000);
        assert_eq!(validator.max_parameters, 1000);
        assert_eq!(validator.parameter_bounds, (-1000.0, 1000.0));
    }

    #[tokio::test]
    async fn test_validation_order() {
        let validator = SafetyValidator::new()
            .with_max_code_length(5)
            .add_forbidden_keyword("bad".to_string());

        // Code length check should come first
        let genome = AgentGenome::new("bad code that is too long".to_string(), vec![]);
        let result = validator.validate(&genome).await;

        assert!(matches!(result, Err(ValidationError::CodeTooLong { .. })));
    }

    #[tokio::test]
    async fn test_loop_detection_case_sensitivity() {
        let validator = SafetyValidator::new();

        // Should detect uppercase patterns
        let genome = AgentGenome::new("WHILE TRUE { }".to_string(), vec![]);
        let result = validator.validate(&genome).await;
        assert!(matches!(result, Err(ValidationError::InfiniteLoopDetected)));
    }

    #[test]
    fn test_syntax_validation_empty_code() {
        let validator = SafetyValidator::new();
        let result = validator.validate_syntax("");
        assert!(result.is_ok());
    }

    #[test]
    fn test_syntax_validation_single_characters() {
        let validator = SafetyValidator::new();

        assert!(validator.validate_syntax("{").is_err());
        assert!(validator.validate_syntax("}").is_err());
        assert!(validator.validate_syntax("[").is_err());
        assert!(validator.validate_syntax("]").is_err());
        assert!(validator.validate_syntax("(").is_err());
        assert!(validator.validate_syntax(")").is_err());
    }

    #[tokio::test]
    async fn test_infinite_loop_check_empty_code() {
        let validator = SafetyValidator::new();
        let result = validator.check_infinite_loops("");
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_forbidden_keywords_modification() {
        let mut validator = SafetyValidator::new();

        // Add a new keyword
        validator = validator.add_forbidden_keyword("custom_danger".to_string());

        let genome = AgentGenome::new("custom_danger()".to_string(), vec![]);
        let result = validator.validate(&genome).await;
        assert!(matches!(result, Err(ValidationError::ForbiddenKeyword(_))));
    }

    #[tokio::test]
    async fn test_validation_with_all_options_disabled() {
        let validator = SafetyValidator::new()
            .with_max_code_length(usize::MAX)
            .with_max_parameters(usize::MAX)
            .with_parameter_bounds(f32::NEG_INFINITY, f32::INFINITY)
            .with_syntax_check(false);

        // Should only check forbidden keywords
        let genome = AgentGenome::new("exec('dangerous')".to_string(), vec![f32::MAX]);
        let result = validator.validate(&genome).await;
        assert!(matches!(result, Err(ValidationError::ForbiddenKeyword(_))));
    }
}
