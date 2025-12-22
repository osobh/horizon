//! Error types for the zero-config intelligence system

use thiserror::Error;

/// Result type for zero-config operations
pub type Result<T> = std::result::Result<T, ZeroConfigError>;

/// Errors that can occur in the zero-config system
#[derive(Debug, Error)]
pub enum ZeroConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("TOML parsing error: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("Language detection failed: {reason}")]
    LanguageDetection { reason: String },

    #[error("Dependency analysis failed: {reason}")]
    DependencyAnalysis { reason: String },

    #[error("Resource estimation failed: {reason}")]
    ResourceEstimation { reason: String },

    #[error("Pattern recognition failed: {reason}")]
    PatternRecognition { reason: String },

    #[error("Configuration generation failed: {reason}")]
    ConfigGeneration { reason: String },

    #[error("Learning system error: {reason}")]
    Learning { reason: String },

    #[error("Tree-sitter parsing error: {reason}")]
    Parsing { reason: String },

    #[error("Invalid file path: {path}")]
    InvalidPath { path: String },

    #[error("Unsupported language: {language}")]
    UnsupportedLanguage { language: String },

    #[error("Machine learning error: {reason}")]
    MachineLearning { reason: String },

    #[error("Pattern similarity calculation failed: {reason}")]
    SimilarityCalculation { reason: String },

    #[error("Configuration validation failed: {reason}")]
    ConfigValidation { reason: String },

    #[error("Knowledge base error: {reason}")]
    KnowledgeBase { reason: String },

    #[error("Memory allocation error: {reason}")]
    Memory { reason: String },

    #[error("Runtime error from exorust-runtime: {0}")]
    Runtime(#[from] exorust_runtime::RuntimeError),

    #[error("Memory error from exorust-memory: {0}")]
    MemoryError(#[from] exorust_memory::MemoryError),

    #[error("Task join error: {0}")]
    JoinError(#[from] tokio::task::JoinError),
}

impl ZeroConfigError {
    /// Create a language detection error
    pub fn language_detection<S: Into<String>>(reason: S) -> Self {
        Self::LanguageDetection {
            reason: reason.into(),
        }
    }

    /// Create a dependency analysis error
    pub fn dependency_analysis<S: Into<String>>(reason: S) -> Self {
        Self::DependencyAnalysis {
            reason: reason.into(),
        }
    }

    /// Create a resource estimation error
    pub fn resource_estimation<S: Into<String>>(reason: S) -> Self {
        Self::ResourceEstimation {
            reason: reason.into(),
        }
    }

    /// Create a pattern recognition error
    pub fn pattern_recognition<S: Into<String>>(reason: S) -> Self {
        Self::PatternRecognition {
            reason: reason.into(),
        }
    }

    /// Create a configuration generation error
    pub fn config_generation<S: Into<String>>(reason: S) -> Self {
        Self::ConfigGeneration {
            reason: reason.into(),
        }
    }

    /// Create a learning system error
    pub fn learning<S: Into<String>>(reason: S) -> Self {
        Self::Learning {
            reason: reason.into(),
        }
    }

    /// Create a parsing error
    pub fn parsing<S: Into<String>>(reason: S) -> Self {
        Self::Parsing {
            reason: reason.into(),
        }
    }

    /// Create an invalid path error
    pub fn invalid_path<S: Into<String>>(path: S) -> Self {
        Self::InvalidPath { path: path.into() }
    }

    /// Create an unsupported language error
    pub fn unsupported_language<S: Into<String>>(language: S) -> Self {
        Self::UnsupportedLanguage {
            language: language.into(),
        }
    }

    /// Create a machine learning error
    pub fn machine_learning<S: Into<String>>(reason: S) -> Self {
        Self::MachineLearning {
            reason: reason.into(),
        }
    }

    /// Create a similarity calculation error
    pub fn similarity_calculation<S: Into<String>>(reason: S) -> Self {
        Self::SimilarityCalculation {
            reason: reason.into(),
        }
    }

    /// Create a configuration validation error
    pub fn config_validation<S: Into<String>>(reason: S) -> Self {
        Self::ConfigValidation {
            reason: reason.into(),
        }
    }

    /// Create a knowledge base error
    pub fn knowledge_base<S: Into<String>>(reason: S) -> Self {
        Self::KnowledgeBase {
            reason: reason.into(),
        }
    }

    /// Create a memory error
    pub fn memory<S: Into<String>>(reason: S) -> Self {
        Self::Memory {
            reason: reason.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;
    use std::io::{Error as IoError, ErrorKind};

    #[test]
    fn test_io_error_conversion() {
        let io_error = IoError::new(ErrorKind::NotFound, "File not found");
        let zero_config_error: ZeroConfigError = io_error.into();

        match zero_config_error {
            ZeroConfigError::Io(_) => (),
            _ => panic!("Expected IO error"),
        }
    }

    #[test]
    fn test_json_error_conversion() {
        let json_str = r#"{"invalid": json"#;
        let json_error = serde_json::from_str::<serde_json::Value>(json_str).unwrap_err();
        let zero_config_error: ZeroConfigError = json_error.into();

        match zero_config_error {
            ZeroConfigError::Json(_) => (),
            _ => panic!("Expected JSON error"),
        }
    }

    #[test]
    fn test_toml_error_conversion() {
        let toml_str = r#"[invalid toml"#;
        let toml_error = toml::from_str::<toml::Value>(toml_str).unwrap_err();
        let zero_config_error: ZeroConfigError = toml_error.into();

        match zero_config_error {
            ZeroConfigError::Toml(_) => (),
            _ => panic!("Expected TOML error"),
        }
    }

    #[test]
    fn test_custom_error_creation() {
        let error = ZeroConfigError::language_detection("Unable to detect language");
        match error {
            ZeroConfigError::LanguageDetection { reason } => {
                assert_eq!(reason, "Unable to detect language");
            }
            _ => panic!("Expected LanguageDetection error"),
        }
    }

    #[test]
    fn test_dependency_analysis_error() {
        let error = ZeroConfigError::dependency_analysis("Failed to parse package.json");
        match error {
            ZeroConfigError::DependencyAnalysis { reason } => {
                assert_eq!(reason, "Failed to parse package.json");
            }
            _ => panic!("Expected DependencyAnalysis error"),
        }
    }

    #[test]
    fn test_resource_estimation_error() {
        let error = ZeroConfigError::resource_estimation("Cannot estimate GPU requirements");
        match error {
            ZeroConfigError::ResourceEstimation { reason } => {
                assert_eq!(reason, "Cannot estimate GPU requirements");
            }
            _ => panic!("Expected ResourceEstimation error"),
        }
    }

    #[test]
    fn test_pattern_recognition_error() {
        let error = ZeroConfigError::pattern_recognition("No similar patterns found");
        match error {
            ZeroConfigError::PatternRecognition { reason } => {
                assert_eq!(reason, "No similar patterns found");
            }
            _ => panic!("Expected PatternRecognition error"),
        }
    }

    #[test]
    fn test_config_generation_error() {
        let error = ZeroConfigError::config_generation("Invalid resource constraints");
        match error {
            ZeroConfigError::ConfigGeneration { reason } => {
                assert_eq!(reason, "Invalid resource constraints");
            }
            _ => panic!("Expected ConfigGeneration error"),
        }
    }

    #[test]
    fn test_learning_error() {
        let error = ZeroConfigError::learning("Pattern storage failed");
        match error {
            ZeroConfigError::Learning { reason } => {
                assert_eq!(reason, "Pattern storage failed");
            }
            _ => panic!("Expected Learning error"),
        }
    }

    #[test]
    fn test_parsing_error() {
        let error = ZeroConfigError::parsing("Tree-sitter parse failed");
        match error {
            ZeroConfigError::Parsing { reason } => {
                assert_eq!(reason, "Tree-sitter parse failed");
            }
            _ => panic!("Expected Parsing error"),
        }
    }

    #[test]
    fn test_invalid_path_error() {
        let error = ZeroConfigError::invalid_path("/nonexistent/path");
        match error {
            ZeroConfigError::InvalidPath { path } => {
                assert_eq!(path, "/nonexistent/path");
            }
            _ => panic!("Expected InvalidPath error"),
        }
    }

    #[test]
    fn test_unsupported_language_error() {
        let error = ZeroConfigError::unsupported_language("cobol");
        match error {
            ZeroConfigError::UnsupportedLanguage { language } => {
                assert_eq!(language, "cobol");
            }
            _ => panic!("Expected UnsupportedLanguage error"),
        }
    }

    #[test]
    fn test_machine_learning_error() {
        let error = ZeroConfigError::machine_learning("Model training failed");
        match error {
            ZeroConfigError::MachineLearning { reason } => {
                assert_eq!(reason, "Model training failed");
            }
            _ => panic!("Expected MachineLearning error"),
        }
    }

    #[test]
    fn test_similarity_calculation_error() {
        let error = ZeroConfigError::similarity_calculation("Embedding dimension mismatch");
        match error {
            ZeroConfigError::SimilarityCalculation { reason } => {
                assert_eq!(reason, "Embedding dimension mismatch");
            }
            _ => panic!("Expected SimilarityCalculation error"),
        }
    }

    #[test]
    fn test_config_validation_error() {
        let error = ZeroConfigError::config_validation("CPU cores must be positive");
        match error {
            ZeroConfigError::ConfigValidation { reason } => {
                assert_eq!(reason, "CPU cores must be positive");
            }
            _ => panic!("Expected ConfigValidation error"),
        }
    }

    #[test]
    fn test_knowledge_base_error() {
        let error = ZeroConfigError::knowledge_base("Database connection failed");
        match error {
            ZeroConfigError::KnowledgeBase { reason } => {
                assert_eq!(reason, "Database connection failed");
            }
            _ => panic!("Expected KnowledgeBase error"),
        }
    }

    #[test]
    fn test_memory_error() {
        let error = ZeroConfigError::memory("Failed to allocate embedding vectors");
        match error {
            ZeroConfigError::Memory { reason } => {
                assert_eq!(reason, "Failed to allocate embedding vectors");
            }
            _ => panic!("Expected Memory error"),
        }
    }

    #[test]
    fn test_error_display() {
        let error = ZeroConfigError::language_detection("Test error");
        let display_str = format!("{}", error);
        assert!(display_str.contains("Language detection failed"));
        assert!(display_str.contains("Test error"));
    }

    #[test]
    fn test_error_debug() {
        let error = ZeroConfigError::config_generation("Debug test");
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("ConfigGeneration"));
        assert!(debug_str.contains("Debug test"));
    }

    #[test]
    fn test_error_source_chain() {
        let io_error = IoError::new(ErrorKind::PermissionDenied, "Access denied");
        let zero_config_error: ZeroConfigError = io_error.into();

        assert!(zero_config_error.source().is_some());
    }

    #[test]
    fn test_result_type_usage() {
        fn test_function() -> Result<String> {
            Ok("success".to_string())
        }

        let result = test_function();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
    }

    #[test]
    fn test_result_type_error() {
        fn test_function() -> Result<String> {
            Err(ZeroConfigError::language_detection("Test failure"))
        }

        let result = test_function();
        assert!(result.is_err());
    }

    #[test]
    fn test_chained_errors() {
        let io_error = IoError::new(ErrorKind::UnexpectedEof, "Unexpected end of file");
        let zero_config_error: ZeroConfigError = io_error.into();

        let chained_error = match zero_config_error {
            ZeroConfigError::Io(e) => {
                ZeroConfigError::parsing(format!("Parse failed due to IO: {}", e))
            }
            _ => panic!("Expected IO error"),
        };

        match chained_error {
            ZeroConfigError::Parsing { reason } => {
                assert!(reason.contains("Parse failed"));
                assert!(reason.contains("Unexpected end of file"));
            }
            _ => panic!("Expected Parsing error"),
        }
    }
}
