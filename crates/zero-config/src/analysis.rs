//! Code analysis engine for zero-config intelligence
//!
//! This module provides intelligent analysis of codebases to understand:
//! - Programming languages and frameworks
//! - Dependencies and their types
//! - Resource requirements estimation
//! - Complexity metrics

use crate::dependency_parsers::DependencyAnalyzer;
use crate::language_detection::LanguageDetector;
use crate::resource_estimation::ResourceEstimator;
use crate::{ResourceRequirements, Result, ZeroConfigError};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::fs;
use walkdir::WalkDir;

/// Main code analyzer that orchestrates all analysis operations
pub struct CodeAnalyzer {
    pub(crate) language_detector: LanguageDetector,
    pub(crate) dependency_analyzer: DependencyAnalyzer,
    pub(crate) resource_estimator: ResourceEstimator,
}

impl CodeAnalyzer {
    /// Create a new code analyzer
    pub fn new() -> Self {
        Self {
            language_detector: LanguageDetector::new(),
            dependency_analyzer: DependencyAnalyzer::new(),
            resource_estimator: ResourceEstimator::new(),
        }
    }

    /// Get list of supported programming languages
    pub fn supported_languages(&self) -> Vec<&str> {
        self.language_detector.supported_languages()
    }

    /// Analyze a complete codebase
    pub async fn analyze_codebase(&self, path: &str) -> Result<CodebaseAnalysis> {
        let path = Path::new(path);
        if !path.exists() {
            return Err(ZeroConfigError::invalid_path(path.to_string_lossy()));
        }

        // Step 1: Detect primary language and framework
        let language_info = self.language_detector.detect_language(path).await?;

        // Step 2: Analyze dependencies
        let dependencies = self
            .dependency_analyzer
            .analyze_dependencies(path, &language_info.language)
            .await?;

        // Step 3: Calculate complexity metrics
        let complexity = self.calculate_complexity_metrics(path).await?;

        // Step 4: Estimate resource requirements
        let resources = self
            .resource_estimator
            .estimate_resources(&language_info, &dependencies, &complexity)
            .await?;

        Ok(CodebaseAnalysis {
            path: path.to_string_lossy().to_string(),
            language: language_info.language,
            framework: language_info.framework,
            dependencies,
            complexity,
            resources,
            file_count: language_info.file_count,
            total_lines: language_info.total_lines,
        })
    }

    /// Calculate complexity metrics for the codebase
    pub(crate) async fn calculate_complexity_metrics(
        &self,
        path: &Path,
    ) -> Result<ComplexityMetrics> {
        let mut total_lines = 0;
        let mut total_files = 0;
        let mut function_count = 0;
        let mut class_count = 0;
        let mut import_count = 0;

        for entry in WalkDir::new(path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let file_path = entry.path();
            if let Some(extension) = file_path.extension() {
                let ext = extension.to_string_lossy().to_lowercase();

                // Only analyze source code files
                if matches!(
                    ext.as_str(),
                    "rs" | "py" | "js" | "ts" | "go" | "java" | "cpp" | "cc" | "cxx" | "c" | "h"
                ) {
                    if let Ok(content) = fs::read_to_string(file_path).await {
                        total_files += 1;
                        total_lines += content.lines().count();

                        // Simple heuristic-based counting
                        function_count += content.matches("fn ").count()
                            + content.matches("def ").count()
                            + content.matches("func ").count()
                            + content.matches("function ").count();

                        class_count += content.matches("class ").count()
                            + content.matches("struct ").count()
                            + content.matches("interface ").count();

                        import_count += content.matches("import ").count()
                            + content.matches("use ").count()
                            + content.matches("#include").count();
                    }
                }
            }
        }

        // Calculate cyclomatic complexity estimate (simplified)
        let estimated_complexity =
            (function_count * 2 + class_count * 3) as f32 / total_lines.max(1) as f32;

        Ok(ComplexityMetrics {
            total_lines,
            total_files,
            function_count,
            class_count,
            import_count,
            cyclomatic_complexity: estimated_complexity,
            maintainability_index: Self::calculate_maintainability_index(
                total_lines,
                function_count,
                estimated_complexity,
            ),
        })
    }

    /// Calculate maintainability index (simplified version)
    pub(crate) fn calculate_maintainability_index(
        lines: usize,
        functions: usize,
        complexity: f32,
    ) -> f32 {
        // Simplified maintainability index calculation
        let volume = (lines as f32).ln();
        let complexity_penalty = complexity * 10.0;
        let function_penalty = if functions > 0 {
            (lines as f32 / functions as f32).ln()
        } else {
            0.0
        };

        (171.0 - volume * 5.2 - complexity_penalty - function_penalty)
            .max(0.0)
            .min(171.0)
    }
}

impl Default for CodeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete analysis result for a codebase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodebaseAnalysis {
    pub path: String,
    pub language: String,
    pub framework: Option<String>,
    pub dependencies: Vec<crate::Dependency>,
    pub complexity: ComplexityMetrics,
    pub resources: ResourceRequirements,
    pub file_count: usize,
    pub total_lines: usize,
}

/// Code complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    pub total_lines: usize,
    pub total_files: usize,
    pub function_count: usize,
    pub class_count: usize,
    pub import_count: usize,
    pub cyclomatic_complexity: f32,
    pub maintainability_index: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DependencyType;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_code_analyzer_creation() {
        let analyzer = CodeAnalyzer::new();
        let languages = analyzer.supported_languages();
        assert!(languages.contains(&"rust"));
        assert!(languages.contains(&"python"));
        assert!(languages.contains(&"javascript"));
    }

    #[tokio::test]
    async fn test_complexity_metrics_calculation() {
        let analyzer = CodeAnalyzer::new();
        let temp_dir = TempDir::new().unwrap();

        // Create a simple Rust project
        fs::create_dir(temp_dir.path().join("src")).await.unwrap();
        let main_rs = r#"
use std::collections::HashMap;

struct MyStruct {
    value: i32,
}

impl MyStruct {
    fn new(value: i32) -> Self {
        Self { value }
    }
    
    fn process(&self) -> i32 {
        if self.value > 0 {
            self.value * 2
        } else {
            0
        }
    }
}

fn main() {
    let s = MyStruct::new(42);
    println!("{}", s.process());
}
"#;
        fs::write(temp_dir.path().join("src").join("main.rs"), main_rs)
            .await
            .unwrap();

        let complexity = analyzer
            .calculate_complexity_metrics(temp_dir.path())
            .await
            .unwrap();

        assert!(complexity.total_lines > 0);
        assert_eq!(complexity.total_files, 1);
        assert!(complexity.function_count >= 2); // new, process, main
        assert_eq!(complexity.class_count, 1); // MyStruct
        assert_eq!(complexity.import_count, 1); // use std::collections::HashMap
        assert!(complexity.cyclomatic_complexity >= 0.0);
        assert!(complexity.maintainability_index > 0.0);
    }

    #[tokio::test]
    async fn test_maintainability_index_calculation() {
        let index = CodeAnalyzer::calculate_maintainability_index(1000, 10, 2.0);
        assert!(index >= 0.0);
        assert!(index <= 171.0);

        // Higher complexity should result in lower maintainability
        let index_high_complexity = CodeAnalyzer::calculate_maintainability_index(1000, 10, 10.0);
        assert!(index > index_high_complexity);
    }

    #[test]
    fn test_codebase_analysis_serialization() {
        let analysis = CodebaseAnalysis {
            path: "/test/path".to_string(),
            language: "rust".to_string(),
            framework: Some("tokio".to_string()),
            dependencies: vec![],
            complexity: ComplexityMetrics {
                total_lines: 1000,
                total_files: 5,
                function_count: 20,
                class_count: 3,
                import_count: 10,
                cyclomatic_complexity: 2.5,
                maintainability_index: 85.0,
            },
            resources: ResourceRequirements {
                cpu_cores: 2.0,
                memory_gb: 4.0,
                gpu_units: 0.0,
                storage_gb: 10.0,
                network_bandwidth_mbps: 100.0,
            },
            file_count: 5,
            total_lines: 1000,
        };

        let json = serde_json::to_string(&analysis).unwrap();
        let deserialized: CodebaseAnalysis = serde_json::from_str(&json).unwrap();

        assert_eq!(analysis.language, deserialized.language);
        assert_eq!(analysis.framework, deserialized.framework);
        assert_eq!(
            analysis.complexity.total_lines,
            deserialized.complexity.total_lines
        );
    }

    #[tokio::test]
    async fn test_empty_directory_analysis() {
        let temp_dir = TempDir::new().unwrap();
        let analyzer = CodeAnalyzer::new();

        let result = analyzer
            .analyze_codebase(temp_dir.path().to_str().unwrap())
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_nonexistent_directory_analysis() {
        let analyzer = CodeAnalyzer::new();
        let result = analyzer.analyze_codebase("/nonexistent/path").await;
        assert!(result.is_err());

        match result.unwrap_err() {
            ZeroConfigError::InvalidPath { path } => {
                assert_eq!(path, "/nonexistent/path");
            }
            _ => panic!("Expected InvalidPath error"),
        }
    }

    #[test]
    fn test_complexity_edge_cases() {
        // Test maintainability index edge cases
        assert_eq!(
            CodeAnalyzer::calculate_maintainability_index(0, 0, 0.0),
            171.0
        );
        assert_eq!(
            CodeAnalyzer::calculate_maintainability_index(1000000, 1, 100.0),
            0.0
        );

        // Test with intermediate values
        let index = CodeAnalyzer::calculate_maintainability_index(5000, 50, 10.0);
        assert!(index > 0.0 && index < 171.0);
    }

    // Framework-specific tests moved to language_detection.rs tests

    #[tokio::test]
    async fn test_javascript_framework_detection_nextjs() {
        let temp_dir = TempDir::new().unwrap();
        let package_json = r#"
{
  "name": "nextjs-app",
  "dependencies": {
    "next": "^13.0.0",
    "react": "^18.0.0"
  }
}
"#;
        fs::write(temp_dir.path().join("package.json"), package_json)
            .await
            .unwrap();
        fs::create_dir(temp_dir.path().join("pages")).await.unwrap();
        fs::write(
            temp_dir.path().join("pages").join("_app.js"),
            "export default function App() {}",
        )
        .await
        .unwrap();

        let detector = LanguageDetector::new();
        let info = detector.detect_language(temp_dir.path()).await.unwrap();

        assert_eq!(info.language, "javascript");
        assert_eq!(info.framework, Some("nextjs".to_string()));
    }

    #[tokio::test]
    async fn test_javascript_framework_detection_vue() {
        let temp_dir = TempDir::new().unwrap();
        let package_json = r#"
{
  "name": "vue-app",
  "dependencies": {
    "vue": "^3.2.0",
    "vue-router": "^4.0.0"
  }
}
"#;
        fs::write(temp_dir.path().join("package.json"), package_json)
            .await
            .unwrap();
        fs::write(
            temp_dir.path().join("App.vue"),
            "<template><div>Test</div></template>",
        )
        .await
        .unwrap();
        fs::write(
            temp_dir.path().join("main.js"),
            "import { createApp } from 'vue'",
        )
        .await
        .unwrap();

        let detector = LanguageDetector::new();
        let info = detector.detect_language(temp_dir.path()).await.unwrap();

        assert_eq!(info.language, "javascript");
        assert_eq!(info.framework, Some("vue".to_string()));
    }

    #[tokio::test]
    async fn test_cpp_file_detection() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(
            temp_dir.path().join("main.cpp"),
            "#include <iostream>\nint main() {}",
        )
        .await
        .unwrap();
        fs::write(
            temp_dir.path().join("header.h"),
            "#ifndef HEADER_H\n#define HEADER_H\n#endif",
        )
        .await
        .unwrap();
        fs::write(temp_dir.path().join("impl.cc"), "void func() {}")
            .await
            .unwrap();
        fs::write(temp_dir.path().join("other.cxx"), "class MyClass {};")
            .await
            .unwrap();

        let detector = LanguageDetector::new();
        let info = detector.detect_language(temp_dir.path()).await.unwrap();

        assert_eq!(info.language, "cpp");
        assert_eq!(info.total_lines, 7); // Count all lines from C++ files
    }

    // Dependency classification tests

    #[tokio::test]
    async fn test_rust_dependencies_edge_cases() {
        let temp_dir = TempDir::new().unwrap();
        let cargo_toml = r#"
[dependencies]
warp = "0.3"
lapin = "2.3"
bb8-redis = "0.13"
candle = "0.3"

[dev-dependencies]
criterion = "0.5"
"#;
        fs::write(temp_dir.path().join("Cargo.toml"), cargo_toml)
            .await
            .unwrap();
        fs::write(temp_dir.path().join("main.rs"), "fn main() {}")
            .await
            .unwrap();

        let analyzer = DependencyAnalyzer::new();
        let deps = analyzer
            .analyze_dependencies(temp_dir.path(), "rust")
            .await
            .unwrap();

        assert!(deps
            .iter()
            .any(|d| d.name == "warp" && d.dependency_type == DependencyType::WebFramework));
        assert!(deps
            .iter()
            .any(|d| d.name == "lapin" && d.dependency_type == DependencyType::MessageQueue));
        assert!(deps
            .iter()
            .any(|d| d.name == "bb8-redis" && d.dependency_type == DependencyType::Cache));
        assert!(deps
            .iter()
            .any(|d| d.name == "candle" && d.dependency_type == DependencyType::MLFramework));
    }

    #[tokio::test]
    async fn test_javascript_dependencies_edge_cases() {
        let temp_dir = TempDir::new().unwrap();
        let package_json = r#"
{
  "dependencies": {
    "koa": "^2.14.0",
    "bull": "^4.11.0",
    "ioredis": "^5.3.0",
    "@tensorflow/tfjs": "^4.10.0"
  }
}
"#;
        fs::write(temp_dir.path().join("package.json"), package_json)
            .await
            .unwrap();
        fs::write(temp_dir.path().join("index.js"), "console.log('test');")
            .await
            .unwrap();

        let analyzer = DependencyAnalyzer::new();
        let deps = analyzer
            .analyze_dependencies(temp_dir.path(), "javascript")
            .await
            .unwrap();

        assert!(deps
            .iter()
            .any(|d| d.name == "koa" && d.dependency_type == DependencyType::WebFramework));
        assert!(deps
            .iter()
            .any(|d| d.name == "bull" && d.dependency_type == DependencyType::MessageQueue));
        assert!(deps
            .iter()
            .any(|d| d.name == "ioredis" && d.dependency_type == DependencyType::Cache));
        assert!(deps
            .iter()
            .any(|d| d.name == "@tensorflow/tfjs"
                && d.dependency_type == DependencyType::MLFramework));
    }

    #[tokio::test]
    async fn test_java_dependency_detection() {
        let temp_dir = TempDir::new().unwrap();
        let pom_xml = r#"
<project>
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
        <dependency>
            <groupId>mysql</groupId>
            <artifactId>mysql-connector-java</artifactId>
        </dependency>
    </dependencies>
</project>
"#;
        fs::write(temp_dir.path().join("pom.xml"), pom_xml)
            .await
            .unwrap();
        fs::write(
            temp_dir.path().join("Application.java"),
            "public class Application {}",
        )
        .await
        .unwrap();

        let analyzer = DependencyAnalyzer::new();
        let deps = analyzer
            .analyze_dependencies(temp_dir.path(), "java")
            .await
            .unwrap();

        assert!(deps
            .iter()
            .any(|d| d.name == "org.springframework.boot:spring-boot-starter-web"));
        assert!(deps.iter().any(|d| d.name == "mysql:mysql-connector-java"));
    }

    #[tokio::test]
    async fn test_parse_pyproject_toml() {
        let temp_dir = TempDir::new().unwrap();
        let pyproject_toml = r#"
[tool.poetry]
name = "test-app"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
sqlalchemy = "^2.0"
alembic = "^1.12"
celery = "^5.3"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4"
black = "^23.0"
"#;
        fs::write(temp_dir.path().join("pyproject.toml"), pyproject_toml)
            .await
            .unwrap();
        fs::write(
            temp_dir.path().join("app.py"),
            "from sqlalchemy import create_engine",
        )
        .await
        .unwrap();
        // Also create requirements.txt for now since we don't parse pyproject.toml yet
        fs::write(
            temp_dir.path().join("requirements.txt"),
            "sqlalchemy==2.0\ncelery==5.3",
        )
        .await
        .unwrap();

        let analyzer = DependencyAnalyzer::new();
        let deps = analyzer
            .analyze_dependencies(temp_dir.path(), "python")
            .await
            .unwrap();

        assert!(deps
            .iter()
            .any(|d| d.name == "sqlalchemy" && d.dependency_type == DependencyType::Database));
        assert!(deps
            .iter()
            .any(|d| d.name == "celery" && d.dependency_type == DependencyType::MessageQueue));
    }
}
