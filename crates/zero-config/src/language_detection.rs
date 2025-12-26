//! Language detection module for zero-config intelligence

use crate::{Result, ZeroConfigError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokio::fs;
use walkdir::WalkDir;

/// Language detection system
pub struct LanguageDetector {
    supported_languages: Vec<String>,
}

impl LanguageDetector {
    pub fn new() -> Self {
        Self {
            supported_languages: vec![
                "rust".to_string(),
                "python".to_string(),
                "javascript".to_string(),
                "typescript".to_string(),
                "go".to_string(),
                "java".to_string(),
                "cpp".to_string(),
                "c".to_string(),
            ],
        }
    }

    pub fn supported_languages(&self) -> Vec<&str> {
        self.supported_languages
            .iter()
            .map(|s| s.as_str())
            .collect()
    }

    /// Detect the primary language and framework of a codebase
    pub async fn detect_language(&self, path: &Path) -> Result<LanguageInfo> {
        let mut file_counts: HashMap<String, usize> = HashMap::new();
        let mut total_lines = 0;
        let mut total_files = 0;
        let mut framework = None;

        for entry in WalkDir::new(path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let file_path = entry.path();
            total_files += 1;

            if let Some(extension) = file_path.extension() {
                let ext = extension.to_string_lossy().to_lowercase();
                *file_counts.entry(ext.clone()).or_insert(0) += 1;

                // Count lines for source files
                if matches!(
                    ext.as_str(),
                    "rs" | "py" | "js" | "ts" | "go" | "java" | "cpp" | "cc" | "cxx" | "c" | "h"
                ) {
                    if let Ok(content) = fs::read_to_string(file_path).await {
                        total_lines += content.lines().count();
                    }
                }
            }

            // Check for specific framework files
            if let Some(filename) = file_path.file_name() {
                let filename = filename.to_string_lossy();
                match filename.as_ref() {
                    "Cargo.toml" => framework = Some("cargo".to_string()),
                    "package.json" => {
                        framework = Some(
                            self.detect_js_framework(file_path)
                                .await
                                .unwrap_or("node".to_string()),
                        )
                    }
                    "requirements.txt" | "pyproject.toml" => {
                        framework = Some(
                            self.detect_python_framework(file_path)
                                .await
                                .unwrap_or("python".to_string()),
                        )
                    }
                    "go.mod" => framework = Some("go-modules".to_string()),
                    "pom.xml" => framework = Some("maven".to_string()),
                    "build.gradle" => framework = Some("gradle".to_string()),
                    _ => {}
                }
            }
        }

        // Determine primary language based on file counts
        let primary_language = self.determine_primary_language(&file_counts)?;

        Ok(LanguageInfo {
            language: primary_language,
            framework,
            file_count: total_files,
            total_lines,
            language_distribution: file_counts,
        })
    }

    /// Determine the primary language from file extension counts
    fn determine_primary_language(&self, file_counts: &HashMap<String, usize>) -> Result<String> {
        let language_mappings = [
            ("rs", "rust"),
            ("py", "python"),
            ("js", "javascript"),
            ("ts", "typescript"),
            ("go", "go"),
            ("java", "java"),
            ("cpp", "cpp"),
            ("cc", "cpp"),
            ("cxx", "cpp"),
            ("c", "c"),
        ];

        let mut language_scores: HashMap<String, usize> = HashMap::new();

        for (ext, lang) in &language_mappings {
            if let Some(&count) = file_counts.get(*ext) {
                *language_scores.entry(lang.to_string()).or_insert(0) += count;
            }
        }

        language_scores
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(lang, _)| lang)
            .ok_or_else(|| {
                ZeroConfigError::language_detection("No recognizable language files found")
            })
    }

    /// Detect JavaScript/TypeScript framework from package.json
    async fn detect_js_framework(&self, package_json_path: &Path) -> Result<String> {
        let content = fs::read_to_string(package_json_path).await?;
        let package: serde_json::Value = serde_json::from_str(&content)?;

        if let Some(deps) = package.get("dependencies").and_then(|d| d.as_object()) {
            // Check Next.js before React since Next.js uses React
            if deps.contains_key("next") {
                return Ok("nextjs".to_string());
            }
            if deps.contains_key("react") {
                return Ok("react".to_string());
            }
            if deps.contains_key("vue") {
                return Ok("vue".to_string());
            }
            if deps.contains_key("angular") || deps.contains_key("@angular/core") {
                return Ok("angular".to_string());
            }
            if deps.contains_key("express") {
                return Ok("express".to_string());
            }
            if deps.contains_key("fastify") {
                return Ok("fastify".to_string());
            }
        }

        Ok("node".to_string())
    }

    /// Detect Python framework from requirements.txt or pyproject.toml
    async fn detect_python_framework(&self, file_path: &Path) -> Result<String> {
        let content = fs::read_to_string(file_path).await?;

        if content.contains("django") {
            Ok("django".to_string())
        } else if content.contains("flask") {
            Ok("flask".to_string())
        } else if content.contains("fastapi") {
            Ok("fastapi".to_string())
        } else if content.contains("tornado") {
            Ok("tornado".to_string())
        } else if content.contains("starlette") {
            Ok("starlette".to_string())
        } else {
            Ok("python".to_string())
        }
    }
}

impl Default for LanguageDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about detected language and framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageInfo {
    pub language: String,
    pub framework: Option<String>,
    pub file_count: usize,
    pub total_lines: usize,
    pub language_distribution: HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_language_detector_rust() {
        let temp_dir = TempDir::new().unwrap();
        let cargo_toml = r#"
[package]
name = "test"
version = "0.1.0"

[dependencies]
tokio = "1.0"
"#;
        fs::write(temp_dir.path().join("Cargo.toml"), cargo_toml)
            .await
            .unwrap();
        fs::create_dir(temp_dir.path().join("src")).await.unwrap();
        fs::write(temp_dir.path().join("src").join("main.rs"), "fn main() {}")
            .await
            .unwrap();

        let detector = LanguageDetector::new();
        let info = detector.detect_language(temp_dir.path()).await.unwrap();

        assert_eq!(info.language, "rust");
        assert_eq!(info.framework, Some("cargo".to_string()));
        assert!(info.total_lines > 0);
    }

    #[tokio::test]
    async fn test_language_detector_python() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(
            temp_dir.path().join("requirements.txt"),
            "fastapi==0.104.1\nuvicorn==0.24.0",
        )
        .await
        .unwrap();
        fs::write(
            temp_dir.path().join("main.py"),
            "from fastapi import FastAPI\napp = FastAPI()",
        )
        .await
        .unwrap();

        let detector = LanguageDetector::new();
        let info = detector.detect_language(temp_dir.path()).await.unwrap();

        assert_eq!(info.language, "python");
        assert_eq!(info.framework, Some("fastapi".to_string()));
    }

    #[tokio::test]
    async fn test_language_detector_javascript() {
        let temp_dir = TempDir::new().unwrap();
        let package_json = r#"
{
  "name": "test-app",
  "dependencies": {
    "express": "^4.18.0",
    "react": "^18.0.0"
  }
}
"#;
        fs::write(temp_dir.path().join("package.json"), package_json)
            .await
            .unwrap();
        fs::write(
            temp_dir.path().join("index.js"),
            "const express = require('express');",
        )
        .await
        .unwrap();

        let detector = LanguageDetector::new();
        let info = detector.detect_language(temp_dir.path()).await.unwrap();

        assert_eq!(info.language, "javascript");
        assert_eq!(info.framework, Some("react".to_string()));
    }

    #[tokio::test]
    async fn test_unsupported_language_handling() {
        let temp_dir = TempDir::new().unwrap();

        // Create files with unsupported extensions
        fs::write(temp_dir.path().join("file.xyz"), "unsupported content")
            .await
            .unwrap();
        fs::write(temp_dir.path().join("data.dat"), "binary data")
            .await
            .unwrap();

        let detector = LanguageDetector::new();
        let result = detector.detect_language(temp_dir.path()).await;

        // Should error when no supported language files found
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_mixed_language_project() {
        let temp_dir = TempDir::new().unwrap();

        // Create a mixed language project with more Python files
        for i in 0..5 {
            fs::write(
                temp_dir.path().join(format!("module{}.py", i)),
                "import os\nprint('test')",
            )
            .await
            .unwrap();
        }

        fs::write(temp_dir.path().join("helper.js"), "console.log('helper');")
            .await
            .unwrap();
        fs::write(temp_dir.path().join("util.rs"), "fn util() {}")
            .await
            .unwrap();

        let detector = LanguageDetector::new();
        let info = detector.detect_language(temp_dir.path()).await.unwrap();

        // Should detect Python as primary language
        assert_eq!(info.language, "python");
        assert_eq!(info.total_lines, 12); // 10 lines from Python + 1 JS + 1 Rust
    }

    #[test]
    fn test_default_implementation() {
        let detector = LanguageDetector::default();
        assert!(!detector.supported_languages().is_empty());
    }
}
