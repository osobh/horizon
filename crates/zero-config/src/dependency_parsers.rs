//! Language-specific dependency parsers

use crate::dependency_classification::*;
use crate::{Dependency, Result};
use std::collections::HashMap;
use std::path::Path;
use tokio::fs;

/// Trait for language-specific dependency parsers
#[async_trait::async_trait]
pub trait DependencyParser: Send + Sync {
    async fn parse_dependencies(&self, path: &Path) -> Result<Vec<Dependency>>;
}

/// Dependency analysis system
pub struct DependencyAnalyzer {
    parsers: HashMap<String, Box<dyn DependencyParser + Send + Sync>>,
}

impl DependencyAnalyzer {
    pub fn new() -> Self {
        let mut parsers: HashMap<String, Box<dyn DependencyParser + Send + Sync>> = HashMap::new();
        parsers.insert("rust".to_string(), Box::new(CargoParser));
        parsers.insert("python".to_string(), Box::new(PythonParser));
        parsers.insert("javascript".to_string(), Box::new(NodeParser));
        parsers.insert("typescript".to_string(), Box::new(NodeParser));
        parsers.insert("go".to_string(), Box::new(GoParser));
        parsers.insert("java".to_string(), Box::new(JavaParser));

        Self { parsers }
    }

    /// Analyze dependencies for a given language
    pub async fn analyze_dependencies(
        &self,
        path: &Path,
        language: &str,
    ) -> Result<Vec<Dependency>> {
        if let Some(parser) = self.parsers.get(language) {
            parser.parse_dependencies(path).await
        } else {
            // Return empty dependencies for unsupported languages
            Ok(vec![])
        }
    }
}

impl Default for DependencyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Cargo.toml parser for Rust projects
pub struct CargoParser;

#[async_trait::async_trait]
impl DependencyParser for CargoParser {
    async fn parse_dependencies(&self, path: &Path) -> Result<Vec<Dependency>> {
        let cargo_toml_path = path.join("Cargo.toml");
        if !cargo_toml_path.exists() {
            return Ok(vec![]);
        }

        let content = match fs::read_to_string(cargo_toml_path).await {
            Ok(c) => c,
            Err(_) => return Ok(vec![]),
        };
        let cargo_toml: toml::Value = match toml::from_str(&content) {
            Ok(t) => t,
            Err(_) => return Ok(vec![]),
        };

        let mut dependencies = Vec::new();

        if let Some(deps) = cargo_toml.get("dependencies").and_then(|d| d.as_table()) {
            for (name, value) in deps {
                let version = match value {
                    toml::Value::String(v) => Some(v.clone()),
                    toml::Value::Table(t) => t
                        .get("version")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string()),
                    _ => None,
                };

                let dep_type = classify_rust_dependency(name);
                dependencies.push(Dependency {
                    name: name.clone(),
                    version,
                    dependency_type: dep_type,
                });
            }
        }

        Ok(dependencies)
    }
}

/// package.json parser for Node.js projects
pub struct NodeParser;

#[async_trait::async_trait]
impl DependencyParser for NodeParser {
    async fn parse_dependencies(&self, path: &Path) -> Result<Vec<Dependency>> {
        let package_json_path = path.join("package.json");
        if !package_json_path.exists() {
            return Ok(vec![]);
        }

        let content = match fs::read_to_string(package_json_path).await {
            Ok(c) => c,
            Err(_) => return Ok(vec![]),
        };
        let package: serde_json::Value = match serde_json::from_str(&content) {
            Ok(p) => p,
            Err(_) => return Ok(vec![]),
        };

        let mut dependencies = Vec::new();

        // Parse both dependencies and devDependencies
        for dep_type_key in &["dependencies", "devDependencies"] {
            if let Some(deps) = package.get(dep_type_key).and_then(|d| d.as_object()) {
                for (name, version) in deps {
                    let version_str = version.as_str().map(|s| s.to_string());
                    let dep_type = classify_node_dependency(name);

                    dependencies.push(Dependency {
                        name: name.clone(),
                        version: version_str,
                        dependency_type: dep_type,
                    });
                }
            }
        }

        Ok(dependencies)
    }
}

/// requirements.txt parser for Python projects
pub struct PythonParser;

#[async_trait::async_trait]
impl DependencyParser for PythonParser {
    async fn parse_dependencies(&self, path: &Path) -> Result<Vec<Dependency>> {
        let requirements_path = path.join("requirements.txt");
        if !requirements_path.exists() {
            return Ok(vec![]);
        }

        let content = fs::read_to_string(requirements_path).await?;
        let mut dependencies = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse requirement line (simplified)
            let parts: Vec<&str> = line.split(&['=', '>', '<', '!', '~'][..]).collect();
            if let Some(name) = parts.first() {
                let name = name.trim();
                let version = if parts.len() > 1 {
                    Some(parts[1].trim().to_string())
                } else {
                    None
                };

                let dep_type = classify_python_dependency(name);
                dependencies.push(Dependency {
                    name: name.to_string(),
                    version,
                    dependency_type: dep_type,
                });
            }
        }

        Ok(dependencies)
    }
}

/// go.mod parser for Go projects
pub struct GoParser;

#[async_trait::async_trait]
impl DependencyParser for GoParser {
    async fn parse_dependencies(&self, path: &Path) -> Result<Vec<Dependency>> {
        let go_mod_path = path.join("go.mod");
        if !go_mod_path.exists() {
            return Ok(vec![]);
        }

        let content = fs::read_to_string(go_mod_path).await?;
        let mut dependencies = Vec::new();
        let mut in_require_block = false;

        for line in content.lines() {
            let line = line.trim();

            if line.starts_with("require (") {
                in_require_block = true;
                continue;
            }

            if line == ")" && in_require_block {
                in_require_block = false;
                continue;
            }

            if line.starts_with("require ") || in_require_block {
                let cleaned_line = line.replace("require ", "");
                let parts: Vec<&str> = cleaned_line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let name = parts[0].to_string();
                    let version = Some(parts[1].to_string());
                    let dep_type = classify_go_dependency(&name);

                    dependencies.push(Dependency {
                        name,
                        version,
                        dependency_type: dep_type,
                    });
                }
            }
        }

        Ok(dependencies)
    }
}

/// pom.xml parser for Java projects (simplified)
pub struct JavaParser;

#[async_trait::async_trait]
impl DependencyParser for JavaParser {
    async fn parse_dependencies(&self, path: &Path) -> Result<Vec<Dependency>> {
        let pom_xml_path = path.join("pom.xml");
        if !pom_xml_path.exists() {
            return Ok(vec![]);
        }

        let content = fs::read_to_string(pom_xml_path).await?;
        let mut dependencies = Vec::new();

        // Very simple XML parsing for Maven dependencies
        // In a real implementation, you'd use a proper XML parser
        let lines: Vec<&str> = content.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i].trim();
            if line.contains("<dependency>") {
                let mut group_id = None;
                let mut artifact_id = None;
                let mut version = None;

                i += 1;
                while i < lines.len() && !lines[i].trim().contains("</dependency>") {
                    let dep_line = lines[i].trim();

                    if let Some(start) = dep_line.find("<groupId>") {
                        if let Some(end) = dep_line.find("</groupId>") {
                            group_id = Some(dep_line[start + 9..end].to_string());
                        }
                    }

                    if let Some(start) = dep_line.find("<artifactId>") {
                        if let Some(end) = dep_line.find("</artifactId>") {
                            artifact_id = Some(dep_line[start + 12..end].to_string());
                        }
                    }

                    if let Some(start) = dep_line.find("<version>") {
                        if let Some(end) = dep_line.find("</version>") {
                            version = Some(dep_line[start + 9..end].to_string());
                        }
                    }

                    i += 1;
                }

                if let (Some(group), Some(artifact)) = (group_id, artifact_id) {
                    let name = format!("{}:{}", group, artifact);
                    let dep_type = classify_java_dependency(&artifact);

                    dependencies.push(Dependency {
                        name,
                        version,
                        dependency_type: dep_type,
                    });
                }
            }
            i += 1;
        }

        Ok(dependencies)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DependencyType;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_cargo_dependency_parser() {
        let temp_dir = TempDir::new().unwrap();
        let cargo_toml = r#"
[package]
name = "test"
version = "0.1.0"

[dependencies]
tokio = "1.0"
serde = { version = "1.0", features = ["derive"] }
sqlx = "0.7"
"#;
        fs::write(temp_dir.path().join("Cargo.toml"), cargo_toml)
            .await
            .unwrap();

        let parser = CargoParser;
        let deps = parser.parse_dependencies(temp_dir.path()).await.unwrap();

        assert_eq!(deps.len(), 3);
        assert!(deps.iter().any(|d| d.name == "tokio"));
        assert!(deps.iter().any(|d| d.name == "serde"));
        assert!(deps.iter().any(|d| d.name == "sqlx"));

        let sqlx_dep = deps.iter().find(|d| d.name == "sqlx").unwrap();
        assert!(matches!(sqlx_dep.dependency_type, DependencyType::Database));
    }

    #[tokio::test]
    async fn test_node_dependency_parser() {
        let temp_dir = TempDir::new().unwrap();
        let package_json = r#"
{
  "dependencies": {
    "express": "^4.18.0",
    "mongoose": "^7.0.0",
    "redis": "^4.6.0"
  },
  "devDependencies": {
    "jest": "^29.0.0"
  }
}
"#;
        fs::write(temp_dir.path().join("package.json"), package_json)
            .await
            .unwrap();

        let parser = NodeParser;
        let deps = parser.parse_dependencies(temp_dir.path()).await.unwrap();

        assert_eq!(deps.len(), 4);
        assert!(deps.iter().any(|d| d.name == "express"));
        assert!(deps.iter().any(|d| d.name == "mongoose"));
        assert!(deps.iter().any(|d| d.name == "redis"));
        assert!(deps.iter().any(|d| d.name == "jest"));

        let express_dep = deps.iter().find(|d| d.name == "express").unwrap();
        assert!(matches!(
            express_dep.dependency_type,
            DependencyType::WebFramework
        ));

        let mongoose_dep = deps.iter().find(|d| d.name == "mongoose").unwrap();
        assert!(matches!(
            mongoose_dep.dependency_type,
            DependencyType::Database
        ));
    }

    #[tokio::test]
    async fn test_python_dependency_parser() {
        let temp_dir = TempDir::new().unwrap();
        let requirements = r#"
fastapi==0.104.1
sqlalchemy==2.0.23
redis==5.0.1
tensorflow==2.13.0
# This is a comment
django>=4.0.0
"#;
        fs::write(temp_dir.path().join("requirements.txt"), requirements)
            .await
            .unwrap();

        let parser = PythonParser;
        let deps = parser.parse_dependencies(temp_dir.path()).await.unwrap();

        assert_eq!(deps.len(), 5);
        assert!(deps.iter().any(|d| d.name == "fastapi"));
        assert!(deps.iter().any(|d| d.name == "sqlalchemy"));
        assert!(deps.iter().any(|d| d.name == "redis"));
        assert!(deps.iter().any(|d| d.name == "tensorflow"));
        assert!(deps.iter().any(|d| d.name == "django"));

        let fastapi_dep = deps.iter().find(|d| d.name == "fastapi").unwrap();
        assert!(matches!(
            fastapi_dep.dependency_type,
            DependencyType::WebFramework
        ));

        let tf_dep = deps.iter().find(|d| d.name == "tensorflow").unwrap();
        assert!(matches!(
            tf_dep.dependency_type,
            DependencyType::MLFramework
        ));
    }

    #[tokio::test]
    async fn test_go_dependency_parser() {
        let temp_dir = TempDir::new().unwrap();
        let go_mod = r#"
module test-app

go 1.19

require (
    github.com/gin-gonic/gin v1.9.1
    gorm.io/gorm v1.25.4
    github.com/go-redis/redis v6.15.9
)
"#;
        fs::write(temp_dir.path().join("go.mod"), go_mod)
            .await
            .unwrap();

        let parser = GoParser;
        let deps = parser.parse_dependencies(temp_dir.path()).await.unwrap();

        assert_eq!(deps.len(), 3);
        assert!(deps.iter().any(|d| d.name.contains("gin")));
        assert!(deps.iter().any(|d| d.name.contains("gorm")));
        assert!(deps.iter().any(|d| d.name.contains("redis")));
    }

    #[tokio::test]
    async fn test_dependency_parser_errors() {
        let temp_dir = TempDir::new().unwrap();

        // Test invalid JSON package.json
        fs::write(temp_dir.path().join("package.json"), "{ invalid json }")
            .await
            .unwrap();
        fs::write(temp_dir.path().join("index.js"), "console.log('test');")
            .await
            .unwrap();
        let analyzer = DependencyAnalyzer::new();
        let result = analyzer
            .analyze_dependencies(temp_dir.path(), "javascript")
            .await;
        // Should handle error gracefully and return empty deps
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());

        // Test invalid TOML Cargo.toml
        fs::write(temp_dir.path().join("Cargo.toml"), "[invalid toml")
            .await
            .unwrap();
        fs::write(temp_dir.path().join("lib.rs"), "// test")
            .await
            .unwrap();
        let result = analyzer.analyze_dependencies(temp_dir.path(), "rust").await;
        // Should handle error gracefully
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }
}
