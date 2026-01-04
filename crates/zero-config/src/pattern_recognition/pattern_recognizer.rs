//! Core pattern recognition implementation

use super::{PatternType, RecognizedPattern, SimilarityEngine};
pub use crate::learning::DeploymentPattern;
use crate::{analysis::CodebaseAnalysis, Result};

/// Pattern recognition engine for finding similar deployments
pub struct PatternRecognizer {
    similarity_engine: SimilarityEngine,
}

impl PatternRecognizer {
    /// Create a new pattern recognizer
    pub fn new() -> Self {
        Self {
            similarity_engine: SimilarityEngine::new(),
        }
    }

    /// Recognize patterns in a codebase analysis
    pub async fn recognize_patterns(
        &self,
        analysis: &CodebaseAnalysis,
    ) -> Result<Vec<RecognizedPattern>> {
        let mut patterns = Vec::new();

        // Recognize language patterns
        patterns.extend(self.recognize_language_patterns(analysis)?);

        // Recognize dependency patterns
        patterns.extend(self.recognize_dependency_patterns(analysis)?);

        // Recognize complexity patterns
        patterns.extend(self.recognize_complexity_patterns(analysis)?);

        // Recognize resource patterns
        patterns.extend(self.recognize_resource_patterns(analysis)?);

        Ok(patterns)
    }

    /// Find similar patterns between two analyses
    pub async fn find_similar_patterns(
        &self,
        analysis: &CodebaseAnalysis,
        pattern_database: &[DeploymentPattern],
    ) -> Result<Vec<(DeploymentPattern, f32)>> {
        let mut similarities = Vec::new();

        for pattern in pattern_database {
            let similarity = self
                .similarity_engine
                .calculate_codebase_similarity(analysis, &pattern.config)
                .await?;

            if similarity > 0.3 {
                // Minimum similarity threshold
                similarities.push((pattern.clone(), similarity));
            }
        }

        // Sort by similarity descending
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(similarities)
    }

    /// Recognize language-specific patterns
    fn recognize_language_patterns(
        &self,
        analysis: &CodebaseAnalysis,
    ) -> Result<Vec<RecognizedPattern>> {
        let mut patterns = Vec::new();

        match analysis.language.as_str() {
            "rust" => {
                patterns.push(RecognizedPattern {
                    pattern_type: PatternType::Language,
                    name: "rust_application".to_string(),
                    confidence: 1.0,
                    description: "Rust application with compiled binary".to_string(),
                    recommendations: vec![
                        "Use minimal container image".to_string(),
                        "Enable static linking for smaller binaries".to_string(),
                        "Consider using scratch or distroless base image".to_string(),
                    ],
                    tags: vec![
                        "compiled".to_string(),
                        "memory-safe".to_string(),
                        "performance".to_string(),
                    ],
                });

                if analysis
                    .framework
                    .as_ref()
                    .is_some_and(|f| f.contains("tokio"))
                {
                    patterns.push(RecognizedPattern {
                        pattern_type: PatternType::Framework,
                        name: "async_rust".to_string(),
                        confidence: 0.9,
                        description: "Async Rust application using Tokio runtime".to_string(),
                        recommendations: vec![
                            "Configure appropriate thread pool size".to_string(),
                            "Monitor async task spawning".to_string(),
                            "Use connection pooling for I/O operations".to_string(),
                        ],
                        tags: vec![
                            "async".to_string(),
                            "concurrent".to_string(),
                            "io-bound".to_string(),
                        ],
                    });
                }
            }
            "python" => {
                patterns.push(RecognizedPattern {
                    pattern_type: PatternType::Language,
                    name: "python_application".to_string(),
                    confidence: 1.0,
                    description: "Python application requiring interpreter".to_string(),
                    recommendations: vec![
                        "Use Python slim or alpine base image".to_string(),
                        "Consider using virtual environments".to_string(),
                        "Monitor memory usage carefully".to_string(),
                    ],
                    tags: vec![
                        "interpreted".to_string(),
                        "dynamic".to_string(),
                        "memory-intensive".to_string(),
                    ],
                });

                if let Some(framework) = &analysis.framework {
                    match framework.as_str() {
                        "django" => {
                            patterns.push(RecognizedPattern {
                                pattern_type: PatternType::Framework,
                                name: "django_web_app".to_string(),
                                confidence: 0.95,
                                description: "Django web application with ORM".to_string(),
                                recommendations: vec![
                                    "Configure database connection pooling".to_string(),
                                    "Set up static file serving".to_string(),
                                    "Consider using gunicorn or uwsgi".to_string(),
                                ],
                                tags: vec!["web".to_string(), "orm".to_string(), "mvc".to_string()],
                            });
                        }
                        "fastapi" => {
                            patterns.push(RecognizedPattern {
                                pattern_type: PatternType::Framework,
                                name: "fastapi_api".to_string(),
                                confidence: 0.95,
                                description: "FastAPI async web API".to_string(),
                                recommendations: vec![
                                    "Enable async database drivers".to_string(),
                                    "Configure CORS if needed".to_string(),
                                    "Use pydantic for data validation".to_string(),
                                ],
                                tags: vec![
                                    "api".to_string(),
                                    "async".to_string(),
                                    "openapi".to_string(),
                                ],
                            });
                        }
                        _ => {}
                    }
                }
            }
            "javascript" | "typescript" => {
                patterns.push(RecognizedPattern {
                    pattern_type: PatternType::Language,
                    name: "node_application".to_string(),
                    confidence: 1.0,
                    description: "Node.js application with npm dependencies".to_string(),
                    recommendations: vec![
                        "Use Node.js LTS version".to_string(),
                        "Optimize npm install with --only=production".to_string(),
                        "Consider using PM2 for process management".to_string(),
                    ],
                    tags: vec![
                        "javascript".to_string(),
                        "npm".to_string(),
                        "event-driven".to_string(),
                    ],
                });
            }
            _ => {}
        }

        Ok(patterns)
    }

    /// Recognize dependency-specific patterns
    fn recognize_dependency_patterns(
        &self,
        analysis: &CodebaseAnalysis,
    ) -> Result<Vec<RecognizedPattern>> {
        let mut patterns = Vec::new();

        // Count dependency types
        let mut db_deps = 0;
        let mut cache_deps = 0;
        let mut web_deps = 0;
        let mut ml_deps = 0;

        for dep in &analysis.dependencies {
            match dep.dependency_type {
                crate::DependencyType::Database => db_deps += 1,
                crate::DependencyType::Cache => cache_deps += 1,
                crate::DependencyType::WebFramework => web_deps += 1,
                crate::DependencyType::MLFramework => ml_deps += 1,
                _ => {}
            }
        }

        // Database patterns
        if db_deps > 0 {
            patterns.push(RecognizedPattern {
                pattern_type: PatternType::Dependency,
                name: "database_application".to_string(),
                confidence: 0.9,
                description: format!("Application with {} database dependencies", db_deps),
                recommendations: vec![
                    "Configure persistent storage".to_string(),
                    "Set up database connection pooling".to_string(),
                    "Consider read replicas for scaling".to_string(),
                ],
                tags: vec![
                    "database".to_string(),
                    "persistent".to_string(),
                    "stateful".to_string(),
                ],
            });
        }

        // Cache patterns
        if cache_deps > 0 {
            patterns.push(RecognizedPattern {
                pattern_type: PatternType::Dependency,
                name: "cache_enabled_application".to_string(),
                confidence: 0.8,
                description: format!("Application using {} caching solutions", cache_deps),
                recommendations: vec![
                    "Configure cache eviction policies".to_string(),
                    "Monitor cache hit rates".to_string(),
                    "Set appropriate TTL values".to_string(),
                ],
                tags: vec![
                    "cache".to_string(),
                    "performance".to_string(),
                    "memory".to_string(),
                ],
            });
        }

        // Web framework patterns
        if web_deps > 0 {
            patterns.push(RecognizedPattern {
                pattern_type: PatternType::Dependency,
                name: "web_service".to_string(),
                confidence: 0.95,
                description: "Web service requiring HTTP exposure".to_string(),
                recommendations: vec![
                    "Configure ingress and load balancing".to_string(),
                    "Set up health checks".to_string(),
                    "Consider rate limiting".to_string(),
                ],
                tags: vec!["web".to_string(), "http".to_string(), "service".to_string()],
            });
        }

        // ML framework patterns
        if ml_deps > 0 {
            patterns.push(RecognizedPattern {
                pattern_type: PatternType::Dependency,
                name: "ml_application".to_string(),
                confidence: 0.85,
                description: "Machine learning application".to_string(),
                recommendations: vec![
                    "Consider GPU acceleration".to_string(),
                    "Allocate sufficient memory for models".to_string(),
                    "Set up model versioning".to_string(),
                ],
                tags: vec![
                    "ml".to_string(),
                    "compute-intensive".to_string(),
                    "gpu".to_string(),
                ],
            });
        }

        // Microservices pattern
        if analysis.dependencies.len() > 8 {
            patterns.push(RecognizedPattern {
                pattern_type: PatternType::Architecture,
                name: "microservice".to_string(),
                confidence: 0.7,
                description: "Complex application with many dependencies".to_string(),
                recommendations: vec![
                    "Enable service mesh".to_string(),
                    "Implement distributed tracing".to_string(),
                    "Set up centralized logging".to_string(),
                ],
                tags: vec![
                    "microservice".to_string(),
                    "complex".to_string(),
                    "distributed".to_string(),
                ],
            });
        }

        Ok(patterns)
    }

    /// Recognize complexity-based patterns
    fn recognize_complexity_patterns(
        &self,
        analysis: &CodebaseAnalysis,
    ) -> Result<Vec<RecognizedPattern>> {
        let mut patterns = Vec::new();

        let complexity = &analysis.complexity;

        // Large codebase pattern
        if complexity.total_lines > 50000 {
            patterns.push(RecognizedPattern {
                pattern_type: PatternType::Complexity,
                name: "large_codebase".to_string(),
                confidence: 0.9,
                description: format!("Large codebase with {} lines", complexity.total_lines),
                recommendations: vec![
                    "Consider breaking into smaller services".to_string(),
                    "Implement comprehensive monitoring".to_string(),
                    "Use staged deployments".to_string(),
                ],
                tags: vec![
                    "large".to_string(),
                    "complex".to_string(),
                    "enterprise".to_string(),
                ],
            });
        }

        // High complexity pattern
        if complexity.cyclomatic_complexity > 5.0 {
            patterns.push(RecognizedPattern {
                pattern_type: PatternType::Complexity,
                name: "high_complexity".to_string(),
                confidence: 0.8,
                description: format!(
                    "High cyclomatic complexity: {:.2}",
                    complexity.cyclomatic_complexity
                ),
                recommendations: vec![
                    "Increase test coverage".to_string(),
                    "Consider refactoring complex functions".to_string(),
                    "Implement careful error handling".to_string(),
                ],
                tags: vec![
                    "complex".to_string(),
                    "high-risk".to_string(),
                    "maintenance".to_string(),
                ],
            });
        }

        // Low maintainability pattern
        if complexity.maintainability_index < 50.0 {
            patterns.push(RecognizedPattern {
                pattern_type: PatternType::Complexity,
                name: "low_maintainability".to_string(),
                confidence: 0.7,
                description: format!(
                    "Low maintainability index: {:.2}",
                    complexity.maintainability_index
                ),
                recommendations: vec![
                    "Focus on code quality improvements".to_string(),
                    "Implement comprehensive testing".to_string(),
                    "Consider architectural refactoring".to_string(),
                ],
                tags: vec![
                    "maintenance".to_string(),
                    "technical-debt".to_string(),
                    "refactoring".to_string(),
                ],
            });
        }

        Ok(patterns)
    }

    /// Recognize resource usage patterns
    fn recognize_resource_patterns(
        &self,
        analysis: &CodebaseAnalysis,
    ) -> Result<Vec<RecognizedPattern>> {
        let mut patterns = Vec::new();

        let resources = &analysis.resources;

        // High memory usage pattern
        if resources.memory_gb > 8.0 {
            patterns.push(RecognizedPattern {
                pattern_type: PatternType::Resource,
                name: "memory_intensive".to_string(),
                confidence: 0.8,
                description: format!("High memory usage: {:.1}GB", resources.memory_gb),
                recommendations: vec![
                    "Monitor memory leaks".to_string(),
                    "Consider memory optimization".to_string(),
                    "Use memory-efficient algorithms".to_string(),
                ],
                tags: vec!["memory-intensive".to_string(), "resource-heavy".to_string()],
            });
        }

        // High CPU usage pattern
        if resources.cpu_cores > 4.0 {
            patterns.push(RecognizedPattern {
                pattern_type: PatternType::Resource,
                name: "cpu_intensive".to_string(),
                confidence: 0.8,
                description: format!("High CPU usage: {:.1} cores", resources.cpu_cores),
                recommendations: vec![
                    "Consider horizontal scaling".to_string(),
                    "Optimize computational algorithms".to_string(),
                    "Use CPU affinity for performance".to_string(),
                ],
                tags: vec!["cpu-intensive".to_string(), "compute-heavy".to_string()],
            });
        }

        // GPU usage pattern
        if resources.gpu_units > 0.0 {
            patterns.push(RecognizedPattern {
                pattern_type: PatternType::Resource,
                name: "gpu_accelerated".to_string(),
                confidence: 0.9,
                description: format!("GPU acceleration: {:.1} units", resources.gpu_units),
                recommendations: vec![
                    "Use GPU-optimized container images".to_string(),
                    "Configure CUDA runtime".to_string(),
                    "Monitor GPU utilization".to_string(),
                ],
                tags: vec![
                    "gpu".to_string(),
                    "acceleration".to_string(),
                    "compute".to_string(),
                ],
            });
        }

        // High network usage pattern
        if resources.network_bandwidth_mbps > 500.0 {
            patterns.push(RecognizedPattern {
                pattern_type: PatternType::Resource,
                name: "network_intensive".to_string(),
                confidence: 0.7,
                description: format!(
                    "High network usage: {:.0}Mbps",
                    resources.network_bandwidth_mbps
                ),
                recommendations: vec![
                    "Implement efficient serialization".to_string(),
                    "Use compression for data transfer".to_string(),
                    "Consider CDN for static content".to_string(),
                ],
                tags: vec!["network-intensive".to_string(), "bandwidth".to_string()],
            });
        }

        Ok(patterns)
    }
}

impl Default for PatternRecognizer {
    fn default() -> Self {
        Self::new()
    }
}
