//! HPC-AI Project definitions and dependency management
//!
//! Defines all 15 projects in the HPC-AI ecosystem with their
//! dependencies and metadata.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Project category for organization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProjectCategory {
    /// Core infrastructure (00-02)
    Core,
    /// Storage and communication (03-05)
    Storage,
    /// GPU management (06)
    Gpu,
    /// Data transfer and orchestration (07-08)
    Orchestration,
    /// Data processing and ML (09-10)
    DataProcessing,
    /// Networking (11-12)
    Networking,
    /// UI and observability (13-14)
    Observability,
}

impl ProjectCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Core => "Core Infrastructure",
            Self::Storage => "Storage & Communication",
            Self::Gpu => "GPU Management",
            Self::Orchestration => "Orchestration",
            Self::DataProcessing => "Data Processing & ML",
            Self::Networking => "Networking",
            Self::Observability => "Observability",
        }
    }
}

/// HPC-AI Project definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    /// Project identifier (e.g., "rnccl", "torch")
    pub id: String,
    /// Directory name (e.g., "05-rnccl")
    pub dir_name: String,
    /// Display name
    pub name: String,
    /// Short description
    pub description: String,
    /// Project category
    pub category: ProjectCategory,
    /// Dependencies on other projects (by id)
    pub dependencies: Vec<String>,
    /// Whether this project is deployable
    pub deployable: bool,
    /// Default port (if applicable)
    pub default_port: Option<u16>,
}

impl Project {
    /// Create a new project definition
    pub fn new(
        id: &str,
        dir_name: &str,
        name: &str,
        description: &str,
        category: ProjectCategory,
    ) -> Self {
        Self {
            id: id.to_string(),
            dir_name: dir_name.to_string(),
            name: name.to_string(),
            description: description.to_string(),
            category,
            dependencies: Vec::new(),
            deployable: true,
            default_port: None,
        }
    }

    /// Add a dependency
    pub fn with_dependency(mut self, dep: &str) -> Self {
        self.dependencies.push(dep.to_string());
        self
    }

    /// Add multiple dependencies
    pub fn with_dependencies(mut self, deps: &[&str]) -> Self {
        self.dependencies.extend(deps.iter().map(|s| s.to_string()));
        self
    }

    /// Set as non-deployable (library only)
    pub fn non_deployable(mut self) -> Self {
        self.deployable = false;
        self
    }

    /// Set default port
    pub fn with_port(mut self, port: u16) -> Self {
        self.default_port = Some(port);
        self
    }
}

/// Get all HPC-AI projects with their dependencies
pub fn get_all_projects() -> Vec<Project> {
    vec![
        // Core Infrastructure (00-02)
        Project::new(
            "rustg",
            "00-rust",
            "Rustg",
            "GPU-accelerated Rust compiler",
            ProjectCategory::Core,
        )
        .non_deployable(),
        Project::new(
            "channels",
            "02-hpc-channels",
            "HPC Channels",
            "Ultra-low latency IPC foundation",
            ProjectCategory::Core,
        )
        .non_deployable(),
        // Storage & Communication (03-05)
        Project::new(
            "parcode",
            "03-parcode",
            "Parcode",
            "Lazy-loading object storage",
            ProjectCategory::Storage,
        )
        .with_dependency("channels"),
        Project::new(
            "rmpi",
            "04-rmpi",
            "RMPI",
            "Rust MPI utilities",
            ProjectCategory::Storage,
        )
        .with_dependency("channels"),
        Project::new(
            "rnccl",
            "05-rnccl",
            "RNCCL",
            "GPU collective communication",
            ProjectCategory::Storage,
        )
        .with_dependency("channels"),
        // GPU Management (06)
        Project::new(
            "slai",
            "06-slai",
            "SLAI",
            "GPU detection and cluster management",
            ProjectCategory::Gpu,
        )
        .with_dependency("rnccl")
        .with_port(9100),
        // Data Transfer & Orchestration (07-08)
        Project::new(
            "warp",
            "07-warp",
            "WARP",
            "GPU-accelerated bulk data transfer",
            ProjectCategory::Orchestration,
        )
        .with_dependency("channels"),
        Project::new(
            "stratoswarm",
            "08-stratoswarm",
            "StratoSwarm",
            "Unified orchestration platform",
            ProjectCategory::Orchestration,
        )
        .with_dependencies(&["channels", "warp"])
        .with_port(8080),
        // Data Processing & ML (09-10)
        Project::new(
            "spark",
            "09-rustyspark",
            "RustySpark",
            "Distributed data processing",
            ProjectCategory::DataProcessing,
        )
        .with_dependencies(&["rmpi", "warp"]),
        Project::new(
            "torch",
            "10-rustytorch",
            "RustyTorch",
            "GPU-accelerated ML training",
            ProjectCategory::DataProcessing,
        )
        .with_dependencies(&["rnccl", "slai", "rmpi"]),
        // Networking (11-12)
        Project::new(
            "vortex",
            "11-vortex",
            "Vortex",
            "Intelligent edge proxy",
            ProjectCategory::Networking,
        )
        .with_dependency("slai")
        .with_port(8443),
        Project::new(
            "nebula",
            "12-nebula",
            "Nebula",
            "Real-time communication",
            ProjectCategory::Networking,
        )
        .with_dependencies(&["stratoswarm", "rmpi"])
        .with_port(9090),
        // Observability (14)
        Project::new(
            "argus",
            "14-argus",
            "Argus",
            "Observability platform",
            ProjectCategory::Observability,
        )
        .with_dependency("channels")
        .with_port(9090),
    ]
}

/// Resolve all dependencies for a set of projects
pub fn resolve_dependencies(selected: &[String]) -> Vec<String> {
    let all_projects = get_all_projects();
    let project_map: std::collections::HashMap<_, _> = all_projects
        .iter()
        .map(|p| (p.id.as_str(), p))
        .collect();

    let mut resolved: HashSet<String> = selected.iter().cloned().collect();
    let mut changed = true;

    // Iteratively add dependencies until no changes
    while changed {
        changed = false;
        let current: Vec<_> = resolved.iter().cloned().collect();
        for id in current {
            if let Some(project) = project_map.get(id.as_str()) {
                for dep in &project.dependencies {
                    if !resolved.contains(dep) {
                        resolved.insert(dep.clone());
                        changed = true;
                    }
                }
            }
        }
    }

    // Return in dependency order (dependencies first)
    let mut result: Vec<String> = resolved.into_iter().collect();

    // Simple topological sort
    result.sort_by(|a, b| {
        let a_deps = project_map.get(a.as_str()).map(|p| p.dependencies.len()).unwrap_or(0);
        let b_deps = project_map.get(b.as_str()).map(|p| p.dependencies.len()).unwrap_or(0);
        a_deps.cmp(&b_deps)
    });

    result
}

/// Get projects by category
pub fn get_projects_by_category(category: ProjectCategory) -> Vec<Project> {
    get_all_projects()
        .into_iter()
        .filter(|p| p.category == category)
        .collect()
}

/// Get deployable projects only
pub fn get_deployable_projects() -> Vec<Project> {
    get_all_projects()
        .into_iter()
        .filter(|p| p.deployable)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_count() {
        let projects = get_all_projects();
        // 13 projects (excluding hpc-cli and horizon which are tools, not services)
        assert_eq!(projects.len(), 13);
    }

    #[test]
    fn test_dependency_resolution() {
        let selected = vec!["torch".to_string()];
        let resolved = resolve_dependencies(&selected);

        // torch depends on rnccl, slai, rmpi
        // rnccl depends on channels
        // slai depends on rnccl
        // rmpi depends on channels
        assert!(resolved.contains(&"torch".to_string()));
        assert!(resolved.contains(&"rnccl".to_string()));
        assert!(resolved.contains(&"slai".to_string()));
        assert!(resolved.contains(&"channels".to_string()));
    }

    #[test]
    fn test_deployable_projects() {
        let deployable = get_deployable_projects();
        // rustg and channels are non-deployable
        assert!(deployable.iter().all(|p| p.id != "rustg" && p.id != "channels"));
    }
}
