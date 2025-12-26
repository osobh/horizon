//! Service dependency resolution and management

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Service dependency definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceDependency {
    /// Dependency ID
    pub id: Uuid,
    /// Service that depends on another
    pub dependent_service_id: Uuid,
    /// Service that is depended upon
    pub dependency_service_id: Uuid,
    /// Type of dependency
    pub dependency_type: DependencyType,
    /// Whether dependency is critical
    pub critical: bool,
}

/// Types of service dependencies
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DependencyType {
    /// Hard dependency - service cannot function without it
    Hard,
    /// Soft dependency - service can function with degraded performance
    Soft,
    /// Optional dependency - service can function normally without it
    Optional,
    /// Circular reference (should be avoided)
    Circular,
}

/// Dependency graph for resolution ordering
pub struct DependencyGraph {
    /// Adjacency list representation
    pub graph: HashMap<Uuid, Vec<Uuid>>,
    /// Service metadata
    pub services: HashMap<Uuid, ServiceMetadata>,
}

/// Metadata for services in dependency graph
#[derive(Debug, Clone)]
pub struct ServiceMetadata {
    pub id: Uuid,
    pub name: String,
    pub tier: super::types::RecoveryTier,
    pub critical: bool,
}

impl DependencyGraph {
    /// Create new dependency graph
    pub fn new() -> Self {
        Self {
            graph: HashMap::new(),
            services: HashMap::new(),
        }
    }

    /// Add service to graph
    pub fn add_service(&mut self, service: ServiceMetadata) {
        self.services.insert(service.id, service.clone());
        self.graph.entry(service.id).or_insert_with(Vec::new);
    }

    /// Add dependency between services
    pub fn add_dependency(&mut self, dependent: Uuid, dependency: Uuid) {
        self.graph.entry(dependent).or_insert_with(Vec::new).push(dependency);
        self.graph.entry(dependency).or_insert_with(Vec::new);
    }

    /// Check for circular dependencies using DFS
    pub fn has_cycles(&self) -> bool {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for &service_id in self.graph.keys() {
            if !visited.contains(&service_id) {
                if self.has_cycle_util(service_id, &mut visited, &mut rec_stack) {
                    return true;
                }
            }
        }
        false
    }

    fn has_cycle_util(
        &self,
        service_id: Uuid,
        visited: &mut HashSet<Uuid>,
        rec_stack: &mut HashSet<Uuid>,
    ) -> bool {
        visited.insert(service_id);
        rec_stack.insert(service_id);

        if let Some(dependencies) = self.graph.get(&service_id) {
            for &dep_id in dependencies {
                if !visited.contains(&dep_id) {
                    if self.has_cycle_util(dep_id, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(&dep_id) {
                    return true;
                }
            }
        }

        rec_stack.remove(&service_id);
        false
    }

    /// Get topological sort order for recovery planning
    pub fn topological_sort(&self) -> Result<Vec<Uuid>, String> {
        if self.has_cycles() {
            return Err("Dependency graph contains cycles".to_string());
        }

        let mut in_degree = HashMap::new();
        let mut queue = std::collections::VecDeque::new();
        let mut result = Vec::new();

        // Calculate in-degrees
        for &service_id in self.graph.keys() {
            in_degree.insert(service_id, 0);
        }

        for dependencies in self.graph.values() {
            for &dep_id in dependencies {
                *in_degree.get_mut(&dep_id).unwrap() += 1;
            }
        }

        // Find nodes with no incoming edges
        for (&service_id, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(service_id);
            }
        }

        // Process queue
        while let Some(service_id) = queue.pop_front() {
            result.push(service_id);

            if let Some(dependencies) = self.graph.get(&service_id) {
                for &dep_id in dependencies {
                    let degree = in_degree.get_mut(&dep_id).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(dep_id);
                    }
                }
            }
        }

        if result.len() != self.graph.len() {
            Err("Failed to resolve all dependencies".to_string())
        } else {
            Ok(result)
        }
    }

    /// Get critical path for recovery planning
    pub fn get_critical_path(&self) -> Vec<Uuid> {
        // Return services sorted by tier priority
        let mut services: Vec<_> = self.services.values().collect();
        services.sort_by(|a, b| {
            // Critical services first, then by tier
            b.critical.cmp(&a.critical)
                .then_with(|| a.tier.default_rto_minutes().cmp(&b.tier.default_rto_minutes()))
        });
        
        services.into_iter().map(|s| s.id).collect()
    }
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}
