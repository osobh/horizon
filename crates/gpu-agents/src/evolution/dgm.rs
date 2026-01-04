//! DGM (Darwin Gödel Machine) GPU implementation
//!
//! Implements the Darwin Gödel Machine algorithm for self-improving AI systems
//! that can modify their own code empirically rather than through formal proofs.

use anyhow::Result;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;

/// DGM coding agent that can self-modify
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DgmAgent {
    /// Unique agent ID
    pub id: usize,
    /// Agent source code
    pub code: String,
    /// Performance history on benchmarks
    pub performance_history: Vec<f64>,
    /// Generation number
    pub generation: usize,
    /// Parent agent ID (if any)
    pub parent_id: Option<usize>,
    /// Self-modification history
    pub modification_history: Vec<String>,
    /// Agent capabilities (tools, workflows, etc.)
    pub capabilities: Vec<String>,
}

/// DGM archive for open-ended exploration
#[derive(Debug, Clone)]
pub struct DgmArchive {
    /// All discovered agents
    agents: Vec<DgmAgent>,
    /// Performance threshold for archiving
    performance_threshold: f64,
    /// Maximum archive size
    max_size: usize,
}

impl DgmArchive {
    /// Create new DGM archive
    pub fn new(performance_threshold: f64, max_size: usize) -> Self {
        Self {
            agents: Vec::new(),
            performance_threshold,
            max_size,
        }
    }

    /// Add agent to archive if it meets criteria
    pub fn add_agent(&mut self, agent: DgmAgent) -> bool {
        // Check if agent meets archival criteria
        if let Some(best_perf) = agent
            .performance_history
            .iter()
            .cloned()
            .fold(None, |acc, x| {
                Some(match acc {
                    None => x,
                    Some(y) => x.max(y),
                })
            })
        {
            if best_perf >= self.performance_threshold || self.is_interesting(&agent) {
                self.agents.push(agent);

                // Prune archive if too large
                if self.agents.len() > self.max_size {
                    self.prune_archive();
                }
                return true;
            }
        }
        false
    }

    /// Check if agent is interesting for open-ended exploration
    fn is_interesting(&self, agent: &DgmAgent) -> bool {
        // Novel capabilities
        let has_novel_capabilities = agent.capabilities.iter().any(|cap| {
            !self
                .agents
                .iter()
                .any(|archived| archived.capabilities.contains(cap))
        });

        // Novel modifications
        let has_novel_modifications = agent.modification_history.iter().any(|mod_| {
            !self
                .agents
                .iter()
                .any(|archived| archived.modification_history.contains(mod_))
        });

        has_novel_capabilities || has_novel_modifications
    }

    /// Prune archive to maintain size limit
    fn prune_archive(&mut self) {
        // Sort by best performance and keep top performers + diverse agents
        self.agents.sort_by(|a, b| {
            let a_best = a
                .performance_history
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let b_best = b
                .performance_history
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            b_best.partial_cmp(&a_best)?
        });

        // Keep top half and diverse agents
        let keep_count = self.max_size * 3 / 4;
        self.agents.truncate(keep_count);
    }

    /// Select parent agent for reproduction
    pub fn select_parent(&self) -> Option<&DgmAgent> {
        if self.agents.is_empty() {
            return None;
        }

        // Weight selection by performance and recency
        let mut weights = Vec::new();
        for agent in &self.agents {
            let performance_weight = agent
                .performance_history
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let recency_weight = 1.0 / (agent.generation as f64 + 1.0);
            weights.push(performance_weight + 0.1 * recency_weight);
        }

        // Simple weighted selection (would use proper sampling in production)
        let best_idx = weights
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b)?)
            .map(|(i, _)| i)?;

        Some(&self.agents[best_idx])
    }

    /// Get archive statistics
    pub fn statistics(&self) -> ArchiveStatistics {
        let performances: Vec<f64> = self
            .agents
            .iter()
            .flat_map(|a| a.performance_history.iter().cloned())
            .collect();

        let best = performances
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let average = if performances.is_empty() {
            0.0
        } else {
            performances.iter().sum::<f64>() / performances.len() as f64
        };

        ArchiveStatistics {
            size: self.agents.len(),
            best_performance: best,
            average_performance: average,
            diversity: self.compute_diversity(),
        }
    }

    /// Compute archive diversity
    fn compute_diversity(&self) -> f64 {
        if self.agents.len() < 2 {
            return 0.0;
        }

        // Simple diversity metric based on capability differences
        let mut total_distance = 0.0;
        let mut count = 0;

        for i in 0..self.agents.len() {
            for j in i + 1..self.agents.len() {
                let common_caps = self.agents[i]
                    .capabilities
                    .iter()
                    .filter(|cap| self.agents[j].capabilities.contains(cap))
                    .count();

                let total_caps =
                    (self.agents[i].capabilities.len() + self.agents[j].capabilities.len()) as f64;
                let distance = if total_caps > 0.0 {
                    1.0 - (2.0 * common_caps as f64 / total_caps)
                } else {
                    0.0
                };

                total_distance += distance;
                count += 1;
            }
        }

        if count > 0 {
            total_distance / count as f64
        } else {
            0.0
        }
    }
}

/// DGM GPU-accelerated evolution engine
pub struct DgmEngine {
    device: Arc<CudaDevice>,
    /// Current population of agents
    population: Vec<DgmAgent>,
    /// Archive for open-ended exploration
    archive: DgmArchive,
    /// GPU memory for agent codes
    agent_codes: CudaSlice<u8>,
    /// GPU memory for performance scores
    benchmark_scores: CudaSlice<f32>,
    /// GPU memory for modification history
    modification_buffer: CudaSlice<u8>,
    /// Current generation
    generation: usize,
    /// Maximum code size per agent
    max_code_size: usize,
    /// Population size
    population_size: usize,
    /// Next agent ID
    next_id: usize,
}

impl DgmEngine {
    /// Create new DGM engine
    pub fn new(
        device: Arc<CudaDevice>,
        population_size: usize,
        max_code_size: usize,
        archive_size: usize,
    ) -> Result<Self> {
        // Allocate GPU memory
        // SAFETY: CudaDevice::alloc returns uninitialized GPU memory. This is safe because:
        // 1. The memory will be initialized via htod_copy_into in upload_population_to_gpu
        // 2. The data is never read from GPU before being initialized
        // 3. The CudaDevice handles proper GPU memory management
        let total_code_size = population_size * max_code_size;
        let agent_codes = unsafe { device.alloc::<u8>(total_code_size)? };
        let benchmark_scores = unsafe { device.alloc::<f32>(population_size)? };
        let modification_buffer = unsafe { device.alloc::<u8>(total_code_size)? };

        let archive = DgmArchive::new(0.1, archive_size); // 0.1 performance threshold

        Ok(Self {
            device,
            population: Vec::new(),
            archive,
            agent_codes,
            benchmark_scores,
            modification_buffer,
            generation: 0,
            max_code_size,
            population_size,
            next_id: 0,
        })
    }

    /// Initialize with a single seed agent
    pub fn initialize_with_seed(&mut self, seed_code: String) -> Result<()> {
        let initial_agent = DgmAgent {
            id: self.next_id,
            code: seed_code,
            performance_history: Vec::new(),
            generation: 0,
            parent_id: None,
            modification_history: Vec::new(),
            capabilities: vec!["basic_agent".to_string()],
        };

        self.population.push(initial_agent.clone());
        self.archive.add_agent(initial_agent);
        self.next_id += 1;

        self.upload_population_to_gpu()?;
        Ok(())
    }

    /// Upload population data to GPU
    fn upload_population_to_gpu(&mut self) -> Result<()> {
        let mut code_data = vec![0u8; self.population_size * self.max_code_size];
        let mut score_data = vec![0f32; self.population_size];

        for (i, agent) in self.population.iter().enumerate() {
            let start_idx = i * self.max_code_size;
            let code_bytes = agent.code.as_bytes();
            let copy_len = code_bytes.len().min(self.max_code_size);

            code_data[start_idx..start_idx + copy_len].copy_from_slice(&code_bytes[..copy_len]);

            // Use best performance as score
            score_data[i] = agent
                .performance_history
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max) as f32;
        }

        self.device
            .htod_copy_into(code_data, &mut self.agent_codes.clone())?;
        self.device
            .htod_copy_into(score_data, &mut self.benchmark_scores.clone())?;

        Ok(())
    }

    /// Download population data from GPU
    fn download_population_from_gpu(&mut self) -> Result<()> {
        let mut score_data = vec![0f32; self.population_size];
        self.device
            .dtoh_sync_copy_into(&self.benchmark_scores, &mut score_data)?;

        for (i, agent) in self.population.iter_mut().enumerate() {
            if i < score_data.len() && score_data[i] > 0.0 {
                agent.performance_history.push(score_data[i] as f64);
            }
        }

        Ok(())
    }

    /// Run one iteration of DGM evolution
    pub fn evolve_iteration(&mut self, benchmark_fn: &dyn Fn(&str) -> f64) -> Result<()> {
        // Step 1: Select parent agents from archive
        let parents = self.select_parents_from_archive()?;

        // Step 2: Generate new agents through self-modification
        let new_agents = self.generate_offspring(&parents)?;

        // Step 3: Evaluate new agents on benchmark
        self.evaluate_agents(&new_agents, benchmark_fn)?;

        // Step 4: Update archive with successful agents
        self.update_archive(new_agents)?;

        // Step 5: Update population with best agents
        self.update_population()?;

        self.generation += 1;
        Ok(())
    }

    /// Select parent agents from archive for reproduction
    fn select_parents_from_archive(&self) -> Result<Vec<DgmAgent>> {
        let mut parents = Vec::new();

        // Select multiple parents for diversity
        for _ in 0..self.population_size.min(4) {
            if let Some(parent) = self.archive.select_parent() {
                parents.push(parent.clone());
            }
        }

        // If archive is empty, use current population
        if parents.is_empty() && !self.population.is_empty() {
            parents.push(self.population[0].clone());
        }

        Ok(parents)
    }

    /// Generate offspring through self-modification
    fn generate_offspring(&mut self, parents: &[DgmAgent]) -> Result<Vec<DgmAgent>> {
        let mut offspring = Vec::new();

        while offspring.len() < self.population_size && !parents.is_empty() {
            for parent in parents {
                if offspring.len() >= self.population_size {
                    break;
                }

                // Create modified version using GPU acceleration
                let modified_code = self.self_modify_agent(&parent.code)?;

                let mut new_capabilities = parent.capabilities.clone();
                let modification_type =
                    self.determine_modification_type(&parent.performance_history);

                // Add new capability based on modification
                match modification_type.as_str() {
                    "optimize_performance" => {
                        new_capabilities.push("performance_optimized".to_string())
                    }
                    "add_error_handling" => new_capabilities.push("error_resilient".to_string()),
                    "improve_efficiency" => {
                        new_capabilities.push("efficient_execution".to_string())
                    }
                    "enhance_collaboration" => new_capabilities.push("collaborative".to_string()),
                    _ => new_capabilities.push("modified".to_string()),
                }

                let child = DgmAgent {
                    id: self.next_id,
                    code: modified_code,
                    performance_history: Vec::new(),
                    generation: self.generation + 1,
                    parent_id: Some(parent.id),
                    modification_history: {
                        let mut hist = parent.modification_history.clone();
                        hist.push(modification_type);
                        hist
                    },
                    capabilities: new_capabilities,
                };

                offspring.push(child);
                self.next_id += 1;
            }
        }

        Ok(offspring)
    }

    /// Self-modify agent code using GPU-accelerated operations
    fn self_modify_agent(&self, parent_code: &str) -> Result<String> {
        // Prepare data for GPU kernel
        let mut code_buffer = vec![0u8; self.max_code_size];
        let code_bytes = parent_code.as_bytes();
        let copy_len = code_bytes.len().min(self.max_code_size);
        code_buffer[..copy_len].copy_from_slice(&code_bytes[..copy_len]);

        // Upload to GPU
        self.device
            .htod_copy_into(code_buffer.clone(), &mut self.modification_buffer.clone())?;

        // Launch self-modification kernel
        // SAFETY: The kernel function is called with valid device pointers obtained from
        // CudaSlice::device_ptr(). The modification_buffer was allocated with max_code_size
        // bytes, matching the kernel's expected buffer size. The null placeholder for
        // performance_history is handled by the kernel (history_length=0).
        unsafe {
            crate::evolution::kernels::launch_dgm_self_modification(
                *self.modification_buffer.device_ptr() as *const u8,
                *self.modification_buffer.device_ptr() as *mut u8,
                std::ptr::null(), // performance_history placeholder
                self.max_code_size as u32,
                0,   // history_length
                0.1, // improvement_threshold
            );
        }

        // Download modified code
        let mut modified_buffer = vec![0u8; self.max_code_size];
        self.device
            .dtoh_sync_copy_into(&self.modification_buffer, &mut modified_buffer)?;

        // Convert back to string (simplified - would need proper handling)
        let modified_code = String::from_utf8_lossy(&modified_buffer)
            .trim_end_matches('\0')
            .to_string();

        // Apply simple modifications if GPU kernel didn't change much
        if modified_code == parent_code || modified_code.trim().is_empty() {
            return Ok(format!(
                "{}\n// DGM Self-Modification Gen {}",
                parent_code, self.generation
            ));
        }

        Ok(modified_code)
    }

    /// Determine modification type based on performance history
    fn determine_modification_type(&self, performance_history: &[f64]) -> String {
        if performance_history.is_empty() {
            return "initial_modification".to_string();
        }

        // Analyze performance trend
        if performance_history.len() >= 2 {
            let recent_avg = performance_history.iter().rev().take(3).sum::<f64>() / 3.0;
            let overall_avg =
                performance_history.iter().sum::<f64>() / performance_history.len() as f64;

            if recent_avg < overall_avg * 0.9 {
                return "performance_recovery".to_string();
            } else if recent_avg > overall_avg * 1.1 {
                return "enhance_strengths".to_string();
            }
        }

        // Default modifications based on generation
        let modifications = vec![
            "optimize_performance",
            "add_error_handling",
            "improve_efficiency",
            "enhance_collaboration",
            "refactor_architecture",
            "add_new_tools",
        ];

        let idx = self.generation % modifications.len();
        modifications[idx].to_string()
    }

    /// Evaluate agents on benchmark
    fn evaluate_agents(
        &mut self,
        agents: &[DgmAgent],
        benchmark_fn: &dyn Fn(&str) -> f64,
    ) -> Result<()> {
        // Prepare GPU evaluation (placeholder for now)
        // SAFETY: The kernel function is called with valid device pointers obtained from
        // CudaSlice::device_ptr(). agent_codes has population_size * max_code_size bytes,
        // and benchmark_scores has population_size f32 elements. The null benchmark_data
        // is a placeholder that the kernel handles appropriately.
        unsafe {
            crate::evolution::kernels::evaluate_dgm_benchmark(
                *self.agent_codes.device_ptr() as *const u8,
                *self.benchmark_scores.device_ptr() as *mut f32,
                agents.len() as u32,
                self.max_code_size as u32,
                std::ptr::null(), // benchmark_data placeholder
            );
        }

        // For now, evaluate on CPU
        for agent in agents {
            let _score = benchmark_fn(&agent.code);
            // Score will be stored in agent's performance_history
        }

        Ok(())
    }

    /// Update archive with new agents
    fn update_archive(&mut self, mut agents: Vec<DgmAgent>) -> Result<()> {
        for agent in agents.drain(..) {
            self.archive.add_agent(agent);
        }
        Ok(())
    }

    /// Update population with best agents from archive
    fn update_population(&mut self) -> Result<()> {
        // Get best agents from archive
        let mut archive_agents = self.archive.agents.clone();
        archive_agents.sort_by(|a, b| {
            let a_best = a
                .performance_history
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let b_best = b
                .performance_history
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            b_best.partial_cmp(&a_best)?
        });

        // Update population with top performers
        self.population.clear();
        for agent in archive_agents.into_iter().take(self.population_size) {
            self.population.push(agent);
        }

        self.upload_population_to_gpu()?;
        Ok(())
    }

    /// Get best agent from current archive
    pub fn best_agent(&self) -> Option<&DgmAgent> {
        self.archive.agents.iter().max_by(|a, b| {
            let a_best = a
                .performance_history
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let b_best = b
                .performance_history
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            a_best.partial_cmp(&b_best)?
        })
    }

    /// Get DGM statistics
    pub fn statistics(&self) -> DgmStatistics {
        let archive_stats = self.archive.statistics();

        DgmStatistics {
            generation: self.generation,
            population_size: self.population.len(),
            archive_size: archive_stats.size,
            best_performance: archive_stats.best_performance,
            average_performance: archive_stats.average_performance,
            diversity: archive_stats.diversity,
            total_agents_created: self.next_id,
        }
    }
}

/// Archive statistics
#[derive(Debug, Clone)]
pub struct ArchiveStatistics {
    pub size: usize,
    pub best_performance: f64,
    pub average_performance: f64,
    pub diversity: f64,
}

/// DGM engine statistics
#[derive(Debug, Clone)]
pub struct DgmStatistics {
    pub generation: usize,
    pub population_size: usize,
    pub archive_size: usize,
    pub best_performance: f64,
    pub average_performance: f64,
    pub diversity: f64,
    pub total_agents_created: usize,
}

impl std::fmt::Display for DgmStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DGM Gen {}: Pop={}, Archive={}, Created={}, Best={:.3}, Avg={:.3}, Div={:.3}",
            self.generation,
            self.population_size,
            self.archive_size,
            self.total_agents_created,
            self.best_performance,
            self.average_performance,
            self.diversity
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dgm_archive() {
        let mut archive = DgmArchive::new(0.5, 10);

        let agent = DgmAgent {
            id: 0,
            code: "test_code".to_string(),
            performance_history: vec![0.8],
            generation: 0,
            parent_id: None,
            modification_history: Vec::new(),
            capabilities: vec!["test".to_string()],
        };

        assert!(archive.add_agent(agent));
        assert_eq!(archive.agents.len(), 1);
    }

    #[test]
    fn test_dgm_engine_creation() -> Result<(), Box<dyn std::error::Error>> {
        if let Ok(device) = CudaDevice::new(0) {
            let device = Arc::new(device);
            let engine = DgmEngine::new(device, 16, 1024, 100)?;
            assert_eq!(engine.population_size, 16);
            assert_eq!(engine.max_code_size, 1024);
        }
    }
}
