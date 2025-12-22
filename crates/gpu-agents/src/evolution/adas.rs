//! ADAS (Automated Design of Agentic Systems) GPU implementation
//!
//! Implements the ADAS algorithm from "Automated Design of Agentic Systems" paper
//! for GPU-accelerated meta-agent evolution and self-improving agentic systems.

use crate::evolution::population::{GpuIndividual, GpuPopulation};
use anyhow::Result;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use std::sync::Arc;

/// ADAS meta-agent for automated system design
#[derive(Debug, Clone)]
pub struct AdasMetaAgent {
    /// Agent ID
    pub id: usize,
    /// Agent code/configuration
    pub code: String,
    /// Performance score on benchmark
    pub performance: f64,
    /// Generation number
    pub generation: usize,
    /// Parent agent ID (if any)
    pub parent_id: Option<usize>,
}

/// ADAS population manager for GPU-accelerated evolution
pub struct AdasPopulation {
    device: Arc<CudaDevice>,
    /// Population of meta-agents
    agents: Vec<AdasMetaAgent>,
    /// GPU memory for agent codes (flattened strings)
    agent_codes: CudaSlice<u8>,
    /// GPU memory for performance scores
    performances: CudaSlice<f32>,
    /// Archive of all discovered agents
    archive: Vec<AdasMetaAgent>,
    /// Current generation
    generation: usize,
    /// Maximum code size per agent
    max_code_size: usize,
    /// Population size
    population_size: usize,
}

impl AdasPopulation {
    /// Create new ADAS population
    pub fn new(
        device: Arc<CudaDevice>,
        population_size: usize,
        max_code_size: usize,
    ) -> Result<Self> {
        // Allocate GPU memory
        let total_code_size = population_size * max_code_size;
        let agent_codes = unsafe { device.alloc::<u8>(total_code_size)? };
        let performances = unsafe { device.alloc::<f32>(population_size)? };

        Ok(Self {
            device,
            agents: Vec::with_capacity(population_size),
            agent_codes,
            performances,
            archive: Vec::new(),
            generation: 0,
            max_code_size,
            population_size,
        })
    }

    /// Initialize with seed agents
    pub fn initialize_with_seeds(&mut self, seed_agents: Vec<String>) -> Result<()> {
        self.agents.clear();

        for (i, code) in seed_agents.into_iter().enumerate() {
            let agent = AdasMetaAgent {
                id: i,
                code: code.clone(),
                performance: 0.0,
                generation: 0,
                parent_id: None,
            };

            self.agents.push(agent.clone());
            self.archive.push(agent);

            if self.agents.len() >= self.population_size {
                break;
            }
        }

        // Upload to GPU
        self.upload_to_gpu()?;

        Ok(())
    }

    /// Upload agent data to GPU
    fn upload_to_gpu(&mut self) -> Result<()> {
        // Prepare flattened code data
        let mut code_data = vec![0u8; self.population_size * self.max_code_size];
        let mut performance_data = vec![0f32; self.population_size];

        for (i, agent) in self.agents.iter().enumerate() {
            let start_idx = i * self.max_code_size;
            let code_bytes = agent.code.as_bytes();
            let copy_len = code_bytes.len().min(self.max_code_size);

            code_data[start_idx..start_idx + copy_len].copy_from_slice(&code_bytes[..copy_len]);

            performance_data[i] = agent.performance as f32;
        }

        // Copy to GPU
        self.device
            .htod_copy_into(code_data, &mut self.agent_codes.clone())?;
        self.device
            .htod_copy_into(performance_data, &mut self.performances.clone())?;

        Ok(())
    }

    /// Download agent data from GPU
    fn download_from_gpu(&mut self) -> Result<()> {
        // Download performance scores
        let mut performance_data = vec![0f32; self.population_size];
        self.device
            .dtoh_sync_copy_into(&self.performances, &mut performance_data)?;

        // Update agent performances
        for (i, agent) in self.agents.iter_mut().enumerate() {
            if i < performance_data.len() {
                agent.performance = performance_data[i] as f64;
            }
        }

        Ok(())
    }

    /// Evolve population using ADAS algorithm
    pub fn evolve_generation(&mut self, benchmark_fn: &dyn Fn(&str) -> f64) -> Result<()> {
        // Step 1: Evaluate current population
        self.evaluate_population(benchmark_fn)?;

        // Step 2: Select top performers for replication
        let top_agents = self.select_top_performers(0.3)?; // Top 30%

        // Step 3: Generate new agents through meta-agent modification
        let new_agents = self.generate_offspring(&top_agents)?;

        // Step 4: Replace population
        self.replace_population(new_agents)?;

        // Step 5: Update archive with all agents
        self.update_archive()?;

        self.generation += 1;

        Ok(())
    }

    /// Evaluate population performance on benchmark
    fn evaluate_population(&mut self, benchmark_fn: &dyn Fn(&str) -> f64) -> Result<()> {
        // Launch GPU kernel for parallel evaluation setup
        unsafe {
            crate::evolution::kernels::prepare_adas_evaluation(
                *self.agent_codes.device_ptr() as *const u8,
                *self.performances.device_ptr() as *mut f32,
                self.population_size as u32,
                self.max_code_size as u32,
            );
        }

        // Evaluate each agent (this part stays on CPU for now)
        for agent in &mut self.agents {
            agent.performance = benchmark_fn(&agent.code);
        }

        // Upload results to GPU
        self.upload_to_gpu()?;

        Ok(())
    }

    /// Select top performing agents
    fn select_top_performers(&self, ratio: f32) -> Result<Vec<AdasMetaAgent>> {
        let num_selected = (self.agents.len() as f32 * ratio).ceil() as usize;

        // Sort by performance
        let mut sorted_agents = self.agents.clone();
        sorted_agents.sort_by(|a, b| b.performance.partial_cmp(&a.performance)?);

        Ok(sorted_agents.into_iter().take(num_selected).collect())
    }

    /// Generate offspring through meta-agent self-modification
    fn generate_offspring(&self, parents: &[AdasMetaAgent]) -> Result<Vec<AdasMetaAgent>> {
        let mut offspring = Vec::new();
        let mut next_id = self.archive.len();

        // Fill population with modified versions
        while offspring.len() < self.population_size {
            for parent in parents {
                if offspring.len() >= self.population_size {
                    break;
                }

                // Generate modified agent
                let modified_code = self.modify_agent_code(&parent.code)?;

                let child = AdasMetaAgent {
                    id: next_id,
                    code: modified_code,
                    performance: 0.0, // Will be evaluated
                    generation: self.generation + 1,
                    parent_id: Some(parent.id),
                };

                offspring.push(child);
                next_id += 1;
            }
        }

        Ok(offspring)
    }

    /// Modify agent code using GPU-accelerated mutations
    fn modify_agent_code(&self, parent_code: &str) -> Result<String> {
        // This is a simplified version - in practice would use LLM for modifications
        let mutations = vec![
            "optimize_performance",
            "add_error_handling",
            "improve_efficiency",
            "enhance_collaboration",
            "refactor_architecture",
        ];

        let mutation_idx =
            (rand::random::<f32>() * mutations.len() as f32) as usize % mutations.len();
        let mutation = mutations[mutation_idx];

        // Apply mutation (simplified - would use LLM in practice)
        let modified = format!("{}\n// ADAS Mutation: {}", parent_code, mutation);

        Ok(modified)
    }

    /// Replace population with new generation
    fn replace_population(&mut self, new_agents: Vec<AdasMetaAgent>) -> Result<()> {
        self.agents = new_agents;
        self.upload_to_gpu()?;
        Ok(())
    }

    /// Update archive with current generation
    fn update_archive(&mut self) -> Result<()> {
        for agent in &self.agents {
            self.archive.push(agent.clone());
        }
        Ok(())
    }

    /// Get best agent from current population
    pub fn best_agent(&self) -> Option<&AdasMetaAgent> {
        self.agents
            .iter()
            .max_by(|a, b| a.performance.partial_cmp(&b.performance)?)
    }

    /// Get best agent from archive
    pub fn best_agent_ever(&self) -> Option<&AdasMetaAgent> {
        self.archive
            .iter()
            .max_by(|a, b| a.performance.partial_cmp(&b.performance)?)
    }

    /// Get population statistics
    pub fn statistics(&self) -> AdasStatistics {
        let performances: Vec<f64> = self.agents.iter().map(|a| a.performance).collect();

        let best = performances
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let worst = performances.iter().cloned().fold(f64::INFINITY, f64::min);
        let average = performances.iter().sum::<f64>() / performances.len() as f64;

        // Calculate diversity as standard deviation
        let variance = performances
            .iter()
            .map(|&p| (p - average).powi(2))
            .sum::<f64>()
            / performances.len() as f64;
        let diversity = variance.sqrt();

        AdasStatistics {
            generation: self.generation,
            population_size: self.agents.len(),
            archive_size: self.archive.len(),
            best_performance: best,
            worst_performance: worst,
            average_performance: average,
            diversity: diversity,
        }
    }
}

/// ADAS population statistics
#[derive(Debug, Clone)]
pub struct AdasStatistics {
    pub generation: usize,
    pub population_size: usize,
    pub archive_size: usize,
    pub best_performance: f64,
    pub worst_performance: f64,
    pub average_performance: f64,
    pub diversity: f64,
}

impl std::fmt::Display for AdasStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ADAS Gen {}: Pop={}, Archive={}, Best={:.3}, Avg={:.3}, Div={:.3}",
            self.generation,
            self.population_size,
            self.archive_size,
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
    fn test_adas_population_creation() -> Result<(), Box<dyn std::error::Error>>  {
        if let Ok(device) = CudaDevice::new(0) {
            let device = Arc::new(device);
            let pop = AdasPopulation::new(device, 32, 1024)?;
            assert_eq!(pop.population_size, 32);
            assert_eq!(pop.max_code_size, 1024);
        }
    }

    #[test]
    fn test_adas_seed_initialization() -> Result<(), Box<dyn std::error::Error>>  {
        if let Ok(device) = CudaDevice::new(0) {
            let device = Arc::new(device);
            let mut pop = AdasPopulation::new(device, 4, 256)?;

            let seeds = vec![
                "agent_1_code".to_string(),
                "agent_2_code".to_string(),
                "agent_3_code".to_string(),
            ];

            pop.initialize_with_seeds(seeds)?;
            assert_eq!(pop.agents.len(), 3);
            assert_eq!(pop.archive.len(), 3);
        }
    }
}
