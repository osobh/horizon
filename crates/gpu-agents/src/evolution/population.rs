//! GPU-optimized population representation

use anyhow::Result;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use std::sync::Arc;

/// GPU-friendly individual representation
#[derive(Debug, Clone)]
pub struct GpuIndividual {
    pub index: usize,
    pub genome: Vec<f32>,
    pub fitness: Option<f64>,
}

/// GPU population using Structure of Arrays (SoA) layout
pub struct GpuPopulation {
    device: Arc<CudaDevice>,
    /// Genome data (flattened: population_size * genome_size)
    genomes: CudaSlice<f32>,
    /// Fitness scores (population_size)
    fitness_scores: CudaSlice<f32>,
    /// Valid fitness flags (population_size)
    fitness_valid: CudaSlice<u8>,
    /// Population metadata
    population_size: usize,
    genome_size: usize,
    /// Host-side cache for best individual
    best_individual_cache: Option<GpuIndividual>,
    /// RNG states for CUDA kernels
    rng_states: Option<*mut u8>,
}

impl GpuPopulation {
    /// Create new GPU population
    pub fn new(
        device: Arc<CudaDevice>,
        population_size: usize,
        genome_size: usize,
    ) -> Result<Self> {
        // Calculate sizes with validation
        let total_genome_size = population_size * genome_size;

        // Check for potential overflow
        if total_genome_size < population_size || total_genome_size < genome_size {
            return Err(anyhow::anyhow!(
                "Integer overflow in genome size calculation: {} * {} = {}",
                population_size,
                genome_size,
                total_genome_size
            ));
        }

        // Allocate GPU memory
        let genomes = unsafe { device.alloc::<f32>(total_genome_size)? };
        let fitness_scores = unsafe { device.alloc::<f32>(population_size)? };
        let fitness_valid = device.alloc_zeros::<u8>(population_size)?;

        // Initialize RNG states for CUDA kernels
        let rng_states =
            unsafe { crate::evolution::kernels::setup_rng_states(population_size as u32, 12345) };
        let rng_states = if rng_states.is_null() {
            None
        } else {
            Some(rng_states)
        };

        Ok(Self {
            device,
            genomes,
            fitness_scores,
            fitness_valid,
            population_size,
            genome_size,
            best_individual_cache: None,
            rng_states,
        })
    }

    /// Initialize population with random genomes
    pub fn initialize_random(&mut self) -> Result<()> {
        if let Some(rng_states) = self.rng_states {
            // Launch kernel to generate random genomes
            unsafe {
                crate::evolution::kernels::launch_random_init(
                    *self.genomes.device_ptr() as *mut f32,
                    self.population_size as u32,
                    self.genome_size as u32,
                    rng_states,
                    std::ptr::null_mut(), // stream
                );
            }

            // Invalidate fitness
            self.invalidate_fitness();
            self.best_individual_cache = None;

            Ok(())
        } else {
            Err(anyhow::anyhow!("RNG states not initialized"))
        }
    }

    /// Check if population has valid fitness scores
    pub fn has_fitness(&self) -> bool {
        // Check first element as indicator (all should be synchronized)
        let mut has_fitness = vec![0u8; 1];
        let first_element_slice = self.fitness_valid.slice(0..1);
        self.device
            .dtoh_sync_copy_into(&first_element_slice, &mut has_fitness)
            .unwrap_or_default();
        has_fitness[0] != 0
    }

    /// Invalidate all fitness scores
    pub fn invalidate_fitness(&mut self) {
        // Set all fitness_valid flags to 0
        let zeros = vec![0u8; self.population_size];
        self.device
            .htod_copy_into(zeros, &mut self.fitness_valid)
            .ok();
        self.best_individual_cache = None;
    }

    /// Get best individual (highest fitness)
    pub fn best_individual(&self) -> Result<GpuIndividual> {
        if !self.has_fitness() {
            return Err(anyhow::anyhow!("Fitness not evaluated"));
        }

        // Allocate device memory for results
        let mut best_idx_device = self.device.alloc_zeros::<u32>(1)?;
        let mut best_value_device = self.device.alloc_zeros::<f32>(1)?;

        // Find index of best fitness on GPU
        unsafe {
            crate::evolution::kernels::find_best_fitness(
                *self.fitness_scores.device_ptr() as *const f32,
                *best_idx_device.device_ptr() as *mut u32,
                *best_value_device.device_ptr() as *mut f32,
                self.population_size as u32,
                std::ptr::null_mut(), // stream
            );
        }

        // Copy results to host
        let mut best_idx = vec![0u32; 1];
        let mut best_value = vec![0f32; 1];
        self.device
            .dtoh_sync_copy_into(&best_idx_device, &mut best_idx)?;
        self.device
            .dtoh_sync_copy_into(&best_value_device, &mut best_value)?;

        let best_idx = best_idx[0] as usize;

        // Copy individual data from GPU
        let genome_start = best_idx * self.genome_size;
        let genome_end = genome_start + self.genome_size;

        let mut genome = vec![0f32; self.genome_size];
        let genome_slice = self.genomes.slice(genome_start..genome_end);
        self.device
            .dtoh_sync_copy_into(&genome_slice, &mut genome)?;

        Ok(GpuIndividual {
            index: best_idx,
            genome,
            fitness: Some(best_value[0] as f64),
        })
    }

    /// Get best fitness value
    pub fn best_fitness(&self) -> Option<f64> {
        if !self.has_fitness() {
            return None;
        }

        // Use cached value if available
        if let Some(ref individual) = self.best_individual_cache {
            return individual.fitness;
        }

        // Otherwise compute
        self.best_individual().ok().and_then(|ind| ind.fitness)
    }

    /// Calculate average fitness
    pub fn average_fitness(&self) -> f64 {
        if !self.has_fitness() {
            return 0.0;
        }

        // Allocate device memory for result
        let mut average_device = self
            .device
            .alloc_zeros::<f32>(1)
            .unwrap_or_else(|_| unsafe { self.device.alloc::<f32>(1)? });

        // Launch kernel to compute average
        unsafe {
            crate::evolution::kernels::compute_average_fitness(
                *self.fitness_scores.device_ptr() as *const f32,
                *average_device.device_ptr() as *mut f32,
                self.population_size as u32,
                std::ptr::null_mut(), // stream
            );
        }

        // Copy result to host
        let mut average = vec![0f32; 1];
        self.device
            .dtoh_sync_copy_into(&average_device, &mut average)
            .unwrap_or_default();
        average[0] as f64
    }

    /// Calculate diversity index
    pub fn diversity_index(&self) -> f64 {
        // Allocate device memory for result
        let mut diversity_device = self
            .device
            .alloc_zeros::<f32>(1)
            .unwrap_or_else(|_| unsafe { self.device.alloc::<f32>(1)? });

        // Launch kernel to compute diversity
        unsafe {
            crate::evolution::kernels::compute_diversity(
                *self.genomes.device_ptr() as *const f32,
                *diversity_device.device_ptr() as *mut f32,
                self.population_size as u32,
                self.genome_size as u32,
                std::ptr::null_mut(), // stream
            );
        }

        // Copy result to host
        let mut diversity = vec![0f32; 1];
        self.device
            .dtoh_sync_copy_into(&diversity_device, &mut diversity)
            .unwrap_or_default();
        diversity[0] as f64
    }

    /// Create offspring from selected parents
    pub fn create_offspring(
        &mut self,
        selected_indices: &CudaSlice<u32>,
        crossover_rate: f32,
    ) -> Result<()> {
        if let Some(rng_states) = self.rng_states {
            // For now, skip crossover and just use selected parents directly
            // TODO: Implement proper crossover with parent pairing

            // Invalidate fitness for new offspring
            self.invalidate_fitness();

            Ok(())
        } else {
            Err(anyhow::anyhow!("RNG states not initialized"))
        }
    }

    /// Get GPU pointers for kernel access
    pub fn gpu_pointers(&self) -> PopulationPointers {
        PopulationPointers {
            genomes: *self.genomes.device_ptr() as *mut f32,
            fitness_scores: *self.fitness_scores.device_ptr() as *mut f32,
            fitness_valid: *self.fitness_valid.device_ptr() as *mut u8,
            population_size: self.population_size,
            genome_size: self.genome_size,
        }
    }

    /// Mark fitness as valid (called after evaluation)
    pub fn mark_fitness_valid(&mut self) {
        // Set all fitness to valid (1)
        let ones = vec![1u8; self.population_size];
        self.device
            .htod_copy_into(ones, &mut self.fitness_valid.clone())
            .ok();
    }
}

/// GPU pointers for kernel access
pub struct PopulationPointers {
    pub genomes: *mut f32,
    pub fitness_scores: *mut f32,
    pub fitness_valid: *mut u8,
    pub population_size: usize,
    pub genome_size: usize,
}

impl Drop for GpuPopulation {
    fn drop(&mut self) {
        // Cleanup RNG states
        if let Some(rng_states) = self.rng_states {
            unsafe {
                crate::evolution::kernels::cleanup_rng_states(rng_states);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_population_creation() {
        // This test requires a GPU
        if let Ok(device) = CudaDevice::new(0) {
            let device = Arc::new(device);
            let pop = GpuPopulation::new(device, 1024, 256)?;
            assert_eq!(pop.population_size, 1024);
            assert_eq!(pop.genome_size, 256);
        }
    }
}
