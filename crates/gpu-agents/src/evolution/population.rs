//! GPU-optimized population representation

use anyhow::Result;
use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
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
    device: Arc<CudaContext>,
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
        device: Arc<CudaContext>,
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
        let stream = device.default_stream();
        // SAFETY: alloc returns uninitialized memory. genomes will be initialized
        // via initialize_random() or htod_copy before any kernel reads.
        let genomes = unsafe { stream.alloc::<f32>(total_genome_size)? };
        // SAFETY: alloc returns uninitialized memory. fitness_scores will be written
        // by fitness evaluation kernels before any reads.
        let fitness_scores = unsafe { stream.alloc::<f32>(population_size)? };
        let fitness_valid = stream.alloc_zeros::<u8>(population_size)?;

        // Initialize RNG states for CUDA kernels
        // SAFETY: setup_rng_states allocates and initializes curandState for each thread.
        // Returns null on failure which we check below. population_size is a valid count.
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
            // Launch kernel to generate random genomes in a block so guard is dropped
            // before we borrow self mutably for invalidate_fitness
            // SAFETY: genomes pointer is valid from CudaSlice allocation.
            // rng_states was initialized in new() and is non-null (checked above).
            // population_size and genome_size match allocation sizes.
            {
                let stream = self.device.default_stream();
                let (genomes_ptr, _guard) = self.genomes.device_ptr_mut(&stream);
                unsafe {
                    crate::evolution::kernels::launch_random_init(
                        genomes_ptr as *mut f32,
                        self.population_size as u32,
                        self.genome_size as u32,
                        rng_states,
                        std::ptr::null_mut(), // stream
                    );
                }
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
        let first_element_slice = self.fitness_valid.slice(0..1);
        let stream = self.device.default_stream();
        let has_fitness: Vec<u8> = stream
            .clone_dtoh(&first_element_slice)
            .unwrap_or_else(|_| vec![0u8]);
        !has_fitness.is_empty() && has_fitness[0] != 0
    }

    /// Invalidate all fitness scores
    pub fn invalidate_fitness(&mut self) {
        // Set all fitness_valid flags to 0
        let zeros = vec![0u8; self.population_size];
        let stream = self.device.default_stream();
        stream
            .memcpy_htod(&zeros, &mut self.fitness_valid)
            .ok();
        self.best_individual_cache = None;
    }

    /// Get best individual (highest fitness)
    pub fn best_individual(&self) -> Result<GpuIndividual> {
        if !self.has_fitness() {
            return Err(anyhow::anyhow!("Fitness not evaluated"));
        }

        // Allocate device memory for results
        let stream = self.device.default_stream();
        let mut best_idx_device = stream.alloc_zeros::<u32>(1)?;
        let mut best_value_device = stream.alloc_zeros::<f32>(1)?;

        // Find index of best fitness on GPU in a block so guards are dropped before clone_dtoh
        // SAFETY: All pointers are valid device pointers from CudaSlice allocations.
        // fitness_scores has population_size elements, best_idx/best_value have 1 element each.
        // population_size matches the actual fitness array size.
        {
            let (fitness_ptr, _g1) = self.fitness_scores.device_ptr(&stream);
            let (best_idx_ptr, _g2) = best_idx_device.device_ptr_mut(&stream);
            let (best_value_ptr, _g3) = best_value_device.device_ptr_mut(&stream);
            unsafe {
                crate::evolution::kernels::find_best_fitness(
                    fitness_ptr as *const f32,
                    best_idx_ptr as *mut u32,
                    best_value_ptr as *mut f32,
                    self.population_size as u32,
                    std::ptr::null_mut(), // stream
                );
            }
        }

        // Copy results to host
        let best_idx: Vec<u32> = stream.clone_dtoh(&best_idx_device)?;
        let best_value: Vec<f32> = stream.clone_dtoh(&best_value_device)?;

        let best_idx = best_idx[0] as usize;

        // Copy individual data from GPU
        let genome_start = best_idx * self.genome_size;
        let genome_end = genome_start + self.genome_size;

        let genome_slice = self.genomes.slice(genome_start..genome_end);
        let stream = self.device.default_stream();
        let genome: Vec<f32> = stream.clone_dtoh(&genome_slice)?;

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
        let stream = self.device.default_stream();
        // SAFETY (in unwrap_or_else): alloc returns uninitialized memory for single f32.
        // The kernel writes the result before we read.
        let mut average_device = stream
            .alloc_zeros::<f32>(1)
            .unwrap_or_else(|_| unsafe { stream.alloc::<f32>(1).unwrap() });

        // Launch kernel to compute average in a block so guards are dropped before clone_dtoh
        // SAFETY: All pointers are valid device pointers from CudaSlice allocations.
        // fitness_scores has population_size elements, average_device has 1 element.
        {
            let (fitness_ptr, _g1) = self.fitness_scores.device_ptr(&stream);
            let (avg_ptr, _g2) = average_device.device_ptr_mut(&stream);
            unsafe {
                crate::evolution::kernels::compute_average_fitness(
                    fitness_ptr as *const f32,
                    avg_ptr as *mut f32,
                    self.population_size as u32,
                    std::ptr::null_mut(), // stream
                );
            }
        }

        // Copy result to host
        let average: Vec<f32> = stream
            .clone_dtoh(&average_device)
            .unwrap_or_else(|_| vec![0f32]);
        if average.is_empty() { 0.0 } else { average[0] as f64 }
    }

    /// Calculate diversity index
    pub fn diversity_index(&self) -> f64 {
        // Allocate device memory for result
        let stream = self.device.default_stream();
        // SAFETY (in unwrap_or_else): alloc returns uninitialized memory for single f32.
        // The kernel writes the result before we read.
        let mut diversity_device = stream
            .alloc_zeros::<f32>(1)
            .unwrap_or_else(|_| unsafe { stream.alloc::<f32>(1).unwrap() });

        // Launch kernel to compute diversity in a block so guards are dropped before clone_dtoh
        // SAFETY: All pointers are valid device pointers from CudaSlice allocations.
        // genomes has population_size * genome_size elements. diversity_device has 1 element.
        {
            let (genomes_ptr, _g1) = self.genomes.device_ptr(&stream);
            let (div_ptr, _g2) = diversity_device.device_ptr_mut(&stream);
            unsafe {
                crate::evolution::kernels::compute_diversity(
                    genomes_ptr as *const f32,
                    div_ptr as *mut f32,
                    self.population_size as u32,
                    self.genome_size as u32,
                    std::ptr::null_mut(), // stream
                );
            }
        }

        // Copy result to host
        let diversity: Vec<f32> = stream
            .clone_dtoh(&diversity_device)
            .unwrap_or_else(|_| vec![0f32]);
        if diversity.is_empty() { 0.0 } else { diversity[0] as f64 }
    }

    /// Create offspring from selected parents using crossover
    ///
    /// Parents are paired sequentially from selected_indices:
    /// - Parent 1 and 2 produce offspring 1 and 2
    /// - Parent 3 and 4 produce offspring 3 and 4
    /// - etc.
    ///
    /// Crossover uses uniform crossover where each gene has crossover_rate
    /// probability of coming from parent 2 instead of parent 1.
    #[allow(unused_variables)]
    pub fn create_offspring(
        &mut self,
        selected_indices: &CudaSlice<u32>,
        crossover_rate: f32,
    ) -> Result<()> {
        if let Some(rng_states) = self.rng_states {
            // Allocate temporary buffer for offspring genomes
            let total_genome_size = self.population_size * self.genome_size;
            let stream = self.device.default_stream();
            // SAFETY: alloc returns uninitialized memory. The crossover kernel will
            // write to all elements before we copy back.
            let mut offspring_genomes = unsafe { stream.alloc::<f32>(total_genome_size)? };

            // Launch crossover kernel in a block so guards are dropped before mutable borrow
            // SAFETY: All pointers are valid:
            // - genomes: from CudaSlice allocation in new()
            // - offspring_genomes: just allocated above
            // - selected_indices: passed in as CudaSlice
            // - rng_states: initialized and non-null (checked above)
            // population_size and genome_size match allocation sizes
            {
                let (genomes_ptr, _g1) = self.genomes.device_ptr(&stream);
                let (offspring_ptr, _g2) = offspring_genomes.device_ptr_mut(&stream);
                let (indices_ptr, _g3) = selected_indices.device_ptr(&stream);
                unsafe {
                    crate::evolution::kernels::launch_crossover(
                        genomes_ptr as *const f32,
                        offspring_ptr as *mut f32,
                        indices_ptr as *const u32,
                        self.population_size as u32,
                        self.genome_size as u32,
                        rng_states,
                        std::ptr::null_mut(), // stream
                    );
                }
            }

            // Copy offspring back to main genome buffer
            // SAFETY: Both slices are valid CudaSlice<f32> of the same size
            stream.memcpy_dtod(&offspring_genomes, &mut self.genomes)?;

            // Invalidate fitness for new offspring
            self.invalidate_fitness();
            self.best_individual_cache = None;

            Ok(())
        } else {
            Err(anyhow::anyhow!("RNG states not initialized"))
        }
    }

    /// Get RNG states pointer for external use (e.g., mutation)
    pub fn rng_states(&self) -> Option<*mut u8> {
        self.rng_states
    }

    /// Get GPU pointers for kernel access
    ///
    /// NOTE: The returned pointers should only be used immediately for kernel launches.
    /// Callers must ensure proper stream synchronization after kernel execution.
    pub fn gpu_pointers(&self) -> PopulationPointers {
        let stream = self.device.default_stream();
        // Get device pointers - guards are dropped at end of this function,
        // but pointers remain valid for immediate kernel use
        let (genomes_ptr, _g1) = self.genomes.device_ptr(&stream);
        let (fitness_ptr, _g2) = self.fitness_scores.device_ptr(&stream);
        let (valid_ptr, _g3) = self.fitness_valid.device_ptr(&stream);
        PopulationPointers {
            genomes: genomes_ptr as *mut f32,
            fitness_scores: fitness_ptr as *mut f32,
            fitness_valid: valid_ptr as *mut u8,
            population_size: self.population_size,
            genome_size: self.genome_size,
        }
    }

    /// Mark fitness as valid (called after evaluation)
    pub fn mark_fitness_valid(&mut self) {
        // Set all fitness to valid (1)
        let ones = vec![1u8; self.population_size];
        let stream = self.device.default_stream();
        stream
            .memcpy_htod(&ones, &mut self.fitness_valid)
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
            // SAFETY: rng_states was allocated by setup_rng_states in new() and is non-null.
            // This frees the CUDA memory for the curandState array.
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
    fn test_gpu_population_creation() -> Result<(), Box<dyn std::error::Error>> {
        // This test requires a GPU
        if let Ok(device) = CudaContext::new(0) {
            let pop = GpuPopulation::new(device, 1024, 256)?;
            assert_eq!(pop.population_size, 1024);
            assert_eq!(pop.genome_size, 256);
        }
        Ok(())
    }
}
