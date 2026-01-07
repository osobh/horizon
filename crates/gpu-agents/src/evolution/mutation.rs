//! GPU mutation operations for evolution

use anyhow::Result;
use cudarc::driver::CudaDevice;
use std::sync::Arc;
use std::time::Instant;

use super::population::{GpuPopulation, PopulationPointers};

/// Mutation strategies
#[derive(Debug, Clone, Copy)]
pub enum MutationStrategy {
    /// Uniform random mutation
    Uniform,
    /// Gaussian mutation
    Gaussian,
    /// Adaptive mutation based on diversity
    Adaptive,
    /// Bit flip mutation
    BitFlip,
}

/// GPU Mutation Engine
pub struct GpuMutationEngine {
    _device: Arc<CudaDevice>,
    mutation_rate: f32,
    mutation_strategy: MutationStrategy,
    /// Tracking mutations per second
    mutations_count: u64,
    last_measurement: Instant,
    mutations_per_second: f64,
}

impl GpuMutationEngine {
    /// Create new mutation engine
    pub fn new(device: Arc<CudaDevice>, mutation_rate: f32) -> Result<Self> {
        Ok(Self {
            _device: device,
            mutation_rate,
            mutation_strategy: MutationStrategy::Uniform,
            mutations_count: 0,
            last_measurement: Instant::now(),
            mutations_per_second: 0.0,
        })
    }

    /// Set mutation strategy
    pub fn set_strategy(&mut self, strategy: MutationStrategy) {
        self.mutation_strategy = strategy;
    }

    /// Set mutation rate
    pub fn set_mutation_rate(&mut self, rate: f32) {
        self.mutation_rate = rate.clamp(0.0, 1.0);
    }

    /// Mutate entire population
    pub fn mutate_population(&mut self, population: &mut GpuPopulation) -> Result<()> {
        let pointers = population.gpu_pointers();
        let start = Instant::now();

        // Get RNG states from population
        let rng_states = population.rng_states()
            .ok_or_else(|| anyhow::anyhow!("RNG states not initialized in population"))?;

        match self.mutation_strategy {
            MutationStrategy::Adaptive => {
                // Calculate diversity for adaptive mutation
                let diversity = population.diversity_index();
                self.adaptive_mutation(&pointers, diversity as f32)?;
            }
            MutationStrategy::Uniform => {
                // Launch uniform mutation kernel
                // SAFETY: pointers.genomes is valid from GpuPopulation allocation.
                // rng_states is non-null (checked above). population_size and genome_size
                // match allocation sizes. mutation_rate is a valid f32.
                unsafe {
                    crate::evolution::kernels::launch_uniform_mutation(
                        pointers.genomes,
                        rng_states,
                        pointers.population_size as u32,
                        pointers.genome_size as u32,
                        self.mutation_rate,
                        std::ptr::null_mut(), // stream
                    );
                }
            }
            MutationStrategy::Gaussian => {
                // Launch Gaussian mutation kernel
                // SAFETY: Same as Uniform - all pointers are valid and sizes match allocations.
                unsafe {
                    crate::evolution::kernels::launch_gaussian_mutation(
                        pointers.genomes,
                        rng_states,
                        pointers.population_size as u32,
                        pointers.genome_size as u32,
                        self.mutation_rate,
                        0.1, // sigma - standard deviation
                        std::ptr::null_mut(), // stream
                    );
                }
            }
            MutationStrategy::BitFlip => {
                // Launch bit flip mutation kernel
                // This treats the genome as bits and flips each with mutation_rate probability
                // SAFETY: Same as above - all pointers valid, sizes match allocations.
                unsafe {
                    crate::evolution::kernels::launch_bitflip_mutation(
                        pointers.genomes as *mut u8,
                        rng_states,
                        pointers.population_size as u32,
                        (pointers.genome_size * std::mem::size_of::<f32>()) as u32, // bytes
                        self.mutation_rate,
                        std::ptr::null_mut(), // stream
                    );
                }
            }
        }

        // Update mutation statistics
        self.update_statistics(pointers.population_size, start);

        // Invalidate fitness after mutation
        population.invalidate_fitness();

        Ok(())
    }

    /// Perform adaptive mutation based on population diversity
    fn adaptive_mutation(&mut self, pointers: &PopulationPointers, diversity: f32) -> Result<()> {
        // Use existing adaptive mutation kernel from evolution_kernel.cu
        // SAFETY: pointers.genomes is a valid device pointer from GpuPopulation allocation.
        // population_size matches the actual genome array size. RNG states are null
        // because the kernel manages its own internal RNG. mutation_rate and diversity
        // are valid f32 values.
        unsafe {
            crate::ffi::launch_adaptive_mutation(
                pointers.genomes as *mut f32, // Assuming float genomes for adaptive mutation
                std::ptr::null_mut(),         // RNG states managed internally
                pointers.population_size as u32,
                self.mutation_rate,
                diversity,
            );
        }
        Ok(())
    }

    /// Mutate specific individuals
    pub fn mutate_individuals(
        &mut self,
        population: &mut GpuPopulation,
        _individual_indices: &[u32],
    ) -> Result<()> {
        // For targeted mutation, we would need a custom kernel
        // For now, use full population mutation as placeholder
        self.mutate_population(population)
    }

    /// Update mutation statistics
    fn update_statistics(&mut self, population_size: usize, _start: Instant) {
        self.mutations_count += population_size as u64;

        let elapsed = self.last_measurement.elapsed().as_secs_f64();
        if elapsed > 1.0 {
            self.mutations_per_second = self.mutations_count as f64 / elapsed;
            self.mutations_count = 0;
            self.last_measurement = Instant::now();
        }
    }

    /// Get mutations per second
    pub fn mutations_per_second(&self) -> f64 {
        self.mutations_per_second
    }

    /// Create custom mutation kernel
    pub fn create_custom_mutation<F>(&mut self, _mutation_fn: F) -> Result<()>
    where
        F: Fn(&mut [u8], f32) + Send + Sync + 'static,
    {
        // Placeholder for custom mutation kernel compilation
        // Would involve:
        // 1. Converting Rust closure to CUDA kernel
        // 2. Compiling to PTX
        // 3. Loading and caching the kernel
        Ok(())
    }
}

/// Mutation parameters for different strategies
#[derive(Debug, Clone)]
pub struct MutationParams {
    /// Base mutation rate
    pub rate: f32,
    /// Mutation strength (for Gaussian)
    pub strength: f32,
    /// Minimum diversity threshold
    pub min_diversity: f32,
    /// Maximum diversity threshold
    pub max_diversity: f32,
}

impl Default for MutationParams {
    fn default() -> Self {
        Self {
            rate: 0.01,
            strength: 0.1,
            min_diversity: 0.1,
            max_diversity: 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutation_engine_creation() -> Result<(), Box<dyn std::error::Error>> {
        if let Ok(device) = CudaDevice::new(0) {
            let device = Arc::new(device);
            let engine = GpuMutationEngine::new(device, 0.01)?;
            assert_eq!(engine.mutation_rate, 0.01);
        }
    }

    #[test]
    fn test_mutation_rate_clamping() -> Result<(), Box<dyn std::error::Error>> {
        if let Ok(device) = CudaDevice::new(0) {
            let device = Arc::new(device);
            let mut engine = GpuMutationEngine::new(device, 0.5)?;

            engine.set_mutation_rate(1.5);
            assert_eq!(engine.mutation_rate, 1.0);

            engine.set_mutation_rate(-0.5);
            assert_eq!(engine.mutation_rate, 0.0);
        }
    }
}
