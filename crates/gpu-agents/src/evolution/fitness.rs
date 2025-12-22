//! GPU fitness evaluation for evolution

use anyhow::Result;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use std::sync::Arc;

use super::population::GpuPopulation;

/// Fitness evaluation types
#[derive(Debug, Clone, Copy)]
pub enum FitnessType {
    /// Maximize sum of genome values
    SumFitness = 0,
    /// Pattern matching fitness
    PatternMatching = 1,
    /// Multi-objective fitness
    MultiObjective = 2,
    /// Custom fitness function
    Custom = 3,
}

/// GPU Fitness Evaluator
pub struct GpuFitnessEvaluator {
    device: Arc<CudaDevice>,
    fitness_objectives: usize,
    fitness_type: FitnessType,
    /// Multi-objective fitness vectors (if applicable)
    fitness_vectors: Option<CudaSlice<f32>>,
}

impl GpuFitnessEvaluator {
    /// Create new fitness evaluator
    pub fn new(device: Arc<CudaDevice>, fitness_objectives: usize) -> Result<Self> {
        let fitness_type = if fitness_objectives > 1 {
            FitnessType::MultiObjective
        } else {
            FitnessType::SumFitness
        };

        Ok(Self {
            device,
            fitness_objectives,
            fitness_type,
            fitness_vectors: None,
        })
    }

    /// Set fitness type
    pub fn set_fitness_type(&mut self, fitness_type: FitnessType) {
        self.fitness_type = fitness_type;
    }

    /// Evaluate fitness for entire population
    pub fn evaluate_population(&mut self, population: &mut GpuPopulation) -> Result<()> {
        let pointers = population.gpu_pointers();

        match self.fitness_type {
            FitnessType::MultiObjective => {
                self.evaluate_multi_objective(population)?;
            }
            _ => {
                // Single objective evaluation
                unsafe {
                    crate::evolution::kernels::launch_fitness_evaluation(
                        pointers.genomes,
                        pointers.fitness_scores,
                        pointers.population_size as u32,
                        pointers.genome_size as u32,
                        std::ptr::null_mut(), // stream
                    );
                }
            }
        }

        // Mark fitness as valid
        population.mark_fitness_valid();

        Ok(())
    }

    /// Evaluate multi-objective fitness
    fn evaluate_multi_objective(&mut self, population: &mut GpuPopulation) -> Result<()> {
        let pointers = population.gpu_pointers();

        // Allocate fitness vectors if not already done
        if self.fitness_vectors.is_none() {
            let vector_size = pointers.population_size * self.fitness_objectives;
            self.fitness_vectors = Some(unsafe { self.device.alloc::<f32>(vector_size)? });
        }

        let fitness_vectors = self.fitness_vectors.as_ref()?;

        // Launch multi-objective evaluation kernel
        unsafe {
            crate::ffi::launch_multi_objective_fitness(
                std::ptr::null_mut(), // We're not using GPUAgent struct here
                *fitness_vectors.device_ptr() as *mut f32,
                pointers.population_size as u32,
                self.fitness_objectives as u32,
                0.1,   // kernel_time_ms (example)
                100.0, // memory_usage_mb (example)
            );
        }

        // Compute scalar fitness from multi-objective results
        self.aggregate_multi_objective_fitness(population, fitness_vectors)?;

        Ok(())
    }

    /// Aggregate multi-objective fitness into scalar fitness
    fn aggregate_multi_objective_fitness(
        &self,
        population: &mut GpuPopulation,
        fitness_vectors: &CudaSlice<f32>,
    ) -> Result<()> {
        let pointers = population.gpu_pointers();

        // For now, use simple weighted sum
        // In the future, could implement NSGA-II or other multi-objective algorithms
        unsafe {
            aggregate_fitness_kernel(
                *fitness_vectors.device_ptr() as *const f32,
                pointers.fitness_scores,
                pointers.population_size as u32,
                self.fitness_objectives as u32,
            );
        }

        Ok(())
    }

    /// Set custom fitness function
    pub fn set_custom_fitness_fn<F>(&mut self, _fitness_fn: F)
    where
        F: Fn(&[u8]) -> f32 + Send + Sync + 'static,
    {
        // For custom fitness functions, we would need to:
        // 1. Compile the function to PTX
        // 2. Load it as a CUDA kernel
        // 3. Call it during evaluation
        // This is a placeholder for future implementation
        self.fitness_type = FitnessType::Custom;
    }
}

// Helper kernel for aggregating multi-objective fitness
extern "C" {
    fn aggregate_fitness_kernel(
        fitness_vectors: *const f32,
        fitness_scores: *mut f32,
        population_size: u32,
        num_objectives: u32,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fitness_evaluator_creation() -> Result<(), Box<dyn std::error::Error>>  {
        if let Ok(device) = CudaDevice::new(0) {
            let device = Arc::new(device);
            let evaluator = GpuFitnessEvaluator::new(device, 1)?;
            assert_eq!(evaluator.fitness_objectives, 1);
        }
    }
}
