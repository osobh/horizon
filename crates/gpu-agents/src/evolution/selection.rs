//! GPU selection strategies for evolution

use anyhow::Result;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DeviceSlice};
use std::sync::Arc;

use super::population::{GpuPopulation, PopulationPointers};

/// Selection strategies
#[derive(Debug, Clone, Copy)]
pub enum SelectionMethod {
    /// Tournament selection
    Tournament { size: u32 },
    /// Roulette wheel selection
    RouletteWheel,
    /// Rank-based selection
    RankBased,
    /// Elite selection
    Elite,
    /// NSGA-II for multi-objective
    NSGA2,
}

/// GPU Selection Strategy
pub struct GpuSelectionStrategy {
    device: Arc<CudaDevice>,
    method: SelectionMethod,
    elite_percentage: f32,
    /// Buffer for selected indices
    selected_indices: Option<CudaSlice<u32>>,
}

impl GpuSelectionStrategy {
    /// Create new selection strategy
    pub fn new(device: Arc<CudaDevice>, elite_percentage: f32) -> Result<Self> {
        Ok(Self {
            device,
            method: SelectionMethod::Tournament { size: 3 },
            elite_percentage: elite_percentage.clamp(0.0, 1.0),
            selected_indices: None,
        })
    }

    /// Set selection method
    pub fn set_method(&mut self, method: SelectionMethod) {
        self.method = method;
    }

    /// Select individuals for next generation
    pub fn select(
        &mut self,
        population: &GpuPopulation,
        selection_size: usize,
    ) -> Result<CudaSlice<u32>> {
        // Allocate selection buffer if needed
        if self.selected_indices.is_none()
            || self.selected_indices.as_ref()?.len() < selection_size
        {
            self.selected_indices = Some(unsafe { self.device.alloc::<u32>(selection_size)? });
        }

        let pointers = population.gpu_pointers();

        // Calculate elite count
        let elite_count = (pointers.population_size as f32 * self.elite_percentage) as u32;

        // Get selection parameters before borrowing
        let method = self.method.clone();

        // Now we can borrow selected_indices mutably
        let selected = self.selected_indices.as_mut()?;

        match method {
            SelectionMethod::Tournament { size } => {
                Self::tournament_selection_kernel(
                    &self.device,
                    &pointers,
                    selected,
                    selection_size,
                    size,
                )?;
            }
            SelectionMethod::Elite => {
                Self::elite_selection_kernel(&self.device, &pointers, selected, elite_count)?;
            }
            SelectionMethod::NSGA2 => {
                Self::nsga2_selection_kernel(&self.device, &pointers, selected, selection_size)?;
            }
            _ => {
                // Fallback to tournament selection
                Self::tournament_selection_kernel(
                    &self.device,
                    &pointers,
                    selected,
                    selection_size,
                    3,
                )?;
            }
        }

        Ok(selected.clone())
    }

    /// Tournament selection kernel
    fn tournament_selection_kernel(
        _device: &Arc<CudaDevice>,
        pointers: &PopulationPointers,
        selected: &mut CudaSlice<u32>,
        selection_size: usize,
        tournament_size: u32,
    ) -> Result<()> {
        unsafe {
            crate::evolution::kernels::launch_tournament_selection(
                pointers.fitness_scores,
                *selected.device_ptr() as *mut u32,
                pointers.population_size as u32,
                selection_size as u32,
                tournament_size,
                std::ptr::null_mut(), // rng_states - TODO: proper RNG state management
                std::ptr::null_mut(), // stream
            );
        }
        Ok(())
    }

    /// Elite selection kernel - preserve best individuals
    fn elite_selection_kernel(
        _device: &Arc<CudaDevice>,
        pointers: &PopulationPointers,
        selected: &mut CudaSlice<u32>,
        elite_count: u32,
    ) -> Result<()> {
        unsafe {
            crate::evolution::kernels::launch_elite_preservation(
                pointers.fitness_scores,
                pointers.fitness_valid,
                *selected.device_ptr() as *mut u32,
                pointers.population_size as u32,
                elite_count,
            );
        }
        Ok(())
    }

    /// NSGA-II selection kernel for multi-objective optimization
    fn nsga2_selection_kernel(
        _device: &Arc<CudaDevice>,
        pointers: &PopulationPointers,
        selected: &mut CudaSlice<u32>,
        selection_size: usize,
    ) -> Result<()> {
        // Use NSGA-II kernel from evolution_kernel.cu
        unsafe {
            crate::ffi::launch_nsga2_selection(
                std::ptr::null_mut(),                // GPUAgent pointer (not used here)
                pointers.fitness_scores as *mut f32, // Treat as fitness vectors
                *selected.device_ptr() as *mut u32,
                pointers.population_size as u32,
                1, // num_objectives (simplified for now)
                selection_size as u32,
            );
        }
        Ok(())
    }

    /// Combine multiple selection methods
    pub fn hybrid_selection(
        &mut self,
        population: &GpuPopulation,
        selection_size: usize,
        methods: &[(SelectionMethod, f32)],
    ) -> Result<CudaSlice<u32>> {
        // Allocate combined selection buffer
        if self.selected_indices.is_none()
            || self.selected_indices.as_ref()?.len() < selection_size
        {
            self.selected_indices = Some(unsafe { self.device.alloc::<u32>(selection_size)? });
        }

        let mut offset = 0;

        // Collect partial selections first
        let mut partial_selections = Vec::new();
        for (method, proportion) in methods {
            let count = (selection_size as f32 * proportion) as usize;
            if offset + count > selection_size {
                break;
            }

            self.method = *method;
            let partial_selection = self.select(population, count)?;
            partial_selections.push((offset, partial_selection));
            offset += count;
        }

        // Now copy all partial selections to the combined buffer
        let selected = self.selected_indices.as_mut()?;
        for (offset, partial_selection) in partial_selections {
            let count = partial_selection.len();
            self.device.dtod_copy(
                &partial_selection,
                &mut selected.slice_mut(offset..offset + count),
            )?;
        }

        Ok(selected.clone())
    }
}

/// Selection parameters
#[derive(Debug, Clone)]
pub struct SelectionParams {
    /// Tournament size
    pub tournament_size: u32,
    /// Elite percentage
    pub elite_percentage: f32,
    /// Selection pressure
    pub pressure: f32,
}

impl Default for SelectionParams {
    fn default() -> Self {
        Self {
            tournament_size: 3,
            elite_percentage: 0.1,
            pressure: 2.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selection_strategy_creation() -> Result<(), Box<dyn std::error::Error>>  {
        if let Ok(device) = CudaDevice::new(0) {
            let device = Arc::new(device);
            let strategy = GpuSelectionStrategy::new(device, 0.1)?;
            assert_eq!(strategy.elite_percentage, 0.1);
        }
    }

    #[test]
    fn test_elite_percentage_clamping() -> Result<(), Box<dyn std::error::Error>>  {
        if let Ok(device) = CudaDevice::new(0) {
            let device = Arc::new(device);
            let strategy = GpuSelectionStrategy::new(device.clone(), 1.5)?;
            assert_eq!(strategy.elite_percentage, 1.0);

            let strategy = GpuSelectionStrategy::new(device, -0.5)?;
            assert_eq!(strategy.elite_percentage, 0.0);
        }
    }
}
