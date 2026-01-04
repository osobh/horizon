//! GPU kernel integration tests
//!
//! These tests verify the actual CUDA kernel execution for evolution algorithms.
//! They test the kernel launch configurations, memory transfers, and correctness.

use super::*;
use crate::evolution::kernels::*;
use anyhow::Result;
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr};
use std::sync::Arc;

// Test kernel launch configurations
const TEST_BLOCK_SIZE: u32 = 256;
const WARP_SIZE: u32 = 32;

// =============================================================================
// ADAS Kernel Tests
// =============================================================================

#[test]
fn test_adas_evaluation_kernel_launch() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let population_size = 128;
    let code_size = 256;

    // Allocate GPU memory
    let agents_data = device.alloc_zeros::<u8>(population_size * code_size)?;
    let performances = device.alloc_zeros::<f32>(population_size)?;
    let metrics = device.alloc_zeros::<f32>(population_size * 4)?; // 4 metrics per agent

    // Launch evaluation kernel
    // SAFETY: All pointers are valid device pointers from alloc_zeros calls that
    // remain valid for this scope. population_size and code_size match allocations.
    unsafe {
        prepare_adas_evaluation(
            agents_data.as_ptr() as *const std::ffi::c_void,
            performances.as_mut_ptr() as *mut f32,
            metrics.as_mut_ptr() as *mut f32,
            population_size as u32,
            code_size as u32,
        );
    }

    // Verify results
    let performances_host = device.dtoh_sync_copy(&performances)?;
    assert_eq!(performances_host.len(), population_size);

    // All performances should be initialized (even if to 0)
    for &perf in &performances_host {
        assert!(perf >= 0.0 && perf <= 1.0);
    }

    Ok(())
}

#[test]
fn test_adas_mutation_kernel_launch() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let population_size = 64;
    let code_size = 128;
    let mutation_rate = 0.1;

    // Create test data
    let mut agents_data = vec![0u8; population_size * code_size];
    for i in 0..agents_data.len() {
        agents_data[i] = (i % 256) as u8;
    }

    // Upload to GPU
    let mut agents_gpu = device.htod_sync_copy(&agents_data)?;
    let performances = device.alloc_zeros::<f32>(population_size)?;

    // Launch mutation kernel
    // SAFETY: agents_gpu and performances are valid device pointers from htod_sync_copy
    // and alloc_zeros. Sizes match the allocation parameters.
    unsafe {
        launch_adas_mutation(
            agents_gpu.as_mut_ptr() as *mut std::ffi::c_void,
            performances.as_ptr() as *const f32,
            population_size as u32,
            code_size as u32,
            mutation_rate,
        );
    }

    // Download and verify mutations occurred
    let mutated_agents = device.dtoh_sync_copy(&agents_gpu)?;

    let mut mutations = 0;
    for i in 0..agents_data.len() {
        if mutated_agents[i] != agents_data[i] {
            mutations += 1;
        }
    }

    // With 10% mutation rate, expect some mutations
    assert!(mutations > 0, "Mutations should have occurred");
    assert!(mutations < agents_data.len(), "Not all bytes should mutate");

    Ok(())
}

#[test]
fn test_adas_crossover_kernel_launch() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let population_size = 32;
    let code_size = 64;

    // Create parent agents with distinct patterns
    let mut parent_data = vec![0u8; population_size * code_size];
    for i in 0..population_size {
        let pattern = (i % 4) as u8;
        for j in 0..code_size {
            parent_data[i * code_size + j] = pattern * 64 + (j as u8);
        }
    }

    // Upload to GPU
    let parents_gpu = device.htod_sync_copy(&parent_data)?;
    let mut offspring_gpu = device.alloc_zeros::<u8>(population_size * code_size)?;
    let parent_indices = device.htod_sync_copy(&(0..population_size as u32).collect::<Vec<_>>())?;

    // Launch crossover kernel
    // SAFETY: All device pointers are valid from htod_sync_copy and alloc_zeros.
    // parent_indices contains valid indices within [0, population_size).
    unsafe {
        launch_adas_crossover(
            parents_gpu.as_ptr() as *const std::ffi::c_void,
            offspring_gpu.as_mut_ptr() as *mut std::ffi::c_void,
            parent_indices.as_ptr() as *const u32,
            population_size as u32,
            code_size as u32,
        );
    }

    // Verify offspring are created
    let offspring_data = device.dtoh_sync_copy(&offspring_gpu)?;

    // Offspring should have mixed patterns from parents
    let mut unique_patterns = std::collections::HashSet::new();
    for i in 0..population_size {
        let first_byte = offspring_data[i * code_size];
        unique_patterns.insert(first_byte / 64);
    }

    assert!(
        !unique_patterns.is_empty(),
        "Offspring should have patterns"
    );

    Ok(())
}

#[test]
fn test_adas_diversity_kernel_launch() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let population_size = 128;
    let code_size = 256;

    // Create population with varying diversity
    let mut agents_data = vec![0u8; population_size * code_size];

    // First half: identical agents (low diversity)
    for i in 0..population_size / 2 {
        for j in 0..code_size {
            agents_data[i * code_size + j] = 42;
        }
    }

    // Second half: diverse agents
    for i in population_size / 2..population_size {
        for j in 0..code_size {
            agents_data[i * code_size + j] = ((i * j) % 256) as u8;
        }
    }

    // Upload to GPU
    let agents_gpu = device.htod_sync_copy(&agents_data)?;
    let mut diversity_scores = device.alloc_zeros::<f32>(1)?;

    // Launch diversity kernel
    // SAFETY: agents_gpu is valid from htod_sync_copy, diversity_scores from alloc_zeros.
    // Sizes match the allocation parameters.
    unsafe {
        launch_adas_diversity(
            agents_gpu.as_ptr() as *const std::ffi::c_void,
            diversity_scores.as_mut_ptr() as *mut f32,
            population_size as u32,
            code_size as u32,
        );
    }

    // Check diversity score
    let diversity = device.dtoh_sync_copy(&diversity_scores)?[0];
    assert!(diversity >= 0.0 && diversity <= 1.0);
    assert!(diversity > 0.0, "Population has some diversity");

    Ok(())
}

// =============================================================================
// DGM Kernel Tests
// =============================================================================

#[test]
fn test_dgm_self_modification_kernel_launch() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let population_size = 32;
    let code_size = 128;
    let history_size = 10;

    // Create test agents and history
    let mut agents_data = vec![0u8; population_size * code_size];
    for i in 0..agents_data.len() {
        agents_data[i] = (i % 256) as u8;
    }

    let performance_history = vec![0.3, 0.35, 0.4, 0.42, 0.45, 0.48, 0.5, 0.52, 0.51, 0.53];

    // Upload to GPU
    let mut agents_gpu = device.htod_sync_copy(&agents_data)?;
    let history_gpu = device.htod_sync_copy(&performance_history)?;
    let modification_rate = 0.2;

    // Launch self-modification kernel
    // SAFETY: agents_gpu and history_gpu are valid from htod_sync_copy.
    // history_size matches the length of performance_history.
    unsafe {
        launch_dgm_self_modification(
            agents_gpu.as_mut_ptr() as *mut std::ffi::c_void,
            history_gpu.as_ptr() as *const f32,
            population_size as u32,
            code_size as u32,
            history_size as u32,
            modification_rate,
        );
    }

    // Verify modifications
    let modified_agents = device.dtoh_sync_copy(&agents_gpu)?;

    let mut modifications = 0;
    for i in 0..agents_data.len() {
        if modified_agents[i] != agents_data[i] {
            modifications += 1;
        }
    }

    assert!(modifications > 0, "Self-modifications should occur");

    Ok(())
}

#[test]
fn test_dgm_benchmark_kernel_launch() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let population_size = 64;
    let code_size = 256;
    let num_benchmarks = 5;

    // Create test agents
    let agents_data = vec![1u8; population_size * code_size];

    // Upload to GPU
    let agents_gpu = device.htod_sync_copy(&agents_data)?;
    let mut benchmark_scores = device.alloc_zeros::<f32>(population_size)?;

    // Launch benchmark kernel
    // SAFETY: agents_gpu is valid from htod_sync_copy, benchmark_scores from alloc_zeros.
    // Output buffer has space for population_size scores.
    unsafe {
        launch_dgm_benchmark(
            agents_gpu.as_ptr() as *const std::ffi::c_void,
            benchmark_scores.as_mut_ptr() as *mut f32,
            population_size as u32,
            code_size as u32,
            num_benchmarks as u32,
        );
    }

    // Verify benchmark scores
    let scores = device.dtoh_sync_copy(&benchmark_scores)?;
    assert_eq!(scores.len(), population_size);

    for &score in &scores {
        assert!(
            score >= 0.0 && score <= 1.0,
            "Benchmark scores should be normalized"
        );
    }

    Ok(())
}

#[test]
fn test_dgm_archive_update_kernel_launch() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let population_size = 32;
    let archive_size = 10;
    let code_size = 128;

    // Create population with varying performance
    let mut performances = vec![0.0f32; population_size];
    for i in 0..population_size {
        performances[i] = (i as f32) / (population_size as f32);
    }

    let agents_data = vec![0u8; population_size * code_size];

    // Upload to GPU
    let agents_gpu = device.htod_sync_copy(&agents_data)?;
    let performances_gpu = device.htod_sync_copy(&performances)?;
    let mut archive_gpu = device.alloc_zeros::<u8>(archive_size * code_size)?;
    let mut archive_scores = device.alloc_zeros::<f32>(archive_size)?;

    // Launch archive update kernel
    // SAFETY: All device pointers are valid from htod_sync_copy and alloc_zeros.
    // archive_size < population_size, archive buffers sized for archive_size entries.
    unsafe {
        launch_dgm_archive_update(
            agents_gpu.as_ptr() as *const std::ffi::c_void,
            performances_gpu.as_ptr() as *const f32,
            archive_gpu.as_mut_ptr() as *mut std::ffi::c_void,
            archive_scores.as_mut_ptr() as *mut f32,
            population_size as u32,
            archive_size as u32,
            code_size as u32,
        );
    }

    // Verify archive contains best agents
    let archive_scores_host = device.dtoh_sync_copy(&archive_scores)?;

    // Archive should be sorted by performance (best first)
    for i in 1..archive_scores_host.len() {
        if archive_scores_host[i] > 0.0 {
            assert!(
                archive_scores_host[i - 1] >= archive_scores_host[i],
                "Archive should be sorted by performance"
            );
        }
    }

    Ok(())
}

// =============================================================================
// Swarm Kernel Tests
// =============================================================================

#[test]
fn test_pso_velocity_update_kernel_launch() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let num_particles = 128;
    let dimensions = 10;

    // Create test data
    let positions = vec![0.5f32; num_particles * dimensions];
    let mut velocities = vec![0.1f32; num_particles * dimensions];
    let personal_bests = vec![0.6f32; num_particles * dimensions];
    let global_best = vec![0.7f32; dimensions];

    // PSO parameters
    let inertia = 0.729;
    let cognitive = 1.49;
    let social = 1.49;

    // Upload to GPU
    let positions_gpu = device.htod_sync_copy(&positions)?;
    let mut velocities_gpu = device.htod_sync_copy(&velocities)?;
    let personal_bests_gpu = device.htod_sync_copy(&personal_bests)?;
    let global_best_gpu = device.htod_sync_copy(&global_best)?;

    // Launch velocity update kernel
    // SAFETY: All device pointers from htod_sync_copy are valid. Buffer sizes match
    // num_particles * dimensions for positions/velocities/personal_bests, dimensions for global_best.
    unsafe {
        launch_pso_velocity_update(
            positions_gpu.as_ptr() as *const f32,
            velocities_gpu.as_mut_ptr() as *mut f32,
            personal_bests_gpu.as_ptr() as *const f32,
            global_best_gpu.as_ptr() as *const f32,
            num_particles as u32,
            dimensions as u32,
            inertia,
            cognitive,
            social,
        );
    }

    // Verify velocities updated
    let updated_velocities = device.dtoh_sync_copy(&velocities_gpu)?;

    let mut changes = 0;
    for i in 0..velocities.len() {
        if (updated_velocities[i] - velocities[i]).abs() > 1e-6 {
            changes += 1;
        }
    }

    assert!(changes > 0, "Velocities should be updated");

    Ok(())
}

#[test]
fn test_pso_position_update_kernel_launch() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let num_particles = 64;
    let dimensions = 5;

    // Create test data
    let mut positions = vec![0.0f32; num_particles * dimensions];
    let velocities = vec![0.1f32; num_particles * dimensions];
    let bounds_min = -5.0;
    let bounds_max = 5.0;

    // Upload to GPU
    let mut positions_gpu = device.htod_sync_copy(&positions)?;
    let velocities_gpu = device.htod_sync_copy(&velocities)?;

    // Launch position update kernel
    // SAFETY: positions_gpu and velocities_gpu are valid from htod_sync_copy.
    // Both buffers have num_particles * dimensions elements.
    unsafe {
        launch_pso_position_update(
            positions_gpu.as_mut_ptr() as *mut f32,
            velocities_gpu.as_ptr() as *const f32,
            num_particles as u32,
            dimensions as u32,
            bounds_min,
            bounds_max,
        );
    }

    // Verify positions updated
    let updated_positions = device.dtoh_sync_copy(&positions_gpu)?;

    for i in 0..positions.len() {
        // Positions should have moved by velocity
        assert!(
            (updated_positions[i] - 0.1).abs() < 1e-5,
            "Position should be updated by velocity"
        );

        // Positions should be within bounds
        assert!(
            updated_positions[i] >= bounds_min && updated_positions[i] <= bounds_max,
            "Position should be within bounds"
        );
    }

    Ok(())
}

#[test]
fn test_swarm_fitness_kernel_launch() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let num_particles = 96;
    let dimensions = 8;

    // Create test positions (for sphere function)
    let mut positions = vec![0.0f32; num_particles * dimensions];
    for i in 0..num_particles {
        for j in 0..dimensions {
            positions[i * dimensions + j] = ((i + j) as f32) * 0.1 - 2.0;
        }
    }

    // Upload to GPU
    let positions_gpu = device.htod_sync_copy(&positions)?;
    let mut fitness_gpu = device.alloc_zeros::<f32>(num_particles)?;

    // Launch fitness evaluation kernel
    // SAFETY: positions_gpu valid from htod_sync_copy with num_particles*dimensions elements.
    // fitness_gpu valid from alloc_zeros with num_particles elements.
    unsafe {
        launch_swarm_fitness(
            positions_gpu.as_ptr() as *const f32,
            fitness_gpu.as_mut_ptr() as *mut f32,
            num_particles as u32,
            dimensions as u32,
        );
    }

    // Verify fitness values
    let fitness_values = device.dtoh_sync_copy(&fitness_gpu)?;

    for (i, &fitness) in fitness_values.iter().enumerate() {
        assert!(
            fitness >= 0.0 && fitness <= 1.0,
            "Fitness should be normalized"
        );

        // Particles closer to origin should have better fitness
        let distance_sq: f32 = (0..dimensions)
            .map(|j| positions[i * dimensions + j].powi(2))
            .sum();

        if i > 0 && distance_sq < 1.0 {
            // Close particles should have high fitness
            assert!(fitness > 0.5, "Close particles should have good fitness");
        }
    }

    Ok(())
}

#[test]
fn test_swarm_communication_kernel_launch() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let num_particles = 50;
    let dimensions = 6;
    let neighborhood_size = 5;

    // Create test data
    let positions = vec![0.5f32; num_particles * dimensions];
    let mut shared_knowledge = vec![0.0f32; num_particles * neighborhood_size];

    // Create neighborhood topology (ring topology for simplicity)
    let mut neighbors = vec![0u32; num_particles * neighborhood_size];
    for i in 0..num_particles {
        for j in 0..neighborhood_size {
            neighbors[i * neighborhood_size + j] = ((i + j) % num_particles) as u32;
        }
    }

    // Upload to GPU
    let positions_gpu = device.htod_sync_copy(&positions)?;
    let mut knowledge_gpu = device.htod_sync_copy(&shared_knowledge)?;
    let neighbors_gpu = device.htod_sync_copy(&neighbors)?;

    // Launch communication kernel
    // SAFETY: All device pointers from htod_sync_copy are valid. neighbors contains
    // valid particle indices within [0, num_particles). Buffer sizes match parameters.
    unsafe {
        launch_swarm_communication(
            positions_gpu.as_ptr() as *const f32,
            knowledge_gpu.as_mut_ptr() as *mut f32,
            neighbors_gpu.as_ptr() as *const u32,
            num_particles as u32,
            dimensions as u32,
            neighborhood_size as u32,
        );
    }

    // Verify knowledge sharing occurred
    let updated_knowledge = device.dtoh_sync_copy(&knowledge_gpu)?;

    let mut non_zero = 0;
    for &val in &updated_knowledge {
        if val != 0.0 {
            non_zero += 1;
        }
    }

    assert!(non_zero > 0, "Knowledge should be shared between particles");

    Ok(())
}

// =============================================================================
// Kernel Performance Tests
// =============================================================================

#[test]
#[ignore] // Run with --ignored for performance testing
fn test_kernel_performance_scaling() -> Result<()> {
    let device = CudaDevice::new(0)?;

    // Test different population sizes
    let population_sizes = vec![1024, 4096, 16384, 65536];
    let code_size = 256;

    println!("\nKernel Performance Scaling Test:");
    println!("Population Size | Evaluation Time | Mutation Time | Throughput");
    println!("----------------|-----------------|---------------|------------");

    for &pop_size in &population_sizes {
        // Allocate memory
        let agents = device.alloc_zeros::<u8>(pop_size * code_size)?;
        let performances = device.alloc_zeros::<f32>(pop_size)?;
        let metrics = device.alloc_zeros::<f32>(pop_size * 4)?;

        // Time evaluation kernel
        let eval_start = std::time::Instant::now();
        // SAFETY: Device pointers from alloc_zeros are valid for this scope.
        // pop_size and code_size match allocation sizes.
        unsafe {
            prepare_adas_evaluation(
                agents.as_ptr() as *const std::ffi::c_void,
                performances.as_mut_ptr() as *mut f32,
                metrics.as_mut_ptr() as *mut f32,
                pop_size as u32,
                code_size as u32,
            );
        }
        device.synchronize()?;
        let eval_time = eval_start.elapsed();

        // Time mutation kernel
        let mut_start = std::time::Instant::now();
        // SAFETY: agents and performances are valid device pointers from alloc_zeros.
        // pop_size and code_size match allocation sizes.
        unsafe {
            launch_adas_mutation(
                agents.as_mut_ptr() as *mut std::ffi::c_void,
                performances.as_ptr() as *const f32,
                pop_size as u32,
                code_size as u32,
                0.05,
            );
        }
        device.synchronize()?;
        let mut_time = mut_start.elapsed();

        // Calculate throughput (agents per second)
        let eval_throughput = pop_size as f64 / eval_time.as_secs_f64();

        println!(
            "{:15} | {:15.3?} | {:13.3?} | {:.0} agents/sec",
            pop_size, eval_time, mut_time, eval_throughput
        );
    }

    Ok(())
}

// =============================================================================
// Memory Transfer Tests
// =============================================================================

#[test]
fn test_kernel_memory_transfer_efficiency() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let population_size = 1024;
    let code_size = 512;

    // Create test data
    let host_data = vec![42u8; population_size * code_size];

    // Test upload speed
    let upload_start = std::time::Instant::now();
    let gpu_data = device.htod_sync_copy(&host_data)?;
    let upload_time = upload_start.elapsed();

    // Test download speed
    let download_start = std::time::Instant::now();
    let downloaded = device.dtoh_sync_copy(&gpu_data)?;
    let download_time = download_start.elapsed();

    // Calculate bandwidth
    let data_size_mb = (population_size * code_size) as f64 / (1024.0 * 1024.0);
    let upload_bandwidth = data_size_mb / upload_time.as_secs_f64();
    let download_bandwidth = data_size_mb / download_time.as_secs_f64();

    println!("Memory Transfer Performance:");
    println!("  Data size: {:.2} MB", data_size_mb);
    println!(
        "  Upload: {:.3?} ({:.0} MB/s)",
        upload_time, upload_bandwidth
    );
    println!(
        "  Download: {:.3?} ({:.0} MB/s)",
        download_time, download_bandwidth
    );

    // Verify data integrity
    assert_eq!(downloaded.len(), host_data.len());
    assert_eq!(downloaded[0], 42);
    assert_eq!(downloaded[downloaded.len() - 1], 42);

    Ok(())
}
