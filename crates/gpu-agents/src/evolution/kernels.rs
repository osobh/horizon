//! CUDA kernel declarations for evolution operations

// External CUDA kernel functions
unsafe extern "C" {
    /// Initialize population with random genomes
    pub fn launch_random_init(
        genomes: *mut f32,
        population_size: u32,
        genome_size: u32,
        rng_states: *mut u8,
        stream: *mut u8,
    );

    /// Find index of best fitness
    pub fn find_best_fitness(
        fitness_scores: *const f32,
        best_index: *mut u32,
        best_value: *mut f32,
        population_size: u32,
        stream: *mut u8,
    );

    /// Compute average fitness
    pub fn compute_average_fitness(
        fitness_scores: *const f32,
        average: *mut f32,
        population_size: u32,
        stream: *mut u8,
    );

    /// Compute population diversity
    pub fn compute_diversity(
        genomes: *const f32,
        diversity: *mut f32,
        population_size: u32,
        genome_size: u32,
        stream: *mut u8,
    );

    /// Setup RNG states for CUDA kernels
    pub fn setup_rng_states(num_states: u32, seed: u32) -> *mut u8;

    /// Cleanup RNG states
    pub fn cleanup_rng_states(states: *mut u8);

    /// Perform crossover operation
    pub fn launch_crossover(
        parent_genomes: *const f32,
        offspring_genomes: *mut f32,
        parent_indices: *const u32,
        num_offspring: u32,
        genome_size: u32,
        rng_states: *mut u8,
        stream: *mut u8,
    );

    /// Evaluate fitness function
    pub fn launch_fitness_evaluation(
        genomes: *const f32,
        fitness_scores: *mut f32,
        population_size: u32,
        genome_size: u32,
        stream: *mut u8,
    );

    /// Perform mutations
    pub fn launch_mutation(
        genomes: *mut f32,
        fitness: *const f32,
        population_size: u32,
        genome_size: u32,
        mutation_rate: f32,
        rng_states: *mut u8,
        stream: *mut u8,
    );

    /// Tournament selection
    pub fn launch_tournament_selection(
        fitness_scores: *const f32,
        selected_indices: *mut u32,
        population_size: u32,
        num_selections: u32,
        tournament_size: u32,
        rng_states: *mut u8,
        stream: *mut u8,
    );

    /// Elite preservation
    pub fn launch_elite_preservation(
        fitness_scores: *const f32,
        fitness_valid: *const u8,
        elite_indices: *mut u32,
        population_size: u32,
        elite_count: u32,
    );

    /// Uniform random mutation
    pub fn launch_uniform_mutation(
        genomes: *mut f32,
        rng_states: *mut u8,
        population_size: u32,
        genome_size: u32,
        mutation_rate: f32,
        stream: *mut u8,
    );

    /// Gaussian mutation
    pub fn launch_gaussian_mutation(
        genomes: *mut f32,
        rng_states: *mut u8,
        population_size: u32,
        genome_size: u32,
        mutation_rate: f32,
        sigma: f32,
        stream: *mut u8,
    );

    /// Bit flip mutation
    pub fn launch_bitflip_mutation(
        genomes: *mut u8,
        rng_states: *mut u8,
        population_size: u32,
        genome_size_bytes: u32,
        mutation_rate: f32,
        stream: *mut u8,
    );

    // ADAS (Automated Design of Agentic Systems) kernels
    /// Prepare ADAS evaluation on GPU
    pub fn prepare_adas_evaluation(
        agent_codes: *const u8,
        performances: *mut f32,
        population_size: u32,
        max_code_size: u32,
    );

    /// Launch ADAS mutation operations
    pub fn launch_adas_mutation(
        parent_codes: *const u8,
        offspring_codes: *mut u8,
        mutation_types: *const u32,
        population_size: u32,
        max_code_size: u32,
        mutation_rate: f32,
    );

    /// Launch ADAS crossover operations  
    pub fn launch_adas_crossover(
        parent1_codes: *const u8,
        parent2_codes: *const u8,
        offspring_codes: *mut u8,
        crossover_points: *const u32,
        population_size: u32,
        max_code_size: u32,
    );

    /// Compute ADAS population diversity
    pub fn compute_adas_diversity(
        agent_codes: *const u8,
        diversity_scores: *mut f32,
        population_size: u32,
        max_code_size: u32,
    );

    // DGM (Darwin Gödel Machine) kernels
    /// Launch DGM self-modification
    pub fn launch_dgm_self_modification(
        agent_code: *const u8,
        modified_code: *mut u8,
        performance_history: *const f32,
        code_size: u32,
        history_length: u32,
        improvement_threshold: f32,
    );

    /// Evaluate DGM benchmark performance
    pub fn evaluate_dgm_benchmark(
        agent_codes: *const u8,
        benchmark_scores: *mut f32,
        population_size: u32,
        code_size: u32,
        benchmark_data: *const u8,
    );

    /// Update DGM archive
    pub fn launch_dgm_archive_update(
        new_agents: *const u8,
        archive: *mut u8,
        performances: *const f32,
        archive_indices: *mut u32,
        new_agent_count: u32,
        archive_size: u32,
        code_size: u32,
    );

    // Swarm optimization kernels
    /// Update PSO velocities
    pub fn launch_pso_velocity_update(
        velocities: *mut f32,
        positions: *const f32,
        personal_best: *const f32,
        global_best: *const f32,
        population_size: u32,
        dimensions: u32,
        inertia: f32,
        cognitive: f32,
        social: f32,
    );

    /// Update PSO positions
    pub fn launch_pso_position_update(
        positions: *mut f32,
        velocities: *const f32,
        population_size: u32,
        dimensions: u32,
        max_velocity: f32,
    );

    /// Launch swarm communication
    pub fn launch_swarm_communication(
        agent_states: *const f32,
        shared_knowledge: *mut f32,
        neighborhood_matrix: *const u32,
        population_size: u32,
        state_size: u32,
    );

    /// Compute swarm fitness
    pub fn compute_swarm_fitness(
        positions: *const f32,
        fitness_scores: *mut f32,
        population_size: u32,
        dimensions: u32,
        target_function: *const f32,
    );
}
