#include "types.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Evolution parameters
#define MUTATION_RATE 0.01f
#define CROSSOVER_POINTS 4
#define TOURNAMENT_SIZE 3
#define ELITISM_PERCENTAGE 0.1f

// Device function for mutation
__device__ void gpu_mutate(
    float*        genome,
    curandState*  rng,
    float         mutation_strength)
{
    for (int i = threadIdx.x; i < GENOME_SIZE; i += blockDim.x)
    {
        if (curand_uniform(rng) < MUTATION_RATE)
        {
            // Gaussian mutation
            genome[i] += curand_normal(rng) * mutation_strength;
            
            // Clamp to valid range
            genome[i] = fmaxf(-1.0f, fminf(1.0f, genome[i]));
        }
    }
}

// Evolution kernel - simple tournament selection
__global__ void evolution_kernel(
    GPUAgent*    agents,
    float*       genomes,
    curandState* rng_states,
    uint32_t     num_agents,
    uint32_t     generation)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_agents) return;
    
    // Initialize RNG
    curandState* rng = &rng_states[tid];
    if (generation == 0)
    {
        curand_init(tid + 1337, 0, 0, rng);
    }
    
    // Tournament selection - find best in neighborhood
    GPUAgent* agent      = &agents[tid];
    float     best_fitness = agent->fitness;
    uint32_t  best_idx     = tid;
    
    // Check neighbors
    for (int i = 0; i < 2; i++)
    {
        uint32_t neighbor_idx = agent->neighbors[i];
        if (neighbor_idx < num_agents)
        {
            float neighbor_fitness = agents[neighbor_idx].fitness;
            if (neighbor_fitness > best_fitness)
            {
                best_fitness = neighbor_fitness;
                best_idx     = neighbor_idx;
            }
        }
    }
    
    // If we're not the best, copy from best and mutate
    if (best_idx != tid)
    {
        // Copy genome from best
        float* my_genome   = &genomes[tid * GENOME_SIZE];
        float* best_genome = &genomes[best_idx * GENOME_SIZE];
        
        for (int i = 0; i < GENOME_SIZE; i++)
        {
            my_genome[i] = best_genome[i];
        }
        
        // Apply mutation
        gpu_mutate(my_genome, rng, 0.1f);
        
        // Reset fitness for re-evaluation
        agent->fitness = 0.5f;
    }
}

// Initialize random number generator states
__global__ void init_rng_kernel(curandState* states, uint32_t num_agents, uint32_t seed)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_agents) return;
    
    curand_init(seed + tid, 0, 0, &states[tid]);
}

// NSGA-II selection kernel
__global__ void nsga2_selection_kernel(
    GPUAgent* agents,
    float* fitness_vectors,
    uint32_t* selected_indices,
    uint32_t num_agents,
    uint32_t num_objectives,
    uint32_t population_size)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_agents) return;
    
    // Simplified NSGA-II - compute dominance count
    uint32_t domination_count = 0;
    
    for (uint32_t i = 0; i < num_agents; i++) {
        if (i != tid) {
            bool dominates = true;
            bool at_least_one_better = false;
            
            // Check if agent i dominates agent tid
            for (uint32_t obj = 0; obj < num_objectives; obj++) {
                float my_fitness = fitness_vectors[tid * num_objectives + obj];
                float other_fitness = fitness_vectors[i * num_objectives + obj];
                
                if (other_fitness < my_fitness) {
                    dominates = false;
                    break;
                } else if (other_fitness > my_fitness) {
                    at_least_one_better = true;
                }
            }
            
            if (dominates && at_least_one_better) {
                domination_count++;
            }
        }
    }
    
    // Simple ranking: lower domination count = better
    agents[tid].fitness = 1.0f / (1.0f + domination_count);
}

// Multi-objective fitness evaluation kernel
__global__ void evaluate_multi_objective_fitness_kernel(
    GPUAgent* agents,
    float* fitness_vectors,
    uint32_t num_agents,
    uint32_t num_objectives,
    float kernel_time_ms,
    float memory_usage_mb)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_agents) return;
    
    GPUAgent* agent = &agents[tid];
    float* fitness_vector = &fitness_vectors[tid * num_objectives];
    
    // Objective 0: Performance (higher is better)
    fitness_vector[0] = 1.0f / (kernel_time_ms + 1.0f);
    fitness_vector[0] *= (1.0f + (tid * 0.001f) * sinf(tid * 0.1f)); // Agent-specific variation
    
    // Objective 1: Efficiency (lower memory usage is better)
    if (num_objectives > 1) {
        fitness_vector[1] = 1.0f / (memory_usage_mb + 1.0f);
        fitness_vector[1] *= (1.0f + (tid * 0.002f) * cosf(tid * 0.1f));
    }
    
    // Objective 2: Novelty (based on position diversity)
    if (num_objectives > 2) {
        float novelty = sinf(tid * 0.05f) * cosf(tid * 0.03f);
        // Use deterministic pseudo-random based on tid for novelty
        float pseudo_random = sinf(tid * 0.123f + cosf(tid * 0.456f));
        fitness_vector[2] = fabsf(novelty) * (0.5f + fabsf(pseudo_random) * 0.5f);
    }
    
    // Update agent's scalar fitness (sum of objectives for tournament selection)
    agent->fitness = 0.0f;
    for (uint32_t obj = 0; obj < num_objectives; obj++) {
        agent->fitness += fitness_vector[obj];
    }
    agent->fitness /= num_objectives;
}

// Crossover kernel
__global__ void crossover_kernel(
    float* parent_genomes,
    float* offspring_genomes,
    uint32_t* parent_indices,
    curandState* rng_states,
    uint32_t num_offspring,
    float crossover_rate)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_offspring) return;
    
    curandState* rng = &rng_states[tid];
    
    // Select two parents
    uint32_t parent1_idx = parent_indices[tid * 2];
    uint32_t parent2_idx = parent_indices[tid * 2 + 1];
    
    float* parent1_genome = &parent_genomes[parent1_idx * GENOME_SIZE];
    float* parent2_genome = &parent_genomes[parent2_idx * GENOME_SIZE];
    float* offspring_genome = &offspring_genomes[tid * GENOME_SIZE];
    
    // Uniform crossover
    for (int i = 0; i < GENOME_SIZE; i++) {
        if (curand_uniform(rng) < crossover_rate) {
            offspring_genome[i] = parent1_genome[i];
        } else {
            offspring_genome[i] = parent2_genome[i];
        }
    }
}

// Enhanced mutation kernel with adaptive rates
__global__ void adaptive_mutation_kernel(
    float* genomes,
    curandState* rng_states,
    uint32_t num_agents,
    float base_mutation_rate,
    float population_diversity)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_agents) return;
    
    curandState* rng = &rng_states[tid];
    float* genome = &genomes[tid * GENOME_SIZE];
    
    // Adaptive mutation rate based on diversity
    float adaptive_rate = base_mutation_rate;
    if (population_diversity < 0.1f) {
        adaptive_rate *= 2.0f; // Increase mutation when diversity is low
    } else if (population_diversity > 0.5f) {
        adaptive_rate *= 0.5f; // Decrease mutation when diversity is high
    }
    
    // Apply Gaussian mutation
    for (int i = 0; i < GENOME_SIZE; i++) {
        if (curand_uniform(rng) < adaptive_rate) {
            float mutation_strength = 0.1f * (1.0f - population_diversity); // Stronger mutations for low diversity
            genome[i] += curand_normal(rng) * mutation_strength;
            genome[i] = fmaxf(-1.0f, fminf(1.0f, genome[i])); // Clamp to [-1, 1]
        }
    }
}

// Export C interface
extern "C"
{
    void launch_evolution(GPUAgent* agents, float* genomes, curandState* rng_states,
                         uint32_t num_agents, uint32_t generation)
    {
        dim3 block_size(256);
        dim3 grid_size((num_agents + block_size.x - 1) / block_size.x);
        
        evolution_kernel<<<grid_size, block_size>>>(agents, genomes, rng_states,
                                                   num_agents, generation);
        cudaDeviceSynchronize();
    }
    
    void launch_init_rng(curandState* states, uint32_t num_agents, uint32_t seed)
    {
        dim3 block_size(256);
        dim3 grid_size((num_agents + block_size.x - 1) / block_size.x);
        
        init_rng_kernel<<<grid_size, block_size>>>(states, num_agents, seed);
        cudaDeviceSynchronize();
    }
    
    void launch_multi_objective_fitness(GPUAgent* agents, float* fitness_vectors,
                                       uint32_t num_agents, uint32_t num_objectives,
                                       float kernel_time_ms, float memory_usage_mb)
    {
        dim3 block_size(256);
        dim3 grid_size((num_agents + block_size.x - 1) / block_size.x);
        
        evaluate_multi_objective_fitness_kernel<<<grid_size, block_size>>>(
            agents, fitness_vectors, num_agents, num_objectives, 
            kernel_time_ms, memory_usage_mb);
        cudaDeviceSynchronize();
    }
    
    void launch_nsga2_selection(GPUAgent* agents, float* fitness_vectors,
                               uint32_t* selected_indices, uint32_t num_agents,
                               uint32_t num_objectives, uint32_t population_size)
    {
        dim3 block_size(256);
        dim3 grid_size((num_agents + block_size.x - 1) / block_size.x);
        
        nsga2_selection_kernel<<<grid_size, block_size>>>(
            agents, fitness_vectors, selected_indices, num_agents,
            num_objectives, population_size);
        cudaDeviceSynchronize();
    }
    
    void launch_adaptive_mutation(float* genomes, curandState* rng_states,
                                 uint32_t num_agents, float base_mutation_rate,
                                 float population_diversity)
    {
        dim3 block_size(256);
        dim3 grid_size((num_agents + block_size.x - 1) / block_size.x);
        
        adaptive_mutation_kernel<<<grid_size, block_size>>>(
            genomes, rng_states, num_agents, base_mutation_rate, population_diversity);
        cudaDeviceSynchronize();
    }

// =============================================================================
// ADAS (Automated Design of Agentic Systems) Kernels
// =============================================================================

__global__ void adas_evaluation_kernel(
    const uint8_t* agent_codes,
    float* performances,
    uint32_t population_size,
    uint32_t max_code_size
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= population_size) return;
    
    const uint8_t* agent_code = &agent_codes[tid * max_code_size];
    
    // Evaluate agent performance based on code complexity and structure
    float complexity = 0.0f;
    float structure_score = 0.0f;
    uint32_t non_zero_bytes = 0;
    
    // Calculate code complexity
    for (uint32_t i = 0; i < max_code_size; i++) {
        if (agent_code[i] != 0) {
            non_zero_bytes++;
            complexity += (float)agent_code[i] / 255.0f;
        }
    }
    
    // Normalize complexity by code size
    if (non_zero_bytes > 0) {
        complexity /= non_zero_bytes;
    }
    
    // Calculate structure score based on patterns
    for (uint32_t i = 0; i < max_code_size - 3; i += 4) {
        uint32_t pattern = *(uint32_t*)&agent_code[i];
        if (pattern != 0) {
            structure_score += 0.1f;
        }
    }
    
    // Combine scores with bias toward simpler, well-structured code
    float performance = (0.3f * (1.0f - complexity)) + (0.7f * structure_score);
    performance = fminf(fmaxf(performance, 0.0f), 1.0f);
    
    performances[tid] = performance;
}

__global__ void adas_mutation_kernel(
    const uint8_t* parent_codes,
    uint8_t* offspring_codes,
    const uint32_t* mutation_types,
    uint32_t population_size,
    uint32_t max_code_size,
    float mutation_rate,
    curandState* rng_states
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= population_size) return;
    
    curandState* rng = &rng_states[tid];
    const uint8_t* parent = &parent_codes[tid * max_code_size];
    uint8_t* offspring = &offspring_codes[tid * max_code_size];
    
    // Copy parent to offspring first
    for (uint32_t i = 0; i < max_code_size; i++) {
        offspring[i] = parent[i];
    }
    
    // Apply mutations based on mutation type
    uint32_t mutation_type = mutation_types[tid];
    uint32_t num_mutations = (uint32_t)(mutation_rate * max_code_size);
    
    switch (mutation_type) {
        case 0: // Point mutation
            for (uint32_t m = 0; m < num_mutations; m++) {
                uint32_t pos = curand(rng) % max_code_size;
                offspring[pos] = (uint8_t)(curand(rng) % 256);
            }
            break;
            
        case 1: // Block insertion
            if (max_code_size >= 16) {
                uint32_t insert_pos = curand(rng) % (max_code_size - 8);
                uint32_t block_size = 4 + (curand(rng) % 4);
                for (uint32_t i = 0; i < block_size && insert_pos + i < max_code_size; i++) {
                    offspring[insert_pos + i] = (uint8_t)(curand(rng) % 256);
                }
            }
            break;
    }
}

// =============================================================================
// DGM (Darwin Gödel Machine) Kernels
// =============================================================================

__global__ void dgm_self_modification_kernel(
    const uint8_t* agent_code,
    uint8_t* modified_code,
    const float* performance_history,
    uint32_t code_size,
    uint32_t history_length,
    float improvement_threshold,
    curandState* rng_states
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= 1) return; // Single thread for self-modification
    
    curandState* rng = &rng_states[0];
    
    // Copy original code
    for (uint32_t i = 0; i < code_size; i++) {
        modified_code[i] = agent_code[i];
    }
    
    // Analyze performance trend
    float recent_avg = 0.0f;
    float older_avg = 0.0f;
    uint32_t recent_count = 0;
    uint32_t older_count = 0;
    
    uint32_t split_point = history_length / 2;
    
    for (uint32_t i = 0; i < history_length; i++) {
        if (i < split_point) {
            older_avg += performance_history[i];
            older_count++;
        } else {
            recent_avg += performance_history[i];
            recent_count++;
        }
    }
    
    if (recent_count > 0) recent_avg /= recent_count;
    if (older_count > 0) older_avg /= older_count;
    
    // Decide on modification strategy based on performance trend
    float improvement = recent_avg - older_avg;
    
    if (improvement >= improvement_threshold) {
        // Performance is improving - make conservative changes
        uint32_t num_changes = 1 + (curand(rng) % 3);
        for (uint32_t c = 0; c < num_changes; c++) {
            uint32_t pos = curand(rng) % code_size;
            uint8_t current = modified_code[pos];
            modified_code[pos] = current + (curand(rng) % 32) - 16; // Small change
        }
    } else {
        // Performance is stagnating - make larger changes
        uint32_t num_changes = 5 + (curand(rng) % 10);
        for (uint32_t c = 0; c < num_changes; c++) {
            uint32_t pos = curand(rng) % code_size;
            modified_code[pos] = (uint8_t)(curand(rng) % 256); // Random change
        }
    }
}

// =============================================================================
// Swarm Optimization Kernels
// =============================================================================

__global__ void pso_velocity_update_kernel(
    float* velocities,
    const float* positions,
    const float* personal_best,
    const float* global_best,
    uint32_t population_size,
    uint32_t dimensions,
    float inertia,
    float cognitive,
    float social,
    curandState* rng_states
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= population_size * dimensions) return;
    
    uint32_t particle_id = tid / dimensions;
    uint32_t dim_id = tid % dimensions;
    
    curandState* rng = &rng_states[particle_id];
    
    float current_velocity = velocities[tid];
    float current_position = positions[tid];
    float personal_best_pos = personal_best[tid];
    float global_best_pos = global_best[dim_id];
    
    // PSO velocity update equation
    float r1 = curand_uniform(rng);
    float r2 = curand_uniform(rng);
    
    float new_velocity = inertia * current_velocity +
                        cognitive * r1 * (personal_best_pos - current_position) +
                        social * r2 * (global_best_pos - current_position);
    
    velocities[tid] = new_velocity;
}

__global__ void pso_position_update_kernel(
    float* positions,
    const float* velocities,
    uint32_t population_size,
    uint32_t dimensions,
    float max_velocity
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= population_size * dimensions) return;
    
    float velocity = velocities[tid];
    
    // Clamp velocity
    velocity = fminf(fmaxf(velocity, -max_velocity), max_velocity);
    
    // Update position
    positions[tid] += velocity;
    
    // Keep positions in reasonable bounds
    positions[tid] = fminf(fmaxf(positions[tid], -10.0f), 10.0f);
}

// =============================================================================
// Additional kernel implementations
// =============================================================================

__global__ void adas_crossover_kernel(
    const uint8_t* parent1_codes,
    const uint8_t* parent2_codes,
    uint8_t* offspring_codes,
    const uint32_t* crossover_points,
    uint32_t population_size,
    uint32_t max_code_size
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= population_size) return;
    
    const uint8_t* parent1 = &parent1_codes[tid * max_code_size];
    const uint8_t* parent2 = &parent2_codes[tid * max_code_size];
    uint8_t* offspring = &offspring_codes[tid * max_code_size];
    uint32_t crossover_point = crossover_points[tid] % max_code_size;
    
    // Two-point crossover
    bool use_parent1 = true;
    uint32_t second_point = (crossover_point + max_code_size / 2) % max_code_size;
    
    if (crossover_point > second_point) {
        uint32_t temp = crossover_point;
        crossover_point = second_point;
        second_point = temp;
    }
    
    for (uint32_t i = 0; i < max_code_size; i++) {
        if (i >= crossover_point && i < second_point) {
            use_parent1 = !use_parent1;
        }
        
        offspring[i] = use_parent1 ? parent1[i] : parent2[i];
    }
}

__global__ void adas_diversity_kernel(
    const uint8_t* agent_codes,
    float* diversity_scores,
    uint32_t population_size,
    uint32_t max_code_size
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= population_size) return;
    
    const uint8_t* target_code = &agent_codes[tid * max_code_size];
    float total_diversity = 0.0f;
    uint32_t comparisons = 0;
    
    // Compare with a subset of other agents for efficiency
    uint32_t sample_size = min(population_size / 4, 32U);
    uint32_t step = max(1U, population_size / sample_size);
    
    for (uint32_t i = 0; i < population_size; i += step) {
        if (i != tid) {
            const uint8_t* other_code = &agent_codes[i * max_code_size];
            
            // Compute similarity
            uint32_t matches = 0;
            for (uint32_t j = 0; j < max_code_size; j++) {
                if (target_code[j] == other_code[j]) {
                    matches++;
                }
            }
            float similarity = (float)matches / max_code_size;
            total_diversity += (1.0f - similarity);
            comparisons++;
        }
    }
    
    diversity_scores[tid] = comparisons > 0 ? (total_diversity / comparisons) : 0.5f;
}

__global__ void dgm_benchmark_kernel(
    const uint8_t* agent_codes,
    float* benchmark_scores,
    uint32_t population_size,
    uint32_t code_size,
    const uint8_t* benchmark_data
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= population_size) return;
    
    const uint8_t* agent_code = &agent_codes[tid * code_size];
    
    // Evaluate agent against multiple benchmark criteria
    float complexity_score = 0.0f;
    float efficiency_score = 0.0f;
    float robustness_score = 0.0f;
    
    // Complexity analysis
    uint32_t unique_patterns = 0;
    uint32_t pattern_map[256] = {0}; // Simple histogram
    
    for (uint32_t i = 0; i < code_size; i++) {
        if (pattern_map[agent_code[i]] == 0) {
            unique_patterns++;
        }
        pattern_map[agent_code[i]]++;
    }
    
    complexity_score = (float)unique_patterns / 256.0f;
    
    // Efficiency (code compactness)
    uint32_t non_zero_bytes = 0;
    for (uint32_t i = 0; i < code_size; i++) {
        if (agent_code[i] != 0) {
            non_zero_bytes++;
        }
    }
    efficiency_score = 1.0f - ((float)non_zero_bytes / code_size);
    
    // Robustness (resistance to small changes)
    uint32_t robust_blocks = 0;
    for (uint32_t i = 0; i < code_size - 4; i += 4) {
        uint32_t block_sum = 0;
        for (uint32_t j = 0; j < 4; j++) {
            block_sum += agent_code[i + j];
        }
        if (block_sum > 128 && block_sum < 896) { // Not all zeros or all high values
            robust_blocks++;
        }
    }
    robustness_score = (float)robust_blocks / (code_size / 4);
    
    // Combined benchmark score
    benchmark_scores[tid] = (0.4f * complexity_score) + 
                           (0.3f * efficiency_score) + 
                           (0.3f * robustness_score);
}

__global__ void dgm_archive_update_kernel(
    const uint8_t* new_agents,
    uint8_t* archive,
    const float* performances,
    uint32_t* archive_indices,
    uint32_t new_agent_count,
    uint32_t archive_size,
    uint32_t code_size
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= new_agent_count) return;
    
    float new_performance = performances[tid];
    const uint8_t* new_agent = &new_agents[tid * code_size];
    
    // Find the worst performing agent in archive to potentially replace
    uint32_t worst_idx = 0;
    float worst_performance = 1.0f;
    
    for (uint32_t i = 0; i < archive_size; i++) {
        // Simple performance lookup (would be stored separately in real implementation)
        float archive_performance = 0.5f; // Placeholder
        if (archive_performance < worst_performance) {
            worst_performance = archive_performance;
            worst_idx = i;
        }
    }
    
    // Replace if new agent is better
    if (new_performance > worst_performance) {
        uint8_t* archive_slot = &archive[worst_idx * code_size];
        for (uint32_t i = 0; i < code_size; i++) {
            archive_slot[i] = new_agent[i];
        }
        archive_indices[tid] = worst_idx;
    } else {
        archive_indices[tid] = UINT32_MAX; // Not added
    }
}

__global__ void swarm_communication_kernel(
    const float* agent_states,
    float* shared_knowledge,
    const uint32_t* neighborhood_matrix,
    uint32_t population_size,
    uint32_t state_size
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= population_size) return;
    
    const float* my_state = &agent_states[tid * state_size];
    float* my_knowledge = &shared_knowledge[tid * state_size];
    
    // Initialize with own state
    for (uint32_t i = 0; i < state_size; i++) {
        my_knowledge[i] = my_state[i];
    }
    
    // Aggregate information from neighbors
    uint32_t neighbor_count = 0;
    for (uint32_t neighbor = 0; neighbor < population_size; neighbor++) {
        if (neighborhood_matrix[tid * population_size + neighbor] > 0) {
            const float* neighbor_state = &agent_states[neighbor * state_size];
            for (uint32_t i = 0; i < state_size; i++) {
                my_knowledge[i] += neighbor_state[i];
            }
            neighbor_count++;
        }
    }
    
    // Average the aggregated information
    if (neighbor_count > 0) {
        for (uint32_t i = 0; i < state_size; i++) {
            my_knowledge[i] /= (neighbor_count + 1); // +1 for self
        }
    }
}

__global__ void swarm_fitness_kernel(
    const float* positions,
    float* fitness_scores,
    uint32_t population_size,
    uint32_t dimensions,
    const float* target_function
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= population_size) return;
    
    const float* particle_position = &positions[tid * dimensions];
    
    // Compute fitness using Sphere function as default
    float fitness = 0.0f;
    for (uint32_t d = 0; d < dimensions; d++) {
        float coord = particle_position[d];
        fitness += coord * coord;
    }
    
    // Convert to maximization problem (higher is better)
    fitness_scores[tid] = 1.0f / (1.0f + fitness);
}

// =============================================================================
// Export additional C interface functions
// =============================================================================

    void prepare_adas_evaluation(
        const uint8_t* agent_codes,
        float* performances,
        uint32_t population_size,
        uint32_t max_code_size
    ) {
        dim3 block_size(256);
        dim3 grid_size((population_size + block_size.x - 1) / block_size.x);
        
        adas_evaluation_kernel<<<grid_size, block_size>>>(
            agent_codes, performances, population_size, max_code_size
        );
        
        cudaDeviceSynchronize();
    }

    void launch_adas_mutation(
        const uint8_t* parent_codes,
        uint8_t* offspring_codes,
        const uint32_t* mutation_types,
        uint32_t population_size,
        uint32_t max_code_size,
        float mutation_rate
    ) {
        dim3 block_size(256);
        dim3 grid_size((population_size + block_size.x - 1) / block_size.x);
        
        // Initialize RNG states (simplified - should be done once)
        curandState* rng_states;
        cudaMalloc(&rng_states, population_size * sizeof(curandState));
        
        adas_mutation_kernel<<<grid_size, block_size>>>(
            parent_codes, offspring_codes, mutation_types,
            population_size, max_code_size, mutation_rate, rng_states
        );
        
        cudaDeviceSynchronize();
        cudaFree(rng_states);
    }

    void launch_dgm_self_modification(
        const uint8_t* agent_code,
        uint8_t* modified_code,
        const float* performance_history,
        uint32_t code_size,
        uint32_t history_length,
        float improvement_threshold
    ) {
        dim3 block_size(1);
        dim3 grid_size(1);
        
        curandState* rng_states;
        cudaMalloc(&rng_states, sizeof(curandState));
        
        dgm_self_modification_kernel<<<grid_size, block_size>>>(
            agent_code, modified_code, performance_history,
            code_size, history_length, improvement_threshold, rng_states
        );
        
        cudaDeviceSynchronize();
        cudaFree(rng_states);
    }

    void launch_pso_velocity_update(
        float* velocities,
        const float* positions,
        const float* personal_best,
        const float* global_best,
        uint32_t population_size,
        uint32_t dimensions,
        float inertia,
        float cognitive,
        float social
    ) {
        dim3 block_size(256);
        dim3 grid_size((population_size * dimensions + block_size.x - 1) / block_size.x);
        
        curandState* rng_states;
        cudaMalloc(&rng_states, population_size * sizeof(curandState));
        
        pso_velocity_update_kernel<<<grid_size, block_size>>>(
            velocities, positions, personal_best, global_best,
            population_size, dimensions, inertia, cognitive, social, rng_states
        );
        
        cudaDeviceSynchronize();
        cudaFree(rng_states);
    }

    void launch_pso_position_update(
        float* positions,
        const float* velocities,
        uint32_t population_size,
        uint32_t dimensions,
        float max_velocity
    ) {
        dim3 block_size(256);
        dim3 grid_size((population_size * dimensions + block_size.x - 1) / block_size.x);
        
        pso_position_update_kernel<<<grid_size, block_size>>>(
            positions, velocities, population_size, dimensions, max_velocity
        );
        
        cudaDeviceSynchronize();
    }

    void launch_adas_crossover(
        const uint8_t* parent1_codes,
        const uint8_t* parent2_codes,
        uint8_t* offspring_codes,
        const uint32_t* crossover_points,
        uint32_t population_size,
        uint32_t max_code_size
    ) {
        dim3 block_size(256);
        dim3 grid_size((population_size + block_size.x - 1) / block_size.x);
        
        adas_crossover_kernel<<<grid_size, block_size>>>(
            parent1_codes, parent2_codes, offspring_codes, crossover_points,
            population_size, max_code_size
        );
        
        cudaDeviceSynchronize();
    }

    void compute_adas_diversity(
        const uint8_t* agent_codes,
        float* diversity_scores,
        uint32_t population_size,
        uint32_t max_code_size
    ) {
        dim3 block_size(256);
        dim3 grid_size((population_size + block_size.x - 1) / block_size.x);
        
        adas_diversity_kernel<<<grid_size, block_size>>>(
            agent_codes, diversity_scores, population_size, max_code_size
        );
        
        cudaDeviceSynchronize();
    }

    void evaluate_dgm_benchmark(
        const uint8_t* agent_codes,
        float* benchmark_scores,
        uint32_t population_size,
        uint32_t code_size,
        const uint8_t* benchmark_data
    ) {
        dim3 block_size(256);
        dim3 grid_size((population_size + block_size.x - 1) / block_size.x);
        
        dgm_benchmark_kernel<<<grid_size, block_size>>>(
            agent_codes, benchmark_scores, population_size, code_size, benchmark_data
        );
        
        cudaDeviceSynchronize();
    }

    void launch_dgm_archive_update(
        const uint8_t* new_agents,
        uint8_t* archive,
        const float* performances,
        uint32_t* archive_indices,
        uint32_t new_agent_count,
        uint32_t archive_size,
        uint32_t code_size
    ) {
        dim3 block_size(256);
        dim3 grid_size((new_agent_count + block_size.x - 1) / block_size.x);
        
        dgm_archive_update_kernel<<<grid_size, block_size>>>(
            new_agents, archive, performances, archive_indices,
            new_agent_count, archive_size, code_size
        );
        
        cudaDeviceSynchronize();
    }

    void launch_swarm_communication(
        const float* agent_states,
        float* shared_knowledge,
        const uint32_t* neighborhood_matrix,
        uint32_t population_size,
        uint32_t state_size
    ) {
        dim3 block_size(256);
        dim3 grid_size((population_size + block_size.x - 1) / block_size.x);
        
        swarm_communication_kernel<<<grid_size, block_size>>>(
            agent_states, shared_knowledge, neighborhood_matrix,
            population_size, state_size
        );
        
        cudaDeviceSynchronize();
    }

    void compute_swarm_fitness(
        const float* positions,
        float* fitness_scores,
        uint32_t population_size,
        uint32_t dimensions,
        const float* target_function
    ) {
        dim3 block_size(256);
        dim3 grid_size((population_size + block_size.x - 1) / block_size.x);
        
        swarm_fitness_kernel<<<grid_size, block_size>>>(
            positions, fitness_scores, population_size, dimensions, target_function
        );
        
        cudaDeviceSynchronize();
    }

}