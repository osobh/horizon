//! Evolution algorithm shaders with embedded ML.
//!
//! These shaders implement evolutionary algorithms with neural network
//! fitness evaluation embedded directly in the compute kernels.
//!
//! # Key Innovation: Embedded ML
//!
//! Instead of evaluating fitness on the CPU or making GPU→CPU→GPU round-trips,
//! we embed small neural networks directly into the evolution kernels.
//! This provides massive speedups for fitness-dominated workloads.
//!
//! # Algorithms Implemented
//!
//! - **Tournament Selection**: Select parents based on fitness
//! - **Crossover**: Blend parent genomes
//! - **Mutation**: Perturb genome values
//! - **ADAS**: Adaptive Dynamic Agent Synthesis
//! - **DGM**: Dynamic Genome Modification
//! - **PSO**: Particle Swarm Optimization

use crate::{BufferBinding, ShaderInfo};

/// Core evolution shader with embedded neural network fitness evaluation.
///
/// This shader combines all evolution operations with inline neural network
/// evaluation, eliminating GPU→CPU→GPU round-trips for fitness computation.
///
/// # Network Architecture
///
/// The embedded network is a simple MLP:
/// - Input: Genome values (up to 64 dimensions)
/// - Hidden 1: 32 neurons with tanh activation
/// - Hidden 2: 16 neurons with tanh activation
/// - Output: 1 neuron with sigmoid (fitness score 0-1)
///
/// # Kernel Functions
///
/// - `evaluate_fitness`: Compute fitness using embedded neural network
/// - `tournament_select`: Select parents via tournament selection
/// - `crossover_uniform`: Uniform crossover of parent genomes
/// - `mutate_gaussian`: Gaussian mutation of genome values
/// - `evolution_step`: Complete evolution generation (all-in-one)
pub const EVOLUTION: &str = r#"
#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Structures
// =============================================================================

// Evolution parameters
struct EvolutionParams {
    uint population_size;
    uint genome_length;
    float mutation_rate;
    float mutation_strength;
    float crossover_rate;
    uint tournament_size;
    uint elitism_count;
    uint generation;
};

// Embedded neural network weights for fitness evaluation
// Architecture: genome_length -> 32 -> 16 -> 1
struct FitnessNetworkWeights {
    // Layer 1: input (64 max) -> hidden (32)
    // Stored as [hidden_idx * 64 + input_idx]
    float layer1_weights[64 * 32];
    float layer1_bias[32];

    // Layer 2: hidden (32) -> hidden (16)
    float layer2_weights[32 * 16];
    float layer2_bias[16];

    // Layer 3: hidden (16) -> output (1)
    float layer3_weights[16];
    float layer3_bias;
};

// Individual in the population
struct Individual {
    uint index;
    float fitness;
    uint parent1;
    uint parent2;
    uint age;
};

// =============================================================================
// Embedded Neural Network Evaluation
// =============================================================================

// Evaluate fitness using embedded neural network
// This is the key innovation - no GPU→CPU→GPU round-trips!
inline float evaluate_fitness_network(
    device const float* genome,
    uint genome_length,
    constant FitnessNetworkWeights& weights
) {
    // Layer 1: genome -> hidden1 (32 neurons)
    float hidden1[32];
    for (uint h = 0; h < 32; h++) {
        float sum = weights.layer1_bias[h];
        uint max_inputs = min(genome_length, 64u);
        for (uint i = 0; i < max_inputs; i++) {
            sum += genome[i] * weights.layer1_weights[h * 64 + i];
        }
        hidden1[h] = tanh(sum);
    }

    // Layer 2: hidden1 -> hidden2 (16 neurons)
    float hidden2[16];
    for (uint h = 0; h < 16; h++) {
        float sum = weights.layer2_bias[h];
        for (uint i = 0; i < 32; i++) {
            sum += hidden1[i] * weights.layer2_weights[h * 32 + i];
        }
        hidden2[h] = tanh(sum);
    }

    // Layer 3: hidden2 -> output (1 neuron)
    float output = weights.layer3_bias;
    for (uint i = 0; i < 16; i++) {
        output += hidden2[i] * weights.layer3_weights[i];
    }

    // Sigmoid activation for fitness in [0, 1]
    return 1.0f / (1.0f + exp(-output));
}

// =============================================================================
// Evolution Kernels
// =============================================================================

// Evaluate fitness for all individuals using embedded neural network
kernel void evaluate_fitness(
    device const float* genomes [[buffer(0)]],
    device float* fitness [[buffer(1)]],
    constant EvolutionParams& params [[buffer(2)]],
    constant FitnessNetworkWeights& weights [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.population_size) return;

    device const float* genome = genomes + tid * params.genome_length;
    fitness[tid] = evaluate_fitness_network(genome, params.genome_length, weights);
}

// Tournament selection - select parent indices
kernel void tournament_select(
    device const float* fitness [[buffer(0)]],
    device uint* selected_parents [[buffer(1)]],
    device uint4* rng_state [[buffer(2)]],
    constant EvolutionParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.population_size) return;

    uint4 state = rng_state[tid];

    // Run tournament
    uint best_idx = philox_uint(state, params.population_size);
    float best_fitness = fitness[best_idx];

    for (uint t = 1; t < params.tournament_size; t++) {
        uint candidate = philox_uint(state, params.population_size);
        float candidate_fitness = fitness[candidate];
        if (candidate_fitness > best_fitness) {
            best_idx = candidate;
            best_fitness = candidate_fitness;
        }
    }

    selected_parents[tid] = best_idx;
    rng_state[tid] = state;
}

// Uniform crossover between two parents
kernel void crossover_uniform(
    device const float* parent_genomes [[buffer(0)]],
    device float* child_genomes [[buffer(1)]],
    device const uint* parent1_indices [[buffer(2)]],
    device const uint* parent2_indices [[buffer(3)]],
    device uint4* rng_state [[buffer(4)]],
    constant EvolutionParams& params [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.population_size) return;

    // Skip elite individuals (they're preserved unchanged)
    if (tid < params.elitism_count) {
        // Copy elite individual unchanged
        device const float* src = parent_genomes + tid * params.genome_length;
        device float* dst = child_genomes + tid * params.genome_length;
        for (uint i = 0; i < params.genome_length; i++) {
            dst[i] = src[i];
        }
        return;
    }

    uint4 state = rng_state[tid];

    uint p1 = parent1_indices[tid];
    uint p2 = parent2_indices[tid];

    device const float* parent1 = parent_genomes + p1 * params.genome_length;
    device const float* parent2 = parent_genomes + p2 * params.genome_length;
    device float* child = child_genomes + tid * params.genome_length;

    // Decide whether to do crossover
    if (philox_uniform(state) < params.crossover_rate) {
        // Uniform crossover
        for (uint i = 0; i < params.genome_length; i++) {
            child[i] = (philox_uniform(state) < 0.5f) ? parent1[i] : parent2[i];
        }
    } else {
        // No crossover - copy parent1
        for (uint i = 0; i < params.genome_length; i++) {
            child[i] = parent1[i];
        }
    }

    rng_state[tid] = state;
}

// Gaussian mutation
kernel void mutate_gaussian(
    device float* genomes [[buffer(0)]],
    device uint4* rng_state [[buffer(1)]],
    constant EvolutionParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.population_size) return;

    // Skip elite individuals
    if (tid < params.elitism_count) return;

    uint4 state = rng_state[tid];
    device float* genome = genomes + tid * params.genome_length;

    for (uint i = 0; i < params.genome_length; i++) {
        if (philox_uniform(state) < params.mutation_rate) {
            // Gaussian mutation
            genome[i] += philox_normal(state) * params.mutation_strength;
            // Clamp to [-1, 1] range
            genome[i] = clamp(genome[i], -1.0f, 1.0f);
        }
    }

    rng_state[tid] = state;
}

// Combined evolution step (all-in-one for efficiency)
// This kernel performs: selection + crossover + mutation + fitness evaluation
kernel void evolution_step(
    device float* genomes [[buffer(0)]],
    device float* new_genomes [[buffer(1)]],
    device float* fitness [[buffer(2)]],
    device uint4* rng_state [[buffer(3)]],
    constant EvolutionParams& params [[buffer(4)]],
    constant FitnessNetworkWeights& weights [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.population_size) return;

    uint4 state = rng_state[tid];

    // Elite individuals are preserved
    if (tid < params.elitism_count) {
        device const float* src = genomes + tid * params.genome_length;
        device float* dst = new_genomes + tid * params.genome_length;
        for (uint i = 0; i < params.genome_length; i++) {
            dst[i] = src[i];
        }
    } else {
        // Tournament selection for parent 1
        uint parent1 = philox_uint(state, params.population_size);
        float best_fitness = fitness[parent1];
        for (uint t = 1; t < params.tournament_size; t++) {
            uint candidate = philox_uint(state, params.population_size);
            if (fitness[candidate] > best_fitness) {
                parent1 = candidate;
                best_fitness = fitness[candidate];
            }
        }

        // Tournament selection for parent 2
        uint parent2 = philox_uint(state, params.population_size);
        best_fitness = fitness[parent2];
        for (uint t = 1; t < params.tournament_size; t++) {
            uint candidate = philox_uint(state, params.population_size);
            if (fitness[candidate] > best_fitness) {
                parent2 = candidate;
                best_fitness = fitness[candidate];
            }
        }

        device const float* p1 = genomes + parent1 * params.genome_length;
        device const float* p2 = genomes + parent2 * params.genome_length;
        device float* child = new_genomes + tid * params.genome_length;

        // Crossover + Mutation
        bool do_crossover = philox_uniform(state) < params.crossover_rate;
        for (uint i = 0; i < params.genome_length; i++) {
            // Crossover
            float gene = do_crossover ?
                ((philox_uniform(state) < 0.5f) ? p1[i] : p2[i]) : p1[i];

            // Mutation
            if (philox_uniform(state) < params.mutation_rate) {
                gene += philox_normal(state) * params.mutation_strength;
                gene = clamp(gene, -1.0f, 1.0f);
            }

            child[i] = gene;
        }
    }

    rng_state[tid] = state;

    // Evaluate fitness of new individual
    device const float* my_genome = new_genomes + tid * params.genome_length;
    fitness[tid] = evaluate_fitness_network(my_genome, params.genome_length, weights);
}
"#;

/// ADAS (Adaptive Dynamic Agent Synthesis) shader.
///
/// ADAS uses embedded neural networks to score agent code complexity
/// and adaptively adjust synthesis parameters.
pub const ADAS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ADAS parameters
struct AdasParams {
    uint population_size;
    uint code_length;
    float complexity_weight;
    float novelty_weight;
    float performance_weight;
    float adaptation_rate;
    uint history_length;
};

// Complexity scoring network weights
struct ComplexityNetworkWeights {
    float layer1_weights[128 * 32];
    float layer1_bias[32];
    float layer2_weights[32 * 8];
    float layer2_bias[8];
    float layer3_weights[8];
    float layer3_bias;
};

// Evaluate code complexity using embedded network
inline float evaluate_complexity(
    device const float* code_features,
    uint feature_length,
    constant ComplexityNetworkWeights& weights
) {
    // Layer 1
    float hidden1[32];
    for (uint h = 0; h < 32; h++) {
        float sum = weights.layer1_bias[h];
        uint max_inputs = min(feature_length, 128u);
        for (uint i = 0; i < max_inputs; i++) {
            sum += code_features[i] * weights.layer1_weights[h * 128 + i];
        }
        hidden1[h] = max(0.0f, sum);  // ReLU
    }

    // Layer 2
    float hidden2[8];
    for (uint h = 0; h < 8; h++) {
        float sum = weights.layer2_bias[h];
        for (uint i = 0; i < 32; i++) {
            sum += hidden1[i] * weights.layer2_weights[h * 32 + i];
        }
        hidden2[h] = max(0.0f, sum);  // ReLU
    }

    // Output
    float output = weights.layer3_bias;
    for (uint i = 0; i < 8; i++) {
        output += hidden2[i] * weights.layer3_weights[i];
    }

    return 1.0f / (1.0f + exp(-output));  // Sigmoid
}

// ADAS scoring kernel
kernel void adas_score(
    device const float* code_features [[buffer(0)]],
    device const float* performance_scores [[buffer(1)]],
    device const float* novelty_scores [[buffer(2)]],
    device float* combined_scores [[buffer(3)]],
    constant AdasParams& params [[buffer(4)]],
    constant ComplexityNetworkWeights& weights [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.population_size) return;

    device const float* features = code_features + tid * params.code_length;
    float complexity = evaluate_complexity(features, params.code_length, weights);

    // Combined score: balance performance, novelty, and simplicity (1 - complexity)
    float score = params.performance_weight * performance_scores[tid]
                + params.novelty_weight * novelty_scores[tid]
                + params.complexity_weight * (1.0f - complexity);

    combined_scores[tid] = score;
}

// ADAS adaptation kernel - adjusts synthesis parameters based on population stats
kernel void adas_adapt(
    device const float* scores [[buffer(0)]],
    device float* synthesis_params [[buffer(1)]],
    constant AdasParams& params [[buffer(2)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    // Compute population statistics (mean, variance)
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    uint count = 0;

    for (uint i = tid; i < params.population_size; i += group_size) {
        float s = scores[i];
        local_sum += s;
        local_sq_sum += s * s;
        count++;
    }

    // Reduce in threadgroup
    shared[lid] = local_sum;
    shared[lid + group_size] = local_sq_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
        float total_sum = 0.0f;
        float total_sq_sum = 0.0f;
        for (uint i = 0; i < group_size; i++) {
            total_sum += shared[i];
            total_sq_sum += shared[i + group_size];
        }

        float mean = total_sum / float(params.population_size);
        float variance = (total_sq_sum / float(params.population_size)) - (mean * mean);

        // Adapt synthesis parameters based on statistics
        // If variance is low, increase exploration (mutation)
        // If mean is low, increase exploitation (selection pressure)
        synthesis_params[0] = synthesis_params[0] * (1.0f - params.adaptation_rate)
                            + (0.5f - variance) * params.adaptation_rate;  // mutation rate
        synthesis_params[1] = synthesis_params[1] * (1.0f - params.adaptation_rate)
                            + mean * params.adaptation_rate;  // selection pressure
    }
}
"#;

/// DGM (Dynamic Genome Modification) shader.
///
/// DGM uses embedded neural networks to decide when and how to modify
/// genome structure during evolution.
pub const DGM: &str = r#"
#include <metal_stdlib>
using namespace metal;

// DGM parameters
struct DgmParams {
    uint population_size;
    uint max_genome_length;
    uint min_genome_length;
    float growth_probability;
    float shrink_probability;
    float modification_threshold;
};

// Modification decision network
struct ModificationNetworkWeights {
    float layer1_weights[64 * 16];
    float layer1_bias[16];
    float layer2_weights[16 * 4];
    float layer2_bias[4];  // [grow, shrink, mutate, keep]
};

// Decide modification type using embedded network
inline uint4 decide_modification(
    device const float* genome,
    uint genome_length,
    float fitness,
    float fitness_delta,
    constant ModificationNetworkWeights& weights
) {
    // Prepare input features: genome stats + fitness info
    float features[64];
    for (uint i = 0; i < 64; i++) {
        if (i < genome_length) {
            features[i] = genome[i];
        } else if (i == 60) {
            features[i] = float(genome_length) / 64.0f;  // normalized length
        } else if (i == 61) {
            features[i] = fitness;
        } else if (i == 62) {
            features[i] = fitness_delta;
        } else {
            features[i] = 0.0f;
        }
    }

    // Layer 1
    float hidden[16];
    for (uint h = 0; h < 16; h++) {
        float sum = weights.layer1_bias[h];
        for (uint i = 0; i < 64; i++) {
            sum += features[i] * weights.layer1_weights[h * 64 + i];
        }
        hidden[h] = tanh(sum);
    }

    // Layer 2 (4 outputs)
    float outputs[4];
    for (uint o = 0; o < 4; o++) {
        float sum = weights.layer2_bias[o];
        for (uint i = 0; i < 16; i++) {
            sum += hidden[i] * weights.layer2_weights[o * 16 + i];
        }
        outputs[o] = exp(sum);  // softmax numerator
    }

    // Softmax
    float total = outputs[0] + outputs[1] + outputs[2] + outputs[3];
    for (uint o = 0; o < 4; o++) {
        outputs[o] /= total;
    }

    // Return probabilities as uint4 (scaled by 1000)
    return uint4(
        uint(outputs[0] * 1000.0f),  // grow
        uint(outputs[1] * 1000.0f),  // shrink
        uint(outputs[2] * 1000.0f),  // mutate
        uint(outputs[3] * 1000.0f)   // keep
    );
}

// DGM modification kernel
kernel void dgm_modify(
    device float* genomes [[buffer(0)]],
    device uint* genome_lengths [[buffer(1)]],
    device const float* fitness [[buffer(2)]],
    device const float* prev_fitness [[buffer(3)]],
    device uint4* rng_state [[buffer(4)]],
    constant DgmParams& params [[buffer(5)]],
    constant ModificationNetworkWeights& weights [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.population_size) return;

    uint4 state = rng_state[tid];
    uint current_length = genome_lengths[tid];
    device float* genome = genomes + tid * params.max_genome_length;

    float current_fitness = fitness[tid];
    float prev_fit = prev_fitness[tid];
    float fitness_delta = current_fitness - prev_fit;

    // Get modification decision from network
    uint4 probs = decide_modification(genome, current_length, current_fitness,
                                       fitness_delta, weights);

    uint random = philox_uint(state, 1000);

    if (random < probs.x && current_length < params.max_genome_length) {
        // Grow: add new gene
        genome[current_length] = philox_uniform(state) * 2.0f - 1.0f;
        genome_lengths[tid] = current_length + 1;
    } else if (random < probs.x + probs.y && current_length > params.min_genome_length) {
        // Shrink: remove last gene
        genome_lengths[tid] = current_length - 1;
    } else if (random < probs.x + probs.y + probs.z) {
        // Mutate: modify random gene
        uint gene_idx = philox_uint(state, current_length);
        genome[gene_idx] += philox_normal(state) * 0.1f;
        genome[gene_idx] = clamp(genome[gene_idx], -1.0f, 1.0f);
    }
    // else: keep unchanged

    rng_state[tid] = state;
}
"#;

/// PSO (Particle Swarm Optimization) shader.
///
/// Standard PSO with optional neural network-based adaptive parameters.
pub const PSO: &str = r#"
#include <metal_stdlib>
using namespace metal;

// PSO parameters
struct PsoParams {
    uint swarm_size;
    uint dimensions;
    float inertia_weight;
    float cognitive_weight;  // c1
    float social_weight;     // c2
    float max_velocity;
    float3 bounds_min;
    float3 bounds_max;
};

// Particle state
struct Particle {
    float position[64];
    float velocity[64];
    float personal_best[64];
    float personal_best_fitness;
    float current_fitness;
};

// PSO velocity update kernel
kernel void pso_update_velocity(
    device Particle* particles [[buffer(0)]],
    device const float* global_best [[buffer(1)]],
    device uint4* rng_state [[buffer(2)]],
    constant PsoParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.swarm_size) return;

    uint4 state = rng_state[tid];
    device Particle& p = particles[tid];

    for (uint d = 0; d < params.dimensions; d++) {
        float r1 = philox_uniform(state);
        float r2 = philox_uniform(state);

        // Standard PSO velocity update
        float cognitive = params.cognitive_weight * r1 *
                         (p.personal_best[d] - p.position[d]);
        float social = params.social_weight * r2 *
                      (global_best[d] - p.position[d]);

        p.velocity[d] = params.inertia_weight * p.velocity[d] + cognitive + social;

        // Clamp velocity
        p.velocity[d] = clamp(p.velocity[d], -params.max_velocity, params.max_velocity);
    }

    rng_state[tid] = state;
}

// PSO position update kernel
kernel void pso_update_position(
    device Particle* particles [[buffer(0)]],
    constant PsoParams& params [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.swarm_size) return;

    device Particle& p = particles[tid];

    for (uint d = 0; d < params.dimensions; d++) {
        p.position[d] += p.velocity[d];

        // Bounce off bounds
        if (p.position[d] < -1.0f) {
            p.position[d] = -1.0f;
            p.velocity[d] *= -0.5f;
        }
        if (p.position[d] > 1.0f) {
            p.position[d] = 1.0f;
            p.velocity[d] *= -0.5f;
        }
    }
}

// PSO fitness evaluation with embedded network
kernel void pso_evaluate_fitness(
    device Particle* particles [[buffer(0)]],
    device float* global_best [[buffer(1)]],
    device atomic_uint* global_best_fitness [[buffer(2)]],
    constant PsoParams& params [[buffer(3)]],
    constant FitnessNetworkWeights& weights [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.swarm_size) return;

    device Particle& p = particles[tid];

    // Evaluate fitness using embedded network
    float fitness = evaluate_fitness_network(p.position, params.dimensions, weights);
    p.current_fitness = fitness;

    // Update personal best
    if (fitness > p.personal_best_fitness) {
        p.personal_best_fitness = fitness;
        for (uint d = 0; d < params.dimensions; d++) {
            p.personal_best[d] = p.position[d];
        }

        // Try to update global best (atomic)
        uint fitness_uint = as_type<uint>(fitness);
        uint old_best = atomic_load_explicit(global_best_fitness, memory_order_relaxed);
        while (fitness_uint > old_best) {
            if (atomic_compare_exchange_weak_explicit(
                global_best_fitness, &old_best, fitness_uint,
                memory_order_relaxed, memory_order_relaxed)) {
                // We won - update global best position
                for (uint d = 0; d < params.dimensions; d++) {
                    global_best[d] = p.position[d];
                }
                break;
            }
        }
    }
}
"#;

/// Shader info for the evolution shader.
pub const EVOLUTION_INFO: ShaderInfo = ShaderInfo {
    name: "evolution",
    description: "Evolutionary algorithm with embedded neural network fitness evaluation",
    kernel_functions: &[
        "evaluate_fitness",
        "tournament_select",
        "crossover_uniform",
        "mutate_gaussian",
        "evolution_step",
    ],
    buffer_bindings: &[
        BufferBinding {
            index: 0,
            name: "genomes",
            description: "Population genomes [pop_size * genome_length]",
            read_only: false,
        },
        BufferBinding {
            index: 1,
            name: "fitness",
            description: "Fitness values [pop_size]",
            read_only: false,
        },
        BufferBinding {
            index: 2,
            name: "params",
            description: "EvolutionParams constant buffer",
            read_only: true,
        },
        BufferBinding {
            index: 3,
            name: "weights",
            description: "FitnessNetworkWeights for embedded neural network",
            read_only: true,
        },
    ],
};

/// Shader info for ADAS.
pub const ADAS_INFO: ShaderInfo = ShaderInfo {
    name: "adas",
    description: "Adaptive Dynamic Agent Synthesis with complexity scoring",
    kernel_functions: &["adas_score", "adas_adapt"],
    buffer_bindings: &[],
};

/// Shader info for DGM.
pub const DGM_INFO: ShaderInfo = ShaderInfo {
    name: "dgm",
    description: "Dynamic Genome Modification with neural decision network",
    kernel_functions: &["dgm_modify"],
    buffer_bindings: &[],
};

/// Shader info for PSO.
pub const PSO_INFO: ShaderInfo = ShaderInfo {
    name: "pso",
    description: "Particle Swarm Optimization with embedded fitness evaluation",
    kernel_functions: &["pso_update_velocity", "pso_update_position", "pso_evaluate_fitness"],
    buffer_bindings: &[],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evolution_shader_content() {
        assert!(EVOLUTION.contains("FitnessNetworkWeights"));
        assert!(EVOLUTION.contains("evaluate_fitness_network"));
        assert!(EVOLUTION.contains("evolution_step"));
        assert!(EVOLUTION.contains("tournament_select"));
        assert!(EVOLUTION.contains("crossover_uniform"));
        assert!(EVOLUTION.contains("mutate_gaussian"));
    }

    #[test]
    fn test_adas_shader_content() {
        assert!(ADAS.contains("ComplexityNetworkWeights"));
        assert!(ADAS.contains("evaluate_complexity"));
        assert!(ADAS.contains("adas_score"));
        assert!(ADAS.contains("adas_adapt"));
    }

    #[test]
    fn test_dgm_shader_content() {
        assert!(DGM.contains("ModificationNetworkWeights"));
        assert!(DGM.contains("decide_modification"));
        assert!(DGM.contains("dgm_modify"));
    }

    #[test]
    fn test_pso_shader_content() {
        assert!(PSO.contains("Particle"));
        assert!(PSO.contains("pso_update_velocity"));
        assert!(PSO.contains("pso_update_position"));
        assert!(PSO.contains("pso_evaluate_fitness"));
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_evolution_shader_compilation() {
        use stratoswarm_metal_core::metal3::{is_available, Metal3Backend};
        use stratoswarm_metal_core::backend::MetalBackend;

        if !is_available() {
            println!("Skipping test - Metal not available");
            return;
        }

        let backend = Metal3Backend::new().expect("Failed to create Metal backend");

        // Combine RNG + evolution shaders
        let source = format!(
            "{}\n{}\n{}",
            crate::common::RNG,
            crate::common::ATOMICS,
            EVOLUTION
        );

        let result = backend.create_compute_pipeline(&source, "evaluate_fitness");
        assert!(result.is_ok(), "Failed to compile evolution shader: {:?}", result.err());

        let result = backend.create_compute_pipeline(&source, "evolution_step");
        assert!(result.is_ok(), "Failed to compile evolution_step: {:?}", result.err());
    }
}
