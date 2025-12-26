//! Agent and swarm behavior shaders.
//!
//! These shaders implement basic agent operations and flocking behavior
//! used in the StratoSwarm agent system.

/// Agent state structure and basic operations.
///
/// Defines the agent data layout and provides kernels for:
/// - Agent initialization
/// - Position and velocity updates
/// - State management
pub const AGENT: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Agent state structure
struct Agent {
    float3 position;
    float3 velocity;
    float energy;
    uint state;       // 0 = idle, 1 = active, 2 = computing, 3 = communicating
    uint agent_id;
    float fitness;
    uint genome_offset;  // Offset into genome buffer
    uint genome_length;  // Length of this agent's genome
};

// Agent parameters
struct AgentParams {
    uint num_agents;
    float max_speed;
    float max_force;
    float energy_decay;
    float min_energy;
    float3 bounds_min;
    float3 bounds_max;
};

// Initialize agents with random positions and velocities
kernel void init_agents(
    device Agent* agents [[buffer(0)]],
    device uint4* rng_state [[buffer(1)]],
    constant AgentParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_agents) return;

    uint4 state = rng_state[tid];

    // Random position within bounds
    float3 range = params.bounds_max - params.bounds_min;
    float3 pos;
    pos.x = params.bounds_min.x + philox_uniform(state) * range.x;
    pos.y = params.bounds_min.y + philox_uniform(state) * range.y;
    pos.z = params.bounds_min.z + philox_uniform(state) * range.z;

    // Random velocity
    float3 vel;
    vel.x = (philox_uniform(state) - 0.5f) * params.max_speed;
    vel.y = (philox_uniform(state) - 0.5f) * params.max_speed;
    vel.z = (philox_uniform(state) - 0.5f) * params.max_speed;

    agents[tid].position = pos;
    agents[tid].velocity = vel;
    agents[tid].energy = 1.0f;
    agents[tid].state = 1;  // Active
    agents[tid].agent_id = tid;
    agents[tid].fitness = 0.0f;

    rng_state[tid] = state;
}

// Update agent positions based on velocity
kernel void update_agents(
    device Agent* agents [[buffer(0)]],
    constant AgentParams& params [[buffer(1)]],
    constant float& dt [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_agents) return;

    Agent agent = agents[tid];

    // Skip inactive agents
    if (agent.state == 0) return;

    // Update position
    agent.position += agent.velocity * dt;

    // Clamp to bounds (wrap or bounce)
    for (int i = 0; i < 3; i++) {
        if (agent.position[i] < params.bounds_min[i]) {
            agent.position[i] = params.bounds_max[i] - (params.bounds_min[i] - agent.position[i]);
        }
        if (agent.position[i] > params.bounds_max[i]) {
            agent.position[i] = params.bounds_min[i] + (agent.position[i] - params.bounds_max[i]);
        }
    }

    // Energy decay
    agent.energy -= params.energy_decay * dt;
    if (agent.energy < params.min_energy) {
        agent.state = 0;  // Deactivate
    }

    agents[tid] = agent;
}

// Apply force to agent velocity
kernel void apply_force(
    device Agent* agents [[buffer(0)]],
    device const float3* forces [[buffer(1)]],
    constant AgentParams& params [[buffer(2)]],
    constant float& dt [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_agents) return;

    Agent agent = agents[tid];
    if (agent.state == 0) return;

    float3 force = forces[tid];

    // Limit force magnitude
    float force_mag = length(force);
    if (force_mag > params.max_force) {
        force = normalize(force) * params.max_force;
    }

    // Apply force to velocity
    agent.velocity += force * dt;

    // Limit speed
    float speed = length(agent.velocity);
    if (speed > params.max_speed) {
        agent.velocity = normalize(agent.velocity) * params.max_speed;
    }

    agents[tid] = agent;
}
"#;

/// Swarm flocking behavior shaders.
///
/// Implements Reynolds' flocking rules:
/// - Separation: Avoid crowding neighbors
/// - Alignment: Steer towards average heading of neighbors
/// - Cohesion: Steer towards average position of neighbors
pub const SWARM: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Flocking parameters
struct FlockParams {
    uint num_agents;
    float separation_radius;
    float alignment_radius;
    float cohesion_radius;
    float separation_weight;
    float alignment_weight;
    float cohesion_weight;
    float max_speed;
    float max_force;
};

// Calculate flocking forces for all agents
kernel void calculate_flocking_forces(
    device const Agent* agents [[buffer(0)]],
    device float3* forces [[buffer(1)]],
    constant FlockParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_agents) return;

    Agent self = agents[tid];
    if (self.state == 0) {
        forces[tid] = float3(0.0f);
        return;
    }

    float3 separation = float3(0.0f);
    float3 alignment = float3(0.0f);
    float3 cohesion = float3(0.0f);

    uint sep_count = 0;
    uint align_count = 0;
    uint coh_count = 0;

    // Check all other agents (can be optimized with spatial hashing)
    for (uint i = 0; i < params.num_agents; i++) {
        if (i == tid) continue;

        Agent other = agents[i];
        if (other.state == 0) continue;

        float3 diff = self.position - other.position;
        float dist = length(diff);

        // Separation
        if (dist < params.separation_radius && dist > 0.0f) {
            separation += normalize(diff) / dist;
            sep_count++;
        }

        // Alignment
        if (dist < params.alignment_radius) {
            alignment += other.velocity;
            align_count++;
        }

        // Cohesion
        if (dist < params.cohesion_radius) {
            cohesion += other.position;
            coh_count++;
        }
    }

    // Compute steering forces
    float3 total_force = float3(0.0f);

    if (sep_count > 0) {
        separation /= float(sep_count);
        separation = normalize(separation) * params.max_speed - self.velocity;
        float sep_mag = length(separation);
        if (sep_mag > params.max_force) {
            separation = normalize(separation) * params.max_force;
        }
        total_force += separation * params.separation_weight;
    }

    if (align_count > 0) {
        alignment /= float(align_count);
        alignment = normalize(alignment) * params.max_speed - self.velocity;
        float align_mag = length(alignment);
        if (align_mag > params.max_force) {
            alignment = normalize(alignment) * params.max_force;
        }
        total_force += alignment * params.alignment_weight;
    }

    if (coh_count > 0) {
        cohesion /= float(coh_count);
        float3 desired = cohesion - self.position;
        desired = normalize(desired) * params.max_speed - self.velocity;
        float coh_mag = length(desired);
        if (coh_mag > params.max_force) {
            desired = normalize(desired) * params.max_force;
        }
        total_force += desired * params.cohesion_weight;
    }

    forces[tid] = total_force;
}

// Simple spatial hash for neighbor lookup optimization
struct SpatialHashParams {
    uint num_agents;
    float cell_size;
    uint3 grid_size;
    float3 bounds_min;
};

// Compute spatial hash cell for an agent
kernel void compute_spatial_hash(
    device const Agent* agents [[buffer(0)]],
    device uint* cell_indices [[buffer(1)]],
    device uint* agent_indices [[buffer(2)]],
    constant SpatialHashParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_agents) return;

    float3 pos = agents[tid].position;
    float3 local_pos = pos - params.bounds_min;

    uint3 cell;
    cell.x = uint(local_pos.x / params.cell_size) % params.grid_size.x;
    cell.y = uint(local_pos.y / params.cell_size) % params.grid_size.y;
    cell.z = uint(local_pos.z / params.cell_size) % params.grid_size.z;

    uint cell_index = cell.x + cell.y * params.grid_size.x + cell.z * params.grid_size.x * params.grid_size.y;

    cell_indices[tid] = cell_index;
    agent_indices[tid] = tid;
}
"#;

/// Shader info for the agent shader.
pub const AGENT_INFO: super::ShaderInfo = super::ShaderInfo {
    name: "agent",
    description: "Basic agent state management and physics",
    kernel_functions: &["init_agents", "update_agents", "apply_force"],
    buffer_bindings: &[
        super::BufferBinding {
            index: 0,
            name: "agents",
            description: "Array of Agent structs",
            read_only: false,
        },
        super::BufferBinding {
            index: 1,
            name: "rng_state / forces",
            description: "RNG state for init, or forces for update",
            read_only: false,
        },
        super::BufferBinding {
            index: 2,
            name: "params",
            description: "AgentParams constant buffer",
            read_only: true,
        },
    ],
};

/// Shader info for the swarm shader.
pub const SWARM_INFO: super::ShaderInfo = super::ShaderInfo {
    name: "swarm",
    description: "Reynolds flocking behavior (separation, alignment, cohesion)",
    kernel_functions: &["calculate_flocking_forces", "compute_spatial_hash"],
    buffer_bindings: &[
        super::BufferBinding {
            index: 0,
            name: "agents",
            description: "Array of Agent structs",
            read_only: true,
        },
        super::BufferBinding {
            index: 1,
            name: "forces",
            description: "Output force vectors",
            read_only: false,
        },
        super::BufferBinding {
            index: 2,
            name: "params",
            description: "FlockParams constant buffer",
            read_only: true,
        },
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_shader_content() {
        assert!(AGENT.contains("struct Agent"));
        assert!(AGENT.contains("init_agents"));
        assert!(AGENT.contains("update_agents"));
        assert!(AGENT.contains("apply_force"));
    }

    #[test]
    fn test_swarm_shader_content() {
        assert!(SWARM.contains("FlockParams"));
        assert!(SWARM.contains("calculate_flocking_forces"));
        assert!(SWARM.contains("separation"));
        assert!(SWARM.contains("alignment"));
        assert!(SWARM.contains("cohesion"));
    }
}
