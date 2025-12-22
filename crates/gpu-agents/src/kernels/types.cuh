#ifndef GPU_AGENTS_TYPES_CUH
#define GPU_AGENTS_TYPES_CUH

#include <cuda_runtime.h>
#include <stdint.h>

// Agent memory configuration
#define MAX_MEMORY_ENTRIES 1024
#define EMBEDDING_SIZE 128
#define MESSAGE_SIZE 32
#define GENOME_SIZE 256
#define MAX_NEIGHBORS 8

// GPU Agent structure - 256 bytes aligned
struct GPUAgent
{
    // Position and movement (24 bytes)
    float3 position;
    float3 velocity;
    
    // Core state (16 bytes)
    float    fitness;
    uint32_t state;
    uint32_t agent_type;
    uint32_t swarm_id;
    
    // Memory offsets (16 bytes)
    uint32_t working_memory_offset;
    uint32_t episodic_memory_offset;
    uint32_t semantic_memory_offset;
    uint32_t genome_offset;
    
    // Communication (72 bytes)
    uint32_t neighbors[2];
    float    shared_data[16];
    
    // Padding to 256 bytes
    char padding[128];
};

static_assert(sizeof(GPUAgent) == 256, "GPUAgent must be 256 bytes");

// Working memory structure
struct GPUWorkingMemory
{
    uint32_t keys[MAX_MEMORY_ENTRIES];
    float    values[MAX_MEMORY_ENTRIES][EMBEDDING_SIZE];
    uint32_t timestamps[MAX_MEMORY_ENTRIES];
    uint32_t access_count[MAX_MEMORY_ENTRIES];
    uint32_t head;
};

// Message structure for CPU-GPU communication
struct CPUGPUMessage
{
    uint32_t msg_type;
    uint64_t sender_id;
    uint64_t target_id;
    float    payload[32];
};

// Swarm configuration
struct SwarmConfig
{
    uint32_t num_agents;
    uint32_t block_size;
    uint32_t evolution_interval;
    float    cohesion_weight;
    float    separation_weight;
    float    alignment_weight;
};

#endif // GPU_AGENTS_TYPES_CUH