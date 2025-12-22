#include "types.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

// LLM model dimensions
#define VOCAB_SIZE 32000
#define EMBEDDING_DIM 768
#define HIDDEN_DIM 2048
#define NUM_HEADS 12
#define NUM_LAYERS 6
#define MAX_SEQ_LEN 512

// LLM model structure on GPU
struct LLMModel {
    float* token_embeddings;     // [VOCAB_SIZE, EMBEDDING_DIM]
    float* attention_weights;    // [NUM_LAYERS, NUM_HEADS, EMBEDDING_DIM, EMBEDDING_DIM]
    float* ffn_weights;         // [NUM_LAYERS, HIDDEN_DIM, EMBEDDING_DIM]
    float* layer_norms;         // [NUM_LAYERS, EMBEDDING_DIM]
    float* output_projection;   // [EMBEDDING_DIM, VOCAB_SIZE]
};

// Agent prompt context
struct AgentPrompt {
    uint32_t agent_id;
    float position[3];
    float velocity[3];
    float fitness;
    uint32_t token_ids[MAX_SEQ_LEN];
    uint32_t seq_length;
};

// LLM inference result
struct LLMResponse {
    uint32_t agent_id;
    uint32_t response_tokens[MAX_SEQ_LEN];
    uint32_t response_length;
    float logits[VOCAB_SIZE];
    float confidence;
};

// Device function for attention computation
__device__ float compute_attention_score(
    const float* query, 
    const float* key, 
    int dim)
{
    float score = 0.0f;
    for (int i = 0; i < dim; i++) {
        score += query[i] * key[i];
    }
    return score / sqrtf((float)dim);
}

// Device function for layer normalization
__device__ void layer_norm(
    float* output,
    const float* input,
    const float* gamma,
    const float* beta,
    int dim)
{
    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) {
        mean += input[i];
    }
    mean /= dim;

    // Compute variance  
    float variance = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= dim;

    // Normalize
    float inv_std = rsqrtf(variance + 1e-8f);
    for (int i = 0; i < dim; i++) {
        output[i] = (input[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// Kernel for embedding lookup
__global__ void embedding_lookup_kernel(
    float* embeddings_out,        // [batch_size, seq_len, embedding_dim]
    const uint32_t* token_ids,    // [batch_size, seq_len]
    const float* embedding_table, // [vocab_size, embedding_dim]
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t embedding_dim)
{
    uint32_t batch_idx = blockIdx.x;
    uint32_t seq_idx = blockIdx.y;
    uint32_t embed_idx = threadIdx.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len || embed_idx >= embedding_dim) {
        return;
    }

    uint32_t token_id = token_ids[batch_idx * seq_len + seq_idx];
    if (token_id < VOCAB_SIZE) {
        embeddings_out[(batch_idx * seq_len + seq_idx) * embedding_dim + embed_idx] = 
            embedding_table[token_id * embedding_dim + embed_idx];
    }
}

// Kernel for multi-head attention
__global__ void multihead_attention_kernel(
    float* attention_out,         // [batch_size, seq_len, embedding_dim]
    const float* query,           // [batch_size, seq_len, embedding_dim]
    const float* key,             // [batch_size, seq_len, embedding_dim]
    const float* value,           // [batch_size, seq_len, embedding_dim]
    const float* attention_weights, // [num_heads, embedding_dim, embedding_dim]
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t embedding_dim,
    uint32_t num_heads)
{
    uint32_t batch_idx = blockIdx.x;
    uint32_t seq_idx = blockIdx.y;
    uint32_t head_idx = blockIdx.z;
    uint32_t dim_idx = threadIdx.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len || 
        head_idx >= num_heads || dim_idx >= embedding_dim / num_heads) {
        return;
    }

    uint32_t head_dim = embedding_dim / num_heads;
    uint32_t global_dim_idx = head_idx * head_dim + dim_idx;

    // Compute attention scores for this position
    float attention_score = 0.0f;
    for (uint32_t k = 0; k < seq_len; k++) {
        float q_val = query[(batch_idx * seq_len + seq_idx) * embedding_dim + global_dim_idx];
        float k_val = key[(batch_idx * seq_len + k) * embedding_dim + global_dim_idx];
        
        float score = compute_attention_score(&q_val, &k_val, 1);
        float v_val = value[(batch_idx * seq_len + k) * embedding_dim + global_dim_idx];
        
        attention_score += expf(score) * v_val;
    }

    attention_out[(batch_idx * seq_len + seq_idx) * embedding_dim + global_dim_idx] = attention_score;
}

// Kernel for feed-forward network
__global__ void feedforward_kernel(
    float* ffn_out,              // [batch_size, seq_len, embedding_dim]
    const float* input,          // [batch_size, seq_len, embedding_dim]
    const float* weights1,       // [embedding_dim, hidden_dim]
    const float* weights2,       // [hidden_dim, embedding_dim]
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t embedding_dim,
    uint32_t hidden_dim)
{
    uint32_t batch_idx = blockIdx.x;
    uint32_t seq_idx = blockIdx.y;
    uint32_t out_dim = threadIdx.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len || out_dim >= embedding_dim) {
        return;
    }

    // First linear layer with ReLU
    float hidden_val = 0.0f;
    for (uint32_t i = 0; i < embedding_dim; i++) {
        float input_val = input[(batch_idx * seq_len + seq_idx) * embedding_dim + i];
        hidden_val += input_val * weights1[i * hidden_dim + out_dim];
    }
    hidden_val = fmaxf(0.0f, hidden_val); // ReLU activation

    // Second linear layer
    float output_val = 0.0f;
    for (uint32_t i = 0; i < hidden_dim; i++) {
        output_val += hidden_val * weights2[i * embedding_dim + out_dim];
    }

    ffn_out[(batch_idx * seq_len + seq_idx) * embedding_dim + out_dim] = output_val;
}

// Kernel for generating responses from logits
__global__ void generate_response_kernel(
    LLMResponse* responses,       // [batch_size]
    const float* logits,         // [batch_size, vocab_size]
    curandState* rng_states,     // [batch_size]
    uint32_t batch_size,
    uint32_t max_tokens)
{
    uint32_t batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size) {
        return;
    }

    curandState local_state = rng_states[batch_idx];
    LLMResponse* response = &responses[batch_idx];

    // Generate tokens using sampling
    for (uint32_t token_pos = 0; token_pos < max_tokens && token_pos < MAX_SEQ_LEN; token_pos++) {
        // Find top-k logits for sampling
        float max_logit = -INFINITY;
        for (uint32_t v = 0; v < VOCAB_SIZE; v++) {
            float logit = logits[batch_idx * VOCAB_SIZE + v];
            if (logit > max_logit) {
                max_logit = logit;
            }
        }

        // Softmax sampling with temperature
        float temperature = 0.7f;
        float sum_exp = 0.0f;
        for (uint32_t v = 0; v < VOCAB_SIZE; v++) {
            float logit = logits[batch_idx * VOCAB_SIZE + v];
            sum_exp += expf((logit - max_logit) / temperature);
        }

        // Sample token
        float rand_val = curand_uniform(&local_state) * sum_exp;
        float cumsum = 0.0f;
        uint32_t selected_token = 0;
        
        for (uint32_t v = 0; v < VOCAB_SIZE; v++) {
            float logit = logits[batch_idx * VOCAB_SIZE + v];
            cumsum += expf((logit - max_logit) / temperature);
            if (cumsum >= rand_val) {
                selected_token = v;
                break;
            }
        }

        response->response_tokens[token_pos] = selected_token;
        
        // Stop if EOS token (assuming token 2 is EOS)
        if (selected_token == 2) {
            response->response_length = token_pos + 1;
            break;
        }
    }

    // Compute confidence as max logit  
    float max_logit = -INFINITY;
    for (uint32_t v = 0; v < VOCAB_SIZE; v++) {
        float logit = logits[batch_idx * VOCAB_SIZE + v];
        if (logit > max_logit) {
            max_logit = logit;
        }
    }
    response->confidence = max_logit;
    rng_states[batch_idx] = local_state;
}

// Main LLM inference kernel
__global__ void llm_inference_kernel(
    LLMResponse* responses,       // [batch_size]
    const AgentPrompt* prompts,   // [batch_size]  
    const LLMModel* model,
    curandState* rng_states,      // [batch_size]
    float* workspace,            // Temporary workspace
    uint32_t batch_size)
{
    uint32_t batch_idx = blockIdx.x;
    
    if (batch_idx >= batch_size) {
        return;
    }

    const AgentPrompt* prompt = &prompts[batch_idx];
    LLMResponse* response = &responses[batch_idx];
    
    response->agent_id = prompt->agent_id;
    
    // For now, generate a simple deterministic response based on agent state
    // In a full implementation, this would run the transformer layers
    
    // Simple heuristic response generation
    float fitness_score = prompt->fitness;
    
    if (fitness_score > 0.8f) {
        // High fitness: continue current strategy
        response->response_tokens[0] = 1001; // "continue"
        response->response_tokens[1] = 1002; // "current"  
        response->response_tokens[2] = 1003; // "strategy"
        response->response_length = 3;
        response->confidence = 0.9f;
    } else if (fitness_score > 0.5f) {
        // Medium fitness: explore nearby
        response->response_tokens[0] = 1004; // "explore" 
        response->response_tokens[1] = 1005; // "nearby"
        response->response_tokens[2] = 1006; // "area"
        response->response_length = 3;
        response->confidence = 0.7f;
    } else {
        // Low fitness: change strategy
        response->response_tokens[0] = 1007; // "change"
        response->response_tokens[1] = 1008; // "strategy"
        response->response_tokens[2] = 1009; // "completely"
        response->response_length = 3;
        response->confidence = 0.6f;
    }
}

extern "C"
{
    void launch_llm_inference(
        LLMResponse* responses,
        const AgentPrompt* prompts,
        const LLMModel* model,
        curandState* rng_states,
        float* workspace,
        uint32_t batch_size)
    {
        dim3 block_size(256);
        dim3 grid_size((batch_size + block_size.x - 1) / block_size.x);
        
        llm_inference_kernel<<<grid_size, block_size>>>(
            responses, prompts, model, rng_states, workspace, batch_size);
        
        cudaDeviceSynchronize();
    }

    void launch_llm_embedding_lookup(
        float* embeddings_out,
        const uint32_t* token_ids,
        const float* embedding_table,
        uint32_t batch_size,
        uint32_t seq_len,
        uint32_t embedding_dim)
    {
        dim3 block_size(256);
        dim3 grid_size(batch_size, seq_len);
        
        embedding_lookup_kernel<<<grid_size, block_size>>>(
            embeddings_out, token_ids, embedding_table, 
            batch_size, seq_len, embedding_dim);
        
        cudaDeviceSynchronize();
    }

    void launch_llm_attention(
        float* attention_out,
        const float* query,
        const float* key, 
        const float* value,
        const float* attention_weights,
        uint32_t batch_size,
        uint32_t seq_len,
        uint32_t embedding_dim,
        uint32_t num_heads)
    {
        dim3 block_size(256);
        dim3 grid_size(batch_size, seq_len, num_heads);
        
        multihead_attention_kernel<<<grid_size, block_size>>>(
            attention_out, query, key, value, attention_weights,
            batch_size, seq_len, embedding_dim, num_heads);
        
        cudaDeviceSynchronize();
    }
}