/**
 * CUDA kernels for GPU-accelerated Huffman encoding and decoding
 * 
 * Provides high-performance parallel Huffman compression operations
 * optimized for streaming data processing.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

// Maximum tree depth for Huffman codes
#define MAX_HUFFMAN_DEPTH 32
#define MAX_ALPHABET_SIZE 256
#define WARP_SIZE 32
#define BLOCK_SIZE 256

// Huffman code structure for GPU
struct HuffmanCode {
    uint32_t bits;      // Packed bits (up to 32 bits)
    uint8_t length;     // Number of bits in the code
};

// Huffman tree node structure for GPU
struct HuffmanTreeNode {
    uint8_t symbol;     // Symbol (for leaf nodes)
    uint8_t is_leaf;    // 1 if leaf node, 0 if internal
    uint16_t left;      // Index of left child (0 if leaf)
    uint16_t right;     // Index of right child (0 if leaf)
};

/**
 * Parallel Huffman encoding kernel
 * 
 * Each thread processes one byte of input data and converts it to
 * its corresponding Huffman code using the precomputed code table.
 */
__global__ void huffman_encode_kernel(
    const uint8_t* input_data,
    uint8_t* output_data,
    const HuffmanCode* code_table,
    uint32_t input_length,
    uint32_t* output_bit_positions,
    uint32_t* total_output_bits
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    
    // Shared memory for bit accumulation within block
    __shared__ uint32_t block_bit_counts[BLOCK_SIZE];
    __shared__ uint32_t block_bit_offset;
    
    // Initialize shared memory
    block_bit_counts[threadIdx.x] = 0;
    __syncthreads();
    
    // First pass: count bits needed for each thread
    for (uint32_t i = tid; i < input_length; i += stride) {
        uint8_t symbol = input_data[i];
        HuffmanCode code = code_table[symbol];
        block_bit_counts[threadIdx.x] += code.length;
    }
    
    __syncthreads();
    
    // Compute prefix sum within block for bit positions
    uint32_t local_offset = 0;
    for (int i = 0; i < threadIdx.x; i++) {
        local_offset += block_bit_counts[i];
    }
    
    // First thread in block computes total bits for block
    if (threadIdx.x == 0) {
        uint32_t block_total = 0;
        for (int i = 0; i < blockDim.x; i++) {
            block_total += block_bit_counts[i];
        }
        block_bit_offset = atomicAdd(total_output_bits, block_total);
    }
    
    __syncthreads();
    
    // Second pass: write encoded bits to output
    uint32_t global_bit_offset = block_bit_offset + local_offset;
    
    for (uint32_t i = tid; i < input_length; i += stride) {
        uint8_t symbol = input_data[i];
        HuffmanCode code = code_table[symbol];
        
        // Write bits to output buffer
        write_bits_to_buffer(output_data, global_bit_offset, code.bits, code.length);
        global_bit_offset += code.length;
    }
}

/**
 * Parallel Huffman decoding kernel
 * 
 * Uses tree traversal to decode compressed bits back to symbols.
 * Each thread processes a portion of the bit stream.
 */
__global__ void huffman_decode_kernel(
    const uint8_t* input_data,
    uint8_t* output_data,
    const HuffmanTreeNode* tree_nodes,
    uint32_t root_index,
    uint32_t input_bit_length,
    uint32_t output_length,
    uint32_t* decode_progress
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    
    // Each thread processes a segment of the bit stream
    uint32_t bits_per_thread = (input_bit_length + stride - 1) / stride;
    uint32_t start_bit = tid * bits_per_thread;
    uint32_t end_bit = min(start_bit + bits_per_thread, input_bit_length);
    
    if (start_bit >= input_bit_length) return;
    
    // State for tree traversal
    uint16_t current_node = root_index;
    uint32_t output_pos = 0;
    
    // Decode bits in this thread's segment
    for (uint32_t bit_pos = start_bit; bit_pos < end_bit; bit_pos++) {
        // Read bit from input
        uint8_t bit = read_bit_from_buffer(input_data, bit_pos);
        
        // Traverse tree based on bit
        const HuffmanTreeNode& node = tree_nodes[current_node];
        if (!node.is_leaf) {
            current_node = bit ? node.right : node.left;
            
            // Check if we've reached a leaf
            if (tree_nodes[current_node].is_leaf) {
                // Found symbol - write to output
                uint32_t global_output_pos = atomicAdd(decode_progress, 1);
                if (global_output_pos < output_length) {
                    output_data[global_output_pos] = tree_nodes[current_node].symbol;
                }
                
                // Reset to root for next symbol
                current_node = root_index;
            }
        }
    }
}

/**
 * Optimized parallel frequency counting kernel
 * 
 * Counts character frequencies in input data for Huffman tree construction.
 * Uses shared memory and atomic operations for efficiency.
 */
__global__ void count_frequencies_kernel(
    const uint8_t* input_data,
    uint32_t input_length,
    uint32_t* frequencies
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    
    // Shared memory for local frequency counting
    __shared__ uint32_t local_frequencies[MAX_ALPHABET_SIZE];
    
    // Initialize shared memory
    if (threadIdx.x < MAX_ALPHABET_SIZE) {
        local_frequencies[threadIdx.x] = 0;
    }
    __syncthreads();
    
    // Count frequencies locally
    for (uint32_t i = tid; i < input_length; i += stride) {
        uint8_t symbol = input_data[i];
        atomicAdd(&local_frequencies[symbol], 1);
    }
    
    __syncthreads();
    
    // Add local counts to global frequencies
    if (threadIdx.x < MAX_ALPHABET_SIZE) {
        if (local_frequencies[threadIdx.x] > 0) {
            atomicAdd(&frequencies[threadIdx.x], local_frequencies[threadIdx.x]);
        }
    }
}

/**
 * Bit manipulation helper functions
 */
__device__ void write_bits_to_buffer(
    uint8_t* buffer,
    uint32_t bit_offset,
    uint32_t bits,
    uint8_t bit_count
) {
    for (uint8_t i = 0; i < bit_count; i++) {
        uint32_t global_bit_pos = bit_offset + i;
        uint32_t byte_pos = global_bit_pos / 8;
        uint32_t bit_pos_in_byte = global_bit_pos % 8;
        
        uint8_t bit = (bits >> (bit_count - 1 - i)) & 1;
        
        if (bit) {
            buffer[byte_pos] |= (1 << (7 - bit_pos_in_byte));
        } else {
            buffer[byte_pos] &= ~(1 << (7 - bit_pos_in_byte));
        }
    }
}

__device__ uint8_t read_bit_from_buffer(
    const uint8_t* buffer,
    uint32_t bit_offset
) {
    uint32_t byte_pos = bit_offset / 8;
    uint32_t bit_pos_in_byte = bit_offset % 8;
    
    return (buffer[byte_pos] >> (7 - bit_pos_in_byte)) & 1;
}

/**
 * Warp-level parallel bit packing kernel
 * 
 * Efficiently packs variable-length Huffman codes into output buffer
 * using warp-level primitives for better performance.
 */
__global__ void warp_parallel_bit_pack_kernel(
    const uint8_t* input_data,
    uint8_t* output_data,
    const HuffmanCode* code_table,
    uint32_t input_length,
    uint32_t* output_byte_count
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t warp_id = tid / WARP_SIZE;
    uint32_t lane_id = tid % WARP_SIZE;
    
    // Shared memory for warp-level coordination
    __shared__ uint32_t warp_bit_buffer[BLOCK_SIZE / WARP_SIZE];
    __shared__ uint8_t warp_bit_count[BLOCK_SIZE / WARP_SIZE];
    __shared__ uint32_t warp_output_pos[BLOCK_SIZE / WARP_SIZE];
    
    if (lane_id == 0) {
        warp_bit_buffer[warp_id] = 0;
        warp_bit_count[warp_id] = 0;
        warp_output_pos[warp_id] = 0;
    }
    
    __syncwarp();
    
    // Process input data in warps
    for (uint32_t i = tid; i < input_length; i += blockDim.x * gridDim.x) {
        uint8_t symbol = input_data[i];
        HuffmanCode code = code_table[symbol];
        
        // Pack bits efficiently using warp primitives
        uint32_t mask = __activemask();
        uint32_t prefix_sum = __popc(mask & ((1U << lane_id) - 1));
        
        // Accumulate bits in warp buffer
        if (lane_id == 0) {
            uint32_t total_bits = __popc(mask) * 8; // Approximate
            if (warp_bit_count[warp_id] + total_bits >= 32) {
                // Flush warp buffer
                uint32_t output_pos = atomicAdd(output_byte_count, 4);
                *((uint32_t*)(output_data + output_pos)) = warp_bit_buffer[warp_id];
                warp_bit_buffer[warp_id] = 0;
                warp_bit_count[warp_id] = 0;
            }
        }
        
        // Add code to warp buffer
        if (mask & (1U << lane_id)) {
            uint32_t shift = 32 - warp_bit_count[warp_id] - code.length;
            atomicOr(&warp_bit_buffer[warp_id], code.bits << shift);
            atomicAdd(&warp_bit_count[warp_id], code.length);
        }
        
        __syncwarp();
    }
    
    // Flush remaining bits
    if (lane_id == 0 && warp_bit_count[warp_id] > 0) {
        uint32_t bytes_needed = (warp_bit_count[warp_id] + 7) / 8;
        uint32_t output_pos = atomicAdd(output_byte_count, bytes_needed);
        
        for (uint32_t b = 0; b < bytes_needed; b++) {
            output_data[output_pos + b] = (warp_bit_buffer[warp_id] >> (24 - b * 8)) & 0xFF;
        }
    }
}

/**
 * Tree traversal optimization kernel
 * 
 * Optimizes decoding by using shared memory caching of frequently
 * accessed tree nodes and branch prediction.
 */
__global__ void optimized_huffman_decode_kernel(
    const uint8_t* input_data,
    uint8_t* output_data,
    const HuffmanTreeNode* tree_nodes,
    uint32_t root_index,
    uint32_t input_bit_length,
    uint32_t output_length
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Cache frequently used tree nodes in shared memory
    __shared__ HuffmanTreeNode cached_nodes[64];
    
    // Load root and first level nodes into cache
    if (threadIdx.x < 64 && threadIdx.x < MAX_ALPHABET_SIZE) {
        cached_nodes[threadIdx.x] = tree_nodes[threadIdx.x];
    }
    
    __syncthreads();
    
    // Process assigned portion of bit stream
    uint32_t bits_per_thread = (input_bit_length + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);
    uint32_t start_bit = tid * bits_per_thread;
    uint32_t end_bit = min(start_bit + bits_per_thread, input_bit_length);
    
    if (start_bit >= input_bit_length) return;
    
    uint16_t current_node = root_index;
    uint32_t output_count = 0;
    
    // Optimized decoding loop
    for (uint32_t bit_pos = start_bit; bit_pos < end_bit; bit_pos++) {
        uint8_t bit = read_bit_from_buffer(input_data, bit_pos);
        
        // Use cached nodes when possible
        const HuffmanTreeNode* node = (current_node < 64) ? 
            &cached_nodes[current_node] : &tree_nodes[current_node];
        
        if (!node->is_leaf) {
            current_node = bit ? node->right : node->left;
            
            // Check if reached leaf
            const HuffmanTreeNode* next_node = (current_node < 64) ?
                &cached_nodes[current_node] : &tree_nodes[current_node];
                
            if (next_node->is_leaf) {
                // Decode successful - output symbol
                if (output_count < output_length) {
                    output_data[start_bit / 8 + output_count] = next_node->symbol;
                    output_count++;
                }
                current_node = root_index;
            }
        }
    }
}

/**
 * Host interface functions for kernel launches
 */
extern "C" {
    /**
     * Launch Huffman encoding kernel
     */
    cudaError_t launch_huffman_encode(
        const uint8_t* d_input,
        uint8_t* d_output,
        const HuffmanCode* d_code_table,
        uint32_t input_length,
        uint32_t* d_output_bits,
        cudaStream_t stream
    ) {
        dim3 block_size(BLOCK_SIZE);
        dim3 grid_size((input_length + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        // Initialize output bit counter
        cudaMemsetAsync(d_output_bits, 0, sizeof(uint32_t), stream);
        
        huffman_encode_kernel<<<grid_size, block_size, 0, stream>>>(
            d_input, d_output, d_code_table, input_length, nullptr, d_output_bits
        );
        
        return cudaGetLastError();
    }
    
    /**
     * Launch Huffman decoding kernel
     */
    cudaError_t launch_huffman_decode(
        const uint8_t* d_input,
        uint8_t* d_output,
        const HuffmanTreeNode* d_tree_nodes,
        uint32_t root_index,
        uint32_t input_bit_length,
        uint32_t output_length,
        cudaStream_t stream
    ) {
        dim3 block_size(BLOCK_SIZE);
        dim3 grid_size((input_bit_length / 8 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        uint32_t* d_progress;
        cudaMallocAsync((void**)&d_progress, sizeof(uint32_t), stream);
        cudaMemsetAsync(d_progress, 0, sizeof(uint32_t), stream);
        
        huffman_decode_kernel<<<grid_size, block_size, 0, stream>>>(
            d_input, d_output, d_tree_nodes, root_index, 
            input_bit_length, output_length, d_progress
        );
        
        cudaFreeAsync(d_progress, stream);
        return cudaGetLastError();
    }
    
    /**
     * Launch frequency counting kernel
     */
    cudaError_t launch_count_frequencies(
        const uint8_t* d_input,
        uint32_t input_length,
        uint32_t* d_frequencies,
        cudaStream_t stream
    ) {
        dim3 block_size(BLOCK_SIZE);
        dim3 grid_size((input_length + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        // Initialize frequency array
        cudaMemsetAsync(d_frequencies, 0, MAX_ALPHABET_SIZE * sizeof(uint32_t), stream);
        
        count_frequencies_kernel<<<grid_size, block_size, 0, stream>>>(
            d_input, input_length, d_frequencies
        );
        
        return cudaGetLastError();
    }
    
    /**
     * Launch optimized warp-parallel encoding
     */
    cudaError_t launch_warp_parallel_encode(
        const uint8_t* d_input,
        uint8_t* d_output,
        const HuffmanCode* d_code_table,
        uint32_t input_length,
        uint32_t* d_output_bytes,
        cudaStream_t stream
    ) {
        dim3 block_size(BLOCK_SIZE);
        dim3 grid_size((input_length + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        cudaMemsetAsync(d_output_bytes, 0, sizeof(uint32_t), stream);
        
        warp_parallel_bit_pack_kernel<<<grid_size, block_size, 0, stream>>>(
            d_input, d_output, d_code_table, input_length, d_output_bytes
        );
        
        return cudaGetLastError();
    }
}

/**
 * Utility kernels for performance optimization
 */

/**
 * Memory coalescing optimization for large data transfers
 */
__global__ void coalesced_memory_copy_kernel(
    const uint8_t* src,
    uint8_t* dst,
    uint32_t length
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    
    // Use 4-byte aligned copies when possible
    if (tid * 4 < length) {
        uint32_t* src32 = (uint32_t*)src;
        uint32_t* dst32 = (uint32_t*)dst;
        
        for (uint32_t i = tid; i * 4 < length; i += stride) {
            dst32[i] = src32[i];
        }
    }
}

/**
 * Bit stream validation kernel
 */
__global__ void validate_huffman_stream_kernel(
    const uint8_t* encoded_data,
    const HuffmanTreeNode* tree_nodes,
    uint32_t root_index,
    uint32_t bit_length,
    uint32_t* validation_result
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid != 0) return; // Only first thread validates
    
    uint16_t current_node = root_index;
    uint32_t valid_symbols = 0;
    
    for (uint32_t bit_pos = 0; bit_pos < bit_length; bit_pos++) {
        uint8_t bit = read_bit_from_buffer(encoded_data, bit_pos);
        
        const HuffmanTreeNode& node = tree_nodes[current_node];
        if (!node.is_leaf) {
            current_node = bit ? node.right : node.left;
            
            if (tree_nodes[current_node].is_leaf) {
                valid_symbols++;
                current_node = root_index;
            }
        } else {
            // Invalid state - shouldn't reach leaf without traversal
            *validation_result = 0;
            return;
        }
    }
    
    *validation_result = valid_symbols;
}