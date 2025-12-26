// GPU kernels for string operations
// Provides high-performance string processing on CUDA hardware

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>

extern "C" {

// =============================================================================
// String Manipulation Kernels
// =============================================================================

/**
 * Convert strings to uppercase
 * Each thread processes one character
 */
__global__ void string_uppercase_kernel(
    const char* input_data,
    char* output_data,
    const uint32_t* string_offsets,
    const uint32_t* string_lengths,
    uint32_t num_strings
) {
    uint32_t string_id = blockIdx.x;
    uint32_t char_id = threadIdx.x;
    
    if (string_id >= num_strings) return;
    
    uint32_t string_length = string_lengths[string_id];
    if (char_id >= string_length) return;
    
    uint32_t offset = string_offsets[string_id];
    char c = input_data[offset + char_id];
    
    // Convert to uppercase
    if (c >= 'a' && c <= 'z') {
        c = c - 'a' + 'A';
    }
    
    output_data[offset + char_id] = c;
}

/**
 * Convert strings to lowercase
 * Each thread processes one character
 */
__global__ void string_lowercase_kernel(
    const char* input_data,
    char* output_data,
    const uint32_t* string_offsets,
    const uint32_t* string_lengths,
    uint32_t num_strings
) {
    uint32_t string_id = blockIdx.x;
    uint32_t char_id = threadIdx.x;
    
    if (string_id >= num_strings) return;
    
    uint32_t string_length = string_lengths[string_id];
    if (char_id >= string_length) return;
    
    uint32_t offset = string_offsets[string_id];
    char c = input_data[offset + char_id];
    
    // Convert to lowercase
    if (c >= 'A' && c <= 'Z') {
        c = c - 'A' + 'a';
    }
    
    output_data[offset + char_id] = c;
}

/**
 * Reverse strings
 * Each thread copies one character from end to beginning
 */
__global__ void string_reverse_kernel(
    const char* input_data,
    char* output_data,
    const uint32_t* string_offsets,
    const uint32_t* string_lengths,
    uint32_t num_strings
) {
    uint32_t string_id = blockIdx.x;
    uint32_t char_id = threadIdx.x;
    
    if (string_id >= num_strings) return;
    
    uint32_t string_length = string_lengths[string_id];
    if (char_id >= string_length) return;
    
    uint32_t input_offset = string_offsets[string_id];
    uint32_t output_offset = string_offsets[string_id];
    
    // Copy character from end to beginning
    char c = input_data[input_offset + (string_length - 1 - char_id)];
    output_data[output_offset + char_id] = c;
}

// =============================================================================
// Pattern Matching Kernels
// =============================================================================

/**
 * Pattern matching using parallel string comparison
 * Each block processes one string, threads cooperate to find pattern
 */
__global__ void string_pattern_match_kernel(
    const char* input_data,
    char* match_results,
    const uint32_t* string_offsets,
    const uint32_t* string_lengths,
    const char* pattern,
    uint32_t pattern_length,
    uint32_t num_strings
) {
    uint32_t string_id = blockIdx.x;
    uint32_t thread_id = threadIdx.x;
    
    if (string_id >= num_strings) return;
    
    uint32_t string_length = string_lengths[string_id];
    uint32_t string_offset = string_offsets[string_id];
    
    __shared__ bool found_match;
    
    if (thread_id == 0) {
        found_match = false;
    }
    __syncthreads();
    
    // Each thread checks a different starting position
    for (uint32_t start_pos = thread_id; 
         start_pos <= string_length - pattern_length && !found_match; 
         start_pos += blockDim.x) {
        
        bool local_match = true;
        
        // Check if pattern matches at this position
        for (uint32_t i = 0; i < pattern_length && local_match; i++) {
            if (input_data[string_offset + start_pos + i] != pattern[i]) {
                local_match = false;
            }
        }
        
        if (local_match) {
            found_match = true;
        }
    }
    
    __syncthreads();
    
    // Write result
    if (thread_id == 0) {
        match_results[string_id] = found_match ? 1 : 0;
    }
}

/**
 * String replacement kernel
 * Replace all occurrences of pattern with replacement
 */
__global__ void string_replace_kernel(
    const char* input_data,
    char* output_data,
    const uint32_t* string_offsets,
    const uint32_t* string_lengths,
    uint32_t* output_lengths,
    const char* pattern,
    uint32_t pattern_length,
    const char* replacement,
    uint32_t replacement_length,
    uint32_t num_strings
) {
    uint32_t string_id = blockIdx.x;
    
    if (string_id >= num_strings) return;
    
    uint32_t string_length = string_lengths[string_id];
    uint32_t input_offset = string_offsets[string_id];
    uint32_t output_offset = string_offsets[string_id];
    
    uint32_t input_pos = 0;
    uint32_t output_pos = 0;
    
    while (input_pos < string_length) {
        // Check if pattern matches at current position
        bool pattern_match = true;
        
        if (input_pos + pattern_length <= string_length) {
            for (uint32_t i = 0; i < pattern_length; i++) {
                if (input_data[input_offset + input_pos + i] != pattern[i]) {
                    pattern_match = false;
                    break;
                }
            }
        } else {
            pattern_match = false;
        }
        
        if (pattern_match) {
            // Copy replacement
            for (uint32_t i = 0; i < replacement_length; i++) {
                output_data[output_offset + output_pos + i] = replacement[i];
            }
            input_pos += pattern_length;
            output_pos += replacement_length;
        } else {
            // Copy original character
            output_data[output_offset + output_pos] = input_data[input_offset + input_pos];
            input_pos++;
            output_pos++;
        }
    }
    
    output_lengths[string_id] = output_pos;
}

// =============================================================================
// Advanced String Operations
// =============================================================================

/**
 * Parallel string length calculation
 */
__global__ void string_length_kernel(
    const char* string_data,
    const uint32_t* string_offsets,
    uint32_t* string_lengths,
    uint32_t num_strings,
    uint32_t max_string_length
) {
    uint32_t string_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (string_id >= num_strings) return;
    
    uint32_t offset = string_offsets[string_id];
    uint32_t length = 0;
    
    // Find null terminator or max length
    while (length < max_string_length && string_data[offset + length] != '\0') {
        length++;
    }
    
    string_lengths[string_id] = length;
}

/**
 * Parallel string comparison for sorting
 */
__device__ int string_compare(
    const char* str1, uint32_t len1,
    const char* str2, uint32_t len2
) {
    uint32_t min_len = min(len1, len2);
    
    for (uint32_t i = 0; i < min_len; i++) {
        if (str1[i] < str2[i]) return -1;
        if (str1[i] > str2[i]) return 1;
    }
    
    if (len1 < len2) return -1;
    if (len1 > len2) return 1;
    return 0;
}

/**
 * Bitonic sort for string arrays
 */
__global__ void string_bitonic_sort_kernel(
    char* string_data,
    uint32_t* string_offsets,
    uint32_t* string_lengths,
    uint32_t* string_indices,
    uint32_t num_strings,
    uint32_t stage,
    uint32_t step,
    bool ascending
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pair_distance = 1 << (step - 1);
    
    if (tid >= num_strings / 2) return;
    
    uint32_t left_id = ((tid / pair_distance) * pair_distance * 2) + (tid % pair_distance);
    uint32_t right_id = left_id + pair_distance;
    
    if (right_id >= num_strings) return;
    
    uint32_t left_idx = string_indices[left_id];
    uint32_t right_idx = string_indices[right_id];
    
    const char* left_str = string_data + string_offsets[left_idx];
    const char* right_str = string_data + string_offsets[right_idx];
    
    uint32_t left_len = string_lengths[left_idx];
    uint32_t right_len = string_lengths[right_idx];
    
    int cmp = string_compare(left_str, left_len, right_str, right_len);
    
    bool direction = ((tid / (1 << stage)) % 2) == 0;
    bool should_swap = (cmp > 0) == (direction == ascending);
    
    if (should_swap) {
        // Swap indices
        string_indices[left_id] = right_idx;
        string_indices[right_id] = left_idx;
    }
}

// =============================================================================
// Utility Kernels
// =============================================================================

/**
 * Pack strings into continuous buffer with metadata
 */
__global__ void pack_strings_kernel(
    const char** input_strings,
    const uint32_t* input_lengths,
    char* output_buffer,
    uint32_t* output_offsets,
    uint32_t num_strings
) {
    uint32_t string_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (string_id >= num_strings) return;
    
    uint32_t offset = output_offsets[string_id];
    uint32_t length = input_lengths[string_id];
    const char* input_str = input_strings[string_id];
    
    // Copy string data
    for (uint32_t i = 0; i < length; i++) {
        output_buffer[offset + i] = input_str[i];
    }
}

/**
 * Calculate prefix sum for string offsets
 */
__global__ void calculate_string_offsets_kernel(
    const uint32_t* string_lengths,
    uint32_t* string_offsets,
    uint32_t num_strings
) {
    // Use CUB for efficient prefix sum
    typedef cub::BlockScan<uint32_t, 256> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t length = (tid < num_strings) ? string_lengths[tid] : 0;
    uint32_t offset;
    
    BlockScan(temp_storage).ExclusiveSum(length, offset);
    
    if (tid < num_strings) {
        string_offsets[tid] = offset;
    }
}

// =============================================================================
// Kernel Launch Wrappers
// =============================================================================

/**
 * Launch uppercase conversion kernel
 */
void launch_string_uppercase_kernel(
    const char* input_data,
    char* output_data,
    const uint32_t* string_offsets,
    const uint32_t* string_lengths,
    uint32_t num_strings,
    uint32_t max_string_length,
    cudaStream_t stream
) {
    dim3 grid(num_strings);
    dim3 block(min(max_string_length, 1024U));
    
    string_uppercase_kernel<<<grid, block, 0, stream>>>(
        input_data, output_data, string_offsets, string_lengths, num_strings
    );
}

/**
 * Launch lowercase conversion kernel
 */
void launch_string_lowercase_kernel(
    const char* input_data,
    char* output_data,
    const uint32_t* string_offsets,
    const uint32_t* string_lengths,
    uint32_t num_strings,
    uint32_t max_string_length,
    cudaStream_t stream
) {
    dim3 grid(num_strings);
    dim3 block(min(max_string_length, 1024U));
    
    string_lowercase_kernel<<<grid, block, 0, stream>>>(
        input_data, output_data, string_offsets, string_lengths, num_strings
    );
}

/**
 * Launch string reverse kernel
 */
void launch_string_reverse_kernel(
    const char* input_data,
    char* output_data,
    const uint32_t* string_offsets,
    const uint32_t* string_lengths,
    uint32_t num_strings,
    uint32_t max_string_length,
    cudaStream_t stream
) {
    dim3 grid(num_strings);
    dim3 block(min(max_string_length, 1024U));
    
    string_reverse_kernel<<<grid, block, 0, stream>>>(
        input_data, output_data, string_offsets, string_lengths, num_strings
    );
}

/**
 * Launch pattern matching kernel
 */
void launch_string_pattern_match_kernel(
    const char* input_data,
    char* match_results,
    const uint32_t* string_offsets,
    const uint32_t* string_lengths,
    const char* pattern,
    uint32_t pattern_length,
    uint32_t num_strings,
    cudaStream_t stream
) {
    dim3 grid(num_strings);
    dim3 block(min(1024U, 256U)); // Use 256 threads per block for pattern matching
    
    string_pattern_match_kernel<<<grid, block, 0, stream>>>(
        input_data, match_results, string_offsets, string_lengths,
        pattern, pattern_length, num_strings
    );
}

/**
 * Launch string replacement kernel
 */
void launch_string_replace_kernel(
    const char* input_data,
    char* output_data,
    const uint32_t* string_offsets,
    const uint32_t* string_lengths,
    uint32_t* output_lengths,
    const char* pattern,
    uint32_t pattern_length,
    const char* replacement,
    uint32_t replacement_length,
    uint32_t num_strings,
    cudaStream_t stream
) {
    dim3 grid(num_strings);
    dim3 block(1); // Single thread per string for replacement
    
    string_replace_kernel<<<grid, block, 0, stream>>>(
        input_data, output_data, string_offsets, string_lengths, output_lengths,
        pattern, pattern_length, replacement, replacement_length, num_strings
    );
}

/**
 * Launch bitonic sort kernel
 */
void launch_string_bitonic_sort_kernel(
    char* string_data,
    uint32_t* string_offsets,
    uint32_t* string_lengths,
    uint32_t* string_indices,
    uint32_t num_strings,
    uint32_t stage,
    uint32_t step,
    bool ascending,
    cudaStream_t stream
) {
    uint32_t threads_per_block = 256;
    uint32_t blocks = (num_strings / 2 + threads_per_block - 1) / threads_per_block;
    
    string_bitonic_sort_kernel<<<blocks, threads_per_block, 0, stream>>>(
        string_data, string_offsets, string_lengths, string_indices,
        num_strings, stage, step, ascending
    );
}

} // extern "C"