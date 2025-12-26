//! Utility shaders.
//!
//! Various utility shaders for common operations.
//!
//! # Phase 5 Implementation
//!
//! This module will contain:
//! - String operations
//! - Huffman compression
//! - Data transformation utilities

/// String comparison and hashing utilities.
pub const STRING: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Simple string hash (FNV-1a variant)
inline uint string_hash(device const char* str, uint len) {
    uint hash = 2166136261u;
    for (uint i = 0; i < len; i++) {
        hash ^= uint(str[i]);
        hash *= 16777619u;
    }
    return hash;
}

// String comparison
inline int string_compare(
    device const char* a, uint len_a,
    device const char* b, uint len_b
) {
    uint min_len = min(len_a, len_b);
    for (uint i = 0; i < min_len; i++) {
        if (a[i] != b[i]) {
            return int(a[i]) - int(b[i]);
        }
    }
    return int(len_a) - int(len_b);
}

// Batch string hashing
kernel void batch_hash_strings(
    device const char* strings [[buffer(0)]],
    device const uint* offsets [[buffer(1)]],
    device const uint* lengths [[buffer(2)]],
    device uint* hashes [[buffer(3)]],
    constant uint& num_strings [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= num_strings) return;

    device const char* str = strings + offsets[tid];
    hashes[tid] = string_hash(str, lengths[tid]);
}
"#;

/// Huffman compression utilities placeholder.
pub const HUFFMAN: &str = r#"
// Huffman compression placeholder - Phase 5 implementation
"#;

/// Data reduction operations.
pub const REDUCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Parallel sum reduction
kernel void reduce_sum(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    threadgroup float* shared [[threadgroup(0)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    // Load into shared memory
    shared[lid] = (tid < n) ? input[tid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction in shared memory
    for (uint stride = group_size / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (lid == 0) {
        output[group_id] = shared[0];
    }
}

// Parallel max reduction
kernel void reduce_max(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device uint* indices [[buffer(2)]],
    threadgroup float* shared_vals [[threadgroup(0)]],
    threadgroup uint* shared_idx [[threadgroup(1)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    // Load into shared memory
    if (tid < n) {
        shared_vals[lid] = input[tid];
        shared_idx[lid] = tid;
    } else {
        shared_vals[lid] = -INFINITY;
        shared_idx[lid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = group_size / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            if (shared_vals[lid + stride] > shared_vals[lid]) {
                shared_vals[lid] = shared_vals[lid + stride];
                shared_idx[lid] = shared_idx[lid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (lid == 0) {
        output[group_id] = shared_vals[0];
        indices[group_id] = shared_idx[0];
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_shader_exists() {
        assert!(STRING.contains("string_hash"));
        assert!(STRING.contains("string_compare"));
    }

    #[test]
    fn test_reduce_shader_exists() {
        assert!(REDUCE.contains("reduce_sum"));
        assert!(REDUCE.contains("reduce_max"));
    }
}
