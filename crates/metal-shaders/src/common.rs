//! Common Metal shader utilities.
//!
//! These shaders provide fundamental operations used across all domain-specific shaders:
//! - Philox RNG for high-quality random number generation
//! - Atomic float operations for lock-free aggregation
//! - Math utilities for common calculations

/// Philox random number generator implementation.
///
/// Philox is a counter-based RNG that produces high-quality random numbers
/// suitable for Monte Carlo simulations and evolutionary algorithms.
///
/// # Usage in Shaders
///
/// ```metal
/// uint4 state = uint4(thread_id, seed, 0, 0);
/// float random = philox_uniform(state);
/// ```
pub const RNG: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Philox 4x32 round function
inline uint4 philox4x32_round(uint4 counter, uint2 key) {
    uint hi0 = mulhi(0xD2511F53u, counter.x);
    uint lo0 = 0xD2511F53u * counter.x;
    uint hi1 = mulhi(0xCD9E8D57u, counter.z);
    uint lo1 = 0xCD9E8D57u * counter.z;

    return uint4(
        hi1 ^ counter.y ^ key.x,
        lo1,
        hi0 ^ counter.w ^ key.y,
        lo0
    );
}

// Full Philox 4x32 with 10 rounds
inline uint4 philox4x32(uint4 counter, uint2 key) {
    for (int i = 0; i < 10; i++) {
        counter = philox4x32_round(counter, key);
        key.x += 0x9E3779B9u;
        key.y += 0xBB67AE85u;
    }
    return counter;
}

// Generate a uniform float in [0, 1)
inline float philox_uniform(thread uint4& state) {
    state = philox4x32(state, uint2(0x12345678u, 0x9ABCDEF0u));
    return float(state.x) / float(0xFFFFFFFFu);
}

// Generate a uniform float in [low, high)
inline float philox_uniform_range(thread uint4& state, float low, float high) {
    return low + philox_uniform(state) * (high - low);
}

// Generate a normal distribution using Box-Muller
inline float philox_normal(thread uint4& state) {
    float u1 = philox_uniform(state);
    float u2 = philox_uniform(state);
    // Avoid log(0)
    u1 = max(u1, 1e-10f);
    return sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI_F * u2);
}

// Generate a normal distribution with mean and stddev
inline float philox_normal_range(thread uint4& state, float mean, float stddev) {
    return mean + philox_normal(state) * stddev;
}

// Generate a random uint in [0, max)
inline uint philox_uint(thread uint4& state, uint max_val) {
    state = philox4x32(state, uint2(0x12345678u, 0x9ABCDEF0u));
    return state.x % max_val;
}
"#;

/// Atomic float operations using compare-and-swap.
///
/// Metal doesn't have native atomic float operations, so we implement them
/// using compare-and-swap loops on atomic_uint.
///
/// # Usage in Shaders
///
/// ```metal
/// device atomic_uint* counter;
/// atomic_add_float(counter, 1.5f);
/// ```
pub const ATOMICS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Atomic add for floats using CAS
inline void atomic_add_float(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    uint desired;
    do {
        desired = as_type<uint>(as_type<float>(expected) + val);
    } while (!atomic_compare_exchange_weak_explicit(
        addr, &expected, desired,
        memory_order_relaxed, memory_order_relaxed
    ));
}

// Atomic max for floats using CAS
inline float atomic_max_float(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    uint desired;
    float old_val;
    do {
        old_val = as_type<float>(expected);
        if (old_val >= val) return old_val;
        desired = as_type<uint>(val);
    } while (!atomic_compare_exchange_weak_explicit(
        addr, &expected, desired,
        memory_order_relaxed, memory_order_relaxed
    ));
    return old_val;
}

// Atomic min for floats using CAS
inline float atomic_min_float(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    uint desired;
    float old_val;
    do {
        old_val = as_type<float>(expected);
        if (old_val <= val) return old_val;
        desired = as_type<uint>(val);
    } while (!atomic_compare_exchange_weak_explicit(
        addr, &expected, desired,
        memory_order_relaxed, memory_order_relaxed
    ));
    return old_val;
}

// Atomic exchange for floats
inline float atomic_exchange_float(device atomic_uint* addr, float val) {
    uint old = atomic_exchange_explicit(addr, as_type<uint>(val), memory_order_relaxed);
    return as_type<float>(old);
}
"#;

/// Common math utilities.
///
/// Provides mathematical functions commonly used in StratoSwarm algorithms.
pub const MATH: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Sigmoid activation function
inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// ReLU activation function
inline float relu(float x) {
    return max(0.0f, x);
}

// Leaky ReLU activation function
inline float leaky_relu(float x, float alpha = 0.01f) {
    return x > 0.0f ? x : alpha * x;
}

// Softmax for a small fixed-size array (inline version)
template<uint N>
inline void softmax(thread float* arr) {
    float max_val = arr[0];
    for (uint i = 1; i < N; i++) {
        max_val = max(max_val, arr[i]);
    }

    float sum = 0.0f;
    for (uint i = 0; i < N; i++) {
        arr[i] = exp(arr[i] - max_val);
        sum += arr[i];
    }

    for (uint i = 0; i < N; i++) {
        arr[i] /= sum;
    }
}

// Cosine similarity between two vectors
inline float cosine_similarity(
    device const float* a,
    device const float* b,
    uint length
) {
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (uint i = 0; i < length; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    float norm = sqrt(norm_a) * sqrt(norm_b);
    return norm > 0.0f ? dot / norm : 0.0f;
}

// Euclidean distance between two vectors
inline float euclidean_distance(
    device const float* a,
    device const float* b,
    uint length
) {
    float sum = 0.0f;
    for (uint i = 0; i < length; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Dot product of two vectors
inline float dot_product(
    device const float* a,
    device const float* b,
    uint length
) {
    float sum = 0.0f;
    for (uint i = 0; i < length; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Vector addition: result = a + b
inline void vec_add(
    device float* result,
    device const float* a,
    device const float* b,
    uint length
) {
    for (uint i = 0; i < length; i++) {
        result[i] = a[i] + b[i];
    }
}

// Vector scaling: result = a * scale
inline void vec_scale(
    device float* result,
    device const float* a,
    float scale,
    uint length
) {
    for (uint i = 0; i < length; i++) {
        result[i] = a[i] * scale;
    }
}
"#;

/// Shader info for the RNG shader.
pub const RNG_INFO: super::ShaderInfo = super::ShaderInfo {
    name: "rng",
    description: "Philox counter-based random number generator",
    kernel_functions: &[],
    buffer_bindings: &[],
};

/// Shader info for the atomics shader.
pub const ATOMICS_INFO: super::ShaderInfo = super::ShaderInfo {
    name: "atomics",
    description: "Atomic float operations using compare-and-swap",
    kernel_functions: &[],
    buffer_bindings: &[],
};

/// Shader info for the math utilities shader.
pub const MATH_INFO: super::ShaderInfo = super::ShaderInfo {
    name: "math",
    description: "Common math utilities for neural networks and similarity",
    kernel_functions: &[],
    buffer_bindings: &[],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_shader_content() {
        assert!(RNG.contains("philox4x32"));
        assert!(RNG.contains("philox_uniform"));
        assert!(RNG.contains("philox_normal"));
    }

    #[test]
    fn test_atomics_shader_content() {
        assert!(ATOMICS.contains("atomic_add_float"));
        assert!(ATOMICS.contains("atomic_max_float"));
        assert!(ATOMICS.contains("atomic_min_float"));
    }

    #[test]
    fn test_math_shader_content() {
        assert!(MATH.contains("sigmoid"));
        assert!(MATH.contains("relu"));
        assert!(MATH.contains("cosine_similarity"));
    }
}
