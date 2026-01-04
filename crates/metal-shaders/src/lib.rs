//! Metal Shading Language shaders for StratoSwarm.
//!
//! This crate provides Metal shader sources for GPU-accelerated operations.
//! Shaders are embedded as string constants for runtime compilation.
//!
//! # Shader Categories
//!
//! - **Common**: RNG, atomics, math utilities used across all shaders
//! - **Agent**: Basic agent operations and swarm behavior
//! - **Evolution**: Evolutionary algorithms with embedded neural networks
//! - **Knowledge**: Knowledge graph operations with embedding networks
//! - **Consensus**: Voting and leader election
//! - **Util**: String operations, compression, etc.
//!
//! # Example
//!
//! ```ignore
//! use stratoswarm_metal_shaders::common;
//! use stratoswarm_metal_core::metal3::{Metal3Backend, Metal3ComputePipeline};
//!
//! let backend = Metal3Backend::new()?;
//! let shader_source = format!(
//!     "{}\n{}\n{}",
//!     common::RNG,
//!     common::ATOMICS,
//!     evolution::EVOLUTION
//! );
//! let pipeline = backend.create_compute_pipeline(&shader_source, "evolution_kernel")?;
//! ```

pub mod agent;
pub mod common;
pub mod consensus;
pub mod evolution;
pub mod knowledge;
pub mod util;

/// Re-export metal-core for convenience
pub use stratoswarm_metal_core as core;

/// Combine multiple shader sources into a single source string.
///
/// This is useful for including common utilities in domain-specific shaders.
pub fn combine_shaders(shaders: &[&str]) -> String {
    shaders.join("\n\n")
}

/// Shader metadata for documentation and introspection.
#[derive(Debug, Clone)]
pub struct ShaderInfo {
    /// Name of the shader.
    pub name: &'static str,
    /// Description of what the shader does.
    pub description: &'static str,
    /// Kernel function names defined in this shader.
    pub kernel_functions: &'static [&'static str],
    /// Buffer bindings used by this shader.
    pub buffer_bindings: &'static [BufferBinding],
}

/// Buffer binding information.
#[derive(Debug, Clone)]
pub struct BufferBinding {
    /// Buffer index.
    pub index: u32,
    /// Name of the binding.
    pub name: &'static str,
    /// Description of the data.
    pub description: &'static str,
    /// Whether this binding is read-only.
    pub read_only: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combine_shaders() {
        let combined = combine_shaders(&[common::RNG, common::ATOMICS]);

        assert!(combined.contains("philox"));
        assert!(combined.contains("atomic_add_float"));
    }

    #[test]
    fn test_shader_compilation() {
        use stratoswarm_metal_core::backend::MetalBackend;
        use stratoswarm_metal_core::metal3::{is_available, Metal3Backend, Metal3ComputePipeline};

        if !is_available() {
            println!("Skipping test - Metal not available");
            return;
        }

        let backend = Metal3Backend::new().expect("Failed to create Metal backend");

        // Test that common shaders + a simple kernel compiles
        let source = format!(
            "{}\n{}\n{}",
            common::RNG,
            common::ATOMICS,
            r#"
            kernel void test_kernel(
                device float* output [[buffer(0)]],
                uint tid [[thread_position_in_grid]]
            ) {
                uint4 state = uint4(tid, 0, 0, 0);
                output[tid] = philox_uniform(state);
            }
            "#
        );

        let result = backend.create_compute_pipeline(&source, "test_kernel");
        assert!(
            result.is_ok(),
            "Failed to compile shader: {:?}",
            result.err()
        );
    }
}
