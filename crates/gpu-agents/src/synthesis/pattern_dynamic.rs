//! Dynamic GPU Pattern Matching - Allocates buffers as needed
//!
//! Simple but functional pattern matching for throughput testing

use crate::synthesis::{AstNode, Match, NodeType, Pattern};
use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use std::collections::HashMap;
use std::sync::Arc;

const NODE_SIZE: usize = 64; // Aligned to 64 bytes

/// Dynamic GPU Pattern Matcher
#[derive(Clone)]
pub struct DynamicGpuPatternMatcher {
    device: Arc<CudaDevice>,
}

impl DynamicGpuPatternMatcher {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Match patterns against ASTs with dynamic allocation
    pub fn match_batch(&self, patterns: &[Pattern], asts: &[AstNode]) -> Result<Vec<Vec<Match>>> {
        // Encode data
        let pattern_data = self.encode_patterns(patterns)?;
        let ast_data = self.encode_asts(asts)?;

        let pattern_count = patterns.len();
        let node_count = ast_data.len() / NODE_SIZE;

        // Allocate exact-size GPU buffers
        let pattern_buffer = unsafe { self.device.alloc::<u8>(pattern_data.len()) }
            .context("Failed to allocate pattern buffer")?;

        let ast_buffer = unsafe { self.device.alloc::<u8>(ast_data.len()) }
            .context("Failed to allocate AST buffer")?;

        let match_buffer = self
            .device
            .alloc_zeros::<u32>(node_count * 2)
            .context("Failed to allocate match buffer")?;

        // Copy to GPU
        self.device
            .htod_copy_into(pattern_data, &mut pattern_buffer.clone())?;
        self.device
            .htod_copy_into(ast_data, &mut ast_buffer.clone())?;

        // Launch kernel
        unsafe {
            crate::synthesis::launch_match_patterns_fast(
                *pattern_buffer.device_ptr() as *const u8,
                *ast_buffer.device_ptr() as *const u8,
                *match_buffer.device_ptr() as *mut u32,
                pattern_count as u32,
                node_count as u32,
            );
        }

        self.device.synchronize()?;

        // Get results
        let mut results = vec![0u32; node_count * 2];
        self.device
            .dtoh_sync_copy_into(&match_buffer, &mut results)?;

        // Extract matches
        self.extract_matches(&results, patterns.len())
    }

    /// Simple pattern encoding
    fn encode_patterns(&self, patterns: &[Pattern]) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();

        for pattern in patterns {
            let start = buffer.len();
            buffer.resize(start + NODE_SIZE, 0);

            self.write_u32(&mut buffer, start + 0, pattern.node_type as u32);
            self.write_u32(&mut buffer, start + 4, self.hash_value(&pattern.value));
            self.write_u32(&mut buffer, start + 8, 0); // No children for simplicity
        }

        Ok(buffer)
    }

    /// Simple AST encoding
    fn encode_asts(&self, asts: &[AstNode]) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();

        for ast in asts {
            self.encode_ast_flat(&mut buffer, ast)?;
        }

        Ok(buffer)
    }

    fn encode_ast_flat(&self, buffer: &mut Vec<u8>, ast: &AstNode) -> Result<()> {
        let start = buffer.len();
        buffer.resize(start + NODE_SIZE, 0);

        self.write_u32(buffer, start + 0, ast.node_type as u32);
        self.write_u32(buffer, start + 4, self.hash_value(&ast.value));
        self.write_u32(buffer, start + 8, 0); // Ignore children for now

        // Encode children as separate nodes
        for child in &ast.children {
            self.encode_ast_flat(buffer, child)?;
        }

        Ok(())
    }

    /// Extract matches
    fn extract_matches(&self, results: &[u32], pattern_count: usize) -> Result<Vec<Vec<Match>>> {
        let mut matches = vec![vec![]; pattern_count];

        for i in 0..results.len() / 2 {
            let node_id = results[i * 2];
            let match_flags = results[i * 2 + 1];

            if match_flags != 0 {
                for p in 0..pattern_count.min(32) {
                    if match_flags & (1 << p) != 0 {
                        matches[p].push(Match {
                            node_id: node_id as usize,
                            bindings: HashMap::new(),
                        });
                    }
                }
            }
        }

        Ok(matches)
    }

    fn write_u32(&self, buffer: &mut [u8], offset: usize, value: u32) {
        buffer[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn hash_value(&self, value: &Option<String>) -> u32 {
        value
            .as_ref()
            .map(|s| {
                let mut hash = 0u32;
                for byte in s.bytes() {
                    hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
                }
                hash
            })
            .unwrap_or(0)
    }
}
