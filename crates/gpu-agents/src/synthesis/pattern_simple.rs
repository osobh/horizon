//! Simplified GPU Pattern Matching Implementation
//!
//! Dynamic buffer allocation for variable-sized patterns

use crate::synthesis::{AstNode, Match, NodeType, Pattern};
use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use std::collections::HashMap;
use std::sync::Arc;

const NODE_SIZE: usize = 52; // 4 + 4 + 4 + 40 bytes per node

/// Simplified GPU Pattern Matcher with dynamic allocation
pub struct SimpleGpuPatternMatcher {
    device: Arc<CudaDevice>,
}

impl SimpleGpuPatternMatcher {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    pub fn match_pattern(&self, pattern: &Pattern, ast: &AstNode) -> Result<Vec<Match>> {
        // Encode pattern and AST
        let pattern_data = self.encode_pattern(pattern)?;
        let ast_data = self.encode_ast(ast)?;

        let num_ast_nodes = ast_data.len() / NODE_SIZE;

        // Allocate GPU buffers with exact sizes
        let pattern_buffer = unsafe { self.device.alloc::<u8>(pattern_data.len()) }
            .context("Failed to allocate pattern buffer")?;

        let ast_buffer = unsafe { self.device.alloc::<u8>(ast_data.len()) }
            .context("Failed to allocate AST buffer")?;

        let match_buffer = self
            .device
            .alloc_zeros::<u32>(num_ast_nodes * 2)
            .context("Failed to allocate match buffer")?;

        // Copy to GPU
        self.device
            .htod_copy_into(pattern_data, &mut pattern_buffer.clone())?;
        self.device
            .htod_copy_into(ast_data, &mut ast_buffer.clone())?;

        // Launch kernel
        unsafe {
            crate::synthesis::launch_match_patterns(
                *pattern_buffer.device_ptr() as *const u8,
                *ast_buffer.device_ptr() as *const u8,
                *match_buffer.device_ptr() as *mut u32,
                1, // Single pattern
                num_ast_nodes as u32,
            );
        }

        // Get results
        self.device.synchronize()?;

        let mut results = vec![0u32; num_ast_nodes * 2];
        self.device
            .dtoh_sync_copy_into(&match_buffer, &mut results)?;

        // Extract matches
        let matches = self.extract_matches(&results)?;
        Ok(matches)
    }

    fn encode_pattern(&self, pattern: &Pattern) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        self.encode_node_simple(&mut buffer, pattern)?;
        // Pad to NODE_SIZE boundary
        while buffer.len() % NODE_SIZE != 0 {
            buffer.push(0);
        }
        Ok(buffer)
    }

    fn encode_ast(&self, ast: &AstNode) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        self.encode_ast_node_simple(&mut buffer, ast)?;
        // Pad to NODE_SIZE boundary
        while buffer.len() % NODE_SIZE != 0 {
            buffer.push(0);
        }
        Ok(buffer)
    }

    fn encode_node_simple(&self, buffer: &mut Vec<u8>, pattern: &Pattern) -> Result<()> {
        // Reserve space for this node
        let start = buffer.len();
        buffer.resize(start + NODE_SIZE, 0);

        // Write node data
        self.write_u32(buffer, start + 0, pattern.node_type as u32);
        self.write_u32(buffer, start + 4, self.hash_value(&pattern.value));
        self.write_u32(buffer, start + 8, pattern.children.len() as u32);

        // Write zeros for child indices initially
        for i in 0..10 {
            self.write_u32(buffer, start + 12 + i * 4, 0);
        }

        // Encode children and update indices
        for (i, child) in pattern.children.iter().enumerate().take(10) {
            let child_idx = buffer.len() / NODE_SIZE;
            self.write_u32(buffer, start + 12 + i * 4, child_idx as u32);
            self.encode_node_simple(buffer, child)?;
        }

        Ok(())
    }

    fn encode_ast_node_simple(&self, buffer: &mut Vec<u8>, ast: &AstNode) -> Result<()> {
        // Reserve space for this node
        let start = buffer.len();
        buffer.resize(start + NODE_SIZE, 0);

        // Write node data
        self.write_u32(buffer, start + 0, ast.node_type as u32);
        self.write_u32(buffer, start + 4, self.hash_value(&ast.value));
        self.write_u32(buffer, start + 8, ast.children.len() as u32);

        // Write zeros for child indices initially
        for i in 0..10 {
            self.write_u32(buffer, start + 12 + i * 4, 0);
        }

        // Encode children and update indices
        for (i, child) in ast.children.iter().enumerate().take(10) {
            let child_idx = buffer.len() / NODE_SIZE;
            self.write_u32(buffer, start + 12 + i * 4, child_idx as u32);
            self.encode_ast_node_simple(buffer, child)?;
        }

        Ok(())
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

    fn extract_matches(&self, results: &[u32]) -> Result<Vec<Match>> {
        let mut matches = Vec::new();

        for i in 0..results.len() / 2 {
            let node_id = results[i * 2];
            let match_flag = results[i * 2 + 1];

            if match_flag != 0 {
                matches.push(Match {
                    node_id: node_id as usize,
                    bindings: HashMap::new(),
                });
            }
        }

        Ok(matches)
    }
}
