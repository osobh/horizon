//! Fixed GPU Pattern Matching Implementation
//! 
//! Corrected encoding for proper child index calculation

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use std::sync::Arc;
use std::collections::HashMap;
use anyhow::{Result, Context};
use crate::synthesis::{Pattern, AstNode, Match, NodeType};

const NODE_SIZE: usize = 52; // 4 + 4 + 4 + 40 bytes per node
const MAX_CHILDREN: usize = 10;

/// Fixed GPU Pattern Matcher
pub struct GpuPatternMatcherV2 {
    device: Arc<CudaDevice>,
    pattern_buffer: CudaSlice<u8>,
    ast_buffer: CudaSlice<u8>,
    match_buffer: CudaSlice<u32>,
    max_nodes: usize,
}

impl GpuPatternMatcherV2 {
    pub fn new(device: Arc<CudaDevice>, max_nodes: usize) -> Result<Self> {
        let buffer_size = max_nodes * NODE_SIZE;
        
        let pattern_buffer = unsafe { device.alloc::<u8>(buffer_size) }
            .context("Failed to allocate pattern buffer")?;
        let ast_buffer = unsafe { device.alloc::<u8>(buffer_size) }
            .context("Failed to allocate AST buffer")?;
        let match_buffer = unsafe { device.alloc::<u32>(max_nodes * 2) }
            .context("Failed to allocate match buffer")?;

        Ok(Self {
            device,
            pattern_buffer,
            ast_buffer,
            match_buffer,
            max_nodes,
        })
    }

    pub fn match_pattern(&self, pattern: &Pattern, ast: &AstNode) -> Result<Vec<Match>> {
        // First pass: count nodes
        let pattern_node_count = self.count_nodes(pattern);
        let ast_node_count = self.count_nodes_ast(ast);
        
        if pattern_node_count > self.max_nodes || ast_node_count > self.max_nodes {
            anyhow::bail!("Too many nodes: pattern={}, ast={}, max={}", 
                         pattern_node_count, ast_node_count, self.max_nodes);
        }
        
        // Second pass: encode with proper indexing
        let mut pattern_data = vec![0u8; pattern_node_count * NODE_SIZE];
        let mut pattern_index = 0;
        self.encode_pattern_fixed(&mut pattern_data, pattern, &mut pattern_index)?;
        
        let mut ast_data = vec![0u8; ast_node_count * NODE_SIZE];
        let mut ast_index = 0;
        self.encode_ast_fixed(&mut ast_data, ast, &mut ast_index)?;
        
        // Copy to GPU
        let mut pattern_slice = self.pattern_buffer.slice(0..pattern_data.len());
        self.device.htod_copy_into(pattern_data, &mut pattern_slice)?;
        
        let mut ast_slice = self.ast_buffer.slice(0..ast_data.len());
        self.device.htod_copy_into(ast_data, &mut ast_slice)?;
        
        // Clear match buffer
        let zeros = vec![0u32; ast_node_count * 2];
        let mut match_slice = self.match_buffer.slice(0..zeros.len());
        self.device.htod_copy_into(zeros, &mut match_slice)?;
        
        // Launch kernel
        unsafe {
            crate::synthesis::launch_match_patterns(
                *self.pattern_buffer.device_ptr() as *const u8,
                *self.ast_buffer.device_ptr() as *const u8,
                *self.match_buffer.device_ptr() as *mut u32,
                1, // Single pattern
                ast_node_count as u32,
            );
        }
        
        // Get results
        self.device.synchronize()?;
        
        let mut results = vec![0u32; ast_node_count * 2];
        self.device.dtoh_sync_copy_into(&match_slice, &mut results)?;
        
        // Extract matches
        let matches = self.extract_matches(&results, pattern, ast)?;
        Ok(matches)
    }
    
    fn count_nodes(&self, pattern: &Pattern) -> usize {
        1 + pattern.children.iter().map(|c| self.count_nodes(c)).sum::<usize>()
    }
    
    fn count_nodes_ast(&self, ast: &AstNode) -> usize {
        1 + ast.children.iter().map(|c| self.count_nodes_ast(c)).sum::<usize>()
    }
    
    fn encode_pattern_fixed(&self, buffer: &mut [u8], pattern: &Pattern, index: &mut usize) -> Result<usize> {
        let my_index = *index;
        *index += 1;
        
        let offset = my_index * NODE_SIZE;
        
        // Write node data
        self.write_u32(buffer, offset + 0, pattern.node_type as u32);
        self.write_u32(buffer, offset + 4, self.hash_value(&pattern.value));
        self.write_u32(buffer, offset + 8, pattern.children.len() as u32);
        
        // Reserve space for child indices
        let child_indices_offset = offset + 12;
        
        // Encode children and collect their indices
        let mut child_indices = vec![0u32; MAX_CHILDREN];
        for (i, child) in pattern.children.iter().enumerate().take(MAX_CHILDREN) {
            child_indices[i] = *index as u32;
            self.encode_pattern_fixed(buffer, child, index)?;
        }
        
        // Write child indices
        for (i, idx) in child_indices.iter().enumerate() {
            self.write_u32(buffer, child_indices_offset + i * 4, *idx);
        }
        
        Ok(my_index)
    }
    
    fn encode_ast_fixed(&self, buffer: &mut [u8], ast: &AstNode, index: &mut usize) -> Result<usize> {
        let my_index = *index;
        *index += 1;
        
        let offset = my_index * NODE_SIZE;
        
        // Write node data
        self.write_u32(buffer, offset + 0, ast.node_type as u32);
        self.write_u32(buffer, offset + 4, self.hash_value(&ast.value));
        self.write_u32(buffer, offset + 8, ast.children.len() as u32);
        
        // Reserve space for child indices
        let child_indices_offset = offset + 12;
        
        // Encode children and collect their indices
        let mut child_indices = vec![0u32; MAX_CHILDREN];
        for (i, child) in ast.children.iter().enumerate().take(MAX_CHILDREN) {
            child_indices[i] = *index as u32;
            self.encode_ast_fixed(buffer, child, index)?;
        }
        
        // Write child indices
        for (i, idx) in child_indices.iter().enumerate() {
            self.write_u32(buffer, child_indices_offset + i * 4, *idx);
        }
        
        Ok(my_index)
    }
    
    fn write_u32(&self, buffer: &mut [u8], offset: usize, value: u32) {
        buffer[offset..offset+4].copy_from_slice(&value.to_le_bytes());
    }
    
    fn hash_value(&self, value: &Option<String>) -> u32 {
        value.as_ref().map(|s| {
            let mut hash = 0u32;
            for byte in s.bytes() {
                hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
            }
            hash
        }).unwrap_or(0)
    }
    
    fn extract_matches(&self, results: &[u32], _pattern: &Pattern, _ast: &AstNode) -> Result<Vec<Match>> {
        let mut matches = Vec::new();
        
        for i in 0..results.len()/2 {
            let node_id = results[i * 2];
            let match_flag = results[i * 2 + 1];
            
            if match_flag != 0 {
                matches.push(Match {
                    node_id: node_id as usize,
                    bindings: HashMap::new(), // TODO: Extract actual bindings
                });
            }
        }
        
        Ok(matches)
    }
}