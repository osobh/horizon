//! GPU Pattern Matching Implementation
//!
//! Parallel pattern matching for AST nodes using GPU acceleration

use crate::synthesis::{AstNode, Match, NodeType, Pattern};
use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};
use std::collections::HashMap;
use std::sync::Arc;

/// GPU Pattern Matcher for parallel AST matching
pub struct GpuPatternMatcher {
    device: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    pattern_buffer: CudaSlice<u8>,
    ast_buffer: CudaSlice<u8>,
    match_buffer: CudaSlice<u32>,
    max_nodes: usize,
}

impl GpuPatternMatcher {
    /// Create a new GPU pattern matcher
    pub fn new(device: Arc<CudaContext>, max_nodes: usize) -> Result<Self> {
        let stream = device.default_stream();
        // Allocate GPU buffers
        // Each node encoded as: [type(4), value_hash(4), child_count(4), children_indices(4*10)]
        let node_size = 4 + 4 + 4 + 4 * 10; // 52 bytes per node
        let buffer_size = max_nodes * node_size;

        // SAFETY: alloc returns uninitialized memory. pattern_buffer will be written via
        // memcpy_htod in match_pattern() before the kernel reads it.
        let pattern_buffer = unsafe { stream.alloc::<u8>(buffer_size) }
            .context("Failed to allocate pattern buffer")?;
        // SAFETY: alloc returns uninitialized memory. ast_buffer will be written via
        // memcpy_htod in match_pattern() before the kernel reads it.
        let ast_buffer =
            unsafe { stream.alloc::<u8>(buffer_size) }.context("Failed to allocate AST buffer")?;
        // SAFETY: alloc returns uninitialized memory. match_buffer will be cleared to zeros
        // in match_pattern() before the kernel writes match results.
        let match_buffer = unsafe { stream.alloc::<u32>(max_nodes * 2) } // [node_id, match_flag]
            .context("Failed to allocate match buffer")?;

        Ok(Self {
            device,
            stream,
            pattern_buffer,
            ast_buffer,
            match_buffer,
            max_nodes,
        })
    }

    /// Match a pattern against an AST on GPU
    pub fn match_pattern(&self, pattern: &Pattern, ast: &AstNode) -> Result<Vec<Match>> {
        // Encode pattern and AST for GPU
        let pattern_data = self.encode_pattern(pattern)?;
        let ast_data = self.encode_ast(ast)?;

        // Get number of AST nodes before moving data
        let num_ast_nodes = ast_data.len() as u32 / 52; // 52 bytes per node

        // Copy to GPU
        self.stream
            .memcpy_htod(&pattern_data, &mut self.pattern_buffer.clone())?;
        self.stream
            .memcpy_htod(&ast_data, &mut self.ast_buffer.clone())?;

        // Clear match buffer
        let zeros = vec![0u32; self.max_nodes * 2];
        self.stream
            .memcpy_htod(&zeros, &mut self.match_buffer.clone())?;
        // SAFETY: All pointers are valid device pointers from CudaSlice allocations:
        // - pattern_buffer: populated via memcpy_htod above
        // - ast_buffer: populated via memcpy_htod above
        // - match_buffer: cleared to zeros above, kernel writes results
        // - num_ast_nodes calculated from actual encoded AST size
        unsafe {
            let (pattern_ptr, _guard1) = self.pattern_buffer.device_ptr(&self.stream);
            let (ast_ptr, _guard2) = self.ast_buffer.device_ptr(&self.stream);
            let (match_ptr, _guard3) = self.match_buffer.device_ptr(&self.stream);
            crate::synthesis::launch_match_patterns(
                pattern_ptr as *const u8,
                ast_ptr as *const u8,
                match_ptr as *mut u32,
                1,             // Single pattern for now
                num_ast_nodes, // Number of AST nodes
            );
        }

        // Synchronize and get results
        self.stream.synchronize()?;

        let results: Vec<u32> = self.stream.clone_dtoh(&self.match_buffer)?;

        // Extract matches and bindings
        let matches = self.extract_matches(&results, pattern, ast)?;

        Ok(matches)
    }

    /// Match pattern against multiple ASTs in parallel
    pub fn match_pattern_parallel(
        &self,
        pattern: &Pattern,
        ast_forest: &[AstNode],
    ) -> Result<Vec<Match>> {
        let mut all_matches = Vec::new();

        // Process in batches that fit in GPU memory
        for ast in ast_forest {
            let matches = self.match_pattern(pattern, ast)?;
            all_matches.extend(matches);
        }

        Ok(all_matches)
    }

    /// Encode pattern for GPU processing
    fn encode_pattern(&self, pattern: &Pattern) -> Result<Vec<u8>> {
        let mut encoded = Vec::new();
        self.encode_pattern_recursive(pattern, &mut encoded)?;
        Ok(encoded)
    }

    fn encode_pattern_recursive(&self, pattern: &Pattern, buffer: &mut Vec<u8>) -> Result<usize> {
        let start_pos = buffer.len();

        // Encode node type (4 bytes)
        buffer.extend_from_slice(&(pattern.node_type as u32).to_le_bytes());

        // Encode value hash (4 bytes)
        let value_hash = pattern
            .value
            .as_ref()
            .map(|v| self.hash_string(v))
            .unwrap_or(0);
        buffer.extend_from_slice(&value_hash.to_le_bytes());

        // Encode child count (4 bytes)
        buffer.extend_from_slice(&(pattern.children.len() as u32).to_le_bytes());

        // Encode children indices (4 bytes each, max 10)
        let mut child_indices = vec![0u32; 10];
        let mut child_positions = Vec::new();

        // First encode all children to get their positions
        for child in pattern.children.iter().take(10) {
            let child_pos = buffer.len() / 52; // Node index (52 bytes per node)
            child_positions.push(child_pos as u32);
            self.encode_pattern_recursive(child, buffer)?;
        }

        // Now fill the child indices array
        for (i, &pos) in child_positions.iter().enumerate() {
            child_indices[i] = pos;
        }

        for idx in child_indices {
            buffer.extend_from_slice(&idx.to_le_bytes());
        }

        Ok(start_pos)
    }

    /// Encode AST for GPU processing
    fn encode_ast(&self, ast: &AstNode) -> Result<Vec<u8>> {
        let mut encoded = Vec::new();
        self.encode_ast_recursive(ast, &mut encoded)?;
        Ok(encoded)
    }

    fn encode_ast_recursive(&self, ast: &AstNode, buffer: &mut Vec<u8>) -> Result<usize> {
        let start_pos = buffer.len();

        // Encode node type (4 bytes)
        buffer.extend_from_slice(&(ast.node_type as u32).to_le_bytes());

        // Encode value hash (4 bytes)
        let value_hash = ast.value.as_ref().map(|v| self.hash_string(v)).unwrap_or(0);
        buffer.extend_from_slice(&value_hash.to_le_bytes());

        // Encode child count (4 bytes)
        buffer.extend_from_slice(&(ast.children.len() as u32).to_le_bytes());

        // Encode children indices (4 bytes each, max 10)
        let mut child_indices = vec![0u32; 10];
        for (i, child) in ast.children.iter().enumerate().take(10) {
            child_indices[i] = (buffer.len() / 40) as u32; // Node index
            self.encode_ast_recursive(child, buffer)?;
        }

        for idx in child_indices {
            buffer.extend_from_slice(&idx.to_le_bytes());
        }

        Ok(start_pos)
    }

    /// Simple string hash function
    fn hash_string(&self, s: &str) -> u32 {
        let mut hash = 0u32;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
        }
        hash
    }

    /// Extract matches and variable bindings from GPU results
    fn extract_matches(
        &self,
        results: &[u32],
        pattern: &Pattern,
        ast: &AstNode,
    ) -> Result<Vec<Match>> {
        let mut matches = Vec::new();

        // Results format: [node_id, match_flag, ...]
        for i in (0..results.len()).step_by(2) {
            if results[i + 1] != 0 {
                // Found a match at node_id
                let node_id = results[i] as usize;

                // Extract variable bindings
                let bindings = self.extract_bindings(pattern, ast, node_id)?;

                matches.push(Match { node_id, bindings });
            }
        }

        Ok(matches)
    }

    /// Extract variable bindings from a match
    fn extract_bindings(
        &self,
        pattern: &Pattern,
        ast: &AstNode,
        _node_id: usize,
    ) -> Result<HashMap<String, String>> {
        let mut bindings = HashMap::new();

        // Simple binding extraction for variables
        if let Some(ref pattern_val) = pattern.value {
            if pattern_val.starts_with('$') {
                if let Some(ref ast_val) = ast.value {
                    bindings.insert(pattern_val.clone(), ast_val.clone());
                }
            }
        }

        // Recursively extract from children
        for (p_child, a_child) in pattern.children.iter().zip(ast.children.iter()) {
            let child_bindings = self.extract_bindings(p_child, a_child, 0)?;
            bindings.extend(child_bindings);
        }

        Ok(bindings)
    }
}

impl Drop for GpuPatternMatcher {
    fn drop(&mut self) {
        // Buffers are automatically freed when CudaSlice is dropped
    }
}
