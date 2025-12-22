//! Optimized GPU Pattern Matching Implementation
//! 
//! High-performance pattern matching targeting 2.6B ops/sec

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, CudaStream};
use std::sync::Arc;
use std::collections::HashMap;
use anyhow::{Result, Context};
use crate::synthesis::{Pattern, AstNode, Match, NodeType};

const NODE_SIZE: usize = 64; // Aligned to 64 bytes for better memory access
const MAX_CHILDREN: usize = 10;
const PATTERNS_PER_BLOCK: usize = 32; // Shared memory cache size
const THREADS_PER_BLOCK: usize = 256;
const NODES_PER_THREAD: usize = 4; // Process multiple nodes per thread

/// Optimized GPU Pattern Matcher for high throughput
pub struct GpuPatternMatcherOptimized {
    device: Arc<CudaDevice>,
    
    // Pre-allocated buffers
    pattern_buffer: CudaSlice<u8>,
    ast_buffer: CudaSlice<u8>,
    match_buffer: CudaSlice<u32>,
    
    // Pinned host memory for faster transfers
    pinned_pattern: Vec<u8>,
    pinned_ast: Vec<u8>,
    pinned_results: Vec<u32>,
    
    // CUDA streams for overlap
    streams: Vec<CudaStream>,
    
    // Buffer capacities
    max_patterns: usize,
    max_nodes: usize,
}

impl GpuPatternMatcherOptimized {
    pub fn new(device: Arc<CudaDevice>, max_patterns: usize, max_nodes: usize) -> Result<Self> {
        // Allocate GPU buffers with proper alignment
        let pattern_buffer_size = max_patterns * NODE_SIZE;
        let ast_buffer_size = max_nodes * NODE_SIZE;
        let match_buffer_size = max_nodes * 2 * std::mem::size_of::<u32>();
        
        let pattern_buffer = unsafe { 
            device.alloc::<u8>(pattern_buffer_size) 
        }.context("Failed to allocate pattern buffer")?;
        
        let ast_buffer = unsafe { 
            device.alloc::<u8>(ast_buffer_size) 
        }.context("Failed to allocate AST buffer")?;
        
        let match_buffer = unsafe { 
            device.alloc::<u32>(max_nodes * 2) 
        }.context("Failed to allocate match buffer")?;
        
        // Allocate pinned host memory
        let pinned_pattern = vec![0u8; pattern_buffer_size];
        let pinned_ast = vec![0u8; ast_buffer_size];
        let pinned_results = vec![0u32; max_nodes * 2];
        
        // Create CUDA streams for overlap
        let streams = (0..4)
            .map(|_| device.fork_default_stream())
            .collect::<Result<Vec<_>, _>>()?;
        
        Ok(Self {
            device,
            pattern_buffer,
            ast_buffer,
            match_buffer,
            pinned_pattern,
            pinned_ast,
            pinned_results,
            streams,
            max_patterns,
            max_nodes,
        })
    }
    
    /// Match multiple patterns against multiple ASTs in batches
    pub fn match_patterns_batch(
        &mut self,
        patterns: &[Pattern],
        ast_forest: &[AstNode],
    ) -> Result<Vec<Vec<Match>>> {
        let mut all_results = Vec::new();
        
        // Process patterns in batches that fit in shared memory
        for pattern_batch in patterns.chunks(PATTERNS_PER_BLOCK) {
            // Encode patterns into pinned memory
            let pattern_count = self.encode_patterns_aligned(pattern_batch)?;
            
            // Process ASTs in batches
            for ast_batch in ast_forest.chunks(self.max_nodes / 4) {
                let node_count = self.encode_asts_aligned(ast_batch)?;
                
                // Async copy to GPU using streams
                let stream_idx = 0;
                let stream = &self.streams[stream_idx];
                
                // Copy patterns (stream 0)
                unsafe {
                    self.device.htod_async_copy_into(
                        &self.pinned_pattern[..pattern_count * NODE_SIZE],
                        &mut self.pattern_buffer.clone(),
                        stream,
                    )?;
                }
                
                // Copy ASTs (stream 1)
                let ast_stream = &self.streams[1];
                unsafe {
                    self.device.htod_async_copy_into(
                        &self.pinned_ast[..node_count * NODE_SIZE],
                        &mut self.ast_buffer.clone(),
                        ast_stream,
                    )?;
                }
                
                // Clear match buffer
                let zero_slice = vec![0u32; node_count * 2];
                self.device.htod_copy_into(zero_slice, &mut self.match_buffer.clone())?;
                
                // Synchronize streams before kernel launch
                stream.synchronize()?;
                ast_stream.synchronize()?;
                
                // Launch optimized kernel
                self.launch_optimized_kernel(pattern_count, node_count)?;
                
                // Copy results back asynchronously
                let result_stream = &self.streams[2];
                unsafe {
                    self.device.dtoh_async_copy_into(
                        &self.match_buffer,
                        &mut self.pinned_results[..node_count * 2],
                        result_stream,
                    )?;
                }
                
                result_stream.synchronize()?;
                
                // Extract matches
                let batch_matches = self.extract_matches_batch(
                    &self.pinned_results[..node_count * 2],
                    pattern_batch,
                    ast_batch,
                )?;
                
                all_results.extend(batch_matches);
            }
        }
        
        Ok(all_results)
    }
    
    /// Encode patterns with proper alignment
    fn encode_patterns_aligned(&mut self, patterns: &[Pattern]) -> Result<usize> {
        let mut offset = 0;
        let mut count = 0;
        
        for pattern in patterns {
            self.encode_pattern_aligned(&mut self.pinned_pattern, pattern, &mut offset)?;
            count += 1;
        }
        
        Ok(count)
    }
    
    /// Encode ASTs with proper alignment
    fn encode_asts_aligned(&mut self, asts: &[AstNode]) -> Result<usize> {
        let mut offset = 0;
        let mut total_nodes = 0;
        
        for ast in asts {
            let nodes = self.encode_ast_aligned(&mut self.pinned_ast, ast, &mut offset)?;
            total_nodes += nodes;
        }
        
        Ok(total_nodes)
    }
    
    /// Encode pattern with 64-byte alignment
    fn encode_pattern_aligned(
        &self,
        buffer: &mut [u8],
        pattern: &Pattern,
        offset: &mut usize,
    ) -> Result<()> {
        let start = *offset;
        
        // Ensure alignment
        if start % 64 != 0 {
            *offset = ((start / 64) + 1) * 64;
        }
        
        // Reserve space
        buffer.resize(buffer.len().max(*offset + NODE_SIZE), 0);
        
        // Write node data
        self.write_u32(buffer, *offset + 0, pattern.node_type as u32);
        self.write_u32(buffer, *offset + 4, self.hash_value(&pattern.value));
        self.write_u32(buffer, *offset + 8, pattern.children.len() as u32);
        
        // Encode children
        let mut child_indices = [0u32; MAX_CHILDREN];
        for (i, child) in pattern.children.iter().enumerate().take(MAX_CHILDREN) {
            *offset += NODE_SIZE;
            child_indices[i] = (*offset / NODE_SIZE) as u32;
            self.encode_pattern_aligned(buffer, child, offset)?;
        }
        
        // Write child indices
        for (i, idx) in child_indices.iter().enumerate() {
            self.write_u32(buffer, start + 12 + i * 4, *idx);
        }
        
        *offset += NODE_SIZE;
        Ok(())
    }
    
    /// Encode AST with 64-byte alignment
    fn encode_ast_aligned(
        &self,
        buffer: &mut [u8],
        ast: &AstNode,
        offset: &mut usize,
    ) -> Result<usize> {
        let start_offset = *offset;
        self.encode_ast_node_aligned(buffer, ast, offset)?;
        Ok((*offset - start_offset) / NODE_SIZE)
    }
    
    fn encode_ast_node_aligned(
        &self,
        buffer: &mut [u8],
        ast: &AstNode,
        offset: &mut usize,
    ) -> Result<()> {
        let start = *offset;
        
        // Ensure alignment
        if start % 64 != 0 {
            *offset = ((start / 64) + 1) * 64;
        }
        
        // Reserve space
        buffer.resize(buffer.len().max(*offset + NODE_SIZE), 0);
        
        // Write node data
        self.write_u32(buffer, *offset + 0, ast.node_type as u32);
        self.write_u32(buffer, *offset + 4, self.hash_value(&ast.value));
        self.write_u32(buffer, *offset + 8, ast.children.len() as u32);
        
        // Encode children
        let mut child_indices = [0u32; MAX_CHILDREN];
        for (i, child) in ast.children.iter().enumerate().take(MAX_CHILDREN) {
            *offset += NODE_SIZE;
            child_indices[i] = (*offset / NODE_SIZE) as u32;
            self.encode_ast_node_aligned(buffer, child, offset)?;
        }
        
        // Write child indices
        for (i, idx) in child_indices.iter().enumerate() {
            self.write_u32(buffer, start + 12 + i * 4, *idx);
        }
        
        *offset += NODE_SIZE;
        Ok(())
    }
    
    /// Launch optimized kernel with proper configuration
    fn launch_optimized_kernel(&self, pattern_count: usize, node_count: usize) -> Result<()> {
        unsafe {
            crate::synthesis::launch_match_patterns_fast(
                *self.pattern_buffer.device_ptr() as *const u8,
                *self.ast_buffer.device_ptr() as *const u8,
                *self.match_buffer.device_ptr() as *mut u32,
                pattern_count as u32,
                node_count as u32,
            );
        }
        
        self.device.synchronize()?;
        Ok(())
    }
    
    /// Extract matches from results
    fn extract_matches_batch(
        &self,
        results: &[u32],
        patterns: &[Pattern],
        asts: &[AstNode],
    ) -> Result<Vec<Vec<Match>>> {
        let mut all_matches = vec![vec![]; patterns.len()];
        
        for i in 0..results.len()/2 {
            let node_id = results[i * 2] as usize;
            let match_flags = results[i * 2 + 1];
            
            // Check which patterns matched (bit flags)
            for (p, _pattern) in patterns.iter().enumerate() {
                if match_flags & (1 << p) != 0 {
                    all_matches[p].push(Match {
                        node_id,
                        bindings: HashMap::new(),
                    });
                }
            }
        }
        
        Ok(all_matches)
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
    
    /// Measure throughput in operations per second
    pub fn benchmark_throughput(&mut self, duration_secs: u64) -> Result<f64> {
        use std::time::{Instant, Duration};
        
        // Create test data
        let patterns: Vec<Pattern> = (0..32).map(|i| Pattern {
            node_type: NodeType::Variable,
            children: vec![],
            value: Some(format!("var{}", i)),
        }).collect();
        
        let asts: Vec<AstNode> = (0..1000).map(|i| AstNode {
            node_type: NodeType::Variable,
            children: vec![],
            value: Some(format!("var{}", i % 32)),
        }).collect();
        
        let start = Instant::now();
        let mut operations = 0u64;
        
        while start.elapsed() < Duration::from_secs(duration_secs) {
            self.match_patterns_batch(&patterns, &asts)?;
            operations += (patterns.len() * asts.len()) as u64;
        }
        
        let elapsed = start.elapsed().as_secs_f64();
        Ok(operations as f64 / elapsed)
    }
}