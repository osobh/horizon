//! Fast GPU Pattern Matching Implementation
//! 
//! Optimized for throughput with simpler API

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use std::sync::Arc;
use std::collections::HashMap;
use anyhow::{Result, Context};
use crate::synthesis::{Pattern, AstNode, Match, NodeType};

const NODE_SIZE: usize = 64; // Aligned to 64 bytes
const MAX_CHILDREN: usize = 10;
const PATTERNS_PER_BATCH: usize = 32;

/// Fast GPU Pattern Matcher
pub struct FastGpuPatternMatcher {
    device: Arc<CudaDevice>,
    
    // Pre-allocated buffers
    pattern_buffer: CudaSlice<u8>,
    ast_buffer: CudaSlice<u8>,
    match_buffer: CudaSlice<u32>,
    
    // Capacities
    max_patterns: usize,
    max_nodes: usize,
}

impl FastGpuPatternMatcher {
    pub fn new(device: Arc<CudaDevice>, max_patterns: usize, max_nodes: usize) -> Result<Self> {
        // Allocate aligned buffers
        // SAFETY: alloc returns uninitialized memory. pattern_buffer will be written
        // via htod_copy_into before any kernel reads in match_batch().
        let pattern_buffer = unsafe {
            device.alloc::<u8>(max_patterns * NODE_SIZE)
        }.context("Failed to allocate pattern buffer")?;
        
        // SAFETY: alloc returns uninitialized memory. ast_buffer will be written
        // via htod_copy_into before any kernel reads in match_batch().
        let ast_buffer = unsafe {
            device.alloc::<u8>(max_nodes * NODE_SIZE)
        }.context("Failed to allocate AST buffer")?;
        
        // SAFETY: alloc returns uninitialized memory. match_buffer is cleared
        // via htod_copy_into with zeros before any kernel reads in match_batch().
        let match_buffer = unsafe {
            device.alloc::<u32>(max_nodes * 2)
        }.context("Failed to allocate match buffer")?;
        
        Ok(Self {
            device,
            pattern_buffer,
            ast_buffer,
            match_buffer,
            max_patterns,
            max_nodes,
        })
    }
    
    /// Fast batch matching
    pub fn match_batch(&self, patterns: &[Pattern], asts: &[AstNode]) -> Result<Vec<Vec<Match>>> {
        // Encode patterns
        let pattern_data = self.encode_patterns_aligned(patterns)?;
        let ast_data = self.encode_asts_aligned(asts)?;
        
        let pattern_count = patterns.len();
        let node_count = ast_data.len() / NODE_SIZE;
        
        // Copy to GPU using slices
        let mut pattern_slice = self.pattern_buffer.slice(0..pattern_data.len());
        self.device.htod_copy_into(pattern_data, &mut pattern_slice)?;
        
        let mut ast_slice = self.ast_buffer.slice(0..ast_data.len());
        self.device.htod_copy_into(ast_data, &mut ast_slice)?;
        
        // Clear match buffer
        let zeros = vec![0u32; node_count * 2];
        let mut match_slice = self.match_buffer.slice(0..zeros.len());
        self.device.htod_copy_into(zeros, &mut match_slice)?;
        
        // Launch fast kernel
        // SAFETY: All pointers are valid device pointers from CudaSlice allocations.
        // pattern_count and node_count match the encoded data sizes. Buffers have
        // been initialized via htod_copy_into before this kernel launch.
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
        
        // Get results
        let mut results = vec![0u32; node_count * 2];
        let result_slice = self.match_buffer.slice(0..results.len());
        self.device.dtoh_sync_copy_into(&result_slice, &mut results)?;
        
        // Extract matches
        self.extract_matches(&results, patterns.len())
    }
    
    /// Encode patterns with 64-byte alignment
    fn encode_patterns_aligned(&self, patterns: &[Pattern]) -> Result<Vec<u8>> {
        let mut buffer = Vec::with_capacity(patterns.len() * NODE_SIZE);
        
        for pattern in patterns {
            let start = buffer.len();
            buffer.resize(start + NODE_SIZE, 0);
            
            self.write_u32(&mut buffer, start + 0, pattern.node_type as u32);
            self.write_u32(&mut buffer, start + 4, self.hash_value(&pattern.value));
            self.write_u32(&mut buffer, start + 8, pattern.children.len() as u32);
            
            // Simple encoding - no recursive children for speed
            for i in 0..MAX_CHILDREN {
                self.write_u32(&mut buffer, start + 12 + i * 4, 0);
            }
        }
        
        Ok(buffer)
    }
    
    /// Encode ASTs with 64-byte alignment
    fn encode_asts_aligned(&self, asts: &[AstNode]) -> Result<Vec<u8>> {
        let mut buffer = Vec::with_capacity(asts.len() * NODE_SIZE * 2);
        
        for ast in asts {
            self.encode_ast_nodes(&mut buffer, ast)?;
        }
        
        Ok(buffer)
    }
    
    fn encode_ast_nodes(&self, buffer: &mut Vec<u8>, ast: &AstNode) -> Result<()> {
        let start = buffer.len();
        buffer.resize(start + NODE_SIZE, 0);
        
        self.write_u32(buffer, start + 0, ast.node_type as u32);
        self.write_u32(buffer, start + 4, self.hash_value(&ast.value));
        self.write_u32(buffer, start + 8, ast.children.len() as u32);
        
        // Encode child indices
        let child_start = buffer.len() / NODE_SIZE;
        for (i, child) in ast.children.iter().enumerate().take(MAX_CHILDREN) {
            self.write_u32(buffer, start + 12 + i * 4, (child_start + i + 1) as u32);
        }
        
        // Encode children
        for child in &ast.children {
            self.encode_ast_nodes(buffer, child)?;
        }
        
        Ok(())
    }
    
    /// Extract matches from GPU results
    fn extract_matches(&self, results: &[u32], pattern_count: usize) -> Result<Vec<Vec<Match>>> {
        let mut matches = vec![vec![]; pattern_count];
        
        for i in 0..results.len()/2 {
            let node_id = results[i * 2];
            let match_flags = results[i * 2 + 1];
            
            if match_flags != 0 {
                // Check which patterns matched
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
    
    /// Benchmark throughput
    pub fn benchmark_throughput(&self, duration_secs: u64) -> Result<f64> {
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
            self.match_batch(&patterns, &asts)?;
            operations += (patterns.len() * asts.len()) as u64;
        }
        
        let elapsed = start.elapsed().as_secs_f64();
        Ok(operations as f64 / elapsed)
    }
}