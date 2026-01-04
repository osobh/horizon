//! GPU AST Transformation Implementation
//!
//! Parallel AST transformations and rewriting on GPU

use crate::synthesis::{AstNode, NodeType, TransformRule};
use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use std::sync::Arc;

/// GPU AST Transformer for parallel tree transformations
pub struct GpuAstTransformer {
    device: Arc<CudaDevice>,
    ast_buffer: CudaSlice<u8>,
    rule_buffer: CudaSlice<u8>,
    output_buffer: CudaSlice<u8>,
    max_nodes: usize,
}

impl GpuAstTransformer {
    /// Create a new GPU AST transformer
    pub fn new(device: Arc<CudaDevice>, max_nodes: usize) -> Result<Self> {
        // Allocate GPU buffers
        let node_size = 4 + 4 + 4 + 4 * 10; // type + value_hash + child_count + children
        let buffer_size = max_nodes * node_size;

        // SAFETY: alloc returns uninitialized memory. ast_buffer will be written
        // via htod_copy_into in transform_ast() before the kernel reads it.
        let ast_buffer =
            unsafe { device.alloc::<u8>(buffer_size) }.context("Failed to allocate AST buffer")?;
        // SAFETY: alloc returns uninitialized memory. rule_buffer will be written
        // via htod_copy_into in transform_ast() before the kernel reads it.
        let rule_buffer =
            unsafe { device.alloc::<u8>(buffer_size) }.context("Failed to allocate rule buffer")?;
        // SAFETY: alloc returns uninitialized memory. output_buffer will be cleared
        // to zeros in transform_ast() before the kernel writes transformed nodes.
        let output_buffer = unsafe { device.alloc::<u8>(buffer_size) }
            .context("Failed to allocate output buffer")?;

        Ok(Self {
            device,
            ast_buffer,
            rule_buffer,
            output_buffer,
            max_nodes,
        })
    }

    /// Transform an AST using a transformation rule
    pub fn transform_ast(&self, ast: &AstNode, rule: &TransformRule) -> Result<AstNode> {
        // Encode AST and rule for GPU
        let ast_data = self.encode_ast(ast)?;
        let rule_data = self.encode_rule(rule)?;

        // Copy to GPU
        self.device
            .htod_copy_into(ast_data.clone(), &mut self.ast_buffer.clone())?;
        self.device
            .htod_copy_into(rule_data.clone(), &mut self.rule_buffer.clone())?;

        // Clear output buffer
        let zeros = vec![0u8; self.max_nodes * 40];
        self.device
            .htod_copy_into(zeros, &mut self.output_buffer.clone())?;

        // Launch transformation kernel
        // SAFETY: All pointers are valid device pointers from CudaSlice allocations:
        // - ast_buffer: populated via htod_copy_into above
        // - rule_buffer: populated via htod_copy_into above
        // - output_buffer: cleared to zeros above, kernel writes transformed result
        // - num_nodes calculated from actual encoded AST size
        unsafe {
            crate::synthesis::launch_transform_ast(
                *self.ast_buffer.device_ptr() as *const u8,
                *self.rule_buffer.device_ptr() as *const u8,
                *self.output_buffer.device_ptr() as *mut u8,
                (ast_data.len() / 40) as u32, // Number of nodes
                1,                            // Single rule for now
            );
        }

        // Synchronize and get results
        self.device.synchronize()?;

        let mut output = vec![0u8; self.max_nodes * 40];
        self.device
            .dtoh_sync_copy_into(&self.output_buffer, &mut output)?;

        // Decode transformed AST
        let result = self.decode_ast(&output)?;

        Ok(result)
    }

    /// Apply multiple transformation rules in sequence
    pub fn transform_ast_multi(&self, ast: &AstNode, rules: &[TransformRule]) -> Result<AstNode> {
        let mut current = ast.clone();

        for rule in rules {
            current = self.transform_ast(&current, rule)?;
        }

        Ok(current)
    }

    /// Traverse AST and apply visitor function
    pub fn traverse_ast<F>(&self, ast: &AstNode, visitor: F) -> Result<()>
    where
        F: Fn(&AstNode, usize) -> Result<()>,
    {
        self.traverse_recursive(ast, &visitor, 0)
    }

    fn traverse_recursive<F>(&self, node: &AstNode, visitor: &F, depth: usize) -> Result<()>
    where
        F: Fn(&AstNode, usize) -> Result<()>,
    {
        visitor(node, depth)?;

        for child in &node.children {
            self.traverse_recursive(child, visitor, depth + 1)?;
        }

        Ok(())
    }

    /// Count nodes in AST
    pub fn count_nodes(&self, ast: &AstNode) -> usize {
        struct Counter {
            count: usize,
        }
        let mut counter = Counter { count: 0 };

        self.count_nodes_recursive(ast, &mut counter.count);
        counter.count
    }

    fn count_nodes_recursive(&self, node: &AstNode, count: &mut usize) {
        *count += 1;
        for child in &node.children {
            self.count_nodes_recursive(child, count);
        }
    }

    /// Find nodes of specific type
    pub fn find_nodes_by_type<'a>(
        &self,
        ast: &'a AstNode,
        node_type: NodeType,
    ) -> Vec<&'a AstNode> {
        let mut nodes = Vec::new();
        self.find_nodes_recursive(ast, node_type, &mut nodes);
        nodes
    }

    fn find_nodes_recursive<'a>(
        &self,
        node: &'a AstNode,
        target_type: NodeType,
        results: &mut Vec<&'a AstNode>,
    ) {
        if node.node_type == target_type {
            results.push(node);
        }

        for child in &node.children {
            self.find_nodes_recursive(child, target_type, results);
        }
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
            child_indices[i] = (buffer.len() / 40) as u32;
            self.encode_ast_recursive(child, buffer)?;
        }

        for idx in child_indices {
            buffer.extend_from_slice(&idx.to_le_bytes());
        }

        Ok(start_pos)
    }

    /// Encode transformation rule for GPU
    fn encode_rule(&self, rule: &TransformRule) -> Result<Vec<u8>> {
        let mut encoded = Vec::new();

        // Encode pattern
        self.encode_pattern_recursive(&rule.pattern, &mut encoded)?;

        // Encode replacement
        self.encode_ast_recursive(&rule.replacement, &mut encoded)?;

        Ok(encoded)
    }

    fn encode_pattern_recursive(
        &self,
        pattern: &crate::synthesis::Pattern,
        buffer: &mut Vec<u8>,
    ) -> Result<usize> {
        let start_pos = buffer.len();

        // Similar to encode_ast_recursive but for Pattern type
        buffer.extend_from_slice(&(pattern.node_type as u32).to_le_bytes());

        let value_hash = pattern
            .value
            .as_ref()
            .map(|v| self.hash_string(v))
            .unwrap_or(0);
        buffer.extend_from_slice(&value_hash.to_le_bytes());

        buffer.extend_from_slice(&(pattern.children.len() as u32).to_le_bytes());

        let mut child_indices = vec![0u32; 10];
        for (i, child) in pattern.children.iter().enumerate().take(10) {
            child_indices[i] = (buffer.len() / 40) as u32;
            self.encode_pattern_recursive(child, buffer)?;
        }

        for idx in child_indices {
            buffer.extend_from_slice(&idx.to_le_bytes());
        }

        Ok(start_pos)
    }

    /// Decode AST from GPU buffer
    fn decode_ast(&self, data: &[u8]) -> Result<AstNode> {
        self.decode_ast_at(data, 0)
    }

    fn decode_ast_at(&self, data: &[u8], offset: usize) -> Result<AstNode> {
        if offset + 40 > data.len() {
            anyhow::bail!("Invalid AST data: buffer too small");
        }

        let node_data = &data[offset..offset + 40];

        // Decode node type
        let node_type_val =
            u32::from_le_bytes([node_data[0], node_data[1], node_data[2], node_data[3]]);
        let node_type = match node_type_val {
            0 => NodeType::Function,
            1 => NodeType::Variable,
            2 => NodeType::Literal,
            3 => NodeType::Block,
            4 => NodeType::Return,
            5 => NodeType::BinaryOp,
            6 => NodeType::UnaryOp,
            7 => NodeType::Call,
            8 => NodeType::If,
            9 => NodeType::Loop,
            _ => anyhow::bail!("Invalid node type: {}", node_type_val),
        };

        // Decode child count
        let child_count =
            u32::from_le_bytes([node_data[8], node_data[9], node_data[10], node_data[11]]);

        // Decode children
        let mut children = Vec::new();
        for i in 0..child_count.min(10) {
            let child_idx_offset = 12 + i as usize * 4;
            let child_idx = u32::from_le_bytes([
                node_data[child_idx_offset],
                node_data[child_idx_offset + 1],
                node_data[child_idx_offset + 2],
                node_data[child_idx_offset + 3],
            ]);

            if child_idx > 0 {
                let child = self.decode_ast_at(data, child_idx as usize * 40)?;
                children.push(child);
            }
        }

        Ok(AstNode {
            node_type,
            children,
            value: None, // TODO: Decode value from hash table
        })
    }

    /// Simple string hash function
    fn hash_string(&self, s: &str) -> u32 {
        let mut hash = 0u32;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
        }
        hash
    }
}

impl Drop for GpuAstTransformer {
    fn drop(&mut self) {
        // Buffers are automatically freed when CudaSlice is dropped
    }
}
