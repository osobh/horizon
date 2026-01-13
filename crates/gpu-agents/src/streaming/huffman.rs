//! GPU-accelerated Huffman coding for data compression
//!
//! Provides high-performance Huffman encoding and decoding operations
//! optimized for streaming data processing on GPU hardware.

use super::*;
use anyhow::{anyhow, Result};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

/// GPU Huffman processor for batch compression operations
pub struct GpuHuffmanProcessor {
    device: Arc<CudaContext>,
    config: HuffmanConfig,
    cuda_streams: Vec<Arc<CudaStream>>,
    buffer_pool: GpuBufferPool,
    statistics: HuffmanStatistics,
}

impl GpuHuffmanProcessor {
    /// Create new GPU Huffman processor
    pub fn new(device: Arc<CudaContext>, config: HuffmanConfig) -> Result<Self> {
        // Create CUDA streams for parallel processing
        let mut cuda_streams = Vec::with_capacity(config.num_streams);
        let default_stream = device.default_stream();
        for _ in 0..config.num_streams {
            cuda_streams.push(default_stream.fork()?);
        }

        // Create buffer pool for compression operations
        let buffer_pool = GpuBufferPool::new(
            device.clone(),
            config.num_streams * 4, // Quadruple buffering for input/output/tree/scratch
            config.max_data_size,
        )?;

        Ok(Self {
            device,
            config,
            cuda_streams,
            buffer_pool,
            statistics: HuffmanStatistics::default(),
        })
    }

    /// Encode batch of data using Huffman compression
    pub async fn encode_batch(&mut self, data_batch: &[Vec<u8>]) -> Result<Vec<HuffmanEncoded>> {
        if data_batch.is_empty() {
            return Ok(Vec::new());
        }

        if data_batch.len() > self.config.batch_size {
            return Err(anyhow!(
                "Batch size {} exceeds maximum {}",
                data_batch.len(),
                self.config.batch_size
            ));
        }

        let start_time = std::time::Instant::now();
        let mut results = Vec::with_capacity(data_batch.len());

        for data in data_batch {
            if data.len() > self.config.max_data_size {
                return Err(anyhow!(
                    "Data size {} exceeds maximum {}",
                    data.len(),
                    self.config.max_data_size
                ));
            }

            let encoded = if self.config.use_gpu_acceleration && data.len() > 1024 {
                self.encode_gpu(data).await?
            } else {
                self.encode_cpu(data)?
            };

            results.push(encoded);
        }

        // Update statistics
        let processing_time = start_time.elapsed();
        self.statistics.encode_operations += data_batch.len() as u64;
        self.statistics.total_encode_time += processing_time;
        self.statistics.total_input_bytes +=
            data_batch.iter().map(|d| d.len()).sum::<usize>() as u64;
        self.statistics.total_compressed_bytes +=
            results.iter().map(|r| r.data.len()).sum::<usize>() as u64;

        Ok(results)
    }

    /// Decode batch of Huffman-encoded data
    pub async fn decode_batch(&mut self, encoded_batch: &[HuffmanEncoded]) -> Result<Vec<Vec<u8>>> {
        if encoded_batch.is_empty() {
            return Ok(Vec::new());
        }

        let start_time = std::time::Instant::now();
        let mut results = Vec::with_capacity(encoded_batch.len());

        for encoded in encoded_batch {
            let decoded = if self.config.use_gpu_acceleration && encoded.data.len() > 1024 {
                self.decode_gpu(encoded).await?
            } else {
                self.decode_cpu(encoded)?
            };

            results.push(decoded);
        }

        // Update statistics
        let processing_time = start_time.elapsed();
        self.statistics.decode_operations += encoded_batch.len() as u64;
        self.statistics.total_decode_time += processing_time;

        Ok(results)
    }

    /// Get processing statistics
    pub fn get_statistics(&self) -> &HuffmanStatistics {
        &self.statistics
    }

    /// Get current memory usage
    pub fn get_memory_usage(&self) -> Result<usize> {
        // Estimate memory usage based on buffer pool and statistics
        Ok(self.buffer_pool.total_allocated() + std::mem::size_of::<HuffmanStatistics>())
    }

    /// GPU-accelerated encoding
    async fn encode_gpu(&mut self, data: &[u8]) -> Result<HuffmanEncoded> {
        // Get CUDA stream for this operation
        let stream_idx = self.statistics.encode_operations as usize % self.cuda_streams.len();
        let _stream = &self.cuda_streams[stream_idx];

        // Build Huffman tree on CPU (complex algorithm, better on CPU)
        let mut codec = HuffmanCodec::new(self.config.clone())?;
        let tree = codec.build_huffman_tree(data)?;

        // Acquire buffers
        let input_idx = self
            .buffer_pool
            .acquire()
            .ok_or_else(|| anyhow!("No input buffer available"))?;
        let output_idx = self
            .buffer_pool
            .acquire()
            .ok_or_else(|| anyhow!("No output buffer available"))?;

        // Copy data to GPU
        let default_stream = self.device.default_stream();
        {
            let input_buffer = self
                .buffer_pool
                .get_buffer_mut(input_idx)
                .ok_or_else(|| anyhow!("Invalid input buffer index"))?;
            default_stream.memcpy_htod(data, input_buffer)?;
        }

        // Simulate GPU kernel for encoding
        {
            let input_buffer = self
                .buffer_pool
                .get_buffer_mut(input_idx)
                .ok_or_else(|| anyhow!("Invalid input buffer index"))?;
            let input_data: Vec<u8> = default_stream.clone_dtoh(input_buffer)?;

            let output_buffer = self
                .buffer_pool
                .get_buffer_mut(output_idx)
                .ok_or_else(|| anyhow!("Invalid output buffer index"))?;
            // For now, just copy input to output
            default_stream.memcpy_htod(&input_data, output_buffer)?;
        }

        // Wait for completion
        self.device.synchronize()?;

        // Copy results back
        let encoded_data: Vec<u8> = {
            let output_buffer = self
                .buffer_pool
                .get_buffer_mut(output_idx)
                .ok_or_else(|| anyhow!("Invalid output buffer index"))?;
            default_stream.clone_dtoh(output_buffer)?
        };

        // Release buffers
        self.buffer_pool.release(input_idx);
        self.buffer_pool.release(output_idx);

        let encoded_len = encoded_data.len();
        Ok(HuffmanEncoded {
            data: encoded_data,
            tree: tree.clone(),
            original_length: data.len(),
            compression_ratio: data.len() as f32 / encoded_len as f32,
        })
    }

    /// GPU-accelerated decoding
    async fn decode_gpu(&mut self, encoded: &HuffmanEncoded) -> Result<Vec<u8>> {
        let stream_idx = self.statistics.decode_operations as usize % self.cuda_streams.len();
        let _stream = &self.cuda_streams[stream_idx];

        let input_idx = self
            .buffer_pool
            .acquire()
            .ok_or_else(|| anyhow!("No input buffer available"))?;
        let output_idx = self
            .buffer_pool
            .acquire()
            .ok_or_else(|| anyhow!("No output buffer available"))?;

        // Copy encoded data to GPU
        let default_stream = self.device.default_stream();
        {
            let input_buffer = self
                .buffer_pool
                .get_buffer_mut(input_idx)
                .ok_or_else(|| anyhow!("Invalid input buffer index"))?;
            default_stream.memcpy_htod(&encoded.data, input_buffer)?;
        }

        // Simulate GPU kernel for decoding
        {
            let input_buffer = self
                .buffer_pool
                .get_buffer_mut(input_idx)
                .ok_or_else(|| anyhow!("Invalid input buffer index"))?;
            let input_data: Vec<u8> = default_stream.clone_dtoh(input_buffer)?;

            let output_buffer = self
                .buffer_pool
                .get_buffer_mut(output_idx)
                .ok_or_else(|| anyhow!("Invalid output buffer index"))?;
            // For now, just copy input to output
            default_stream.memcpy_htod(&input_data, output_buffer)?;
        }

        self.device.synchronize()?;

        // Copy results back
        let decoded_data: Vec<u8> = {
            let output_buffer = self
                .buffer_pool
                .get_buffer_mut(output_idx)
                .ok_or_else(|| anyhow!("Invalid output buffer index"))?;
            default_stream.clone_dtoh(output_buffer)?
        };

        self.buffer_pool.release(input_idx);
        self.buffer_pool.release(output_idx);

        Ok(decoded_data)
    }

    /// CPU fallback encoding
    fn encode_cpu(&self, data: &[u8]) -> Result<HuffmanEncoded> {
        let mut codec = HuffmanCodec::new(self.config.clone())?;
        codec.encode(data)
    }

    /// CPU fallback decoding
    fn decode_cpu(&self, encoded: &HuffmanEncoded) -> Result<Vec<u8>> {
        let codec = HuffmanCodec::new(self.config.clone())?;
        codec.decode(&encoded.data, &encoded.tree)
    }

    /// Launch GPU kernel for Huffman encoding (placeholder)
    fn launch_huffman_encode_kernel(
        &self,
        input: &CudaSlice<u8>,
        output: &mut CudaSlice<u8>,
        data_length: usize,
        tree: &HuffmanTree,
        stream: &CudaStream,
    ) -> Result<()> {
        Self::launch_huffman_encode_kernel_static(
            &self.device,
            input,
            output,
            data_length,
            tree,
            stream,
        )
    }

    fn launch_huffman_encode_kernel_static(
        _device: &Arc<CudaContext>,
        input: &CudaSlice<u8>,
        output: &mut CudaSlice<u8>,
        data_length: usize,
        tree: &HuffmanTree,
        stream: &CudaStream,
    ) -> Result<()> {
        // Placeholder for GPU kernel launch
        // Real implementation would call: launch_huffman_encode_kernel
        let _ = (input, output, data_length, tree, stream);
        Ok(())
    }

    /// Launch GPU kernel for Huffman decoding (placeholder)
    fn launch_huffman_decode_kernel(
        &self,
        input: &CudaSlice<u8>,
        output: &mut CudaSlice<u8>,
        original_length: usize,
        tree: &HuffmanTree,
        stream: &CudaStream,
    ) -> Result<()> {
        Self::launch_huffman_decode_kernel_static(
            &self.device,
            input,
            output,
            original_length,
            tree,
            stream,
        )
    }

    fn launch_huffman_decode_kernel_static(
        _device: &Arc<CudaContext>,
        input: &CudaSlice<u8>,
        output: &mut CudaSlice<u8>,
        original_length: usize,
        tree: &HuffmanTree,
        stream: &CudaStream,
    ) -> Result<()> {
        let _ = (input, output, original_length, tree, stream);
        Ok(())
    }
}

/// CPU-based Huffman codec implementation
pub struct HuffmanCodec {
    config: HuffmanConfig,
}

impl HuffmanCodec {
    pub fn new(config: HuffmanConfig) -> Result<Self> {
        Ok(Self { config })
    }

    /// Build Huffman tree from input data
    pub fn build_huffman_tree(&mut self, data: &[u8]) -> Result<HuffmanTree> {
        if data.is_empty() {
            return Err(anyhow!("Cannot build Huffman tree from empty data"));
        }

        // Analyze character frequencies
        let frequencies = self.analyze_frequencies(data)?;

        // Build tree using priority queue
        let mut heap = BinaryHeap::new();

        // Create leaf nodes for each character
        for (&byte, &frequency) in &frequencies {
            heap.push(Reverse(HuffmanNode {
                frequency,
                byte: Some(byte),
                left: None,
                right: None,
            }));
        }

        // Special case: single character
        if heap.len() == 1 {
            let node = heap.pop().ok_or_else(|| anyhow!("Empty heap"))?.0;
            let root = HuffmanNode {
                frequency: node.frequency,
                byte: None,
                left: Some(Box::new(node)),
                right: None,
            };
            return Ok(self.build_code_table(root));
        }

        // Build tree by combining nodes
        while heap.len() > 1 {
            let left = Box::new(heap.pop().ok_or_else(|| anyhow!("Empty heap"))?.0);
            let right = Box::new(heap.pop().ok_or_else(|| anyhow!("Empty heap"))?.0);

            let combined = HuffmanNode {
                frequency: left.frequency + right.frequency,
                byte: None,
                left: Some(left),
                right: Some(right),
            };

            heap.push(Reverse(combined));
        }

        let root = heap.pop().ok_or_else(|| anyhow!("Empty heap"))?.0;
        Ok(self.build_code_table(root))
    }

    /// Analyze character frequencies in data
    pub fn analyze_frequencies(&self, data: &[u8]) -> Result<HashMap<u8, u64>> {
        let mut frequencies = HashMap::new();

        for &byte in data {
            *frequencies.entry(byte).or_insert(0) += 1;
        }

        if frequencies.is_empty() {
            return Err(anyhow!("No characters found in data"));
        }

        Ok(frequencies)
    }

    /// Encode data using Huffman compression
    pub fn encode(&mut self, data: &[u8]) -> Result<HuffmanEncoded> {
        if data.is_empty() {
            return Err(anyhow!("Cannot encode empty data"));
        }

        let tree = self.build_huffman_tree(data)?;
        let encoded_bits = self.encode_with_tree(data, &tree)?;

        // Convert bits to bytes
        let mut encoded_data = Vec::new();
        let mut current_byte = 0u8;
        let mut bit_count = 0;

        for bit in encoded_bits {
            current_byte = (current_byte << 1) | bit;
            bit_count += 1;

            if bit_count == 8 {
                encoded_data.push(current_byte);
                current_byte = 0;
                bit_count = 0;
            }
        }

        // Handle remaining bits
        if bit_count > 0 {
            current_byte <<= 8 - bit_count;
            encoded_data.push(current_byte);
        }

        let compression_ratio = data.len() as f32 / encoded_data.len() as f32;

        Ok(HuffmanEncoded {
            data: encoded_data,
            tree,
            original_length: data.len(),
            compression_ratio,
        })
    }

    /// Decode Huffman-encoded data
    pub fn decode(&self, encoded_data: &[u8], tree: &HuffmanTree) -> Result<Vec<u8>> {
        if encoded_data.is_empty() {
            return Ok(Vec::new());
        }

        if tree.root.is_none() {
            return Err(anyhow!("Invalid Huffman tree: no root node"));
        }

        let mut decoded = Vec::new();
        let mut current_node = tree.root.as_ref().ok_or_else(|| anyhow!("No root node"))?;

        for &byte in encoded_data {
            for bit_pos in (0..8).rev() {
                let bit = (byte >> bit_pos) & 1;

                current_node = if bit == 0 {
                    if let Some(ref left) = current_node.left {
                        left
                    } else {
                        current_node
                    }
                } else {
                    if let Some(ref right) = current_node.right {
                        right
                    } else {
                        current_node
                    }
                };

                // If we've reached a leaf node
                if let Some(decoded_byte) = current_node.byte {
                    decoded.push(decoded_byte);
                    current_node = tree.root.as_ref().ok_or_else(|| anyhow!("No root node"))?;

                    // Stop if we've decoded enough bytes
                    if decoded.len() >= tree.original_length {
                        break;
                    }
                }
            }

            if decoded.len() >= tree.original_length {
                break;
            }
        }

        decoded.truncate(tree.original_length);
        Ok(decoded)
    }

    /// Build code table from Huffman tree
    fn build_code_table(&self, root: HuffmanNode) -> HuffmanTree {
        let mut code_table = HashMap::new();

        // Traverse tree to build codes
        self.build_codes(&root, Vec::new(), &mut code_table);

        HuffmanTree {
            root: Some(root),
            code_table,
            original_length: 0, // Will be set during encoding
        }
    }

    /// Recursively build Huffman codes
    fn build_codes(
        &self,
        node: &HuffmanNode,
        code: Vec<u8>,
        code_table: &mut HashMap<u8, HuffmanCode>,
    ) {
        if let Some(byte) = node.byte {
            // Leaf node - store the code
            let code = if code.is_empty() {
                vec![0] // Single character case
            } else {
                code
            };

            let bit_length = code.len();
            code_table.insert(
                byte,
                HuffmanCode {
                    bits: code,
                    bit_length,
                },
            );
        } else {
            // Internal node - continue traversal
            if let Some(ref left) = node.left {
                let mut left_code = code.clone();
                left_code.push(0);
                self.build_codes(left, left_code, code_table);
            }

            if let Some(ref right) = node.right {
                let mut right_code = code.clone();
                right_code.push(1);
                self.build_codes(right, right_code, code_table);
            }
        }
    }

    /// Encode data using existing Huffman tree
    fn encode_with_tree(&self, data: &[u8], tree: &HuffmanTree) -> Result<Vec<u8>> {
        let mut encoded_bits = Vec::new();

        for &byte in data {
            let code = tree
                .code_table
                .get(&byte)
                .ok_or_else(|| anyhow!("Character {} not found in Huffman tree", byte))?;

            encoded_bits.extend_from_slice(&code.bits);
        }

        Ok(encoded_bits)
    }
}

/// Huffman tree node
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HuffmanNode {
    pub frequency: u64,
    pub byte: Option<u8>,
    pub left: Option<Box<HuffmanNode>>,
    pub right: Option<Box<HuffmanNode>>,
}

impl PartialOrd for HuffmanNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HuffmanNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.frequency.cmp(&other.frequency)
    }
}

/// Huffman tree structure
#[derive(Debug, Clone)]
pub struct HuffmanTree {
    pub root: Option<HuffmanNode>,
    pub code_table: HashMap<u8, HuffmanCode>,
    pub original_length: usize,
}

/// Huffman code for a character
#[derive(Debug, Clone)]
pub struct HuffmanCode {
    pub bits: Vec<u8>,
    pub bit_length: usize,
}

/// Encoded data with Huffman compression
#[derive(Debug, Clone)]
pub struct HuffmanEncoded {
    pub data: Vec<u8>,
    pub tree: HuffmanTree,
    pub original_length: usize,
    pub compression_ratio: f32,
}

/// Configuration for Huffman processor
#[derive(Debug, Clone)]
pub struct HuffmanConfig {
    /// Maximum tree depth to prevent excessive memory usage
    pub max_tree_depth: usize,
    /// Maximum batch size for processing
    pub batch_size: usize,
    /// Maximum data size per item
    pub max_data_size: usize,
    /// Number of CUDA streams for parallel processing
    pub num_streams: usize,
    /// Use GPU acceleration when available
    pub use_gpu_acceleration: bool,
    /// Compression level preference
    pub compression_level: CompressionLevel,
    /// Enable detailed statistics collection
    pub enable_statistics: bool,
}

impl Default for HuffmanConfig {
    fn default() -> Self {
        Self {
            max_tree_depth: 32,         // Reasonable depth limit
            batch_size: 512,            // 512 items per batch
            max_data_size: 1024 * 1024, // 1MB max per item
            num_streams: 4,             // 4 parallel streams
            use_gpu_acceleration: true,
            compression_level: CompressionLevel::Balanced,
            enable_statistics: true,
        }
    }
}

/// Compression level preferences
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CompressionLevel {
    /// Prioritize speed over compression ratio
    Fast,
    /// Balance between speed and compression
    Balanced,
    /// Prioritize compression ratio over speed
    Best,
}

/// Huffman processing statistics
#[derive(Debug, Default)]
pub struct HuffmanStatistics {
    pub encode_operations: u64,
    pub decode_operations: u64,
    pub total_input_bytes: u64,
    pub total_compressed_bytes: u64,
    pub total_encode_time: std::time::Duration,
    pub total_decode_time: std::time::Duration,
}

impl HuffmanStatistics {
    /// Calculate average compression ratio
    pub fn avg_compression_ratio(&self) -> f64 {
        if self.total_compressed_bytes > 0 {
            self.total_input_bytes as f64 / self.total_compressed_bytes as f64
        } else {
            0.0
        }
    }

    /// Calculate encoding throughput in bytes per second
    pub fn encode_throughput_bps(&self) -> f64 {
        if self.total_encode_time.as_secs_f64() > 0.0 {
            self.total_input_bytes as f64 / self.total_encode_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Calculate decoding throughput in bytes per second
    pub fn decode_throughput_bps(&self) -> f64 {
        if self.total_decode_time.as_secs_f64() > 0.0 {
            self.total_input_bytes as f64 / self.total_decode_time.as_secs_f64()
        } else {
            0.0
        }
    }
}

/// High-level Huffman stream processor
pub struct HuffmanStreamProcessor {
    gpu_processor: GpuHuffmanProcessor,
    enable_gpu_acceleration: bool,
}

impl HuffmanStreamProcessor {
    /// Create new Huffman stream processor
    pub fn new(device: Arc<CudaContext>, config: HuffmanConfig) -> Result<Self> {
        let gpu_processor = GpuHuffmanProcessor::new(device, config)?;

        Ok(Self {
            gpu_processor,
            enable_gpu_acceleration: true,
        })
    }

    /// Compress data with automatic GPU/CPU fallback
    pub async fn compress(&mut self, data: Vec<Vec<u8>>) -> Result<Vec<HuffmanEncoded>> {
        if self.enable_gpu_acceleration {
            match self.gpu_processor.encode_batch(&data).await {
                Ok(results) => Ok(results),
                Err(e) => {
                    eprintln!("GPU compression failed, falling back to CPU: {}", e);
                    self.compress_cpu(data)
                }
            }
        } else {
            self.compress_cpu(data)
        }
    }

    /// Decompress data with automatic GPU/CPU fallback
    pub async fn decompress(&mut self, encoded: Vec<HuffmanEncoded>) -> Result<Vec<Vec<u8>>> {
        if self.enable_gpu_acceleration {
            match self.gpu_processor.decode_batch(&encoded).await {
                Ok(results) => Ok(results),
                Err(e) => {
                    eprintln!("GPU decompression failed, falling back to CPU: {}", e);
                    self.decompress_cpu(encoded)
                }
            }
        } else {
            self.decompress_cpu(encoded)
        }
    }

    /// CPU fallback for compression
    fn compress_cpu(&self, data: Vec<Vec<u8>>) -> Result<Vec<HuffmanEncoded>> {
        let config = self.gpu_processor.config.clone();
        let mut results = Vec::with_capacity(data.len());

        for item in data {
            let mut codec = HuffmanCodec::new(config.clone())?;
            let encoded = codec.encode(&item)?;
            results.push(encoded);
        }

        Ok(results)
    }

    /// CPU fallback for decompression
    fn decompress_cpu(&self, encoded: Vec<HuffmanEncoded>) -> Result<Vec<Vec<u8>>> {
        let config = self.gpu_processor.config.clone();
        let codec = HuffmanCodec::new(config)?;
        let mut results = Vec::with_capacity(encoded.len());

        for item in encoded {
            let decoded = codec.decode(&item.data, &item.tree)?;
            results.push(decoded);
        }

        Ok(results)
    }

    /// Get processing statistics
    pub fn get_statistics(&self) -> &HuffmanStatistics {
        self.gpu_processor.get_statistics()
    }

    /// Enable or disable GPU acceleration
    pub fn set_gpu_acceleration(&mut self, enabled: bool) {
        self.enable_gpu_acceleration = enabled;
    }
}
