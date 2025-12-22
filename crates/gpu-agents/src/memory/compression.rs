//! Compression Module for Tier-Specific Data Compression
//!
//! Implements LZ4 and ZSTD compression for storage tiers

use anyhow::{Context, Result};
use std::io::{Read, Write};

/// Compress data using LZ4 algorithm
pub fn compress_lz4(data: &[u8]) -> Result<Vec<u8>> {
    let max_compressed_size = lz4::block::compress_bound(data.len())?;
    let mut compressed = vec![0u8; max_compressed_size + 8]; // +8 for uncompressed size header

    // Write uncompressed size as header
    let size_bytes = (data.len() as u64).to_le_bytes();
    compressed[..8].copy_from_slice(&size_bytes);

    // Compress data
    let compressed_size = lz4::block::compress_to_buffer(data, None, false, &mut compressed[8..])?;

    compressed.truncate(8 + compressed_size);
    Ok(compressed)
}

/// Decompress LZ4 compressed data
pub fn decompress_lz4(compressed: &[u8], expected_size: usize) -> Result<Vec<u8>> {
    if compressed.len() < 8 {
        anyhow::bail!("Invalid compressed data: too short");
    }

    // Read uncompressed size from header
    let mut size_bytes = [0u8; 8];
    size_bytes.copy_from_slice(&compressed[..8]);
    let uncompressed_size = u64::from_le_bytes(size_bytes) as usize;

    // Validate expected size if provided
    if expected_size > 0 && uncompressed_size != expected_size {
        anyhow::bail!(
            "Size mismatch: expected {}, got {}",
            expected_size,
            uncompressed_size
        );
    }

    // Decompress data
    let mut decompressed = vec![0u8; uncompressed_size];
    let decompressed_size = lz4::block::decompress_to_buffer(
        &compressed[8..],
        Some(uncompressed_size as i32),
        &mut decompressed,
    )?;

    if decompressed_size != uncompressed_size {
        anyhow::bail!(
            "Decompression size mismatch: expected {}, got {}",
            uncompressed_size,
            decompressed_size
        );
    }

    Ok(decompressed)
}

/// Compress data using ZSTD algorithm
pub fn compress_zstd(data: &[u8], level: i32) -> Result<Vec<u8>> {
    let mut encoder = zstd::Encoder::new(Vec::new(), level)?;
    encoder.write_all(data)?;
    let compressed = encoder.finish()?;
    Ok(compressed)
}

/// Decompress ZSTD compressed data
pub fn decompress_zstd(compressed: &[u8], expected_size: usize) -> Result<Vec<u8>> {
    let mut decoder = zstd::Decoder::new(compressed)?;
    let mut decompressed = Vec::new();

    if expected_size > 0 {
        decompressed.reserve(expected_size);
    }

    decoder.read_to_end(&mut decompressed)?;

    // Validate expected size if provided
    if expected_size > 0 && decompressed.len() != expected_size {
        anyhow::bail!(
            "Size mismatch: expected {}, got {}",
            expected_size,
            decompressed.len()
        );
    }

    Ok(decompressed)
}

/// GPU-accelerated compression placeholder
/// In production, this would use nvCOMP or similar GPU compression library
pub struct GpuCompressor {
    algorithm: CompressionAlgorithm,
    device_id: u32,
}

impl GpuCompressor {
    /// Create new GPU compressor
    pub fn new(algorithm: CompressionAlgorithm, device_id: u32) -> Self {
        Self {
            algorithm,
            device_id,
        }
    }

    /// Compress data on GPU
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        // In real implementation, would:
        // 1. Copy data to GPU
        // 2. Launch compression kernel
        // 3. Copy compressed data back

        match self.algorithm {
            CompressionAlgorithm::None => Ok(data.to_vec()),
            CompressionAlgorithm::Lz4 => compress_lz4(data),
            CompressionAlgorithm::Zstd(level) => compress_zstd(data, level),
        }
    }

    /// Decompress data on GPU
    pub fn decompress(&self, compressed: &[u8], expected_size: usize) -> Result<Vec<u8>> {
        match self.algorithm {
            CompressionAlgorithm::None => Ok(compressed.to_vec()),
            CompressionAlgorithm::Lz4 => decompress_lz4(compressed, expected_size),
            CompressionAlgorithm::Zstd(_) => decompress_zstd(compressed, expected_size),
        }
    }
}

/// Compression statistics
#[derive(Debug, Default)]
pub struct CompressionStats {
    pub total_compressed: u64,
    pub total_uncompressed: u64,
    pub compression_time_us: u64,
    pub decompression_time_us: u64,
    pub compression_ratio: f32,
}

impl CompressionStats {
    /// Update compression ratio
    pub fn update_ratio(&mut self) {
        if self.total_uncompressed > 0 {
            self.compression_ratio = self.total_compressed as f32 / self.total_uncompressed as f32;
        }
    }

    /// Add compression result
    pub fn add_compression(&mut self, uncompressed_size: u64, compressed_size: u64, time_us: u64) {
        self.total_uncompressed += uncompressed_size;
        self.total_compressed += compressed_size;
        self.compression_time_us += time_us;
        self.update_ratio();
    }

    /// Add decompression result
    pub fn add_decompression(&mut self, time_us: u64) {
        self.decompression_time_us += time_us;
    }

    /// Get average compression ratio
    pub fn average_ratio(&self) -> f32 {
        self.compression_ratio
    }

    /// Get space savings percentage
    pub fn space_savings(&self) -> f32 {
        if self.total_uncompressed > 0 {
            (1.0 - self.compression_ratio) * 100.0
        } else {
            0.0
        }
    }
}

/// Batch compression for multiple pages
pub struct BatchCompressor {
    algorithm: CompressionAlgorithm,
    stats: CompressionStats,
}

impl BatchCompressor {
    /// Create new batch compressor
    pub fn new(algorithm: CompressionAlgorithm) -> Self {
        Self {
            algorithm,
            stats: CompressionStats::default(),
        }
    }

    /// Compress batch of pages
    pub fn compress_batch(&mut self, pages: &[Vec<u8>]) -> Result<Vec<Vec<u8>>> {
        let mut compressed_pages = Vec::with_capacity(pages.len());

        for page in pages {
            let start = std::time::Instant::now();

            let compressed = match self.algorithm {
                CompressionAlgorithm::None => page.clone(),
                CompressionAlgorithm::Lz4 => compress_lz4(page)?,
                CompressionAlgorithm::Zstd(level) => compress_zstd(page, level)?,
            };

            let elapsed = start.elapsed().as_micros() as u64;
            self.stats
                .add_compression(page.len() as u64, compressed.len() as u64, elapsed);

            compressed_pages.push(compressed);
        }

        Ok(compressed_pages)
    }

    /// Decompress batch of pages
    pub fn decompress_batch(
        &mut self,
        compressed_pages: &[Vec<u8>],
        expected_size: usize,
    ) -> Result<Vec<Vec<u8>>> {
        let mut decompressed_pages = Vec::with_capacity(compressed_pages.len());

        for compressed in compressed_pages {
            let start = std::time::Instant::now();

            let decompressed = match self.algorithm {
                CompressionAlgorithm::None => compressed.clone(),
                CompressionAlgorithm::Lz4 => decompress_lz4(compressed, expected_size)?,
                CompressionAlgorithm::Zstd(_) => decompress_zstd(compressed, expected_size)?,
            };

            let elapsed = start.elapsed().as_micros() as u64;
            self.stats.add_decompression(elapsed);

            decompressed_pages.push(decompressed);
        }

        Ok(decompressed_pages)
    }

    /// Get compression statistics
    pub fn get_stats(&self) -> &CompressionStats {
        &self.stats
    }
}

/// Adaptive compression selector
pub struct AdaptiveCompressor {
    current_algorithm: CompressionAlgorithm,
    stats_by_algorithm: std::collections::HashMap<String, CompressionStats>,
    selection_threshold: f32,
}

impl AdaptiveCompressor {
    /// Create new adaptive compressor
    pub fn new(initial_algorithm: CompressionAlgorithm) -> Self {
        Self {
            current_algorithm: initial_algorithm,
            stats_by_algorithm: std::collections::HashMap::new(),
            selection_threshold: 0.1, // 10% improvement threshold
        }
    }

    /// Compress with adaptive algorithm selection
    pub fn compress(&mut self, data: &[u8]) -> Result<Vec<u8>> {
        // Try current algorithm
        let start = std::time::Instant::now();
        let compressed = match self.current_algorithm {
            CompressionAlgorithm::None => data.to_vec(),
            CompressionAlgorithm::Lz4 => compress_lz4(data)?,
            CompressionAlgorithm::Zstd(level) => compress_zstd(data, level)?,
        };
        let elapsed = start.elapsed().as_micros() as u64;

        // Update stats
        let algo_name = format!("{:?}", self.current_algorithm);
        let stats = self
            .stats_by_algorithm
            .entry(algo_name)
            .or_insert_with(CompressionStats::default);
        stats.add_compression(data.len() as u64, compressed.len() as u64, elapsed);

        // Periodically evaluate algorithm selection
        if stats.total_compressed > 100 * 1024 * 1024 {
            // Every 100MB
            self.evaluate_algorithms();
        }

        Ok(compressed)
    }

    /// Evaluate and potentially switch algorithms
    fn evaluate_algorithms(&mut self) {
        let mut best_algorithm = self.current_algorithm;
        let mut best_score = f32::MAX;

        for (algo_name, stats) in &self.stats_by_algorithm {
            // Score based on compression ratio and time
            let score = stats.compression_ratio
                + (stats.compression_time_us as f32 / stats.total_uncompressed as f32);

            if score < best_score * (1.0 - self.selection_threshold) {
                best_score = score;
                // Parse algorithm from name (simplified)
                if algo_name.contains("Lz4") {
                    best_algorithm = CompressionAlgorithm::Lz4;
                } else if algo_name.contains("Zstd") {
                    best_algorithm = CompressionAlgorithm::Zstd(3);
                }
            }
        }

        if best_algorithm != self.current_algorithm {
            println!("Switching compression algorithm to {:?}", best_algorithm);
            self.current_algorithm = best_algorithm;
        }
    }
}

use super::CompressionAlgorithm;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lz4_compression() {
        let data = b"Hello, World! This is a test of LZ4 compression.";
        let compressed = compress_lz4(data)?;
        assert!(compressed.len() < data.len() + 8); // Should be smaller + header

        let decompressed = decompress_lz4(&compressed, data.len())?;
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_zstd_compression() {
        let data = vec![42u8; 1024]; // Repetitive data compresses well
        let compressed = compress_zstd(&data, 3)?;
        assert!(compressed.len() < data.len() / 10); // Should compress very well

        let decompressed = decompress_zstd(&compressed, data.len())?;
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_batch_compression() {
        let pages = vec![vec![1u8; 4096], vec![2u8; 4096], vec![3u8; 4096]];

        let mut compressor = BatchCompressor::new(CompressionAlgorithm::Lz4);
        let compressed = compressor.compress_batch(&pages)?;
        assert_eq!(compressed.len(), pages.len());

        let decompressed = compressor.decompress_batch(&compressed, 4096)?;
        assert_eq!(decompressed, pages);

        let stats = compressor.get_stats();
        assert!(stats.compression_ratio < 0.5); // Good compression on repetitive data
        assert!(stats.space_savings() > 50.0);
    }

    #[test]
    fn test_compression_stats() {
        let mut stats = CompressionStats::default();
        stats.add_compression(1000, 500, 100);
        assert_eq!(stats.compression_ratio, 0.5);
        assert_eq!(stats.space_savings(), 50.0);

        stats.add_compression(2000, 500, 200);
        assert_eq!(stats.total_uncompressed, 3000);
        assert_eq!(stats.total_compressed, 1000);
        assert_eq!(stats.compression_ratio, 1.0 / 3.0);
    }
}
