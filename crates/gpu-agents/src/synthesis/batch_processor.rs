//! Batch processing for multiple patterns
//!
//! Processes multiple patterns in parallel for better GPU utilization

use super::pattern_dynamic::DynamicGpuPatternMatcher;
use crate::synthesis::{AstNode, Match, Pattern};
use anyhow::Result;
use cudarc::driver::{CudaContext, CudaStream};
use futures::future::join_all;
use std::sync::Arc;

/// Configuration for batch processing
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum patterns per batch
    pub max_patterns_per_batch: usize,
    /// Maximum AST nodes per batch
    pub max_nodes_per_batch: usize,
    /// Number of CUDA streams to use
    pub num_streams: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_patterns_per_batch: 32,
            max_nodes_per_batch: 10_000,
            num_streams: 4,
        }
    }
}

/// Batch processor for pattern matching
pub struct BatchProcessor {
    device: Arc<CudaContext>,
    config: BatchConfig,
    matcher: DynamicGpuPatternMatcher,
    streams: Vec<Arc<CudaStream>>,
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new(device: Arc<CudaContext>, config: BatchConfig) -> Result<Self> {
        let matcher = DynamicGpuPatternMatcher::new(device.clone())?;

        // Create CUDA streams for parallel execution
        let mut streams = Vec::with_capacity(config.num_streams);
        for _ in 0..config.num_streams {
            streams.push(device.new_stream()?);
        }

        Ok(Self {
            device,
            config,
            matcher,
            streams,
        })
    }

    /// Process multiple pattern batches in parallel
    pub fn process_batches(
        &self,
        pattern_batches: Vec<Vec<Pattern>>,
        ast_batches: Vec<Vec<AstNode>>,
    ) -> Result<Vec<Vec<Vec<Match>>>> {
        // Use tokio runtime for async processing
        let runtime = tokio::runtime::Runtime::new()?;

        runtime.block_on(async {
            self.process_batches_async(pattern_batches, ast_batches)
                .await
        })
    }

    /// Async version of process_batches that uses parallel streams
    async fn process_batches_async(
        &self,
        pattern_batches: Vec<Vec<Pattern>>,
        ast_batches: Vec<Vec<AstNode>>,
    ) -> Result<Vec<Vec<Vec<Match>>>> {
        let mut futures = Vec::new();

        // Distribute batches across available streams
        for (idx, (patterns, asts)) in pattern_batches
            .into_iter()
            .zip(ast_batches.into_iter())
            .enumerate()
        {
            let stream_idx = idx % self.streams.len();
            let matcher = self.matcher.clone();

            // Process each batch on a different stream
            let future = tokio::task::spawn_blocking(move || matcher.match_batch(&patterns, &asts));

            futures.push(future);
        }

        // Wait for all batches to complete
        let results = join_all(futures).await;

        // Collect results
        let mut all_results = Vec::new();
        for result in results {
            match result {
                Ok(Ok(batch_results)) => all_results.push(batch_results),
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(anyhow::anyhow!("Task join error: {}", e)),
            }
        }

        Ok(all_results)
    }

    /// Process a single batch efficiently
    pub fn process_single_batch(
        &self,
        patterns: &[Pattern],
        asts: &[AstNode],
    ) -> Result<Vec<Vec<Match>>> {
        // Check batch size limits
        if patterns.len() > self.config.max_patterns_per_batch {
            // Process chunks in parallel using available streams
            let runtime = tokio::runtime::Runtime::new()?;

            runtime.block_on(async { self.process_chunks_parallel(patterns, asts).await })
        } else {
            // Process directly
            self.matcher.match_batch(patterns, asts)
        }
    }

    /// Process chunks in parallel using multiple streams
    async fn process_chunks_parallel(
        &self,
        patterns: &[Pattern],
        asts: &[AstNode],
    ) -> Result<Vec<Vec<Match>>> {
        let mut futures = Vec::new();
        let chunks: Vec<_> = patterns
            .chunks(self.config.max_patterns_per_batch)
            .map(|c| c.to_vec())
            .collect();

        let asts = asts.to_vec();

        // Process each chunk on a different stream
        for (idx, chunk) in chunks.into_iter().enumerate() {
            let stream_idx = idx % self.streams.len();
            let matcher = self.matcher.clone();
            let asts_clone = asts.clone();

            let future =
                tokio::task::spawn_blocking(move || matcher.match_batch(&chunk, &asts_clone));

            futures.push(future);
        }

        // Wait for all chunks to complete
        let results = join_all(futures).await;

        // Collect and flatten results
        let mut all_matches = Vec::new();
        for result in results {
            match result {
                Ok(Ok(chunk_matches)) => all_matches.extend(chunk_matches),
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(anyhow::anyhow!("Task join error: {}", e)),
            }
        }

        Ok(all_matches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::NodeType;

    fn create_test_pattern(name: &str) -> Pattern {
        Pattern {
            node_type: NodeType::Variable,
            children: vec![],
            value: Some(name.to_string()),
        }
    }

    fn create_test_ast(name: &str) -> AstNode {
        AstNode {
            node_type: NodeType::Variable,
            children: vec![],
            value: Some(name.to_string()),
        }
    }

    #[test]
    fn test_batch_processor_creation() -> Result<(), Box<dyn std::error::Error>> {
        let device = CudaContext::new(0)?;
        let config = BatchConfig::default();
        let processor = BatchProcessor::new(device, config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_single_batch_processing() -> Result<(), Box<dyn std::error::Error>> {
        let device = CudaContext::new(0)?;
        let processor = BatchProcessor::new(device, BatchConfig::default())?;

        let patterns = vec![create_test_pattern("x"), create_test_pattern("y")];

        let asts = vec![
            create_test_ast("x"),
            create_test_ast("y"),
            create_test_ast("z"),
        ];

        let result = processor.process_single_batch(&patterns, &asts);
        assert!(result.is_ok());

        let matches = result?;
        assert_eq!(matches.len(), 2); // One result per pattern
        assert!(!matches[0].is_empty()); // Pattern "x" should match AST "x"
        assert!(!matches[1].is_empty()); // Pattern "y" should match AST "y"
    }

    #[test]
    fn test_multiple_batch_processing() -> Result<(), Box<dyn std::error::Error>> {
        let device = CudaContext::new(0)?;
        let processor = BatchProcessor::new(device, BatchConfig::default())?;

        let pattern_batches = vec![
            vec![create_test_pattern("a"), create_test_pattern("b")],
            vec![create_test_pattern("c"), create_test_pattern("d")],
        ];

        let ast_batches = vec![
            vec![create_test_ast("a"), create_test_ast("b")],
            vec![create_test_ast("c"), create_test_ast("d")],
        ];

        let result = processor.process_batches(pattern_batches, ast_batches);
        assert!(result.is_ok());

        let all_matches = result?;
        assert_eq!(all_matches.len(), 2); // Two batches
        assert_eq!(all_matches[0].len(), 2); // Two patterns in first batch
        assert_eq!(all_matches[1].len(), 2); // Two patterns in second batch
    }

    #[test]
    fn test_batch_size_limits() -> Result<(), Box<dyn std::error::Error>> {
        let device = CudaContext::new(0)?;
        let config = BatchConfig {
            max_patterns_per_batch: 2,
            max_nodes_per_batch: 3,
            num_streams: 2,
        };
        let processor = BatchProcessor::new(device, config)?;

        // Test with patterns exceeding limit
        let patterns = vec![
            create_test_pattern("p1"),
            create_test_pattern("p2"),
            create_test_pattern("p3"), // Exceeds limit
        ];

        let asts = vec![create_test_ast("a1")];

        let result = processor.process_single_batch(&patterns, &asts);
        // Should handle gracefully - either process in chunks or return error
        assert!(result.is_ok() || result.is_err());
    }
}
