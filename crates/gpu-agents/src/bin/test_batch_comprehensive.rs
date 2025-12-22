//! Comprehensive batch processor tests

use cudarc::driver::CudaDevice;
use gpu_agents::synthesis::batch_processor::{BatchConfig, BatchProcessor};
use gpu_agents::synthesis::{AstNode, Match, NodeType, Pattern};
use std::time::Instant;

fn create_patterns(count: usize) -> Vec<Pattern> {
    (0..count)
        .map(|i| Pattern {
            node_type: NodeType::Variable,
            children: vec![],
            value: Some(format!("var{}", i)),
        })
        .collect()
}

fn create_asts(count: usize) -> Vec<AstNode> {
    (0..count)
        .map(|i| AstNode {
            node_type: NodeType::Variable,
            children: vec![],
            value: Some(format!("var{}", i % 10)), // Some will match patterns
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    println!("🧪 Comprehensive Batch Processor Tests");
    println!("======================================");

    let device = CudaDevice::new(0)?;

    // Test 1: Single batch with exact matches
    println!("\n1. Testing single batch with exact matches...");
    {
        let config = BatchConfig::default();
        let processor = BatchProcessor::new(device.clone(), config)?;

        let patterns = vec![
            Pattern {
                node_type: NodeType::Variable,
                children: vec![],
                value: Some("x".to_string()),
            },
            Pattern {
                node_type: NodeType::Variable,
                children: vec![],
                value: Some("y".to_string()),
            },
        ];

        let asts = vec![
            AstNode {
                node_type: NodeType::Variable,
                children: vec![],
                value: Some("x".to_string()),
            },
            AstNode {
                node_type: NodeType::Variable,
                children: vec![],
                value: Some("y".to_string()),
            },
            AstNode {
                node_type: NodeType::Variable,
                children: vec![],
                value: Some("z".to_string()),
            },
        ];

        let matches = processor.process_single_batch(&patterns, &asts)?;
        println!("   Patterns: {}, ASTs: {}", patterns.len(), asts.len());
        println!(
            "   Matches per pattern: [{}, {}]",
            matches[0].len(),
            matches[1].len()
        );
        assert_eq!(matches.len(), 2);
        assert!(!matches[0].is_empty(), "Pattern 'x' should have matches");
        assert!(!matches[1].is_empty(), "Pattern 'y' should have matches");
        println!("   ✅ Single batch test passed");
    }

    // Test 2: Batch size limit handling
    println!("\n2. Testing batch size limit handling...");
    {
        let config = BatchConfig {
            max_patterns_per_batch: 10,
            max_nodes_per_batch: 100,
            num_streams: 2,
        };
        let processor = BatchProcessor::new(device.clone(), config)?;

        let patterns = create_patterns(25); // Exceeds limit of 10
        let asts = create_asts(50);

        let start = Instant::now();
        let matches = processor.process_single_batch(&patterns, &asts)?;
        let elapsed = start.elapsed();

        println!("   Patterns: {} (limit: 10)", patterns.len());
        println!("   Processed in: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
        assert_eq!(matches.len(), patterns.len());
        println!("   ✅ Batch limit test passed");
    }

    // Test 3: Multiple batch processing
    println!("\n3. Testing multiple batch processing...");
    {
        let config = BatchConfig::default();
        let processor = BatchProcessor::new(device.clone(), config)?;

        let pattern_batches = vec![create_patterns(5), create_patterns(10), create_patterns(3)];

        let ast_batches = vec![create_asts(20), create_asts(30), create_asts(15)];

        let start = Instant::now();
        let all_matches =
            processor.process_batches(pattern_batches.clone(), ast_batches.clone())?;
        let elapsed = start.elapsed();

        println!("   Batches: {}", pattern_batches.len());
        println!(
            "   Total patterns: {}",
            pattern_batches.iter().map(|b| b.len()).sum::<usize>()
        );
        println!(
            "   Total ASTs: {}",
            ast_batches.iter().map(|b| b.len()).sum::<usize>()
        );
        println!("   Processed in: {:.2}ms", elapsed.as_secs_f64() * 1000.0);

        assert_eq!(all_matches.len(), 3);
        assert_eq!(all_matches[0].len(), 5);
        assert_eq!(all_matches[1].len(), 10);
        assert_eq!(all_matches[2].len(), 3);
        println!("   ✅ Multiple batch test passed");
    }

    // Test 4: Performance comparison
    println!("\n4. Testing performance with different batch sizes...");
    {
        let patterns = create_patterns(100);
        let asts = create_asts(1000);

        // Small batches
        let config1 = BatchConfig {
            max_patterns_per_batch: 10,
            max_nodes_per_batch: 1000,
            num_streams: 4,
        };
        let processor1 = BatchProcessor::new(device.clone(), config1)?;

        let start1 = Instant::now();
        let _ = processor1.process_single_batch(&patterns, &asts)?;
        let time1 = start1.elapsed();

        // Large batches
        let config2 = BatchConfig {
            max_patterns_per_batch: 100,
            max_nodes_per_batch: 10000,
            num_streams: 4,
        };
        let processor2 = BatchProcessor::new(device.clone(), config2)?;

        let start2 = Instant::now();
        let _ = processor2.process_single_batch(&patterns, &asts)?;
        let time2 = start2.elapsed();

        println!(
            "   Small batches (10): {:.2}ms",
            time1.as_secs_f64() * 1000.0
        );
        println!(
            "   Large batches (100): {:.2}ms",
            time2.as_secs_f64() * 1000.0
        );
        println!(
            "   Speedup: {:.2}x",
            time1.as_secs_f64() / time2.as_secs_f64()
        );
        println!("   ✅ Performance test completed");
    }

    println!("\n✅ All tests passed!");
    Ok(())
}
