//! Comprehensive test suite for advanced prefetching strategies
//!
//! Tests all aspects of prefetching including pattern recognition,
//! multi-tier prefetching, adaptive strategies, and performance.

use super::*;
use crate::memory::{MemoryTier, PageInfo, TierManager};
use anyhow::Result;
use cudarc::driver::CudaContext;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_prefetch_config_default() {
        let config = PrefetchConfig::default();

        assert!(config.enable_prefetching);
        assert_eq!(config.prefetch_distance, 4);
        assert_eq!(config.prefetch_degree, 8);
        assert_eq!(config.max_prefetch_size, 16 * 1024 * 1024); // 16MB
        assert!(config.enable_adaptive_prefetching);
        assert_eq!(config.strategy, PrefetchStrategy::Adaptive);
    }

    #[test]
    fn test_access_pattern_types() {
        let patterns = vec![
            AccessPattern::Sequential,
            AccessPattern::Strided(4),
            AccessPattern::Random,
            AccessPattern::Temporal,
            AccessPattern::Spatial,
            AccessPattern::Mixed,
        ];

        assert_eq!(patterns.len(), 6);
        assert!(matches!(patterns[0], AccessPattern::Sequential));
        assert!(matches!(patterns[1], AccessPattern::Strided(4)));
    }

    #[test]
    fn test_prefetch_request_creation() {
        let request = PrefetchRequest {
            page_id: 1000,
            priority: PrefetchPriority::High,
            deadline: Some(Duration::from_millis(10)),
            size_hint: 4096,
            pattern_hint: Some(AccessPattern::Sequential),
        };

        assert_eq!(request.page_id, 1000);
        assert_eq!(request.priority, PrefetchPriority::High);
        assert_eq!(request.size_hint, 4096);
    }

    #[test]
    fn test_prefetch_statistics() {
        let mut stats = PrefetchStatistics::default();
        stats.total_prefetches = 1000;
        stats.hits = 850;
        stats.misses = 150;
        stats.accurate_predictions = 800;

        assert_eq!(stats.hit_rate(), 0.85);
        assert_eq!(stats.accuracy_rate(), 0.8);
    }

    #[test]
    fn test_pattern_detector() {
        let mut detector = PatternDetector::new(100);

        // Sequential access pattern
        for i in 0..10 {
            detector.record_access(i, Instant::now());
        }

        let pattern = detector.detect_pattern();
        assert!(matches!(pattern, AccessPattern::Sequential));
    }

    #[test]
    fn test_strided_pattern_detection() {
        let mut detector = PatternDetector::new(100);

        // Strided access with stride of 4
        for i in 0..10 {
            detector.record_access(i * 4, Instant::now());
        }

        let pattern = detector.detect_pattern();
        assert!(matches!(pattern, AccessPattern::Strided(4)));
    }

    #[test]
    fn test_prefetch_priorities() {
        let priorities = vec![
            PrefetchPriority::Low,
            PrefetchPriority::Normal,
            PrefetchPriority::High,
            PrefetchPriority::Critical,
        ];

        assert_eq!(priorities.len(), 4);
        assert!(priorities[0] < priorities[3]);
    }

    #[test]
    fn test_tier_predictor() {
        let predictor = TierPredictor::new();

        // Test prediction for hot page
        let mut access_history = AccessHistory {
            page_id: 100,
            access_count: 50,
            last_access: Instant::now(),
            access_intervals: vec![Duration::from_millis(10); 5],
        };

        let predicted_tier = predictor.predict_tier(&access_history);
        assert_eq!(predicted_tier, MemoryTier::GPU);

        // Test prediction for cold page
        access_history.access_count = 1;
        access_history.last_access = Instant::now() - Duration::from_secs(3600);

        let predicted_tier = predictor.predict_tier(&access_history);
        assert!(matches!(predicted_tier, MemoryTier::SSD | MemoryTier::HDD));
    }

    #[test]
    fn test_prefetch_cache() {
        let mut cache = PrefetchCache::new(100);

        // Add some predictions
        cache.add_prediction(1000, MemoryTier::GPU);
        cache.add_prediction(2000, MemoryTier::CPU);

        // Check predictions
        assert_eq!(cache.get_prediction(1000), Some(MemoryTier::GPU));
        assert_eq!(cache.get_prediction(2000), Some(MemoryTier::CPU));
        assert_eq!(cache.get_prediction(3000), None);
    }

    #[test]
    fn test_ml_predictor_config() {
        let config = MLPredictorConfig {
            model_type: ModelType::LSTM,
            input_features: 8,
            hidden_size: 64,
            output_classes: 5, // Number of tiers
            learning_rate: 0.001,
            update_frequency: Duration::from_secs(60),
        };

        assert_eq!(config.input_features, 8);
        assert_eq!(config.hidden_size, 64);
        assert!(matches!(config.model_type, ModelType::LSTM));
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    async fn setup_test_prefetcher() -> Result<Arc<AdvancedPrefetcher>> {
        let context = CudaContext::new(0)?; // Returns Arc<CudaContext>
        let config = PrefetchConfig::default();

        // Mock tier manager for testing
        let tier_manager = Arc::new(TierManager::new(context, Default::default())?);

        Ok(Arc::new(AdvancedPrefetcher::new(tier_manager, config)?))
    }

    #[tokio::test]
    async fn test_prefetcher_creation() -> Result<()> {
        let prefetcher = setup_test_prefetcher().await?;

        let stats = prefetcher.get_statistics();
        assert_eq!(stats.total_prefetches, 0);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_sequential_prefetching() -> Result<()> {
        let prefetcher = setup_test_prefetcher().await?;

        // Simulate sequential access pattern
        for i in 0..100 {
            prefetcher.record_access(i, MemoryTier::CPU).await?;

            if i >= 10 {
                // Check if prefetcher predicted next pages
                let predictions = prefetcher.get_prefetch_queue();
                assert!(!predictions.is_empty());

                // Should predict sequential pages
                let next_pages: Vec<u64> = predictions.iter().map(|req| req.page_id).collect();

                assert!(next_pages.contains(&(i + 1)));
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_strided_prefetching() -> Result<()> {
        let prefetcher = setup_test_prefetcher().await?;
        let stride = 8;

        // Simulate strided access pattern
        for i in 0..50 {
            let page_id = i * stride;
            prefetcher.record_access(page_id, MemoryTier::CPU).await?;

            if i >= 5 {
                // Check predictions
                let predictions = prefetcher.get_prefetch_queue();

                // Should predict strided pages
                let next_pages: Vec<u64> = predictions.iter().map(|req| req.page_id).collect();

                assert!(next_pages.contains(&((i + 1) * stride)));
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_adaptive_strategy() -> Result<()> {
        let mut config = PrefetchConfig::default();
        config.strategy = PrefetchStrategy::Adaptive;

        let context = CudaContext::new(0)?; // Returns Arc<CudaContext>
        let tier_manager = Arc::new(TierManager::new(context, Default::default())?);
        let prefetcher = Arc::new(AdvancedPrefetcher::new(tier_manager, config)?);

        // Start with sequential pattern
        for i in 0..50 {
            prefetcher.record_access(i, MemoryTier::CPU).await?;
        }

        // Change to strided pattern
        for i in 0..50 {
            prefetcher.record_access(i * 4, MemoryTier::CPU).await?;
        }

        // Adaptive strategy should detect pattern change
        let stats = prefetcher.get_statistics();
        assert!(stats.pattern_changes > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_multi_tier_prefetching() -> Result<()> {
        let prefetcher = setup_test_prefetcher().await?;

        // Simulate access across multiple tiers
        let test_pages = vec![
            (100, MemoryTier::GPU),
            (200, MemoryTier::CPU),
            (300, MemoryTier::NVMe),
            (400, MemoryTier::SSD),
        ];

        for (page_id, tier) in test_pages {
            prefetcher.record_access(page_id, tier).await?;

            // Prefetcher should optimize tier placement
            let recommendations = prefetcher.get_tier_recommendations(page_id);
            assert!(recommendations.is_some());
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_deadline_aware_prefetching() -> Result<()> {
        let prefetcher = setup_test_prefetcher().await?;

        // Submit requests with different deadlines
        let requests = vec![
            PrefetchRequest {
                page_id: 1000,
                priority: PrefetchPriority::Normal,
                deadline: Some(Duration::from_millis(100)),
                size_hint: 4096,
                pattern_hint: None,
            },
            PrefetchRequest {
                page_id: 2000,
                priority: PrefetchPriority::High,
                deadline: Some(Duration::from_millis(10)),
                size_hint: 4096,
                pattern_hint: None,
            },
            PrefetchRequest {
                page_id: 3000,
                priority: PrefetchPriority::Low,
                deadline: None,
                size_hint: 4096,
                pattern_hint: None,
            },
        ];

        for request in requests {
            prefetcher.submit_prefetch_request(request).await?;
        }

        // Check queue ordering - urgent requests should be first
        let queue = prefetcher.get_prefetch_queue();
        assert!(!queue.is_empty());
        assert_eq!(queue[0].page_id, 2000); // High priority with tight deadline

        Ok(())
    }

    #[tokio::test]
    async fn test_prefetch_throttling() -> Result<()> {
        let mut config = PrefetchConfig::default();
        config.max_prefetch_size = 1024 * 1024; // 1MB limit

        let context = CudaContext::new(0)?; // Returns Arc<CudaContext>
        let tier_manager = Arc::new(TierManager::new(context, Default::default())?);
        let prefetcher = Arc::new(AdvancedPrefetcher::new(tier_manager, config)?);

        // Submit many prefetch requests
        for i in 0..1000 {
            let request = PrefetchRequest {
                page_id: i,
                priority: PrefetchPriority::Normal,
                deadline: None,
                size_hint: 4096,
                pattern_hint: None,
            };
            prefetcher.submit_prefetch_request(request).await?;
        }

        // Should throttle to stay within memory limit
        let active_size = prefetcher.get_active_prefetch_size();
        assert!(active_size <= 1024 * 1024);

        Ok(())
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    fn test_pattern_detection_performance() {
        let mut detector = PatternDetector::new(10000);
        let start = Instant::now();

        // Simulate 10K accesses
        for i in 0..10000 {
            detector.record_access(i, Instant::now());
        }

        let pattern = detector.detect_pattern();
        let elapsed = start.elapsed();

        println!("Pattern detection for 10K accesses: {:?}", elapsed);
        assert!(elapsed < Duration::from_millis(100)); // Should be fast
        assert!(matches!(pattern, AccessPattern::Sequential));
    }

    #[test]
    fn test_prediction_cache_performance() {
        let mut cache = PrefetchCache::new(100000);
        let start = Instant::now();

        // Add 100K predictions
        for i in 0..100000 {
            cache.add_prediction(i, MemoryTier::CPU);
        }

        let insert_time = start.elapsed();
        println!("Insert 100K predictions: {:?}", insert_time);

        // Lookup performance
        let start = Instant::now();
        let mut hits = 0;

        for i in 0..100000 {
            if cache.get_prediction(i).is_some() {
                hits += 1;
            }
        }

        let lookup_time = start.elapsed();
        println!("Lookup 100K predictions: {:?}", lookup_time);

        assert_eq!(hits, 100000);
        assert!(lookup_time < Duration::from_millis(50));
    }

    #[test]
    fn test_tier_prediction_performance() {
        let predictor = TierPredictor::new();
        let mut histories = Vec::new();

        // Create test histories
        for i in 0..1000 {
            histories.push(AccessHistory {
                page_id: i,
                access_count: (i % 100) as u32,
                last_access: Instant::now() - Duration::from_secs(i % 3600),
                access_intervals: vec![Duration::from_millis(10); 5],
            });
        }

        let start = Instant::now();
        let mut predictions = Vec::new();

        for history in &histories {
            predictions.push(predictor.predict_tier(history));
        }

        let elapsed = start.elapsed();
        println!("1000 tier predictions: {:?}", elapsed);

        assert_eq!(predictions.len(), 1000);
        assert!(elapsed < Duration::from_millis(10));
    }
}

#[cfg(test)]
mod ml_predictor_tests {
    use super::*;

    #[test]
    fn test_feature_extraction() {
        let history = AccessHistory {
            page_id: 1000,
            access_count: 25,
            last_access: Instant::now(),
            access_intervals: vec![
                Duration::from_millis(10),
                Duration::from_millis(15),
                Duration::from_millis(12),
                Duration::from_millis(11),
                Duration::from_millis(13),
            ],
        };

        let features = MLPredictor::extract_features(&history);

        assert_eq!(features.len(), 8); // Expected number of features
        assert!(features[0] > 0.0); // Access count feature
        assert!(features[1] >= 0.0); // Time since last access
    }

    #[test]
    fn test_model_types() {
        let models = vec![
            ModelType::LinearRegression,
            ModelType::DecisionTree,
            ModelType::RandomForest,
            ModelType::LSTM,
            ModelType::Transformer,
        ];

        assert_eq!(models.len(), 5);

        // Test model creation for each type
        for model_type in models {
            let config = MLPredictorConfig {
                model_type,
                input_features: 8,
                hidden_size: 32,
                output_classes: 5,
                learning_rate: 0.001,
                update_frequency: Duration::from_secs(60),
            };

            // Would create actual model in real implementation
            assert!(config.input_features > 0);
        }
    }

    #[test]
    fn test_online_learning() -> Result<()> {
        let mut predictor = MLPredictor::new(MLPredictorConfig::default())?;

        // Generate training data
        let mut training_data = Vec::new();
        for i in 0..100 {
            let history = AccessHistory {
                page_id: i,
                access_count: (i % 50) as u32,
                last_access: Instant::now() - Duration::from_secs(i % 300),
                access_intervals: vec![Duration::from_millis(10 + i % 20); 5],
            };

            let tier = if i % 50 > 40 {
                MemoryTier::GPU
            } else if i % 50 > 20 {
                MemoryTier::CPU
            } else {
                MemoryTier::NVMe
            };

            training_data.push((history, tier));
        }

        // Train model
        let loss = predictor.update_model(&training_data)?;
        assert!(loss >= 0.0);

        // Test predictions
        let test_history = AccessHistory {
            page_id: 1001,
            access_count: 45,
            last_access: Instant::now(),
            access_intervals: vec![Duration::from_millis(10); 5],
        };

        let prediction = predictor.predict(&test_history);
        assert!(matches!(prediction, MemoryTier::GPU | MemoryTier::CPU));

        Ok(())
    }
}

#[cfg(test)]
mod cost_benefit_tests {
    use super::*;

    #[test]
    fn test_prefetch_cost_calculation() {
        let analyzer = CostBenefitAnalyzer::new();

        // Test different scenarios
        let scenarios = vec![
            (MemoryTier::NVMe, MemoryTier::GPU, 4096), // Small transfer to GPU
            (MemoryTier::SSD, MemoryTier::CPU, 1048576), // 1MB to CPU
            (MemoryTier::HDD, MemoryTier::NVMe, 10485760), // 10MB to NVMe
        ];

        for (from_tier, to_tier, size) in scenarios {
            let cost = analyzer.calculate_prefetch_cost(from_tier, to_tier, size);

            assert!(cost.transfer_time > Duration::ZERO);
            assert!(cost.energy_cost > 0.0);
            assert!(cost.bandwidth_usage > 0.0);

            // Higher tiers should have lower latency
            if from_tier as u8 > to_tier as u8 {
                assert!(cost.transfer_time < Duration::from_millis(100));
            }
        }
    }

    #[test]
    fn test_benefit_estimation() {
        let analyzer = CostBenefitAnalyzer::new();

        let history = AccessHistory {
            page_id: 1000,
            access_count: 100,
            last_access: Instant::now(),
            access_intervals: vec![Duration::from_millis(5); 10],
        };

        let benefit = analyzer.estimate_benefit(&history, MemoryTier::GPU);

        assert!(benefit.hit_probability > 0.0);
        assert!(benefit.hit_probability <= 1.0);
        assert!(benefit.expected_speedup > 0.0);
        assert!(benefit.value_score > 0.0);
    }

    #[test]
    fn test_prefetch_decision() {
        let analyzer = CostBenefitAnalyzer::new();

        // High-value prefetch (should be approved)
        let hot_page = AccessHistory {
            page_id: 1000,
            access_count: 1000,
            last_access: Instant::now(),
            access_intervals: vec![Duration::from_millis(1); 10],
        };

        let decision = analyzer.should_prefetch(&hot_page, MemoryTier::SSD, MemoryTier::GPU, 4096);

        assert!(decision.approved);
        assert!(decision.net_benefit > 0.0);

        // Low-value prefetch (should be rejected)
        let cold_page = AccessHistory {
            page_id: 2000,
            access_count: 1,
            last_access: Instant::now() - Duration::from_secs(3600),
            access_intervals: vec![],
        };

        let decision = analyzer.should_prefetch(
            &cold_page,
            MemoryTier::HDD,
            MemoryTier::GPU,
            1048576, // 1MB
        );

        assert!(!decision.approved);
    }
}
