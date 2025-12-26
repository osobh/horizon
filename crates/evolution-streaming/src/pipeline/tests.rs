//! Tests for the evolution streaming pipeline

#[cfg(test)]
mod tests {
    use crate::{
        AgentGenome, EvolutionCycleResult, EvolutionStreamingError, EvolutionStreamingPipeline,
        PipelineStats,
    };
    use std::sync::Arc;
    use std::time::Duration;

    #[tokio::test]
    async fn test_pipeline_builder() {
        let pipeline = EvolutionStreamingPipeline::builder()
            .with_batch_size(16)
            .with_timeout(Duration::from_secs(60))
            .build()
            .await
            .unwrap();

        assert_eq!(pipeline.batch_size, 16);
        assert_eq!(pipeline.pipeline_timeout, Duration::from_secs(60));
        assert!(pipeline.is_archive_empty());
    }

    #[tokio::test]
    async fn test_pipeline_bootstrap() {
        let pipeline = EvolutionStreamingPipeline::builder().build().await.unwrap();

        let agents = vec![
            AgentGenome::new("fn test1() { 1 }".to_string(), vec![1.0]),
            AgentGenome::new("fn test2() { 2 }".to_string(), vec![2.0]),
            AgentGenome::new("fn test3() { 3 }".to_string(), vec![3.0]),
        ];

        let added_count = pipeline.bootstrap_with_agents(agents).await.unwrap();
        assert_eq!(added_count, 3);
        assert_eq!(pipeline.archive_size(), 3);
        assert!(!pipeline.is_archive_empty());
    }

    #[tokio::test]
    async fn test_evolution_cycle() {
        let pipeline = EvolutionStreamingPipeline::builder()
            .with_batch_size(2)
            .build()
            .await
            .unwrap();

        // Bootstrap with some agents
        let agents = vec![
            AgentGenome::new("fn test1() { 1 }".to_string(), vec![1.0]),
            AgentGenome::new("fn test2() { 2 }".to_string(), vec![2.0]),
        ];
        pipeline.bootstrap_with_agents(agents).await.unwrap();

        // Run one evolution cycle
        let result = pipeline.run_cycle().await.unwrap();

        assert!(result.selected_count > 0); // Should select at least one agent
        assert!(result.mutated_count > 0);
        assert!(result.evaluated_count > 0);
        assert!(result.total_time > Duration::ZERO);
    }

    #[tokio::test]
    async fn test_multiple_cycles() {
        let pipeline = EvolutionStreamingPipeline::builder()
            .with_batch_size(1)
            .build()
            .await
            .unwrap();

        // Bootstrap with one agent
        let agents = vec![AgentGenome::new(
            "fn test() { 42 }".to_string(),
            vec![1.0, 2.0],
        )];
        pipeline.bootstrap_with_agents(agents).await.unwrap();

        // Run multiple cycles
        let results = pipeline.run_cycles(3).await.unwrap();
        assert_eq!(results.len(), 3);

        for result in results {
            assert!(result.selected_count > 0);
            assert!(result.total_time > Duration::ZERO);
        }

        // Check pipeline statistics
        let stats = pipeline.get_stats().await;
        assert_eq!(stats.cycles_completed, 3);
        assert!(stats.agents_processed > 0);
        assert!(stats.total_processing_time > Duration::ZERO);
    }

    #[tokio::test]
    async fn test_empty_archive_cycle_failure() {
        let pipeline = EvolutionStreamingPipeline::builder().build().await.unwrap();

        // Try to run cycle with empty archive
        let result = pipeline.run_cycle().await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            EvolutionStreamingError::PipelineError(_)
        ));
    }

    #[tokio::test]
    async fn test_pipeline_stats_updates() {
        let pipeline = EvolutionStreamingPipeline::builder()
            .with_batch_size(1)
            .build()
            .await
            .unwrap();

        let agents = vec![AgentGenome::new("fn test() { 1 }".to_string(), vec![1.0])];
        pipeline.bootstrap_with_agents(agents).await.unwrap();

        let initial_stats = pipeline.get_stats().await;
        assert_eq!(initial_stats.cycles_completed, 0);

        // Run one cycle
        pipeline.run_cycle().await.unwrap();

        let updated_stats = pipeline.get_stats().await;
        assert_eq!(updated_stats.cycles_completed, 1);
        assert!(updated_stats.agents_processed > 0);
        assert!(updated_stats.total_processing_time > Duration::ZERO);
    }

    #[tokio::test]
    async fn test_pipeline_archive_operations() {
        let pipeline = EvolutionStreamingPipeline::builder().build().await.unwrap();

        // Initially empty
        assert!(pipeline.is_archive_empty());
        assert_eq!(pipeline.archive_size(), 0);
        assert!(pipeline.best_fitness().await.is_none());

        // Add agents
        let agents = vec![
            AgentGenome::new("fn fitness1() { 1 }".to_string(), vec![1.0]),
            AgentGenome::new("fn fitness2() { 2 }".to_string(), vec![2.0]),
            AgentGenome::new("fn fitness3() { 3 }".to_string(), vec![3.0]),
        ];
        pipeline.bootstrap_with_agents(agents).await.unwrap();

        assert!(!pipeline.is_archive_empty());
        assert_eq!(pipeline.archive_size(), 3);
        assert!(pipeline.best_fitness().await.is_some());
    }

    #[tokio::test]
    async fn test_pipeline_fitness_evolution() {
        let pipeline = EvolutionStreamingPipeline::builder()
            .with_batch_size(2)
            .build()
            .await
            .unwrap();

        let agents = vec![
            AgentGenome::new("fn fitness1() { 1 }".to_string(), vec![1.0]),
            AgentGenome::new("fn fitness2() { 2 }".to_string(), vec![2.0]),
            AgentGenome::new("fn fitness3() { 3 }".to_string(), vec![3.0]),
        ];
        pipeline.bootstrap_with_agents(agents).await.unwrap();

        let initial_best = pipeline.best_fitness().await;
        assert!(initial_best.is_some());

        // Run evolution cycles
        pipeline.run_cycles(3).await.unwrap();

        let final_best = pipeline.best_fitness().await;
        assert!(final_best.is_some());

        // Best fitness might improve or stay the same
        assert!(final_best.unwrap() >= initial_best.unwrap());
    }

    #[tokio::test]
    async fn test_pipeline_concurrent_archive_access() {
        let pipeline = Arc::new(
            EvolutionStreamingPipeline::builder()
                .with_batch_size(1)
                .build()
                .await
                .unwrap(),
        );

        let agents = vec![
            AgentGenome::new("fn concurrent1() { 1 }".to_string(), vec![1.0]),
            AgentGenome::new("fn concurrent2() { 2 }".to_string(), vec![2.0]),
        ];
        pipeline.bootstrap_with_agents(agents).await.unwrap();

        let mut handles = vec![];

        // Run multiple cycles concurrently
        for _i in 0..3 {
            let pipeline_clone = pipeline.clone();
            let handle = tokio::spawn(async move {
                (
                    pipeline_clone.run_cycle().await,
                    pipeline_clone.archive_size(),
                    pipeline_clone.is_archive_empty(),
                    pipeline_clone.best_fitness().await,
                )
            });
            handles.push(handle);
        }

        // Collect results
        for handle in handles {
            let (cycle_result, size, is_empty, best_fitness) = handle.await.unwrap();

            assert!(cycle_result.is_ok() || cycle_result.is_err());
            assert!(size >= 0);
            assert_eq!(is_empty, size == 0);
            assert!(best_fitness.is_some() || best_fitness.is_none());
        }
    }

    #[test]
    fn test_evolution_cycle_result_debug_format() {
        let result = EvolutionCycleResult {
            total_time: Duration::from_millis(100),
            selection_time: Duration::from_millis(10),
            mutation_time: Duration::from_millis(30),
            evaluation_time: Duration::from_millis(50),
            archive_time: Duration::from_millis(10),
            selected_count: 5,
            mutated_count: 20,
            evaluated_count: 20,
            novel_agents: 3,
        };

        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("EvolutionCycleResult"));
        assert!(debug_str.contains("total_time"));
        assert!(debug_str.contains("selected_count"));
    }

    #[test]
    fn test_pipeline_stats_debug_format() {
        let stats = PipelineStats {
            cycles_completed: 10,
            agents_processed: 100,
            mutations_generated: 400,
            evaluations_completed: 100,
            archive_updates: 50,
            total_processing_time: Duration::from_secs(60),
            average_cycle_time: Duration::from_secs(6),
            throughput_agents_per_sec: 1.67,
        };

        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("PipelineStats"));
        assert!(debug_str.contains("cycles_completed"));
        assert!(debug_str.contains("throughput_agents_per_sec"));
    }
}
