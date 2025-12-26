//! Tests for swarm behavior visualization

#[cfg(test)]
mod tests {
    use gpu_agents::{
        ChartType, DataExportFormat, GpuSwarm, GpuSwarmConfig, RenderingBackend,
        VisualizationConfig, VisualizationManager,
    };

    use tempfile::TempDir;

    #[test]
    fn test_visualization_config_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = VisualizationConfig {
            output_directory: temp_dir.path().to_path_buf(),
            enable_real_time: true,
            update_interval_ms: 100,
            max_history_frames: 1000,
            chart_types: vec![
                ChartType::SwarmDistribution,
                ChartType::FitnessEvolution,
                ChartType::DiversityMetrics,
                ChartType::PerformanceTimeline,
            ],
            export_format: DataExportFormat::JSON,
            rendering_backend: RenderingBackend::Plotters,
            image_width: 800,
            image_height: 600,
            enable_animation: true,
            fps: 30,
            color_scheme: "default".to_string(),
        };

        assert_eq!(config.update_interval_ms, 100);
        assert_eq!(config.chart_types.len(), 4);
        assert_eq!(config.export_format, DataExportFormat::JSON);
        assert_eq!(config.rendering_backend, RenderingBackend::Plotters);
        assert!(config.enable_real_time);
        assert!(config.enable_animation);
        assert_eq!(config.fps, 30);
    }

    #[test]
    fn test_visualization_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = VisualizationConfig::default_with_output_dir(temp_dir.path());
        let manager = VisualizationManager::new(config);

        assert!(manager.is_ok());
        let manager = manager?;

        assert_eq!(manager.frame_count(), 0);
        assert!(!manager.is_recording());
        assert_eq!(manager.supported_chart_types().len(), 4);
    }

    #[test]
    fn test_swarm_data_capture() {
        let temp_dir = TempDir::new().unwrap();
        let config = VisualizationConfig::default_with_output_dir(temp_dir.path());
        let mut manager = VisualizationManager::new(config).unwrap();

        // Create test swarm
        let mut swarm = GpuSwarm::new(GpuSwarmConfig::default())?;
        swarm.initialize(100)?;

        // Run swarm to generate some data
        for _ in 0..10 {
            swarm.step().unwrap();
        }

        // Capture swarm data
        let frame_data = manager.capture_frame(&swarm).unwrap();

        assert_eq!(frame_data.agent_count, 100);
        assert!(frame_data.timestamp > 0);
        assert!(!frame_data.agent_positions.is_empty());
        assert!(!frame_data.fitness_values.is_empty());
        assert!(frame_data.average_fitness >= 0.0);
        assert!(frame_data.diversity_score >= 0.0);

        // Verify agent positions are reasonable
        for position in &frame_data.agent_positions {
            assert_eq!(position.len(), 3); // x, y, z coordinates
            for &coord in position {
                assert!(coord.is_finite());
            }
        }

        // Verify fitness values are valid
        for &fitness in &frame_data.fitness_values {
            assert!(fitness >= 0.0);
            assert!(fitness.is_finite());
        }
    }

    #[test]
    fn test_chart_generation() {
        let temp_dir = TempDir::new().unwrap();
        let config = VisualizationConfig::default_with_output_dir(temp_dir.path());
        let mut manager = VisualizationManager::new(config).unwrap();

        // Create test swarm and generate data
        let mut swarm = GpuSwarm::new(GpuSwarmConfig::default())?;
        swarm.initialize(50)?;

        // Capture multiple frames to build history
        for _step in 0..20 {
            swarm.step().unwrap();
            manager.capture_frame(&swarm).unwrap();
        }

        // Generate different types of charts
        let chart_types = vec![
            ChartType::SwarmDistribution,
            ChartType::FitnessEvolution,
            ChartType::DiversityMetrics,
            ChartType::PerformanceTimeline,
        ];

        for chart_type in chart_types {
            let chart_path = manager.generate_chart(chart_type.clone()).unwrap();

            assert!(chart_path.exists());
            assert!(chart_path.extension().unwrap() == "png");

            // Check file is not empty
            let metadata = std::fs::metadata(&chart_path).unwrap();
            assert!(metadata.len() > 0);

            println!("Generated chart: {:?} at {:?}", chart_type, chart_path);
        }

        // Verify frame history
        assert_eq!(manager.frame_count(), 20);
    }

    #[test]
    fn test_real_time_visualization() {
        let temp_dir = TempDir::new().unwrap();
        let config = VisualizationConfig {
            enable_real_time: true,
            update_interval_ms: 50, // Fast updates for testing
            ..VisualizationConfig::default_with_output_dir(temp_dir.path())
        };
        let mut manager = VisualizationManager::new(config)?;

        // Create test swarm
        let mut swarm = GpuSwarm::new(GpuSwarmConfig::default())?;
        swarm.initialize(30).unwrap();

        // Start real-time visualization
        manager.start_real_time_visualization().unwrap();
        assert!(manager.is_recording());

        let start_time = std::time::Instant::now();

        // Run simulation for a short time
        while start_time.elapsed().as_millis() < 200 {
            swarm.step().unwrap();
            manager.update_real_time(&swarm).unwrap();

            // Small delay to simulate real-time processing
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Stop real-time visualization
        manager.stop_real_time_visualization().unwrap();
        assert!(!manager.is_recording());

        // Check that frames were captured
        assert!(manager.frame_count() > 0);

        // Verify real-time charts were generated
        let output_files = std::fs::read_dir(temp_dir.path()).unwrap();
        let chart_files: Vec<_> = output_files
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .map(|ext| ext == "png")
                    .unwrap_or(false)
            })
            .collect();

        assert!(!chart_files.is_empty(), "No chart files were generated");
    }

    #[test]
    fn test_data_export() {
        let temp_dir = TempDir::new().unwrap();
        let config = VisualizationConfig::default_with_output_dir(temp_dir.path());
        let mut manager = VisualizationManager::new(config).unwrap();

        // Create test swarm and generate data
        let mut swarm = GpuSwarm::new(GpuSwarmConfig::default())?;
        swarm.initialize(25)?;

        // Capture frames
        for _ in 0..15 {
            swarm.step().unwrap();
            manager.capture_frame(&swarm).unwrap();
        }

        // Export data in different formats
        let formats = vec![
            DataExportFormat::JSON,
            DataExportFormat::CSV,
            DataExportFormat::Parquet,
        ];

        for format in formats {
            let export_path = manager.export_data(format.clone()).unwrap();

            assert!(export_path.exists());

            let expected_extension = match format {
                DataExportFormat::JSON => "json",
                DataExportFormat::CSV => "csv",
                DataExportFormat::Parquet => "parquet",
            };

            assert_eq!(export_path.extension().unwrap(), expected_extension);

            // Check file is not empty
            let metadata = std::fs::metadata(&export_path).unwrap();
            assert!(metadata.len() > 0);

            println!("Exported data: {:?} to {:?}", format, export_path);
        }

        // Test data integrity for JSON export
        let json_path = manager.export_data(DataExportFormat::JSON).unwrap();
        let json_content = std::fs::read_to_string(&json_path).unwrap();

        // Should be valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&json_content).unwrap();
        assert!(parsed.is_object());

        // Should contain expected fields
        let obj = parsed.as_object().unwrap();
        assert!(obj.contains_key("frames"));
        assert!(obj.contains_key("metadata"));
        assert!(obj.contains_key("summary"));
    }

    #[test]
    fn test_animation_generation() {
        let temp_dir = TempDir::new().unwrap();
        let config = VisualizationConfig {
            enable_animation: true,
            fps: 10, // Lower FPS for faster testing
            ..VisualizationConfig::default_with_output_dir(temp_dir.path())
        };
        let mut manager = VisualizationManager::new(config)?;

        // Create test swarm
        let mut swarm = GpuSwarm::new(GpuSwarmConfig::default())?;
        swarm.initialize(40).unwrap();

        // Capture enough frames for animation
        for _ in 0..30 {
            swarm.step().unwrap();
            manager.capture_frame(&swarm).unwrap();
        }

        // Generate animation
        let animation_path = manager
            .generate_animation(ChartType::SwarmDistribution)
            .unwrap();

        assert!(animation_path.exists());
        assert_eq!(animation_path.extension().unwrap(), "gif");

        // Check file is not empty
        let metadata = std::fs::metadata(&animation_path).unwrap();
        assert!(metadata.len() > 0);

        println!("Generated animation at {:?}", animation_path);
    }

    #[test]
    fn test_performance_dashboard() {
        let temp_dir = TempDir::new().unwrap();
        let config = VisualizationConfig::default_with_output_dir(temp_dir.path());
        let mut manager = VisualizationManager::new(config).unwrap();

        // Create test swarm
        let mut swarm = GpuSwarm::new(GpuSwarmConfig::default())?;
        swarm.initialize(80)?;

        // Generate performance data
        for _ in 0..25 {
            swarm.step().unwrap();
            manager.capture_frame(&swarm).unwrap();
        }

        // Generate performance dashboard
        let dashboard_path = manager.generate_performance_dashboard().unwrap();

        assert!(dashboard_path.exists());
        assert_eq!(dashboard_path.extension().unwrap(), "html");

        // Read HTML content
        let html_content = std::fs::read_to_string(&dashboard_path).unwrap();

        // Check for expected dashboard elements
        assert!(html_content.contains("<html"));
        assert!(html_content.contains("Performance Dashboard"));
        assert!(html_content.contains("Agent Count"));
        assert!(html_content.contains("Fitness Evolution"));
        assert!(html_content.contains("GPU Memory Usage"));

        println!("Generated dashboard at {:?}", dashboard_path);
    }

    #[test]
    fn test_visualization_metrics() {
        let temp_dir = TempDir::new().unwrap();
        let config = VisualizationConfig::default_with_output_dir(temp_dir.path());
        let mut manager = VisualizationManager::new(config).unwrap();

        // Create test swarm
        let mut swarm = GpuSwarm::new(GpuSwarmConfig::default())?;
        swarm.initialize(60)?;

        let start_time = std::time::Instant::now();

        // Generate data and charts
        for _ in 0..10 {
            swarm.step().unwrap();
            manager.capture_frame(&swarm).unwrap();
        }

        // Generate some charts
        manager.generate_chart(ChartType::FitnessEvolution).unwrap();
        manager
            .generate_chart(ChartType::SwarmDistribution)
            .unwrap();

        let elapsed = start_time.elapsed();

        // Get visualization metrics
        let metrics = manager.get_visualization_metrics().unwrap();

        assert_eq!(metrics.total_frames_captured, 10);
        assert_eq!(metrics.charts_generated, 2);
        assert!(metrics.total_processing_time_ms >= 0.0);
        assert!(metrics.average_frame_capture_time_ms >= 0.0);
        assert!(metrics.data_size_bytes > 0);
        assert_eq!(metrics.export_formats_used.len(), 0); // No exports yet

        // Processing should be reasonably fast
        assert!(
            elapsed.as_millis() < 5000,
            "Visualization processing too slow"
        );

        println!("Visualization metrics: {:?}", metrics);
    }

    #[test]
    fn test_multi_chart_dashboard() {
        let temp_dir = TempDir::new().unwrap();
        let config = VisualizationConfig {
            chart_types: vec![
                ChartType::SwarmDistribution,
                ChartType::FitnessEvolution,
                ChartType::DiversityMetrics,
                ChartType::PerformanceTimeline,
            ],
            ..VisualizationConfig::default_with_output_dir(temp_dir.path())
        };
        let mut manager = VisualizationManager::new(config).unwrap();

        // Create test swarm
        let mut swarm = GpuSwarm::new(GpuSwarmConfig::default()).unwrap();
        swarm.initialize(120).unwrap();

        // Generate diverse data
        for _ in 0..20 {
            swarm.step().unwrap();
            manager.capture_frame(&swarm).unwrap();
        }

        // Generate multi-chart dashboard
        let dashboard_path = manager.generate_multi_chart_dashboard().unwrap();

        assert!(dashboard_path.exists());
        assert_eq!(dashboard_path.extension().unwrap(), "html");

        // Read and verify dashboard content
        let html_content = std::fs::read_to_string(&dashboard_path).unwrap();

        // Should contain all chart types
        assert!(html_content.contains("Swarm Distribution"));
        assert!(html_content.contains("Fitness Evolution"));
        assert!(html_content.contains("Diversity Metrics"));
        assert!(html_content.contains("Performance Timeline"));

        // Should contain interactive elements
        assert!(html_content.contains("javascript") || html_content.contains("script"));

        println!("Generated multi-chart dashboard at {:?}", dashboard_path);
    }

    #[test]
    fn test_memory_efficient_visualization() {
        let temp_dir = TempDir::new().unwrap();
        let config = VisualizationConfig {
            max_history_frames: 50, // Limit history to test memory management
            ..VisualizationConfig::default_with_output_dir(temp_dir.path())
        };
        let mut manager = VisualizationManager::new(config)?;

        // Create test swarm
        let mut swarm = GpuSwarm::new(GpuSwarmConfig::default())?;
        swarm.initialize(100)?;

        // Capture more frames than the limit
        for i in 0..80 {
            swarm.step().unwrap();
            manager.capture_frame(&swarm).unwrap();

            // Memory usage should not grow unbounded
            let current_frames = manager.frame_count();
            assert!(
                current_frames <= 50,
                "Frame count {} exceeds limit at step {}",
                current_frames,
                i
            );
        }

        // Should have exactly the max number of frames
        assert_eq!(manager.frame_count(), 50);

        // Generate chart to ensure data is still valid
        let chart_path = manager.generate_chart(ChartType::FitnessEvolution).unwrap();
        assert!(chart_path.exists());

        println!(
            "Memory-efficient visualization completed with {} frames",
            manager.frame_count()
        );
    }
}
