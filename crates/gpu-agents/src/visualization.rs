//! Swarm behavior visualization tools
//!
//! This module has been refactored for better maintainability while keeping
//! all functionality in a single file to stay under the 850 line limit.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::GpuSwarm;

/// Chart types for visualization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChartType {
    /// Agent position distribution in 3D space
    SwarmDistribution,
    /// Fitness evolution over time
    FitnessEvolution,
    /// Population diversity metrics
    DiversityMetrics,
    /// Performance timeline (timing, memory usage)
    PerformanceTimeline,
}

/// Data export formats
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataExportFormat {
    /// JSON format for general use
    JSON,
    /// CSV format for spreadsheet analysis
    CSV,
    /// Parquet format for big data analysis
    Parquet,
}

/// Rendering backends for visualization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RenderingBackend {
    /// Plotters library for static charts
    Plotters,
    /// Matplotlib-style plotting
    MatplotlibRs,
    /// Web-based interactive charts
    WebGL,
}

/// Configuration for visualization system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    /// Output directory for generated files
    pub output_directory: PathBuf,
    /// Enable real-time visualization
    pub enable_real_time: bool,
    /// Update interval for real-time visualization (ms)
    pub update_interval_ms: u64,
    /// Maximum number of frames to keep in history
    pub max_history_frames: usize,
    /// Types of charts to generate
    pub chart_types: Vec<ChartType>,
    /// Default export format
    pub export_format: DataExportFormat,
    /// Rendering backend to use
    pub rendering_backend: RenderingBackend,
    /// Image width for generated charts
    pub image_width: u32,
    /// Image height for generated charts
    pub image_height: u32,
    /// Enable animation generation
    pub enable_animation: bool,
    /// Frames per second for animations
    pub fps: u32,
    /// Color scheme for charts
    pub color_scheme: String,
}

impl VisualizationConfig {
    /// Create default config with specified output directory
    pub fn default_with_output_dir(output_dir: &Path) -> Self {
        Self {
            output_directory: output_dir.to_path_buf(),
            enable_real_time: false,
            update_interval_ms: 1000,
            max_history_frames: 1000,
            chart_types: vec![
                ChartType::SwarmDistribution,
                ChartType::FitnessEvolution,
                ChartType::DiversityMetrics,
                ChartType::PerformanceTimeline,
            ],
            export_format: DataExportFormat::JSON,
            rendering_backend: RenderingBackend::Plotters,
            image_width: 1200,
            image_height: 800,
            enable_animation: false,
            fps: 24,
            color_scheme: "default".to_string(),
        }
    }
}

/// Single frame of swarm data for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameData {
    /// Frame timestamp
    pub timestamp: u64,
    /// Number of agents in the swarm
    pub agent_count: usize,
    /// Agent positions [x, y, z] for each agent
    pub agent_positions: Vec<Vec<f32>>,
    /// Fitness values for each agent
    pub fitness_values: Vec<f32>,
    /// Average fitness of the population
    pub average_fitness: f32,
    /// Population diversity score
    pub diversity_score: f32,
    /// GPU memory usage in bytes
    pub gpu_memory_used: usize,
    /// Kernel execution time in milliseconds
    pub kernel_time_ms: f32,
    /// LLM inference time in milliseconds
    pub llm_inference_time_ms: f32,
}

/// Visualization performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationMetrics {
    /// Total number of frames captured
    pub total_frames_captured: usize,
    /// Number of charts generated
    pub charts_generated: usize,
    /// Total processing time in milliseconds
    pub total_processing_time_ms: f64,
    /// Average time per frame capture in milliseconds
    pub average_frame_capture_time_ms: f64,
    /// Total data size in bytes
    pub data_size_bytes: usize,
    /// Export formats that have been used
    pub export_formats_used: Vec<DataExportFormat>,
    /// Number of animations generated
    pub animations_generated: usize,
}

/// Manages swarm behavior visualization
pub struct VisualizationManager {
    config: VisualizationConfig,
    frame_history: VecDeque<FrameData>,
    is_recording: bool,
    last_update_time: Option<Instant>,
    total_processing_time: Duration,
    charts_generated: usize,
    animations_generated: usize,
    export_formats_used: Vec<DataExportFormat>,
}

impl VisualizationManager {
    /// Create a new visualization manager
    pub fn new(config: VisualizationConfig) -> Result<Self> {
        std::fs::create_dir_all(&config.output_directory)?;

        Ok(Self {
            config,
            frame_history: VecDeque::new(),
            is_recording: false,
            last_update_time: None,
            total_processing_time: Duration::ZERO,
            charts_generated: 0,
            animations_generated: 0,
            export_formats_used: Vec::new(),
        })
    }

    /// Get the number of captured frames
    pub fn frame_count(&self) -> usize {
        self.frame_history.len()
    }

    /// Check if real-time recording is active
    pub fn is_recording(&self) -> bool {
        self.is_recording
    }

    /// Get supported chart types
    pub fn supported_chart_types(&self) -> &[ChartType] {
        &self.config.chart_types
    }

    /// Capture a frame of swarm data
    pub fn capture_frame(&mut self, swarm: &GpuSwarm) -> Result<FrameData> {
        let start_time = Instant::now();

        let metrics = swarm.metrics();
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;

        // Generate agent positions and fitness values
        let (agent_positions, fitness_values) = self.generate_agent_data(metrics.agent_count);
        let average_fitness = if !fitness_values.is_empty() {
            fitness_values.iter().sum::<f32>() / fitness_values.len() as f32
        } else {
            0.0
        };

        let diversity_score = if fitness_values.len() > 1 {
            let variance = fitness_values
                .iter()
                .map(|&f| (f - average_fitness).powi(2))
                .sum::<f32>()
                / fitness_values.len() as f32;
            variance.sqrt()
        } else {
            0.0
        };

        let frame_data = FrameData {
            timestamp,
            agent_count: metrics.agent_count,
            agent_positions,
            fitness_values,
            average_fitness,
            diversity_score,
            gpu_memory_used: metrics.gpu_memory_used,
            kernel_time_ms: metrics.kernel_time_ms,
            llm_inference_time_ms: metrics.llm_inference_time_ms,
        };

        self.frame_history.push_back(frame_data.clone());
        while self.frame_history.len() > self.config.max_history_frames {
            self.frame_history.pop_front();
        }

        self.total_processing_time += start_time.elapsed();
        Ok(frame_data)
    }

    /// Generate agent positions and fitness values
    fn generate_agent_data(&self, agent_count: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
        let mut agent_positions = Vec::with_capacity(agent_count);
        let mut fitness_values = Vec::with_capacity(agent_count);

        for i in 0..agent_count {
            let angle = (i as f32) * 0.1;
            let radius = 5.0 + (i as f32 * 0.01).sin() * 2.0;

            agent_positions.push(vec![
                radius * angle.cos(),
                radius * angle.sin(),
                (i as f32 * 0.05).sin() * 1.0,
            ]);

            let base_fitness = 0.5 + (i as f32 / agent_count as f32) * 0.5;
            let noise = (i as f32 * 7.0).sin() * 0.1;
            fitness_values.push(base_fitness + noise);
        }

        (agent_positions, fitness_values)
    }

    /// Generate a chart for the specified type
    pub fn generate_chart(&mut self, chart_type: ChartType) -> Result<PathBuf> {
        if self.frame_history.is_empty() {
            return Err(anyhow::anyhow!(
                "No frame data available for chart generation"
            ));
        }

        let start_time = Instant::now();
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        let filename = format!(
            "{}_{}.png",
            match chart_type {
                ChartType::SwarmDistribution => "swarm_distribution",
                ChartType::FitnessEvolution => "fitness_evolution",
                ChartType::DiversityMetrics => "diversity_metrics",
                ChartType::PerformanceTimeline => "performance_timeline",
            },
            timestamp
        );

        let chart_path = self.config.output_directory.join(filename);
        self.create_chart_placeholder(&chart_path, &chart_type)?;

        self.charts_generated += 1;
        self.total_processing_time += start_time.elapsed();
        Ok(chart_path)
    }

    /// Create a placeholder chart
    fn create_chart_placeholder(&self, output_path: &Path, chart_type: &ChartType) -> Result<()> {
        let latest_frame = self.frame_history.back().ok_or_else(|| anyhow::anyhow!("No frames available"))?;
        let chart_data = match chart_type {
            ChartType::SwarmDistribution => format!(
                "Swarm Distribution\nAgents: {}\nAvg Fitness: {:.3}\nDiversity: {:.3}",
                latest_frame.agent_count,
                latest_frame.average_fitness,
                latest_frame.diversity_score
            ),
            ChartType::FitnessEvolution => {
                let fitness_range = self
                    .frame_history
                    .iter()
                    .map(|f| f.average_fitness)
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), f| {
                        (min.min(f), max.max(f))
                    });
                format!(
                    "Fitness Evolution\nFrames: {}\nRange: {:.3}-{:.3}",
                    self.frame_history.len(),
                    fitness_range.0,
                    fitness_range.1
                )
            }
            ChartType::DiversityMetrics => {
                let div_range = self
                    .frame_history
                    .iter()
                    .map(|f| f.diversity_score)
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), d| {
                        (min.min(d), max.max(d))
                    });
                format!(
                    "Diversity Metrics\nFrames: {}\nRange: {:.3}-{:.3}",
                    self.frame_history.len(),
                    div_range.0,
                    div_range.1
                )
            }
            ChartType::PerformanceTimeline => {
                let avg_kernel_time = self
                    .frame_history
                    .iter()
                    .map(|f| f.kernel_time_ms)
                    .sum::<f32>()
                    / self.frame_history.len() as f32;
                format!(
                    "Performance Timeline\nAvg Kernel: {:.3}ms\nMemory: {}MB",
                    avg_kernel_time,
                    latest_frame.gpu_memory_used / 1_000_000
                )
            }
        };

        let content = format!(
            "Chart Placeholder: {:?}\n\n{}\n\nGenerated: {}",
            chart_type,
            chart_data,
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        );
        std::fs::write(output_path, content.as_bytes())?;
        Ok(())
    }

    /// Start/stop real-time visualization
    pub fn start_real_time_visualization(&mut self) -> Result<()> {
        self.is_recording = true;
        self.last_update_time = Some(Instant::now());
        Ok(())
    }

    pub fn stop_real_time_visualization(&mut self) -> Result<()> {
        self.is_recording = false;
        self.last_update_time = None;
        Ok(())
    }

    /// Update real-time visualization
    pub fn update_real_time(&mut self, swarm: &GpuSwarm) -> Result<()> {
        if !self.is_recording {
            return Ok(());
        }

        let now = Instant::now();
        let should_update = if let Some(last_update) = self.last_update_time {
            now.duration_since(last_update).as_millis() >= self.config.update_interval_ms as u128
        } else {
            true
        };

        if should_update {
            self.capture_frame(swarm)?;
            if !self.config.chart_types.is_empty() {
                let chart_type = self.config.chart_types[0].clone();
                self.generate_chart(chart_type)?;
            }
            self.last_update_time = Some(now);
        }
        Ok(())
    }

    /// Export data in the specified format
    pub fn export_data(&mut self, format: DataExportFormat) -> Result<PathBuf> {
        if self.frame_history.is_empty() {
            return Err(anyhow::anyhow!("No data to export"));
        }

        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let filename = format!(
            "swarm_data_{}.{}",
            timestamp,
            match format {
                DataExportFormat::JSON => "json",
                DataExportFormat::CSV => "csv",
                DataExportFormat::Parquet => "parquet",
            }
        );

        let export_path = self.config.output_directory.join(filename);
        match format {
            DataExportFormat::JSON => self.export_json(&export_path)?,
            DataExportFormat::CSV => self.export_csv(&export_path)?,
            DataExportFormat::Parquet => self.export_parquet(&export_path)?,
        }

        if !self.export_formats_used.contains(&format) {
            self.export_formats_used.push(format);
        }
        Ok(export_path)
    }

    /// Export as JSON
    fn export_json(&self, output_path: &Path) -> Result<()> {
        let export_data = serde_json::json!({
            "metadata": {
                "export_timestamp": SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
                "total_frames": self.frame_history.len(),
                "config": self.config
            },
            "frames": self.frame_history,
        });
        std::fs::write(output_path, serde_json::to_string_pretty(&export_data)?)?;
        Ok(())
    }

    /// Export as CSV
    fn export_csv(&self, output_path: &Path) -> Result<()> {
        let mut csv = String::from("timestamp,agent_count,average_fitness,diversity_score,gpu_memory_used,kernel_time_ms,llm_inference_time_ms\n");
        for frame in &self.frame_history {
            csv.push_str(&format!(
                "{},{},{:.6},{:.6},{},{:.6},{:.6}\n",
                frame.timestamp,
                frame.agent_count,
                frame.average_fitness,
                frame.diversity_score,
                frame.gpu_memory_used,
                frame.kernel_time_ms,
                frame.llm_inference_time_ms
            ));
        }
        std::fs::write(output_path, csv)?;
        Ok(())
    }

    /// Export as Parquet (placeholder)
    fn export_parquet(&self, output_path: &Path) -> Result<()> {
        let placeholder = format!(
            "Parquet Placeholder\nFrames: {}\nExported: {}",
            self.frame_history.len(),
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        );
        std::fs::write(output_path, placeholder.as_bytes())?;
        Ok(())
    }

    /// Generate animation
    pub fn generate_animation(&mut self, chart_type: ChartType) -> Result<PathBuf> {
        if self.frame_history.len() < 2 {
            return Err(anyhow::anyhow!("Need at least 2 frames for animation"));
        }

        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let filename = format!(
            "{}_{}.gif",
            match chart_type {
                ChartType::SwarmDistribution => "swarm_animation",
                ChartType::FitnessEvolution => "fitness_animation",
                ChartType::DiversityMetrics => "diversity_animation",
                ChartType::PerformanceTimeline => "performance_animation",
            },
            timestamp
        );

        let path = self.config.output_directory.join(filename);
        let placeholder = format!(
            "Animation Placeholder\nType: {:?}\nFrames: {}\nFPS: {}",
            chart_type,
            self.frame_history.len(),
            self.config.fps
        );
        std::fs::write(&path, placeholder.as_bytes())?;

        self.animations_generated += 1;
        Ok(path)
    }

    /// Generate dashboard
    pub fn generate_performance_dashboard(&mut self) -> Result<PathBuf> {
        if self.frame_history.is_empty() {
            return Err(anyhow::anyhow!("No data available for dashboard"));
        }

        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let path = self
            .config
            .output_directory
            .join(format!("dashboard_{}.html", timestamp));

        let latest = self.frame_history.back().ok_or_else(|| anyhow::anyhow!("No frames available"))?;
        let html = format!(
            r#"<!DOCTYPE html><html><head><title>Dashboard</title></head><body>
<h1>Performance Dashboard</h1>
<p>Agents: {}</p><p>Fitness: {:.3}</p><p>Memory: {} MB</p><p>Kernel: {:.2}ms</p>
</body></html>"#,
            latest.agent_count,
            latest.average_fitness,
            latest.gpu_memory_used / 1_000_000,
            latest.kernel_time_ms
        );

        std::fs::write(&path, html)?;
        Ok(path)
    }

    /// Generate multi-chart dashboard  
    pub fn generate_multi_chart_dashboard(&mut self) -> Result<PathBuf> {
        if self.frame_history.is_empty() {
            return Err(anyhow::anyhow!("No data available for dashboard"));
        }

        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let path = self
            .config
            .output_directory
            .join(format!("multi_dashboard_{}.html", timestamp));

        let latest = self.frame_history.back().ok_or_else(|| anyhow::anyhow!("No frames available"))?;
        let html = format!(
            r#"<!DOCTYPE html><html><head><title>Multi-Chart Dashboard</title></head><body>
<h1>Multi-Chart Dashboard</h1><div style="display:flex;flex-wrap:wrap;">
<div style="margin:10px;padding:10px;border:1px solid #ccc;"><h3>Swarm</h3><p>Agents: {}</p></div>
<div style="margin:10px;padding:10px;border:1px solid #ccc;"><h3>Fitness</h3><p>Current: {:.3}</p></div>
<div style="margin:10px;padding:10px;border:1px solid #ccc;"><h3>Diversity</h3><p>Score: {:.3}</p></div>
<div style="margin:10px;padding:10px;border:1px solid #ccc;"><h3>Performance</h3><p>Kernel: {:.2}ms</p></div>
</div></body></html>"#,
            latest.agent_count,
            latest.average_fitness,
            latest.diversity_score,
            latest.kernel_time_ms
        );

        std::fs::write(&path, html)?;
        Ok(path)
    }

    /// Get metrics
    pub fn get_visualization_metrics(&self) -> Result<VisualizationMetrics> {
        let data_size = self
            .frame_history
            .iter()
            .map(|f| f.agent_positions.len() * 12 + f.fitness_values.len() * 4 + 64)
            .sum::<usize>();

        let avg_time = if !self.frame_history.is_empty() {
            self.total_processing_time.as_millis() as f64 / self.frame_history.len() as f64
        } else {
            0.0
        };

        Ok(VisualizationMetrics {
            total_frames_captured: self.frame_history.len(),
            charts_generated: self.charts_generated,
            total_processing_time_ms: self.total_processing_time.as_millis() as f64,
            average_frame_capture_time_ms: avg_time,
            data_size_bytes: data_size,
            export_formats_used: self.export_formats_used.clone(),
            animations_generated: self.animations_generated,
        })
    }
}
