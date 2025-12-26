//! Analysis algorithm implementations

use super::types::*;
use crate::replay::KernelMetrics;
use std::collections::HashMap;
use uuid::Uuid;

/// Analysis implementation helper
pub struct AnalysisImpl<'a> {
    config: &'a super::engine::AnalysisConfig,
}

impl<'a> AnalysisImpl<'a> {
    pub fn new(config: &'a super::engine::AnalysisConfig) -> Self {
        Self { config }
    }

    /// Analyze kernel metrics
    pub fn analyze_kernel_metrics(&self, metrics: &KernelMetrics) -> Vec<Finding> {
        let mut findings = Vec::new();

        // Check for performance issues
        if metrics.occupancy_percent < 50.0 {
            findings.push(Finding {
                finding_id: Uuid::new_v4(),
                category: FindingCategory::Performance,
                title: "Low GPU Occupancy".to_string(),
                description: format!(
                    "GPU occupancy is {}%, which is below optimal levels (>75%)",
                    metrics.occupancy_percent
                ),
                impact: Impact {
                    performance_impact: (75.0 - metrics.occupancy_percent) / 75.0,
                    memory_impact: 0.0,
                    correctness_impact: 0.0,
                    overall_score: (75.0 - metrics.occupancy_percent) / 75.0 * 0.8,
                },
                evidence: {
                    let mut evidence = HashMap::new();
                    evidence.insert(
                        "occupancy_percent".to_string(),
                        serde_json::Value::Number(
                            serde_json::Number::from_f64(metrics.occupancy_percent).unwrap(),
                        ),
                    );
                    evidence
                },
                location: None,
            });
        }

        // Check for memory bandwidth issues
        if metrics.memory_bandwidth_gb_s < 100.0 {
            findings.push(Finding {
                finding_id: Uuid::new_v4(),
                category: FindingCategory::Memory,
                title: "Low Memory Bandwidth".to_string(),
                description: format!(
                    "Memory bandwidth is only {:.1} GB/s, which is below typical GPU capabilities (>200 GB/s)",
                    metrics.memory_bandwidth_gb_s
                ),
                impact: Impact {
                    performance_impact: 0.4,
                    memory_impact: 0.6,
                    correctness_impact: 0.0,
                    overall_score: 0.5,
                },
                evidence: {
                    let mut evidence = HashMap::new();
                    evidence.insert(
                        "bandwidth_gb_s".to_string(),
                        serde_json::Value::Number(
                            serde_json::Number::from_f64(metrics.memory_bandwidth_gb_s).unwrap(),
                        ),
                    );
                    evidence
                },
                location: None,
            });
        }

        // Check for excessive execution time
        if metrics.average_execution_time_ms > 1000.0 {
            findings.push(Finding {
                finding_id: Uuid::new_v4(),
                category: FindingCategory::Performance,
                title: "High Kernel Execution Time".to_string(),
                description: format!(
                    "Average kernel execution time is {:.1}ms, which may indicate optimization opportunities",
                    metrics.average_execution_time_ms
                ),
                impact: Impact {
                    performance_impact: 0.6,
                    memory_impact: 0.0,
                    correctness_impact: 0.0,
                    overall_score: 0.6,
                },
                evidence: {
                    let mut evidence = HashMap::new();
                    evidence.insert(
                        "avg_execution_ms".to_string(),
                        serde_json::Value::Number(
                            serde_json::Number::from_f64(metrics.average_execution_time_ms).unwrap(),
                        ),
                    );
                    evidence.insert(
                        "launch_count".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(metrics.launch_count)),
                    );
                    evidence
                },
                location: None,
            });
        }

        findings
    }

    /// Generate recommendations from findings
    pub fn generate_recommendations(&self, findings: &[Finding]) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        for finding in findings {
            match finding.category {
                FindingCategory::Performance => {
                    if finding.title.contains("Low GPU Occupancy") {
                        recommendations.push(Recommendation {
                            recommendation_id: Uuid::new_v4(),
                            title: "Increase Block Size".to_string(),
                            description: "Consider increasing the block size to improve GPU occupancy. Try multiples of 32 up to 1024.".to_string(),
                            action_type: ActionType::ParameterTuning,
                            priority: Priority::High,
                            estimated_improvement: EstimatedImprovement {
                                performance_gain_percent: 25.0,
                                memory_savings_percent: 0.0,
                                reliability_improvement: 0.1,
                                confidence: 0.8,
                            },
                            implementation_effort: ImplementationEffort {
                                complexity: Complexity::Simple,
                                estimated_hours: 2.0,
                                risk_level: RiskLevel::Low,
                                dependencies: vec!["kernel_configuration".to_string()],
                            },
                        });
                    }

                    if finding.title.contains("High Kernel Execution Time") {
                        recommendations.push(Recommendation {
                            recommendation_id: Uuid::new_v4(),
                            title: "Optimize Kernel Algorithm".to_string(),
                            description: "Review kernel implementation for optimization opportunities such as memory coalescing, shared memory usage, or algorithm improvements.".to_string(),
                            action_type: ActionType::CodeChange,
                            priority: Priority::Medium,
                            estimated_improvement: EstimatedImprovement {
                                performance_gain_percent: 40.0,
                                memory_savings_percent: 10.0,
                                reliability_improvement: 0.0,
                                confidence: 0.6,
                            },
                            implementation_effort: ImplementationEffort {
                                complexity: Complexity::Complex,
                                estimated_hours: 16.0,
                                risk_level: RiskLevel::Medium,
                                dependencies: vec![
                                    "kernel_source_code".to_string(),
                                    "profiling_tools".to_string(),
                                ],
                            },
                        });
                    }
                }
                FindingCategory::Memory => {
                    if finding.title.contains("Low Memory Bandwidth") {
                        recommendations.push(Recommendation {
                            recommendation_id: Uuid::new_v4(),
                            title: "Improve Memory Access Patterns".to_string(),
                            description: "Optimize memory access patterns for better coalescing. Consider using shared memory or texture memory where appropriate.".to_string(),
                            action_type: ActionType::CodeChange,
                            priority: Priority::High,
                            estimated_improvement: EstimatedImprovement {
                                performance_gain_percent: 30.0,
                                memory_savings_percent: 15.0,
                                reliability_improvement: 0.0,
                                confidence: 0.7,
                            },
                            implementation_effort: ImplementationEffort {
                                complexity: Complexity::Moderate,
                                estimated_hours: 8.0,
                                risk_level: RiskLevel::Low,
                                dependencies: vec!["kernel_source_code".to_string()],
                            },
                        });
                    }
                }
                _ => {}
            }
        }

        recommendations
    }

    /// Determine overall severity from findings
    pub fn determine_severity(&self, findings: &[Finding]) -> Severity {
        if findings.is_empty() {
            return Severity::Info;
        }

        let max_impact = findings
            .iter()
            .map(|f| f.impact.overall_score)
            .fold(0.0, f64::max);

        if max_impact > 0.8 {
            Severity::Critical
        } else if max_impact > 0.6 {
            Severity::High
        } else if max_impact > 0.4 {
            Severity::Medium
        } else if max_impact > 0.2 {
            Severity::Low
        } else {
            Severity::Info
        }
    }

    /// Calculate confidence level for analysis
    pub fn calculate_confidence(&self, findings: &[Finding], data_points: u64) -> f64 {
        if data_points == 0 {
            return 0.0;
        }

        let base_confidence = match data_points {
            0..=10 => 0.3,
            11..=50 => 0.5,
            51..=100 => 0.7,
            101..=500 => 0.8,
            501..=1000 => 0.9,
            _ => 0.95,
        };

        // Adjust confidence based on findings consistency
        let avg_confidence = if !findings.is_empty() {
            findings.iter().map(|f| f.impact.overall_score).sum::<f64>() / findings.len() as f64
        } else {
            1.0
        };

        base_confidence * avg_confidence.min(1.0)
    }

    /// Detect anomalies in metric values
    pub fn detect_metric_anomaly(
        &self,
        metric_name: &str,
        value: f64,
        historical_values: &[(u64, f64)],
    ) -> Option<AnomalyResult> {
        if historical_values.len() < self.config.trend_min_points {
            return None;
        }

        // Calculate mean and standard deviation
        let values: Vec<f64> = historical_values.iter().map(|(_, v)| *v).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        // Check if value is anomalous
        let z_score = if std_dev > 0.0 {
            (value - mean).abs() / std_dev
        } else {
            0.0
        };

        if z_score > self.config.anomaly_threshold {
            let anomaly_type = if value > mean {
                AnomalyType::Spike
            } else {
                AnomalyType::Drop
            };

            Some(AnomalyResult {
                anomaly_id: Uuid::new_v4(),
                anomaly_type,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                metric_name: metric_name.to_string(),
                expected_value: mean,
                actual_value: value,
                deviation_score: z_score,
                confidence: z_score.min(5.0) / 5.0, // Normalize to 0-1
                context: HashMap::new(),
            })
        } else {
            None
        }
    }

    /// Analyze trend in time series data
    pub fn analyze_trend(&self, data_points: &[(u64, f64)]) -> Option<TrendAnalysis> {
        if data_points.len() < self.config.trend_min_points {
            return None;
        }

        // Simple linear regression
        let n = data_points.len() as f64;
        let sum_x: f64 = data_points.iter().map(|(x, _)| *x as f64).sum();
        let sum_y: f64 = data_points.iter().map(|(_, y)| *y).sum();
        let sum_xx: f64 = data_points.iter().map(|(x, _)| (*x as f64).powi(2)).sum();
        let sum_xy: f64 = data_points.iter().map(|(x, y)| *x as f64 * *y).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x.powi(2));
        let _intercept = (sum_y - slope * sum_x) / n;

        // Determine trend direction
        let trend_direction = if slope.abs() < 0.01 {
            TrendDirection::Stable
        } else if slope > 0.0 {
            TrendDirection::Increasing
        } else {
            TrendDirection::Decreasing
        };

        // Calculate R-squared for confidence
        let y_mean = sum_y / n;
        let ss_tot: f64 = data_points.iter().map(|(_, y)| (y - y_mean).powi(2)).sum();
        let ss_res: f64 = data_points
            .iter()
            .map(|(x, y)| {
                let y_pred = slope * (*x as f64) + _intercept;
                (y - y_pred).powi(2)
            })
            .sum();

        let r_squared = if ss_tot > 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        Some(TrendAnalysis {
            metric_name: "unnamed".to_string(),
            trend_direction,
            rate_of_change: slope,
            prediction_window: 3600, // 1 hour default
            confidence: r_squared,
            data_points: data_points.to_vec(),
        })
    }
}
