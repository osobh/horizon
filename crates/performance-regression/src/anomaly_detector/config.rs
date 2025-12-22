//! Configuration for anomaly detection
pub struct AnomalyConfig {
    pub sensitivity_threshold: f64,
    pub detection_window: std::time::Duration,
    pub baseline_window: std::time::Duration,
    pub ml_algorithm: super::models::MLAlgorithm,
    pub statistical_method: String,
    pub min_samples: usize,
    pub confidence_interval: f64,
    pub feature_extraction: Vec<String>,
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        Self {
            sensitivity_threshold: 2.5,
            detection_window: std::time::Duration::from_secs(3600),
            baseline_window: std::time::Duration::from_secs(86400 * 7),
            ml_algorithm: super::models::MLAlgorithm::IsolationForest,
            statistical_method: "z-score".to_string(),
            min_samples: 100,
            confidence_interval: 0.95,
            feature_extraction: vec![],
        }
    }
}
