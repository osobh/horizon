//! Machine learning models for anomaly detection
#[derive(Debug, Clone)]
pub enum MLAlgorithm {
    IsolationForest,
    LSTM,
    Autoencoder,
    OneClassSVM,
    LocalOutlierFactor,
}

pub struct ModelState {
    pub algorithm: MLAlgorithm,
    pub is_trained: bool,
    pub last_trained: Option<chrono::DateTime<chrono::Utc>>,
    pub training_samples: usize,
    pub model_version: String,
}

pub struct StatisticalBaseline {
    pub mean: f64,
    pub std_dev: f64,
    pub median: f64,
    pub q1: f64,
    pub q3: f64,
    pub min: f64,
    pub max: f64,
}

pub struct IsolationForestModel {
    pub n_estimators: usize,
    pub max_samples: usize,
    pub contamination: f64,
    pub max_features: f64,
}

pub struct LSTMModel {
    pub sequence_length: usize,
    pub hidden_units: usize,
    pub num_layers: usize,
}
