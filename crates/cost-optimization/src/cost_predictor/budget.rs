//! Budget forecasting and assessment

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use super::recommendations::CostRecommendation;
use super::risk::{RiskFactor, RiskLevel};
use super::types::CostMetricType;

/// Budget forecast
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetForecast {
    /// Forecast ID
    pub id: Uuid,
    /// Created at
    pub created_at: DateTime<Utc>,
    /// Time period
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    /// Predicted total cost
    pub predicted_cost: f64,
    /// Cost breakdown by category
    pub cost_breakdown: HashMap<CostMetricType, f64>,
    /// Risk assessment
    pub risk_assessment: RiskAssessment,
    /// Recommendations
    pub recommendations: Vec<CostRecommendation>,
}

/// Risk assessment for budget
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    /// Overall risk level
    pub risk_level: RiskLevel,
    /// Probability of budget overrun
    pub overrun_probability: f64,
    /// Expected variance
    pub expected_variance: f64,
    /// Risk factors
    pub risk_factors: Vec<RiskFactor>,
}
