//! Result explanation and interpretation for business goals

use crate::error::{BusinessError, BusinessResult};
use crate::goal::{BusinessGoal, Criterion};
use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
        ChatCompletionRequestUserMessage, CreateChatCompletionRequest,
    },
    Client,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Result explanation system for business goals
pub struct ResultExplainer {
    llm_client: Client<OpenAIConfig>,
    model: String,
    system_prompt: String,
    templates: ExplanationTemplates,
}

/// Explained results for a business goal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplainedResult {
    /// Goal ID
    pub goal_id: String,
    /// Executive summary
    pub executive_summary: String,
    /// Detailed findings
    pub detailed_findings: Vec<Finding>,
    /// Success criteria analysis
    pub criteria_analysis: Vec<CriterionAnalysis>,
    /// Key insights
    pub key_insights: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<Recommendation>,
    /// Data visualizations suggested
    pub visualizations: Vec<VisualizationSuggestion>,
    /// Business impact assessment
    pub business_impact: BusinessImpact,
    /// Technical details
    pub technical_details: TechnicalDetails,
    /// Generated timestamp
    pub generated_at: DateTime<Utc>,
    /// Explanation confidence score (0-1)
    pub confidence_score: f64,
}

/// Individual finding from goal execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    /// Finding ID
    pub finding_id: String,
    /// Finding title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Finding category
    pub category: FindingCategory,
    /// Importance level
    pub importance: ImportanceLevel,
    /// Supporting evidence
    pub evidence: Vec<Evidence>,
    /// Statistical significance (if applicable)
    pub statistical_significance: Option<f64>,
    /// Confidence in this finding
    pub confidence: f64,
}

/// Analysis of success criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriterionAnalysis {
    /// The criterion being analyzed
    pub criterion: Criterion,
    /// Whether the criterion was met
    pub met: bool,
    /// Actual achieved value
    pub achieved_value: Option<f64>,
    /// Target value
    pub target_value: f64,
    /// Explanation of the result
    pub explanation: String,
    /// Contributing factors
    pub contributing_factors: Vec<String>,
}

/// Business recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Recommendation ID
    pub recommendation_id: String,
    /// Short title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Priority level
    pub priority: RecommendationPriority,
    /// Expected impact
    pub expected_impact: String,
    /// Implementation effort
    pub implementation_effort: EffortLevel,
    /// Timeline for implementation
    pub timeline: String,
    /// Required resources
    pub required_resources: Vec<String>,
    /// Success metrics
    pub success_metrics: Vec<String>,
}

/// Visualization suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationSuggestion {
    /// Visualization type
    pub viz_type: VisualizationType,
    /// Title for the visualization
    pub title: String,
    /// Description of what to show
    pub description: String,
    /// Data fields to include
    pub data_fields: Vec<String>,
    /// Suggested insights to highlight
    pub insights_to_highlight: Vec<String>,
}

/// Business impact assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessImpact {
    /// Overall impact score (0-10)
    pub impact_score: f64,
    /// Revenue impact
    pub revenue_impact: Option<RevenueImpact>,
    /// Cost impact
    pub cost_impact: Option<CostImpact>,
    /// Risk assessment
    pub risk_assessment: RiskAssessment,
    /// Strategic alignment
    pub strategic_alignment: String,
    /// Stakeholder impact
    pub stakeholder_impact: Vec<StakeholderImpact>,
}

/// Technical execution details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalDetails {
    /// Execution duration
    pub execution_duration: std::time::Duration,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,
    /// Error analysis
    pub error_analysis: Option<ErrorAnalysis>,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
}

/// Supporting evidence for findings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    /// Evidence type
    pub evidence_type: EvidenceType,
    /// Description
    pub description: String,
    /// Data source
    pub source: String,
    /// Reliability score
    pub reliability: f64,
    /// Timestamp when evidence was collected
    pub collected_at: DateTime<Utc>,
}

/// Revenue impact details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevenueImpact {
    /// Estimated revenue change
    pub estimated_change: f64,
    /// Currency
    pub currency: String,
    /// Time period
    pub time_period: String,
    /// Confidence in estimate
    pub confidence: f64,
}

/// Cost impact details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostImpact {
    /// Estimated cost change
    pub estimated_change: f64,
    /// Currency
    pub currency: String,
    /// Cost category
    pub category: String,
    /// Time period
    pub time_period: String,
}

/// Risk assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    /// Overall risk level
    pub risk_level: RiskLevel,
    /// Identified risks
    pub risks: Vec<IdentifiedRisk>,
    /// Mitigation strategies
    pub mitigation_strategies: Vec<String>,
}

/// Stakeholder impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StakeholderImpact {
    /// Stakeholder group
    pub stakeholder: String,
    /// Impact description
    pub impact: String,
    /// Impact severity
    pub severity: ImpactSeverity,
}

/// Resource utilization summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// Memory utilization percentage
    pub memory_utilization: f64,
    /// GPU utilization percentage
    pub gpu_utilization: f64,
    /// Network utilization
    pub network_utilization: f64,
    /// Cost efficiency score
    pub cost_efficiency: f64,
}

/// Error analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorAnalysis {
    /// Error rate
    pub error_rate: f64,
    /// Common error types
    pub common_errors: Vec<String>,
    /// Error impact assessment
    pub impact_assessment: String,
    /// Remediation suggestions
    pub remediation_suggestions: Vec<String>,
}

/// Quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Data quality score
    pub data_quality_score: f64,
    /// Result consistency score
    pub consistency_score: f64,
    /// Completeness score
    pub completeness_score: f64,
    /// Accuracy score
    pub accuracy_score: f64,
}

/// Identified risk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentifiedRisk {
    /// Risk description
    pub description: String,
    /// Probability (0-1)
    pub probability: f64,
    /// Impact severity
    pub impact: RiskImpact,
    /// Risk category
    pub category: String,
}

/// Explanation templates for different goal types
#[derive(Debug, Clone)]
pub struct ExplanationTemplates {
    pub data_analysis: String,
    pub machine_learning: String,
    pub business_intelligence: String,
    pub optimization: String,
    pub default: String,
}

// Enums for categorization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FindingCategory {
    Insight,
    Anomaly,
    Trend,
    Correlation,
    Prediction,
    Performance,
    Quality,
    Risk,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum ImportanceLevel {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum RecommendationPriority {
    Low = 1,
    Medium = 2,
    High = 3,
    Urgent = 4,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EffortLevel {
    Minimal,
    Low,
    Medium,
    High,
    Extensive,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VisualizationType {
    BarChart,
    LineChart,
    ScatterPlot,
    Histogram,
    HeatMap,
    Dashboard,
    TreeMap,
    Network,
    Correlation,
    TimeSeries,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EvidenceType {
    Statistical,
    Observational,
    Experimental,
    Historical,
    Comparative,
    Qualitative,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum RiskLevel {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RiskImpact {
    Negligible,
    Minor,
    Moderate,
    Major,
    Severe,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ImpactSeverity {
    Positive,
    Neutral,
    Negative,
    Mixed,
}

impl ResultExplainer {
    /// Create a new result explainer
    pub fn new(api_key: Option<String>) -> BusinessResult<Self> {
        let config = match api_key {
            Some(key) => OpenAIConfig::new().with_api_key(key),
            None => OpenAIConfig::new(),
        };

        let llm_client = Client::with_config(config);

        let system_prompt = r#"
You are an expert business analyst and data scientist specializing in explaining complex analytical results 
in clear, actionable terms for business stakeholders. Your task is to:

1. Analyze goal execution results and provide executive-level summaries
2. Identify key insights, patterns, and actionable recommendations
3. Assess business impact and strategic implications
4. Suggest appropriate visualizations and next steps
5. Translate technical findings into business language

Focus on:
- Clear, jargon-free explanations
- Quantified business impact where possible
- Actionable recommendations with priorities
- Risk assessment and mitigation strategies
- Data-driven insights with confidence levels

Respond with structured JSON containing all analysis components.
"#.trim().to_string();

        let templates = ExplanationTemplates {
            data_analysis: "Analyze data processing results focusing on patterns, trends, and business insights.".to_string(),
            machine_learning: "Explain ML model performance, predictions, and recommendations for model improvement.".to_string(),
            business_intelligence: "Provide BI dashboard insights with KPI analysis and strategic recommendations.".to_string(),
            optimization: "Explain optimization results, efficiency gains, and implementation recommendations.".to_string(),
            default: "Provide comprehensive analysis of goal results with business-focused insights.".to_string(),
        };

        Ok(Self {
            llm_client,
            model: "gpt-4".to_string(),
            system_prompt,
            templates,
        })
    }

    /// Explain goal execution results
    pub async fn explain_results(
        &self,
        goal: &BusinessGoal,
        execution_data: &HashMap<String, serde_json::Value>,
    ) -> BusinessResult<ExplainedResult> {
        debug!("Explaining results for goal: {}", goal.goal_id);

        #[cfg(feature = "mock")]
        {
            return Ok(self.create_mock_explanation(goal, execution_data));
        }

        let template = self.get_template_for_goal(goal);
        let analysis = self
            .generate_llm_explanation(goal, execution_data, &template)
            .await?;

        let mut explained_result = self.parse_llm_response(&analysis)?;
        explained_result.goal_id = goal.goal_id.clone();
        explained_result.generated_at = Utc::now();

        // Add technical details from execution data
        explained_result.technical_details = self.extract_technical_details(execution_data);

        // Analyze success criteria
        explained_result.criteria_analysis = self.analyze_success_criteria(goal, execution_data);

        info!("Generated explanation for goal: {}", goal.goal_id);
        Ok(explained_result)
    }

    /// Generate explanation using LLM
    async fn generate_llm_explanation(
        &self,
        goal: &BusinessGoal,
        execution_data: &HashMap<String, serde_json::Value>,
        template: &str,
    ) -> BusinessResult<String> {
        let goal_json = serde_json::to_string_pretty(goal)?;
        let execution_json = serde_json::to_string_pretty(execution_data)?;

        let user_message = format!(
            "{}\n\nGoal Details:\n{}\n\nExecution Results:\n{}",
            template, goal_json, execution_json
        );

        let messages = vec![
            ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                content: async_openai::types::ChatCompletionRequestSystemMessageContent::Text(
                    self.system_prompt.clone(),
                ),
                name: None,
            }),
            ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                    user_message,
                ),
                name: None,
            }),
        ];

        let request = CreateChatCompletionRequest {
            model: self.model.clone(),
            messages,
            max_completion_tokens: Some(4096),
            temperature: Some(0.3),
            ..Default::default()
        };

        let response = self.llm_client.chat().create(request).await?;

        let content = response
            .choices
            .first()
            .and_then(|choice| choice.message.content.as_ref())
            .ok_or_else(|| BusinessError::ResultExplanationError {
                goal_id: goal.goal_id.clone(),
                reason: "No response content from LLM".to_string(),
            })?;

        Ok(content.clone())
    }

    /// Parse LLM response into structured explanation
    fn parse_llm_response(&self, response: &str) -> BusinessResult<ExplainedResult> {
        // Try to parse as JSON first
        if let Ok(explained) = serde_json::from_str::<ExplainedResult>(response) {
            return Ok(explained);
        }

        // Fallback: create explanation from text response
        warn!("Failed to parse LLM response as JSON, creating fallback explanation");

        Ok(ExplainedResult {
            goal_id: String::new(), // Will be set by caller
            executive_summary: response.lines().take(3).collect::<Vec<_>>().join(" "),
            detailed_findings: vec![Finding {
                finding_id: "fallback-1".to_string(),
                title: "Analysis Results".to_string(),
                description: response.to_string(),
                category: FindingCategory::Insight,
                importance: ImportanceLevel::Medium,
                evidence: Vec::new(),
                statistical_significance: None,
                confidence: 0.7,
            }],
            criteria_analysis: Vec::new(),
            key_insights: Vec::new(),
            recommendations: Vec::new(),
            visualizations: Vec::new(),
            business_impact: BusinessImpact::default(),
            technical_details: TechnicalDetails::default(),
            generated_at: Utc::now(),
            confidence_score: 0.7,
        })
    }

    /// Get appropriate template for goal category
    fn get_template_for_goal(&self, goal: &BusinessGoal) -> String {
        match goal.category {
            crate::goal::GoalCategory::DataAnalysis => self.templates.data_analysis.clone(),
            crate::goal::GoalCategory::MachineLearning => self.templates.machine_learning.clone(),
            crate::goal::GoalCategory::BusinessIntelligence => {
                self.templates.business_intelligence.clone()
            }
            crate::goal::GoalCategory::Optimization => self.templates.optimization.clone(),
            _ => self.templates.default.clone(),
        }
    }

    /// Extract technical details from execution data
    fn extract_technical_details(
        &self,
        execution_data: &HashMap<String, serde_json::Value>,
    ) -> TechnicalDetails {
        let duration = execution_data
            .get("execution_duration")
            .and_then(|v| v.as_u64())
            .map(std::time::Duration::from_secs)
            .unwrap_or(std::time::Duration::from_secs(0));

        let mut performance_metrics = HashMap::new();
        if let Some(metrics) = execution_data
            .get("performance_metrics")
            .and_then(|v| v.as_object())
        {
            for (key, value) in metrics {
                if let Some(num) = value.as_f64() {
                    performance_metrics.insert(key.clone(), num);
                }
            }
        }

        TechnicalDetails {
            execution_duration: duration,
            resource_utilization: ResourceUtilization::default(),
            performance_metrics,
            error_analysis: None,
            quality_metrics: QualityMetrics::default(),
        }
    }

    /// Analyze success criteria against results
    fn analyze_success_criteria(
        &self,
        goal: &BusinessGoal,
        execution_data: &HashMap<String, serde_json::Value>,
    ) -> Vec<CriterionAnalysis> {
        goal.success_criteria
            .iter()
            .map(|criterion| {
                let (met, achieved_value, target_value, explanation) = match criterion {
                    Criterion::Accuracy { min_accuracy } => {
                        let achieved = execution_data
                            .get("accuracy")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0);
                        (
                            achieved >= *min_accuracy,
                            Some(achieved),
                            *min_accuracy,
                            format!(
                                "Achieved accuracy: {:.3}, Target: {:.3}",
                                achieved, min_accuracy
                            ),
                        )
                    }
                    Criterion::Completion { percentage } => {
                        let achieved = execution_data
                            .get("completion_percentage")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0);
                        (
                            achieved >= *percentage as f64,
                            Some(achieved),
                            *percentage as f64,
                            format!(
                                "Achieved completion: {:.1}%, Target: {:.1}%",
                                achieved, percentage
                            ),
                        )
                    }
                    _ => (
                        true,
                        None,
                        0.0,
                        "Criterion analysis not implemented".to_string(),
                    ),
                };

                CriterionAnalysis {
                    criterion: criterion.clone(),
                    met,
                    achieved_value,
                    target_value,
                    explanation,
                    contributing_factors: Vec::new(),
                }
            })
            .collect()
    }

    #[cfg(feature = "mock")]
    fn create_mock_explanation(
        &self,
        goal: &BusinessGoal,
        _execution_data: &HashMap<String, serde_json::Value>,
    ) -> ExplainedResult {
        let findings = vec![
            Finding {
                finding_id: "finding-1".to_string(),
                title: "Significant Pattern Identified".to_string(),
                description: "Analysis revealed a strong correlation between variables A and B"
                    .to_string(),
                category: FindingCategory::Correlation,
                importance: ImportanceLevel::High,
                evidence: vec![Evidence {
                    evidence_type: EvidenceType::Statistical,
                    description: "Correlation coefficient: 0.85".to_string(),
                    source: "Statistical analysis".to_string(),
                    reliability: 0.95,
                    collected_at: Utc::now(),
                }],
                statistical_significance: Some(0.001),
                confidence: 0.92,
            },
            Finding {
                finding_id: "finding-2".to_string(),
                title: "Performance Optimization Opportunity".to_string(),
                description: "Algorithm efficiency can be improved by 23%".to_string(),
                category: FindingCategory::Performance,
                importance: ImportanceLevel::Medium,
                evidence: vec![Evidence {
                    evidence_type: EvidenceType::Experimental,
                    description: "Benchmark testing results".to_string(),
                    source: "Performance tests".to_string(),
                    reliability: 0.88,
                    collected_at: Utc::now(),
                }],
                statistical_significance: None,
                confidence: 0.78,
            },
        ];

        let recommendations = vec![Recommendation {
            recommendation_id: "rec-1".to_string(),
            title: "Implement Advanced Analytics".to_string(),
            description: "Deploy machine learning models to leverage identified patterns"
                .to_string(),
            priority: RecommendationPriority::High,
            expected_impact: "15-20% improvement in prediction accuracy".to_string(),
            implementation_effort: EffortLevel::Medium,
            timeline: "3-4 months".to_string(),
            required_resources: vec![
                "Data Science Team".to_string(),
                "ML Infrastructure".to_string(),
            ],
            success_metrics: vec![
                "Accuracy improvement".to_string(),
                "Processing time reduction".to_string(),
            ],
        }];

        let visualizations = vec![
            VisualizationSuggestion {
                viz_type: VisualizationType::ScatterPlot,
                title: "Variable Correlation Analysis".to_string(),
                description: "Scatter plot showing correlation between variables A and B"
                    .to_string(),
                data_fields: vec!["variable_a".to_string(), "variable_b".to_string()],
                insights_to_highlight: vec!["Strong positive correlation".to_string()],
            },
            VisualizationSuggestion {
                viz_type: VisualizationType::Dashboard,
                title: "Performance Metrics Dashboard".to_string(),
                description: "Real-time dashboard showing key performance indicators".to_string(),
                data_fields: vec![
                    "accuracy".to_string(),
                    "processing_time".to_string(),
                    "throughput".to_string(),
                ],
                insights_to_highlight: vec![
                    "Performance trends".to_string(),
                    "Optimization opportunities".to_string(),
                ],
            },
        ];

        ExplainedResult {
            goal_id: goal.goal_id.clone(),
            executive_summary: "Analysis completed successfully with significant insights identified. Strong correlations discovered with actionable optimization opportunities.".to_string(),
            detailed_findings: findings,
            criteria_analysis: Vec::new(),
            key_insights: vec![
                "Variable correlation strength of 0.85 indicates predictive potential".to_string(),
                "Algorithm optimization can yield 23% performance improvement".to_string(),
                "Data quality scores exceed 90% threshold for reliability".to_string(),
            ],
            recommendations,
            visualizations,
            business_impact: BusinessImpact {
                impact_score: 7.5,
                revenue_impact: Some(RevenueImpact {
                    estimated_change: 150000.0,
                    currency: "USD".to_string(),
                    time_period: "Annual".to_string(),
                    confidence: 0.75,
                }),
                cost_impact: Some(CostImpact {
                    estimated_change: -25000.0,
                    currency: "USD".to_string(),
                    category: "Operational efficiency".to_string(),
                    time_period: "Annual".to_string(),
                }),
                risk_assessment: RiskAssessment {
                    risk_level: RiskLevel::Low,
                    risks: vec![
                        IdentifiedRisk {
                            description: "Data quality degradation over time".to_string(),
                            probability: 0.3,
                            impact: RiskImpact::Minor,
                            category: "Data".to_string(),
                        }
                    ],
                    mitigation_strategies: vec!["Implement data quality monitoring".to_string()],
                },
                strategic_alignment: "Strongly aligns with data-driven decision making initiatives".to_string(),
                stakeholder_impact: vec![
                    StakeholderImpact {
                        stakeholder: "Data Science Team".to_string(),
                        impact: "Enhanced analytical capabilities".to_string(),
                        severity: ImpactSeverity::Positive,
                    }
                ],
            },
            technical_details: TechnicalDetails {
                execution_duration: std::time::Duration::from_secs(2 * 3600),
                resource_utilization: ResourceUtilization {
                    cpu_utilization: 65.0,
                    memory_utilization: 45.0,
                    gpu_utilization: 80.0,
                    network_utilization: 25.0,
                    cost_efficiency: 0.85,
                },
                performance_metrics: {
                    let mut metrics = HashMap::new();
                    metrics.insert("accuracy".to_string(), 0.94);
                    metrics.insert("precision".to_string(), 0.92);
                    metrics.insert("recall".to_string(), 0.89);
                    metrics.insert("processing_time".to_string(), 45.5);
                    metrics
                },
                error_analysis: Some(ErrorAnalysis {
                    error_rate: 0.03,
                    common_errors: vec!["Data format inconsistencies".to_string()],
                    impact_assessment: "Minor impact on overall results".to_string(),
                    remediation_suggestions: vec!["Implement data validation".to_string()],
                }),
                quality_metrics: QualityMetrics {
                    data_quality_score: 0.92,
                    consistency_score: 0.89,
                    completeness_score: 0.95,
                    accuracy_score: 0.94,
                },
            },
            generated_at: Utc::now(),
            confidence_score: 0.88,
        }
    }

    /// Set custom model
    pub fn set_model(&mut self, model: String) {
        self.model = model;
    }

    /// Get explanation summary
    pub fn get_summary(&self, explained: &ExplainedResult) -> String {
        format!(
            "Goal {} analysis: {} findings, {} recommendations. Impact score: {:.1}/10. Confidence: {:.1}%",
            explained.goal_id,
            explained.detailed_findings.len(),
            explained.recommendations.len(),
            explained.business_impact.impact_score,
            explained.confidence_score * 100.0
        )
    }
}

impl Default for BusinessImpact {
    fn default() -> Self {
        Self {
            impact_score: 5.0,
            revenue_impact: None,
            cost_impact: None,
            risk_assessment: RiskAssessment {
                risk_level: RiskLevel::Medium,
                risks: Vec::new(),
                mitigation_strategies: Vec::new(),
            },
            strategic_alignment: "Alignment assessment pending".to_string(),
            stakeholder_impact: Vec::new(),
        }
    }
}

impl Default for TechnicalDetails {
    fn default() -> Self {
        Self {
            execution_duration: std::time::Duration::from_secs(0),
            resource_utilization: ResourceUtilization::default(),
            performance_metrics: HashMap::new(),
            error_analysis: None,
            quality_metrics: QualityMetrics::default(),
        }
    }
}

impl Default for ResourceUtilization {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            gpu_utilization: 0.0,
            network_utilization: 0.0,
            cost_efficiency: 1.0,
        }
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            data_quality_score: 1.0,
            consistency_score: 1.0,
            completeness_score: 1.0,
            accuracy_score: 1.0,
        }
    }
}

impl Default for ResultExplainer {
    fn default() -> Self {
        Self::new(None).expect("Failed to create default result explainer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goal::{BusinessGoal, GoalCategory, GoalPriority};

    fn create_test_explainer() -> ResultExplainer {
        ResultExplainer::new(Some("test-api-key".to_string())).unwrap()
    }

    fn create_test_goal() -> BusinessGoal {
        let mut goal = BusinessGoal::new(
            "Analyze customer data for insights".to_string(),
            "test@example.com".to_string(),
        );
        goal.category = GoalCategory::DataAnalysis;
        goal.priority = GoalPriority::High;
        goal
    }

    fn create_test_execution_data() -> HashMap<String, serde_json::Value> {
        let mut data = HashMap::new();
        data.insert(
            "accuracy".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(0.94).unwrap()),
        );
        data.insert(
            "completion_percentage".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(95.0).unwrap()),
        );
        data.insert(
            "execution_duration".to_string(),
            serde_json::Value::Number(serde_json::Number::from(7200)),
        ); // 2 hours

        let mut performance_metrics = serde_json::Map::new();
        performance_metrics.insert(
            "precision".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(0.92).unwrap()),
        );
        performance_metrics.insert(
            "recall".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(0.89).unwrap()),
        );
        data.insert(
            "performance_metrics".to_string(),
            serde_json::Value::Object(performance_metrics),
        );

        data
    }

    #[test]
    fn test_explainer_creation() {
        let explainer = create_test_explainer();
        assert_eq!(explainer.model, "gpt-4");
        assert!(!explainer.system_prompt.is_empty());
        assert!(!explainer.templates.data_analysis.is_empty());
    }

    #[test]
    fn test_explainer_creation_without_api_key() {
        let explainer = ResultExplainer::new(None);
        assert!(explainer.is_ok());
    }

    #[test]
    fn test_get_template_for_goal() {
        let explainer = create_test_explainer();
        let mut goal = create_test_goal();

        goal.category = GoalCategory::DataAnalysis;
        assert_eq!(
            explainer.get_template_for_goal(&goal),
            explainer.templates.data_analysis
        );

        goal.category = GoalCategory::MachineLearning;
        assert_eq!(
            explainer.get_template_for_goal(&goal),
            explainer.templates.machine_learning
        );

        goal.category = GoalCategory::BusinessIntelligence;
        assert_eq!(
            explainer.get_template_for_goal(&goal),
            explainer.templates.business_intelligence
        );

        goal.category = GoalCategory::Optimization;
        assert_eq!(
            explainer.get_template_for_goal(&goal),
            explainer.templates.optimization
        );

        goal.category = GoalCategory::Research;
        assert_eq!(
            explainer.get_template_for_goal(&goal),
            explainer.templates.default
        );
    }

    #[test]
    fn test_extract_technical_details() {
        let explainer = create_test_explainer();
        let execution_data = create_test_execution_data();

        let details = explainer.extract_technical_details(&execution_data);
        assert_eq!(
            details.execution_duration,
            std::time::Duration::from_secs(7200)
        );
        assert_eq!(details.performance_metrics.len(), 2);
        assert_eq!(details.performance_metrics.get("precision"), Some(&0.92));
        assert_eq!(details.performance_metrics.get("recall"), Some(&0.89));
    }

    #[test]
    fn test_analyze_success_criteria() {
        let explainer = create_test_explainer();
        let mut goal = create_test_goal();
        goal.add_criterion(Criterion::Accuracy { min_accuracy: 0.9 });
        goal.add_criterion(Criterion::Completion { percentage: 90.0 });

        let execution_data = create_test_execution_data();
        let analysis = explainer.analyze_success_criteria(&goal, &execution_data);

        assert_eq!(analysis.len(), 2);

        // Accuracy criterion
        assert!(analysis[0].met); // 0.94 >= 0.9
        assert_eq!(analysis[0].achieved_value, Some(0.94));
        assert_eq!(analysis[0].target_value, 0.9);

        // Completion criterion
        assert!(analysis[1].met); // 95.0 >= 90.0
        assert_eq!(analysis[1].achieved_value, Some(95.0));
        assert_eq!(analysis[1].target_value, 90.0);
    }

    #[test]
    fn test_analyze_success_criteria_unmet() {
        let explainer = create_test_explainer();
        let mut goal = create_test_goal();
        goal.add_criterion(Criterion::Accuracy { min_accuracy: 0.98 }); // Higher than achieved 0.94

        let execution_data = create_test_execution_data();
        let analysis = explainer.analyze_success_criteria(&goal, &execution_data);

        assert_eq!(analysis.len(), 1);
        assert!(!analysis[0].met); // 0.94 < 0.98
        assert_eq!(analysis[0].achieved_value, Some(0.94));
        assert_eq!(analysis[0].target_value, 0.98);
    }

    #[cfg(feature = "mock")]
    #[tokio::test]
    async fn test_explain_results_mock() {
        let explainer = create_test_explainer();
        let goal = create_test_goal();
        let execution_data = create_test_execution_data();

        let result = explainer.explain_results(&goal, &execution_data).await;
        assert!(result.is_ok());

        let explained = result.unwrap();
        assert_eq!(explained.goal_id, goal.goal_id);
        assert!(!explained.executive_summary.is_empty());
        assert!(!explained.detailed_findings.is_empty());
        assert!(!explained.key_insights.is_empty());
        assert!(!explained.recommendations.is_empty());
        assert!(!explained.visualizations.is_empty());
        assert!(explained.confidence_score > 0.0);
    }

    #[test]
    fn test_set_model() {
        let mut explainer = create_test_explainer();
        explainer.set_model("gpt-3.5-turbo".to_string());
        assert_eq!(explainer.model, "gpt-3.5-turbo");
    }

    #[test]
    fn test_get_summary() {
        let explainer = create_test_explainer();
        let explained = ExplainedResult {
            goal_id: "test-goal".to_string(),
            executive_summary: "Test summary".to_string(),
            detailed_findings: vec![Finding {
                finding_id: "f1".to_string(),
                title: "Test Finding".to_string(),
                description: "Test".to_string(),
                category: FindingCategory::Insight,
                importance: ImportanceLevel::High,
                evidence: Vec::new(),
                statistical_significance: None,
                confidence: 0.9,
            }],
            criteria_analysis: Vec::new(),
            key_insights: Vec::new(),
            recommendations: vec![Recommendation {
                recommendation_id: "r1".to_string(),
                title: "Test Rec".to_string(),
                description: "Test".to_string(),
                priority: RecommendationPriority::High,
                expected_impact: "Test".to_string(),
                implementation_effort: EffortLevel::Low,
                timeline: "1 week".to_string(),
                required_resources: Vec::new(),
                success_metrics: Vec::new(),
            }],
            visualizations: Vec::new(),
            business_impact: BusinessImpact {
                impact_score: 8.5,
                ..Default::default()
            },
            technical_details: TechnicalDetails::default(),
            generated_at: Utc::now(),
            confidence_score: 0.92,
        };

        let summary = explainer.get_summary(&explained);
        assert!(summary.contains("test-goal"));
        assert!(summary.contains("1 findings"));
        assert!(summary.contains("1 recommendations"));
        assert!(summary.contains("8.5/10"));
        assert!(summary.contains("92.0%"));
    }

    #[test]
    fn test_finding_serialization() {
        let finding = Finding {
            finding_id: "test-finding".to_string(),
            title: "Test Finding".to_string(),
            description: "Test description".to_string(),
            category: FindingCategory::Correlation,
            importance: ImportanceLevel::High,
            evidence: vec![Evidence {
                evidence_type: EvidenceType::Statistical,
                description: "Test evidence".to_string(),
                source: "Test source".to_string(),
                reliability: 0.95,
                collected_at: Utc::now(),
            }],
            statistical_significance: Some(0.001),
            confidence: 0.92,
        };

        let serialized = serde_json::to_string(&finding).unwrap();
        let deserialized: Finding = serde_json::from_str(&serialized).unwrap();

        assert_eq!(finding.finding_id, deserialized.finding_id);
        assert_eq!(finding.category, deserialized.category);
        assert_eq!(finding.importance, deserialized.importance);
        assert_eq!(finding.confidence, deserialized.confidence);
    }

    #[test]
    fn test_recommendation_serialization() {
        let recommendation = Recommendation {
            recommendation_id: "test-rec".to_string(),
            title: "Test Recommendation".to_string(),
            description: "Test description".to_string(),
            priority: RecommendationPriority::Urgent,
            expected_impact: "High impact".to_string(),
            implementation_effort: EffortLevel::Medium,
            timeline: "3 months".to_string(),
            required_resources: vec!["Resource 1".to_string(), "Resource 2".to_string()],
            success_metrics: vec!["Metric 1".to_string()],
        };

        let serialized = serde_json::to_string(&recommendation).unwrap();
        let deserialized: Recommendation = serde_json::from_str(&serialized).unwrap();

        assert_eq!(
            recommendation.recommendation_id,
            deserialized.recommendation_id
        );
        assert_eq!(recommendation.priority, deserialized.priority);
        assert_eq!(
            recommendation.implementation_effort,
            deserialized.implementation_effort
        );
    }

    #[test]
    fn test_business_impact_default() {
        let impact = BusinessImpact::default();
        assert_eq!(impact.impact_score, 5.0);
        assert_eq!(impact.risk_assessment.risk_level, RiskLevel::Medium);
        assert!(impact.revenue_impact.is_none());
        assert!(impact.cost_impact.is_none());
    }

    #[test]
    fn test_technical_details_default() {
        let details = TechnicalDetails::default();
        assert_eq!(
            details.execution_duration,
            std::time::Duration::from_secs(0)
        );
        assert_eq!(details.resource_utilization.cpu_utilization, 0.0);
        assert_eq!(details.quality_metrics.data_quality_score, 1.0);
        assert!(details.error_analysis.is_none());
    }

    #[test]
    fn test_enum_serialization() {
        // Test FindingCategory
        let categories = vec![
            FindingCategory::Insight,
            FindingCategory::Anomaly,
            FindingCategory::Trend,
            FindingCategory::Correlation,
            FindingCategory::Prediction,
            FindingCategory::Performance,
            FindingCategory::Quality,
            FindingCategory::Risk,
        ];

        for category in categories {
            let serialized = serde_json::to_string(&category).unwrap();
            let deserialized: FindingCategory = serde_json::from_str(&serialized).unwrap();
            assert_eq!(category, deserialized);
        }

        // Test ImportanceLevel ordering
        assert!(ImportanceLevel::Critical > ImportanceLevel::High);
        assert!(ImportanceLevel::High > ImportanceLevel::Medium);
        assert!(ImportanceLevel::Medium > ImportanceLevel::Low);

        // Test VisualizationType
        let viz_types = vec![
            VisualizationType::BarChart,
            VisualizationType::LineChart,
            VisualizationType::ScatterPlot,
            VisualizationType::Dashboard,
        ];

        for viz_type in viz_types {
            let serialized = serde_json::to_string(&viz_type).unwrap();
            let deserialized: VisualizationType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(viz_type, deserialized);
        }
    }

    #[test]
    fn test_risk_assessment_serialization() {
        let risk_assessment = RiskAssessment {
            risk_level: RiskLevel::High,
            risks: vec![IdentifiedRisk {
                description: "Test risk".to_string(),
                probability: 0.7,
                impact: RiskImpact::Major,
                category: "Technical".to_string(),
            }],
            mitigation_strategies: vec!["Mitigation 1".to_string()],
        };

        let serialized = serde_json::to_string(&risk_assessment).unwrap();
        let deserialized: RiskAssessment = serde_json::from_str(&serialized).unwrap();

        assert_eq!(risk_assessment.risk_level, deserialized.risk_level);
        assert_eq!(risk_assessment.risks.len(), deserialized.risks.len());
        assert_eq!(
            risk_assessment.mitigation_strategies,
            deserialized.mitigation_strategies
        );
    }

    #[test]
    fn test_default_explainer() {
        let explainer = ResultExplainer::default();
        assert_eq!(explainer.model, "gpt-4");
        assert!(!explainer.system_prompt.is_empty());
    }

    #[test]
    fn test_visualization_suggestion() {
        let viz = VisualizationSuggestion {
            viz_type: VisualizationType::HeatMap,
            title: "Correlation Matrix".to_string(),
            description: "Shows correlations between variables".to_string(),
            data_fields: vec!["var1".to_string(), "var2".to_string()],
            insights_to_highlight: vec!["Strong correlations".to_string()],
        };

        let serialized = serde_json::to_string(&viz).unwrap();
        let deserialized: VisualizationSuggestion = serde_json::from_str(&serialized).unwrap();

        assert_eq!(viz.viz_type, deserialized.viz_type);
        assert_eq!(viz.title, deserialized.title);
        assert_eq!(viz.data_fields, deserialized.data_fields);
    }
}
