//! Integration tests for AI safety compliance

use async_trait::async_trait;
use chrono::Utc;
use stratoswarm_evolution_global::{
    ai_safety_compliance::{
        BiasDetectionResult, BiasDetector, BiasType, ComplianceStatus, EthicalAssessment,
        EthicalAssessor, EthicalPrinciple, SafetyComplianceConfig, SafetyComplianceManager,
        SafetyRule, SafetyRuleType, SafetySeverity,
    },
    error::EvolutionGlobalResult,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

/// Mock bias detector for integration tests
struct IntegrationBiasDetector {
    bias_scores: Arc<Mutex<HashMap<String, f64>>>,
}

impl IntegrationBiasDetector {
    fn new() -> Self {
        Self {
            bias_scores: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    async fn set_bias_score(&self, model_id: &str, score: f64) {
        let mut scores = self.bias_scores.lock().await;
        scores.insert(model_id.to_string(), score);
    }
}

#[async_trait]
impl BiasDetector for IntegrationBiasDetector {
    async fn detect_bias(&self, model_id: &str) -> EvolutionGlobalResult<BiasDetectionResult> {
        let scores = self.bias_scores.lock().await;
        let bias_score = scores.get(model_id).cloned().unwrap_or(0.3);

        let bias_types = if bias_score > 0.5 {
            vec![BiasType::Gender, BiasType::Age]
        } else {
            vec![]
        };

        Ok(BiasDetectionResult {
            model_id: model_id.to_string(),
            bias_score,
            bias_types,
            affected_groups: vec!["test_group".to_string()],
            mitigation_suggestions: vec!["Rebalance training data".to_string()],
            confidence: 0.9,
            timestamp: Utc::now(),
        })
    }

    async fn suggest_mitigation(
        &self,
        _bias_result: &BiasDetectionResult,
    ) -> EvolutionGlobalResult<Vec<String>> {
        Ok(vec![
            "Implement data augmentation".to_string(),
            "Add fairness constraints".to_string(),
        ])
    }
}

/// Mock ethical assessor for integration tests
struct IntegrationEthicalAssessor {
    ethical_scores: Arc<Mutex<HashMap<String, f64>>>,
}

impl IntegrationEthicalAssessor {
    fn new() -> Self {
        Self {
            ethical_scores: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    async fn set_ethical_score(&self, model_id: &str, score: f64) {
        let mut scores = self.ethical_scores.lock().await;
        scores.insert(model_id.to_string(), score);
    }
}

#[async_trait]
impl EthicalAssessor for IntegrationEthicalAssessor {
    async fn assess_ethics(&self, model_id: &str) -> EvolutionGlobalResult<EthicalAssessment> {
        let scores = self.ethical_scores.lock().await;
        let ethical_score = scores.get(model_id).cloned().unwrap_or(0.85);

        Ok(EthicalAssessment {
            model_id: model_id.to_string(),
            assessment_id: Uuid::new_v4(),
            ethical_score,
            principles_checked: vec![
                EthicalPrinciple::Fairness,
                EthicalPrinciple::Transparency,
                EthicalPrinciple::Privacy,
            ],
            violations: vec![],
            recommendations: vec!["Improve model explainability".to_string()],
            assessor: "IntegrationTest".to_string(),
            timestamp: Utc::now(),
        })
    }

    async fn validate_principle(
        &self,
        _model_id: &str,
        principle: EthicalPrinciple,
    ) -> EvolutionGlobalResult<bool> {
        match principle {
            EthicalPrinciple::Fairness => Ok(true),
            EthicalPrinciple::Privacy => Ok(true),
            _ => Ok(false),
        }
    }
}

fn create_test_safety_rule(rule_type: SafetyRuleType, severity: SafetySeverity) -> SafetyRule {
    SafetyRule {
        id: Uuid::new_v4(),
        name: format!("Test Rule {:?}", rule_type),
        description: "Integration test rule".to_string(),
        rule_type,
        severity,
        parameters: HashMap::new(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        active: true,
    }
}

#[tokio::test]
async fn test_complete_safety_compliance_workflow() {
    let config = SafetyComplianceConfig::default();
    let bias_detector = Arc::new(IntegrationBiasDetector::new());
    let ethical_assessor = Arc::new(IntegrationEthicalAssessor::new());

    let manager =
        SafetyComplianceManager::new(config, bias_detector.clone(), ethical_assessor.clone())
            .unwrap();

    // Add safety rules
    let bias_rule = create_test_safety_rule(SafetyRuleType::BiasDetection, SafetySeverity::High);
    let ethics_rule =
        create_test_safety_rule(SafetyRuleType::EthicalCompliance, SafetySeverity::Critical);

    manager.add_safety_rule(bias_rule.clone()).await.unwrap();
    manager.add_safety_rule(ethics_rule.clone()).await.unwrap();

    // Set up test model with good scores
    bias_detector.set_bias_score("model-good", 0.2).await;
    ethical_assessor.set_ethical_score("model-good", 0.9).await;

    // Validate model
    let validation_results = manager.validate_model("model-good").await.unwrap();
    assert_eq!(validation_results.len(), 2);
    assert!(validation_results.iter().all(|r| r.passed));

    // Check compliance status
    let status = manager.get_compliance_status("model-good").await.unwrap();
    assert_eq!(status, ComplianceStatus::Compliant);
}

#[tokio::test]
async fn test_bias_detection_workflow() {
    let config = SafetyComplianceConfig {
        bias_threshold: 0.5,
        ..Default::default()
    };
    let bias_detector = Arc::new(IntegrationBiasDetector::new());
    let ethical_assessor = Arc::new(IntegrationEthicalAssessor::new());

    let manager =
        SafetyComplianceManager::new(config, bias_detector.clone(), ethical_assessor).unwrap();

    // Test model with high bias
    bias_detector.set_bias_score("model-biased", 0.8).await;
    let bias_result = manager.detect_model_bias("model-biased").await.unwrap();
    assert_eq!(bias_result.bias_score, 0.8);
    assert!(!bias_result.bias_types.is_empty());

    // Test model with low bias
    bias_detector.set_bias_score("model-fair", 0.2).await;
    let bias_result = manager.detect_model_bias("model-fair").await.unwrap();
    assert_eq!(bias_result.bias_score, 0.2);
    assert!(bias_result.bias_types.is_empty());
}

#[tokio::test]
async fn test_ethical_compliance_workflow() {
    let config = SafetyComplianceConfig {
        ethical_threshold: 0.8,
        ..Default::default()
    };
    let bias_detector = Arc::new(IntegrationBiasDetector::new());
    let ethical_assessor = Arc::new(IntegrationEthicalAssessor::new());

    let manager =
        SafetyComplianceManager::new(config, bias_detector, ethical_assessor.clone()).unwrap();

    // Test model with high ethical score
    ethical_assessor
        .set_ethical_score("model-ethical", 0.95)
        .await;
    let assessment = manager
        .assess_ethical_compliance("model-ethical")
        .await
        .unwrap();
    assert_eq!(assessment.ethical_score, 0.95);
    assert!(assessment.violations.is_empty());

    // Test model with low ethical score
    ethical_assessor
        .set_ethical_score("model-unethical", 0.6)
        .await;
    let assessment = manager
        .assess_ethical_compliance("model-unethical")
        .await
        .unwrap();
    assert_eq!(assessment.ethical_score, 0.6);
}

#[tokio::test]
async fn test_safety_rule_management() {
    let config = SafetyComplianceConfig::default();
    let bias_detector = Arc::new(IntegrationBiasDetector::new());
    let ethical_assessor = Arc::new(IntegrationEthicalAssessor::new());

    let manager = SafetyComplianceManager::new(config, bias_detector, ethical_assessor).unwrap();

    // Add multiple rules
    let rules = vec![
        create_test_safety_rule(SafetyRuleType::BiasDetection, SafetySeverity::High),
        create_test_safety_rule(SafetyRuleType::OutputSafety, SafetySeverity::Medium),
        create_test_safety_rule(SafetyRuleType::DataPrivacy, SafetySeverity::Critical),
    ];

    for rule in &rules {
        manager.add_safety_rule(rule.clone()).await.unwrap();
    }

    // Get active rules
    let active_rules = manager.get_active_safety_rules().await.unwrap();
    assert_eq!(active_rules.len(), 3);

    // Remove a rule
    manager.remove_safety_rule(rules[0].id).await.unwrap();
    let active_rules = manager.get_active_safety_rules().await.unwrap();
    assert_eq!(active_rules.len(), 2);
}

#[tokio::test]
async fn test_compliance_report_generation() {
    let config = SafetyComplianceConfig::default();
    let bias_detector = Arc::new(IntegrationBiasDetector::new());
    let ethical_assessor = Arc::new(IntegrationEthicalAssessor::new());

    let manager = SafetyComplianceManager::new(config, bias_detector, ethical_assessor).unwrap();

    // Add rules and validate models
    let rule = create_test_safety_rule(SafetyRuleType::BiasDetection, SafetySeverity::High);
    manager.add_safety_rule(rule).await.unwrap();

    // Validate multiple models
    for i in 0..3 {
        manager
            .validate_model(&format!("model-{}", i))
            .await
            .unwrap();
    }

    // Generate global report
    let report = manager.generate_compliance_report(None).await.unwrap();
    assert!(report.contains_key("total_safety_rules"));
    assert!(report.contains_key("monitored_models"));

    // Generate model-specific report
    let report = manager
        .generate_compliance_report(Some("model-0".to_string()))
        .await
        .unwrap();
    assert!(report.contains_key("model_id"));
    assert!(report.contains_key("compliance_status"));
}

#[tokio::test]
async fn test_strict_mode_compliance() {
    let config = SafetyComplianceConfig {
        strict_mode: true,
        ..Default::default()
    };
    let bias_detector = Arc::new(IntegrationBiasDetector::new());
    let ethical_assessor = Arc::new(IntegrationEthicalAssessor::new());

    let manager =
        SafetyComplianceManager::new(config, bias_detector.clone(), ethical_assessor.clone())
            .unwrap();

    // Add rule that will fail
    let rule = create_test_safety_rule(SafetyRuleType::BiasDetection, SafetySeverity::High);
    manager.add_safety_rule(rule).await.unwrap();

    // Set high bias score
    bias_detector.set_bias_score("model-strict", 0.8).await;

    // Validate model
    let results = manager.validate_model("model-strict").await.unwrap();
    assert!(!results[0].passed);

    // In strict mode, any failure means non-compliant
    let status = manager.get_compliance_status("model-strict").await.unwrap();
    assert_eq!(status, ComplianceStatus::NonCompliant);
}

#[tokio::test]
async fn test_audit_log_retention() {
    let config = SafetyComplianceConfig {
        audit_retention_days: 1,
        ..Default::default()
    };
    let bias_detector = Arc::new(IntegrationBiasDetector::new());
    let ethical_assessor = Arc::new(IntegrationEthicalAssessor::new());

    let manager = SafetyComplianceManager::new(config, bias_detector, ethical_assessor).unwrap();

    // Add rules to generate audit entries
    for i in 0..5 {
        let rule = create_test_safety_rule(SafetyRuleType::BiasDetection, SafetySeverity::Low);
        manager.add_safety_rule(rule).await.unwrap();
    }

    // Clean up old entries (none should be old enough yet)
    let cleaned = manager.cleanup_audit_log().await.unwrap();
    assert_eq!(cleaned, 0);
}

#[tokio::test]
async fn test_disabled_safety_compliance() {
    let config = SafetyComplianceConfig {
        enabled: false,
        ..Default::default()
    };
    let bias_detector = Arc::new(IntegrationBiasDetector::new());
    let ethical_assessor = Arc::new(IntegrationEthicalAssessor::new());

    let manager = SafetyComplianceManager::new(config, bias_detector, ethical_assessor).unwrap();

    // Try to add rule with disabled system
    let rule = create_test_safety_rule(SafetyRuleType::BiasDetection, SafetySeverity::High);
    let result = manager.add_safety_rule(rule).await;
    assert!(result.is_err());

    // Validate model returns empty results when disabled
    let results = manager.validate_model("model-test").await.unwrap();
    assert!(results.is_empty());

    // Compliance status is exempt when disabled
    let status = manager.get_compliance_status("model-test").await.unwrap();
    assert_eq!(status, ComplianceStatus::Exempt);
}
