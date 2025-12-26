//! Context analysis for intelligent mode switching

use super::types::*;
use crate::error::{EvolutionEngineError, EvolutionEngineResult};
use std::collections::HashMap;
use std::time::Duration;

/// Analyzes context to recommend optimal validation modes
pub struct ContextAnalyzer {
    config: IntegrationConfig,
    performance_history: Vec<ValidationHistoryEntry>,
}

impl ContextAnalyzer {
    /// Create new context analyzer
    pub fn new(config: IntegrationConfig) -> EvolutionEngineResult<Self> {
        Ok(Self {
            config,
            performance_history: Vec::new(),
        })
    }

    /// Get configured context window size
    pub fn get_window_size(&self) -> usize {
        self.config.context_window_size
    }

    /// Recommend validation mode based on context
    pub fn recommend_mode(&self, context: &ContextMetrics) -> EvolutionEngineResult<ModeDecision> {
        let mut scores = HashMap::new();

        // Score each validation mode
        scores.insert(
            ValidationMode::Empirical,
            self.score_empirical_mode(context),
        );
        scores.insert(ValidationMode::FormalProof, self.score_formal_mode(context));
        scores.insert(ValidationMode::Hybrid, self.score_hybrid_mode(context));

        // Find the best mode
        let (recommended_mode, confidence) = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(mode, score)| (mode.clone(), *score))
            .unwrap_or((ValidationMode::Empirical, 0.5));

        let rationale = self.generate_rationale(&recommended_mode, context);
        let expected_benefits = self.get_expected_benefits(&recommended_mode);
        let potential_risks = self.get_potential_risks(&recommended_mode);
        let resource_estimate = self.estimate_resources(&recommended_mode, context);

        Ok(ModeDecision {
            recommended_mode,
            confidence,
            rationale,
            expected_benefits,
            potential_risks,
            resource_estimate,
        })
    }

    /// Recommend mode for a specific validation request
    pub fn recommend_mode_for_request(
        &self,
        request: &ValidationRequest,
    ) -> EvolutionEngineResult<ModeDecision> {
        let mut decision = self.recommend_mode(&request.context)?;

        // Adjust recommendation based on request specifics
        match request.criticality {
            CriticalityLevel::Critical => {
                // Critical changes prefer formal validation or hybrid
                if decision.recommended_mode == ValidationMode::Empirical {
                    decision.recommended_mode = ValidationMode::Hybrid;
                    decision
                        .rationale
                        .push_str(" Upgraded to hybrid mode for critical modification.");
                }
                decision.confidence = (decision.confidence + 0.1).min(1.0);
            }
            CriticalityLevel::Low => {
                // Low criticality can use lightweight empirical validation
                if decision.recommended_mode == ValidationMode::FormalProof {
                    decision.recommended_mode = ValidationMode::Empirical;
                    decision
                        .rationale
                        .push_str(" Downgraded to empirical mode for low-criticality change.");
                }
            }
            _ => {} // Medium/High use default recommendation
        }

        // Consider time budget
        if let Some(time_budget) = &request.time_budget {
            if *time_budget < self.config.formal_proof_timeout / 2 {
                decision.recommended_mode = ValidationMode::Empirical;
                decision
                    .rationale
                    .push_str(" Time budget favors empirical validation.");
            }
        }

        Ok(decision)
    }

    /// Analyze performance trends from validation history
    pub fn analyze_performance_trends(
        &self,
        context: &ContextMetrics,
    ) -> EvolutionEngineResult<String> {
        let mut analysis = String::new();

        // Analyze recent performance by mode
        let mut mode_performance: HashMap<ValidationMode, (u32, u32, f64)> = HashMap::new();

        for entry in &context.validation_history {
            let (successes, total, total_confidence) = mode_performance
                .entry(entry.mode.clone())
                .or_insert((0, 0, 0.0));

            *total += 1;
            *total_confidence += entry.confidence;
            if entry.success {
                *successes += 1;
            }
        }

        // Generate performance summary
        for (mode, (successes, total, total_confidence)) in mode_performance {
            if total > 0 {
                let success_rate = successes as f64 / total as f64;
                let avg_confidence = total_confidence / total as f64;

                analysis.push_str(&format!(
                    "{:?} mode: {:.1}% success rate, {:.2} avg confidence over {} validations. ",
                    mode,
                    success_rate * 100.0,
                    avg_confidence,
                    total
                ));
            }
        }

        if analysis.is_empty() {
            analysis.push_str("Insufficient validation history for trend analysis.");
        }

        Ok(analysis)
    }

    // Helper methods for scoring different modes

    fn score_empirical_mode(&self, context: &ContextMetrics) -> f64 {
        let mut score = 0.5; // Base score

        // Favor empirical under resource constraints
        if context.system_load > 0.8 {
            score += 0.3;
        }

        // Favor empirical for simpler modifications
        if context.complexity_estimate < 500 {
            score += 0.2;
        }

        // Favor empirical under time pressure
        if context.time_pressure > 0.7 {
            score += 0.25;
        }

        // Historical success rate
        if let Some(&success_rate) = context.mode_success_rates.get(&ValidationMode::Empirical) {
            score += (success_rate - 0.5) * 0.4; // Weight historical performance
        }

        // Available resources
        if context.available_resources.memory_bytes < 2_000_000_000 {
            // Less than 2GB
            score += 0.15;
        }

        score.min(1.0).max(0.0)
    }

    fn score_formal_mode(&self, context: &ContextMetrics) -> f64 {
        let mut score = 0.4; // Lower base score due to resource requirements

        // Favor formal for complex modifications
        if context.complexity_estimate > 1000 {
            score += 0.3;
        }

        // Favor formal when resources are abundant
        if context.system_load < 0.5 && context.available_resources.memory_bytes > 4_000_000_000 {
            score += 0.2;
        }

        // Favor formal when time is not critical
        if context.time_pressure < 0.3 {
            score += 0.15;
        }

        // Historical success rate
        if let Some(&success_rate) = context.mode_success_rates.get(&ValidationMode::FormalProof) {
            score += (success_rate - 0.5) * 0.4;
        }

        // Penalize if timeout threshold is likely to be exceeded
        if context.available_resources.time_budget < self.config.formal_proof_timeout {
            score -= 0.2;
        }

        score.min(1.0).max(0.0)
    }

    fn score_hybrid_mode(&self, context: &ContextMetrics) -> f64 {
        let mut score = 0.6; // Higher base score as it combines both approaches

        // Hybrid is good for medium complexity
        if context.complexity_estimate >= 300 && context.complexity_estimate <= 800 {
            score += 0.2;
        }

        // Hybrid works well with moderate resources
        if context.system_load >= 0.3 && context.system_load <= 0.7 {
            score += 0.15;
        }

        // Historical success rate
        if let Some(&success_rate) = context.mode_success_rates.get(&ValidationMode::Hybrid) {
            score += (success_rate - 0.5) * 0.3;
        }

        // Favor hybrid when both approaches had mixed results historically
        let empirical_rate = context
            .mode_success_rates
            .get(&ValidationMode::Empirical)
            .unwrap_or(&0.5);
        let formal_rate = context
            .mode_success_rates
            .get(&ValidationMode::FormalProof)
            .unwrap_or(&0.5);

        if (empirical_rate - formal_rate).abs() < 0.2 {
            score += 0.1; // Both approaches are similarly effective
        }

        score.min(1.0).max(0.0)
    }

    fn generate_rationale(&self, mode: &ValidationMode, context: &ContextMetrics) -> String {
        match mode {
            ValidationMode::Empirical => {
                let mut reasons = Vec::new();

                if context.system_load > 0.8 {
                    reasons.push("high system load");
                }
                if context.complexity_estimate < 500 {
                    reasons.push("moderate complexity");
                }
                if context.time_pressure > 0.7 {
                    reasons.push("time constraints");
                }
                if context.available_resources.memory_bytes < 2_000_000_000 {
                    reasons.push("limited memory");
                }

                if reasons.is_empty() {
                    "Empirical validation selected based on historical performance".to_string()
                } else {
                    format!(
                        "Empirical validation recommended due to: {}",
                        reasons.join(", ")
                    )
                }
            }
            ValidationMode::FormalProof => {
                let mut reasons = Vec::new();

                if context.complexity_estimate > 1000 {
                    reasons.push("high complexity");
                }
                if context.system_load < 0.5 {
                    reasons.push("abundant resources");
                }
                if context.time_pressure < 0.3 {
                    reasons.push("sufficient time");
                }

                if reasons.is_empty() {
                    "Formal proof selected for maximum rigor".to_string()
                } else {
                    format!("Formal proof recommended due to: {}", reasons.join(", "))
                }
            }
            ValidationMode::Hybrid => {
                "Hybrid approach balances thoroughness with efficiency".to_string()
            }
            ValidationMode::Adaptive => {
                "Adaptive mode will choose optimal approach per validation".to_string()
            }
        }
    }

    fn get_expected_benefits(&self, mode: &ValidationMode) -> Vec<String> {
        match mode {
            ValidationMode::Empirical => vec![
                "Fast validation times".to_string(),
                "Low resource consumption".to_string(),
                "Good practical coverage".to_string(),
                "Scales well with complexity".to_string(),
            ],
            ValidationMode::FormalProof => vec![
                "Mathematical certainty".to_string(),
                "Complete theoretical coverage".to_string(),
                "Provable correctness".to_string(),
                "High confidence results".to_string(),
            ],
            ValidationMode::Hybrid => vec![
                "Balanced approach".to_string(),
                "Multiple validation perspectives".to_string(),
                "Good coverage with efficiency".to_string(),
                "Fallback mechanisms".to_string(),
            ],
            ValidationMode::Adaptive => vec![
                "Context-optimal performance".to_string(),
                "Dynamic optimization".to_string(),
                "Resource-aware operation".to_string(),
            ],
        }
    }

    fn get_potential_risks(&self, mode: &ValidationMode) -> Vec<String> {
        match mode {
            ValidationMode::Empirical => vec![
                "May miss edge cases".to_string(),
                "Statistical confidence limitations".to_string(),
                "Test coverage gaps possible".to_string(),
            ],
            ValidationMode::FormalProof => vec![
                "Resource intensive".to_string(),
                "Long computation times".to_string(),
                "May timeout on complex problems".to_string(),
                "Requires formal specifications".to_string(),
            ],
            ValidationMode::Hybrid => vec![
                "Higher resource usage".to_string(),
                "Complex coordination".to_string(),
                "Potential mode conflicts".to_string(),
            ],
            ValidationMode::Adaptive => vec![
                "Mode switching overhead".to_string(),
                "Unpredictable behavior".to_string(),
            ],
        }
    }

    fn estimate_resources(&self, mode: &ValidationMode, context: &ContextMetrics) -> ResourceUsage {
        match mode {
            ValidationMode::Empirical => ResourceUsage {
                cpu_time: Duration::from_secs(30),
                peak_memory: 200_000_000, // 200MB
                proof_steps: None,
                test_count: Some(50),
            },
            ValidationMode::FormalProof => ResourceUsage {
                cpu_time: Duration::from_secs(180),
                peak_memory: 800_000_000, // 800MB
                proof_steps: Some(context.complexity_estimate * 2),
                test_count: None,
            },
            ValidationMode::Hybrid => ResourceUsage {
                cpu_time: Duration::from_secs(90),
                peak_memory: 400_000_000, // 400MB
                proof_steps: Some(context.complexity_estimate),
                test_count: Some(25),
            },
            ValidationMode::Adaptive => ResourceUsage {
                cpu_time: Duration::from_secs(60),
                peak_memory: 300_000_000, // 300MB
                proof_steps: Some(context.complexity_estimate / 2),
                test_count: Some(30),
            },
        }
    }
}
