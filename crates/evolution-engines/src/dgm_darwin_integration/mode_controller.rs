//! Main controller for Darwin-Gödel integration and mode switching

use super::context_analyzer::ContextAnalyzer;
use super::types::*;
use super::validation_bridge::ValidationBridge;
use crate::error::{EvolutionEngineError, EvolutionEngineResult};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};
use uuid::Uuid;

/// Main controller for Darwin-Gödel integration
pub struct DarwinGodelController {
    config: IntegrationConfig,
    current_mode: ValidationMode,
    context_analyzer: ContextAnalyzer,
    validation_bridge: ValidationBridge,
    last_mode_switch: Option<SystemTime>,
    switch_history: Vec<ModeSwitch>,
    integration_metrics: IntegrationMetrics,
}

impl DarwinGodelController {
    /// Create new Darwin-Gödel controller
    pub fn new(config: IntegrationConfig) -> EvolutionEngineResult<Self> {
        let current_mode = config.default_mode.clone();
        let context_analyzer = ContextAnalyzer::new(config.clone())?;
        let validation_bridge = ValidationBridge::new(config.clone())?;

        let integration_metrics = IntegrationMetrics {
            total_validations: 0,
            validations_by_mode: HashMap::new(),
            success_rates_by_mode: HashMap::new(),
            avg_times_by_mode: HashMap::new(),
            mode_switches: 0,
            switch_success_rate: 1.0,
            resource_efficiency_by_mode: HashMap::new(),
        };

        Ok(Self {
            config,
            current_mode,
            context_analyzer,
            validation_bridge,
            last_mode_switch: None,
            switch_history: Vec::new(),
            integration_metrics,
        })
    }

    /// Get current validation mode
    pub fn get_current_mode(&self) -> ValidationMode {
        self.current_mode.clone()
    }

    /// Check if automatic mode switching is enabled
    pub fn is_auto_switching_enabled(&self) -> bool {
        self.config.auto_switching
    }

    /// Process a validation request
    pub fn process_validation_request(
        &mut self,
        request: ValidationRequest,
    ) -> EvolutionEngineResult<ValidationResponse> {
        let start_time = Instant::now();

        // Determine which mode to use
        let mode_to_use = self.determine_validation_mode(&request)?;

        // Perform validation using the determined mode
        let validation_result = match mode_to_use {
            ValidationMode::Empirical => self.validation_bridge.validate_empirical(&request)?,
            ValidationMode::FormalProof => self.validation_bridge.validate_formal(&request)?,
            ValidationMode::Hybrid => self.validation_bridge.validate_hybrid(&request)?,
            ValidationMode::Adaptive => {
                // For adaptive mode, let context analyzer decide
                let decision = self.context_analyzer.recommend_mode_for_request(&request)?;
                match decision.recommended_mode {
                    ValidationMode::Empirical => {
                        self.validation_bridge.validate_empirical(&request)?
                    }
                    ValidationMode::FormalProof => {
                        self.validation_bridge.validate_formal(&request)?
                    }
                    _ => self.validation_bridge.validate_hybrid(&request)?,
                }
            }
        };

        let validation_time = start_time.elapsed();
        let confidence = validation_result
            .statistical_significance
            .confidence_interval
            .1;

        // Create response
        let response = ValidationResponse {
            request_id: request.id.clone(),
            mode_used: mode_to_use.clone(),
            result: validation_result,
            confidence,
            validation_time,
            resource_usage: ResourceUsage {
                cpu_time: validation_time,
                peak_memory: 100_000_000, // Simplified
                proof_steps: if matches!(mode_to_use, ValidationMode::FormalProof) {
                    Some(request.context.complexity_estimate)
                } else {
                    None
                },
                test_count: if matches!(mode_to_use, ValidationMode::Empirical) {
                    Some(10)
                } else {
                    None
                },
            },
            recommendations: self.generate_recommendations(&request, &mode_to_use, confidence),
        };

        // Update metrics
        self.update_integration_metrics(&response);

        Ok(response)
    }

    /// Manually switch validation mode
    pub fn switch_mode(
        &mut self,
        new_mode: ValidationMode,
        reason: SwitchReason,
        context: &ContextMetrics,
    ) -> EvolutionEngineResult<ModeSwitch> {
        let now = SystemTime::now();

        // Check cooldown period
        if let Some(last_switch) = self.last_mode_switch {
            if now
                .duration_since(last_switch)
                .map_err(|e| EvolutionEngineError::Other(format!("Time error: {}", e)))?
                < self.config.switching_cooldown
            {
                return Err(EvolutionEngineError::Other(
                    "Mode switch cooldown period not elapsed".to_string(),
                ));
            }
        }

        let mode_switch = ModeSwitch {
            id: Uuid::new_v4().to_string(),
            timestamp: now,
            from_mode: self.current_mode.clone(),
            to_mode: new_mode.clone(),
            reason,
            context: context.clone(),
        };

        // Perform the switch
        self.current_mode = new_mode;
        self.last_mode_switch = Some(now);
        self.switch_history.push(mode_switch.clone());
        self.integration_metrics.mode_switches += 1;

        Ok(mode_switch)
    }

    /// Get integration metrics
    pub fn get_integration_metrics(&self) -> EvolutionEngineResult<IntegrationMetrics> {
        Ok(self.integration_metrics.clone())
    }

    // Helper methods

    fn determine_validation_mode(
        &mut self,
        request: &ValidationRequest,
    ) -> EvolutionEngineResult<ValidationMode> {
        // If request has a preferred mode, respect it
        if let Some(preferred) = &request.preferred_mode {
            return Ok(preferred.clone());
        }

        // Use current mode unless auto-switching is enabled
        if !self.config.auto_switching {
            return Ok(self.current_mode.clone());
        }

        // For adaptive mode, always use context analyzer
        if matches!(self.current_mode, ValidationMode::Adaptive) {
            let decision = self.context_analyzer.recommend_mode_for_request(request)?;
            return Ok(decision.recommended_mode);
        }

        // Check if we should switch modes based on context
        let decision = self.context_analyzer.recommend_mode_for_request(request)?;

        // Switch if confidence in recommendation is high and mode is different
        if decision.confidence > 0.8 && decision.recommended_mode != self.current_mode {
            // Perform automatic switch
            let _switch = self.switch_mode(
                decision.recommended_mode.clone(),
                SwitchReason::AutoOptimization,
                &request.context,
            )?;

            return Ok(decision.recommended_mode);
        }

        Ok(self.current_mode.clone())
    }

    fn generate_recommendations(
        &self,
        request: &ValidationRequest,
        mode_used: &ValidationMode,
        confidence: f64,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Confidence-based recommendations
        if confidence < 0.7 {
            recommendations
                .push("Consider using hybrid validation for better confidence".to_string());
        }

        // Mode-specific recommendations
        match mode_used {
            ValidationMode::Empirical => {
                if request.criticality == CriticalityLevel::Critical {
                    recommendations.push(
                        "Consider formal verification for critical modifications".to_string(),
                    );
                }
                if confidence > 0.9 {
                    recommendations
                        .push("High empirical confidence - validation successful".to_string());
                }
            }
            ValidationMode::FormalProof => {
                if request.context.time_pressure > 0.8 {
                    recommendations.push(
                        "Consider empirical validation for time-critical modifications".to_string(),
                    );
                }
                if confidence == 1.0 {
                    recommendations
                        .push("Formal proof successful - maximum confidence achieved".to_string());
                }
            }
            ValidationMode::Hybrid => {
                recommendations
                    .push("Hybrid approach provides balanced validation coverage".to_string());
            }
            ValidationMode::Adaptive => {
                recommendations
                    .push("Adaptive mode selected optimal validation approach".to_string());
            }
        }

        // Resource-based recommendations
        if request.context.system_load > 0.9 {
            recommendations
                .push("High system load detected - consider lighter validation modes".to_string());
        }

        // Complexity-based recommendations
        if request.context.complexity_estimate > 1000
            && matches!(mode_used, ValidationMode::Empirical)
        {
            recommendations.push(
                "High complexity modification may benefit from formal verification".to_string(),
            );
        }

        recommendations
    }

    fn update_integration_metrics(&mut self, response: &ValidationResponse) {
        self.integration_metrics.total_validations += 1;

        // Update validations by mode
        *self
            .integration_metrics
            .validations_by_mode
            .entry(response.mode_used.clone())
            .or_insert(0) += 1;

        // Update success rates
        let current_success_rate = self
            .integration_metrics
            .success_rates_by_mode
            .entry(response.mode_used.clone())
            .or_insert(0.0);

        let validation_success = if response.result.success_rate > 0.5 {
            1.0
        } else {
            0.0
        };
        let mode_validations = *self
            .integration_metrics
            .validations_by_mode
            .get(&response.mode_used)
            .unwrap_or(&1);

        // Running average
        *current_success_rate = (*current_success_rate * (mode_validations - 1) as f64
            + validation_success)
            / mode_validations as f64;

        // Update average times
        let current_avg_time = self
            .integration_metrics
            .avg_times_by_mode
            .entry(response.mode_used.clone())
            .or_insert(Duration::from_secs(0));

        let avg_nanos = (current_avg_time.as_nanos() * (mode_validations - 1) as u128
            + response.validation_time.as_nanos())
            / mode_validations as u128;
        *current_avg_time = Duration::from_nanos(avg_nanos.try_into().unwrap_or(u64::MAX));

        // Update resource efficiency
        let efficiency =
            response.result.success_rate * (1.0 / response.validation_time.as_secs_f64()).min(1.0);
        let current_efficiency = self
            .integration_metrics
            .resource_efficiency_by_mode
            .entry(response.mode_used.clone())
            .or_insert(0.0);

        *current_efficiency = (*current_efficiency * (mode_validations - 1) as f64 + efficiency)
            / mode_validations as f64;
    }
}
