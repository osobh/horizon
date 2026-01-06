//! Settings Bridge
//!
//! Integrates governor (policy management) and quota-manager with Horizon.
//! Provides access to RBAC/ABAC policies, resource quotas, and application settings.
//!
//! Currently uses mock data until governor and quota-manager are fully integrated.

use crate::error::HorizonError;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Policy type.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyType {
    Rbac,
    Abac,
    Quota,
    RateLimit,
}

/// Policy action.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyAction {
    Allow,
    Deny,
    RequireApproval,
}

/// Policy condition operator.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditionOperator {
    Equals,
    NotEquals,
    Contains,
    GreaterThan,
    LessThan,
    In,
    NotIn,
}

/// Policy condition.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PolicyCondition {
    pub field: String,
    pub operator: ConditionOperator,
    pub value: serde_json::Value,
}

/// Access policy.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Policy {
    pub id: String,
    pub name: String,
    pub description: String,
    #[serde(rename = "type")]
    pub policy_type: PolicyType,
    pub action: PolicyAction,
    pub conditions: Vec<PolicyCondition>,
    pub enabled: bool,
    pub priority: u32,
    pub created_at: String,
    pub updated_at: String,
}

/// Resource quota.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Quota {
    pub id: String,
    pub name: String,
    pub resource_type: QuotaResourceType,
    pub limit: f64,
    pub used: f64,
    pub unit: String,
    pub scope: QuotaScope,
    pub scope_id: Option<String>,
    pub reset_period: ResetPeriod,
    pub enabled: bool,
}

/// Quota resource type.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuotaResourceType {
    Gpu,
    Cpu,
    Memory,
    Storage,
    Network,
}

/// Quota scope.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuotaScope {
    User,
    Team,
    Project,
    Global,
}

/// Reset period.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResetPeriod {
    Hourly,
    Daily,
    Weekly,
    Monthly,
    Never,
}

/// Application settings.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AppSettings {
    pub theme: Theme,
    pub notifications_enabled: bool,
    pub auto_refresh_interval_secs: u32,
    pub default_cluster: Option<String>,
    pub telemetry_enabled: bool,
    pub log_level: LogLevel,
}

/// Theme.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Theme {
    Light,
    Dark,
    System,
}

/// Log level.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

/// Settings summary.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SettingsSummary {
    pub policies: Vec<Policy>,
    pub quotas: Vec<Quota>,
    pub app_settings: AppSettings,
    pub policy_evaluation_count: u64,
    pub quota_violations_count: u64,
}

/// Bridge to settings services.
pub struct SettingsBridge {
    state: Arc<RwLock<MockSettingsState>>,
}

struct MockSettingsState {
    policies: Vec<Policy>,
    quotas: Vec<Quota>,
    app_settings: AppSettings,
}

impl MockSettingsState {
    fn new() -> Self {
        let now = chrono::Utc::now().to_rfc3339();

        let policies = vec![
            Policy {
                id: "policy-gpu-access".to_string(),
                name: "GPU Access Control".to_string(),
                description: "Controls access to GPU resources based on team membership".to_string(),
                policy_type: PolicyType::Rbac,
                action: PolicyAction::Allow,
                conditions: vec![PolicyCondition {
                    field: "team".to_string(),
                    operator: ConditionOperator::In,
                    value: serde_json::json!(["ml-research", "infrastructure"]),
                }],
                enabled: true,
                priority: 100,
                created_at: now.clone(),
                updated_at: now.clone(),
            },
            Policy {
                id: "policy-cost-limit".to_string(),
                name: "Cost Limit Policy".to_string(),
                description: "Requires approval for jobs exceeding $500".to_string(),
                policy_type: PolicyType::Abac,
                action: PolicyAction::RequireApproval,
                conditions: vec![PolicyCondition {
                    field: "estimated_cost".to_string(),
                    operator: ConditionOperator::GreaterThan,
                    value: serde_json::json!(500),
                }],
                enabled: true,
                priority: 90,
                created_at: now.clone(),
                updated_at: now.clone(),
            },
            Policy {
                id: "policy-rate-limit".to_string(),
                name: "API Rate Limit".to_string(),
                description: "Limits API calls to 1000/minute per user".to_string(),
                policy_type: PolicyType::RateLimit,
                action: PolicyAction::Deny,
                conditions: vec![PolicyCondition {
                    field: "requests_per_minute".to_string(),
                    operator: ConditionOperator::GreaterThan,
                    value: serde_json::json!(1000),
                }],
                enabled: true,
                priority: 80,
                created_at: now.clone(),
                updated_at: now.clone(),
            },
        ];

        let quotas = vec![
            Quota {
                id: "quota-gpu-ml".to_string(),
                name: "ML Team GPU Quota".to_string(),
                resource_type: QuotaResourceType::Gpu,
                limit: 100.0,
                used: 78.5,
                unit: "hours/day".to_string(),
                scope: QuotaScope::Team,
                scope_id: Some("ml-research".to_string()),
                reset_period: ResetPeriod::Daily,
                enabled: true,
            },
            Quota {
                id: "quota-storage-global".to_string(),
                name: "Global Storage Quota".to_string(),
                resource_type: QuotaResourceType::Storage,
                limit: 10000.0,
                used: 6543.0,
                unit: "GB".to_string(),
                scope: QuotaScope::Global,
                scope_id: None,
                reset_period: ResetPeriod::Never,
                enabled: true,
            },
            Quota {
                id: "quota-memory-project".to_string(),
                name: "LLM Project Memory Quota".to_string(),
                resource_type: QuotaResourceType::Memory,
                limit: 512.0,
                used: 384.0,
                unit: "GB".to_string(),
                scope: QuotaScope::Project,
                scope_id: Some("llm-training".to_string()),
                reset_period: ResetPeriod::Never,
                enabled: true,
            },
        ];

        let app_settings = AppSettings {
            theme: Theme::Dark,
            notifications_enabled: true,
            auto_refresh_interval_secs: 30,
            default_cluster: Some("production".to_string()),
            telemetry_enabled: true,
            log_level: LogLevel::Info,
        };

        Self {
            policies,
            quotas,
            app_settings,
        }
    }
}

impl SettingsBridge {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(MockSettingsState::new())),
        }
    }

    pub async fn get_summary(&self) -> SettingsSummary {
        let state = self.state.read().await;
        SettingsSummary {
            policies: state.policies.clone(),
            quotas: state.quotas.clone(),
            app_settings: state.app_settings.clone(),
            policy_evaluation_count: 145_230,
            quota_violations_count: 12,
        }
    }

    pub async fn get_policies(&self) -> Vec<Policy> {
        let state = self.state.read().await;
        state.policies.clone()
    }

    pub async fn get_quotas(&self) -> Vec<Quota> {
        let state = self.state.read().await;
        state.quotas.clone()
    }

    pub async fn get_app_settings(&self) -> AppSettings {
        let state = self.state.read().await;
        state.app_settings.clone()
    }

    pub async fn create_policy(&self, policy: Policy) -> Result<Policy, HorizonError> {
        // Validate policy name
        if policy.name.trim().is_empty() {
            return Err(HorizonError::InvalidConfig(
                "Policy name cannot be empty".to_string(),
            ));
        }

        let mut state = self.state.write().await;
        // Use UUID instead of timestamp to prevent ID collisions
        let new_policy = Policy {
            id: format!("policy-{}", Uuid::new_v4()),
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
            ..policy
        };
        state.policies.push(new_policy.clone());
        Ok(new_policy)
    }

    pub async fn update_policy(
        &self,
        id: String,
        updates: serde_json::Value,
    ) -> Result<(), HorizonError> {
        if id.trim().is_empty() {
            return Err(HorizonError::InvalidConfig(
                "Policy ID cannot be empty".to_string(),
            ));
        }

        let mut state = self.state.write().await;
        let policy = state
            .policies
            .iter_mut()
            .find(|p| p.id == id)
            .ok_or_else(|| HorizonError::NotFound(format!("Policy '{}' not found", id)))?;

        if let Some(name) = updates.get("name").and_then(|v| v.as_str()) {
            if name.trim().is_empty() {
                return Err(HorizonError::InvalidConfig(
                    "Policy name cannot be empty".to_string(),
                ));
            }
            policy.name = name.to_string();
        }
        if let Some(description) = updates.get("description").and_then(|v| v.as_str()) {
            policy.description = description.to_string();
        }
        if let Some(enabled) = updates.get("enabled").and_then(|v| v.as_bool()) {
            policy.enabled = enabled;
        }
        if let Some(priority) = updates.get("priority").and_then(|v| v.as_u64()) {
            if priority == 0 {
                return Err(HorizonError::InvalidConfig(
                    "Priority must be greater than 0".to_string(),
                ));
            }
            policy.priority = priority as u32;
        }
        policy.updated_at = chrono::Utc::now().to_rfc3339();
        Ok(())
    }

    pub async fn delete_policy(&self, id: String) -> Result<(), HorizonError> {
        if id.trim().is_empty() {
            return Err(HorizonError::InvalidConfig(
                "Policy ID cannot be empty".to_string(),
            ));
        }

        let mut state = self.state.write().await;
        let initial_len = state.policies.len();
        state.policies.retain(|p| p.id != id);

        if state.policies.len() == initial_len {
            return Err(HorizonError::NotFound(format!("Policy '{}' not found", id)));
        }

        Ok(())
    }

    pub async fn toggle_policy(&self, id: String, enabled: bool) -> Result<(), HorizonError> {
        if id.trim().is_empty() {
            return Err(HorizonError::InvalidConfig(
                "Policy ID cannot be empty".to_string(),
            ));
        }

        let mut state = self.state.write().await;
        let policy = state
            .policies
            .iter_mut()
            .find(|p| p.id == id)
            .ok_or_else(|| HorizonError::NotFound(format!("Policy '{}' not found", id)))?;

        policy.enabled = enabled;
        policy.updated_at = chrono::Utc::now().to_rfc3339();
        Ok(())
    }

    pub async fn set_quota(&self, quota: Quota) -> Result<Quota, HorizonError> {
        // Validate quota name and limit
        if quota.name.trim().is_empty() {
            return Err(HorizonError::InvalidConfig(
                "Quota name cannot be empty".to_string(),
            ));
        }
        if quota.limit <= 0.0 {
            return Err(HorizonError::InvalidConfig(
                "Quota limit must be greater than 0".to_string(),
            ));
        }

        let mut state = self.state.write().await;
        // Use UUID instead of timestamp to prevent ID collisions
        let new_quota = Quota {
            id: format!("quota-{}", Uuid::new_v4()),
            used: 0.0,
            ..quota
        };
        state.quotas.push(new_quota.clone());
        Ok(new_quota)
    }

    pub async fn update_quota(
        &self,
        id: String,
        updates: serde_json::Value,
    ) -> Result<(), HorizonError> {
        if id.trim().is_empty() {
            return Err(HorizonError::InvalidConfig(
                "Quota ID cannot be empty".to_string(),
            ));
        }

        let mut state = self.state.write().await;
        let quota = state
            .quotas
            .iter_mut()
            .find(|q| q.id == id)
            .ok_or_else(|| HorizonError::NotFound(format!("Quota '{}' not found", id)))?;

        if let Some(name) = updates.get("name").and_then(|v| v.as_str()) {
            if name.trim().is_empty() {
                return Err(HorizonError::InvalidConfig(
                    "Quota name cannot be empty".to_string(),
                ));
            }
            quota.name = name.to_string();
        }
        if let Some(limit) = updates.get("limit").and_then(|v| v.as_f64()) {
            if limit <= 0.0 {
                return Err(HorizonError::InvalidConfig(
                    "Quota limit must be greater than 0".to_string(),
                ));
            }
            quota.limit = limit;
        }
        if let Some(enabled) = updates.get("enabled").and_then(|v| v.as_bool()) {
            quota.enabled = enabled;
        }
        Ok(())
    }

    pub async fn delete_quota(&self, id: String) -> Result<(), HorizonError> {
        if id.trim().is_empty() {
            return Err(HorizonError::InvalidConfig(
                "Quota ID cannot be empty".to_string(),
            ));
        }

        let mut state = self.state.write().await;
        let initial_len = state.quotas.len();
        state.quotas.retain(|q| q.id != id);

        if state.quotas.len() == initial_len {
            return Err(HorizonError::NotFound(format!("Quota '{}' not found", id)));
        }

        Ok(())
    }

    pub async fn update_app_settings(&self, updates: serde_json::Value) -> Result<(), HorizonError> {
        let mut state = self.state.write().await;
        if let Some(notifications) = updates.get("notifications_enabled").and_then(|v| v.as_bool()) {
            state.app_settings.notifications_enabled = notifications;
        }
        if let Some(interval) = updates.get("auto_refresh_interval_secs").and_then(|v| v.as_u64()) {
            // Validate interval is reasonable (between 1 second and 1 hour)
            if interval == 0 || interval > 3600 {
                return Err(HorizonError::InvalidConfig(
                    "Refresh interval must be between 1 and 3600 seconds".to_string(),
                ));
            }
            state.app_settings.auto_refresh_interval_secs = interval as u32;
        }
        if let Some(telemetry) = updates.get("telemetry_enabled").and_then(|v| v.as_bool()) {
            state.app_settings.telemetry_enabled = telemetry;
        }
        if let Some(theme) = updates.get("theme").and_then(|v| v.as_str()) {
            state.app_settings.theme = match theme {
                "light" => Theme::Light,
                "system" => Theme::System,
                _ => Theme::Dark,
            };
        }
        if let Some(log_level) = updates.get("log_level").and_then(|v| v.as_str()) {
            state.app_settings.log_level = match log_level {
                "debug" => LogLevel::Debug,
                "warn" => LogLevel::Warn,
                "error" => LogLevel::Error,
                _ => LogLevel::Info,
            };
        }
        Ok(())
    }
}

impl Default for SettingsBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_summary() {
        let bridge = SettingsBridge::new();
        let summary = bridge.get_summary().await;

        assert!(!summary.policies.is_empty());
        assert!(!summary.quotas.is_empty());
        assert!(summary.policy_evaluation_count > 0);
    }

    #[tokio::test]
    async fn test_get_policies() {
        let bridge = SettingsBridge::new();
        let policies = bridge.get_policies().await;

        assert!(policies.len() >= 3, "Should have at least 3 policies");
        for policy in &policies {
            assert!(!policy.id.is_empty());
            assert!(!policy.name.is_empty());
        }
    }

    #[tokio::test]
    async fn test_get_quotas() {
        let bridge = SettingsBridge::new();
        let quotas = bridge.get_quotas().await;

        assert!(quotas.len() >= 3, "Should have at least 3 quotas");
        for quota in &quotas {
            assert!(!quota.id.is_empty());
            assert!(!quota.name.is_empty());
            assert!(quota.limit > 0.0);
        }
    }

    #[tokio::test]
    async fn test_get_app_settings() {
        let bridge = SettingsBridge::new();
        let settings = bridge.get_app_settings().await;

        assert!(settings.auto_refresh_interval_secs > 0);
    }

    #[tokio::test]
    async fn test_create_policy_success() {
        let bridge = SettingsBridge::new();

        let new_policy = Policy {
            id: "temp".to_string(),
            name: "Test Policy".to_string(),
            description: "A test policy".to_string(),
            policy_type: PolicyType::Rbac,
            action: PolicyAction::Allow,
            conditions: vec![],
            enabled: true,
            priority: 50,
            created_at: String::new(),
            updated_at: String::new(),
        };

        let result = bridge.create_policy(new_policy.clone()).await;
        assert!(result.is_ok());

        let created = result.unwrap();
        assert!(created.id.starts_with("policy-"));
        assert_eq!(created.name, "Test Policy");
        assert_eq!(created.priority, 50);

        // Verify it was added
        let policies = bridge.get_policies().await;
        assert!(policies.iter().any(|p| p.name == "Test Policy"));
    }

    #[tokio::test]
    async fn test_create_policy_validation_empty_name() {
        let bridge = SettingsBridge::new();

        let invalid_policy = Policy {
            id: "temp".to_string(),
            name: "".to_string(),
            description: "Test".to_string(),
            policy_type: PolicyType::Rbac,
            action: PolicyAction::Allow,
            conditions: vec![],
            enabled: true,
            priority: 50,
            created_at: String::new(),
            updated_at: String::new(),
        };

        let result = bridge.create_policy(invalid_policy).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), HorizonError::InvalidConfig(_)));
    }

    #[tokio::test]
    async fn test_update_policy_success() {
        let bridge = SettingsBridge::new();

        let policies = bridge.get_policies().await;
        let test_policy = policies.first().expect("Should have at least one policy");

        let updates = serde_json::json!({
            "name": "Updated Policy Name",
            "priority": 200
        });

        let result = bridge.update_policy(test_policy.id.clone(), updates).await;
        assert!(result.is_ok());

        // Verify update
        let updated_policies = bridge.get_policies().await;
        let updated = updated_policies
            .iter()
            .find(|p| p.id == test_policy.id)
            .expect("Policy should still exist");
        assert_eq!(updated.name, "Updated Policy Name");
        assert_eq!(updated.priority, 200);
    }

    #[tokio::test]
    async fn test_update_policy_validation_empty_name() {
        let bridge = SettingsBridge::new();

        let policies = bridge.get_policies().await;
        let test_policy = policies.first().expect("Should have at least one policy");

        let updates = serde_json::json!({
            "name": ""
        });

        let result = bridge.update_policy(test_policy.id.clone(), updates).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), HorizonError::InvalidConfig(_)));
    }

    #[tokio::test]
    async fn test_update_policy_validation_zero_priority() {
        let bridge = SettingsBridge::new();

        let policies = bridge.get_policies().await;
        let test_policy = policies.first().expect("Should have at least one policy");

        let updates = serde_json::json!({
            "priority": 0
        });

        let result = bridge.update_policy(test_policy.id.clone(), updates).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), HorizonError::InvalidConfig(_)));
    }

    #[tokio::test]
    async fn test_update_policy_not_found() {
        let bridge = SettingsBridge::new();

        let updates = serde_json::json!({
            "name": "Updated Name"
        });

        let result = bridge
            .update_policy("nonexistent-policy-id".to_string(), updates)
            .await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), HorizonError::NotFound(_)));
    }

    #[tokio::test]
    async fn test_delete_policy_success() {
        let bridge = SettingsBridge::new();

        let policies = bridge.get_policies().await;
        let initial_count = policies.len();
        let test_policy = policies.first().expect("Should have at least one policy");

        let result = bridge.delete_policy(test_policy.id.clone()).await;
        assert!(result.is_ok());

        // Verify deletion
        let updated_policies = bridge.get_policies().await;
        assert_eq!(updated_policies.len(), initial_count - 1);
        assert!(!updated_policies.iter().any(|p| p.id == test_policy.id));
    }

    #[tokio::test]
    async fn test_delete_policy_not_found() {
        let bridge = SettingsBridge::new();

        let result = bridge.delete_policy("nonexistent-policy-id".to_string()).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), HorizonError::NotFound(_)));
    }

    #[tokio::test]
    async fn test_toggle_policy_success() {
        let bridge = SettingsBridge::new();

        let policies = bridge.get_policies().await;
        let test_policy = policies.first().expect("Should have at least one policy");
        let original_state = test_policy.enabled;

        // Toggle to opposite state
        let result = bridge
            .toggle_policy(test_policy.id.clone(), !original_state)
            .await;
        assert!(result.is_ok());

        // Verify toggle
        let updated_policies = bridge.get_policies().await;
        let updated = updated_policies
            .iter()
            .find(|p| p.id == test_policy.id)
            .expect("Policy should still exist");
        assert_eq!(updated.enabled, !original_state);
    }

    #[tokio::test]
    async fn test_set_quota_success() {
        let bridge = SettingsBridge::new();

        let new_quota = Quota {
            id: "temp".to_string(),
            name: "Test Quota".to_string(),
            resource_type: QuotaResourceType::Gpu,
            limit: 100.0,
            used: 0.0,
            unit: "hours".to_string(),
            scope: QuotaScope::Team,
            scope_id: Some("test-team".to_string()),
            reset_period: ResetPeriod::Daily,
            enabled: true,
        };

        let result = bridge.set_quota(new_quota.clone()).await;
        assert!(result.is_ok());

        let created = result.unwrap();
        assert!(created.id.starts_with("quota-"));
        assert_eq!(created.name, "Test Quota");
        assert_eq!(created.limit, 100.0);
        assert_eq!(created.used, 0.0); // Should always start at 0

        // Verify it was added
        let quotas = bridge.get_quotas().await;
        assert!(quotas.iter().any(|q| q.name == "Test Quota"));
    }

    #[tokio::test]
    async fn test_set_quota_validation_empty_name() {
        let bridge = SettingsBridge::new();

        let invalid_quota = Quota {
            id: "temp".to_string(),
            name: "".to_string(),
            resource_type: QuotaResourceType::Gpu,
            limit: 100.0,
            used: 0.0,
            unit: "hours".to_string(),
            scope: QuotaScope::Team,
            scope_id: None,
            reset_period: ResetPeriod::Daily,
            enabled: true,
        };

        let result = bridge.set_quota(invalid_quota).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), HorizonError::InvalidConfig(_)));
    }

    #[tokio::test]
    async fn test_set_quota_validation_zero_limit() {
        let bridge = SettingsBridge::new();

        let invalid_quota = Quota {
            id: "temp".to_string(),
            name: "Test Quota".to_string(),
            resource_type: QuotaResourceType::Gpu,
            limit: 0.0,
            used: 0.0,
            unit: "hours".to_string(),
            scope: QuotaScope::Team,
            scope_id: None,
            reset_period: ResetPeriod::Daily,
            enabled: true,
        };

        let result = bridge.set_quota(invalid_quota).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), HorizonError::InvalidConfig(_)));
    }

    #[tokio::test]
    async fn test_update_quota_success() {
        let bridge = SettingsBridge::new();

        let quotas = bridge.get_quotas().await;
        let test_quota = quotas.first().expect("Should have at least one quota");

        let updates = serde_json::json!({
            "name": "Updated Quota Name",
            "limit": 200.0
        });

        let result = bridge.update_quota(test_quota.id.clone(), updates).await;
        assert!(result.is_ok());

        // Verify update
        let updated_quotas = bridge.get_quotas().await;
        let updated = updated_quotas
            .iter()
            .find(|q| q.id == test_quota.id)
            .expect("Quota should still exist");
        assert_eq!(updated.name, "Updated Quota Name");
        assert_eq!(updated.limit, 200.0);
    }

    #[tokio::test]
    async fn test_update_quota_validation_zero_limit() {
        let bridge = SettingsBridge::new();

        let quotas = bridge.get_quotas().await;
        let test_quota = quotas.first().expect("Should have at least one quota");

        let updates = serde_json::json!({
            "limit": 0.0
        });

        let result = bridge.update_quota(test_quota.id.clone(), updates).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), HorizonError::InvalidConfig(_)));
    }

    #[tokio::test]
    async fn test_delete_quota_success() {
        let bridge = SettingsBridge::new();

        let quotas = bridge.get_quotas().await;
        let initial_count = quotas.len();
        let test_quota = quotas.first().expect("Should have at least one quota");

        let result = bridge.delete_quota(test_quota.id.clone()).await;
        assert!(result.is_ok());

        // Verify deletion
        let updated_quotas = bridge.get_quotas().await;
        assert_eq!(updated_quotas.len(), initial_count - 1);
        assert!(!updated_quotas.iter().any(|q| q.id == test_quota.id));
    }

    #[tokio::test]
    async fn test_update_app_settings_success() {
        let bridge = SettingsBridge::new();

        let updates = serde_json::json!({
            "notifications_enabled": false,
            "auto_refresh_interval_secs": 60,
            "theme": "light"
        });

        let result = bridge.update_app_settings(updates).await;
        assert!(result.is_ok());

        // Verify update
        let settings = bridge.get_app_settings().await;
        assert!(!settings.notifications_enabled);
        assert_eq!(settings.auto_refresh_interval_secs, 60);
        assert!(matches!(settings.theme, Theme::Light));
    }

    #[tokio::test]
    async fn test_update_app_settings_validation_invalid_interval() {
        let bridge = SettingsBridge::new();

        // Test interval = 0
        let updates = serde_json::json!({
            "auto_refresh_interval_secs": 0
        });
        let result = bridge.update_app_settings(updates).await;
        assert!(result.is_err());

        // Test interval > 3600
        let updates = serde_json::json!({
            "auto_refresh_interval_secs": 5000
        });
        let result = bridge.update_app_settings(updates).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_policy_id_uniqueness() {
        let bridge = SettingsBridge::new();

        let policy1 = Policy {
            id: "temp".to_string(),
            name: "Policy 1".to_string(),
            description: "Test".to_string(),
            policy_type: PolicyType::Rbac,
            action: PolicyAction::Allow,
            conditions: vec![],
            enabled: true,
            priority: 50,
            created_at: String::new(),
            updated_at: String::new(),
        };

        let policy2 = Policy {
            id: "temp".to_string(),
            name: "Policy 2".to_string(),
            description: "Test".to_string(),
            policy_type: PolicyType::Rbac,
            action: PolicyAction::Allow,
            conditions: vec![],
            enabled: true,
            priority: 50,
            created_at: String::new(),
            updated_at: String::new(),
        };

        let created1 = bridge.create_policy(policy1).await.unwrap();
        let created2 = bridge.create_policy(policy2).await.unwrap();

        // IDs should be unique (using UUID)
        assert_ne!(created1.id, created2.id);
    }

    #[tokio::test]
    async fn test_quota_id_uniqueness() {
        let bridge = SettingsBridge::new();

        let quota1 = Quota {
            id: "temp".to_string(),
            name: "Quota 1".to_string(),
            resource_type: QuotaResourceType::Gpu,
            limit: 100.0,
            used: 0.0,
            unit: "hours".to_string(),
            scope: QuotaScope::Team,
            scope_id: None,
            reset_period: ResetPeriod::Daily,
            enabled: true,
        };

        let quota2 = Quota {
            id: "temp".to_string(),
            name: "Quota 2".to_string(),
            resource_type: QuotaResourceType::Cpu,
            limit: 200.0,
            used: 0.0,
            unit: "cores".to_string(),
            scope: QuotaScope::Team,
            scope_id: None,
            reset_period: ResetPeriod::Daily,
            enabled: true,
        };

        let created1 = bridge.set_quota(quota1).await.unwrap();
        let created2 = bridge.set_quota(quota2).await.unwrap();

        // IDs should be unique (using UUID)
        assert_ne!(created1.id, created2.id);
    }
}
