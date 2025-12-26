//! Performance alert generation and delivery for regression detection

use crate::error::{PerformanceRegressionError, PerformanceRegressionResult};
use crate::metrics_collector::{MetricStatistics, MetricType};
use chrono::{DateTime, Duration, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{error, info, warn};
use dashmap::DashMap;

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning alert
    Warning,
    /// Critical alert
    Critical,
    /// Emergency alert requiring immediate attention
    Emergency,
}

/// Alert notification channels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NotificationChannel {
    /// Email notification
    Email { recipients: Vec<String> },
    /// Slack webhook notification
    Slack {
        webhook_url: String,
        channel: Option<String>,
    },
    /// PagerDuty integration
    PagerDuty { routing_key: String },
    /// Generic webhook
    Webhook {
        url: String,
        headers: HashMap<String, String>,
    },
    /// SMS notification
    Sms { numbers: Vec<String> },
}

/// Alert condition for triggering notifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertCondition {
    /// Unique identifier for the condition
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Metric type to monitor
    pub metric_type: MetricType,
    /// Threshold value
    pub threshold: f64,
    /// Comparison operator
    pub operator: ComparisonOperator,
    /// Number of consecutive breaches before alerting
    pub consecutive_breaches: usize,
    /// Time window for evaluation (in seconds)
    pub evaluation_window: u64,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Notification channels to use
    pub channels: Vec<NotificationChannel>,
    /// Additional tags for filtering
    pub tags: HashMap<String, String>,
}

/// Comparison operators for alert conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    Equal,
    NotEqual,
}

/// Alert instance representing a triggered alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Unique alert ID
    pub id: String,
    /// Alert condition that triggered this alert
    pub condition_id: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert title
    pub title: String,
    /// Detailed message
    pub message: String,
    /// Metric data that triggered the alert
    pub metric_data: MetricStatistics,
    /// Timestamp when alert was triggered
    pub triggered_at: DateTime<Utc>,
    /// Timestamp when alert was resolved (if applicable)
    pub resolved_at: Option<DateTime<Utc>>,
    /// Additional context
    pub context: HashMap<String, String>,
}

/// Alert manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertManagerConfig {
    /// Rate limit: maximum alerts per minute
    pub rate_limit_per_minute: usize,
    /// Deduplication window in seconds
    pub deduplication_window: u64,
    /// Alert retention period in hours
    pub retention_hours: u64,
    /// Enable alert escalation
    pub enable_escalation: bool,
    /// Escalation timeout in minutes
    pub escalation_timeout_minutes: u64,
    /// Maximum retry attempts for failed notifications
    pub max_retry_attempts: usize,
    /// Retry delay in seconds
    pub retry_delay_seconds: u64,
}

impl Default for AlertManagerConfig {
    fn default() -> Self {
        Self {
            rate_limit_per_minute: 10,
            deduplication_window: 300, // 5 minutes
            retention_hours: 168,      // 1 week
            enable_escalation: true,
            escalation_timeout_minutes: 15,
            max_retry_attempts: 3,
            retry_delay_seconds: 60,
        }
    }
}

/// Alert manager for handling performance alerts
pub struct AlertManager {
    config: AlertManagerConfig,
    conditions: Arc<DashMap<String, AlertCondition>>,
    active_alerts: Arc<DashMap<String, Alert>>,
    alert_history: Arc<RwLock<VecDeque<Alert>>>,
    breach_counters: Arc<DashMap<String, usize>>,
    rate_limiter: Arc<DashMap<String, VecDeque<DateTime<Utc>>>>,
    notification_sender: mpsc::UnboundedSender<(Alert, Vec<NotificationChannel>)>,
}

impl AlertManager {
    /// Create a new alert manager
    pub fn new(config: AlertManagerConfig) -> PerformanceRegressionResult<Self> {
        let (tx, mut rx) = mpsc::unbounded_channel();

        // Background task for sending notifications
        tokio::spawn(async move {
            while let Some((alert, channels)) = rx.recv().await {
                Self::send_notifications(alert, channels).await;
            }
        });

        Ok(Self {
            config,
            conditions: Arc::new(DashMap::new()),
            active_alerts: Arc::new(DashMap::new()),
            alert_history: Arc::new(RwLock::new(VecDeque::new())),
            breach_counters: Arc::new(DashMap::new()),
            rate_limiter: Arc::new(DashMap::new()),
            notification_sender: tx,
        })
    }

    /// Add an alert condition
    pub fn add_condition(&self, condition: AlertCondition) -> PerformanceRegressionResult<()> {
        self.conditions.insert(condition.id.clone(), condition);
        Ok(())
    }

    /// Remove an alert condition
    pub fn remove_condition(&self, condition_id: &str) -> PerformanceRegressionResult<()> {
        self.conditions.remove(condition_id);
        Ok(())
    }

    /// Get all alert conditions
    pub fn get_conditions(&self) -> Vec<AlertCondition> {
        self.conditions.iter().map(|entry| entry.value().clone()).collect()
    }

    /// Evaluate alert conditions against metric statistics
    pub async fn evaluate_alert_conditions(
        &self,
        stats: &MetricStatistics,
    ) -> PerformanceRegressionResult<()> {
        let conditions: Vec<AlertCondition> = self.conditions
            .iter()
            .filter(|entry| entry.value().metric_type == stats.metric_type)
            .map(|entry| entry.value().clone())
            .collect();

        for condition in conditions {
            self.evaluate_single_condition(&condition, stats).await?;
        }

        Ok(())
    }

    /// Evaluate a single alert condition
    async fn evaluate_single_condition(
        &self,
        condition: &AlertCondition,
        stats: &MetricStatistics,
    ) -> PerformanceRegressionResult<()> {
        let breach = self.check_threshold_breach(condition, stats.average.0)?;

        if breach {
            self.handle_threshold_breach(condition, stats).await?;
        } else {
            self.reset_breach_counter(&condition.id);
        }

        Ok(())
    }

    /// Check if threshold is breached
    fn check_threshold_breach(
        &self,
        condition: &AlertCondition,
        value: f64,
    ) -> PerformanceRegressionResult<bool> {
        let breach = match condition.operator {
            ComparisonOperator::GreaterThan => value > condition.threshold,
            ComparisonOperator::GreaterThanOrEqual => value >= condition.threshold,
            ComparisonOperator::LessThan => value < condition.threshold,
            ComparisonOperator::LessThanOrEqual => value <= condition.threshold,
            ComparisonOperator::Equal => (value - condition.threshold).abs() < f64::EPSILON,
            ComparisonOperator::NotEqual => (value - condition.threshold).abs() >= f64::EPSILON,
        };

        Ok(breach)
    }

    /// Handle threshold breach
    async fn handle_threshold_breach(
        &self,
        condition: &AlertCondition,
        stats: &MetricStatistics,
    ) -> PerformanceRegressionResult<()> {
        let should_alert = {
            let mut count = self.breach_counters
                .entry(condition.id.clone())
                .or_insert(0);
            *count += 1;
            *count >= condition.consecutive_breaches
        };

        if should_alert {
            self.create_and_send_alert(condition, stats).await?;
        }

        Ok(())
    }

    /// Reset breach counter for a condition
    fn reset_breach_counter(&self, condition_id: &str) {
        self.breach_counters.remove(condition_id);
    }

    /// Create and send an alert
    async fn create_and_send_alert(
        &self,
        condition: &AlertCondition,
        stats: &MetricStatistics,
    ) -> PerformanceRegressionResult<()> {
        // Check for duplicates
        if self.is_duplicate_alert(condition, stats)? {
            info!("Skipping duplicate alert for condition: {}", condition.id);
            return Ok(());
        }

        // Check rate limit
        if !self.check_rate_limit(&condition.id)? {
            warn!("Rate limit exceeded for condition: {}", condition.id);
            return Ok(());
        }

        let alert = Alert {
            id: uuid::Uuid::new_v4().to_string(),
            condition_id: condition.id.clone(),
            severity: condition.severity,
            title: format!("{} - {} Alert", condition.name, condition.severity_string()),
            message: self.format_alert_message(condition, stats),
            metric_data: stats.clone(),
            triggered_at: Utc::now(),
            resolved_at: None,
            context: condition.tags.clone(),
        };

        // Store active alert
        self.store_active_alert(alert.clone())?;

        // Send notifications
        self.send_alert(alert, condition.channels.clone()).await?;

        // Reset breach counter after successful alert
        self.reset_breach_counter(&condition.id);

        Ok(())
    }

    /// Check if alert is a duplicate
    fn is_duplicate_alert(
        &self,
        condition: &AlertCondition,
        _stats: &MetricStatistics,
    ) -> PerformanceRegressionResult<bool> {
        let dedup_window = Duration::seconds(self.config.deduplication_window as i64);
        let now = Utc::now();

        for entry in self.active_alerts.iter() {
            let alert = entry.value();
            if alert.condition_id == condition.id
                && alert.resolved_at.is_none()
                && now - alert.triggered_at < dedup_window
            {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Check rate limit for alerts
    fn check_rate_limit(&self, condition_id: &str) -> PerformanceRegressionResult<bool> {
        let now = Utc::now();
        let one_minute_ago = now - Duration::seconds(60);

        let mut timestamps = self.rate_limiter
            .entry(condition_id.to_string())
            .or_insert_with(VecDeque::new);

        // Remove old timestamps
        timestamps.retain(|ts| *ts > one_minute_ago);

        if timestamps.len() >= self.config.rate_limit_per_minute {
            return Ok(false);
        }

        timestamps.push_back(now);
        Ok(true)
    }

    /// Store active alert
    fn store_active_alert(&self, alert: Alert) -> PerformanceRegressionResult<()> {
        self.active_alerts.insert(alert.id.clone(), alert.clone());

        let mut alert_history = self.alert_history.write();
        alert_history.push_back(alert);

        // Maintain history size
        let retention_cutoff = Utc::now() - Duration::hours(self.config.retention_hours as i64);
        alert_history.retain(|a| a.triggered_at > retention_cutoff);

        Ok(())
    }

    /// Send alert through configured channels
    pub async fn send_alert(
        &self,
        alert: Alert,
        channels: Vec<NotificationChannel>,
    ) -> PerformanceRegressionResult<()> {
        self.notification_sender
            .send((alert, channels))
            .map_err(|e| PerformanceRegressionError::AlertDeliveryFailed {
                channel: "internal".to_string(),
                details: e.to_string(),
            })?;
        Ok(())
    }

    /// Send notifications through various channels
    async fn send_notifications(alert: Alert, channels: Vec<NotificationChannel>) {
        for channel in channels {
            match Self::send_to_channel(&alert, &channel).await {
                Ok(_) => info!("Alert sent successfully via {:?}", channel),
                Err(e) => error!("Failed to send alert via {:?}: {}", channel, e),
            }
        }
    }

    /// Send alert to specific channel
    async fn send_to_channel(
        alert: &Alert,
        channel: &NotificationChannel,
    ) -> PerformanceRegressionResult<()> {
        match channel {
            NotificationChannel::Email { recipients } => {
                Self::send_email_notification(alert, recipients).await
            }
            NotificationChannel::Slack {
                webhook_url,
                channel,
            } => Self::send_slack_notification(alert, webhook_url, channel.as_deref()).await,
            NotificationChannel::PagerDuty { routing_key } => {
                Self::send_pagerduty_notification(alert, routing_key).await
            }
            NotificationChannel::Webhook { url, headers } => {
                Self::send_webhook_notification(alert, url, headers).await
            }
            NotificationChannel::Sms { numbers } => {
                Self::send_sms_notification(alert, numbers).await
            }
        }
    }

    /// Send email notification
    async fn send_email_notification(
        alert: &Alert,
        recipients: &[String],
    ) -> PerformanceRegressionResult<()> {
        // Simulate email sending
        info!("Sending email alert to {:?}: {}", recipients, alert.title);
        Ok(())
    }

    /// Send Slack notification
    async fn send_slack_notification(
        alert: &Alert,
        webhook_url: &str,
        channel: Option<&str>,
    ) -> PerformanceRegressionResult<()> {
        // Simulate Slack webhook
        info!("Sending Slack alert to {}: {}", webhook_url, alert.title);
        if let Some(ch) = channel {
            info!("Target channel: {}", ch);
        }
        Ok(())
    }

    /// Send PagerDuty notification
    async fn send_pagerduty_notification(
        alert: &Alert,
        routing_key: &str,
    ) -> PerformanceRegressionResult<()> {
        // Simulate PagerDuty API call
        info!(
            "Sending PagerDuty alert with key {}: {}",
            routing_key, alert.title
        );
        Ok(())
    }

    /// Send webhook notification
    async fn send_webhook_notification(
        alert: &Alert,
        url: &str,
        headers: &HashMap<String, String>,
    ) -> PerformanceRegressionResult<()> {
        // Simulate webhook call
        info!("Sending webhook alert to {}: {}", url, alert.title);
        info!("Headers: {:?}", headers);
        Ok(())
    }

    /// Send SMS notification
    async fn send_sms_notification(
        alert: &Alert,
        numbers: &[String],
    ) -> PerformanceRegressionResult<()> {
        // Simulate SMS sending
        info!("Sending SMS alert to {:?}: {}", numbers, alert.title);
        Ok(())
    }

    /// Format alert message
    fn format_alert_message(&self, condition: &AlertCondition, stats: &MetricStatistics) -> String {
        format!(
            "Alert condition '{}' triggered for metric {:?}. \
             Current value: {:.2} {} threshold: {:.2}. \
             Statistics - Min: {:.2}, Max: {:.2}, Avg: {:.2}, P95: {:.2}, P99: {:.2}",
            condition.name,
            condition.metric_type,
            stats.average.0,
            match condition.operator {
                ComparisonOperator::GreaterThan | ComparisonOperator::GreaterThanOrEqual =>
                    "exceeds",
                ComparisonOperator::LessThan | ComparisonOperator::LessThanOrEqual => "below",
                ComparisonOperator::Equal => "equals",
                ComparisonOperator::NotEqual => "differs from",
            },
            condition.threshold,
            stats.minimum.0,
            stats.maximum.0,
            stats.average.0,
            stats.p95.0,
            stats.p99.0
        )
    }

    /// Manage alert escalation
    pub async fn manage_alert_escalation(&self) -> PerformanceRegressionResult<()> {
        if !self.config.enable_escalation {
            return Ok(());
        }

        let escalation_timeout = Duration::minutes(self.config.escalation_timeout_minutes as i64);
        let now = Utc::now();

        // Collect alerts to escalate (to avoid holding lock during async operations)
        let alerts_to_escalate: Vec<_> = self.active_alerts.iter()
            .filter(|entry| {
                let alert = entry.value();
                alert.resolved_at.is_none() && now - alert.triggered_at > escalation_timeout
            })
            .map(|entry| entry.value().clone())
            .collect();

        for alert in alerts_to_escalate {
            self.escalate_alert(&alert).await?;
        }

        Ok(())
    }

    /// Escalate an alert
    async fn escalate_alert(&self, alert: &Alert) -> PerformanceRegressionResult<()> {
        info!("Escalating alert: {}", alert.id);

        // Find the original condition to get escalation channels
        if let Some(condition) = self.conditions.get(&alert.condition_id) {
            // For now, re-send to all channels (in production, would have separate escalation channels)
            self.send_alert(alert.clone(), condition.channels.clone())
                .await?;
        }

        Ok(())
    }

    /// Deduplicate alerts to prevent spam
    pub fn deduplicate_alerts(&self) -> PerformanceRegressionResult<Vec<Alert>> {
        let mut unique_alerts = HashMap::new();

        for entry in self.active_alerts.iter() {
            let alert = entry.value();
            let key = format!("{}-{}", alert.condition_id, alert.severity as u8);
            unique_alerts.insert(key, alert.clone());
        }

        Ok(unique_alerts.into_values().collect())
    }

    /// Track alert history
    pub fn track_alert_history(&self) -> Vec<Alert> {
        self.alert_history.read().iter().cloned().collect()
    }

    /// Configure notification channels
    pub fn configure_notification_channels(
        &self,
        condition_id: &str,
        channels: Vec<NotificationChannel>,
    ) -> PerformanceRegressionResult<()> {
        if let Some(mut condition) = self.conditions.get_mut(condition_id) {
            condition.channels = channels;
            Ok(())
        } else {
            Err(PerformanceRegressionError::AlertConditionNotFound {
                condition_id: condition_id.to_string(),
            })
        }
    }

    /// Resolve an active alert
    pub fn resolve_alert(&self, alert_id: &str) -> PerformanceRegressionResult<()> {
        if let Some(mut alert) = self.active_alerts.get_mut(alert_id) {
            alert.resolved_at = Some(Utc::now());
            Ok(())
        } else {
            Err(PerformanceRegressionError::AlertNotFound {
                alert_id: alert_id.to_string(),
            })
        }
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> Vec<Alert> {
        self.active_alerts.iter().map(|entry| entry.value().clone()).collect()
    }

    /// Get alert by ID
    pub fn get_alert(&self, alert_id: &str) -> Option<Alert> {
        self.active_alerts.get(alert_id).map(|r| r.clone())
    }

    /// Clear resolved alerts
    pub fn clear_resolved_alerts(&self) -> PerformanceRegressionResult<()> {
        self.active_alerts.retain(|_, alert| alert.resolved_at.is_none());
        Ok(())
    }
}

/// Extension trait for AlertCondition
impl AlertCondition {
    fn severity_string(&self) -> &'static str {
        match self.severity {
            AlertSeverity::Info => "Info",
            AlertSeverity::Warning => "Warning",
            AlertSeverity::Critical => "Critical",
            AlertSeverity::Emergency => "Emergency",
        }
    }
}

use uuid;

#[cfg(test)]
mod tests {
    use super::*;
    use ordered_float::OrderedFloat;

    fn create_test_config() -> AlertManagerConfig {
        AlertManagerConfig {
            rate_limit_per_minute: 5,
            deduplication_window: 60,
            retention_hours: 24,
            enable_escalation: true,
            escalation_timeout_minutes: 10,
            max_retry_attempts: 2,
            retry_delay_seconds: 30,
        }
    }

    fn create_test_condition(id: &str, metric_type: MetricType) -> AlertCondition {
        AlertCondition {
            id: id.to_string(),
            name: format!("Test condition {}", id),
            metric_type,
            threshold: 80.0,
            operator: ComparisonOperator::GreaterThan,
            consecutive_breaches: 2,
            evaluation_window: 300,
            severity: AlertSeverity::Warning,
            channels: vec![NotificationChannel::Email {
                recipients: vec!["test@example.com".to_string()],
            }],
            tags: HashMap::new(),
        }
    }

    fn create_test_stats(metric_type: MetricType, value: f64) -> MetricStatistics {
        MetricStatistics {
            metric_type,
            average: OrderedFloat(value),
            minimum: OrderedFloat(value - 10.0),
            maximum: OrderedFloat(value + 10.0),
            std_deviation: OrderedFloat(5.0),
            p95: OrderedFloat(value + 8.0),
            p99: OrderedFloat(value + 9.0),
            sample_count: 100,
            window_start: Utc::now() - Duration::minutes(5),
            window_end: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_alert_manager_creation() {
        let config = create_test_config();
        let manager = AlertManager::new(config);
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_add_and_remove_condition() {
        let manager = AlertManager::new(create_test_config()).unwrap();
        let condition = create_test_condition("test1", MetricType::CpuUsage);

        // Add condition
        assert!(manager.add_condition(condition.clone()).is_ok());
        assert_eq!(manager.get_conditions().len(), 1);

        // Remove condition
        assert!(manager.remove_condition("test1").is_ok());
        assert_eq!(manager.get_conditions().len(), 0);
    }

    #[tokio::test]
    async fn test_threshold_breach_detection() {
        let manager = AlertManager::new(create_test_config()).unwrap();
        let condition = create_test_condition("test1", MetricType::CpuUsage);

        // Test various operators
        assert!(manager.check_threshold_breach(&condition, 85.0).unwrap()); // 85 > 80

        let mut condition_lt = condition.clone();
        condition_lt.operator = ComparisonOperator::LessThan;
        assert!(manager.check_threshold_breach(&condition_lt, 75.0).unwrap()); // 75 < 80

        let mut condition_eq = condition.clone();
        condition_eq.operator = ComparisonOperator::Equal;
        assert!(manager.check_threshold_breach(&condition_eq, 80.0).unwrap()); // 80 == 80
    }

    #[tokio::test]
    async fn test_consecutive_breach_requirement() {
        let manager = AlertManager::new(create_test_config()).unwrap();
        let condition = create_test_condition("test1", MetricType::CpuUsage);
        manager.add_condition(condition.clone()).unwrap();

        let stats = create_test_stats(MetricType::CpuUsage, 85.0);

        // First breach - no alert
        manager.evaluate_alert_conditions(&stats).await.unwrap();
        assert_eq!(manager.get_active_alerts().len(), 0);

        // Second breach - should trigger alert
        manager.evaluate_alert_conditions(&stats).await.unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        assert_eq!(manager.get_active_alerts().len(), 1);
    }

    #[tokio::test]
    async fn test_alert_deduplication() {
        let manager = AlertManager::new(create_test_config()).unwrap();
        let condition = create_test_condition("test1", MetricType::CpuUsage);
        manager.add_condition(condition.clone()).unwrap();

        let stats = create_test_stats(MetricType::CpuUsage, 85.0);

        // Trigger alert twice
        manager.evaluate_alert_conditions(&stats).await.unwrap();
        manager.evaluate_alert_conditions(&stats).await.unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        // Second alert should be deduplicated
        assert_eq!(manager.get_active_alerts().len(), 1);
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let mut config = create_test_config();
        config.rate_limit_per_minute = 2;
        let manager = AlertManager::new(config).unwrap();

        let condition = create_test_condition("test1", MetricType::CpuUsage);
        manager.add_condition(condition.clone()).unwrap();

        // Clear any existing alerts to ensure clean state
        manager.clear_resolved_alerts().unwrap();

        // Trigger multiple alerts rapidly
        for i in 0..5 {
            let stats = create_test_stats(MetricType::CpuUsage, 85.0 + i as f64);
            manager.evaluate_alert_conditions(&stats).await.unwrap();
            manager.evaluate_alert_conditions(&stats).await.unwrap(); // Trigger consecutive breach

            // Resolve the alert to allow new ones
            if let Some(alert) = manager.get_active_alerts().first() {
                manager.resolve_alert(&alert.id).unwrap();
            }
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Should have limited alerts due to rate limiting
        let history = manager.track_alert_history();
        assert!(history.len() <= 2);
    }

    #[tokio::test]
    async fn test_multiple_notification_channels() {
        let manager = AlertManager::new(create_test_config()).unwrap();
        let mut condition = create_test_condition("test1", MetricType::ResponseTime);

        condition.channels = vec![
            NotificationChannel::Email {
                recipients: vec!["alert@example.com".to_string()],
            },
            NotificationChannel::Slack {
                webhook_url: "https://hooks.slack.com/test".to_string(),
                channel: Some("#alerts".to_string()),
            },
            NotificationChannel::PagerDuty {
                routing_key: "test-key".to_string(),
            },
        ];

        manager.add_condition(condition).unwrap();

        let stats = create_test_stats(MetricType::ResponseTime, 100.0);
        manager.evaluate_alert_conditions(&stats).await.unwrap();
        manager.evaluate_alert_conditions(&stats).await.unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        assert_eq!(manager.get_active_alerts().len(), 1);
    }

    #[tokio::test]
    async fn test_alert_severity_levels() {
        let manager = AlertManager::new(create_test_config()).unwrap();

        let severities = vec![
            AlertSeverity::Info,
            AlertSeverity::Warning,
            AlertSeverity::Critical,
            AlertSeverity::Emergency,
        ];

        for (i, severity) in severities.iter().enumerate() {
            let mut condition = create_test_condition(&format!("test{}", i), MetricType::ErrorRate);
            condition.severity = *severity;
            manager.add_condition(condition).unwrap();
        }

        assert_eq!(manager.get_conditions().len(), 4);

        // Verify severity ordering
        assert!(AlertSeverity::Info < AlertSeverity::Warning);
        assert!(AlertSeverity::Warning < AlertSeverity::Critical);
        assert!(AlertSeverity::Critical < AlertSeverity::Emergency);
    }

    #[tokio::test]
    async fn test_alert_resolution() {
        let manager = AlertManager::new(create_test_config()).unwrap();
        let condition = create_test_condition("test1", MetricType::CpuUsage);
        manager.add_condition(condition).unwrap();

        let stats = create_test_stats(MetricType::CpuUsage, 85.0);
        manager.evaluate_alert_conditions(&stats).await.unwrap();
        manager.evaluate_alert_conditions(&stats).await.unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let alert = manager.get_active_alerts().first().unwrap().clone();
        assert!(alert.resolved_at.is_none());

        // Resolve alert
        manager.resolve_alert(&alert.id).unwrap();

        let resolved_alert = manager.get_alert(&alert.id).unwrap();
        assert!(resolved_alert.resolved_at.is_some());
    }

    #[tokio::test]
    async fn test_alert_history_tracking() {
        let mut config = create_test_config();
        config.deduplication_window = 1; // Short deduplication window
        let manager = AlertManager::new(config).unwrap();
        let condition = create_test_condition("test1", MetricType::MemoryUsage);
        manager.add_condition(condition).unwrap();

        // Trigger multiple alerts with sleep between to avoid deduplication
        for i in 0..3 {
            // Wait longer than deduplication window before each alert
            if i > 0 {
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
            }

            let stats = create_test_stats(MetricType::MemoryUsage, 85.0 + i as f64 * 10.0);
            manager.evaluate_alert_conditions(&stats).await.unwrap();
            manager.evaluate_alert_conditions(&stats).await.unwrap();

            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        let history = manager.track_alert_history();
        assert!(history.len() >= 3);
    }

    #[tokio::test]
    async fn test_metric_type_filtering() {
        let manager = AlertManager::new(create_test_config()).unwrap();

        // Add conditions for different metric types
        manager
            .add_condition(create_test_condition("cpu", MetricType::CpuUsage))
            .unwrap();
        manager
            .add_condition(create_test_condition("memory", MetricType::MemoryUsage))
            .unwrap();

        // Evaluate only CPU metrics
        let cpu_stats = create_test_stats(MetricType::CpuUsage, 85.0);
        manager.evaluate_alert_conditions(&cpu_stats).await.unwrap();
        manager.evaluate_alert_conditions(&cpu_stats).await.unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        // Should only have CPU alert
        let alerts = manager.get_active_alerts();
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].condition_id, "cpu");
    }

    #[tokio::test]
    async fn test_custom_metric_alerts() {
        let manager = AlertManager::new(create_test_config()).unwrap();
        let mut condition =
            create_test_condition("custom1", MetricType::Custom("api_latency".to_string()));
        condition.threshold = 500.0;
        manager.add_condition(condition).unwrap();

        let stats = create_test_stats(MetricType::Custom("api_latency".to_string()), 600.0);
        manager.evaluate_alert_conditions(&stats).await.unwrap();
        manager.evaluate_alert_conditions(&stats).await.unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        assert_eq!(manager.get_active_alerts().len(), 1);
    }

    #[tokio::test]
    async fn test_alert_context_and_tags() {
        let manager = AlertManager::new(create_test_config()).unwrap();
        let mut condition = create_test_condition("test1", MetricType::ResponseTime);
        condition
            .tags
            .insert("service".to_string(), "api".to_string());
        condition
            .tags
            .insert("environment".to_string(), "production".to_string());
        manager.add_condition(condition).unwrap();

        let stats = create_test_stats(MetricType::ResponseTime, 85.0);
        manager.evaluate_alert_conditions(&stats).await.unwrap();
        manager.evaluate_alert_conditions(&stats).await.unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let alerts = manager.get_active_alerts();
        let alert = alerts.first().unwrap();
        assert_eq!(alert.context.get("service").unwrap(), "api");
        assert_eq!(alert.context.get("environment").unwrap(), "production");
    }

    #[tokio::test]
    async fn test_webhook_notification_with_headers() {
        let manager = AlertManager::new(create_test_config()).unwrap();
        let mut condition = create_test_condition("test1", MetricType::ErrorRate);
        condition.threshold = 5.0; // Set threshold lower than test value

        let mut headers = HashMap::new();
        headers.insert("Authorization".to_string(), "Bearer token123".to_string());
        headers.insert("Content-Type".to_string(), "application/json".to_string());

        condition.channels = vec![NotificationChannel::Webhook {
            url: "https://example.com/webhook".to_string(),
            headers,
        }];

        manager.add_condition(condition).unwrap();

        let stats = create_test_stats(MetricType::ErrorRate, 10.0);
        manager.evaluate_alert_conditions(&stats).await.unwrap();
        manager.evaluate_alert_conditions(&stats).await.unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        assert_eq!(manager.get_active_alerts().len(), 1);
    }

    #[tokio::test]
    async fn test_sms_notification() {
        let manager = AlertManager::new(create_test_config()).unwrap();
        let mut condition = create_test_condition("test1", MetricType::DiskIops);

        condition.channels = vec![NotificationChannel::Sms {
            numbers: vec!["+1234567890".to_string(), "+0987654321".to_string()],
        }];

        manager.add_condition(condition).unwrap();

        let stats = create_test_stats(MetricType::DiskIops, 1000.0);
        manager.evaluate_alert_conditions(&stats).await.unwrap();
        manager.evaluate_alert_conditions(&stats).await.unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        assert_eq!(manager.get_active_alerts().len(), 1);
    }

    #[tokio::test]
    async fn test_clear_resolved_alerts() {
        let manager = AlertManager::new(create_test_config()).unwrap();
        let condition = create_test_condition("test1", MetricType::NetworkBandwidth);
        manager.add_condition(condition).unwrap();

        // Create multiple alerts
        for i in 0..3 {
            let stats = create_test_stats(MetricType::NetworkBandwidth, 85.0 + i as f64);
            manager.evaluate_alert_conditions(&stats).await.unwrap();
            manager.evaluate_alert_conditions(&stats).await.unwrap();
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        let alerts = manager.get_active_alerts();
        assert!(alerts.len() >= 1);

        // Resolve first alert
        manager.resolve_alert(&alerts[0].id).unwrap();

        // Clear resolved alerts
        manager.clear_resolved_alerts().unwrap();

        // Should have fewer active alerts
        let remaining_alerts = manager.get_active_alerts();
        assert!(remaining_alerts.len() < alerts.len());
        assert!(remaining_alerts.iter().all(|a| a.resolved_at.is_none()));
    }

    #[tokio::test]
    async fn test_configure_notification_channels() {
        let manager = AlertManager::new(create_test_config()).unwrap();
        let condition = create_test_condition("test1", MetricType::CpuUsage);
        manager.add_condition(condition).unwrap();

        let new_channels = vec![NotificationChannel::Slack {
            webhook_url: "https://new-webhook.slack.com".to_string(),
            channel: Some("#new-channel".to_string()),
        }];

        manager
            .configure_notification_channels("test1", new_channels.clone())
            .unwrap();

        let updated_condition = manager
            .get_conditions()
            .into_iter()
            .find(|c| c.id == "test1")
            .unwrap();

        assert_eq!(updated_condition.channels, new_channels);
    }

    #[tokio::test]
    async fn test_alert_message_formatting() {
        let manager = AlertManager::new(create_test_config()).unwrap();
        let condition = create_test_condition("test1", MetricType::ResponseTime);
        let stats = create_test_stats(MetricType::ResponseTime, 150.0);

        let message = manager.format_alert_message(&condition, &stats);
        assert!(message.contains("Test condition test1"));
        assert!(message.contains("150.00"));
        assert!(message.contains("exceeds"));
        assert!(message.contains("80.00"));
    }

    #[tokio::test]
    async fn test_comparison_operators() {
        let manager = AlertManager::new(create_test_config()).unwrap();

        let operators = vec![
            (ComparisonOperator::GreaterThan, 85.0, true),
            (ComparisonOperator::GreaterThan, 75.0, false),
            (ComparisonOperator::LessThan, 75.0, true),
            (ComparisonOperator::LessThan, 85.0, false),
            (ComparisonOperator::GreaterThanOrEqual, 80.0, true),
            (ComparisonOperator::LessThanOrEqual, 80.0, true),
            (ComparisonOperator::Equal, 80.0, true),
            (ComparisonOperator::NotEqual, 85.0, true),
        ];

        for (operator, value, expected) in operators {
            let mut condition = create_test_condition("test", MetricType::CpuUsage);
            condition.operator = operator;
            let result = manager.check_threshold_breach(&condition, value).unwrap();
            assert_eq!(
                result, expected,
                "Failed for {:?} with value {}",
                operator, value
            );
        }
    }

    #[tokio::test]
    async fn test_evaluation_window() {
        let manager = AlertManager::new(create_test_config()).unwrap();
        let mut condition = create_test_condition("test1", MetricType::Throughput);
        condition.evaluation_window = 60; // 1 minute window
        manager.add_condition(condition).unwrap();

        let stats = create_test_stats(MetricType::Throughput, 85.0);
        manager.evaluate_alert_conditions(&stats).await.unwrap();
        manager.evaluate_alert_conditions(&stats).await.unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        assert_eq!(manager.get_active_alerts().len(), 1);
    }

    #[tokio::test]
    async fn test_alert_escalation() {
        let mut config = create_test_config();
        config.escalation_timeout_minutes = 0; // Immediate escalation for testing
        let manager = AlertManager::new(config).unwrap();

        let condition = create_test_condition("test1", MetricType::CpuUsage);
        manager.add_condition(condition).unwrap();

        let stats = create_test_stats(MetricType::CpuUsage, 85.0);
        manager.evaluate_alert_conditions(&stats).await.unwrap();
        manager.evaluate_alert_conditions(&stats).await.unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        // Trigger escalation
        manager.manage_alert_escalation().await.unwrap();

        // Alert should still be active after escalation
        assert_eq!(manager.get_active_alerts().len(), 1);
    }

    #[tokio::test]
    async fn test_alert_retention() {
        let mut config = create_test_config();
        config.retention_hours = 1; // 1 hour retention
        let manager = AlertManager::new(config).unwrap();

        let condition = create_test_condition("test1", MetricType::MemoryUsage);
        manager.add_condition(condition).unwrap();

        let stats = create_test_stats(MetricType::MemoryUsage, 85.0);
        manager.evaluate_alert_conditions(&stats).await.unwrap();
        manager.evaluate_alert_conditions(&stats).await.unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // History should be maintained
        let history = manager.track_alert_history();
        assert!(history.len() >= 1);

        // Alerts within retention period should be kept
        let active_alerts = manager.get_active_alerts();
        assert!(active_alerts.len() >= 1);
    }

    #[tokio::test]
    async fn test_breach_counter_reset() {
        let manager = AlertManager::new(create_test_config()).unwrap();
        let condition = create_test_condition("test1", MetricType::ErrorRate);
        manager.add_condition(condition).unwrap();

        // First breach
        let high_stats = create_test_stats(MetricType::ErrorRate, 85.0);
        manager
            .evaluate_alert_conditions(&high_stats)
            .await
            .unwrap();

        // Normal value - should reset counter
        let normal_stats = create_test_stats(MetricType::ErrorRate, 50.0);
        manager
            .evaluate_alert_conditions(&normal_stats)
            .await
            .unwrap();

        // Another breach - should start counting from 1 again
        manager
            .evaluate_alert_conditions(&high_stats)
            .await
            .unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        // Should not have triggered alert yet (needs 2 consecutive)
        assert_eq!(manager.get_active_alerts().len(), 0);
    }

    #[tokio::test]
    async fn test_alert_deduplicate_method() {
        let manager = AlertManager::new(create_test_config()).unwrap();

        // Create multiple conditions with same severity
        for i in 0..3 {
            let mut condition = create_test_condition(&format!("test{}", i), MetricType::CpuUsage);
            condition.severity = AlertSeverity::Critical;
            manager.add_condition(condition).unwrap();
        }

        // Trigger alerts for all conditions
        let stats = create_test_stats(MetricType::CpuUsage, 85.0);
        for _ in 0..2 {
            manager.evaluate_alert_conditions(&stats).await.unwrap();
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let deduplicated = manager.deduplicate_alerts().unwrap();
        assert!(deduplicated.len() <= 3);
    }

    #[tokio::test]
    async fn test_concurrent_alert_evaluation() {
        let manager = Arc::new(AlertManager::new(create_test_config()).unwrap());

        // Add multiple conditions
        for i in 0..5 {
            let condition = create_test_condition(
                &format!("test{}", i),
                MetricType::Custom(format!("metric{}", i)),
            );
            manager.add_condition(condition).unwrap();
        }

        // Evaluate conditions concurrently
        let mut handles = vec![];
        for i in 0..5 {
            let manager_clone = manager.clone();
            let handle = tokio::spawn(async move {
                let stats = create_test_stats(MetricType::Custom(format!("metric{}", i)), 85.0);
                for _ in 0..2 {
                    manager_clone
                        .evaluate_alert_conditions(&stats)
                        .await
                        .unwrap();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        // Should have alerts for all conditions
        assert_eq!(manager.get_active_alerts().len(), 5);
    }

    #[tokio::test]
    async fn test_alert_condition_update() {
        let manager = AlertManager::new(create_test_config()).unwrap();
        let mut condition = create_test_condition("test1", MetricType::ResponseTime);
        condition.threshold = 100.0;
        manager.add_condition(condition.clone()).unwrap();

        // Test with value below threshold
        let stats = create_test_stats(MetricType::ResponseTime, 50.0);
        manager.evaluate_alert_conditions(&stats).await.unwrap();
        manager.evaluate_alert_conditions(&stats).await.unwrap();
        assert_eq!(manager.get_active_alerts().len(), 0);

        // Update threshold
        condition.threshold = 40.0;
        manager.add_condition(condition).unwrap(); // This replaces the existing condition

        // Now the same value should trigger an alert
        manager.evaluate_alert_conditions(&stats).await.unwrap();
        manager.evaluate_alert_conditions(&stats).await.unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        assert_eq!(manager.get_active_alerts().len(), 1);
    }
}
