//! Monitoring-related types for AI Assistant Zero-Config Integration
//! Split from types.rs to keep files under 750 lines

use std::collections::HashMap;
use std::time::Duration;

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfiguration {
    pub metrics_collection: MetricsConfiguration,
    pub logging_configuration: LoggingConfiguration,
    pub alerting_configuration: AlertingConfiguration,
    pub tracing_configuration: TracingConfiguration,
    pub dashboard_configuration: DashboardConfiguration,
}

/// Metrics collection configuration
#[derive(Debug, Clone)]
pub struct MetricsConfiguration {
    pub enabled: bool,
    pub collection_interval: Duration,
    pub retention_period: Duration,
    pub custom_metrics: Vec<CustomMetric>,
    pub exporters: Vec<MetricsExporter>,
}

/// Custom metrics definition
#[derive(Debug, Clone)]
pub struct CustomMetric {
    pub name: String,
    pub metric_type: MetricType,
    pub description: String,
    pub labels: Vec<String>,
    pub collection_method: CollectionMethod,
}

/// Metric types
#[derive(Debug, Clone, PartialEq)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
}

/// Collection methods
#[derive(Debug, Clone, PartialEq)]
pub enum CollectionMethod {
    Pull,
    Push,
    Streaming,
}

/// Metrics exporters
#[derive(Debug, Clone, PartialEq)]
pub enum MetricsExporter {
    Prometheus,
    InfluxDB,
    CloudWatch,
    DataDog,
    NewRelic,
    Grafana,
}

/// Logging configuration
#[derive(Debug, Clone)]
pub struct LoggingConfiguration {
    pub log_level: super::LogLevel,
    pub log_format: LogFormat,
    pub structured_logging: bool,
    pub log_aggregation: LogAggregationConfiguration,
    pub log_retention: LogRetentionConfiguration,
}

/// Log formats
#[derive(Debug, Clone, PartialEq)]
pub enum LogFormat {
    JSON,
    Plain,
    Structured,
    Custom,
}

/// Log aggregation configuration
#[derive(Debug, Clone)]
pub struct LogAggregationConfiguration {
    pub enabled: bool,
    pub aggregation_service: LogAggregationService,
    pub shipping_interval: Duration,
    pub buffer_size: u32,
}

/// Log aggregation services
#[derive(Debug, Clone, PartialEq)]
pub enum LogAggregationService {
    ElasticSearch,
    Splunk,
    CloudWatch,
    DataDog,
    Fluentd,
    Logstash,
}

/// Log retention configuration
#[derive(Debug, Clone)]
pub struct LogRetentionConfiguration {
    pub retention_period: Duration,
    pub archival_enabled: bool,
    pub archival_storage: Option<String>,
    pub compression_enabled: bool,
}

/// Alerting configuration
#[derive(Debug, Clone)]
pub struct AlertingConfiguration {
    pub enabled: bool,
    pub alert_rules: Vec<AlertRule>,
    pub notification_channels: Vec<NotificationChannel>,
    pub escalation_policies: Vec<EscalationPolicy>,
}

/// Alert rules
#[derive(Debug, Clone)]
pub struct AlertRule {
    pub name: String,
    pub condition: AlertCondition,
    pub severity: AlertSeverity,
    pub notification_channels: Vec<String>,
    pub cooldown_period: Duration,
}

/// Alert conditions
#[derive(Debug, Clone)]
pub struct AlertCondition {
    pub metric_name: String,
    pub operator: ComparisonOperator,
    pub threshold: f64,
    pub evaluation_period: Duration,
    pub evaluation_frequency: Duration,
}

/// Comparison operators
#[derive(Debug, Clone, PartialEq)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Equal,
    NotEqual,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum AlertSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Notification channels
#[derive(Debug, Clone)]
pub struct NotificationChannel {
    pub name: String,
    pub channel_type: NotificationChannelType,
    pub configuration: HashMap<String, String>,
    pub enabled: bool,
}

/// Notification channel types
#[derive(Debug, Clone, PartialEq)]
pub enum NotificationChannelType {
    Email,
    Slack,
    PagerDuty,
    Webhook,
    SMS,
    Phone,
}

/// Escalation policies
#[derive(Debug, Clone)]
pub struct EscalationPolicy {
    pub name: String,
    pub escalation_rules: Vec<EscalationRule>,
}

/// Escalation rules
#[derive(Debug, Clone)]
pub struct EscalationRule {
    pub delay: Duration,
    pub notification_channels: Vec<String>,
    pub auto_resolve: bool,
}

/// Tracing configuration
#[derive(Debug, Clone)]
pub struct TracingConfiguration {
    pub enabled: bool,
    pub sampling_rate: f64,
    pub trace_exporters: Vec<TraceExporter>,
    pub custom_attributes: HashMap<String, String>,
}

/// Trace exporters
#[derive(Debug, Clone, PartialEq)]
pub enum TraceExporter {
    Jaeger,
    Zipkin,
    CloudTrace,
    DataDog,
    NewRelic,
    XRay,
}

/// Dashboard configuration
#[derive(Debug, Clone)]
pub struct DashboardConfiguration {
    pub enabled: bool,
    pub dashboard_provider: DashboardProvider,
    pub custom_dashboards: Vec<CustomDashboard>,
    pub auto_generated_dashboards: bool,
}

/// Dashboard providers
#[derive(Debug, Clone, PartialEq)]
pub enum DashboardProvider {
    Grafana,
    Kibana,
    CloudWatch,
    DataDog,
    NewRelic,
    Custom,
}

/// Custom dashboard definition
#[derive(Debug, Clone)]
pub struct CustomDashboard {
    pub name: String,
    pub description: String,
    pub panels: Vec<DashboardPanel>,
    pub refresh_interval: Duration,
}

/// Dashboard panels
#[derive(Debug, Clone)]
pub struct DashboardPanel {
    pub title: String,
    pub panel_type: PanelType,
    pub metrics: Vec<String>,
    pub time_range: TimeRange,
    pub visualization_options: HashMap<String, String>,
}

/// Panel types
#[derive(Debug, Clone, PartialEq)]
pub enum PanelType {
    Graph,
    SingleStat,
    Table,
    Heatmap,
    Logs,
    Gauge,
}

/// Time ranges for panels
#[derive(Debug, Clone)]
pub struct TimeRange {
    pub from: TimeRangeValue,
    pub to: TimeRangeValue,
}

/// Time range values
#[derive(Debug, Clone, PartialEq)]
pub enum TimeRangeValue {
    Now,
    Relative(Duration),
    Absolute(String),
}