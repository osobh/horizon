use crate::{types::TimeSeries, DataPoint, Result, TsdbError};
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for InfluxDB client
#[derive(Debug, Clone)]
pub struct InfluxDbConfig {
    /// Base URL (e.g., "http://localhost:8086")
    pub url: String,
    /// Organization name
    pub org: String,
    /// Bucket name
    pub bucket: String,
    /// API token (optional for testing)
    pub token: Option<String>,
}

impl InfluxDbConfig {
    /// Create a new configuration
    pub fn new(url: impl Into<String>, org: impl Into<String>, bucket: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            org: org.into(),
            bucket: bucket.into(),
            token: None,
        }
    }

    /// Add an API token
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }
}

/// InfluxDB client for querying time-series data
#[derive(Debug, Clone)]
pub struct InfluxDbClient {
    config: InfluxDbConfig,
    client: Client,
}

impl InfluxDbClient {
    /// Create a new InfluxDB client
    pub fn new(config: InfluxDbConfig) -> Result<Self> {
        Ok(Self {
            config,
            client: Client::new(),
        })
    }

    /// Check if InfluxDB is healthy
    #[tracing::instrument(skip(self), err)]
    pub async fn health(&self) -> Result<HealthResponse> {
        let url = format!("{}/health", self.config.url);

        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            return Err(TsdbError::Connection(format!(
                "Health check failed with status: {}",
                response.status()
            )));
        }

        let health = response.json().await?;
        Ok(health)
    }

    /// Write data points to InfluxDB
    #[tracing::instrument(skip(self, points), fields(points_count = points.len()), err)]
    pub async fn write_points(&self, measurement: &str, points: &[DataPoint]) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }

        let url = format!("{}/api/v2/write", self.config.url);

        // Convert points to Line Protocol format
        let line_protocol = self.points_to_line_protocol(measurement, points);

        let mut request = self
            .client
            .post(&url)
            .header("Content-Type", "text/plain; charset=utf-8")
            .query(&[("org", &self.config.org), ("bucket", &self.config.bucket)]);

        if let Some(token) = &self.config.token {
            request = request.header("Authorization", format!("Token {}", token));
        }

        let response = request.body(line_protocol).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(TsdbError::Write(format!(
                "Write failed with status {}: {}",
                status, error_text
            )));
        }

        tracing::debug!("Successfully wrote {} points to InfluxDB", points.len());
        Ok(())
    }

    /// Convert DataPoints to InfluxDB Line Protocol format
    /// Format: <measurement>[,<tag_key>=<tag_value>...] <field_key>=<field_value>[,<field_key>=<field_value>...] [<timestamp>]
    fn points_to_line_protocol(&self, measurement: &str, points: &[DataPoint]) -> String {
        points
            .iter()
            .map(|point| self.point_to_line_protocol(measurement, point))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Convert a single DataPoint to Line Protocol
    fn point_to_line_protocol(&self, measurement: &str, point: &DataPoint) -> String {
        let mut line = measurement.to_string();

        // Add tags (sorted for consistency)
        if !point.tags.is_empty() {
            let mut tag_keys: Vec<_> = point.tags.keys().collect();
            tag_keys.sort();

            for key in tag_keys {
                if let Some(value) = point.tags.get(key) {
                    line.push(',');
                    line.push_str(&escape_tag_key(key));
                    line.push('=');
                    line.push_str(&escape_tag_value(value));
                }
            }
        }

        // Add field (always "value" for our DataPoint)
        line.push(' ');
        line.push_str(&format!("value={}", point.value));

        // Add timestamp (nanoseconds)
        let timestamp_nanos = point
            .timestamp
            .timestamp_nanos_opt()
            .unwrap_or(point.timestamp.timestamp() * 1_000_000_000);
        line.push(' ');
        line.push_str(&timestamp_nanos.to_string());

        line
    }

    /// Execute a Flux query and return time series
    #[tracing::instrument(skip(self), err)]
    pub async fn query(&self, flux_query: &str) -> Result<Vec<TimeSeries>> {
        let url = format!("{}/api/v2/query", self.config.url);

        let mut request = self
            .client
            .post(&url)
            .header("Accept", "application/json")
            .header("Content-Type", "application/vnd.flux")
            .query(&[("org", &self.config.org)]);

        if let Some(token) = &self.config.token {
            request = request.header("Authorization", format!("Token {}", token));
        }

        let response = request.body(flux_query.to_string()).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(TsdbError::Query(format!(
                "Query failed with status {}: {}",
                status, error_text
            )));
        }

        let query_result: QueryResult = response.json().await?;
        let time_series = self.parse_query_result(query_result)?;

        Ok(time_series)
    }

    /// Parse InfluxDB query result into time series
    fn parse_query_result(&self, result: QueryResult) -> Result<Vec<TimeSeries>> {
        let mut series_map: HashMap<String, TimeSeries> = HashMap::new();

        for table in result.results {
            let series_name = table
                .tags
                .get("_measurement")
                .or(table.tags.get("_field"))
                .cloned()
                .unwrap_or_else(|| "unknown".to_string());

            let series = series_map
                .entry(series_name.clone())
                .or_insert_with(|| TimeSeries::new(series_name));

            for row in table.records {
                if let (Some(time), Some(value)) = (row.time, row.value) {
                    let point = DataPoint::with_tags(time, value, row.tags);
                    series.add_point(point);
                }
            }
        }

        let mut time_series: Vec<TimeSeries> = series_map.into_values().collect();

        // Sort points within each series
        for series in &mut time_series {
            series.sort();
        }

        Ok(time_series)
    }
}

/// Health check response from InfluxDB
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Health status ("pass" or "fail")
    pub status: String,
    /// InfluxDB version
    #[serde(default)]
    pub version: String,
    /// Additional message
    #[serde(default)]
    pub message: String,
}

/// Query result from InfluxDB
#[derive(Debug, Clone, Deserialize)]
struct QueryResult {
    results: Vec<TableResult>,
}

#[derive(Debug, Clone, Deserialize)]
struct TableResult {
    #[serde(default)]
    tags: HashMap<String, String>,
    records: Vec<RecordResult>,
}

#[derive(Debug, Clone, Deserialize)]
struct RecordResult {
    #[serde(rename = "_time")]
    time: Option<DateTime<Utc>>,
    #[serde(rename = "_value")]
    value: Option<f64>,
    #[serde(flatten)]
    tags: HashMap<String, String>,
}

/// Escape special characters in tag keys/values for Line Protocol
fn escape_tag_key(s: &str) -> String {
    s.replace(',', "\\,")
        .replace('=', "\\=")
        .replace(' ', "\\ ")
}

/// Escape special characters in tag values for Line Protocol
fn escape_tag_value(s: &str) -> String {
    s.replace(',', "\\,")
        .replace('=', "\\=")
        .replace(' ', "\\ ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_influxdb_config_builder() {
        let config = InfluxDbConfig::new("http://localhost:8086", "test-org", "test-bucket")
            .with_token("test-token");

        assert_eq!(config.url, "http://localhost:8086");
        assert_eq!(config.org, "test-org");
        assert_eq!(config.bucket, "test-bucket");
        assert_eq!(config.token, Some("test-token".to_string()));
    }

    #[test]
    fn test_influxdb_client_creation() {
        let config = InfluxDbConfig::new("http://localhost:8086", "test-org", "test-bucket");
        let client = InfluxDbClient::new(config);

        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn test_health_check_invalid_url() {
        let config =
            InfluxDbConfig::new("http://invalid-url-12345:8086", "test-org", "test-bucket");
        let client = InfluxDbClient::new(config).unwrap();

        let result = client.health().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_query_invalid_url() {
        let config =
            InfluxDbConfig::new("http://invalid-url-12345:8086", "test-org", "test-bucket");
        let client = InfluxDbClient::new(config).unwrap();

        let result = client.query("from(bucket: \"test\")").await;
        assert!(result.is_err());
    }

    #[test]
    fn test_point_to_line_protocol_simple() {
        let config = InfluxDbConfig::new("http://localhost:8086", "test-org", "test-bucket");
        let client = InfluxDbClient::new(config).unwrap();

        let timestamp = DateTime::parse_from_rfc3339("2025-01-15T10:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let point = DataPoint::new(timestamp, 85.5);

        let line = client.point_to_line_protocol("gpu_utilization", &point);

        // Should be: gpu_utilization value=85.5 <timestamp_nanos>
        assert!(line.starts_with("gpu_utilization value=85.5"));
        assert!(line.contains("1736935200000000000")); // 2025-01-15T10:00:00Z in nanoseconds
    }

    #[test]
    fn test_point_to_line_protocol_with_tags() {
        let config = InfluxDbConfig::new("http://localhost:8086", "test-org", "test-bucket");
        let client = InfluxDbClient::new(config).unwrap();

        let timestamp = DateTime::parse_from_rfc3339("2025-01-15T10:00:00Z")
            .unwrap()
            .with_timezone(&Utc);

        let mut tags = HashMap::new();
        tags.insert("host".to_string(), "node-1".to_string());
        tags.insert("gpu".to_string(), "0".to_string());

        let point = DataPoint::with_tags(timestamp, 85.5, tags);
        let line = client.point_to_line_protocol("gpu_utilization", &point);

        // Tags should be sorted alphabetically
        // Should be: gpu_utilization,gpu=0,host=node-1 value=85.5 <timestamp>
        assert!(line.contains("gpu_utilization,gpu=0,host=node-1"));
        assert!(line.contains("value=85.5"));
    }

    #[test]
    fn test_points_to_line_protocol_multiple() {
        let config = InfluxDbConfig::new("http://localhost:8086", "test-org", "test-bucket");
        let client = InfluxDbClient::new(config).unwrap();

        let timestamp1 = DateTime::parse_from_rfc3339("2025-01-15T10:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let timestamp2 = DateTime::parse_from_rfc3339("2025-01-15T10:01:00Z")
            .unwrap()
            .with_timezone(&Utc);

        let points = vec![
            DataPoint::new(timestamp1, 85.5),
            DataPoint::new(timestamp2, 90.0),
        ];

        let protocol = client.points_to_line_protocol("test_metric", &points);

        // Should have two lines separated by newline
        let lines: Vec<&str> = protocol.split('\n').collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("value=85.5"));
        assert!(lines[1].contains("value=90"));
    }

    #[test]
    fn test_escape_tag_key() {
        assert_eq!(escape_tag_key("simple"), "simple");
        assert_eq!(escape_tag_key("with space"), "with\\ space");
        assert_eq!(escape_tag_key("with,comma"), "with\\,comma");
        assert_eq!(escape_tag_key("with=equals"), "with\\=equals");
        assert_eq!(escape_tag_key("all, =special"), "all\\,\\ \\=special");
    }

    #[test]
    fn test_escape_tag_value() {
        assert_eq!(escape_tag_value("simple"), "simple");
        assert_eq!(escape_tag_value("with space"), "with\\ space");
        assert_eq!(escape_tag_value("with,comma"), "with\\,comma");
        assert_eq!(escape_tag_value("with=equals"), "with\\=equals");
    }

    #[tokio::test]
    async fn test_write_points_invalid_url() {
        let config =
            InfluxDbConfig::new("http://invalid-url-12345:8086", "test-org", "test-bucket");
        let client = InfluxDbClient::new(config).unwrap();

        let timestamp = DateTime::parse_from_rfc3339("2025-01-15T10:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let points = vec![DataPoint::new(timestamp, 85.5)];

        let result = client.write_points("test_metric", &points).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_write_points_empty() {
        let config = InfluxDbConfig::new("http://localhost:8086", "test-org", "test-bucket");
        let client = InfluxDbClient::new(config).unwrap();

        let points: Vec<DataPoint> = vec![];
        let result = client.write_points("test_metric", &points).await;

        // Empty points should succeed without making a request
        assert!(result.is_ok());
    }
}
