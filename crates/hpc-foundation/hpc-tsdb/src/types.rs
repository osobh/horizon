use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Represents a time range for querying time-series data
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeRange {
    /// Start time (inclusive)
    pub start: DateTime<Utc>,
    /// End time (exclusive)
    pub end: DateTime<Utc>,
}

impl TimeRange {
    /// Create a new time range
    pub fn new(start: DateTime<Utc>, end: DateTime<Utc>) -> crate::Result<Self> {
        if start >= end {
            return Err(crate::TsdbError::InvalidTimeRange(format!(
                "start ({}) must be before end ({})",
                start, end
            )));
        }
        Ok(Self { start, end })
    }

    /// Create a time range for the last N hours
    pub fn last_hours(hours: i64) -> Self {
        let end = Utc::now();
        let start = end - chrono::Duration::hours(hours);
        Self { start, end }
    }

    /// Create a time range for the last N days
    pub fn last_days(days: i64) -> Self {
        let end = Utc::now();
        let start = end - chrono::Duration::days(days);
        Self { start, end }
    }

    /// Get the duration of this time range in seconds
    pub fn duration_secs(&self) -> i64 {
        (self.end - self.start).num_seconds()
    }
}

/// A single data point in a time series
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataPoint {
    /// Timestamp of the data point
    pub timestamp: DateTime<Utc>,
    /// Value at this timestamp
    pub value: f64,
    /// Optional tags for grouping/filtering
    #[serde(default)]
    pub tags: std::collections::HashMap<String, String>,
}

impl DataPoint {
    /// Create a new data point
    pub fn new(timestamp: DateTime<Utc>, value: f64) -> Self {
        Self {
            timestamp,
            value,
            tags: std::collections::HashMap::new(),
        }
    }

    /// Create a data point with tags
    pub fn with_tags(
        timestamp: DateTime<Utc>,
        value: f64,
        tags: std::collections::HashMap<String, String>,
    ) -> Self {
        Self {
            timestamp,
            value,
            tags,
        }
    }
}

/// A time series is a collection of data points
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeSeries {
    /// Name/identifier of the time series
    pub name: String,
    /// Data points in chronological order
    pub points: Vec<DataPoint>,
}

impl TimeSeries {
    /// Create a new time series
    pub fn new(name: String) -> Self {
        Self {
            name,
            points: Vec::new(),
        }
    }

    /// Add a data point to the time series
    pub fn add_point(&mut self, point: DataPoint) {
        self.points.push(point);
    }

    /// Get the values as a vector (for forecasting)
    pub fn values(&self) -> Vec<f64> {
        self.points.iter().map(|p| p.value).collect()
    }

    /// Get the number of data points
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if the time series is empty
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Sort points by timestamp (ascending)
    pub fn sort(&mut self) {
        self.points.sort_by_key(|p| p.timestamp);
    }
}

/// Aggregation functions for time-series data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Aggregation {
    /// Mean/average value
    Mean,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Sum of values
    Sum,
    /// Count of values
    Count,
    /// Median value
    Median,
    /// First value
    First,
    /// Last value
    Last,
}

impl Aggregation {
    /// Convert to InfluxDB Flux function name
    pub fn to_flux_fn(&self) -> &'static str {
        match self {
            Aggregation::Mean => "mean",
            Aggregation::Min => "min",
            Aggregation::Max => "max",
            Aggregation::Sum => "sum",
            Aggregation::Count => "count",
            Aggregation::Median => "median",
            Aggregation::First => "first",
            Aggregation::Last => "last",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_range_validation() {
        let now = Utc::now();
        let past = now - chrono::Duration::hours(1);

        // Valid range
        let range = TimeRange::new(past, now);
        assert!(range.is_ok());

        // Invalid range (start after end)
        let range = TimeRange::new(now, past);
        assert!(range.is_err());

        // Invalid range (start equals end)
        let range = TimeRange::new(now, now);
        assert!(range.is_err());
    }

    #[test]
    fn test_time_range_last_hours() {
        let range = TimeRange::last_hours(24);
        assert!(range.duration_secs() > 86000); // ~24 hours in seconds
        assert!(range.duration_secs() < 87000);
    }

    #[test]
    fn test_time_range_last_days() {
        let range = TimeRange::last_days(7);
        assert!(range.duration_secs() > 604000); // ~7 days in seconds
        assert!(range.duration_secs() < 605000);
    }

    #[test]
    fn test_data_point_creation() {
        let now = Utc::now();
        let point = DataPoint::new(now, 42.5);

        assert_eq!(point.timestamp, now);
        assert_eq!(point.value, 42.5);
        assert!(point.tags.is_empty());
    }

    #[test]
    fn test_data_point_with_tags() {
        let now = Utc::now();
        let mut tags = std::collections::HashMap::new();
        tags.insert("host".to_string(), "gpu-node-01".to_string());
        tags.insert("gpu".to_string(), "0".to_string());

        let point = DataPoint::with_tags(now, 85.0, tags.clone());

        assert_eq!(point.timestamp, now);
        assert_eq!(point.value, 85.0);
        assert_eq!(point.tags, tags);
    }

    #[test]
    fn test_time_series_operations() {
        let mut ts = TimeSeries::new("gpu_utilization".to_string());
        assert_eq!(ts.len(), 0);
        assert!(ts.is_empty());

        let now = Utc::now();
        ts.add_point(DataPoint::new(now, 50.0));
        ts.add_point(DataPoint::new(now + chrono::Duration::hours(1), 75.0));
        ts.add_point(DataPoint::new(now + chrono::Duration::hours(2), 60.0));

        assert_eq!(ts.len(), 3);
        assert!(!ts.is_empty());
        assert_eq!(ts.values(), vec![50.0, 75.0, 60.0]);
    }

    #[test]
    fn test_time_series_sort() {
        let mut ts = TimeSeries::new("test".to_string());
        let now = Utc::now();

        // Add points out of order
        ts.add_point(DataPoint::new(now + chrono::Duration::hours(2), 60.0));
        ts.add_point(DataPoint::new(now, 50.0));
        ts.add_point(DataPoint::new(now + chrono::Duration::hours(1), 75.0));

        ts.sort();

        assert_eq!(ts.values(), vec![50.0, 75.0, 60.0]);
    }

    #[test]
    fn test_aggregation_to_flux_fn() {
        assert_eq!(Aggregation::Mean.to_flux_fn(), "mean");
        assert_eq!(Aggregation::Min.to_flux_fn(), "min");
        assert_eq!(Aggregation::Max.to_flux_fn(), "max");
        assert_eq!(Aggregation::Sum.to_flux_fn(), "sum");
        assert_eq!(Aggregation::Count.to_flux_fn(), "count");
        assert_eq!(Aggregation::Median.to_flux_fn(), "median");
        assert_eq!(Aggregation::First.to_flux_fn(), "first");
        assert_eq!(Aggregation::Last.to_flux_fn(), "last");
    }
}
