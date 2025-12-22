use crate::types::{Aggregation, TimeRange};

/// Builder for InfluxDB Flux queries
#[derive(Debug, Clone)]
pub struct QueryBuilder {
    bucket: String,
    measurement: Option<String>,
    field: Option<String>,
    time_range: Option<TimeRange>,
    filters: Vec<(String, String)>,
    aggregation: Option<Aggregation>,
    window: Option<String>,
    group_by: Vec<String>,
}

impl QueryBuilder {
    /// Create a new query builder for the specified bucket
    pub fn new(bucket: impl Into<String>) -> Self {
        Self {
            bucket: bucket.into(),
            measurement: None,
            field: None,
            time_range: None,
            filters: Vec::new(),
            aggregation: None,
            window: None,
            group_by: Vec::new(),
        }
    }

    /// Set the measurement to query
    pub fn measurement(mut self, measurement: impl Into<String>) -> Self {
        self.measurement = Some(measurement.into());
        self
    }

    /// Set the field to query
    pub fn field(mut self, field: impl Into<String>) -> Self {
        self.field = Some(field.into());
        self
    }

    /// Set the time range
    pub fn time_range(mut self, range: TimeRange) -> Self {
        self.time_range = Some(range);
        self
    }

    /// Add a tag filter
    pub fn filter(mut self, tag: impl Into<String>, value: impl Into<String>) -> Self {
        self.filters.push((tag.into(), value.into()));
        self
    }

    /// Set aggregation function
    pub fn aggregate(mut self, agg: Aggregation) -> Self {
        self.aggregation = Some(agg);
        self
    }

    /// Set aggregation window (e.g., "1h", "5m", "1d")
    pub fn window(mut self, window: impl Into<String>) -> Self {
        self.window = Some(window.into());
        self
    }

    /// Add a group-by tag
    pub fn group_by(mut self, tag: impl Into<String>) -> Self {
        self.group_by.push(tag.into());
        self
    }

    /// Build the Flux query string
    pub fn build(self) -> crate::Result<String> {
        let mut query = String::new();

        // Start with bucket
        query.push_str(&format!("from(bucket: \"{}\")\n", self.bucket));

        // Add time range
        if let Some(range) = self.time_range {
            let start = range.start.to_rfc3339();
            let stop = range.end.to_rfc3339();
            query.push_str(&format!("  |> range(start: {}, stop: {})\n", start, stop));
        } else {
            // Default to last hour if no range specified
            query.push_str("  |> range(start: -1h)\n");
        }

        // Filter by measurement
        if let Some(measurement) = &self.measurement {
            query.push_str(&format!(
                "  |> filter(fn: (r) => r[\"_measurement\"] == \"{}\")\n",
                measurement
            ));
        }

        // Filter by field
        if let Some(field) = &self.field {
            query.push_str(&format!(
                "  |> filter(fn: (r) => r[\"_field\"] == \"{}\")\n",
                field
            ));
        }

        // Add tag filters
        for (tag, value) in &self.filters {
            query.push_str(&format!(
                "  |> filter(fn: (r) => r[\"{}\"] == \"{}\")\n",
                tag, value
            ));
        }

        // Add aggregation window if specified
        if let Some(window) = &self.window {
            if let Some(agg) = &self.aggregation {
                query.push_str(&format!(
                    "  |> aggregateWindow(every: {}, fn: {})\n",
                    window,
                    agg.to_flux_fn()
                ));
            }
        } else if let Some(agg) = &self.aggregation {
            // Group by specified tags or use default grouping
            if !self.group_by.is_empty() {
                let tags = self
                    .group_by
                    .iter()
                    .map(|t| format!("\"{}\"", t))
                    .collect::<Vec<_>>()
                    .join(", ");
                query.push_str(&format!("  |> group(columns: [{}])\n", tags));
            }
            query.push_str(&format!("  |> {}()\n", agg.to_flux_fn()));
        }

        Ok(query)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    #[test]
    fn test_query_builder_basic() {
        let query = QueryBuilder::new("metrics")
            .measurement("gpu_utilization")
            .field("value")
            .build()
            .unwrap();

        println!("Query:\n{}", query);

        assert!(query.contains("from(bucket: \"metrics\")"));
        assert!(query.contains("r[\"_measurement\"] == \"gpu_utilization\""));
        assert!(query.contains("r[\"_field\"] == \"value\""));
    }

    #[test]
    fn test_query_builder_with_time_range() {
        let start = Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2025, 1, 2, 0, 0, 0).unwrap();
        let range = TimeRange::new(start, end).unwrap();

        let query = QueryBuilder::new("metrics")
            .measurement("gpu_utilization")
            .time_range(range)
            .build()
            .unwrap();

        assert!(query.contains("range(start: 2025-01-01T00:00:00+00:00, stop: 2025-01-02T00:00:00+00:00)"));
    }

    #[test]
    fn test_query_builder_with_filters() {
        let query = QueryBuilder::new("metrics")
            .measurement("gpu_utilization")
            .filter("host", "gpu-node-01")
            .filter("gpu", "0")
            .build()
            .unwrap();

        assert!(query.contains("r[\"host\"] == \"gpu-node-01\""));
        assert!(query.contains("r[\"gpu\"] == \"0\""));
    }

    #[test]
    fn test_query_builder_with_aggregation() {
        let query = QueryBuilder::new("metrics")
            .measurement("gpu_utilization")
            .aggregate(Aggregation::Mean)
            .build()
            .unwrap();

        assert!(query.contains("mean()"));
    }

    #[test]
    fn test_query_builder_with_window_aggregation() {
        let query = QueryBuilder::new("metrics")
            .measurement("gpu_utilization")
            .aggregate(Aggregation::Mean)
            .window("1h")
            .build()
            .unwrap();

        assert!(query.contains("aggregateWindow(every: 1h, fn: mean)"));
    }

    #[test]
    fn test_query_builder_with_group_by() {
        let query = QueryBuilder::new("metrics")
            .measurement("gpu_utilization")
            .aggregate(Aggregation::Mean)
            .group_by("host")
            .group_by("gpu")
            .build()
            .unwrap();

        assert!(query.contains("group(columns: [\"host\", \"gpu\"])"));
        assert!(query.contains("mean()"));
    }

    #[test]
    fn test_query_builder_complex() {
        let start = Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2025, 1, 7, 0, 0, 0).unwrap();
        let range = TimeRange::new(start, end).unwrap();

        let query = QueryBuilder::new("metrics")
            .measurement("gpu_utilization")
            .field("value")
            .time_range(range)
            .filter("host", "gpu-node-01")
            .aggregate(Aggregation::Mean)
            .window("1d")
            .build()
            .unwrap();

        // Verify all components are present
        assert!(query.contains("from(bucket: \"metrics\")"));
        assert!(query.contains("range(start:"));
        assert!(query.contains("r[\"_measurement\"] == \"gpu_utilization\""));
        assert!(query.contains("r[\"_field\"] == \"value\""));
        assert!(query.contains("r[\"host\"] == \"gpu-node-01\""));
        assert!(query.contains("aggregateWindow(every: 1d, fn: mean)"));
    }
}
