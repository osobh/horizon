# horizon-tsdbx

Time-series database abstractions for Horizon GPU capacity management.

**Version**: 0.1.0
**Status**: Production Ready ✅
**Tests**: 28 passing

## Overview

`horizon-tsdbx` provides trait-based abstractions for time-series databases with a primary implementation for InfluxDB. It enables both reading and writing historical metrics for capacity forecasting and analysis.

## Features

- **InfluxDB Client**: Full-featured client with health checks, queries, and writes
- **Flux Query Support**: Execute Flux queries and parse results into time series
- **Write API**: Write data points using InfluxDB Line Protocol
- **Type-Safe Models**: `DataPoint`, `TimeSeries`, `TimeRange`, `Aggregation`
- **Query Builder**: Fluent API for constructing Flux queries
- **Comprehensive Tests**: 28 tests covering all functionality

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
horizon-tsdbx = { path = "../../crates/tsdbx" }
```

## Quick Start

### Initialize Client

```rust
use horizon_tsdbx::{InfluxDbClient, client::influxdb::InfluxDbConfig};

let config = InfluxDbConfig::new(
    "http://localhost:8086",
    "my-org",
    "my-bucket"
).with_token("my-token");

let client = InfluxDbClient::new(config)?;
```

### Write Data Points

```rust
use horizon_tsdbx::DataPoint;
use chrono::Utc;
use std::collections::HashMap;

// Create a data point with tags
let mut tags = HashMap::new();
tags.insert("host".to_string(), "gpu-node-01".to_string());
tags.insert("gpu".to_string(), "0".to_string());

let point = DataPoint::with_tags(Utc::now(), 85.5, tags);

// Write to InfluxDB
client.write_points("gpu_utilization", &[point]).await?;
```

### Query Data

```rust
use horizon_tsdbx::QueryBuilder;

let query = QueryBuilder::new("my-bucket")
    .measurement("gpu_utilization")
    .field("value")
    .range_hours(24)
    .aggregation(horizon_tsdbx::Aggregation::Mean)
    .window("5m")
    .build();

let time_series = client.query(&query).await?;
```

### Health Check

```rust
let health = client.health().await?;
println!("InfluxDB status: {}", health.status);
```

## API Reference

### InfluxDbConfig

Configuration for the InfluxDB client.

```rust
pub struct InfluxDbConfig {
    pub url: String,      // Base URL (e.g., "http://localhost:8086")
    pub org: String,      // Organization name
    pub bucket: String,   // Bucket name
    pub token: Option<String>, // API token (optional)
}
```

**Methods**:
- `new(url, org, bucket)` - Create configuration
- `with_token(token)` - Add API token

### InfluxDbClient

Client for interacting with InfluxDB.

```rust
impl InfluxDbClient {
    pub fn new(config: InfluxDbConfig) -> Result<Self>;
    pub async fn health(&self) -> Result<HealthResponse>;
    pub async fn write_points(&self, measurement: &str, points: &[DataPoint]) -> Result<()>;
    pub async fn query(&self, flux_query: &str) -> Result<Vec<TimeSeries>>;
}
```

### DataPoint

Represents a single time-series data point.

```rust
pub struct DataPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub tags: HashMap<String, String>,
}
```

**Methods**:
- `new(timestamp, value)` - Create without tags
- `with_tags(timestamp, value, tags)` - Create with tags

### TimeSeries

A collection of data points.

```rust
pub struct TimeSeries {
    pub name: String,
    pub points: Vec<DataPoint>,
}
```

**Methods**:
- `new(name)` - Create empty series
- `add_point(point)` - Add data point
- `values()` - Extract values as Vec<f64>
- `len()` - Number of points
- `is_empty()` - Check if empty
- `sort()` - Sort by timestamp

### TimeRange

Represents a time range for queries.

```rust
pub struct TimeRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}
```

**Methods**:
- `new(start, end)` - Create with validation
- `last_hours(hours)` - Create for last N hours
- `last_days(days)` - Create for last N days
- `duration_secs()` - Get duration in seconds

### Aggregation

Enum for aggregation functions.

```rust
pub enum Aggregation {
    Mean,
    Min,
    Max,
    Sum,
    Count,
    Median,
    First,
    Last,
}
```

**Methods**:
- `to_flux_fn()` - Convert to Flux function name

### QueryBuilder

Fluent API for building Flux queries.

```rust
let query = QueryBuilder::new("bucket")
    .measurement("cpu_usage")
    .field("value")
    .range_hours(24)
    .filter("host", "node-1")
    .aggregation(Aggregation::Mean)
    .window("5m")
    .group_by(&["host"])
    .build();
```

**Methods**:
- `new(bucket)` - Start builder
- `measurement(name)` - Set measurement
- `field(name)` - Set field
- `range(start, end)` - Set time range
- `range_hours(hours)` - Last N hours
- `range_days(days)` - Last N days
- `filter(tag, value)` - Add tag filter
- `aggregation(agg)` - Set aggregation
- `window(duration)` - Set window size
- `group_by(tags)` - Group by tags
- `build()` - Generate Flux query

## InfluxDB Line Protocol

The write API uses InfluxDB Line Protocol format:

```
<measurement>[,<tag_key>=<tag_value>...] <field_key>=<field_value>[,<field_key>=<field_value>...] [<timestamp>]
```

**Example**:

```
gpu_utilization,host=node-1,gpu=0 value=85.5 1736935200000000000
```

**Special Characters**:
- Commas, equals signs, and spaces in tag keys/values are automatically escaped
- Timestamps are in nanoseconds

## Error Handling

The crate uses a custom error type:

```rust
pub enum TsdbError {
    Connection(String),
    Query(String),
    Write(String),
    Parse(String),
    Config(String),
    Http(Box<reqwest::Error>),
    Json(serde_json::Error),
    InvalidTimeRange(String),
    MissingField(String),
}
```

All functions return `Result<T, TsdbError>`.

## Examples

### Write Multiple Points with Different Tags

```rust
use horizon_tsdbx::DataPoint;
use chrono::Utc;
use std::collections::HashMap;

let now = Utc::now();
let mut points = Vec::new();

// GPU 0
let mut tags0 = HashMap::new();
tags0.insert("host".to_string(), "node-1".to_string());
tags0.insert("gpu".to_string(), "0".to_string());
points.push(DataPoint::with_tags(now, 85.5, tags0));

// GPU 1
let mut tags1 = HashMap::new();
tags1.insert("host".to_string(), "node-1".to_string());
tags1.insert("gpu".to_string(), "1".to_string());
points.push(DataPoint::with_tags(now, 92.3, tags1));

// Write both points in one request
client.write_points("gpu_metrics", &points).await?;
```

### Query with Aggregation and Grouping

```rust
use horizon_tsdbx::{QueryBuilder, Aggregation};

let query = QueryBuilder::new("metrics")
    .measurement("gpu_metrics")
    .field("value")
    .range_hours(24)
    .filter("metric_name", "utilization")
    .aggregation(Aggregation::Mean)
    .window("5m")
    .group_by(&["resource_id"])
    .build();

let time_series = client.query(&query).await?;

for series in time_series {
    println!("Series: {}", series.name);
    println!("Points: {}", series.len());
    println!("Values: {:?}", series.values());
}
```

### Create Time Range Queries

```rust
use horizon_tsdbx::TimeRange;
use chrono::{Duration, Utc};

// Last 7 days
let range = TimeRange::last_days(7);

// Custom range
let start = Utc::now() - Duration::hours(48);
let end = Utc::now();
let range = TimeRange::new(start, end)?;

println!("Duration: {} seconds", range.duration_secs());
```

## Testing

Run the test suite:

```bash
cargo test -p horizon-tsdbx
```

**Test Coverage**:
- Client configuration: 2 tests
- Health checks: 1 test
- Query execution: 1 test
- Write operations: 3 tests
- Line Protocol conversion: 4 tests
- Escaping functions: 2 tests
- Type models: 9 tests
- Query builder: 6 tests

**Total**: 28 tests, all passing ✅

## Integration with Data Ingestion Service

The Data Ingestion Service uses `horizon-tsdbx` to persist uploaded metrics:

```rust
use horizon_tsdbx::{InfluxDbClient, DataPoint};

// Convert MetricRecord to DataPoint
let datapoint = metric_record.to_datapoint();

// Write to InfluxDB
let measurement = format!("{}_metrics", resource_type);
client.write_points(&measurement, &[datapoint]).await?;
```

Metrics are grouped by `resource_type` (gpu, cpu, memory, etc.) into separate measurements for efficient querying.

## Performance Characteristics

- **Write Throughput**: Batched writes recommended (100-1000 points per request)
- **Line Protocol Overhead**: Minimal (~50 bytes per point)
- **Query Latency**: Depends on time range and aggregation complexity
- **Memory Usage**: ~100 bytes per DataPoint in memory

## Configuration

### Environment Variables

```bash
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your-token
INFLUXDB_ORG=horizon
INFLUXDB_BUCKET=metrics
```

### Connection Pooling

The `reqwest::Client` used internally handles connection pooling automatically.

## Dependencies

- `reqwest` - HTTP client
- `serde` / `serde_json` - Serialization
- `chrono` - Time handling
- `thiserror` / `anyhow` - Error handling
- `tracing` - Instrumentation
- `tokio` - Async runtime

## Contributing

When adding new features:
1. Add tests for new functionality
2. Update this README
3. Ensure all tests pass: `cargo test -p horizon-tsdbx`
4. Run `cargo clippy` to check for warnings

## Future Enhancements

### Short-term
- [ ] Batch write optimization with configurable batch size
- [ ] Retry logic with exponential backoff
- [ ] Connection pooling configuration
- [ ] Custom timeout settings

### Medium-term
- [ ] Support for multiple InfluxDB versions (v1 and v2)
- [ ] Async query streaming for large result sets
- [ ] Query result caching
- [ ] Metrics for monitoring client performance

### Long-term
- [ ] Support for other time-series databases (Prometheus, TimescaleDB)
- [ ] Distributed query execution
- [ ] Advanced query optimization

## License

Part of the Horizon project.

---

**Prepared By**: Claude Code (Anthropic)
**Status**: Production Ready ✅
**Test Coverage**: 28 tests passing
