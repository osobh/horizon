# Monitoring

Comprehensive system monitoring and profiling for StratoSwarm with Prometheus integration.

## Overview

The `monitoring` crate provides real-time observability into StratoSwarm's distributed system. It collects metrics from all components, profiles performance bottlenecks, traces distributed operations, and exports data to standard monitoring systems like Prometheus and Grafana.

## Features

- **Prometheus Metrics**: Standard metrics export with custom collectors
- **Performance Profiling**: CPU, memory, and GPU profiling capabilities
- **Distributed Tracing**: Track operations across multiple nodes
- **Custom Metrics**: Define application-specific measurements
- **Real-time Dashboards**: Pre-built Grafana dashboards
- **Alerting Rules**: Configurable alerts for anomalies
- **Low Overhead**: Minimal impact on system performance
- **GPU Monitoring**: NVIDIA GPU metrics collection

## Usage

### Basic Metrics Collection

```rust
use monitoring::{MetricsCollector, Counter, Gauge, Histogram};

// Initialize metrics collector
let collector = MetricsCollector::new()
    .with_prefix("stratoswarm")
    .with_labels(vec![("node", "gpu-01")]);

// Counter for events
let requests = Counter::new("requests_total", "Total requests processed");
requests.inc();

// Gauge for current values  
let memory_usage = Gauge::new("memory_bytes", "Current memory usage");
memory_usage.set(1073741824); // 1GB

// Histogram for distributions
let latency = Histogram::new("request_latency_seconds", "Request latency");
latency.observe(0.045); // 45ms
```

### GPU Monitoring

```rust
use monitoring::gpu::{GpuMonitor, GpuMetrics};

// Create GPU monitor
let gpu_monitor = GpuMonitor::new()?;

// Collect GPU metrics
let metrics = gpu_monitor.collect_all()?;
for (device_id, device_metrics) in metrics {
    println!("GPU {}: {}% utilization, {}°C",
        device_id,
        device_metrics.utilization,
        device_metrics.temperature
    );
}

// Monitor specific GPU
let gpu_0 = gpu_monitor.device(0)?;
println!("Memory: {}/{} MB", 
    gpu_0.memory_used_mb(),
    gpu_0.memory_total_mb()
);
```

### Performance Profiling

```rust
use monitoring::{Profiler, ProfileScope};

// Create profiler
let profiler = Profiler::new();

// Profile a code section
{
    let _scope = ProfileScope::new(&profiler, "critical_section");
    // Your code here
    expensive_operation();
} // Automatically records duration

// Get profile results
let report = profiler.report();
println!("Critical section: {}ms avg, {}ms p99",
    report.avg_duration_ms("critical_section"),
    report.p99_duration_ms("critical_section")
);
```

### Distributed Tracing

```rust
use monitoring::{Tracer, Span, SpanContext};

// Initialize tracer
let tracer = Tracer::new("stratoswarm");

// Start root span
let root_span = tracer.start_span("process_request");

// Create child span
let child_span = tracer.start_span("database_query")
    .with_parent(&root_span)
    .with_tag("db.type", "postgres");

// Add events to span
child_span.add_event("query_started");
let result = query_database().await?;
child_span.add_event("query_completed");

// Spans automatically close when dropped
```

### Custom Metrics

```rust
use monitoring::{MetricBuilder, MetricType};

// Define custom metric
let gpu_kernel_launches = MetricBuilder::new()
    .name("gpu_kernel_launches_total")
    .metric_type(MetricType::Counter)
    .help("Number of GPU kernel launches")
    .label("kernel_type")
    .label("device_id")
    .build();

// Use custom metric
gpu_kernel_launches
    .with_labels(&[("kernel_type", "matrix_mul"), ("device_id", "0")])
    .inc();
```

## Prometheus Integration

### Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'stratoswarm'
    static_configs:
      - targets: ['localhost:9090']
```

### Metric Export

```rust
use monitoring::prometheus::{PrometheusExporter, ExporterConfig};

// Configure exporter
let config = ExporterConfig::default()
    .port(9090)
    .path("/metrics")
    .refresh_interval(Duration::from_secs(10));

// Start exporter
let exporter = PrometheusExporter::new(config);
exporter.start().await?;
```

## Pre-built Metrics

The crate provides standard metrics out of the box:

### System Metrics
- CPU usage (per core)
- Memory usage (RSS, heap, stack)
- Disk I/O (read/write bytes and ops)
- Network I/O (bytes in/out, packets)
- File descriptors

### Agent Metrics
- Active agents count
- Agent lifecycle events
- Message throughput
- Goal completion rate
- Evolution fitness scores

### GPU Metrics
- Utilization percentage
- Memory usage
- Temperature
- Power consumption
- PCIe throughput
- Kernel execution time

### Cluster Metrics
- Node count by type
- Inter-node latency
- Bandwidth utilization
- Job queue depth
- Resource allocation

## Grafana Dashboards

Pre-built dashboards available in `dashboards/`:

- **System Overview**: Overall cluster health
- **GPU Performance**: Detailed GPU metrics
- **Agent Activity**: Agent lifecycle and performance
- **Network Traffic**: Inter-node communication
- **Evolution Progress**: Evolution engine metrics

Import dashboards:
```bash
# Import all dashboards
./scripts/import_dashboards.sh

# Or import specific dashboard
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @dashboards/gpu_performance.json
```

## Alerting Rules

Configure alerts for common issues:

```yaml
groups:
  - name: stratoswarm_alerts
    rules:
      - alert: HighGPUTemperature
        expr: gpu_temperature_celsius > 85
        for: 5m
        annotations:
          summary: "GPU temperature critical"
          
      - alert: LowMemory
        expr: (node_memory_available_bytes / node_memory_total_bytes) < 0.1
        for: 5m
        annotations:
          summary: "Less than 10% memory available"
          
      - alert: AgentCrashLoop
        expr: increase(agent_crashes_total[5m]) > 5
        annotations:
          summary: "Agent crashing repeatedly"
```

## Performance Overhead

Monitoring overhead is minimal:
- **Metrics Collection**: <0.1% CPU overhead
- **Profiling**: <1% when enabled
- **Tracing**: ~100ns per span
- **Memory**: ~10MB base + 1KB per metric

## Configuration

```toml
[monitoring]
# Metrics settings
metrics_port = 9090
metrics_path = "/metrics"
collection_interval = "10s"

# Profiling settings
profiling_enabled = true
profile_sample_rate = 0.1  # 10% sampling

# Tracing settings
tracing_enabled = true
trace_sample_rate = 0.01  # 1% sampling
jaeger_endpoint = "http://localhost:14268/api/traces"

# GPU monitoring
gpu_monitoring_enabled = true
gpu_poll_interval = "5s"

# Retention
metrics_retention_days = 30
```

## Testing

```bash
# Run all tests
cargo test

# Test metrics collection
cargo test metrics

# Test GPU monitoring (requires GPU)
cargo test gpu --features gpu_required

# Benchmark overhead
cargo bench overhead
```

## Coverage

Current test coverage: 90%+ (Excellent)

Well-tested areas:
- Metric types and operations
- Prometheus format export
- Profiler accuracy
- Configuration parsing

Hardware-dependent:
- GPU metrics (requires NVIDIA GPU)
- Some performance counters

## CLI Tools

The crate includes helpful CLI tools:

```bash
# Monitor real-time metrics
cargo run --bin stratoswarm-monitor

# Export metrics snapshot
cargo run --bin metrics-export -- --format json > metrics.json

# Analyze performance profile
cargo run --bin profile-analyzer -- trace.pprof
```

## Integration

Used throughout StratoSwarm:
- All crates include monitoring hooks
- Automatic instrumentation for key operations
- Standardized metric naming conventions
- Consistent label schemes

## Best Practices

1. **Use standard metric names**: Follow Prometheus naming conventions
2. **Add meaningful labels**: But avoid high-cardinality labels
3. **Set up alerts early**: Don't wait for problems to occur
4. **Profile in production**: But with low sampling rates
5. **Trace critical paths**: Focus on user-facing operations

## License

MIT