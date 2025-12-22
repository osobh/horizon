//! # Horizon Tracing and Metrics
//!
//! This crate provides unified tracing and metrics initialization for all Horizon services.
//! It combines structured logging (via `tracing`), OpenTelemetry export, and Prometheus metrics
//! into a single, easy-to-use API.
//!
//! ## Features
//!
//! - **Layered Tracing**: Combines stdout formatting, environment filtering, and OpenTelemetry export
//! - **Metrics Export**: Prometheus HTTP endpoint for metrics scraping
//! - **Service Identification**: Automatic service name injection into spans
//! - **Guard Pattern**: Automatic cleanup on drop (flushes OTLP, etc.)
//! - **Graceful Degradation**: Continues to work even if OTLP endpoint is unavailable
//! - **Thread Safety**: Safe to use in concurrent environments
//!
//! ## Usage
//!
//! ```rust,no_run
//! use hpc_tracing::{TracingConfig, init, init_metrics};
//! use tracing::info;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Initialize metrics endpoint
//!     init_metrics("0.0.0.0:9090".parse()?)?;
//!
//!     // Initialize tracing
//!     let config = TracingConfig {
//!         service_name: "my-service".to_string(),
//!         log_level: "info".to_string(),
//!         otlp_endpoint: Some("http://localhost:4317".to_string()),
//!     };
//!     let _guard = init(config)?;
//!
//!     // Use tracing and metrics
//!     info!("Service started");
//!     metrics::increment_counter!("requests_total");
//!
//!     Ok(())
//! }
//! ```

use hpc_error::{HpcError, Result};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing_subscriber::{
    layer::SubscriberExt, util::SubscriberInitExt, EnvFilter,
};

/// Global flag to track if tracing has been initialized
static TRACING_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Global flag to track if metrics have been initialized
static METRICS_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Configuration for tracing initialization.
///
/// ## Fields
///
/// - `service_name`: The name of the service (used in spans and metrics). Must be non-empty
///   and contain only alphanumeric characters, hyphens, and underscores.
/// - `log_level`: The minimum log level to emit. Valid values: "trace", "debug", "info", "warn", "error".
/// - `otlp_endpoint`: Optional OpenTelemetry Protocol (OTLP) endpoint for distributed tracing.
///   If provided but unreachable, tracing will continue with local logging only.
///
/// ## Example
///
/// ```rust
/// use hpc_tracing::TracingConfig;
///
/// let config = TracingConfig {
///     service_name: "telemetry-collector".to_string(),
///     log_level: "info".to_string(),
///     otlp_endpoint: Some("http://jaeger:4317".to_string()),
/// };
/// ```
#[derive(Clone, Debug)]
pub struct TracingConfig {
    pub service_name: String,
    pub log_level: String,
    pub otlp_endpoint: Option<String>,
}

impl TracingConfig {
    /// Validates the configuration.
    ///
    /// ## Errors
    ///
    /// - `HpcError::InvalidInput` if `service_name` is empty or contains invalid characters
    /// - `HpcError::InvalidInput` if `log_level` is not a valid log level
    fn validate(&self) -> Result<()> {
        // Validate service name
        if self.service_name.is_empty() {
            return Err(HpcError::InvalidInput {
                field: "service_name".to_string(),
                reason: "service name cannot be empty".to_string(),
            });
        }

        if !self
            .service_name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return Err(HpcError::InvalidInput {
                field: "service_name".to_string(),
                reason: "service name must contain only alphanumeric characters, hyphens, and underscores".to_string(),
            });
        }

        // Validate log level
        let valid_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_levels.contains(&self.log_level.as_str()) {
            return Err(HpcError::InvalidInput {
                field: "log_level".to_string(),
                reason: format!(
                    "invalid log level '{}', must be one of: {}",
                    self.log_level,
                    valid_levels.join(", ")
                ),
            });
        }

        Ok(())
    }
}

/// Guard for tracing cleanup.
///
/// When dropped, this guard will flush any pending spans and shut down the tracer provider.
/// This ensures that all telemetry data is exported before the application exits.
///
/// ## Example
///
/// ```rust,no_run
/// use hpc_tracing::{TracingConfig, init};
///
/// fn main() -> anyhow::Result<()> {
///     let config = TracingConfig {
///         service_name: "my-service".to_string(),
///         log_level: "info".to_string(),
///         otlp_endpoint: None,
///     };
///     let _guard = init(config)?;
///
///     // Tracing is active here
///
///     // When _guard is dropped, tracing is cleaned up
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct TracingGuard;

impl Drop for TracingGuard {
    fn drop(&mut self) {
        // Shutdown the global tracer provider
        opentelemetry::global::shutdown_tracer_provider();
    }
}

/// Initializes the tracing subsystem with the provided configuration.
///
/// This function sets up:
/// - A formatted stdout logger
/// - Environment-based log level filtering
/// - OpenTelemetry export (if configured)
/// - Service name injection into all spans
///
/// ## Thread Safety
///
/// This function uses global state and can only be called once successfully.
/// Subsequent calls will return an error.
///
/// ## Graceful Degradation
///
/// If the OTLP endpoint is configured but unreachable, the function will log a warning
/// and continue with local logging only. This ensures that services can start even if
/// the observability backend is unavailable.
///
/// ## Errors
///
/// - `HpcError::InvalidInput` if the configuration is invalid
/// - `HpcError::Telemetry` if tracing has already been initialized
/// - `HpcError::Telemetry` if the subscriber cannot be set
///
/// ## Example
///
/// ```rust,no_run
/// use hpc_tracing::{TracingConfig, init};
/// use tracing::{info, instrument};
///
/// #[instrument]
/// fn process_data(id: u64) {
///     info!(id, "Processing data");
/// }
///
/// fn main() -> anyhow::Result<()> {
///     let config = TracingConfig {
///         service_name: "data-processor".to_string(),
///         log_level: "debug".to_string(),
///         otlp_endpoint: Some("http://localhost:4317".to_string()),
///     };
///     let _guard = init(config)?;
///
///     process_data(42);
///
///     Ok(())
/// }
/// ```
pub fn init(config: TracingConfig) -> Result<TracingGuard> {
    // Validate configuration
    config.validate()?;

    // Check if already initialized
    if TRACING_INITIALIZED.swap(true, Ordering::SeqCst) {
        return Err(HpcError::Telemetry(
            "tracing has already been initialized".to_string(),
        ));
    }

    // Create environment filter
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&config.log_level));

    // Build subscriber differently based on whether OTLP is configured
    if let Some(endpoint) = config.otlp_endpoint {
        match setup_otlp_layer(&config.service_name, &endpoint) {
            Ok(Some(otel_layer)) => {
                // Create subscriber with OTLP layer
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(tracing_subscriber::fmt::layer()
                        .with_target(true)
                        .with_line_number(true)
                        .with_thread_ids(true))
                    .with(otel_layer)
                    .try_init()
                    .map_err(|e| {
                        HpcError::Telemetry(format!("failed to initialize tracing: {}", e))
                    })?;
            }
            Ok(None) | Err(_) => {
                // OTLP setup failed but we continue with local logging
                eprintln!(
                    "Warning: Failed to connect to OTLP endpoint {}, continuing with local logging only",
                    endpoint
                );
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(tracing_subscriber::fmt::layer()
                        .with_target(true)
                        .with_line_number(true)
                        .with_thread_ids(true))
                    .try_init()
                    .map_err(|e| {
                        HpcError::Telemetry(format!("failed to initialize tracing: {}", e))
                    })?;
            }
        }
    } else {
        // No OTLP endpoint, just use local logging
        tracing_subscriber::registry()
            .with(env_filter)
            .with(tracing_subscriber::fmt::layer()
                .with_target(true)
                .with_line_number(true)
                .with_thread_ids(true))
            .try_init()
            .map_err(|e| {
                HpcError::Telemetry(format!("failed to initialize tracing: {}", e))
            })?;
    }

    Ok(TracingGuard)
}

/// Sets up the OpenTelemetry layer.
///
/// Returns `Ok(Some(layer))` if successful, `Ok(None)` if the endpoint is unreachable,
/// or `Err` if there's a configuration error.
fn setup_otlp_layer<S>(
    service_name: &str,
    endpoint: &str,
) -> Result<Option<tracing_opentelemetry::OpenTelemetryLayer<S, opentelemetry_sdk::trace::Tracer>>>
where
    S: tracing::Subscriber + for<'span> tracing_subscriber::registry::LookupSpan<'span>,
{
    use opentelemetry::KeyValue;
    use opentelemetry_otlp::WithExportConfig;
    use opentelemetry_sdk::trace::Config;
    use opentelemetry_sdk::Resource;
    use opentelemetry_sdk::runtime;

    // Create resource with service name
    let resource = Resource::new(vec![
        KeyValue::new("service.name", service_name.to_string()),
    ]);

    // Configure trace config
    let trace_config = Config::default().with_resource(resource);

    // Attempt to create OTLP exporter
    let exporter = opentelemetry_otlp::new_exporter()
        .tonic()
        .with_endpoint(endpoint);

    // Build and install the tracer - this returns a Tracer, not TracerProvider
    // The provider is set globally by install_batch
    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(exporter)
        .with_trace_config(trace_config)
        .install_batch(runtime::Tokio)
        .map_err(|e| {
            // If we can't create the pipeline, return error for graceful degradation
            eprintln!("Failed to create OTLP pipeline: {}", e);
            HpcError::Telemetry(format!("OTLP pipeline creation failed: {}", e))
        })?;

    // Create the OpenTelemetry layer
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    Ok(Some(otel_layer))
}

/// Initializes the Prometheus metrics exporter.
///
/// This function starts an HTTP server on the specified address that exposes
/// metrics in Prometheus format at the `/metrics` endpoint.
///
/// ## Thread Safety
///
/// This function uses global state and can only be called once successfully.
/// Subsequent calls will return an error.
///
/// ## Errors
///
/// - `HpcError::Telemetry` if metrics have already been initialized
/// - `HpcError::Telemetry` if the metrics exporter cannot be installed
/// - `HpcError::Network` if the address is already in use
///
/// ## Example
///
/// ```rust,no_run
/// use hpc_tracing::init_metrics;
/// use std::net::SocketAddr;
///
/// fn main() -> anyhow::Result<()> {
///     let addr: SocketAddr = "0.0.0.0:9090".parse()?;
///     init_metrics(addr)?;
///
///     // Record metrics
///     metrics::increment_counter!("requests_total", "endpoint" => "/api");
///     metrics::gauge!("active_connections", 42.0);
///     metrics::histogram!("request_duration_ms", 123.45);
///
///     // Metrics are now available at http://0.0.0.0:9090/metrics
///     Ok(())
/// }
/// ```
pub fn init_metrics(addr: SocketAddr) -> Result<()> {
    // Check if already initialized
    if METRICS_INITIALIZED.swap(true, Ordering::SeqCst) {
        return Err(HpcError::Telemetry(
            "metrics have already been initialized".to_string(),
        ));
    }

    // Try to bind to the address early to catch port-in-use errors
    let listener = std::net::TcpListener::bind(addr)
        .map_err(|e| {
            METRICS_INITIALIZED.store(false, Ordering::SeqCst);
            if e.kind() == std::io::ErrorKind::AddrInUse {
                HpcError::Network(format!("address {} already in use", addr))
            } else {
                HpcError::Network(format!("failed to bind to {}: {}", addr, e))
            }
        })?;

    // Install the Prometheus recorder
    let builder = metrics_exporter_prometheus::PrometheusBuilder::new();
    let handle = builder
        .install_recorder()
        .map_err(|e| {
            METRICS_INITIALIZED.store(false, Ordering::SeqCst);
            HpcError::Telemetry(format!("failed to install metrics recorder: {}", e))
        })?;

    // Spawn the HTTP server in a background task
    tokio::spawn(async move {
        use axum::{Router, routing::get};

        // Create a simple axum app with a /metrics endpoint
        let app = Router::new()
            .route("/metrics", get(move || async move {
                handle.render()
            }));

        // Convert to tokio listener
        let listener = match tokio::net::TcpListener::from_std(listener) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Failed to convert listener: {}", e);
                return;
            }
        };

        // Run the server
        if let Err(e) = axum::serve(listener, app).await {
            eprintln!("Metrics server error: {}", e);
        }
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation_empty_service_name() {
        let config = TracingConfig {
            service_name: String::new(),
            log_level: "info".to_string(),
            otlp_endpoint: None,
        };

        let result = config.validate();
        assert!(result.is_err());
        match result.unwrap_err() {
            HpcError::InvalidInput { field, .. } => {
                assert_eq!(field, "service_name");
            }
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[test]
    fn test_config_validation_invalid_characters() {
        let config = TracingConfig {
            service_name: "invalid service name!".to_string(),
            log_level: "info".to_string(),
            otlp_endpoint: None,
        };

        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_config_validation_invalid_log_level() {
        let config = TracingConfig {
            service_name: "test-service".to_string(),
            log_level: "invalid".to_string(),
            otlp_endpoint: None,
        };

        let result = config.validate();
        assert!(result.is_err());
        match result.unwrap_err() {
            HpcError::InvalidInput { field, .. } => {
                assert_eq!(field, "log_level");
            }
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[test]
    fn test_config_validation_valid() {
        let config = TracingConfig {
            service_name: "test-service".to_string(),
            log_level: "info".to_string(),
            otlp_endpoint: None,
        };

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_guard_is_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<TracingGuard>();
        assert_sync::<TracingGuard>();
    }

    #[test]
    fn test_config_is_clone() {
        let config = TracingConfig {
            service_name: "test".to_string(),
            log_level: "info".to_string(),
            otlp_endpoint: None,
        };
        let _cloned = config.clone();
    }
}
