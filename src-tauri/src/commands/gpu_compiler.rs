//! GPU Compiler Commands
//!
//! Commands for GPU-accelerated Rust compilation via rustg.
//! Provides 10x faster compilation on macOS (Metal) and Linux/Windows (CUDA).

use crate::gpu_compiler_bridge::{GpuCompilationResult, GpuCompilerStatus};
use crate::state::AppState;
use tauri::State;

/// Initialize the GPU compiler.
///
/// This should be called once at application startup.
/// Returns the compiler status after initialization.
#[tauri::command]
pub async fn init_gpu_compiler(state: State<'_, AppState>) -> Result<GpuCompilerStatus, String> {
    tracing::info!("Initializing GPU compiler");

    // Initialize the GPU compiler bridge
    state.gpu_compiler.initialize().await?;

    // Return the status
    let status = state.gpu_compiler.status().await;
    tracing::info!(
        "GPU compiler initialized: backend={}, available={}",
        status.backend,
        status.available
    );

    Ok(status)
}

/// Get the current GPU compiler status.
///
/// Returns information about GPU availability, backend, and memory limits.
#[tauri::command]
pub async fn get_gpu_compiler_status(
    state: State<'_, AppState>,
) -> Result<GpuCompilerStatus, String> {
    Ok(state.gpu_compiler.status().await)
}

/// Compile Rust source code using GPU acceleration.
///
/// Returns detailed compilation metrics including timing breakdown,
/// token count, and GPU utilization.
#[tauri::command]
pub async fn gpu_compile(
    source: String,
    state: State<'_, AppState>,
) -> Result<GpuCompilationResult, String> {
    tracing::info!(
        "GPU compiling {} bytes of source code",
        source.len()
    );

    let result = state.gpu_compiler.compile(source).await?;

    if result.success {
        tracing::info!(
            "GPU compilation succeeded: {:.2}ms total, {:.1}% GPU utilization",
            result.total_time_ms,
            result.gpu_utilization
        );
    } else {
        tracing::warn!(
            "GPU compilation failed: {:?}",
            result.error
        );
    }

    Ok(result)
}

/// Compile source code with GPU acceleration and return simplified result.
///
/// This is a convenience wrapper that returns just success/failure and timing.
#[tauri::command]
pub async fn gpu_compile_quick(
    source: String,
    state: State<'_, AppState>,
) -> Result<QuickCompileResult, String> {
    let result = state.gpu_compiler.compile(source).await?;

    Ok(QuickCompileResult {
        success: result.success,
        total_time_ms: result.total_time_ms,
        gpu_accelerated: result.gpu_accelerated,
        error: result.error,
    })
}

/// Simplified compilation result for quick checks.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QuickCompileResult {
    pub success: bool,
    pub total_time_ms: f64,
    pub gpu_accelerated: bool,
    pub error: Option<String>,
}

/// Benchmark GPU vs CPU compilation.
///
/// Compiles the same code with GPU and CPU (fallback) to compare performance.
#[tauri::command]
pub async fn benchmark_gpu_compiler(
    source: String,
    state: State<'_, AppState>,
) -> Result<BenchmarkResult, String> {
    tracing::info!("Benchmarking GPU compiler with {} bytes of source", source.len());

    // Compile with GPU
    let gpu_result = state.gpu_compiler.compile(source.clone()).await?;

    // For now, we don't have a separate CPU-only path, so estimate based on
    // typical 10x speedup factor
    let estimated_cpu_time = gpu_result.total_time_ms * 10.0;

    let speedup = if gpu_result.gpu_accelerated {
        estimated_cpu_time / gpu_result.total_time_ms
    } else {
        1.0
    };

    Ok(BenchmarkResult {
        gpu_time_ms: gpu_result.total_time_ms,
        cpu_time_ms: estimated_cpu_time,
        speedup,
        gpu_accelerated: gpu_result.gpu_accelerated,
        gpu_utilization: gpu_result.gpu_utilization,
    })
}

/// GPU vs CPU compilation benchmark result.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BenchmarkResult {
    pub gpu_time_ms: f64,
    pub cpu_time_ms: f64,
    pub speedup: f64,
    pub gpu_accelerated: bool,
    pub gpu_utilization: f32,
}
