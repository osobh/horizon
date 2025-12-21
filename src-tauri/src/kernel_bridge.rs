//! Kernel Bridge
//!
//! Integrates the notebook-kernel crate with Horizon using hpc-channels.

#[cfg(feature = "embedded-kernel")]
use notebook_kernel::{
    ExecuteRequest, ExecuteResult as KernelExecuteResult, ExecutorConfig, KernelExecutor,
};

use hpc_channels::{channels, KernelMessage};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Bridge to the embedded notebook kernel.
pub struct KernelBridge {
    #[cfg(feature = "embedded-kernel")]
    executor: Arc<RwLock<Option<KernelExecutor>>>,
    #[cfg(not(feature = "embedded-kernel"))]
    _phantom: std::marker::PhantomData<()>,
}

impl KernelBridge {
    /// Create a new kernel bridge.
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "embedded-kernel")]
            executor: Arc::new(RwLock::new(None)),
            #[cfg(not(feature = "embedded-kernel"))]
            _phantom: std::marker::PhantomData,
        }
    }

    /// Initialize the kernel.
    #[cfg(feature = "embedded-kernel")]
    pub async fn initialize(&self) -> Result<(), String> {
        let config = ExecutorConfig::default();
        match KernelExecutor::new(config).await {
            Ok(executor) => {
                let mut guard = self.executor.write().await;
                *guard = Some(executor);
                tracing::info!("Kernel initialized successfully");
                Ok(())
            }
            Err(e) => {
                tracing::error!("Failed to initialize kernel: {:?}", e);
                Err(format!("Failed to initialize kernel: {:?}", e))
            }
        }
    }

    #[cfg(not(feature = "embedded-kernel"))]
    pub async fn initialize(&self) -> Result<(), String> {
        tracing::warn!("Kernel feature not enabled, using mock execution");
        Ok(())
    }

    /// Execute code in the kernel.
    #[cfg(feature = "embedded-kernel")]
    pub async fn execute(&self, code: String, silent: bool) -> Result<ExecutionOutput, String> {
        let guard = self.executor.read().await;
        let executor = guard
            .as_ref()
            .ok_or_else(|| "Kernel not initialized".to_string())?;

        let request = ExecuteRequest {
            code,
            silent,
            store_history: !silent,
            user_expressions: std::collections::HashMap::new(),
            allow_stdin: false,
        };

        match executor.execute(request).await {
            Ok(result) => Ok(ExecutionOutput::from_kernel_result(result)),
            Err(e) => {
                tracing::error!("Execution failed: {:?}", e);
                Ok(ExecutionOutput {
                    success: false,
                    outputs: vec![Output {
                        output_type: OutputType::Error,
                        content: format!("{:?}", e),
                    }],
                    execution_count: 0,
                    duration_ms: 0,
                })
            }
        }
    }

    #[cfg(not(feature = "embedded-kernel"))]
    pub async fn execute(&self, code: String, _silent: bool) -> Result<ExecutionOutput, String> {
        // Mock execution when kernel feature is disabled
        tracing::info!("Mock execution: {}", &code[..code.len().min(50)]);

        let outputs = if code.contains("println!") {
            vec![Output {
                output_type: OutputType::Stdout,
                content: "Hello from Horizon! (mock)\n".to_string(),
            }]
        } else {
            vec![Output {
                output_type: OutputType::Result,
                content: "()".to_string(),
            }]
        };

        Ok(ExecutionOutput {
            success: true,
            outputs,
            execution_count: 1,
            duration_ms: 5,
        })
    }

    /// Restart the kernel.
    #[cfg(feature = "embedded-kernel")]
    pub async fn restart(&self) -> Result<(), String> {
        let guard = self.executor.read().await;
        if let Some(executor) = guard.as_ref() {
            executor
                .restart()
                .await
                .map_err(|e| format!("Failed to restart kernel: {:?}", e))?;
            tracing::info!("Kernel restarted");
        }
        Ok(())
    }

    #[cfg(not(feature = "embedded-kernel"))]
    pub async fn restart(&self) -> Result<(), String> {
        tracing::info!("Mock kernel restart");
        Ok(())
    }

    /// Check if the kernel is initialized.
    #[allow(dead_code)]
    #[cfg(feature = "embedded-kernel")]
    pub async fn is_initialized(&self) -> bool {
        self.executor.read().await.is_some()
    }

    #[allow(dead_code)]
    #[cfg(not(feature = "embedded-kernel"))]
    pub async fn is_initialized(&self) -> bool {
        true // Mock is always "initialized"
    }
}

impl Default for KernelBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Output from kernel execution.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExecutionOutput {
    pub success: bool,
    pub outputs: Vec<Output>,
    pub execution_count: u64,
    pub duration_ms: u64,
}

#[cfg(feature = "embedded-kernel")]
impl ExecutionOutput {
    fn from_kernel_result(result: KernelExecuteResult) -> Self {
        let mut outputs = Vec::new();

        if let Some(stdout) = result.stdout {
            if !stdout.is_empty() {
                outputs.push(Output {
                    output_type: OutputType::Stdout,
                    content: stdout,
                });
            }
        }

        if let Some(stderr) = result.stderr {
            if !stderr.is_empty() {
                outputs.push(Output {
                    output_type: OutputType::Stderr,
                    content: stderr,
                });
            }
        }

        // If no outputs, add a default result
        if outputs.is_empty() && result.status == "ok" {
            outputs.push(Output {
                output_type: OutputType::Result,
                content: "()".to_string(),
            });
        }

        Self {
            success: result.status == "ok",
            outputs,
            execution_count: result.execution_count,
            duration_ms: result.execution_time_ms.unwrap_or(0),
        }
    }
}

/// Single output item.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Output {
    pub output_type: OutputType,
    pub content: String,
}

/// Type of output.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputType {
    Stdout,
    Stderr,
    Result,
    Error,
    Display,
}

/// Start the kernel message handler task.
///
/// This spawns a background task that listens for KernelMessage events
/// on the hpc-channels and dispatches them to the kernel.
#[allow(dead_code)]
pub async fn start_kernel_handler(bridge: Arc<KernelBridge>) {
    // Create kernel channels
    let (_tx, mut rx) = hpc_channels::channel::<KernelMessage>(channels::KERNEL_EXECUTE);
    let output_tx = hpc_channels::broadcast::<KernelMessage>(channels::KERNEL_OUTPUT, 256);

    tracing::info!("Kernel handler started on channel: {}", channels::KERNEL_EXECUTE);

    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            match msg {
                KernelMessage::Execute { code, cell_id } => {
                    tracing::debug!("Received execute request for cell: {}", cell_id);

                    match bridge.execute(code.clone(), false).await {
                        Ok(output) => {
                            // Send output back via broadcast channel
                            for o in &output.outputs {
                                let stream = match o.output_type {
                                    OutputType::Stdout => {
                                        hpc_channels::OutputStream::Stdout
                                    }
                                    OutputType::Stderr => {
                                        hpc_channels::OutputStream::Stderr
                                    }
                                    _ => hpc_channels::OutputStream::Display,
                                };

                                let _ = output_tx.send(KernelMessage::Output {
                                    cell_id: cell_id.clone(),
                                    content: o.content.clone(),
                                    stream,
                                });
                            }

                            // Send completion message
                            let _ = output_tx.send(KernelMessage::ExecutionComplete {
                                cell_id: cell_id.clone(),
                                success: output.success,
                                duration_ms: output.duration_ms,
                            });
                        }
                        Err(e) => {
                            let _ = output_tx.send(KernelMessage::Output {
                                cell_id: cell_id.clone(),
                                content: e,
                                stream: hpc_channels::OutputStream::Stderr,
                            });
                        }
                    }
                }
                KernelMessage::Interrupt => {
                    tracing::info!("Kernel interrupt requested");
                    // TODO: Implement interrupt
                }
                KernelMessage::Restart => {
                    tracing::info!("Kernel restart requested");
                    if let Err(e) = bridge.restart().await {
                        tracing::error!("Restart failed: {}", e);
                    }
                }
                _ => {}
            }
        }
    });
}
