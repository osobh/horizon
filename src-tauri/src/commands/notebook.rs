//! Notebook Kernel Commands
//!
//! Commands for executing Rust code in the embedded notebook kernel.

use crate::kernel_bridge::OutputType as BridgeOutputType;
use crate::state::{AppState, VariableInfo};
use serde::{Deserialize, Serialize};
use tauri::State;

#[derive(Debug, Serialize, Deserialize)]
pub struct CellOutput {
    pub output_type: OutputType,
    pub content: String,
    pub execution_count: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputType {
    Stdout,
    Stderr,
    Result,
    Error,
    Display,
}

impl From<BridgeOutputType> for OutputType {
    fn from(ot: BridgeOutputType) -> Self {
        match ot {
            BridgeOutputType::Stdout => OutputType::Stdout,
            BridgeOutputType::Stderr => OutputType::Stderr,
            BridgeOutputType::Result => OutputType::Result,
            BridgeOutputType::Error => OutputType::Error,
            BridgeOutputType::Display => OutputType::Display,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub success: bool,
    pub outputs: Vec<CellOutput>,
    pub execution_count: u64,
    pub duration_ms: u64,
}

/// Execute a code cell in the notebook kernel.
#[tauri::command]
pub async fn execute_cell(code: String, state: State<'_, AppState>) -> Result<ExecutionResult, String> {
    tracing::info!("Executing cell: {}", &code[..code.len().min(50)]);

    // Execute via the kernel bridge
    let output = state.kernel.execute(code.clone(), false).await?;

    // Update execution count in notebook state
    let mut notebook = state.notebook.write().await;
    notebook.execution_count = output.execution_count.max(notebook.execution_count + 1);
    let count = notebook.execution_count;

    // Convert outputs
    let outputs: Vec<CellOutput> = output
        .outputs
        .into_iter()
        .map(|o| CellOutput {
            output_type: o.output_type.into(),
            content: o.content,
            execution_count: count,
        })
        .collect();

    // Extract variables from code (simple heuristic for let bindings)
    if code.contains("let ") {
        let var_name = code
            .split("let ")
            .nth(1)
            .and_then(|s| s.split([':', '=', ' ']).next())
            .unwrap_or("x");

        // Check if variable already exists
        if !notebook.variables.iter().any(|v| v.name == var_name) {
            notebook.variables.push(VariableInfo {
                name: var_name.to_string(),
                var_type: "inferred".to_string(),
                size_bytes: 0,
                preview: "...".to_string(),
            });
        }
    }

    Ok(ExecutionResult {
        success: output.success,
        outputs,
        execution_count: count,
        duration_ms: output.duration_ms,
    })
}

/// Get all variables currently in scope.
#[tauri::command]
pub async fn get_variables(state: State<'_, AppState>) -> Result<Vec<VariableInfo>, String> {
    let notebook = state.notebook.read().await;
    Ok(notebook.variables.clone())
}

/// Restart the notebook kernel.
#[tauri::command]
pub async fn restart_kernel(state: State<'_, AppState>) -> Result<(), String> {
    tracing::info!("Restarting kernel");

    // Restart the kernel via bridge
    state.kernel.restart().await?;

    // Clear notebook state
    let mut notebook = state.notebook.write().await;
    notebook.execution_count = 0;
    notebook.variables.clear();
    notebook.kernel_running = true;

    Ok(())
}

/// Interrupt the current execution.
#[tauri::command]
pub async fn interrupt_kernel(_state: State<'_, AppState>) -> Result<(), String> {
    // TODO: Implement interrupt via kernel bridge
    tracing::info!("Interrupting kernel execution");
    Ok(())
}
