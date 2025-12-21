//! Horizon - Unified HPC-AI Platform
//!
//! A Tauri-based desktop application providing a single pane of glass
//! to the entire HPC-AI stack.

#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod cluster_bridge;
mod commands;
mod events;
mod kernel_bridge;
mod state;
mod storage_bridge;
mod training_bridge;

use events::MetricsCollector;
use state::AppState;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    tracing::info!("Starting Horizon - HPC-AI Unified Platform");

    // Create app state
    let app_state = AppState::new();
    let kernel = Arc::clone(&app_state.kernel);

    // Create metrics collector (updates every 2 seconds)
    let metrics_collector = Arc::new(MetricsCollector::new(2000));

    tauri::Builder::default()
        .manage(app_state)
        .manage(metrics_collector.clone())
        .setup(move |app| {
            let app_handle = app.handle().clone();

            // Initialize kernel in background
            let kernel_clone = Arc::clone(&kernel);
            tauri::async_runtime::spawn(async move {
                tracing::info!("Initializing embedded kernel...");
                if let Err(e) = kernel_clone.initialize().await {
                    tracing::error!("Failed to initialize kernel: {}", e);
                } else {
                    tracing::info!("Kernel initialized successfully");
                }
            });

            // Start metrics collection
            let metrics = Arc::clone(&metrics_collector);
            tauri::async_runtime::spawn(async move {
                metrics.start(app_handle).await;
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // Cluster commands
            commands::cluster::get_cluster_status,
            commands::cluster::list_nodes,
            commands::cluster::get_node,
            commands::cluster::get_cluster_stats,
            commands::cluster::connect_cluster,
            commands::cluster::disconnect_cluster,
            // Notebook commands
            commands::notebook::execute_cell,
            commands::notebook::get_variables,
            commands::notebook::restart_kernel,
            commands::notebook::interrupt_kernel,
            // Training commands
            commands::training::start_training,
            commands::training::get_training_status,
            commands::training::list_training_jobs,
            commands::training::get_training_summary,
            commands::training::pause_training,
            commands::training::resume_training,
            commands::training::cancel_training,
            // Storage commands
            commands::storage::upload_file,
            commands::storage::download_file,
            commands::storage::get_transfer,
            commands::storage::list_transfers,
            commands::storage::list_active_transfers,
            commands::storage::get_storage_stats,
            commands::storage::pause_transfer,
            commands::storage::resume_transfer,
            commands::storage::cancel_transfer,
            commands::storage::list_files,
            // System commands
            commands::system::detect_gpus,
            commands::system::get_system_info,
        ])
        .run(tauri::generate_context!())
        .expect("error while running horizon");
}
