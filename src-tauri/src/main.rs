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
mod data_pipeline_bridge;
mod edge_proxy_bridge;
mod ephemeral_bridge;
mod events;
mod evolution_bridge;
mod gpu_compiler_bridge;
mod kernel_bridge;
mod nebula_bridge;
mod slai_bridge;
mod state;
mod storage_bridge;
mod tensor_mesh_bridge;
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
    let gpu_compiler = Arc::clone(&app_state.gpu_compiler);

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

            // Initialize GPU compiler in background
            let gpu_compiler_clone = Arc::clone(&gpu_compiler);
            tauri::async_runtime::spawn(async move {
                tracing::info!("Initializing GPU compiler...");
                if let Err(e) = gpu_compiler_clone.initialize().await {
                    tracing::warn!("GPU compiler initialization warning: {}", e);
                } else {
                    let status = gpu_compiler_clone.status().await;
                    tracing::info!(
                        "GPU compiler initialized: backend={}, available={}",
                        status.backend,
                        status.available
                    );
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
            // GPU Compiler commands
            commands::gpu_compiler::init_gpu_compiler,
            commands::gpu_compiler::get_gpu_compiler_status,
            commands::gpu_compiler::gpu_compile,
            commands::gpu_compiler::gpu_compile_quick,
            commands::gpu_compiler::benchmark_gpu_compiler,
            // Evolution commands
            commands::evolution::get_evolution_status,
            commands::evolution::get_adas_metrics,
            commands::evolution::get_dgm_metrics,
            commands::evolution::get_swarm_metrics,
            commands::evolution::get_evolution_events,
            commands::evolution::simulate_evolution_step,
            // Nebula commands (RDMA + ZK + Mesh)
            commands::nebula::get_nebula_status,
            commands::nebula::get_rdma_stats,
            commands::nebula::get_zk_stats,
            commands::nebula::get_mesh_topology,
            commands::nebula::simulate_nebula_activity,
            // Tensor Mesh commands (RMPI + RDMA distributed training)
            commands::tensor_mesh::get_tensor_mesh_status,
            commands::tensor_mesh::get_collective_stats,
            commands::tensor_mesh::get_active_transfers,
            commands::tensor_mesh::get_tensor_nodes,
            commands::tensor_mesh::simulate_tensor_mesh_activity,
            // Data Pipeline commands (GPU-accelerated encryption - Synergy 4)
            commands::data_pipeline::get_data_pipeline_status,
            commands::data_pipeline::get_pipeline_stats,
            commands::data_pipeline::get_pipeline_stages,
            commands::data_pipeline::get_pipeline_jobs,
            commands::data_pipeline::simulate_pipeline_activity,
            // Edge Proxy commands (Vortex + SLAI - Synergy 5)
            commands::edge_proxy::get_edge_proxy_status,
            commands::edge_proxy::get_proxy_status,
            commands::edge_proxy::get_brain_status,
            commands::edge_proxy::get_failure_predictions,
            commands::edge_proxy::get_backend_health,
            commands::edge_proxy::get_routing_decisions,
            commands::edge_proxy::simulate_edge_proxy_activity,
            // Scheduler commands (SLAI GPU scheduling)
            commands::scheduler::scheduler_detect_gpus,
            commands::scheduler::scheduler_list_gpus,
            commands::scheduler::submit_scheduler_job,
            commands::scheduler::get_scheduler_job,
            commands::scheduler::list_scheduler_jobs,
            commands::scheduler::list_jobs_by_tenant,
            commands::scheduler::cancel_scheduler_job,
            commands::scheduler::create_tenant,
            commands::scheduler::get_tenant,
            commands::scheduler::list_tenants,
            commands::scheduler::suspend_tenant,
            commands::scheduler::resume_tenant,
            commands::scheduler::get_scheduler_summary,
            commands::scheduler::get_fair_share,
            // Ephemeral session commands
            commands::ephemeral::create_ephemeral_session,
            commands::ephemeral::get_invite_url,
            commands::ephemeral::join_ephemeral_session,
            commands::ephemeral::get_ephemeral_presence,
            commands::ephemeral::get_ephemeral_session,
            commands::ephemeral::list_ephemeral_sessions,
            commands::ephemeral::end_ephemeral_session,
            commands::ephemeral::leave_ephemeral_session,
            commands::ephemeral::update_ephemeral_activity,
            commands::ephemeral::get_ephemeral_stats,
            commands::ephemeral::cleanup_expired_sessions,
        ])
        .run(tauri::generate_context!())
        .expect("error while running horizon");
}
