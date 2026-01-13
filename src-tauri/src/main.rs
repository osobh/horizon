//! Horizon - Unified HPC-AI Platform
//!
//! A Tauri-based desktop application providing a single pane of glass
//! to the entire HPC-AI stack.

#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod argus_bridge;
mod cluster_bridge;
mod hpcci_bridge;
mod rustyspark_bridge;
mod commands;
mod costs_bridge;
mod data_pipeline_bridge;
mod edge_proxy_bridge;
mod ephemeral_bridge;
mod error;
mod events;
mod evolution_bridge;
mod gpu_compiler_bridge;
mod intelligence_bridge;
mod kernel_bridge;
mod nebula_bridge;
mod settings_bridge;
mod slai_bridge;
mod state;
mod storage_bridge;
mod stratoswarm_bridge;
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
    let training = Arc::clone(&app_state.training);
    let nebula = Arc::clone(&app_state.nebula);
    let tensor_mesh = Arc::clone(&app_state.tensor_mesh);
    let storage = Arc::clone(&app_state.storage);
    let ephemeral = Arc::clone(&app_state.ephemeral);
    let slai = Arc::clone(&app_state.slai);
    let argus = Arc::clone(&app_state.argus);
    let hpcci = Arc::clone(&app_state.hpcci);
    let rustyspark = Arc::clone(&app_state.rustyspark);
    let stratoswarm = Arc::clone(&app_state.stratoswarm);

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

            // Initialize training bridge in background
            let training_clone = Arc::clone(&training);
            tauri::async_runtime::spawn(async move {
                tracing::info!("Initializing training bridge...");
                if let Err(e) = training_clone.initialize().await {
                    tracing::error!("Failed to initialize training: {}", e);
                } else {
                    tracing::info!("Training bridge initialized successfully");
                }
            });

            // Initialize nebula bridge in background
            let nebula_clone = Arc::clone(&nebula);
            tauri::async_runtime::spawn(async move {
                tracing::info!("Initializing nebula bridge...");
                if let Err(e) = nebula_clone.initialize().await {
                    tracing::error!("Failed to initialize nebula: {}", e);
                } else {
                    tracing::info!("Nebula bridge initialized successfully");
                }
            });

            // Initialize tensor mesh bridge in background
            let tensor_mesh_clone = Arc::clone(&tensor_mesh);
            tauri::async_runtime::spawn(async move {
                tracing::info!("Initializing tensor mesh bridge...");
                if let Err(e) = tensor_mesh_clone.initialize().await {
                    tracing::error!("Failed to initialize tensor mesh: {}", e);
                } else {
                    tracing::info!("Tensor mesh bridge initialized successfully");
                }
            });

            // Initialize storage bridge in background
            let storage_clone = Arc::clone(&storage);
            tauri::async_runtime::spawn(async move {
                tracing::info!("Initializing storage bridge...");
                if let Err(e) = storage_clone.initialize().await {
                    tracing::error!("Failed to initialize storage: {}", e);
                } else {
                    tracing::info!("Storage bridge initialized successfully");
                }
            });

            // Initialize SLAI bridge in background
            let slai_clone = Arc::clone(&slai);
            tauri::async_runtime::spawn(async move {
                tracing::info!("Initializing SLAI bridge...");
                if let Err(e) = slai_clone.initialize().await {
                    tracing::error!("Failed to initialize SLAI: {}", e);
                } else {
                    tracing::info!("SLAI bridge initialized successfully");
                }
            });

            // Initialize Argus bridge in background
            let argus_clone = Arc::clone(&argus);
            tauri::async_runtime::spawn(async move {
                tracing::info!("Initializing Argus bridge...");
                if let Err(e) = argus_clone.initialize().await {
                    tracing::error!("Failed to initialize Argus: {}", e);
                } else {
                    tracing::info!("Argus bridge initialized successfully");
                }
            });

            // Initialize HPC-CI bridge in background
            let hpcci_clone = Arc::clone(&hpcci);
            tauri::async_runtime::spawn(async move {
                tracing::info!("Initializing HPC-CI bridge...");
                if let Err(e) = hpcci_clone.initialize().await {
                    tracing::error!("Failed to initialize HPC-CI: {}", e);
                } else {
                    tracing::info!("HPC-CI bridge initialized successfully");
                }
            });

            // Initialize RustySpark bridge in background
            let rustyspark_clone = Arc::clone(&rustyspark);
            tauri::async_runtime::spawn(async move {
                tracing::info!("Initializing RustySpark bridge...");
                if let Err(e) = rustyspark_clone.initialize().await {
                    tracing::error!("Failed to initialize RustySpark: {}", e);
                } else {
                    tracing::info!("RustySpark bridge initialized successfully");
                }
            });

            // Initialize StratoSwarm bridge in background
            let stratoswarm_clone = Arc::clone(&stratoswarm);
            tauri::async_runtime::spawn(async move {
                tracing::info!("Initializing StratoSwarm bridge...");
                if let Err(e) = stratoswarm_clone.initialize().await {
                    tracing::error!("Failed to initialize StratoSwarm: {}", e);
                } else {
                    tracing::info!("StratoSwarm bridge initialized successfully");
                }
            });

            // Start metrics collection
            let metrics = Arc::clone(&metrics_collector);
            tauri::async_runtime::spawn(async move {
                metrics.start(app_handle).await;
            });

            // Start ephemeral session cleanup worker (runs every 60 seconds)
            let ephemeral_clone = Arc::clone(&ephemeral);
            tauri::async_runtime::spawn(async move {
                loop {
                    tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
                    let (sessions, invites) = ephemeral_clone.cleanup_expired().await;
                    if sessions > 0 || invites > 0 {
                        tracing::info!(
                            "Ephemeral cleanup: {} sessions, {} invites expired",
                            sessions,
                            invites
                        );
                    }
                }
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
            // Ephemeral commands (Time-limited collaboration)
            commands::ephemeral::list_ephemeral_sessions,
            commands::ephemeral::get_ephemeral_session,
            commands::ephemeral::create_ephemeral_session,
            commands::ephemeral::end_ephemeral_session,
            commands::ephemeral::get_invite_url,
            commands::ephemeral::join_ephemeral_session,
            commands::ephemeral::leave_ephemeral_session,
            commands::ephemeral::get_ephemeral_presence,
            commands::ephemeral::update_ephemeral_activity,
            commands::ephemeral::get_ephemeral_stats,
            commands::ephemeral::cleanup_expired_sessions,
            // SLAI commands (GPU scheduler and job management)
            commands::slai::get_slai_stats,
            commands::slai::get_slai_gpus,
            commands::slai::get_slai_fair_share,
            commands::slai::list_slai_tenants,
            commands::slai::create_slai_tenant,
            commands::slai::list_slai_jobs,
            commands::slai::submit_slai_job,
            commands::slai::cancel_slai_job,
            commands::slai::schedule_slai_next,
            // Argus commands (observability, metrics, alerts)
            commands::argus::get_argus_status,
            commands::argus::set_argus_server_url,
            commands::argus::query_argus_metrics,
            commands::argus::query_argus_metrics_range,
            commands::argus::get_argus_alerts,
            commands::argus::get_argus_targets,
            // HPC-CI commands (pipeline management)
            commands::hpcci::get_hpcci_status,
            commands::hpcci::set_hpcci_server_url,
            commands::hpcci::list_hpcci_pipelines,
            commands::hpcci::get_hpcci_pipeline,
            commands::hpcci::trigger_hpcci_pipeline,
            commands::hpcci::cancel_hpcci_pipeline,
            commands::hpcci::retry_hpcci_pipeline,
            commands::hpcci::list_hpcci_agents,
            commands::hpcci::drain_hpcci_agent,
            commands::hpcci::enable_hpcci_agent,
            commands::hpcci::get_hpcci_approvals,
            commands::hpcci::submit_hpcci_approval,
            commands::hpcci::get_hpcci_dashboard_summary,
            commands::hpcci::get_hpcci_pipeline_logs,
            // RustySpark commands (data processing jobs)
            commands::rustyspark::get_rustyspark_status,
            commands::rustyspark::set_rustyspark_server_url,
            commands::rustyspark::get_rustyspark_summary,
            commands::rustyspark::list_spark_jobs,
            commands::rustyspark::get_spark_job,
            commands::rustyspark::get_spark_job_stages,
            commands::rustyspark::get_spark_stage_tasks,
            commands::rustyspark::cancel_spark_job,
            // Cost Intelligence commands
            commands::costs::get_cost_summary,
            commands::costs::get_cost_attributions,
            commands::costs::get_cost_forecasts,
            commands::costs::get_budget_alerts,
            commands::costs::list_cost_reports,
            commands::costs::generate_chargeback_report,
            commands::costs::generate_showback_report,
            commands::costs::set_budget_threshold,
            // Intelligence commands
            commands::intelligence::get_intelligence_summary,
            commands::intelligence::get_idle_resources,
            commands::intelligence::get_profit_margins,
            commands::intelligence::get_vendor_utilizations,
            commands::intelligence::get_executive_kpis,
            commands::intelligence::get_intelligence_alerts,
            commands::intelligence::acknowledge_intelligence_alert,
            commands::intelligence::terminate_idle_resource,
            // Settings commands
            commands::settings::get_settings_summary,
            commands::settings::get_policies,
            commands::settings::get_quotas,
            commands::settings::get_app_settings,
            commands::settings::create_policy,
            commands::settings::update_policy,
            commands::settings::delete_policy,
            commands::settings::toggle_policy,
            commands::settings::set_quota,
            commands::settings::update_quota,
            commands::settings::delete_quota,
            commands::settings::update_app_settings,
            // StratoSwarm commands (agent visualization and evolution)
            commands::stratoswarm::get_stratoswarm_status,
            commands::stratoswarm::set_stratoswarm_cluster_url,
            commands::stratoswarm::get_swarm_stats,
            commands::stratoswarm::list_swarm_agents,
            commands::stratoswarm::get_swarm_agent,
            commands::stratoswarm::get_swarm_evolution_events,
            commands::stratoswarm::get_active_agent_tasks,
            commands::stratoswarm::trigger_agent_evolution,
            commands::stratoswarm::simulate_swarm_activity,
        ])
        .run(tauri::generate_context!())
        .expect("error while running horizon");
}
