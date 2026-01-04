//! API server for agent health, metrics, and management endpoints

use std::convert::Infallible;
use std::net::SocketAddr;
use tracing::{debug, info};
use uuid::Uuid;
use warp::Filter;

use crate::build_job::BuildJob;
use crate::profile::HardwareProfiler;
use crate::wireguard::{AddPeerRequest, RemovePeerRequest, WireGuardConfigRequest};
use crate::Result;

use super::SwarmletAgent;

impl SwarmletAgent {
    /// API server loop - provides local HTTP API for health checks and metrics
    pub(super) async fn api_server_loop(&self) -> Result<()> {
        // Create API routes
        let health_status = self.health_status.clone();
        let command_executor = self.command_executor.clone();
        let mut shutdown_signal = self.shutdown_signal.clone();

        let health_route = warp::path("health").and(warp::get()).and_then(move || {
            let health_status = health_status.clone();
            async move {
                let health = health_status.read().await;
                let response = serde_json::to_string(&*health)
                    .unwrap_or_else(|_| r#"{"error": "serialization_failed"}"#.to_string());

                Ok::<_, Infallible>(warp::reply::with_header(
                    warp::reply::html(response),
                    "content-type",
                    "application/json",
                ))
            }
        });

        let metrics_route = warp::path("metrics").and(warp::get()).map(|| {
            // Return Prometheus-style metrics
            format!(
                "# HELP swarmlet_uptime_seconds Uptime in seconds\n\
                     # TYPE swarmlet_uptime_seconds counter\n\
                     swarmlet_uptime_seconds {}\n",
                chrono::Utc::now().timestamp()
            )
        });

        // Command execution routes
        let execute_command_route = warp::path!("api" / "v1" / "execute")
            .and(warp::post())
            .and(warp::body::json())
            .and_then(move |request: crate::command::CommandRequest| {
                let command_executor = command_executor.clone();
                async move {
                    match command_executor.execute_command(request).await {
                        Ok(result) => {
                            let json = serde_json::to_string(&result).unwrap_or_else(|_| {
                                r#"{"error": "serialization_failed"}"#.to_string()
                            });
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(json, warp::http::StatusCode::OK),
                                "content-type",
                                "application/json",
                            ))
                        }
                        Err(e) => {
                            let error_response = format!(r#"{{"error": "{e}"}}"#);
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::BAD_REQUEST,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        let command_executor_shell = self.command_executor.clone();
        let execute_shell_route = warp::path!("api" / "v1" / "shell")
            .and(warp::post())
            .and(warp::body::json())
            .and_then(move |request: serde_json::Value| {
                let command_executor = command_executor_shell.clone();
                async move {
                    if let Some(script) = request.get("script").and_then(|s| s.as_str()) {
                        match command_executor.execute_shell(script).await {
                            Ok(result) => {
                                let json = serde_json::to_string(&result).unwrap_or_else(|_| {
                                    r#"{"error": "serialization_failed"}"#.to_string()
                                });
                                Ok::<_, Infallible>(warp::reply::with_header(
                                    warp::reply::with_status(json, warp::http::StatusCode::OK),
                                    "content-type",
                                    "application/json",
                                ))
                            }
                            Err(e) => {
                                let error_response = format!(r#"{{"error": "{e}"}}"#);
                                Ok::<_, Infallible>(warp::reply::with_header(
                                    warp::reply::with_status(
                                        error_response,
                                        warp::http::StatusCode::BAD_REQUEST,
                                    ),
                                    "content-type",
                                    "application/json",
                                ))
                            }
                        }
                    } else {
                        let error_response = r#"{"error": "Missing 'script' field"}"#.to_string();
                        Ok::<_, Infallible>(warp::reply::with_header(
                            warp::reply::with_status(
                                error_response,
                                warp::http::StatusCode::BAD_REQUEST,
                            ),
                            "content-type",
                            "application/json",
                        ))
                    }
                }
            });

        // WireGuard routes
        let wg_manager_configure = self.wireguard_manager.clone();
        let wireguard_configure_route = warp::path!("api" / "v1" / "wireguard" / "configure")
            .and(warp::post())
            .and(warp::body::json())
            .and_then(move |request: WireGuardConfigRequest| {
                let wg_manager = wg_manager_configure.clone();
                async move {
                    match wg_manager.apply_config(request).await {
                        Ok(response) => {
                            let json = serde_json::to_string(&response).unwrap_or_else(|_| {
                                r#"{"error": "serialization_failed"}"#.to_string()
                            });
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(json, warp::http::StatusCode::OK),
                                "content-type",
                                "application/json",
                            ))
                        }
                        Err(e) => {
                            let error_response = format!(r#"{{"error": "{}"}}"#, e);
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        let wg_manager_status = self.wireguard_manager.clone();
        let wireguard_status_route = warp::path!("api" / "v1" / "wireguard" / "status" / String)
            .and(warp::get())
            .and_then(move |interface_name: String| {
                let wg_manager = wg_manager_status.clone();
                async move {
                    match wg_manager.get_status(&interface_name).await {
                        Ok(status) => {
                            let json = serde_json::to_string(&status).unwrap_or_else(|_| {
                                r#"{"error": "serialization_failed"}"#.to_string()
                            });
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(json, warp::http::StatusCode::OK),
                                "content-type",
                                "application/json",
                            ))
                        }
                        Err(e) => {
                            let error_response = format!(r#"{{"error": "{}"}}"#, e);
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        let wg_manager_add_peer = self.wireguard_manager.clone();
        let wireguard_add_peer_route = warp::path!("api" / "v1" / "wireguard" / "peer")
            .and(warp::post())
            .and(warp::body::json())
            .and_then(move |request: AddPeerRequest| {
                let wg_manager = wg_manager_add_peer.clone();
                async move {
                    match wg_manager.add_peer(request).await {
                        Ok(()) => {
                            let json = r#"{"success": true}"#.to_string();
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(json, warp::http::StatusCode::OK),
                                "content-type",
                                "application/json",
                            ))
                        }
                        Err(e) => {
                            let error_response = format!(r#"{{"error": "{}"}}"#, e);
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        let wg_manager_remove_peer = self.wireguard_manager.clone();
        let wireguard_remove_peer_route = warp::path!("api" / "v1" / "wireguard" / "peer")
            .and(warp::delete())
            .and(warp::body::json())
            .and_then(move |request: RemovePeerRequest| {
                let wg_manager = wg_manager_remove_peer.clone();
                async move {
                    match wg_manager.remove_peer(request).await {
                        Ok(()) => {
                            let json = r#"{"success": true}"#.to_string();
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(json, warp::http::StatusCode::OK),
                                "content-type",
                                "application/json",
                            ))
                        }
                        Err(e) => {
                            let error_response = format!(r#"{{"error": "{}"}}"#, e);
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        let wg_manager_list_peers = self.wireguard_manager.clone();
        let wireguard_list_peers_route = warp::path!("api" / "v1" / "wireguard" / "peers" / String)
            .and(warp::get())
            .and_then(move |interface_name: String| {
                let wg_manager = wg_manager_list_peers.clone();
                async move {
                    match wg_manager.list_peers(&interface_name).await {
                        Ok(peers) => {
                            let json = serde_json::to_string(&peers).unwrap_or_else(|_| {
                                r#"{"error": "serialization_failed"}"#.to_string()
                            });
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(json, warp::http::StatusCode::OK),
                                "content-type",
                                "application/json",
                            ))
                        }
                        Err(e) => {
                            let error_response = format!(r#"{{"error": "{}"}}"#, e);
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        // Workload routes
        let workload_manager_list = self.workload_manager.clone();
        let workloads_list_route = warp::path!("api" / "v1" / "workloads")
            .and(warp::get())
            .and_then(move || {
                let wm = workload_manager_list.clone();
                async move {
                    let workloads = wm.get_active_workloads().await;
                    let json = serde_json::to_string(&workloads)
                        .unwrap_or_else(|_| r#"{"error": "serialization_failed"}"#.to_string());
                    Ok::<_, Infallible>(warp::reply::with_header(
                        warp::reply::with_status(json, warp::http::StatusCode::OK),
                        "content-type",
                        "application/json",
                    ))
                }
            });

        let workload_manager_stop = self.workload_manager.clone();
        let workloads_stop_route = warp::path!("api" / "v1" / "workloads" / String / "stop")
            .and(warp::post())
            .and_then(move |workload_id: String| {
                let wm = workload_manager_stop.clone();
                async move {
                    match Uuid::parse_str(&workload_id) {
                        Ok(id) => match wm.stop_workload(id).await {
                            Ok(()) => {
                                let json = r#"{"success": true}"#.to_string();
                                Ok::<_, Infallible>(warp::reply::with_header(
                                    warp::reply::with_status(json, warp::http::StatusCode::OK),
                                    "content-type",
                                    "application/json",
                                ))
                            }
                            Err(e) => {
                                let error_response = format!(r#"{{"error": "{}"}}"#, e);
                                Ok::<_, Infallible>(warp::reply::with_header(
                                    warp::reply::with_status(
                                        error_response,
                                        warp::http::StatusCode::NOT_FOUND,
                                    ),
                                    "content-type",
                                    "application/json",
                                ))
                            }
                        },
                        Err(_) => {
                            let error_response = r#"{"error": "invalid_workload_id"}"#.to_string();
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::BAD_REQUEST,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        // Build job routes
        let build_manager_submit = self.build_job_manager.clone();
        let builds_submit_route = warp::path!("api" / "v1" / "builds")
            .and(warp::post())
            .and(warp::body::json())
            .and_then(move |job: BuildJob| {
                let bm = build_manager_submit.clone();
                async move {
                    match bm.submit_build(job).await {
                        Ok(job_id) => {
                            let json = format!(r#"{{"job_id": "{}"}}"#, job_id);
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(json, warp::http::StatusCode::ACCEPTED),
                                "content-type",
                                "application/json",
                            ))
                        }
                        Err(e) => {
                            let error_response = format!(r#"{{"error": "{}"}}"#, e);
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::BAD_REQUEST,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        let build_manager_list = self.build_job_manager.clone();
        let builds_list_route = warp::path!("api" / "v1" / "builds")
            .and(warp::get())
            .and_then(move || {
                let bm = build_manager_list.clone();
                async move {
                    let jobs = bm.list_active_jobs().await;
                    let json = serde_json::to_string(&jobs)
                        .unwrap_or_else(|_| r#"{"error": "serialization_failed"}"#.to_string());
                    Ok::<_, Infallible>(warp::reply::with_header(
                        warp::reply::with_status(json, warp::http::StatusCode::OK),
                        "content-type",
                        "application/json",
                    ))
                }
            });

        let build_manager_get = self.build_job_manager.clone();
        let builds_get_route = warp::path!("api" / "v1" / "builds" / String)
            .and(warp::get())
            .and_then(move |job_id: String| {
                let bm = build_manager_get.clone();
                async move {
                    match Uuid::parse_str(&job_id) {
                        Ok(id) => match bm.get_job(id).await {
                            Some(job) => {
                                let json = serde_json::to_string(&job).unwrap_or_else(|_| {
                                    r#"{"error": "serialization_failed"}"#.to_string()
                                });
                                Ok::<_, Infallible>(warp::reply::with_header(
                                    warp::reply::with_status(json, warp::http::StatusCode::OK),
                                    "content-type",
                                    "application/json",
                                ))
                            }
                            None => {
                                let error_response = r#"{"error": "job_not_found"}"#.to_string();
                                Ok::<_, Infallible>(warp::reply::with_header(
                                    warp::reply::with_status(
                                        error_response,
                                        warp::http::StatusCode::NOT_FOUND,
                                    ),
                                    "content-type",
                                    "application/json",
                                ))
                            }
                        },
                        Err(_) => {
                            let error_response = r#"{"error": "invalid_job_id"}"#.to_string();
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::BAD_REQUEST,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        let build_manager_cancel = self.build_job_manager.clone();
        let builds_cancel_route = warp::path!("api" / "v1" / "builds" / String)
            .and(warp::delete())
            .and_then(move |job_id: String| {
                let bm = build_manager_cancel.clone();
                async move {
                    match Uuid::parse_str(&job_id) {
                        Ok(id) => match bm.cancel_build(id).await {
                            Ok(()) => {
                                let json = r#"{"success": true}"#.to_string();
                                Ok::<_, Infallible>(warp::reply::with_header(
                                    warp::reply::with_status(json, warp::http::StatusCode::OK),
                                    "content-type",
                                    "application/json",
                                ))
                            }
                            Err(e) => {
                                let error_response = format!(r#"{{"error": "{}"}}"#, e);
                                Ok::<_, Infallible>(warp::reply::with_header(
                                    warp::reply::with_status(
                                        error_response,
                                        warp::http::StatusCode::NOT_FOUND,
                                    ),
                                    "content-type",
                                    "application/json",
                                ))
                            }
                        },
                        Err(_) => {
                            let error_response = r#"{"error": "invalid_job_id"}"#.to_string();
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::BAD_REQUEST,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        let build_manager_logs = self.build_job_manager.clone();
        let builds_logs_route = warp::path!("api" / "v1" / "builds" / String / "logs")
            .and(warp::get())
            .and_then(move |job_id: String| {
                let bm = build_manager_logs.clone();
                async move {
                    match Uuid::parse_str(&job_id) {
                        Ok(id) => match bm.get_logs(id).await {
                            Some(logs) => {
                                let json = serde_json::to_string(&logs).unwrap_or_else(|_| {
                                    r#"{"error": "serialization_failed"}"#.to_string()
                                });
                                Ok::<_, Infallible>(warp::reply::with_header(
                                    warp::reply::with_status(json, warp::http::StatusCode::OK),
                                    "content-type",
                                    "application/json",
                                ))
                            }
                            None => {
                                let error_response = r#"{"error": "job_not_found"}"#.to_string();
                                Ok::<_, Infallible>(warp::reply::with_header(
                                    warp::reply::with_status(
                                        error_response,
                                        warp::http::StatusCode::NOT_FOUND,
                                    ),
                                    "content-type",
                                    "application/json",
                                ))
                            }
                        },
                        Err(_) => {
                            let error_response = r#"{"error": "invalid_job_id"}"#.to_string();
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::BAD_REQUEST,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        // WebSocket route for real-time build log streaming
        let log_streamer_ws = self.log_streamer.clone();
        let builds_logs_stream_route =
            warp::path!("api" / "v1" / "builds" / String / "logs" / "stream")
                .and(warp::ws())
                .map(move |job_id: String, ws: warp::ws::Ws| {
                    let streamer = log_streamer_ws.clone();
                    ws.on_upgrade(move |socket| async move {
                        match Uuid::parse_str(&job_id) {
                            Ok(id) => {
                                streamer.handle_connection(id, socket).await;
                            }
                            Err(_) => {
                                tracing::warn!("Invalid job ID for WebSocket: {}", job_id);
                            }
                        }
                    })
                });

        // Source cache routes
        let build_manager_sources_list = self.build_job_manager.clone();
        let sources_list_route = warp::path!("api" / "v1" / "sources")
            .and(warp::get())
            .and_then(move || {
                let bm = build_manager_sources_list.clone();
                async move {
                    let sources = bm.cache_manager().list_cached_sources().await;
                    let json = serde_json::to_string(&sources)
                        .unwrap_or_else(|_| r#"{"error": "serialization_failed"}"#.to_string());
                    Ok::<_, Infallible>(warp::reply::with_header(
                        warp::reply::with_status(json, warp::http::StatusCode::OK),
                        "content-type",
                        "application/json",
                    ))
                }
            });

        let build_manager_sources_get = self.build_job_manager.clone();
        let sources_get_route = warp::path!("api" / "v1" / "sources" / String)
            .and(warp::get())
            .and_then(move |hash: String| {
                let bm = build_manager_sources_get.clone();
                async move {
                    match bm.cache_manager().get_source_metadata(&hash).await {
                        Some(metadata) => {
                            let json = serde_json::to_string(&metadata).unwrap_or_else(|_| {
                                r#"{"error": "serialization_failed"}"#.to_string()
                            });
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(json, warp::http::StatusCode::OK),
                                "content-type",
                                "application/json",
                            ))
                        }
                        None => {
                            let error_response = r#"{"error": "source_not_found"}"#.to_string();
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::NOT_FOUND,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        let build_manager_sources_delete = self.build_job_manager.clone();
        let sources_delete_route = warp::path!("api" / "v1" / "sources" / String)
            .and(warp::delete())
            .and_then(move |hash: String| {
                let bm = build_manager_sources_delete.clone();
                async move {
                    match bm.cache_manager().delete_cached_source(&hash).await {
                        Ok(Some(metadata)) => {
                            let response = serde_json::json!({
                                "success": true,
                                "deleted": metadata
                            });
                            let json = serde_json::to_string(&response)
                                .unwrap_or_else(|_| r#"{"success": true}"#.to_string());
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(json, warp::http::StatusCode::OK),
                                "content-type",
                                "application/json",
                            ))
                        }
                        Ok(None) => {
                            let error_response = r#"{"error": "source_not_found"}"#.to_string();
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::NOT_FOUND,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                        Err(e) => {
                            let error_response = format!(r#"{{"error": "{}"}}"#, e);
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        // Artifact cache routes
        let build_manager_artifacts_list = self.build_job_manager.clone();
        let artifacts_list_route = warp::path!("api" / "v1" / "artifacts")
            .and(warp::get())
            .and_then(move || {
                let bm = build_manager_artifacts_list.clone();
                async move {
                    let artifacts = bm.cache_manager().list_artifact_caches().await;
                    let json = serde_json::to_string(&artifacts)
                        .unwrap_or_else(|_| r#"{"error": "serialization_failed"}"#.to_string());
                    Ok::<_, Infallible>(warp::reply::with_header(
                        warp::reply::with_status(json, warp::http::StatusCode::OK),
                        "content-type",
                        "application/json",
                    ))
                }
            });

        let build_manager_artifacts_get = self.build_job_manager.clone();
        let artifacts_get_route = warp::path!("api" / "v1" / "artifacts" / String)
            .and(warp::get())
            .and_then(move |cache_key: String| {
                let bm = build_manager_artifacts_get.clone();
                async move {
                    match bm.cache_manager().get_artifact_metadata(&cache_key).await {
                        Some(metadata) => {
                            let json = serde_json::to_string(&metadata).unwrap_or_else(|_| {
                                r#"{"error": "serialization_failed"}"#.to_string()
                            });
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(json, warp::http::StatusCode::OK),
                                "content-type",
                                "application/json",
                            ))
                        }
                        None => {
                            let error_response = r#"{"error": "artifact_not_found"}"#.to_string();
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::NOT_FOUND,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        let build_manager_artifacts_delete = self.build_job_manager.clone();
        let artifacts_delete_route = warp::path!("api" / "v1" / "artifacts" / String)
            .and(warp::delete())
            .and_then(move |cache_key: String| {
                let bm = build_manager_artifacts_delete.clone();
                async move {
                    match bm.cache_manager().delete_artifact_cache(&cache_key).await {
                        Ok(Some(metadata)) => {
                            let response = serde_json::json!({
                                "success": true,
                                "deleted": metadata
                            });
                            let json = serde_json::to_string(&response)
                                .unwrap_or_else(|_| r#"{"success": true}"#.to_string());
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(json, warp::http::StatusCode::OK),
                                "content-type",
                                "application/json",
                            ))
                        }
                        Ok(None) => {
                            let error_response = r#"{"error": "artifact_not_found"}"#.to_string();
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::NOT_FOUND,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                        Err(e) => {
                            let error_response = format!(r#"{{"error": "{}"}}"#, e);
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        // Detailed metrics route (Prometheus format)
        let health_status_detailed = self.health_status.clone();
        let workload_manager_metrics = self.workload_manager.clone();
        let detailed_metrics_route = warp::path!("api" / "v1" / "metrics" / "detailed")
            .and(warp::get())
            .and_then(move || {
                let health = health_status_detailed.clone();
                let wm = workload_manager_metrics.clone();
                async move {
                    let h = health.read().await;
                    let workload_count = wm.active_workload_count().await;

                    let metrics = format!(
                        "# HELP swarmlet_cpu_usage_percent CPU usage percentage\n\
                         # TYPE swarmlet_cpu_usage_percent gauge\n\
                         swarmlet_cpu_usage_percent {}\n\
                         # HELP swarmlet_memory_usage_gb Memory usage in GB\n\
                         # TYPE swarmlet_memory_usage_gb gauge\n\
                         swarmlet_memory_usage_gb {}\n\
                         # HELP swarmlet_disk_usage_gb Disk usage in GB\n\
                         # TYPE swarmlet_disk_usage_gb gauge\n\
                         swarmlet_disk_usage_gb {}\n\
                         # HELP swarmlet_workloads_active Number of active workloads\n\
                         # TYPE swarmlet_workloads_active gauge\n\
                         swarmlet_workloads_active {}\n\
                         # HELP swarmlet_uptime_seconds Uptime in seconds\n\
                         # TYPE swarmlet_uptime_seconds counter\n\
                         swarmlet_uptime_seconds {}\n\
                         # HELP swarmlet_errors_total Total error count\n\
                         # TYPE swarmlet_errors_total counter\n\
                         swarmlet_errors_total {}\n",
                        h.cpu_usage_percent,
                        h.memory_usage_gb,
                        h.disk_usage_gb,
                        workload_count,
                        h.uptime_seconds,
                        h.errors_count
                    );

                    Ok::<_, Infallible>(warp::reply::with_header(
                        warp::reply::with_status(metrics, warp::http::StatusCode::OK),
                        "content-type",
                        "text/plain; version=0.0.4; charset=utf-8",
                    ))
                }
            });

        // Hardware profile route
        let node_id_hardware = self.join_result.node_id;
        let hardware_route = warp::path!("api" / "v1" / "hardware")
            .and(warp::get())
            .and_then(move || {
                let node_id = node_id_hardware;
                async move {
                    let mut profiler = HardwareProfiler::new();
                    match profiler.profile().await {
                        Ok(mut profile) => {
                            // Use the actual node_id instead of randomly generated one
                            profile.node_id = node_id;
                            let json = serde_json::to_string(&profile).unwrap_or_else(|_| {
                                r#"{"error": "serialization_failed"}"#.to_string()
                            });
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(json, warp::http::StatusCode::OK),
                                "content-type",
                                "application/json",
                            ))
                        }
                        Err(e) => {
                            let error_response = format!(r#"{{"error": "{}"}}"#, e);
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        let routes = health_route
            .or(metrics_route)
            .or(execute_command_route)
            .or(execute_shell_route)
            .or(wireguard_configure_route)
            .or(wireguard_status_route)
            .or(wireguard_add_peer_route)
            .or(wireguard_remove_peer_route)
            .or(wireguard_list_peers_route)
            .or(workloads_list_route)
            .or(workloads_stop_route)
            .or(builds_submit_route)
            .or(builds_list_route)
            .or(builds_get_route)
            .or(builds_cancel_route)
            .or(builds_logs_route)
            .or(builds_logs_stream_route)
            .or(sources_list_route)
            .or(sources_get_route)
            .or(sources_delete_route)
            .or(artifacts_list_route)
            .or(artifacts_get_route)
            .or(artifacts_delete_route)
            .or(detailed_metrics_route)
            .or(hardware_route);

        // Start server
        let port = self.config.api_port.unwrap_or(8080);
        let addr: SocketAddr = ([0, 0, 0, 0], port).into();

        info!("Starting API server on {}", addr);

        tokio::select! {
            _ = warp::serve(routes).run(addr) => {
                debug!("API server completed");
            }
            _ = shutdown_signal.changed() => {
                debug!("API server shutting down");
            }
        }

        Ok(())
    }
}
