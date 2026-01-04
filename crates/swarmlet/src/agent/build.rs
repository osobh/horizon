//! Build job handling via hpc-channels

#[cfg(feature = "hpc-channels")]
use crate::build_job::BuildJobStatus;
#[cfg(feature = "hpc-channels")]
use crate::hpc_bridge::{
    build_job_from_submit, BuildChannelBridge, SharedArtifactTransferBridge,
    SharedBuildChannelBridge,
};
#[cfg(feature = "hpc-channels")]
use hpc_channels::messages::BuildMessage;

use super::SwarmletAgent;

impl SwarmletAgent {
    /// HPC-Channels build job listener - receives jobs from hpc-ci (1-5µs latency)
    ///
    /// This is the main integration point between hpc-ci and stratoswarm.
    /// Jobs submitted via hpc-channels bypass REST API for ultra-low latency.
    #[cfg(feature = "hpc-channels")]
    pub(super) async fn hpc_channels_build_listener(&self) -> Result<()> {
        // Subscribe to build job submissions from hpc-ci
        let mut build_rx = match BuildChannelBridge::subscribe_submissions() {
            Some(rx) => rx,
            None => {
                // Channel doesn't exist yet, wait and retry
                info!("Waiting for hpc.build.submit channel to be created...");
                tokio::time::sleep(Duration::from_secs(1)).await;

                // Try to create the channel by broadcasting
                let _ = hpc_channels::broadcast::<BuildMessage>(
                    hpc_channels::channels::BUILD_SUBMIT,
                    256,
                );

                match BuildChannelBridge::subscribe_submissions() {
                    Some(rx) => rx,
                    None => {
                        warn!("Could not subscribe to build submit channel");
                        return Ok(());
                    }
                }
            }
        };

        let mut shutdown_signal = self.shutdown_signal.clone();

        info!("HPC-Channels build listener started, listening on hpc.build.submit");

        loop {
            tokio::select! {
                result = build_rx.recv() => {
                    match result {
                        Ok(msg) => {
                            self.handle_hpc_build_message(msg).await;
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                            warn!("Build listener lagged by {} messages", n);
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                            info!("Build submit channel closed");
                            break;
                        }
                    }
                }
                _ = shutdown_signal.changed() => {
                    debug!("HPC-Channels build listener shutting down");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Handle an incoming build message from hpc-ci
    #[cfg(feature = "hpc-channels")]
    async fn handle_hpc_build_message(&self, msg: BuildMessage) {
        // Only process Submit messages
        if let Some(job) = build_job_from_submit(&msg) {
            let job_id = job.id;

            info!(
                "Received build job from hpc-ci: {} (command: {:?})",
                job_id, job.command
            );

            // Publish queued status
            self.build_bridge.publish_queued(&job_id, 0);

            // Submit to local BuildJobManager
            match self.build_job_manager.submit_build(job).await {
                Ok(_) => {
                    // Publish started status
                    self.build_bridge.publish_started(&job_id);

                    // Start monitoring this job for completion
                    let build_manager = self.build_job_manager.clone();
                    let build_bridge = self.build_bridge.clone();
                    let artifact_bridge = self.artifact_bridge.clone();

                    tokio::spawn(async move {
                        Self::monitor_build_completion(
                            job_id,
                            build_manager,
                            build_bridge,
                            artifact_bridge,
                        )
                        .await;
                    });
                }
                Err(e) => {
                    error!("Failed to submit build job {}: {}", job_id, e);
                    self.build_bridge.publish_failed(&job_id, &e.to_string());
                }
            }
        }
    }

    /// Monitor a build job for completion and publish status updates + logs
    #[cfg(feature = "hpc-channels")]
    #[allow(unused_variables)] // artifact_bridge only used with warp-transfer feature
    async fn monitor_build_completion(
        job_id: Uuid,
        build_manager: Arc<BuildJobManager>,
        build_bridge: SharedBuildChannelBridge,
        artifact_bridge: SharedArtifactTransferBridge,
    ) {
        use crate::build_job::LogStream;
        use hpc_channels::messages::BuildLogLevel;

        let mut last_phase = String::new();
        let mut logs_sent = 0usize; // Track how many logs we've already sent
        let start_time = std::time::Instant::now();

        loop {
            // Wait a bit before checking
            tokio::time::sleep(Duration::from_millis(500)).await;

            // Stream any new logs
            if let Some(logs) = build_manager.get_logs(job_id).await {
                for log in logs.iter().skip(logs_sent) {
                    // Convert LogStream to BuildLogLevel
                    let level = match log.stream {
                        LogStream::Stdout => BuildLogLevel::Info,
                        LogStream::Stderr => BuildLogLevel::Warn,
                        LogStream::System => BuildLogLevel::Info,
                    };
                    let source = match log.stream {
                        LogStream::Stdout => "cargo",
                        LogStream::Stderr => "cargo",
                        LogStream::System => "system",
                    };
                    build_bridge.publish_log(&job_id, level, &log.message, source);
                }
                logs_sent = logs.len();
            }

            // Get job status
            match build_manager.get_job_status(job_id).await {
                Some(status) => {
                    // Convert status to phase string and progress percentage
                    let (phase_str, percent, is_terminal) = match &status {
                        BuildJobStatus::Queued => ("Queued".to_string(), 0, false),
                        BuildJobStatus::PreparingEnvironment => {
                            ("PreparingEnvironment".to_string(), 10, false)
                        }
                        BuildJobStatus::FetchingSource => ("FetchingSource".to_string(), 20, false),
                        BuildJobStatus::ProvisioningToolchain => {
                            ("ProvisioningToolchain".to_string(), 30, false)
                        }
                        BuildJobStatus::Building => ("Building".to_string(), 50, false),
                        BuildJobStatus::Testing => ("Testing".to_string(), 75, false),
                        BuildJobStatus::CollectingArtifacts => {
                            ("CollectingArtifacts".to_string(), 90, false)
                        }
                        BuildJobStatus::Completed => ("Completed".to_string(), 100, true),
                        BuildJobStatus::Failed { .. } => ("Failed".to_string(), 0, true),
                        BuildJobStatus::Cancelled => ("Cancelled".to_string(), 0, true),
                        BuildJobStatus::TimedOut => ("TimedOut".to_string(), 0, true),
                    };

                    // Publish progress if phase changed
                    if phase_str != last_phase && !is_terminal {
                        last_phase = phase_str.clone();
                        build_bridge.publish_progress(&job_id, &phase_str, percent);
                    }

                    // Handle terminal states
                    match status {
                        BuildJobStatus::Completed => {
                            let duration_ms = start_time.elapsed().as_millis() as u64;

                            // Get artifacts from the full job info
                            let artifact_paths: Vec<std::path::PathBuf> =
                                match build_manager.get_job(job_id).await {
                                    Some(active_job) => active_job
                                        .artifacts
                                        .iter()
                                        .map(|a| a.path.clone())
                                        .collect(),
                                    None => vec![],
                                };
                            let artifacts: Vec<String> = artifact_paths
                                .iter()
                                .map(|p| p.display().to_string())
                                .collect();

                            // Try to store artifacts to warp (if warp-transfer feature enabled)
                            #[cfg(feature = "warp-transfer")]
                            let manifest = if !artifact_paths.is_empty() {
                                // Use temp directory for warp staging
                                let staging_dir = std::env::temp_dir()
                                    .join("swarmlet-warp")
                                    .join(job_id.to_string());
                                if let Err(e) = std::fs::create_dir_all(&staging_dir) {
                                    warn!("Failed to create warp staging dir: {}", e);
                                }

                                // Upload artifacts to warp storage
                                match artifact_bridge
                                    .upload_artifacts(
                                        &job_id.to_string(),
                                        &job_id.to_string(), // pipeline_id (use job_id as fallback)
                                        "build",             // stage
                                        &artifact_paths,
                                        &staging_dir,
                                    )
                                    .await
                                {
                                    Ok(m) => {
                                        info!(
                                            "Stored {} artifacts to warp (merkle_root: {}, {} bytes)",
                                            m.files.len(),
                                            m.merkle_root,
                                            m.total_bytes
                                        );
                                        Some(m)
                                    }
                                    Err(e) => {
                                        warn!("Failed to store artifacts to warp: {}", e);
                                        None
                                    }
                                }
                            } else {
                                None
                            };

                            #[cfg(not(feature = "warp-transfer"))]
                            let manifest: Option<
                                hpc_channels::messages::ArtifactManifest,
                            > = None;

                            // Log and publish completion with manifest if available
                            let artifact_count =
                                manifest.as_ref().map(|m| m.files.len()).unwrap_or(0);

                            if manifest.is_some() {
                                build_bridge.publish_completed_with_manifest(
                                    &job_id,
                                    true,
                                    duration_ms,
                                    artifacts,
                                    manifest,
                                );
                            } else {
                                build_bridge.publish_completed(
                                    &job_id,
                                    true,
                                    duration_ms,
                                    artifacts,
                                );
                            }

                            info!(
                                "Build job {} completed (duration: {}ms, warp artifacts: {})",
                                job_id, duration_ms, artifact_count
                            );
                            return;
                        }
                        BuildJobStatus::Failed { ref error } => {
                            build_bridge.publish_failed(&job_id, error);
                            error!("Build job {} failed: {}", job_id, error);
                            return;
                        }
                        BuildJobStatus::Cancelled => {
                            build_bridge.publish_cancelled(&job_id);
                            info!("Build job {} was cancelled", job_id);
                            return;
                        }
                        BuildJobStatus::TimedOut => {
                            build_bridge.publish_failed(&job_id, "Build job timed out");
                            error!("Build job {} timed out", job_id);
                            return;
                        }
                        _ => {
                            // Continue monitoring
                        }
                    }
                }
                None => {
                    // Job not found, it may have been cleaned up
                    warn!("Build job {} not found in manager", job_id);
                    return;
                }
            }

            // Timeout after 1 hour
            if start_time.elapsed() > Duration::from_secs(3600) {
                build_bridge.publish_failed(&job_id, "Build timeout exceeded (1 hour)");
                error!("Build job {} timed out", job_id);
                return;
            }
        }
    }

    /// Listen for build cancellation requests from hpc-ci
    #[cfg(feature = "hpc-channels")]
    pub(super) async fn hpc_channels_cancel_listener(&self) -> Result<()> {
        // Subscribe to build cancellation requests from hpc-ci
        let mut cancel_rx = match BuildChannelBridge::subscribe_cancellations() {
            Some(rx) => rx,
            None => {
                // Channel doesn't exist yet, wait and retry
                info!("Waiting for hpc.build.cancel channel to be created...");
                tokio::time::sleep(Duration::from_secs(1)).await;

                // Try to create the channel by broadcasting
                let _ = hpc_channels::broadcast::<BuildMessage>(
                    hpc_channels::channels::BUILD_CANCEL,
                    256,
                );

                match BuildChannelBridge::subscribe_cancellations() {
                    Some(rx) => rx,
                    None => {
                        warn!("Could not subscribe to build cancel channel");
                        return Ok(());
                    }
                }
            }
        };

        let mut shutdown_signal = self.shutdown_signal.clone();

        info!("HPC-Channels cancel listener started, listening on hpc.build.cancel");

        loop {
            tokio::select! {
                result = cancel_rx.recv() => {
                    match result {
                        Ok(msg) => {
                            self.handle_cancel_request(msg).await;
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                            warn!("Cancel listener lagged by {} messages", n);
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                            info!("Build cancel channel closed");
                            break;
                        }
                    }
                }
                _ = shutdown_signal.changed() => {
                    debug!("HPC-Channels cancel listener shutting down");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Handle a build cancellation request from hpc-ci
    #[cfg(feature = "hpc-channels")]
    async fn handle_cancel_request(&self, msg: BuildMessage) {
        if let BuildMessage::CancelRequest { job_id, reason, .. } = msg {
            let job_uuid = match Uuid::parse_str(&job_id) {
                Ok(id) => id,
                Err(_) => {
                    warn!("Invalid job ID in cancel request: {}", job_id);
                    return;
                }
            };

            info!(
                "Received cancel request for job {}: {:?}",
                job_id,
                reason.as_deref().unwrap_or("no reason provided")
            );

            // Cancel the build in the BuildJobManager
            match self.build_job_manager.cancel_build(job_uuid).await {
                Ok(()) => {
                    info!("Build job {} cancelled successfully", job_id);
                    self.build_bridge.publish_cancelled(&job_uuid);
                }
                Err(e) => {
                    warn!("Failed to cancel build job {}: {}", job_id, e);
                }
            }
        }
    }
}
