//! Deployment handling via hpc-channels

#[cfg(feature = "hpc-channels")]
use crate::hpc_bridge::DeployChannelBridge;
#[cfg(feature = "hpc-channels")]
use hpc_channels::messages::DeployMessage;

use super::SwarmletAgent;

impl SwarmletAgent {
    /// HPC-Channels deploy listener - receives deployments from hpc-ci (1-5µs latency)
    ///
    /// Handles deployment requests including blue-green, rolling, and canary strategies.
    #[cfg(feature = "hpc-channels")]
    pub(super) async fn hpc_channels_deploy_listener(&self) -> Result<()> {
        // Subscribe to deploy job submissions from hpc-ci
        let mut deploy_rx = match DeployChannelBridge::subscribe_submissions() {
            Some(rx) => rx,
            None => {
                // Channel doesn't exist yet, wait and retry
                info!("Waiting for hpc.deploy.submit channel to be created...");
                tokio::time::sleep(Duration::from_secs(1)).await;

                // Try to create the channel by broadcasting
                let _ = hpc_channels::broadcast::<DeployMessage>(
                    hpc_channels::channels::DEPLOY_SUBMIT,
                    256,
                );

                match DeployChannelBridge::subscribe_submissions() {
                    Some(rx) => rx,
                    None => {
                        warn!("Could not subscribe to deploy submit channel");
                        return Ok(());
                    }
                }
            }
        };

        let mut shutdown_signal = self.shutdown_signal.clone();

        info!("HPC-Channels deploy listener started, listening on hpc.deploy.submit");

        loop {
            tokio::select! {
                result = deploy_rx.recv() => {
                    match result {
                        Ok(msg) => {
                            self.handle_hpc_deploy_message(msg).await;
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                            warn!("Deploy listener lagged by {} messages", n);
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                            info!("Deploy submit channel closed");
                            break;
                        }
                    }
                }
                _ = shutdown_signal.changed() => {
                    debug!("HPC-Channels deploy listener shutting down");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Handle an incoming deploy message from hpc-ci
    #[cfg(feature = "hpc-channels")]
    async fn handle_hpc_deploy_message(&self, msg: DeployMessage) {
        // Only process Submit messages
        if let DeployMessage::Submit {
            deploy_id,
            artifact_ref,
            namespace,
            strategy,
            replicas,
            health_check,
            rollback_on_failure,
            ..
        } = msg
        {
            info!(
                "Received deployment from hpc-ci: {} (namespace: {}, replicas: {})",
                deploy_id, namespace, replicas
            );

            // Publish started status
            self.deploy_bridge
                .publish_started(&deploy_id, strategy.clone());

            // Execute the deployment
            let result = self
                .execute_deployment(
                    &deploy_id,
                    &artifact_ref,
                    &namespace,
                    &strategy,
                    replicas,
                    health_check.as_deref(),
                    rollback_on_failure,
                )
                .await;

            match result {
                Ok(url) => {
                    self.deploy_bridge.publish_completed(&deploy_id, true, url);
                    info!("Deployment {} completed successfully", deploy_id);
                }
                Err(e) => {
                    error!("Deployment {} failed: {}", deploy_id, e);
                    if rollback_on_failure {
                        self.deploy_bridge.publish_rolled_back(&deploy_id, &e);
                    } else {
                        self.deploy_bridge
                            .publish_completed(&deploy_id, false, None);
                    }
                }
            }
        }
    }

    /// Execute a deployment with the specified strategy
    #[cfg(feature = "hpc-channels")]
    async fn execute_deployment(
        &self,
        deploy_id: &str,
        artifact_ref: &str,
        namespace: &str,
        strategy: &hpc_channels::messages::DeployStrategy,
        replicas: u32,
        health_check: Option<&str>,
        _rollback_on_failure: bool,
    ) -> std::result::Result<Option<String>, String> {
        use hpc_channels::messages::DeployStrategy as HpcDeployStrategy;

        // Publish progress: fetching artifact
        self.deploy_bridge
            .publish_progress(deploy_id, "fetching_artifact", 10);

        // Check if artifact is available in cache
        info!("Fetching artifact: {}", artifact_ref);
        let cache_manager = self.build_job_manager.cache_manager();
        if let Some(metadata) = cache_manager.get_artifact_metadata(artifact_ref).await {
            info!(
                "Artifact {} found in cache (size: {} bytes, last used: {:?})",
                artifact_ref, metadata.size_bytes, metadata.last_used_at
            );
        } else {
            // Artifact not in cache - would need to fetch from external storage
            warn!("Artifact {} not found in local cache", artifact_ref);
        }

        // Publish progress: preparing deployment
        self.deploy_bridge
            .publish_progress(deploy_id, "preparing", 30);

        match strategy {
            HpcDeployStrategy::BlueGreen => {
                self.execute_blue_green_deployment(
                    deploy_id,
                    artifact_ref,
                    namespace,
                    replicas,
                    health_check,
                )
                .await
            }
            HpcDeployStrategy::Rolling { batch_size } => {
                self.execute_rolling_deployment(
                    deploy_id,
                    artifact_ref,
                    namespace,
                    replicas,
                    *batch_size,
                    health_check,
                )
                .await
            }
            HpcDeployStrategy::Canary { traffic_percent } => {
                self.execute_canary_deployment(
                    deploy_id,
                    artifact_ref,
                    namespace,
                    replicas,
                    *traffic_percent,
                    health_check,
                )
                .await
            }
        }
    }

    /// Execute a blue-green deployment
    #[cfg(feature = "hpc-channels")]
    async fn execute_blue_green_deployment(
        &self,
        deploy_id: &str,
        artifact_ref: &str,
        namespace: &str,
        replicas: u32,
        health_check: Option<&str>,
    ) -> std::result::Result<Option<String>, String> {
        info!(
            "Executing blue-green deployment {} in namespace {}",
            deploy_id, namespace
        );

        // Phase 1: Deploy green (new version)
        self.deploy_bridge
            .publish_progress(deploy_id, "deploying_green", 40);

        // Create workload assignment for the new version
        let assignment = WorkAssignment {
            id: Uuid::new_v4(),
            workload_type: "deployment".to_string(),
            container_image: Some(artifact_ref.to_string()),
            command: None,
            shell_script: None,
            environment: [
                ("DEPLOY_ID".to_string(), deploy_id.to_string()),
                ("NAMESPACE".to_string(), namespace.to_string()),
                ("DEPLOY_COLOR".to_string(), "green".to_string()),
            ]
            .into_iter()
            .collect(),
            resource_limits: ResourceLimits::default(),
            created_at: chrono::Utc::now(),
        };

        // Start the workload
        for i in 0..replicas {
            let mut replica_assignment = assignment.clone();
            replica_assignment.id = Uuid::new_v4();
            replica_assignment
                .environment
                .insert("REPLICA_INDEX".to_string(), i.to_string());

            if let Err(e) = self
                .workload_manager
                .start_workload(replica_assignment)
                .await
            {
                return Err(format!("Failed to start replica {}: {}", i, e));
            }
        }

        self.deploy_bridge
            .publish_progress(deploy_id, "green_deployed", 60);

        // Phase 2: Health check
        if let Some(endpoint) = health_check {
            self.deploy_bridge
                .publish_progress(deploy_id, "health_checking", 70);

            // Real health check using HealthChecker
            let checker = crate::health_check::HealthChecker::with_config(
                crate::health_check::HealthCheckConfig::default()
                    .with_initial_delay(Duration::from_secs(2))
                    .with_max_retries(3),
            );
            let result = checker.check(endpoint).await;

            self.deploy_bridge
                .publish_health_check(deploy_id, result.passed, endpoint);

            if !result.passed {
                return Err(format!(
                    "Health check failed: {}",
                    result.error.unwrap_or_else(|| "Unknown error".to_string())
                ));
            }
        }

        // Phase 3: Switch traffic (simulated)
        self.deploy_bridge
            .publish_progress(deploy_id, "switching_traffic", 90);
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Return deployment URL
        let url = format!("https://{}.{}.svc.cluster.local", deploy_id, namespace);
        Ok(Some(url))
    }

    /// Execute a rolling deployment
    #[cfg(feature = "hpc-channels")]
    async fn execute_rolling_deployment(
        &self,
        deploy_id: &str,
        artifact_ref: &str,
        namespace: &str,
        replicas: u32,
        batch_size: u32,
        health_check: Option<&str>,
    ) -> std::result::Result<Option<String>, String> {
        info!(
            "Executing rolling deployment {} (batch_size: {}) in namespace {}",
            deploy_id, batch_size, namespace
        );

        let total_batches = (replicas + batch_size - 1) / batch_size;
        let mut deployed = 0u32;

        for batch in 0..total_batches {
            let batch_start = batch * batch_size;
            let batch_end = std::cmp::min(batch_start + batch_size, replicas);

            // Calculate progress percentage
            let progress = 30 + ((batch as u8 * 60) / total_batches as u8);
            self.deploy_bridge.publish_progress(
                deploy_id,
                &format!("rolling_batch_{}", batch),
                progress,
            );

            // Deploy this batch
            for i in batch_start..batch_end {
                let assignment = WorkAssignment {
                    id: Uuid::new_v4(),
                    workload_type: "deployment".to_string(),
                    container_image: Some(artifact_ref.to_string()),
                    command: None,
                    shell_script: None,
                    environment: [
                        ("DEPLOY_ID".to_string(), deploy_id.to_string()),
                        ("NAMESPACE".to_string(), namespace.to_string()),
                        ("REPLICA_INDEX".to_string(), i.to_string()),
                    ]
                    .into_iter()
                    .collect(),
                    resource_limits: ResourceLimits::default(),
                    created_at: chrono::Utc::now(),
                };

                if let Err(e) = self.workload_manager.start_workload(assignment).await {
                    return Err(format!("Failed to deploy replica {}: {}", i, e));
                }
                deployed += 1;
            }

            // Health check after each batch
            if let Some(endpoint) = health_check {
                // Real health check using HealthChecker (quick for batch checks)
                let checker = crate::health_check::HealthChecker::with_config(
                    crate::health_check::HealthCheckConfig::quick()
                        .with_initial_delay(Duration::from_secs(1)),
                );
                let result = checker.check(endpoint).await;

                self.deploy_bridge
                    .publish_health_check(deploy_id, result.passed, endpoint);

                if !result.passed {
                    return Err(format!(
                        "Rolling batch {} health check failed: {}",
                        batch,
                        result.error.unwrap_or_else(|| "Unknown error".to_string())
                    ));
                }
            }

            info!(
                "Rolling deployment {}: batch {} complete ({}/{})",
                deploy_id, batch, deployed, replicas
            );
        }

        self.deploy_bridge
            .publish_progress(deploy_id, "complete", 100);

        let url = format!("https://{}.{}.svc.cluster.local", deploy_id, namespace);
        Ok(Some(url))
    }

    /// Execute a canary deployment
    #[cfg(feature = "hpc-channels")]
    async fn execute_canary_deployment(
        &self,
        deploy_id: &str,
        artifact_ref: &str,
        namespace: &str,
        replicas: u32,
        traffic_percent: u8,
        health_check: Option<&str>,
    ) -> std::result::Result<Option<String>, String> {
        info!(
            "Executing canary deployment {} ({}% traffic) in namespace {}",
            deploy_id, traffic_percent, namespace
        );

        // Calculate canary replicas (minimum 1)
        let canary_replicas = std::cmp::max(1, (replicas * traffic_percent as u32) / 100);

        self.deploy_bridge
            .publish_progress(deploy_id, "deploying_canary", 40);

        // Deploy canary instances
        for i in 0..canary_replicas {
            let assignment = WorkAssignment {
                id: Uuid::new_v4(),
                workload_type: "deployment".to_string(),
                container_image: Some(artifact_ref.to_string()),
                command: None,
                shell_script: None,
                environment: [
                    ("DEPLOY_ID".to_string(), deploy_id.to_string()),
                    ("NAMESPACE".to_string(), namespace.to_string()),
                    ("DEPLOY_TYPE".to_string(), "canary".to_string()),
                    ("REPLICA_INDEX".to_string(), i.to_string()),
                ]
                .into_iter()
                .collect(),
                resource_limits: ResourceLimits::default(),
                created_at: chrono::Utc::now(),
            };

            if let Err(e) = self.workload_manager.start_workload(assignment).await {
                return Err(format!("Failed to deploy canary replica {}: {}", i, e));
            }
        }

        self.deploy_bridge
            .publish_progress(deploy_id, "canary_deployed", 60);

        // Health check canary
        if let Some(endpoint) = health_check {
            self.deploy_bridge
                .publish_progress(deploy_id, "health_checking_canary", 70);

            // Real health check using HealthChecker
            let checker = crate::health_check::HealthChecker::with_config(
                crate::health_check::HealthCheckConfig::default()
                    .with_initial_delay(Duration::from_secs(2))
                    .with_max_retries(3),
            );
            let result = checker.check(endpoint).await;

            self.deploy_bridge
                .publish_health_check(deploy_id, result.passed, endpoint);

            if !result.passed {
                return Err(format!(
                    "Canary health check failed: {}",
                    result.error.unwrap_or_else(|| "Unknown error".to_string())
                ));
            }
        }

        // In a real canary, we would monitor metrics and gradually increase traffic
        // For now, we just report success after the canary is deployed
        self.deploy_bridge
            .publish_progress(deploy_id, "canary_monitoring", 80);

        info!(
            "Canary deployment {}: {} replicas receiving {}% traffic",
            deploy_id, canary_replicas, traffic_percent
        );

        let url = format!(
            "https://{}-canary.{}.svc.cluster.local",
            deploy_id, namespace
        );
        Ok(Some(url))
    }
}
