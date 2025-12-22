//! Configuration generation types for AI Assistant Zero-Config Integration
//! Split from types.rs to keep files under 750 lines

/// Generated configuration for deployment
#[derive(Debug, Clone)]
pub struct GeneratedConfiguration {
    pub deployment_config: super::DeploymentConfiguration,
    pub infrastructure_config: super::InfrastructureConfiguration,
    pub security_config: super::SecurityConfiguration,
    pub monitoring_config: super::MonitoringConfiguration,
    pub networking_config: super::NetworkConfiguration,
    pub storage_config: super::StorageConfiguration,
    pub scaling_config: super::ScalingConfiguration,
    pub cost_optimization: super::CostOptimization,
}