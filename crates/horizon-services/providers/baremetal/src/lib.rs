use hpc_provider::{
    Availability, CapacityProvider, HealthStatus, Instance, InstanceState, ProviderError,
    ProviderResult, ProvisionResult, ProvisionSpec, Quote, QuoteRequest, ServiceQuotas, SpotPrices,
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BareMetalConfig {
    pub datacenter: String,
    pub ipmi_endpoint: Option<String>,
}

impl BareMetalConfig {
    pub fn new(datacenter: String) -> Self {
        Self {
            datacenter,
            ipmi_endpoint: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalServer {
    pub id: String,
    pub hostname: String,
    pub ipmi_address: String,
    pub gpu_type: String,
    pub gpu_count: usize,
    pub status: ServerStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ServerStatus {
    Available,
    InUse,
    Maintenance,
    Offline,
}

pub struct BareMetalProvider {
    config: BareMetalConfig,
    inventory: Arc<Mutex<HashMap<String, PhysicalServer>>>,
    allocations: Arc<Mutex<HashMap<String, String>>>, // instance_id -> server_id
    quotas: Arc<Mutex<ServiceQuotas>>,
}

impl BareMetalProvider {
    pub async fn new(config: BareMetalConfig) -> ProviderResult<Self> {
        let mut inventory = HashMap::new();

        // Initialize with sample physical servers
        for i in 1..=10 {
            let server = PhysicalServer {
                id: format!("server-{:03}", i),
                hostname: format!("gpu-node-{:03}.datacenter.local", i),
                ipmi_address: format!("10.0.0.{}", 100 + i),
                gpu_type: if i % 3 == 0 {
                    "H100".to_string()
                } else if i % 2 == 0 {
                    "A100".to_string()
                } else {
                    "V100".to_string()
                },
                gpu_count: 8,
                status: ServerStatus::Available,
            };
            inventory.insert(server.id.clone(), server);
        }

        Ok(Self {
            config,
            inventory: Arc::new(Mutex::new(inventory)),
            allocations: Arc::new(Mutex::new(HashMap::new())),
            quotas: Arc::new(Mutex::new(ServiceQuotas {
                max_instances: 10,
                current_instances: 0,
                max_vcpus: 320,
                current_vcpus: 0,
                max_gpus: 80,
                current_gpus: 0,
            })),
        })
    }

    fn calculate_hourly_rate(gpu_type: &str, gpu_count: usize) -> Decimal {
        let per_gpu_rate = match gpu_type {
            "H100" => dec!(10.0),
            "A100" => dec!(3.5),
            "V100" => dec!(2.5),
            _ => dec!(1.0),
        };
        per_gpu_rate * Decimal::from(gpu_count)
    }

    #[allow(dead_code)]
    fn find_available_server(&self, gpu_type: &str) -> Option<PhysicalServer> {
        let inventory = self.inventory.lock().unwrap();
        inventory
            .values()
            .find(|s| s.status == ServerStatus::Available && s.gpu_type == gpu_type)
            .cloned()
    }
}

#[async_trait::async_trait]
impl CapacityProvider for BareMetalProvider {
    fn name(&self) -> &str {
        "baremetal"
    }

    async fn get_quote(&self, request: &QuoteRequest) -> ProviderResult<Quote> {
        if request.count == 0 {
            return Err(ProviderError::InvalidRequest(
                "count must be greater than 0".to_string(),
            ));
        }

        // For bare-metal, instance_type is GPU type (H100, A100, V100)
        let hourly_rate = Self::calculate_hourly_rate(&request.instance_type, 8);

        let inventory = self.inventory.lock().unwrap();
        let available_count = inventory
            .values()
            .filter(|s| s.status == ServerStatus::Available && s.gpu_type == request.instance_type)
            .count();

        let availability = if available_count >= request.count {
            Availability::Available
        } else if available_count > 0 {
            Availability::Limited
        } else {
            Availability::Unavailable
        };

        let lead_time = if matches!(availability, Availability::Available) {
            0
        } else {
            24
        };

        Ok(Quote {
            provider: "baremetal".to_string(),
            instance_type: request.instance_type.clone(),
            region: self.config.datacenter.clone(),
            hourly_rate,
            spot_rate: None, // No spot pricing for bare-metal
            availability,
            lead_time_hours: lead_time,
        })
    }

    async fn provision(&self, spec: &ProvisionSpec) -> ProviderResult<ProvisionResult> {
        if spec.count == 0 {
            return Err(ProviderError::InvalidRequest(
                "count must be greater than 0".to_string(),
            ));
        }

        let mut quotas = self.quotas.lock().unwrap();
        if quotas.current_instances + spec.count > quotas.max_instances {
            return Err(ProviderError::QuotaExceeded(
                "physical server quota exceeded".to_string(),
            ));
        }

        let mut instances = Vec::new();
        let mut inventory = self.inventory.lock().unwrap();
        let mut allocations = self.allocations.lock().unwrap();

        for _ in 0..spec.count {
            // Find an available server
            let server_id = inventory
                .iter()
                .find(|(_, s)| {
                    s.status == ServerStatus::Available && s.gpu_type == spec.instance_type
                })
                .map(|(id, _)| id.clone())
                .ok_or_else(|| ProviderError::Unavailable("no available servers".to_string()))?;

            let server = inventory.get_mut(&server_id).unwrap();
            server.status = ServerStatus::InUse;

            let instance_id = format!("bm-instance-{}", server_id);

            let instance = Instance {
                id: instance_id.clone(),
                instance_type: spec.instance_type.clone(),
                region: spec.region.clone(),
                public_ip: None, // Bare-metal typically uses private networking
                private_ip: server.ipmi_address.clone(),
                state: InstanceState::Running, // Bare-metal is instantly "running"
            };

            allocations.insert(instance_id, server_id);
            instances.push(instance);
        }

        quotas.current_instances += spec.count;

        let hourly_rate = Self::calculate_hourly_rate(&spec.instance_type, 8);
        let total_cost_estimate = hourly_rate * Decimal::from(spec.count);

        Ok(ProvisionResult {
            instances,
            total_cost_estimate,
        })
    }

    async fn deprovision(&self, instance_id: &str) -> ProviderResult<()> {
        if instance_id.is_empty() {
            return Err(ProviderError::InvalidRequest(
                "instance_id cannot be empty".to_string(),
            ));
        }

        let mut allocations = self.allocations.lock().unwrap();
        let server_id = allocations.remove(instance_id).ok_or_else(|| {
            ProviderError::NotFound(format!("instance {} not found", instance_id))
        })?;

        let mut inventory = self.inventory.lock().unwrap();
        if let Some(server) = inventory.get_mut(&server_id) {
            server.status = ServerStatus::Available;
        }

        let mut quotas = self.quotas.lock().unwrap();
        quotas.current_instances = quotas.current_instances.saturating_sub(1);

        Ok(())
    }

    async fn check_spot_prices(
        &self,
        instance_type: &str,
        _region: &str,
    ) -> ProviderResult<SpotPrices> {
        // Bare-metal doesn't have spot pricing, but we return regular pricing
        let hourly_rate = Self::calculate_hourly_rate(instance_type, 8);

        Ok(SpotPrices {
            instance_type: instance_type.to_string(),
            region: self.config.datacenter.clone(),
            current_price: hourly_rate,
            average_price: hourly_rate,
            min_price: hourly_rate,
            max_price: hourly_rate,
        })
    }

    async fn get_quotas(&self) -> ProviderResult<ServiceQuotas> {
        Ok(self.quotas.lock().unwrap().clone())
    }

    async fn health(&self) -> ProviderResult<HealthStatus> {
        let inventory = self.inventory.lock().unwrap();
        let offline_count = inventory
            .values()
            .filter(|s| s.status == ServerStatus::Offline)
            .count();

        let healthy = offline_count == 0;

        Ok(HealthStatus {
            healthy,
            message: if healthy {
                "All physical servers operational".to_string()
            } else {
                format!("{} servers offline", offline_count)
            },
        })
    }
}

/// Calculate hourly rate for a GPU type and count
pub fn calculate_hourly_rate(gpu_type: &str, gpu_count: usize) -> Decimal {
    BareMetalProvider::calculate_hourly_rate(gpu_type, gpu_count)
}
