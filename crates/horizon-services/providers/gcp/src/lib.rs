use hpc_provider::{
    Availability, CapacityProvider, HealthStatus, Instance, InstanceState, ProviderError,
    ProviderResult, ProvisionResult, ProvisionSpec, Quote, QuoteRequest, ServiceQuotas,
    SpotPrices,
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcpConfig {
    pub project_id: String,
    pub zone: String,
    pub credentials_path: Option<String>,
}

impl GcpConfig {
    pub fn new(project_id: String, zone: String) -> Self {
        Self {
            project_id,
            zone,
            credentials_path: None,
        }
    }
}

struct PricingData {
    on_demand: HashMap<String, Decimal>,
    preemptible: HashMap<String, Decimal>,
}

impl PricingData {
    fn new() -> Self {
        let mut on_demand = HashMap::new();
        let mut preemptible = HashMap::new();

        // GCP A100 instances
        on_demand.insert("a2-highgpu-1g".to_string(), dec!(3.67)); // 1x A100
        on_demand.insert("a2-highgpu-2g".to_string(), dec!(7.35)); // 2x A100
        on_demand.insert("a2-highgpu-4g".to_string(), dec!(14.69)); // 4x A100
        on_demand.insert("a2-highgpu-8g".to_string(), dec!(29.39)); // 8x A100
        on_demand.insert("a2-megagpu-16g".to_string(), dec!(58.78)); // 16x A100

        preemptible.insert("a2-highgpu-1g".to_string(), dec!(1.10));
        preemptible.insert("a2-highgpu-2g".to_string(), dec!(2.21));
        preemptible.insert("a2-highgpu-4g".to_string(), dec!(4.41));
        preemptible.insert("a2-highgpu-8g".to_string(), dec!(8.82));
        preemptible.insert("a2-megagpu-16g".to_string(), dec!(17.63));

        // GCP V100 instances
        on_demand.insert("n1-highmem-8-v100-1".to_string(), dec!(2.48));
        on_demand.insert("n1-highmem-8-v100-2".to_string(), dec!(4.96));
        on_demand.insert("n1-highmem-8-v100-4".to_string(), dec!(9.92));

        preemptible.insert("n1-highmem-8-v100-1".to_string(), dec!(0.74));
        preemptible.insert("n1-highmem-8-v100-2".to_string(), dec!(1.49));
        preemptible.insert("n1-highmem-8-v100-4".to_string(), dec!(2.98));

        Self {
            on_demand,
            preemptible,
        }
    }

    fn get_on_demand_price(&self, machine_type: &str) -> Option<Decimal> {
        self.on_demand.get(machine_type).copied()
    }

    fn get_preemptible_price(&self, machine_type: &str) -> Option<Decimal> {
        self.preemptible.get(machine_type).copied()
    }

    fn is_supported(&self, machine_type: &str) -> bool {
        self.on_demand.contains_key(machine_type)
    }
}

pub struct GcpProvider {
    #[allow(dead_code)]
    config: GcpConfig,
    pricing: PricingData,
    instances: Arc<Mutex<HashMap<String, Instance>>>,
    instance_counter: Arc<Mutex<u64>>,
    quotas: Arc<Mutex<ServiceQuotas>>,
}

impl GcpProvider {
    pub async fn new(config: GcpConfig) -> ProviderResult<Self> {
        Ok(Self {
            config,
            pricing: PricingData::new(),
            instances: Arc::new(Mutex::new(HashMap::new())),
            instance_counter: Arc::new(Mutex::new(0)),
            quotas: Arc::new(Mutex::new(ServiceQuotas {
                max_instances: 100,
                current_instances: 0,
                max_vcpus: 400,
                current_vcpus: 0,
                max_gpus: 80,
                current_gpus: 0,
            })),
        })
    }

    fn generate_instance_id(&self) -> String {
        let mut counter = self.instance_counter.lock().unwrap();
        *counter += 1;
        format!("gcp-instance-{}", *counter)
    }
}

#[async_trait::async_trait]
impl CapacityProvider for GcpProvider {
    fn name(&self) -> &str {
        "gcp"
    }

    async fn get_quote(&self, request: &QuoteRequest) -> ProviderResult<Quote> {
        if request.count == 0 {
            return Err(ProviderError::InvalidRequest(
                "count must be greater than 0".to_string(),
            ));
        }

        let hourly_rate = self
            .pricing
            .get_on_demand_price(&request.instance_type)
            .ok_or_else(|| {
                ProviderError::InvalidRequest(format!(
                    "machine type {} not supported",
                    request.instance_type
                ))
            })?;

        let spot_rate = if request.spot {
            self.pricing.get_preemptible_price(&request.instance_type)
        } else {
            None
        };

        let quotas = self.quotas.lock().unwrap();
        let availability = if quotas.current_instances + request.count <= quotas.max_instances {
            Availability::Available
        } else {
            Availability::Limited
        };

        Ok(Quote {
            provider: "gcp".to_string(),
            instance_type: request.instance_type.clone(),
            region: request.region.clone(),
            hourly_rate,
            spot_rate,
            availability,
            lead_time_hours: 0,
        })
    }

    async fn provision(&self, spec: &ProvisionSpec) -> ProviderResult<ProvisionResult> {
        if spec.count == 0 {
            return Err(ProviderError::InvalidRequest(
                "count must be greater than 0".to_string(),
            ));
        }

        if !self.pricing.is_supported(&spec.instance_type) {
            return Err(ProviderError::InvalidRequest(format!(
                "machine type {} not supported",
                spec.instance_type
            )));
        }

        let mut quotas = self.quotas.lock().unwrap();
        if quotas.current_instances + spec.count > quotas.max_instances {
            return Err(ProviderError::QuotaExceeded(
                "instance quota exceeded".to_string(),
            ));
        }

        let rate = if spec.spot {
            self.pricing
                .get_preemptible_price(&spec.instance_type)
                .unwrap_or_else(|| {
                    self.pricing
                        .get_on_demand_price(&spec.instance_type)
                        .unwrap()
                })
        } else {
            self.pricing
                .get_on_demand_price(&spec.instance_type)
                .unwrap()
        };

        let mut instances = Vec::new();
        let mut instances_map = self.instances.lock().unwrap();

        for i in 0..spec.count {
            let instance_id = self.generate_instance_id();
            let instance = Instance {
                id: instance_id.clone(),
                instance_type: spec.instance_type.clone(),
                region: spec.region.clone(),
                public_ip: Some(format!("35.{}.{}.{}", i / 65536, (i / 256) % 256, i % 256)),
                private_ip: format!("10.128.{}.{}", (i / 256) % 256, i % 256),
                state: InstanceState::Pending,
            };
            instances_map.insert(instance_id, instance.clone());
            instances.push(instance);
        }

        quotas.current_instances += spec.count;

        let total_cost_estimate = rate * Decimal::from(spec.count);

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

        let mut instances = self.instances.lock().unwrap();
        if instances.remove(instance_id).is_some() {
            let mut quotas = self.quotas.lock().unwrap();
            quotas.current_instances = quotas.current_instances.saturating_sub(1);
            Ok(())
        } else {
            Err(ProviderError::NotFound(format!(
                "instance {} not found",
                instance_id
            )))
        }
    }

    async fn check_spot_prices(
        &self,
        instance_type: &str,
        region: &str,
    ) -> ProviderResult<SpotPrices> {
        if instance_type.is_empty() {
            return Err(ProviderError::InvalidRequest(
                "instance_type cannot be empty".to_string(),
            ));
        }

        let base_price = self
            .pricing
            .get_preemptible_price(instance_type)
            .ok_or_else(|| {
                ProviderError::InvalidRequest(format!(
                    "preemptible pricing not available for {}",
                    instance_type
                ))
            })?;

        Ok(SpotPrices {
            instance_type: instance_type.to_string(),
            region: region.to_string(),
            current_price: base_price * dec!(1.05),
            average_price: base_price * dec!(1.10),
            min_price: base_price * dec!(0.90),
            max_price: base_price * dec!(1.30),
        })
    }

    async fn get_quotas(&self) -> ProviderResult<ServiceQuotas> {
        Ok(self.quotas.lock().unwrap().clone())
    }

    async fn health(&self) -> ProviderResult<HealthStatus> {
        Ok(HealthStatus {
            healthy: true,
            message: "GCP Compute Engine operational".to_string(),
        })
    }
}

