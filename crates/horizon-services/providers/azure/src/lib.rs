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
pub struct AzureConfig {
    pub subscription_id: String,
    pub resource_group: String,
    pub location: String,
}

impl AzureConfig {
    pub fn new(subscription_id: String, resource_group: String, location: String) -> Self {
        Self {
            subscription_id,
            resource_group,
            location,
        }
    }
}

struct PricingData {
    on_demand: HashMap<String, Decimal>,
    spot: HashMap<String, Decimal>,
}

impl PricingData {
    fn new() -> Self {
        let mut on_demand = HashMap::new();
        let mut spot = HashMap::new();

        // Azure NC v3 series (V100 GPUs)
        on_demand.insert("Standard_NC6s_v3".to_string(), dec!(3.06)); // 1x V100
        on_demand.insert("Standard_NC12s_v3".to_string(), dec!(6.12)); // 2x V100
        on_demand.insert("Standard_NC24s_v3".to_string(), dec!(12.24)); // 4x V100

        spot.insert("Standard_NC6s_v3".to_string(), dec!(0.92));
        spot.insert("Standard_NC12s_v3".to_string(), dec!(1.84));
        spot.insert("Standard_NC24s_v3".to_string(), dec!(3.67));

        // Azure ND A100 v4 series
        on_demand.insert("Standard_ND96asr_v4".to_string(), dec!(27.20)); // 8x A100
        spot.insert("Standard_ND96asr_v4".to_string(), dec!(8.16));

        // Azure ND H100 v5 series
        on_demand.insert("Standard_ND96isr_H100_v5".to_string(), dec!(81.60)); // 8x H100
        spot.insert("Standard_ND96isr_H100_v5".to_string(), dec!(24.48));

        Self { on_demand, spot }
    }

    fn get_on_demand_price(&self, vm_size: &str) -> Option<Decimal> {
        self.on_demand.get(vm_size).copied()
    }

    fn get_spot_price(&self, vm_size: &str) -> Option<Decimal> {
        self.spot.get(vm_size).copied()
    }

    fn is_supported(&self, vm_size: &str) -> bool {
        self.on_demand.contains_key(vm_size)
    }
}

pub struct AzureProvider {
    #[allow(dead_code)]
    config: AzureConfig,
    pricing: PricingData,
    instances: Arc<Mutex<HashMap<String, Instance>>>,
    instance_counter: Arc<Mutex<u64>>,
    quotas: Arc<Mutex<ServiceQuotas>>,
}

impl AzureProvider {
    pub async fn new(config: AzureConfig) -> ProviderResult<Self> {
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
        format!("azure-vm-{}", *counter)
    }
}

#[async_trait::async_trait]
impl CapacityProvider for AzureProvider {
    fn name(&self) -> &str {
        "azure"
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
                    "VM size {} not supported",
                    request.instance_type
                ))
            })?;

        let spot_rate = if request.spot {
            self.pricing.get_spot_price(&request.instance_type)
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
            provider: "azure".to_string(),
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
                "VM size {} not supported",
                spec.instance_type
            )));
        }

        let mut quotas = self.quotas.lock().unwrap();
        if quotas.current_instances + spec.count > quotas.max_instances {
            return Err(ProviderError::QuotaExceeded(
                "VM quota exceeded".to_string(),
            ));
        }

        let rate = if spec.spot {
            self.pricing
                .get_spot_price(&spec.instance_type)
                .unwrap_or_else(|| self.pricing.get_on_demand_price(&spec.instance_type).unwrap())
        } else {
            self.pricing.get_on_demand_price(&spec.instance_type).unwrap()
        };

        let mut instances = Vec::new();
        let mut instances_map = self.instances.lock().unwrap();

        for i in 0..spec.count {
            let instance_id = self.generate_instance_id();
            let instance = Instance {
                id: instance_id.clone(),
                instance_type: spec.instance_type.clone(),
                region: spec.region.clone(),
                public_ip: Some(format!("20.{}.{}.{}", i / 65536, (i / 256) % 256, i % 256)),
                private_ip: format!("10.1.{}.{}", (i / 256) % 256, i % 256),
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
                "VM {} not found",
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

        let base_price = self.pricing.get_spot_price(instance_type).ok_or_else(|| {
            ProviderError::InvalidRequest(format!(
                "spot pricing not available for {}",
                instance_type
            ))
        })?;

        Ok(SpotPrices {
            instance_type: instance_type.to_string(),
            region: region.to_string(),
            current_price: base_price * dec!(1.08),
            average_price: base_price * dec!(1.12),
            min_price: base_price * dec!(0.85),
            max_price: base_price * dec!(1.40),
        })
    }

    async fn get_quotas(&self) -> ProviderResult<ServiceQuotas> {
        Ok(self.quotas.lock().unwrap().clone())
    }

    async fn health(&self) -> ProviderResult<HealthStatus> {
        Ok(HealthStatus {
            healthy: true,
            message: "Azure Compute operational".to_string(),
        })
    }
}

