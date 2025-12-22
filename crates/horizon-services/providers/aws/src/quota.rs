use hpc_provider::{ProviderResult, ServiceQuotas};
use std::sync::Arc;
use std::sync::Mutex;

pub struct QuotaManager {
    quotas: Arc<Mutex<ServiceQuotas>>,
}

impl QuotaManager {
    pub fn new() -> Self {
        Self {
            quotas: Arc::new(Mutex::new(ServiceQuotas {
                max_instances: 100,
                current_instances: 0,
                max_vcpus: 400,
                current_vcpus: 0,
                max_gpus: 80,
                current_gpus: 0,
            })),
        }
    }

    pub fn get_quotas(&self) -> ProviderResult<ServiceQuotas> {
        Ok(self.quotas.lock().unwrap().clone())
    }

    pub fn increment_instances(&self, count: usize) {
        let mut quotas = self.quotas.lock().unwrap();
        quotas.current_instances += count;
    }

    pub fn decrement_instances(&self, count: usize) {
        let mut quotas = self.quotas.lock().unwrap();
        quotas.current_instances = quotas.current_instances.saturating_sub(count);
    }

    pub fn has_capacity(&self, required: usize) -> bool {
        let quotas = self.quotas.lock().unwrap();
        quotas.current_instances + required <= quotas.max_instances
    }
}

impl Default for QuotaManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quota_manager_creation() {
        let manager = QuotaManager::new();
        let quotas = manager.get_quotas().unwrap();
        assert_eq!(quotas.max_instances, 100);
        assert_eq!(quotas.current_instances, 0);
    }

    #[test]
    fn test_increment_instances() {
        let manager = QuotaManager::new();
        manager.increment_instances(5);
        let quotas = manager.get_quotas().unwrap();
        assert_eq!(quotas.current_instances, 5);
    }

    #[test]
    fn test_decrement_instances() {
        let manager = QuotaManager::new();
        manager.increment_instances(10);
        manager.decrement_instances(3);
        let quotas = manager.get_quotas().unwrap();
        assert_eq!(quotas.current_instances, 7);
    }

    #[test]
    fn test_decrement_below_zero() {
        let manager = QuotaManager::new();
        manager.decrement_instances(5);
        let quotas = manager.get_quotas().unwrap();
        assert_eq!(quotas.current_instances, 0);
    }

    #[test]
    fn test_has_capacity() {
        let manager = QuotaManager::new();
        assert!(manager.has_capacity(50));
        assert!(manager.has_capacity(100));
        assert!(!manager.has_capacity(101));
    }

    #[test]
    fn test_has_capacity_after_increment() {
        let manager = QuotaManager::new();
        manager.increment_instances(90);
        assert!(manager.has_capacity(10));
        assert!(!manager.has_capacity(11));
    }
}
