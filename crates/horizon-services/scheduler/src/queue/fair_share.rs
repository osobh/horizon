use std::collections::HashMap;
use chrono::{DateTime, Duration, Utc};

/// Resource identifier for tracking usage per resource type
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResourceKey {
    GpuHours,
    CpuHours,
    MemoryGbHours,
    StorageGbHours,
    NetworkGbpsHours,
    Custom(String),
}

impl ResourceKey {
    pub fn as_str(&self) -> String {
        match self {
            ResourceKey::GpuHours => "gpu_hours".to_string(),
            ResourceKey::CpuHours => "cpu_hours".to_string(),
            ResourceKey::MemoryGbHours => "memory_gb_hours".to_string(),
            ResourceKey::StorageGbHours => "storage_gb_hours".to_string(),
            ResourceKey::NetworkGbpsHours => "network_gbps_hours".to_string(),
            ResourceKey::Custom(s) => s.clone(),
        }
    }
}

/// Fair-share calculator using Weighted Fair Queuing (WFQ) algorithm
/// Now supports multiple resource types for comprehensive fair-share tracking
pub struct FairShareCalculator {
    /// Historical usage by user per resource type: (user_id, resource_type) -> usage
    usage_history: HashMap<(String, ResourceKey), f64>,
    /// User weights/shares (default 1.0 for equal share)
    user_weights: HashMap<String, f64>,
    /// Decay factor for historical usage (0.0-1.0)
    decay_factor: f64,
    /// Last decay timestamp
    last_decay: DateTime<Utc>,
    /// Decay interval in hours
    decay_interval_hours: i64,
}

impl FairShareCalculator {
    pub fn new() -> Self {
        Self {
            usage_history: HashMap::new(),
            user_weights: HashMap::new(),
            decay_factor: 0.5, // Half-life decay
            last_decay: Utc::now(),
            decay_interval_hours: 24,
        }
    }

    pub fn with_decay(mut self, factor: f64, interval_hours: i64) -> Self {
        self.decay_factor = factor;
        self.decay_interval_hours = interval_hours;
        self
    }

    /// Set user weight (share allocation)
    pub fn set_user_weight(&mut self, user_id: &str, weight: f64) {
        self.user_weights.insert(user_id.to_string(), weight);
    }

    /// Record resource usage for a user
    pub fn record_usage(&mut self, user_id: &str, resource_key: ResourceKey, hours: f64) {
        let key = (user_id.to_string(), resource_key);
        *self.usage_history.entry(key).or_insert(0.0) += hours;
    }

    /// Record GPU usage for a user (backward compatible)
    pub fn record_gpu_usage(&mut self, user_id: &str, gpu_hours: f64) {
        self.record_usage(user_id, ResourceKey::GpuHours, gpu_hours);
    }

    /// Calculate fair-share priority for a user across all resource types (higher is better)
    /// Priority = weight / (total_normalized_usage + 1.0)
    pub fn calculate_priority(&mut self, user_id: &str) -> f64 {
        self.apply_decay();

        let weight = self.user_weights.get(user_id).copied().unwrap_or(1.0);

        // Sum usage across all resource types for this user
        let total_usage: f64 = self.usage_history
            .iter()
            .filter(|((uid, _), _)| uid == user_id)
            .map(|(_, usage)| usage)
            .sum();

        // Avoid division by zero and give advantage to users with less usage
        weight / (total_usage + 1.0)
    }

    /// Calculate fair-share priority for a user for a specific resource type
    pub fn calculate_priority_for_resource(&mut self, user_id: &str, resource_key: &ResourceKey) -> f64 {
        self.apply_decay();

        let weight = self.user_weights.get(user_id).copied().unwrap_or(1.0);
        let key = (user_id.to_string(), resource_key.clone());
        let usage = self.usage_history.get(&key).copied().unwrap_or(0.0);

        // Avoid division by zero and give advantage to users with less usage
        weight / (usage + 1.0)
    }

    /// Apply exponential decay to historical usage
    fn apply_decay(&mut self) {
        let now = Utc::now();
        let elapsed = now.signed_duration_since(self.last_decay);

        if elapsed >= Duration::hours(self.decay_interval_hours) {
            let decay_count = elapsed.num_hours() / self.decay_interval_hours;
            let total_decay = self.decay_factor.powi(decay_count as i32);

            for usage in self.usage_history.values_mut() {
                *usage *= total_decay;
            }

            self.last_decay = now;
        }
    }

    /// Get current total usage for a user across all resources
    pub fn get_usage(&mut self, user_id: &str) -> f64 {
        self.apply_decay();
        self.usage_history
            .iter()
            .filter(|((uid, _), _)| uid == user_id)
            .map(|(_, usage)| usage)
            .sum()
    }

    /// Get current usage for a user for a specific resource type
    pub fn get_usage_for_resource(&mut self, user_id: &str, resource_key: &ResourceKey) -> f64 {
        self.apply_decay();
        let key = (user_id.to_string(), resource_key.clone());
        self.usage_history.get(&key).copied().unwrap_or(0.0)
    }

    /// Get breakdown of usage by resource type for a user
    pub fn get_usage_breakdown(&mut self, user_id: &str) -> HashMap<ResourceKey, f64> {
        self.apply_decay();
        self.usage_history
            .iter()
            .filter(|((uid, _), _)| uid == user_id)
            .map(|((_, resource_key), usage)| (resource_key.clone(), *usage))
            .collect()
    }

    /// Get all user priorities
    pub fn get_all_priorities(&mut self) -> HashMap<String, f64> {
        let mut users: Vec<String> = self
            .usage_history
            .keys()
            .map(|(user_id, _)| user_id.clone())
            .chain(self.user_weights.keys().cloned())
            .collect();
        users.sort();
        users.dedup();

        users
            .into_iter()
            .map(|user| {
                let priority = self.calculate_priority(&user);
                (user, priority)
            })
            .collect()
    }

    /// Get all user priorities for a specific resource type
    pub fn get_all_priorities_for_resource(&mut self, resource_key: &ResourceKey) -> HashMap<String, f64> {
        let mut users: Vec<String> = self
            .usage_history
            .keys()
            .filter(|(_, rkey)| rkey == resource_key)
            .map(|(user_id, _)| user_id.clone())
            .chain(self.user_weights.keys().cloned())
            .collect();
        users.sort();
        users.dedup();

        users
            .into_iter()
            .map(|user| {
                let priority = self.calculate_priority_for_resource(&user, resource_key);
                (user, priority)
            })
            .collect()
    }

    /// Reset all usage history
    pub fn reset(&mut self) {
        self.usage_history.clear();
        self.last_decay = Utc::now();
    }
}

impl Default for FairShareCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equal_share_no_usage() {
        let mut calc = FairShareCalculator::new();

        let p1 = calc.calculate_priority("user1");
        let p2 = calc.calculate_priority("user2");

        // Equal priority for users with no usage
        assert!((p1 - p2).abs() < 0.001);
    }

    #[test]
    fn test_priority_decreases_with_usage() {
        let mut calc = FairShareCalculator::new();

        let p1_before = calc.calculate_priority("user1");

        calc.record_gpu_usage("user1", 10.0);

        let p1_after = calc.calculate_priority("user1");

        assert!(p1_after < p1_before);
    }

    #[test]
    fn test_user_with_less_usage_gets_higher_priority() {
        let mut calc = FairShareCalculator::new();

        calc.record_gpu_usage("heavy_user", 100.0);
        calc.record_gpu_usage("light_user", 10.0);

        let heavy = calc.calculate_priority("heavy_user");
        let light = calc.calculate_priority("light_user");

        assert!(light > heavy);
    }

    #[test]
    fn test_user_weights() {
        let mut calc = FairShareCalculator::new();

        calc.set_user_weight("premium", 2.0);
        calc.set_user_weight("standard", 1.0);

        calc.record_gpu_usage("premium", 10.0);
        calc.record_gpu_usage("standard", 10.0);

        let premium = calc.calculate_priority("premium");
        let standard = calc.calculate_priority("standard");

        // Premium user should have higher priority even with same usage
        assert!(premium > standard);
    }

    #[test]
    fn test_record_usage_accumulates() {
        let mut calc = FairShareCalculator::new();

        calc.record_gpu_usage("user1", 5.0);
        calc.record_gpu_usage("user1", 3.0);

        let usage = calc.get_usage("user1");
        assert!((usage - 8.0).abs() < 0.001);
    }

    #[test]
    fn test_get_all_priorities() {
        let mut calc = FairShareCalculator::new();

        calc.record_gpu_usage("user1", 10.0);
        calc.record_gpu_usage("user2", 20.0);
        calc.set_user_weight("user3", 2.0);

        let priorities = calc.get_all_priorities();

        assert!(priorities.contains_key("user1"));
        assert!(priorities.contains_key("user2"));
        assert!(priorities.contains_key("user3"));
        assert!(priorities.len() >= 3);
    }

    #[test]
    fn test_reset() {
        let mut calc = FairShareCalculator::new();

        calc.record_gpu_usage("user1", 100.0);
        calc.record_gpu_usage("user2", 50.0);

        calc.reset();

        assert_eq!(calc.get_usage("user1"), 0.0);
        assert_eq!(calc.get_usage("user2"), 0.0);
    }

    #[test]
    fn test_priority_formula() {
        let mut calc = FairShareCalculator::new();

        // For user with weight 1.0 and usage 0.0
        let p = calc.calculate_priority("new_user");
        assert!((p - 1.0).abs() < 0.001); // 1.0 / (0.0 + 1.0) = 1.0

        // For user with weight 1.0 and usage 9.0
        calc.record_gpu_usage("existing_user", 9.0);
        let p = calc.calculate_priority("existing_user");
        assert!((p - 0.1).abs() < 0.001); // 1.0 / (9.0 + 1.0) = 0.1
    }

    // New tests for multi-resource tracking

    #[test]
    fn test_multi_resource_tracking() {
        let mut calc = FairShareCalculator::new();

        calc.record_usage("user1", ResourceKey::GpuHours, 10.0);
        calc.record_usage("user1", ResourceKey::CpuHours, 5.0);
        calc.record_usage("user1", ResourceKey::MemoryGbHours, 20.0);

        let total_usage = calc.get_usage("user1");
        assert!((total_usage - 35.0).abs() < 0.001);

        let gpu_usage = calc.get_usage_for_resource("user1", &ResourceKey::GpuHours);
        assert!((gpu_usage - 10.0).abs() < 0.001);

        let cpu_usage = calc.get_usage_for_resource("user1", &ResourceKey::CpuHours);
        assert!((cpu_usage - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_usage_breakdown() {
        let mut calc = FairShareCalculator::new();

        calc.record_usage("user1", ResourceKey::GpuHours, 10.0);
        calc.record_usage("user1", ResourceKey::CpuHours, 5.0);
        calc.record_usage("user1", ResourceKey::StorageGbHours, 100.0);

        let breakdown = calc.get_usage_breakdown("user1");

        assert_eq!(breakdown.len(), 3);
        assert_eq!(breakdown.get(&ResourceKey::GpuHours), Some(&10.0));
        assert_eq!(breakdown.get(&ResourceKey::CpuHours), Some(&5.0));
        assert_eq!(breakdown.get(&ResourceKey::StorageGbHours), Some(&100.0));
    }

    #[test]
    fn test_priority_for_specific_resource() {
        let mut calc = FairShareCalculator::new();

        calc.record_usage("user1", ResourceKey::GpuHours, 10.0);
        calc.record_usage("user1", ResourceKey::CpuHours, 50.0);

        // User1 has higher GPU priority (less GPU usage)
        let gpu_priority = calc.calculate_priority_for_resource("user1", &ResourceKey::GpuHours);
        // User1 has lower CPU priority (more CPU usage)
        let cpu_priority = calc.calculate_priority_for_resource("user1", &ResourceKey::CpuHours);

        assert!(gpu_priority > cpu_priority);
    }

    #[test]
    fn test_multi_resource_fair_share_comparison() {
        let mut calc = FairShareCalculator::new();

        // User1: Heavy GPU user, light CPU user
        calc.record_usage("user1", ResourceKey::GpuHours, 100.0);
        calc.record_usage("user1", ResourceKey::CpuHours, 10.0);

        // User2: Light GPU user, heavy CPU user
        calc.record_usage("user2", ResourceKey::GpuHours, 10.0);
        calc.record_usage("user2", ResourceKey::CpuHours, 100.0);

        // For GPU scheduling, user2 should have higher priority
        let gpu_p1 = calc.calculate_priority_for_resource("user1", &ResourceKey::GpuHours);
        let gpu_p2 = calc.calculate_priority_for_resource("user2", &ResourceKey::GpuHours);
        assert!(gpu_p2 > gpu_p1);

        // For CPU scheduling, user1 should have higher priority
        let cpu_p1 = calc.calculate_priority_for_resource("user1", &ResourceKey::CpuHours);
        let cpu_p2 = calc.calculate_priority_for_resource("user2", &ResourceKey::CpuHours);
        assert!(cpu_p1 > cpu_p2);
    }

    #[test]
    fn test_backward_compatible_gpu_usage() {
        let mut calc = FairShareCalculator::new();

        // Old API
        calc.record_gpu_usage("user1", 10.0);

        // Should be recorded as GpuHours
        let gpu_usage = calc.get_usage_for_resource("user1", &ResourceKey::GpuHours);
        assert!((gpu_usage - 10.0).abs() < 0.001);
    }
}
