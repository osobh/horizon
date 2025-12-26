//! Capability-based access control for ephemeral identities.
//!
//! Capabilities follow the pattern `action:resource` and support wildcards.
//! They can only be reduced (never expanded) when deriving new tokens.

use std::collections::HashSet;

use chrono::{DateTime, Datelike, NaiveTime, Utc, Weekday};
use serde::{Deserialize, Serialize};

use crate::error::{EphemeralError, Result};

/// A single capability representing an allowed action on a resource.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Capability {
    /// The action (e.g., "read", "write", "delete", "*").
    pub action: String,
    /// The resource (e.g., "notebooks", "models/*", "*").
    pub resource: String,
}

impl Capability {
    /// Creates a new capability.
    #[must_use]
    pub fn new(action: impl Into<String>, resource: impl Into<String>) -> Self {
        Self {
            action: action.into(),
            resource: resource.into(),
        }
    }

    /// Creates a wildcard capability (allows everything).
    #[must_use]
    pub fn wildcard() -> Self {
        Self {
            action: "*".to_string(),
            resource: "*".to_string(),
        }
    }

    /// Parses a capability from a string like "read:notebooks".
    ///
    /// # Errors
    ///
    /// Returns an error if the string format is invalid.
    pub fn parse(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Err(EphemeralError::CapabilityDenied(format!(
                "Invalid capability format: {s}"
            )));
        }
        Ok(Self {
            action: parts[0].to_string(),
            resource: parts[1].to_string(),
        })
    }

    /// Converts the capability to its string representation.
    #[must_use]
    pub fn to_string_repr(&self) -> String {
        format!("{}:{}", self.action, self.resource)
    }

    /// Checks if this capability allows the given action on the resource.
    #[must_use]
    pub fn allows(&self, action: &str, resource: &str) -> bool {
        let action_matches = self.action == "*" || self.action == action;
        let resource_matches = self.matches_resource(resource);
        action_matches && resource_matches
    }

    /// Checks if this capability's resource pattern matches the given resource.
    fn matches_resource(&self, resource: &str) -> bool {
        if self.resource == "*" {
            return true;
        }

        if self.resource.ends_with("/*") {
            let prefix = &self.resource[..self.resource.len() - 1];
            return resource.starts_with(prefix) || resource == &self.resource[..self.resource.len() - 2];
        }

        if self.resource.ends_with('*') {
            let prefix = &self.resource[..self.resource.len() - 1];
            return resource.starts_with(prefix);
        }

        self.resource == resource
    }

    /// Checks if this capability is a subset of another capability.
    #[must_use]
    pub fn is_subset_of(&self, other: &Capability) -> bool {
        // Wildcard parent includes everything
        if other.action == "*" && other.resource == "*" {
            return true;
        }

        // Action must match or parent is wildcard
        let action_ok = other.action == "*" || self.action == other.action;

        // Resource must be within parent's scope
        let resource_ok = other.matches_resource(&self.resource);

        action_ok && resource_ok
    }
}

impl std::fmt::Display for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.action, self.resource)
    }
}

/// Rate limits for ephemeral identity operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RateLimits {
    /// Maximum operations per minute.
    pub operations_per_minute: u32,
    /// Maximum bytes transferred per minute.
    pub bytes_per_minute: u64,
    /// Maximum concurrent connections.
    pub max_connections: u32,
    /// Per-resource limits (resource pattern -> ops/min).
    #[serde(default)]
    pub resource_limits: std::collections::HashMap<String, u32>,
}

impl Default for RateLimits {
    fn default() -> Self {
        Self {
            operations_per_minute: 60,
            bytes_per_minute: 10 * 1024 * 1024, // 10 MB/min
            max_connections: 5,
            resource_limits: std::collections::HashMap::new(),
        }
    }
}

impl RateLimits {
    /// Creates rate limits with custom operations per minute.
    #[must_use]
    pub fn with_ops_per_minute(ops: u32) -> Self {
        Self {
            operations_per_minute: ops,
            ..Default::default()
        }
    }

    /// Sets the bytes per minute limit.
    #[must_use]
    pub fn with_bytes_per_minute(mut self, bytes: u64) -> Self {
        self.bytes_per_minute = bytes;
        self
    }

    /// Sets the max connections limit.
    #[must_use]
    pub fn with_max_connections(mut self, max: u32) -> Self {
        self.max_connections = max;
        self
    }

    /// Adds a resource-specific rate limit.
    #[must_use]
    pub fn with_resource_limit(mut self, resource: impl Into<String>, limit: u32) -> Self {
        self.resource_limits.insert(resource.into(), limit);
        self
    }

    /// Returns the stricter of two rate limits.
    #[must_use]
    pub fn intersect(&self, other: &RateLimits) -> RateLimits {
        let mut resource_limits = self.resource_limits.clone();
        for (k, v) in &other.resource_limits {
            resource_limits
                .entry(k.clone())
                .and_modify(|e| *e = (*e).min(*v))
                .or_insert(*v);
        }

        RateLimits {
            operations_per_minute: self.operations_per_minute.min(other.operations_per_minute),
            bytes_per_minute: self.bytes_per_minute.min(other.bytes_per_minute),
            max_connections: self.max_connections.min(other.max_connections),
            resource_limits,
        }
    }

    /// Gets the limit for a specific resource, falling back to default.
    #[must_use]
    pub fn limit_for_resource(&self, resource: &str) -> u32 {
        // Check exact match first
        if let Some(&limit) = self.resource_limits.get(resource) {
            return limit;
        }

        // Check wildcard patterns
        for (pattern, &limit) in &self.resource_limits {
            if pattern.ends_with('*') {
                let prefix = &pattern[..pattern.len() - 1];
                if resource.starts_with(prefix) {
                    return limit;
                }
            }
        }

        self.operations_per_minute
    }
}

/// Time restrictions for when ephemeral access is allowed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeRestrictions {
    /// Allowed start time (inclusive).
    pub start_time: Option<NaiveTime>,
    /// Allowed end time (exclusive).
    pub end_time: Option<NaiveTime>,
    /// Allowed days of the week (empty = all days).
    #[serde(default)]
    pub allowed_days: HashSet<Weekday>,
    /// Timezone for time calculations.
    #[serde(default = "default_timezone")]
    pub timezone: String,
    /// Blackout periods when access is never allowed.
    #[serde(default)]
    pub blackout_periods: Vec<BlackoutPeriod>,
}

fn default_timezone() -> String {
    "UTC".to_string()
}

/// A blackout period when access is not allowed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlackoutPeriod {
    /// Start of blackout (inclusive).
    pub start: DateTime<Utc>,
    /// End of blackout (exclusive).
    pub end: DateTime<Utc>,
    /// Reason for the blackout.
    pub reason: Option<String>,
}

impl Default for TimeRestrictions {
    fn default() -> Self {
        Self {
            start_time: None,
            end_time: None,
            allowed_days: HashSet::new(),
            timezone: "UTC".to_string(),
            blackout_periods: Vec::new(),
        }
    }
}

impl TimeRestrictions {
    /// Creates time restrictions with no limitations.
    #[must_use]
    pub fn unrestricted() -> Self {
        Self::default()
    }

    /// Creates business hours restrictions (9 AM - 5 PM, Mon-Fri).
    #[must_use]
    pub fn business_hours() -> Self {
        let mut allowed_days = HashSet::new();
        allowed_days.insert(Weekday::Mon);
        allowed_days.insert(Weekday::Tue);
        allowed_days.insert(Weekday::Wed);
        allowed_days.insert(Weekday::Thu);
        allowed_days.insert(Weekday::Fri);

        Self {
            start_time: Some(NaiveTime::from_hms_opt(9, 0, 0).unwrap()),
            end_time: Some(NaiveTime::from_hms_opt(17, 0, 0).unwrap()),
            allowed_days,
            timezone: "UTC".to_string(),
            blackout_periods: Vec::new(),
        }
    }

    /// Sets the allowed time window.
    #[must_use]
    pub fn with_time_window(mut self, start: NaiveTime, end: NaiveTime) -> Self {
        self.start_time = Some(start);
        self.end_time = Some(end);
        self
    }

    /// Adds an allowed day.
    #[must_use]
    pub fn with_allowed_day(mut self, day: Weekday) -> Self {
        self.allowed_days.insert(day);
        self
    }

    /// Adds a blackout period.
    #[must_use]
    pub fn with_blackout(mut self, start: DateTime<Utc>, end: DateTime<Utc>, reason: Option<String>) -> Self {
        self.blackout_periods.push(BlackoutPeriod { start, end, reason });
        self
    }

    /// Checks if access is allowed at the given time.
    #[must_use]
    pub fn is_allowed_at(&self, time: DateTime<Utc>) -> bool {
        // Check blackout periods first
        for blackout in &self.blackout_periods {
            if time >= blackout.start && time < blackout.end {
                return false;
            }
        }

        // Check allowed days (empty means all days allowed)
        if !self.allowed_days.is_empty() {
            let weekday = time.weekday();
            if !self.allowed_days.contains(&weekday) {
                return false;
            }
        }

        // Check time window
        let time_of_day = time.time();
        if let (Some(start), Some(end)) = (self.start_time, self.end_time) {
            if start <= end {
                // Normal range (e.g., 9:00 - 17:00)
                if time_of_day < start || time_of_day >= end {
                    return false;
                }
            } else {
                // Overnight range (e.g., 22:00 - 06:00)
                if time_of_day < start && time_of_day >= end {
                    return false;
                }
            }
        }

        true
    }

    /// Returns the stricter of two time restrictions.
    #[must_use]
    pub fn intersect(&self, other: &TimeRestrictions) -> TimeRestrictions {
        // Combine allowed days (intersection, or keep if one is empty)
        let allowed_days = if self.allowed_days.is_empty() {
            other.allowed_days.clone()
        } else if other.allowed_days.is_empty() {
            self.allowed_days.clone()
        } else {
            self.allowed_days.intersection(&other.allowed_days).copied().collect()
        };

        // Take the later start time and earlier end time
        let start_time = match (self.start_time, other.start_time) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, None) => a,
            (None, b) => b,
        };

        let end_time = match (self.end_time, other.end_time) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, None) => a,
            (None, b) => b,
        };

        // Combine all blackout periods
        let mut blackout_periods = self.blackout_periods.clone();
        blackout_periods.extend(other.blackout_periods.clone());

        TimeRestrictions {
            start_time,
            end_time,
            allowed_days,
            timezone: self.timezone.clone(),
            blackout_periods,
        }
    }
}

/// A set of capabilities with rate limits and time restrictions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilitySet {
    /// Allowed capabilities.
    pub capabilities: HashSet<Capability>,
    /// Explicitly denied patterns (take precedence over capabilities).
    pub denials: HashSet<String>,
    /// Rate limits for this capability set.
    pub rate_limits: RateLimits,
    /// Time restrictions for when capabilities are active.
    pub time_restrictions: TimeRestrictions,
}

impl Default for CapabilitySet {
    fn default() -> Self {
        Self::new()
    }
}

impl CapabilitySet {
    /// Creates an empty capability set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            capabilities: HashSet::new(),
            denials: HashSet::new(),
            rate_limits: RateLimits::default(),
            time_restrictions: TimeRestrictions::default(),
        }
    }

    /// Creates a capability set with full access (admin).
    #[must_use]
    pub fn full_access() -> Self {
        let mut caps = HashSet::new();
        caps.insert(Capability::wildcard());
        Self {
            capabilities: caps,
            denials: HashSet::new(),
            rate_limits: RateLimits::default(),
            time_restrictions: TimeRestrictions::default(),
        }
    }

    /// Adds a capability to the set.
    #[must_use]
    pub fn with_capability(mut self, cap: Capability) -> Self {
        self.capabilities.insert(cap);
        self
    }

    /// Adds a denial pattern.
    #[must_use]
    pub fn with_denial(mut self, pattern: impl Into<String>) -> Self {
        self.denials.insert(pattern.into());
        self
    }

    /// Sets the rate limits.
    #[must_use]
    pub fn with_rate_limits(mut self, limits: RateLimits) -> Self {
        self.rate_limits = limits;
        self
    }

    /// Sets the time restrictions.
    #[must_use]
    pub fn with_time_restrictions(mut self, restrictions: TimeRestrictions) -> Self {
        self.time_restrictions = restrictions;
        self
    }

    /// Checks if this capability set allows the action on the resource.
    #[must_use]
    pub fn allows(&self, action: &str, resource: &str) -> bool {
        // Check denials first (they take precedence)
        for denial in &self.denials {
            if let Ok(cap) = Capability::parse(denial) {
                if cap.allows(action, resource) {
                    return false;
                }
            }
        }

        // Check if any capability allows this action
        self.capabilities.iter().any(|cap| cap.allows(action, resource))
    }

    /// Checks if access is allowed at the current time.
    #[must_use]
    pub fn is_time_allowed(&self) -> bool {
        self.time_restrictions.is_allowed_at(Utc::now())
    }

    /// Validates an action and returns an error if denied.
    ///
    /// # Errors
    ///
    /// Returns an error if the action is not allowed or time restrictions apply.
    pub fn validate(&self, action: &str, resource: &str) -> Result<()> {
        if !self.is_time_allowed() {
            return Err(EphemeralError::TimeWindowViolation);
        }

        if !self.allows(action, resource) {
            return Err(EphemeralError::CapabilityDenied(format!(
                "{action}:{resource}"
            )));
        }

        Ok(())
    }

    /// Creates an intersection (subset) with another capability set.
    /// Used when deriving a more restricted token from a parent.
    #[must_use]
    pub fn intersect(&self, other: &CapabilitySet) -> CapabilitySet {
        // Only keep capabilities that exist in both sets
        let capabilities = self
            .capabilities
            .iter()
            .filter(|cap| {
                other.capabilities.iter().any(|other_cap| cap.is_subset_of(other_cap))
            })
            .cloned()
            .collect();

        // Combine denials
        let mut denials = self.denials.clone();
        denials.extend(other.denials.clone());

        CapabilitySet {
            capabilities,
            denials,
            rate_limits: self.rate_limits.intersect(&other.rate_limits),
            time_restrictions: self.time_restrictions.intersect(&other.time_restrictions),
        }
    }

    /// Checks if this capability set is a subset of another.
    #[must_use]
    pub fn is_subset_of(&self, other: &CapabilitySet) -> bool {
        self.capabilities.iter().all(|cap| {
            other.capabilities.iter().any(|other_cap| cap.is_subset_of(other_cap))
        })
    }

    /// Returns the number of capabilities in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.capabilities.len()
    }

    /// Returns true if there are no capabilities.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.capabilities.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Capability Tests ===

    #[test]
    fn test_capability_new() {
        let cap = Capability::new("read", "notebooks");
        assert_eq!(cap.action, "read");
        assert_eq!(cap.resource, "notebooks");
    }

    #[test]
    fn test_capability_wildcard() {
        let cap = Capability::wildcard();
        assert_eq!(cap.action, "*");
        assert_eq!(cap.resource, "*");
    }

    #[test]
    fn test_capability_parse() {
        let cap = Capability::parse("write:models/gpt4").unwrap();
        assert_eq!(cap.action, "write");
        assert_eq!(cap.resource, "models/gpt4");
    }

    #[test]
    fn test_capability_parse_invalid() {
        let result = Capability::parse("invalid-format");
        assert!(result.is_err());
    }

    #[test]
    fn test_capability_to_string() {
        let cap = Capability::new("delete", "jobs/*");
        assert_eq!(cap.to_string(), "delete:jobs/*");
    }

    #[test]
    fn test_capability_allows_exact_match() {
        let cap = Capability::new("read", "notebooks");
        assert!(cap.allows("read", "notebooks"));
        assert!(!cap.allows("write", "notebooks"));
        assert!(!cap.allows("read", "models"));
    }

    #[test]
    fn test_capability_allows_action_wildcard() {
        let cap = Capability::new("*", "notebooks");
        assert!(cap.allows("read", "notebooks"));
        assert!(cap.allows("write", "notebooks"));
        assert!(cap.allows("delete", "notebooks"));
        assert!(!cap.allows("read", "models"));
    }

    #[test]
    fn test_capability_allows_resource_wildcard() {
        let cap = Capability::new("read", "*");
        assert!(cap.allows("read", "notebooks"));
        assert!(cap.allows("read", "models"));
        assert!(cap.allows("read", "anything/nested/deep"));
        assert!(!cap.allows("write", "notebooks"));
    }

    #[test]
    fn test_capability_allows_full_wildcard() {
        let cap = Capability::wildcard();
        assert!(cap.allows("read", "notebooks"));
        assert!(cap.allows("write", "models"));
        assert!(cap.allows("delete", "users/123"));
    }

    #[test]
    fn test_capability_allows_prefix_wildcard() {
        let cap = Capability::new("read", "models/*");
        assert!(cap.allows("read", "models"));
        assert!(cap.allows("read", "models/gpt4"));
        assert!(cap.allows("read", "models/llama/7b"));
        assert!(!cap.allows("read", "notebooks"));
    }

    #[test]
    fn test_capability_allows_suffix_wildcard() {
        let cap = Capability::new("read", "project-*");
        assert!(cap.allows("read", "project-alpha"));
        assert!(cap.allows("read", "project-beta-v2"));
        assert!(!cap.allows("read", "notebooks"));
    }

    #[test]
    fn test_capability_is_subset_of() {
        let parent = Capability::wildcard();
        let child = Capability::new("read", "notebooks");
        assert!(child.is_subset_of(&parent));
        assert!(!parent.is_subset_of(&child));

        let action_wildcard = Capability::new("*", "notebooks");
        assert!(child.is_subset_of(&action_wildcard));

        let resource_wildcard = Capability::new("read", "*");
        assert!(child.is_subset_of(&resource_wildcard));
    }

    // === RateLimits Tests ===

    #[test]
    fn test_rate_limits_default() {
        let limits = RateLimits::default();
        assert_eq!(limits.operations_per_minute, 60);
        assert_eq!(limits.bytes_per_minute, 10 * 1024 * 1024);
        assert_eq!(limits.max_connections, 5);
    }

    #[test]
    fn test_rate_limits_builder() {
        let limits = RateLimits::with_ops_per_minute(100)
            .with_bytes_per_minute(1024 * 1024)
            .with_max_connections(10)
            .with_resource_limit("api/heavy", 10);

        assert_eq!(limits.operations_per_minute, 100);
        assert_eq!(limits.bytes_per_minute, 1024 * 1024);
        assert_eq!(limits.max_connections, 10);
        assert_eq!(limits.resource_limits.get("api/heavy"), Some(&10));
    }

    #[test]
    fn test_rate_limits_intersect() {
        let a = RateLimits::with_ops_per_minute(100)
            .with_resource_limit("api", 50);
        let b = RateLimits::with_ops_per_minute(50)
            .with_resource_limit("api", 75)
            .with_resource_limit("data", 30);

        let result = a.intersect(&b);
        assert_eq!(result.operations_per_minute, 50); // min(100, 50)
        assert_eq!(result.resource_limits.get("api"), Some(&50)); // min(50, 75)
        assert_eq!(result.resource_limits.get("data"), Some(&30));
    }

    #[test]
    fn test_rate_limits_for_resource() {
        let limits = RateLimits::with_ops_per_minute(100)
            .with_resource_limit("api/heavy", 10)
            .with_resource_limit("api/*", 30);

        assert_eq!(limits.limit_for_resource("api/heavy"), 10);
        assert_eq!(limits.limit_for_resource("api/light"), 30);
        assert_eq!(limits.limit_for_resource("other"), 100);
    }

    // === TimeRestrictions Tests ===

    #[test]
    fn test_time_restrictions_default() {
        let restrictions = TimeRestrictions::default();
        assert!(restrictions.start_time.is_none());
        assert!(restrictions.end_time.is_none());
        assert!(restrictions.allowed_days.is_empty());
    }

    #[test]
    fn test_time_restrictions_business_hours() {
        let restrictions = TimeRestrictions::business_hours();
        assert_eq!(
            restrictions.start_time,
            Some(NaiveTime::from_hms_opt(9, 0, 0).unwrap())
        );
        assert_eq!(
            restrictions.end_time,
            Some(NaiveTime::from_hms_opt(17, 0, 0).unwrap())
        );
        assert!(restrictions.allowed_days.contains(&Weekday::Mon));
        assert!(restrictions.allowed_days.contains(&Weekday::Fri));
        assert!(!restrictions.allowed_days.contains(&Weekday::Sat));
    }

    #[test]
    fn test_time_restrictions_is_allowed_unrestricted() {
        let restrictions = TimeRestrictions::unrestricted();
        let now = Utc::now();
        assert!(restrictions.is_allowed_at(now));
    }

    #[test]
    fn test_time_restrictions_is_allowed_during_window() {
        let restrictions = TimeRestrictions::default()
            .with_time_window(
                NaiveTime::from_hms_opt(9, 0, 0).unwrap(),
                NaiveTime::from_hms_opt(17, 0, 0).unwrap(),
            );

        // 10:00 should be allowed
        let time = Utc::now()
            .date_naive()
            .and_hms_opt(10, 0, 0)
            .unwrap()
            .and_utc();
        assert!(restrictions.is_allowed_at(time));

        // 20:00 should not be allowed
        let time = Utc::now()
            .date_naive()
            .and_hms_opt(20, 0, 0)
            .unwrap()
            .and_utc();
        assert!(!restrictions.is_allowed_at(time));
    }

    #[test]
    fn test_time_restrictions_blackout() {
        let now = Utc::now();
        let restrictions = TimeRestrictions::unrestricted()
            .with_blackout(
                now - chrono::Duration::hours(1),
                now + chrono::Duration::hours(1),
                Some("Maintenance".to_string()),
            );

        assert!(!restrictions.is_allowed_at(now));
        assert!(restrictions.is_allowed_at(now + chrono::Duration::hours(2)));
    }

    #[test]
    fn test_time_restrictions_intersect() {
        let a = TimeRestrictions::default()
            .with_time_window(
                NaiveTime::from_hms_opt(8, 0, 0).unwrap(),
                NaiveTime::from_hms_opt(18, 0, 0).unwrap(),
            )
            .with_allowed_day(Weekday::Mon)
            .with_allowed_day(Weekday::Tue);

        let b = TimeRestrictions::default()
            .with_time_window(
                NaiveTime::from_hms_opt(10, 0, 0).unwrap(),
                NaiveTime::from_hms_opt(16, 0, 0).unwrap(),
            )
            .with_allowed_day(Weekday::Mon)
            .with_allowed_day(Weekday::Wed);

        let result = a.intersect(&b);

        // Start should be later (10:00)
        assert_eq!(
            result.start_time,
            Some(NaiveTime::from_hms_opt(10, 0, 0).unwrap())
        );
        // End should be earlier (16:00)
        assert_eq!(
            result.end_time,
            Some(NaiveTime::from_hms_opt(16, 0, 0).unwrap())
        );
        // Only Mon should be in intersection
        assert!(result.allowed_days.contains(&Weekday::Mon));
        assert!(!result.allowed_days.contains(&Weekday::Tue));
        assert!(!result.allowed_days.contains(&Weekday::Wed));
    }

    // === CapabilitySet Tests ===

    #[test]
    fn test_capability_set_new() {
        let set = CapabilitySet::new();
        assert!(set.is_empty());
    }

    #[test]
    fn test_capability_set_full_access() {
        let set = CapabilitySet::full_access();
        assert!(!set.is_empty());
        assert!(set.allows("anything", "anywhere"));
    }

    #[test]
    fn test_capability_set_with_capability() {
        let set = CapabilitySet::new()
            .with_capability(Capability::new("read", "notebooks"))
            .with_capability(Capability::new("write", "notebooks"));

        assert_eq!(set.len(), 2);
        assert!(set.allows("read", "notebooks"));
        assert!(set.allows("write", "notebooks"));
        assert!(!set.allows("delete", "notebooks"));
    }

    #[test]
    fn test_capability_set_with_denial() {
        let set = CapabilitySet::full_access()
            .with_denial("delete:*");

        assert!(set.allows("read", "anything"));
        assert!(set.allows("write", "anything"));
        assert!(!set.allows("delete", "anything"));
    }

    #[test]
    fn test_capability_set_validate_success() {
        let set = CapabilitySet::new()
            .with_capability(Capability::new("read", "notebooks"));

        let result = set.validate("read", "notebooks");
        assert!(result.is_ok());
    }

    #[test]
    fn test_capability_set_validate_denied() {
        let set = CapabilitySet::new()
            .with_capability(Capability::new("read", "notebooks"));

        let result = set.validate("delete", "notebooks");
        assert!(matches!(result, Err(EphemeralError::CapabilityDenied(_))));
    }

    #[test]
    fn test_capability_set_intersect() {
        let parent = CapabilitySet::new()
            .with_capability(Capability::new("read", "*"))
            .with_capability(Capability::new("write", "notebooks"));

        let request = CapabilitySet::new()
            .with_capability(Capability::new("read", "notebooks"))
            .with_capability(Capability::new("read", "models"))
            .with_capability(Capability::new("delete", "notebooks")); // Not in parent

        // Intersect keeps capabilities from request that are subsets of parent
        let result = request.intersect(&parent);

        // Only capabilities that exist in both should remain
        assert!(result.allows("read", "notebooks")); // Subset of read:*
        assert!(result.allows("read", "models")); // Subset of read:*
        assert!(!result.allows("delete", "notebooks")); // Not a subset of any parent cap
        assert!(!result.allows("write", "notebooks")); // In parent but not in request
    }

    #[test]
    fn test_capability_set_is_subset_of() {
        let full = CapabilitySet::full_access();
        let limited = CapabilitySet::new()
            .with_capability(Capability::new("read", "notebooks"));

        assert!(limited.is_subset_of(&full));
        assert!(!full.is_subset_of(&limited));
    }

    #[test]
    fn test_capability_set_serialization() {
        let set = CapabilitySet::new()
            .with_capability(Capability::new("read", "notebooks"))
            .with_rate_limits(RateLimits::with_ops_per_minute(100));

        let json = serde_json::to_string(&set).unwrap();
        let restored: CapabilitySet = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.len(), 1);
        assert_eq!(restored.rate_limits.operations_per_minute, 100);
    }
}
