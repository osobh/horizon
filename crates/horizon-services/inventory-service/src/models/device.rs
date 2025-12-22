use serde::{Deserialize, Serialize};
use sqlx::types::{chrono::NaiveDateTime, Json};
use sqlx::FromRow;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, sqlx::Type)]
#[sqlx(type_name = "device_type", rename_all = "snake_case")]
pub enum DeviceType {
    Server,
    Desktop,
    Laptop,
    RaspberryPi,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailabilitySchedule {
    /// Days of week when device is typically available (0=Sunday, 6=Saturday)
    pub days: Vec<u8>,
    /// Hours of day when device is typically available (0-23)
    pub hours: Vec<u8>,
    /// Timezone (e.g., "America/New_York")
    pub timezone: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct DeviceAsset {
    pub id: Uuid,
    pub asset_type: String,
    pub device_type: Option<DeviceType>,
    pub hostname: Option<String>,
    pub has_battery: Option<bool>,
    pub power_profile: Option<String>,
    pub availability_schedule: Option<Json<AvailabilitySchedule>>,
    pub created_at: NaiveDateTime,
    pub updated_at: NaiveDateTime,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct UptimeSession {
    pub id: i64,
    pub asset_id: Uuid,
    pub online_at: NaiveDateTime,
    pub offline_at: Option<NaiveDateTime>,
    pub duration_minutes: Option<i32>,
    pub battery_percent: Option<i16>,
    pub charging: Option<bool>,
    pub thermal_state: Option<String>,
    pub created_at: NaiveDateTime,
}

impl UptimeSession {
    /// Create a new uptime session (when node comes online)
    pub fn new(asset_id: Uuid, online_at: NaiveDateTime) -> Self {
        Self {
            id: 0, // Will be assigned by database
            asset_id,
            online_at,
            offline_at: None,
            duration_minutes: None,
            battery_percent: None,
            charging: None,
            thermal_state: None,
            created_at: online_at,
        }
    }

    /// Mark session as ended (when node goes offline)
    pub fn end_session(&mut self, offline_at: NaiveDateTime) {
        self.offline_at = Some(offline_at);

        // Calculate duration in minutes
        let duration = offline_at.signed_duration_since(self.online_at);
        self.duration_minutes = Some(duration.num_minutes() as i32);
    }

    /// Set battery metrics
    pub fn set_battery_metrics(&mut self, battery_percent: i16, charging: bool) {
        self.battery_percent = Some(battery_percent);
        self.charging = Some(charging);
    }

    /// Set thermal state
    pub fn set_thermal_state(&mut self, state: &str) {
        self.thermal_state = Some(state.to_string());
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct UptimeStats {
    pub asset_id: Uuid,
    pub total_sessions: i64,
    pub avg_session_duration_minutes: Option<f64>,
    pub total_uptime_minutes: Option<i64>,
    pub uptime_percent_30d: Option<f64>,
    pub reliability_score: Option<f64>,
    pub last_online_at: Option<NaiveDateTime>,
    pub typical_online_hours: Option<Vec<i32>>,
    pub updated_at: NaiveDateTime,
}

impl UptimeStats {
    /// Determine node tier based on reliability score
    pub fn suggested_tier(&self) -> NodeTier {
        match self.reliability_score.unwrap_or(0.0) {
            score if score >= 0.9 => NodeTier::Tier0,
            score if score >= 0.7 => NodeTier::Tier1,
            score if score >= 0.4 => NodeTier::Tier2,
            _ => NodeTier::Tier3,
        }
    }

    /// Check if node is likely available at given hour (0-23)
    pub fn is_likely_available_at_hour(&self, hour: i32) -> bool {
        if let Some(ref hours) = self.typical_online_hours {
            hours.contains(&hour)
        } else {
            false
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeTier {
    Tier0, // Servers (always-on, high reliability, 90%+ uptime)
    Tier1, // Desktops (work hours, good reliability, 70-90% uptime)
    Tier2, // Laptops (intermittent, medium reliability, 40-70% uptime)
    Tier3, // Raspberry Pis or experimental (low reliability, <40% uptime)
}

impl NodeTier {
    /// Get reliability SLA for this tier
    pub fn reliability_sla(&self) -> f64 {
        match self {
            NodeTier::Tier0 => 0.95,
            NodeTier::Tier1 => 0.80,
            NodeTier::Tier2 => 0.55,
            NodeTier::Tier3 => 0.30,
        }
    }

    /// Get priority for job placement (higher is better)
    pub fn placement_priority(&self) -> u8 {
        match self {
            NodeTier::Tier0 => 100,
            NodeTier::Tier1 => 70,
            NodeTier::Tier2 => 40,
            NodeTier::Tier3 => 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_uptime_session_creation() {
        let asset_id = Uuid::new_v4();
        let online_at = Utc::now().naive_utc();

        let session = UptimeSession::new(asset_id, online_at);

        assert_eq!(session.asset_id, asset_id);
        assert_eq!(session.online_at, online_at);
        assert!(session.offline_at.is_none());
        assert!(session.duration_minutes.is_none());
    }

    #[test]
    fn test_uptime_session_end() {
        let asset_id = Uuid::new_v4();
        let online_at = Utc::now().naive_utc();
        let mut session = UptimeSession::new(asset_id, online_at);

        let offline_at = online_at + chrono::Duration::hours(2);
        session.end_session(offline_at);

        assert_eq!(session.offline_at, Some(offline_at));
        assert_eq!(session.duration_minutes, Some(120));
    }

    #[test]
    fn test_uptime_session_battery_metrics() {
        let asset_id = Uuid::new_v4();
        let online_at = Utc::now().naive_utc();
        let mut session = UptimeSession::new(asset_id, online_at);

        session.set_battery_metrics(75, true);

        assert_eq!(session.battery_percent, Some(75));
        assert_eq!(session.charging, Some(true));
    }

    #[test]
    fn test_uptime_session_thermal_state() {
        let asset_id = Uuid::new_v4();
        let online_at = Utc::now().naive_utc();
        let mut session = UptimeSession::new(asset_id, online_at);

        session.set_thermal_state("hot");

        assert_eq!(session.thermal_state, Some("hot".to_string()));
    }

    #[test]
    fn test_node_tier_reliability_sla() {
        assert_eq!(NodeTier::Tier0.reliability_sla(), 0.95);
        assert_eq!(NodeTier::Tier1.reliability_sla(), 0.80);
        assert_eq!(NodeTier::Tier2.reliability_sla(), 0.55);
        assert_eq!(NodeTier::Tier3.reliability_sla(), 0.30);
    }

    #[test]
    fn test_node_tier_placement_priority() {
        assert_eq!(NodeTier::Tier0.placement_priority(), 100);
        assert_eq!(NodeTier::Tier1.placement_priority(), 70);
        assert_eq!(NodeTier::Tier2.placement_priority(), 40);
        assert_eq!(NodeTier::Tier3.placement_priority(), 10);
    }

    #[test]
    fn test_uptime_stats_suggested_tier() {
        let asset_id = Uuid::new_v4();
        let now = Utc::now().naive_utc();

        // High reliability (Tier0)
        let stats = UptimeStats {
            asset_id,
            total_sessions: 100,
            avg_session_duration_minutes: Some(480.0),
            total_uptime_minutes: Some(40000),
            uptime_percent_30d: Some(95.0),
            reliability_score: Some(0.95),
            last_online_at: Some(now),
            typical_online_hours: Some(vec![9, 10, 11, 12, 13, 14, 15, 16, 17]),
            updated_at: now,
        };

        assert_eq!(stats.suggested_tier(), NodeTier::Tier0);

        // Medium reliability (Tier2)
        let stats2 = UptimeStats {
            reliability_score: Some(0.50),
            ..stats.clone()
        };

        assert_eq!(stats2.suggested_tier(), NodeTier::Tier2);
    }

    #[test]
    fn test_uptime_stats_is_likely_available() {
        let asset_id = Uuid::new_v4();
        let now = Utc::now().naive_utc();

        let stats = UptimeStats {
            asset_id,
            total_sessions: 50,
            avg_session_duration_minutes: Some(300.0),
            total_uptime_minutes: Some(15000),
            uptime_percent_30d: Some(75.0),
            reliability_score: Some(0.75),
            last_online_at: Some(now),
            typical_online_hours: Some(vec![9, 10, 11, 12, 13]),
            updated_at: now,
        };

        assert!(stats.is_likely_available_at_hour(10));
        assert!(!stats.is_likely_available_at_hour(22));
    }

    #[test]
    fn test_device_type_serialization() {
        let device = DeviceType::Laptop;
        let json = serde_json::to_string(&device).unwrap();
        let deserialized: DeviceType = serde_json::from_str(&json).unwrap();

        assert_eq!(device, deserialized);
    }

    #[test]
    fn test_availability_schedule_serialization() {
        let schedule = AvailabilitySchedule {
            days: vec![1, 2, 3, 4, 5], // Monday-Friday
            hours: vec![9, 10, 11, 12, 13, 14, 15, 16, 17], // 9am-5pm
            timezone: "America/New_York".to_string(),
        };

        let json = serde_json::to_string(&schedule).unwrap();
        let deserialized: AvailabilitySchedule = serde_json::from_str(&json).unwrap();

        assert_eq!(schedule.days, deserialized.days);
        assert_eq!(schedule.hours, deserialized.hours);
        assert_eq!(schedule.timezone, deserialized.timezone);
    }

    #[test]
    fn test_uptime_session_duration_calculation() {
        let asset_id = Uuid::new_v4();
        let online_at = Utc::now().naive_utc();
        let mut session = UptimeSession::new(asset_id, online_at);

        // 5 hours = 300 minutes
        let offline_at = online_at + chrono::Duration::hours(5);
        session.end_session(offline_at);

        assert_eq!(session.duration_minutes, Some(300));
    }

    #[test]
    fn test_node_tier_ordering() {
        assert!(NodeTier::Tier0.placement_priority() > NodeTier::Tier1.placement_priority());
        assert!(NodeTier::Tier1.placement_priority() > NodeTier::Tier2.placement_priority());
        assert!(NodeTier::Tier2.placement_priority() > NodeTier::Tier3.placement_priority());
    }

    #[test]
    fn test_uptime_stats_zero_reliability() {
        let asset_id = Uuid::new_v4();
        let now = Utc::now().naive_utc();

        let stats = UptimeStats {
            asset_id,
            total_sessions: 0,
            avg_session_duration_minutes: None,
            total_uptime_minutes: None,
            uptime_percent_30d: None,
            reliability_score: Some(0.0),
            last_online_at: None,
            typical_online_hours: None,
            updated_at: now,
        };

        assert_eq!(stats.suggested_tier(), NodeTier::Tier3);
    }
}
