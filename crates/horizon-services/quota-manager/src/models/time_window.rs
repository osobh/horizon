//! Time window scheduling for ephemeral quotas.
//!
//! Provides time-based constraints for when ephemeral resources can be used,
//! supporting business hours, recurring schedules, and blackout periods.

use chrono::{DateTime, Datelike, NaiveDate, NaiveTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use utoipa::ToSchema;
use uuid::Uuid;

use crate::error::{HpcError, QuotaErrorExt, Result};

/// Type of recurrence for a time window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type, ToSchema)]
#[sqlx(type_name = "text", rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum RecurrenceType {
    /// No recurrence - one-time window
    Once,
    /// Daily recurrence
    Daily,
    /// Weekly recurrence on specific days
    Weekly,
    /// Monthly recurrence on specific dates
    Monthly,
}

impl RecurrenceType {
    pub fn as_str(&self) -> &'static str {
        match self {
            RecurrenceType::Once => "once",
            RecurrenceType::Daily => "daily",
            RecurrenceType::Weekly => "weekly",
            RecurrenceType::Monthly => "monthly",
        }
    }

    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "once" => Ok(RecurrenceType::Once),
            "daily" => Ok(RecurrenceType::Daily),
            "weekly" => Ok(RecurrenceType::Weekly),
            "monthly" => Ok(RecurrenceType::Monthly),
            _ => Err(HpcError::invalid_input(
                "recurrence_type",
                format!("Invalid recurrence type: {}", s),
            )),
        }
    }
}

/// A time window defining when ephemeral resources are accessible.
#[derive(Debug, Clone, FromRow, Serialize, Deserialize, ToSchema)]
pub struct TimeWindow {
    /// Unique identifier for this time window
    pub id: Uuid,
    /// Name for this time window (e.g., "Business Hours", "Weekend Maintenance")
    pub name: String,
    /// Optional description
    pub description: Option<String>,
    /// Organization or tenant this window belongs to
    pub tenant_id: Uuid,
    /// Start time in the specified timezone (e.g., 09:00)
    pub start_time: NaiveTime,
    /// End time in the specified timezone (e.g., 17:00)
    pub end_time: NaiveTime,
    /// IANA timezone identifier (e.g., "America/New_York", "UTC")
    pub timezone: String,
    /// Type of recurrence for this window
    pub recurrence_type: RecurrenceType,
    /// For weekly recurrence: days of the week (0=Sunday, 6=Saturday) as JSON array
    #[sqlx(skip)]
    pub allowed_days: Vec<u8>,
    /// For monthly recurrence: days of the month (1-31) as JSON array
    #[sqlx(skip)]
    pub allowed_dates: Vec<u8>,
    /// Specific dates when this window is inactive, serialized as ISO date strings
    #[sqlx(skip)]
    pub blackout_dates: Vec<NaiveDate>,
    /// Whether this time window is currently active
    pub is_active: bool,
    /// When this window was created
    pub created_at: DateTime<Utc>,
    /// Last modification time
    pub updated_at: DateTime<Utc>,
}

/// Request to create a new time window.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CreateTimeWindowRequest {
    pub name: String,
    pub description: Option<String>,
    pub tenant_id: Uuid,
    pub start_time: NaiveTime,
    pub end_time: NaiveTime,
    pub timezone: String,
    pub recurrence_type: RecurrenceType,
    pub allowed_days: Option<Vec<u8>>,
    pub allowed_dates: Option<Vec<u8>>,
    pub blackout_dates: Option<Vec<NaiveDate>>,
}

/// Request to update an existing time window.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct UpdateTimeWindowRequest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub start_time: Option<NaiveTime>,
    pub end_time: Option<NaiveTime>,
    pub timezone: Option<String>,
    pub recurrence_type: Option<RecurrenceType>,
    pub allowed_days: Option<Vec<u8>>,
    pub allowed_dates: Option<Vec<u8>>,
    pub blackout_dates: Option<Vec<NaiveDate>>,
    pub is_active: Option<bool>,
}

/// Result of checking if a time window allows access at a given moment.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct TimeWindowCheckResult {
    /// Whether access is currently allowed
    pub allowed: bool,
    /// If not allowed, when the next allowed window starts
    pub next_window_start: Option<DateTime<Utc>>,
    /// If allowed, when the current window ends
    pub current_window_end: Option<DateTime<Utc>>,
    /// Human-readable reason for the result
    pub reason: String,
}

impl TimeWindow {
    /// Create a new time window with the given parameters.
    pub fn new(
        name: impl Into<String>,
        tenant_id: Uuid,
        start_time: NaiveTime,
        end_time: NaiveTime,
        timezone: impl Into<String>,
        recurrence_type: RecurrenceType,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            description: None,
            tenant_id,
            start_time,
            end_time,
            timezone: timezone.into(),
            recurrence_type,
            allowed_days: Vec::new(),
            allowed_dates: Vec::new(),
            blackout_dates: Vec::new(),
            is_active: true,
            created_at: now,
            updated_at: now,
        }
    }

    /// Create a business hours window (Mon-Fri, 9am-5pm).
    pub fn business_hours(name: impl Into<String>, tenant_id: Uuid, timezone: impl Into<String>) -> Self {
        let mut window = Self::new(
            name,
            tenant_id,
            NaiveTime::from_hms_opt(9, 0, 0).unwrap(),
            NaiveTime::from_hms_opt(17, 0, 0).unwrap(),
            timezone,
            RecurrenceType::Weekly,
        );
        // Monday through Friday (1-5 in chrono weekday)
        window.allowed_days = vec![1, 2, 3, 4, 5];
        window
    }

    /// Create a 24/7 unrestricted window.
    pub fn unrestricted(name: impl Into<String>, tenant_id: Uuid) -> Self {
        Self::new(
            name,
            tenant_id,
            NaiveTime::from_hms_opt(0, 0, 0).unwrap(),
            NaiveTime::from_hms_opt(23, 59, 59).unwrap(),
            "UTC",
            RecurrenceType::Daily,
        )
    }

    /// Add a description to this window.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the allowed days for weekly recurrence.
    pub fn with_allowed_days(mut self, days: Vec<u8>) -> Self {
        self.allowed_days = days;
        self
    }

    /// Set the allowed dates for monthly recurrence.
    pub fn with_allowed_dates(mut self, dates: Vec<u8>) -> Self {
        self.allowed_dates = dates;
        self
    }

    /// Add blackout dates when the window is inactive.
    pub fn with_blackout_dates(mut self, dates: Vec<NaiveDate>) -> Self {
        self.blackout_dates = dates;
        self
    }

    /// Validate this time window configuration.
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(HpcError::invalid_configuration("Time window name cannot be empty"));
        }

        // Validate that end time is after start time (or handle overnight windows)
        if self.start_time == self.end_time {
            return Err(HpcError::invalid_configuration(
                "Start time and end time cannot be the same",
            ));
        }

        // Validate allowed_days for weekly recurrence
        if self.recurrence_type == RecurrenceType::Weekly && self.allowed_days.is_empty() {
            return Err(HpcError::invalid_configuration(
                "Weekly recurrence requires at least one allowed day",
            ));
        }

        // Validate day values (0-6)
        for &day in &self.allowed_days {
            if day > 6 {
                return Err(HpcError::invalid_configuration(format!(
                    "Invalid day value: {}. Must be 0-6 (0=Sunday)",
                    day
                )));
            }
        }

        // Validate allowed_dates for monthly recurrence
        if self.recurrence_type == RecurrenceType::Monthly && self.allowed_dates.is_empty() {
            return Err(HpcError::invalid_configuration(
                "Monthly recurrence requires at least one allowed date",
            ));
        }

        // Validate date values (1-31)
        for &date in &self.allowed_dates {
            if date < 1 || date > 31 {
                return Err(HpcError::invalid_configuration(format!(
                    "Invalid date value: {}. Must be 1-31",
                    date
                )));
            }
        }

        Ok(())
    }

    /// Check if the given datetime falls within this time window.
    pub fn is_allowed(&self, datetime: DateTime<Utc>) -> TimeWindowCheckResult {
        if !self.is_active {
            return TimeWindowCheckResult {
                allowed: false,
                next_window_start: None,
                current_window_end: None,
                reason: "Time window is inactive".to_string(),
            };
        }

        let date = datetime.date_naive();
        let time = datetime.time();

        // Check blackout dates
        if self.blackout_dates.contains(&date) {
            return TimeWindowCheckResult {
                allowed: false,
                next_window_start: None,
                current_window_end: None,
                reason: format!("Date {} is in the blackout list", date),
            };
        }

        // Check recurrence type
        let day_allowed = match self.recurrence_type {
            RecurrenceType::Once => true, // Always allowed for one-time windows
            RecurrenceType::Daily => true,
            RecurrenceType::Weekly => {
                let weekday = datetime.weekday().num_days_from_sunday() as u8;
                self.allowed_days.contains(&weekday)
            }
            RecurrenceType::Monthly => {
                let day_of_month = datetime.day() as u8;
                self.allowed_dates.contains(&day_of_month)
            }
        };

        if !day_allowed {
            return TimeWindowCheckResult {
                allowed: false,
                next_window_start: None,
                current_window_end: None,
                reason: "Current day/date is not in the allowed list".to_string(),
            };
        }

        // Check time window
        let time_allowed = if self.start_time <= self.end_time {
            // Normal window (e.g., 9am to 5pm)
            time >= self.start_time && time <= self.end_time
        } else {
            // Overnight window (e.g., 10pm to 6am)
            time >= self.start_time || time <= self.end_time
        };

        if time_allowed {
            TimeWindowCheckResult {
                allowed: true,
                next_window_start: None,
                current_window_end: None, // Would need timezone conversion to calculate properly
                reason: "Current time is within the allowed window".to_string(),
            }
        } else {
            TimeWindowCheckResult {
                allowed: false,
                next_window_start: None,
                current_window_end: None,
                reason: format!(
                    "Current time {} is outside the window {}-{}",
                    time, self.start_time, self.end_time
                ),
            }
        }
    }

    /// Calculate the total available hours per week for this window.
    pub fn weekly_hours(&self) -> Decimal {
        let duration_seconds = if self.start_time <= self.end_time {
            self.end_time.signed_duration_since(self.start_time)
        } else {
            // Overnight window: calculate as (midnight - start) + (end - midnight)
            let to_midnight = NaiveTime::from_hms_opt(23, 59, 59)
                .unwrap()
                .signed_duration_since(self.start_time);
            let from_midnight = self
                .end_time
                .signed_duration_since(NaiveTime::from_hms_opt(0, 0, 0).unwrap());
            to_midnight + from_midnight
        };

        let daily_hours = Decimal::from(duration_seconds.num_seconds()) / Decimal::from(3600);

        match self.recurrence_type {
            RecurrenceType::Once => daily_hours,
            RecurrenceType::Daily => daily_hours * Decimal::from(7),
            RecurrenceType::Weekly => daily_hours * Decimal::from(self.allowed_days.len()),
            RecurrenceType::Monthly => {
                // Approximate: average 4.33 weeks per month
                daily_hours * Decimal::from(self.allowed_dates.len()) / Decimal::from(4)
            }
        }
    }

    /// Deactivate this time window.
    pub fn deactivate(&mut self) {
        self.is_active = false;
        self.updated_at = Utc::now();
    }

    /// Activate this time window.
    pub fn activate(&mut self) {
        self.is_active = true;
        self.updated_at = Utc::now();
    }

    /// Add a blackout date.
    pub fn add_blackout_date(&mut self, date: NaiveDate) {
        if !self.blackout_dates.contains(&date) {
            self.blackout_dates.push(date);
            self.updated_at = Utc::now();
        }
    }

    /// Remove a blackout date.
    pub fn remove_blackout_date(&mut self, date: &NaiveDate) -> bool {
        let initial_len = self.blackout_dates.len();
        self.blackout_dates.retain(|d| d != date);
        if self.blackout_dates.len() != initial_len {
            self.updated_at = Utc::now();
            true
        } else {
            false
        }
    }
}

/// Builder for creating time windows with fluent API.
pub struct TimeWindowBuilder {
    name: String,
    description: Option<String>,
    tenant_id: Uuid,
    start_time: NaiveTime,
    end_time: NaiveTime,
    timezone: String,
    recurrence_type: RecurrenceType,
    allowed_days: Vec<u8>,
    allowed_dates: Vec<u8>,
    blackout_dates: Vec<NaiveDate>,
}

impl TimeWindowBuilder {
    pub fn new(name: impl Into<String>, tenant_id: Uuid) -> Self {
        Self {
            name: name.into(),
            description: None,
            tenant_id,
            start_time: NaiveTime::from_hms_opt(0, 0, 0).unwrap(),
            end_time: NaiveTime::from_hms_opt(23, 59, 59).unwrap(),
            timezone: "UTC".to_string(),
            recurrence_type: RecurrenceType::Daily,
            allowed_days: Vec::new(),
            allowed_dates: Vec::new(),
            blackout_dates: Vec::new(),
        }
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn hours(mut self, start_hour: u32, end_hour: u32) -> Self {
        self.start_time = NaiveTime::from_hms_opt(start_hour, 0, 0).unwrap();
        self.end_time = NaiveTime::from_hms_opt(end_hour, 0, 0).unwrap();
        self
    }

    pub fn time_range(mut self, start: NaiveTime, end: NaiveTime) -> Self {
        self.start_time = start;
        self.end_time = end;
        self
    }

    pub fn timezone(mut self, tz: impl Into<String>) -> Self {
        self.timezone = tz.into();
        self
    }

    pub fn daily(mut self) -> Self {
        self.recurrence_type = RecurrenceType::Daily;
        self
    }

    pub fn weekly(mut self, days: Vec<u8>) -> Self {
        self.recurrence_type = RecurrenceType::Weekly;
        self.allowed_days = days;
        self
    }

    pub fn weekdays(self) -> Self {
        self.weekly(vec![1, 2, 3, 4, 5])
    }

    pub fn weekends(self) -> Self {
        self.weekly(vec![0, 6])
    }

    pub fn monthly(mut self, dates: Vec<u8>) -> Self {
        self.recurrence_type = RecurrenceType::Monthly;
        self.allowed_dates = dates;
        self
    }

    pub fn blackout(mut self, dates: Vec<NaiveDate>) -> Self {
        self.blackout_dates = dates;
        self
    }

    pub fn build(self) -> Result<TimeWindow> {
        let window = TimeWindow {
            id: Uuid::new_v4(),
            name: self.name,
            description: self.description,
            tenant_id: self.tenant_id,
            start_time: self.start_time,
            end_time: self.end_time,
            timezone: self.timezone,
            recurrence_type: self.recurrence_type,
            allowed_days: self.allowed_days,
            allowed_dates: self.allowed_dates,
            blackout_dates: self.blackout_dates,
            is_active: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        window.validate()?;
        Ok(window)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn tenant_id() -> Uuid {
        Uuid::new_v4()
    }

    #[test]
    fn test_recurrence_type_from_str() {
        assert_eq!(RecurrenceType::from_str("once").unwrap(), RecurrenceType::Once);
        assert_eq!(RecurrenceType::from_str("daily").unwrap(), RecurrenceType::Daily);
        assert_eq!(RecurrenceType::from_str("weekly").unwrap(), RecurrenceType::Weekly);
        assert_eq!(RecurrenceType::from_str("monthly").unwrap(), RecurrenceType::Monthly);
        assert!(RecurrenceType::from_str("invalid").is_err());
    }

    #[test]
    fn test_time_window_new() {
        let window = TimeWindow::new(
            "Test Window",
            tenant_id(),
            NaiveTime::from_hms_opt(9, 0, 0).unwrap(),
            NaiveTime::from_hms_opt(17, 0, 0).unwrap(),
            "UTC",
            RecurrenceType::Daily,
        );

        assert_eq!(window.name, "Test Window");
        assert!(window.is_active);
        assert!(window.allowed_days.is_empty());
    }

    #[test]
    fn test_time_window_business_hours() {
        let window = TimeWindow::business_hours("Business Hours", tenant_id(), "America/New_York");

        assert_eq!(window.name, "Business Hours");
        assert_eq!(window.start_time, NaiveTime::from_hms_opt(9, 0, 0).unwrap());
        assert_eq!(window.end_time, NaiveTime::from_hms_opt(17, 0, 0).unwrap());
        assert_eq!(window.timezone, "America/New_York");
        assert_eq!(window.recurrence_type, RecurrenceType::Weekly);
        assert_eq!(window.allowed_days, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_time_window_unrestricted() {
        let window = TimeWindow::unrestricted("24/7 Access", tenant_id());

        assert_eq!(window.name, "24/7 Access");
        assert_eq!(window.recurrence_type, RecurrenceType::Daily);
        assert_eq!(window.timezone, "UTC");
    }

    #[test]
    fn test_time_window_validate_empty_name() {
        let window = TimeWindow::new(
            "",
            tenant_id(),
            NaiveTime::from_hms_opt(9, 0, 0).unwrap(),
            NaiveTime::from_hms_opt(17, 0, 0).unwrap(),
            "UTC",
            RecurrenceType::Daily,
        );

        assert!(window.validate().is_err());
    }

    #[test]
    fn test_time_window_validate_same_start_end() {
        let window = TimeWindow::new(
            "Test",
            tenant_id(),
            NaiveTime::from_hms_opt(9, 0, 0).unwrap(),
            NaiveTime::from_hms_opt(9, 0, 0).unwrap(),
            "UTC",
            RecurrenceType::Daily,
        );

        assert!(window.validate().is_err());
    }

    #[test]
    fn test_time_window_validate_weekly_no_days() {
        let window = TimeWindow::new(
            "Test",
            tenant_id(),
            NaiveTime::from_hms_opt(9, 0, 0).unwrap(),
            NaiveTime::from_hms_opt(17, 0, 0).unwrap(),
            "UTC",
            RecurrenceType::Weekly,
        );

        assert!(window.validate().is_err());
    }

    #[test]
    fn test_time_window_validate_invalid_day() {
        let mut window = TimeWindow::new(
            "Test",
            tenant_id(),
            NaiveTime::from_hms_opt(9, 0, 0).unwrap(),
            NaiveTime::from_hms_opt(17, 0, 0).unwrap(),
            "UTC",
            RecurrenceType::Weekly,
        );
        window.allowed_days = vec![7]; // Invalid: max is 6

        assert!(window.validate().is_err());
    }

    #[test]
    fn test_time_window_validate_monthly_no_dates() {
        let window = TimeWindow::new(
            "Test",
            tenant_id(),
            NaiveTime::from_hms_opt(9, 0, 0).unwrap(),
            NaiveTime::from_hms_opt(17, 0, 0).unwrap(),
            "UTC",
            RecurrenceType::Monthly,
        );

        assert!(window.validate().is_err());
    }

    #[test]
    fn test_time_window_validate_invalid_date() {
        let mut window = TimeWindow::new(
            "Test",
            tenant_id(),
            NaiveTime::from_hms_opt(9, 0, 0).unwrap(),
            NaiveTime::from_hms_opt(17, 0, 0).unwrap(),
            "UTC",
            RecurrenceType::Monthly,
        );
        window.allowed_dates = vec![32]; // Invalid: max is 31

        assert!(window.validate().is_err());
    }

    #[test]
    fn test_time_window_validate_valid() {
        let window = TimeWindow::business_hours("Test", tenant_id(), "UTC");
        assert!(window.validate().is_ok());
    }

    #[test]
    fn test_time_window_is_allowed_inactive() {
        let mut window = TimeWindow::unrestricted("Test", tenant_id());
        window.is_active = false;

        let result = window.is_allowed(Utc::now());
        assert!(!result.allowed);
        assert!(result.reason.contains("inactive"));
    }

    #[test]
    fn test_time_window_is_allowed_blackout() {
        let today = Utc::now().date_naive();
        let window = TimeWindow::unrestricted("Test", tenant_id()).with_blackout_dates(vec![today]);

        let result = window.is_allowed(Utc::now());
        assert!(!result.allowed);
        assert!(result.reason.contains("blackout"));
    }

    #[test]
    fn test_time_window_is_allowed_daily_within_hours() {
        use chrono::TimeZone;

        // Create a window from 9am to 5pm daily
        let window = TimeWindow::new(
            "Test",
            tenant_id(),
            NaiveTime::from_hms_opt(9, 0, 0).unwrap(),
            NaiveTime::from_hms_opt(17, 0, 0).unwrap(),
            "UTC",
            RecurrenceType::Daily,
        );

        // Test at 12:00 UTC - should be allowed
        let noon = Utc.with_ymd_and_hms(2024, 1, 15, 12, 0, 0).unwrap();
        let result = window.is_allowed(noon);
        assert!(result.allowed);
    }

    #[test]
    fn test_time_window_is_allowed_daily_outside_hours() {
        use chrono::TimeZone;

        let window = TimeWindow::new(
            "Test",
            tenant_id(),
            NaiveTime::from_hms_opt(9, 0, 0).unwrap(),
            NaiveTime::from_hms_opt(17, 0, 0).unwrap(),
            "UTC",
            RecurrenceType::Daily,
        );

        // Test at 20:00 UTC - should not be allowed
        let evening = Utc.with_ymd_and_hms(2024, 1, 15, 20, 0, 0).unwrap();
        let result = window.is_allowed(evening);
        assert!(!result.allowed);
    }

    #[test]
    fn test_time_window_is_allowed_weekly_correct_day() {
        use chrono::TimeZone;

        let mut window = TimeWindow::new(
            "Test",
            tenant_id(),
            NaiveTime::from_hms_opt(9, 0, 0).unwrap(),
            NaiveTime::from_hms_opt(17, 0, 0).unwrap(),
            "UTC",
            RecurrenceType::Weekly,
        );
        window.allowed_days = vec![1]; // Monday only

        // 2024-01-15 is a Monday
        let monday_noon = Utc.with_ymd_and_hms(2024, 1, 15, 12, 0, 0).unwrap();
        let result = window.is_allowed(monday_noon);
        assert!(result.allowed);
    }

    #[test]
    fn test_time_window_is_allowed_weekly_wrong_day() {
        use chrono::TimeZone;

        let mut window = TimeWindow::new(
            "Test",
            tenant_id(),
            NaiveTime::from_hms_opt(9, 0, 0).unwrap(),
            NaiveTime::from_hms_opt(17, 0, 0).unwrap(),
            "UTC",
            RecurrenceType::Weekly,
        );
        window.allowed_days = vec![1]; // Monday only

        // 2024-01-16 is a Tuesday
        let tuesday_noon = Utc.with_ymd_and_hms(2024, 1, 16, 12, 0, 0).unwrap();
        let result = window.is_allowed(tuesday_noon);
        assert!(!result.allowed);
    }

    #[test]
    fn test_time_window_is_allowed_monthly() {
        use chrono::TimeZone;

        let mut window = TimeWindow::new(
            "Test",
            tenant_id(),
            NaiveTime::from_hms_opt(9, 0, 0).unwrap(),
            NaiveTime::from_hms_opt(17, 0, 0).unwrap(),
            "UTC",
            RecurrenceType::Monthly,
        );
        window.allowed_dates = vec![15]; // 15th of each month

        let fifteenth = Utc.with_ymd_and_hms(2024, 1, 15, 12, 0, 0).unwrap();
        let result = window.is_allowed(fifteenth);
        assert!(result.allowed);

        let fourteenth = Utc.with_ymd_and_hms(2024, 1, 14, 12, 0, 0).unwrap();
        let result = window.is_allowed(fourteenth);
        assert!(!result.allowed);
    }

    #[test]
    fn test_time_window_weekly_hours_daily() {
        let window = TimeWindow::new(
            "Test",
            tenant_id(),
            NaiveTime::from_hms_opt(9, 0, 0).unwrap(),
            NaiveTime::from_hms_opt(17, 0, 0).unwrap(),
            "UTC",
            RecurrenceType::Daily,
        );

        // 8 hours per day * 7 days = 56 hours
        let weekly = window.weekly_hours();
        assert_eq!(weekly, dec!(56));
    }

    #[test]
    fn test_time_window_weekly_hours_business() {
        let window = TimeWindow::business_hours("Test", tenant_id(), "UTC");

        // 8 hours per day * 5 days = 40 hours
        let weekly = window.weekly_hours();
        assert_eq!(weekly, dec!(40));
    }

    #[test]
    fn test_time_window_deactivate_activate() {
        let mut window = TimeWindow::unrestricted("Test", tenant_id());
        assert!(window.is_active);

        window.deactivate();
        assert!(!window.is_active);

        window.activate();
        assert!(window.is_active);
    }

    #[test]
    fn test_time_window_blackout_management() {
        let mut window = TimeWindow::unrestricted("Test", tenant_id());
        let date1 = NaiveDate::from_ymd_opt(2024, 12, 25).unwrap();
        let date2 = NaiveDate::from_ymd_opt(2024, 12, 26).unwrap();

        window.add_blackout_date(date1);
        assert_eq!(window.blackout_dates.len(), 1);

        // Adding same date again should not duplicate
        window.add_blackout_date(date1);
        assert_eq!(window.blackout_dates.len(), 1);

        window.add_blackout_date(date2);
        assert_eq!(window.blackout_dates.len(), 2);

        assert!(window.remove_blackout_date(&date1));
        assert_eq!(window.blackout_dates.len(), 1);

        // Removing non-existent date returns false
        assert!(!window.remove_blackout_date(&date1));
    }

    #[test]
    fn test_time_window_builder_basic() {
        let window = TimeWindowBuilder::new("Test", tenant_id())
            .hours(9, 17)
            .daily()
            .build()
            .unwrap();

        assert_eq!(window.name, "Test");
        assert_eq!(window.recurrence_type, RecurrenceType::Daily);
    }

    #[test]
    fn test_time_window_builder_business_hours() {
        let window = TimeWindowBuilder::new("Business", tenant_id())
            .description("Standard business hours")
            .hours(9, 17)
            .timezone("America/New_York")
            .weekdays()
            .build()
            .unwrap();

        assert_eq!(window.description, Some("Standard business hours".to_string()));
        assert_eq!(window.timezone, "America/New_York");
        assert_eq!(window.recurrence_type, RecurrenceType::Weekly);
        assert_eq!(window.allowed_days, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_time_window_builder_weekends() {
        let window = TimeWindowBuilder::new("Weekend", tenant_id())
            .hours(0, 23)
            .weekends()
            .build()
            .unwrap();

        assert_eq!(window.allowed_days, vec![0, 6]);
    }

    #[test]
    fn test_time_window_builder_monthly() {
        let window = TimeWindowBuilder::new("Monthly", tenant_id())
            .hours(9, 17)
            .monthly(vec![1, 15])
            .build()
            .unwrap();

        assert_eq!(window.recurrence_type, RecurrenceType::Monthly);
        assert_eq!(window.allowed_dates, vec![1, 15]);
    }

    #[test]
    fn test_time_window_builder_validation_fails() {
        let result = TimeWindowBuilder::new("", tenant_id()).daily().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_time_window_serialization() {
        let window = TimeWindow::business_hours("Test", tenant_id(), "UTC");
        let json = serde_json::to_string(&window).unwrap();
        let deserialized: TimeWindow = serde_json::from_str(&json).unwrap();

        assert_eq!(window.name, deserialized.name);
        assert_eq!(window.recurrence_type, deserialized.recurrence_type);
        assert_eq!(window.allowed_days, deserialized.allowed_days);
    }
}
