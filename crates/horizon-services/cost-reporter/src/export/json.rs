use crate::error::Result;
use crate::models::summary::CostAttribution;

pub struct JsonExporter;

impl JsonExporter {
    pub fn new() -> Self {
        Self
    }

    pub fn export(&self, attributions: &[CostAttribution]) -> Result<String> {
        Ok(serde_json::to_string_pretty(attributions)?)
    }
}

impl Default for JsonExporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use rust_decimal_macros::dec;
    use uuid::Uuid;

    #[test]
    fn test_json_export_empty() {
        let exporter = JsonExporter::new();
        let result = exporter.export(&[]).unwrap();

        assert_eq!(result, "[]");
    }

    #[test]
    fn test_json_export_single_attribution() {
        let now = Utc::now();
        let attr = CostAttribution {
            id: Uuid::new_v4(),
            job_id: Some(Uuid::new_v4()),
            user_id: "user123".to_string(),
            team_id: Some("team456".to_string()),
            customer_id: None,
            gpu_cost: dec!(100.50),
            cpu_cost: dec!(20.25),
            network_cost: dec!(5.75),
            storage_cost: dec!(3.50),
            total_cost: dec!(130.00),
            period_start: now,
            period_end: now,
            created_at: now,
            updated_at: now,
        };

        let exporter = JsonExporter::new();
        let result = exporter.export(&[attr]).unwrap();

        // Verify valid JSON
        let parsed: Vec<CostAttribution> = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].user_id, "user123");
        assert_eq!(parsed[0].gpu_cost, dec!(100.50));
    }

    #[test]
    fn test_json_export_multiple_attributions() {
        let now = Utc::now();
        let attrs = vec![
            CostAttribution {
                id: Uuid::new_v4(),
                job_id: None,
                user_id: "user1".to_string(),
                team_id: None,
                customer_id: None,
                gpu_cost: dec!(50.00),
                cpu_cost: dec!(10.00),
                network_cost: dec!(2.00),
                storage_cost: dec!(1.00),
                total_cost: dec!(63.00),
                period_start: now,
                period_end: now,
                created_at: now,
                updated_at: now,
            },
            CostAttribution {
                id: Uuid::new_v4(),
                job_id: None,
                user_id: "user2".to_string(),
                team_id: None,
                customer_id: None,
                gpu_cost: dec!(75.00),
                cpu_cost: dec!(15.00),
                network_cost: dec!(3.00),
                storage_cost: dec!(2.00),
                total_cost: dec!(95.00),
                period_start: now,
                period_end: now,
                created_at: now,
                updated_at: now,
            },
        ];

        let exporter = JsonExporter::new();
        let result = exporter.export(&attrs).unwrap();

        // Verify valid JSON with correct count
        let parsed: Vec<CostAttribution> = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].user_id, "user1");
        assert_eq!(parsed[1].user_id, "user2");
    }
}
