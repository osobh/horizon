use crate::error::{HpcError, Result, ReporterErrorExt};
use crate::models::summary::CostAttribution;
use csv::Writer;

pub struct CsvExporter;

impl CsvExporter {
    pub fn new() -> Self {
        Self
    }

    pub fn export(&self, attributions: &[CostAttribution]) -> Result<String> {
        let mut wtr = Writer::from_writer(vec![]);

        // Write header
        wtr.write_record([
            "ID",
            "Job ID",
            "User ID",
            "Team ID",
            "Customer ID",
            "GPU Cost",
            "CPU Cost",
            "Network Cost",
            "Storage Cost",
            "Total Cost",
            "Period Start",
            "Period End",
            "Created At",
        ])?;

        // Write data rows
        for attr in attributions {
            wtr.write_record([
                attr.id.to_string(),
                attr.job_id.map(|id| id.to_string()).unwrap_or_default(),
                attr.user_id.clone(),
                attr.team_id.clone().unwrap_or_default(),
                attr.customer_id.clone().unwrap_or_default(),
                attr.gpu_cost.to_string(),
                attr.cpu_cost.to_string(),
                attr.network_cost.to_string(),
                attr.storage_cost.to_string(),
                attr.total_cost.to_string(),
                attr.period_start.to_rfc3339(),
                attr.period_end.to_rfc3339(),
                attr.created_at.to_rfc3339(),
            ])?;
        }

        let data = wtr.into_inner().map_err(|e| {
            HpcError::export_error(format!("CSV writer error: {}", e))
        })?;
        String::from_utf8(data).map_err(|e| {
            HpcError::export_error(format!("UTF-8 conversion error: {}", e))
        })
    }
}

impl Default for CsvExporter {
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
    fn test_csv_export_empty() {
        let exporter = CsvExporter::new();
        let result = exporter.export(&[]).unwrap();

        // Should have header only
        let lines: Vec<&str> = result.lines().collect();
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("ID"));
        assert!(lines[0].contains("Total Cost"));
    }

    #[test]
    fn test_csv_export_single_attribution() {
        let now = Utc::now();
        let attr = CostAttribution {
            id: Uuid::new_v4(),
            job_id: Some(Uuid::new_v4()),
            user_id: "user123".to_string(),
            team_id: Some("team456".to_string()),
            customer_id: Some("customer789".to_string()),
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

        let exporter = CsvExporter::new();
        let result = exporter.export(&[attr]).unwrap();

        let lines: Vec<&str> = result.lines().collect();
        assert_eq!(lines.len(), 2); // header + 1 data row

        // Check data row contains expected values
        let data_line = lines[1];
        assert!(data_line.contains("user123"));
        assert!(data_line.contains("team456"));
        assert!(data_line.contains("100.50"));
    }

    #[test]
    fn test_csv_export_multiple_attributions() {
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
                team_id: Some("team1".to_string()),
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

        let exporter = CsvExporter::new();
        let result = exporter.export(&attrs).unwrap();

        let lines: Vec<&str> = result.lines().collect();
        assert_eq!(lines.len(), 3); // header + 2 data rows

        assert!(result.contains("user1"));
        assert!(result.contains("user2"));
        assert!(result.contains("50.00"));
        assert!(result.contains("75.00"));
    }

    #[test]
    fn test_csv_export_valid_format() {
        let now = Utc::now();
        let attr = CostAttribution {
            id: Uuid::new_v4(),
            job_id: None,
            user_id: "test_user".to_string(),
            team_id: None,
            customer_id: None,
            gpu_cost: dec!(10.00),
            cpu_cost: dec!(5.00),
            network_cost: dec!(1.00),
            storage_cost: dec!(0.50),
            total_cost: dec!(16.50),
            period_start: now,
            period_end: now,
            created_at: now,
            updated_at: now,
        };

        let exporter = CsvExporter::new();
        let result = exporter.export(&[attr]).unwrap();

        // Verify it's valid CSV by parsing it back
        let mut reader = csv::Reader::from_reader(result.as_bytes());
        let headers = reader.headers().unwrap();
        assert!(!headers.is_empty());

        let record_count = reader.records().count();
        assert_eq!(record_count, 1);
    }
}
