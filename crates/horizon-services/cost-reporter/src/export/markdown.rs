use crate::error::Result;
use crate::models::summary::CostAttribution;
use rust_decimal::Decimal;

pub struct MarkdownExporter;

impl MarkdownExporter {
    pub fn new() -> Self {
        Self
    }

    pub fn export(&self, attributions: &[CostAttribution]) -> Result<String> {
        let mut output = String::new();

        // Title
        output.push_str("# Cost Attribution Report\n\n");

        if attributions.is_empty() {
            output.push_str("*No cost attributions found.*\n");
            return Ok(output);
        }

        // Summary statistics
        let total_cost: Decimal = attributions.iter().map(|a| a.total_cost).sum();
        let total_gpu: Decimal = attributions.iter().map(|a| a.gpu_cost).sum();
        let total_cpu: Decimal = attributions.iter().map(|a| a.cpu_cost).sum();
        let total_network: Decimal = attributions.iter().map(|a| a.network_cost).sum();
        let total_storage: Decimal = attributions.iter().map(|a| a.storage_cost).sum();

        output.push_str("## Summary\n\n");
        output.push_str(&format!("- **Total Records**: {}\n", attributions.len()));
        output.push_str(&format!("- **Total Cost**: ${}\n", total_cost));
        output.push_str(&format!("- **GPU Cost**: ${}\n", total_gpu));
        output.push_str(&format!("- **CPU Cost**: ${}\n", total_cpu));
        output.push_str(&format!("- **Network Cost**: ${}\n", total_network));
        output.push_str(&format!("- **Storage Cost**: ${}\n\n", total_storage));

        // Detailed table
        output.push_str("## Detailed Breakdown\n\n");
        output.push_str("| User ID | Team ID | GPU Cost | CPU Cost | Network Cost | Storage Cost | Total Cost |\n");
        output.push_str("|---------|---------|----------|----------|--------------|--------------|------------|\n");

        for attr in attributions {
            output.push_str(&format!(
                "| {} | {} | ${} | ${} | ${} | ${} | ${} |\n",
                attr.user_id,
                attr.team_id.clone().unwrap_or_else(|| "-".to_string()),
                attr.gpu_cost,
                attr.cpu_cost,
                attr.network_cost,
                attr.storage_cost,
                attr.total_cost,
            ));
        }

        output.push('\n');

        Ok(output)
    }
}

impl Default for MarkdownExporter {
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
    fn test_markdown_export_empty() {
        let exporter = MarkdownExporter::new();
        let result = exporter.export(&[]).unwrap();

        assert!(result.contains("# Cost Attribution Report"));
        assert!(result.contains("No cost attributions found"));
    }

    #[test]
    fn test_markdown_export_single_attribution() {
        let now = Utc::now();
        let attr = CostAttribution {
            id: Uuid::new_v4(),
            job_id: None,
            user_id: "user123".to_string(),
            team_id: Some("team456".to_string()),
            customer_id: None,
            gpu_cost: dec!(100.00),
            cpu_cost: dec!(20.00),
            network_cost: dec!(5.00),
            storage_cost: dec!(3.00),
            total_cost: dec!(128.00),
            period_start: now,
            period_end: now,
            created_at: now,
            updated_at: now,
        };

        let exporter = MarkdownExporter::new();
        let result = exporter.export(&[attr]).unwrap();

        assert!(result.contains("# Cost Attribution Report"));
        assert!(result.contains("## Summary"));
        assert!(result.contains("Total Records**: 1"));
        assert!(result.contains("Total Cost**: $128"));
        assert!(result.contains("## Detailed Breakdown"));
        assert!(result.contains("user123"));
        assert!(result.contains("team456"));
        assert!(result.contains("$100"));
    }

    #[test]
    fn test_markdown_export_multiple_attributions() {
        let now = Utc::now();
        let attrs = vec![
            CostAttribution {
                id: Uuid::new_v4(),
                job_id: None,
                user_id: "user1".to_string(),
                team_id: Some("team1".to_string()),
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

        let exporter = MarkdownExporter::new();
        let result = exporter.export(&attrs).unwrap();

        assert!(result.contains("Total Records**: 2"));
        assert!(result.contains("Total Cost**: $158")); // 63 + 95
        assert!(result.contains("user1"));
        assert!(result.contains("user2"));
        assert!(result.contains("team1"));
        assert!(result.contains("| -")); // For missing team on user2
    }

    #[test]
    fn test_markdown_has_table_structure() {
        let now = Utc::now();
        let attr = CostAttribution {
            id: Uuid::new_v4(),
            job_id: None,
            user_id: "test".to_string(),
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

        let exporter = MarkdownExporter::new();
        let result = exporter.export(&[attr]).unwrap();

        // Check for table structure
        assert!(result.contains("| User ID"));
        assert!(result.contains("|---------|"));
        assert!(result.contains("| test |"));
    }
}
