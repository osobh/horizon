use chrono::Utc;
use cost_reporter::{
    export::{CsvExporter, JsonExporter, MarkdownExporter},
    models::summary::CostAttribution,
};
use rust_decimal_macros::dec;
use uuid::Uuid;

#[test]
fn test_csv_export_empty() {
    let exporter = CsvExporter::new();
    let result = exporter.export(&[]).unwrap();

    let lines: Vec<&str> = result.lines().collect();
    assert_eq!(lines.len(), 1); // Header only
    assert!(lines[0].contains("ID"));
    assert!(lines[0].contains("Total Cost"));
}

#[test]
fn test_csv_export_single_record() {
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

    let exporter = CsvExporter::new();
    let result = exporter.export(&[attr]).unwrap();

    let lines: Vec<&str> = result.lines().collect();
    assert_eq!(lines.len(), 2); // Header + 1 data row

    assert!(result.contains("user123"));
    assert!(result.contains("team456"));
    assert!(result.contains("100.50"));
}

#[test]
fn test_csv_export_multiple_records() {
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
    assert_eq!(lines.len(), 3); // Header + 2 data rows

    assert!(result.contains("user1"));
    assert!(result.contains("user2"));
}

#[test]
fn test_json_export_empty() {
    let exporter = JsonExporter::new();
    let result = exporter.export(&[]).unwrap();
    assert_eq!(result, "[]");
}

#[test]
fn test_json_export_valid() {
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

    let exporter = JsonExporter::new();
    let result = exporter.export(&[attr]).unwrap();

    // Verify it's valid JSON
    let parsed: Vec<CostAttribution> = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].user_id, "user123");
    assert_eq!(parsed[0].gpu_cost, dec!(100.00));
}

#[test]
fn test_markdown_export_empty() {
    let exporter = MarkdownExporter::new();
    let result = exporter.export(&[]).unwrap();

    assert!(result.contains("# Cost Attribution Report"));
    assert!(result.contains("No cost attributions found"));
}

#[test]
fn test_markdown_export_with_data() {
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
}

#[test]
fn test_markdown_export_table_structure() {
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

#[test]
fn test_csv_roundtrip() {
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
    let csv_data = exporter.export(&[attr]).unwrap();

    // Verify it's valid CSV by parsing it back
    let mut reader = csv::Reader::from_reader(csv_data.as_bytes());
    let headers = reader.headers().unwrap();
    assert!(!headers.is_empty());

    let record_count = reader.records().count();
    assert_eq!(record_count, 1);
}
