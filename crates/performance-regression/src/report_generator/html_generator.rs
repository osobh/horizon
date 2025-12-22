//! HTML generation functionality for reports and dashboards

use super::report_types::{PerformanceReport, SectionContentType};
use super::types::DashboardWidget;
use crate::error::PerformanceRegressionResult;
use crate::metrics_collector::MetricDataPoint;
use std::collections::HashMap;

/// Generate HTML report from performance report
pub fn generate_html_report(report: &PerformanceReport) -> PerformanceRegressionResult<String> {
    let mut html = format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <title>{}</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2 {{ color: #333; }}
        .section {{ margin-bottom: 30px; }}
        .metadata {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>{}</h1>
    <div class="metadata">
        <p><strong>Generated:</strong> {}</p>
        <p><strong>Period:</strong> {} to {}</p>
    </div>
"#,
        report.title,
        report.title,
        report.generated_at.format("%Y-%m-%d %H:%M:%S UTC"),
        report.time_range.0.format("%Y-%m-%d %H:%M:%S UTC"),
        report.time_range.1.format("%Y-%m-%d %H:%M:%S UTC")
    );

    for section in &report.sections {
        html.push_str(&format!(
            "<div class=\"section\">\n<h2>{}</h2>\n",
            section.title
        ));

        match section.content_type {
            SectionContentType::Summary => {
                html.push_str("<div class=\"summary\">\n");
                if let Ok(summary) = serde_json::from_value::<HashMap<String, serde_json::Value>>(
                    section.data.clone(),
                ) {
                    for (key, value) in summary {
                        html.push_str(&format!("<p><strong>{}:</strong> {}</p>\n", key, value));
                    }
                }
                html.push_str("</div>\n");
            }
            SectionContentType::Table => {
                html.push_str("<table>\n");
                // In a real implementation, we'd properly format the table
                html.push_str(&format!("<tr><td>{}</td></tr>\n", section.data));
                html.push_str("</table>\n");
            }
            _ => {
                html.push_str(&format!(
                    "<pre>{}</pre>\n",
                    serde_json::to_string_pretty(&section.data)?
                ));
            }
        }

        html.push_str("</div>\n");
    }

    html.push_str("</body>\n</html>");
    Ok(html)
}

/// Generate markdown report from performance report
pub fn generate_markdown_report(report: &PerformanceReport) -> PerformanceRegressionResult<String> {
    let mut markdown = format!("# {}\n\n", report.title);
    markdown.push_str(&format!(
        "**Generated:** {}\n\n",
        report.generated_at.format("%Y-%m-%d %H:%M:%S UTC")
    ));
    markdown.push_str(&format!(
        "**Period:** {} to {}\n\n",
        report.time_range.0.format("%Y-%m-%d %H:%M:%S UTC"),
        report.time_range.1.format("%Y-%m-%d %H:%M:%S UTC")
    ));

    for section in &report.sections {
        markdown.push_str(&format!("## {}\n\n", section.title));

        match section.content_type {
            SectionContentType::Summary => {
                if let Ok(summary) = serde_json::from_value::<HashMap<String, serde_json::Value>>(
                    section.data.clone(),
                ) {
                    for (key, value) in summary {
                        markdown.push_str(&format!("- **{}:** {}\n", key, value));
                    }
                }
            }
            _ => {
                markdown.push_str("```json\n");
                markdown.push_str(&serde_json::to_string_pretty(&section.data)?);
                markdown.push_str("\n```\n");
            }
        }
        markdown.push_str("\n");
    }

    Ok(markdown)
}

/// Generate dashboard HTML
pub fn generate_dashboard_html(
    title: &str,
    widgets: &[DashboardWidget],
    _metrics_data: &[MetricDataPoint],
) -> PerformanceRegressionResult<String> {
    let mut html = format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <title>{}</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .dashboard {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: 20px; }}
        .widget {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
        .widget-title {{ font-weight: bold; margin-bottom: 10px; }}
    </style>
</head>
<body>
    <h1>{}</h1>
    <div class="dashboard">
"#,
        title, title
    );

    for widget in widgets {
        let style = format!(
            "grid-column: span {}; grid-row: span {};",
            widget.size.0, widget.size.1
        );

        html.push_str(&format!(
            r#"
        <div class="widget" style="{}">
            <div class="widget-title">{}</div>
            <div class="widget-content">
                <!-- Chart placeholder for {} -->
                <p>Chart Type: {:?}</p>
                <p>Metrics: {:?}</p>
            </div>
        </div>
"#,
            style, widget.title, widget.id, widget.chart_type, widget.metrics
        ));
    }

    html.push_str(
        r#"
    </div>
</body>
</html>"#,
    );

    Ok(html)
}

/// Generate CSV content from metrics data
pub fn generate_csv_content(
    metrics_data: &[MetricDataPoint],
) -> PerformanceRegressionResult<String> {
    let mut csv = String::from("metric_type,timestamp,value\n");

    for metric in metrics_data {
        csv.push_str(&format!(
            "{:?},{},{}\n",
            metric.metric_type,
            metric.timestamp.to_rfc3339(),
            metric.value
        ));
    }

    Ok(csv)
}
