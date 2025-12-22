pub mod csv;
pub mod json;
pub mod markdown;

pub use self::csv::CsvExporter;
pub use self::json::JsonExporter;
pub use self::markdown::MarkdownExporter;

use crate::error::Result;
use crate::models::summary::CostAttribution;

/// Trait for export formats
pub trait Exporter {
    fn export(&self, attributions: &[CostAttribution]) -> Result<String>;
}
