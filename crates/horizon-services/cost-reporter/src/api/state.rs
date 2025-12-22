use sqlx::PgPool;
use std::sync::Arc;

use crate::config::ReporterConfig;
use crate::db::{Repository, ViewManager};
use crate::export::{CsvExporter, JsonExporter, MarkdownExporter};
use crate::reports::{ChargebackGenerator, CostForecaster, ShowbackGenerator, TrendAnalyzer};

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<ReporterConfig>,
    pub db: PgPool,
    pub repository: Arc<Repository>,
    pub view_manager: Arc<ViewManager>,
    pub showback_generator: Arc<ShowbackGenerator>,
    pub chargeback_generator: Arc<ChargebackGenerator>,
    pub trend_analyzer: Arc<TrendAnalyzer>,
    pub forecaster: Arc<CostForecaster>,
    pub csv_exporter: Arc<CsvExporter>,
    pub json_exporter: Arc<JsonExporter>,
    pub markdown_exporter: Arc<MarkdownExporter>,
}

impl AppState {
    pub fn new(config: ReporterConfig, pool: PgPool) -> Self {
        let repository = Arc::new(Repository::new(pool.clone()));
        let view_manager = Arc::new(ViewManager::new(pool.clone()));

        let showback_generator = Arc::new(ShowbackGenerator::new(repository.as_ref().clone()));
        let chargeback_generator = Arc::new(ChargebackGenerator::new(repository.as_ref().clone()));
        let trend_analyzer = Arc::new(TrendAnalyzer::new());
        let forecaster = Arc::new(CostForecaster::new());

        let csv_exporter = Arc::new(CsvExporter::new());
        let json_exporter = Arc::new(JsonExporter::new());
        let markdown_exporter = Arc::new(MarkdownExporter::new());

        Self {
            config: Arc::new(config),
            db: pool,
            repository,
            view_manager,
            showback_generator,
            chargeback_generator,
            trend_analyzer,
            forecaster,
            csv_exporter,
            json_exporter,
            markdown_exporter,
        }
    }
}
