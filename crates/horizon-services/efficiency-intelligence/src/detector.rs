use crate::db::EfficiencyRepository;
use crate::error::Result;
use crate::models::*;

pub struct WasteDetector {
    repository: EfficiencyRepository,
}

impl WasteDetector {
    pub fn new(repository: EfficiencyRepository) -> Self {
        Self { repository }
    }

    pub async fn scan(&self) -> Result<Vec<WasteDetection>> {
        self.repository.list_detections().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_detector_creation() {
        let pool = sqlx::PgPool::connect_lazy("postgres://localhost/test").unwrap();
        let repo = EfficiencyRepository::new(pool);
        let _detector = WasteDetector::new(repo);
    }
}
