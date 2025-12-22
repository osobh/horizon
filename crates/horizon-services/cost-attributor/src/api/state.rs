use crate::{config::Config, db::Repository};
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub repository: Arc<Repository>,
    pub config: Arc<Config>,
}

impl AppState {
    pub fn new(repository: Repository, config: Config) -> Self {
        Self {
            repository: Arc::new(repository),
            config: Arc::new(config),
        }
    }
}
