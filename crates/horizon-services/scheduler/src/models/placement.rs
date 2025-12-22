use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq)]
pub struct PlacementDecision {
    pub job_id: Uuid,
    pub node_ids: Vec<Uuid>,
    pub gpu_ids: Vec<Uuid>,
    pub score: f64,
}

impl PlacementDecision {
    pub fn new(job_id: Uuid) -> Self {
        Self {
            job_id,
            node_ids: Vec::new(),
            gpu_ids: Vec::new(),
            score: 0.0,
        }
    }
}
