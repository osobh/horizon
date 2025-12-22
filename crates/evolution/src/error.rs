//! Evolution error types

use thiserror::Error;

#[derive(Error, Debug)]
pub enum EvolutionError {
    #[error("Population empty")]
    PopulationEmpty,

    #[error("Invalid mutation parameters")]
    InvalidMutation,

    #[error("Fitness evaluation failed: {reason}")]
    FitnessEvaluationFailed { reason: String },
}
