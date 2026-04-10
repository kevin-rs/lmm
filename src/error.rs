use thiserror::Error;

#[derive(Debug, Error)]
pub enum LmmError {
    #[error("Simulation failure: {0}")]
    Simulation(String),
    #[error("Discovery error: {0}")]
    Discovery(String),
    #[error("Perception error: {0}")]
    Perception(String),
    #[error("World model error: {0}")]
    WorldModel(String),
    #[error("Neural operator error: {0}")]
    Operator(String),
    #[error("Consciousness loop error: {0}")]
    Consciousness(String),
    #[error("Invalid mathematical expression")]
    InvalidExpression,
    #[error("Computation timeout")]
    Timeout,
}

pub type Result<T> = std::result::Result<T, LmmError>;
