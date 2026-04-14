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
    #[error("Convergence failure after {0} iterations")]
    ConvergenceError(usize),
    #[error("Invalid dimension: expected {expected}, got {got}")]
    InvalidDimension { expected: usize, got: usize },
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Causal graph error: {0}")]
    CausalError(String),
    #[error("Division by zero")]
    DivisionByZero,
}

pub type Result<T> = std::result::Result<T, LmmError>;
