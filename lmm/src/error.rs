// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Error Types
//!
//! This module defines the unified error type [`LmmError`] used throughout the `lmm` crate,
//! along with the convenience [`Result`] type alias. Every subsystem maps its failure
//! conditions onto a variant of [`LmmError`] so that callers can handle errors with a single
//! `match` branch.
//!
//! # See Also
//! - [`thiserror`](https://docs.rs/thiserror) - derive macro used to automatically generate `std::error::Error` and `Display` implementations.

use thiserror::Error;

/// The unified error type for the `lmm` crate.
///
/// Each variant corresponds to a distinct failure domain. String payloads carry
/// human-readable context; structured variants expose typed fields where useful.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::error::{LmmError, Result};
///
/// fn divide(a: f64, b: f64) -> Result<f64> {
///     if b == 0.0 {
///         return Err(LmmError::DivisionByZero);
///     }
///     Ok(a / b)
/// }
///
/// assert!(divide(4.0, 2.0).is_ok());
/// assert!(matches!(divide(1.0, 0.0), Err(LmmError::DivisionByZero)));
/// ```
#[derive(Debug, Error)]
pub enum LmmError {
    /// A numerical simulation step failed (e.g. shape mismatch in a `Tensor` operation).
    #[error("Simulation failure: {0}")]
    Simulation(String),

    /// The symbolic regression or equation discovery process encountered an error.
    #[error("Discovery error: {0}")]
    Discovery(String),

    /// A perception or I/O operation failed (e.g. empty input, file not found).
    #[error("Perception error: {0}")]
    Perception(String),

    /// The internal world-model state update failed.
    #[error("World model error: {0}")]
    WorldModel(String),

    /// A neural-operator convolution or Fourier transform failed.
    #[error("Neural operator error: {0}")]
    Operator(String),

    /// The conscious perception-action loop encountered an inconsistency.
    #[error("Consciousness loop error: {0}")]
    Consciousness(String),

    /// A mathematical expression could not be parsed or evaluated.
    #[error("Invalid mathematical expression")]
    InvalidExpression,

    /// An iterative algorithm did not converge within the permitted number of steps.
    #[error("Computation timeout")]
    Timeout,

    /// An iterative algorithm failed to converge after `{0}` iterations.
    #[error("Convergence failure after {0} iterations")]
    ConvergenceError(usize),

    /// A tensor or matrix dimension was wrong; carries the expected and actual sizes.
    #[error("Invalid dimension: expected {expected}, got {got}")]
    InvalidDimension {
        /// The dimension that was required.
        expected: usize,
        /// The dimension that was actually provided.
        got: usize,
    },

    /// A string could not be parsed into the expected type.
    #[error("Parse error: {0}")]
    ParseError(String),

    /// A causal graph operation failed (e.g. cycle detected, unknown node name).
    #[error("Causal graph error: {0}")]
    CausalError(String),

    /// Division by zero was attempted during expression evaluation.
    #[error("Division by zero")]
    DivisionByZero,
}

/// Convenience `Result` alias that pins the error type to [`LmmError`].
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::error::{LmmError, Result};
///
/// fn parse_positive(s: &str) -> Result<f64> {
///     let v: f64 = s.parse().map_err(|e: std::num::ParseFloatError| {
///         LmmError::ParseError(e.to_string())
///     })?;
///     if v < 0.0 {
///         return Err(LmmError::InvalidExpression);
///     }
///     Ok(v)
/// }
///
/// assert_eq!(parse_positive("3.14").unwrap(), 3.14);
/// assert!(parse_positive("-1").is_err());
/// ```
pub type Result<T> = std::result::Result<T, LmmError>;

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
