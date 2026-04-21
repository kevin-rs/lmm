// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `GoalEvaluator` - convergence checker.
//!
//! In control-theory terms this is the **comparator** block that asks:
//! "Is the plant output close enough to the setpoint?" When the Jaccard
//! token-distance between the agent goal and the current observation drops
//! below `convergence_threshold`, the loop is declared converged.
//!
//! ## Examples
//!
//! ```rust
//! use lmm_agent::cognition::goal::GoalEvaluator;
//!
//! let eval = GoalEvaluator::new(0.3);
//! assert!(eval.is_converged(0.1));
//! assert!(!eval.is_converged(0.5));
//!
//! let error = GoalEvaluator::error("Rust is fast", "Rust is memory safe and fast");
//! assert!(error < 1.0 && error >= 0.0);
//! ```
//!
//! ## See Also
//!
//! * [Goal pursuit - Wikipedia](https://en.wikipedia.org/wiki/Goal_pursuit)
//! * [Intelligent agent - Wikipedia](https://en.wikipedia.org/wiki/Intelligent_agent)
//! * [Convergence (mathematics) - Wikipedia](https://en.wikipedia.org/wiki/Convergence#Mathematics)

/// Decides when the `ThinkLoop` has converged on a satisfactory answer.
///
/// # Examples
///
/// ```rust
/// use lmm_agent::cognition::goal::GoalEvaluator;
///
/// let eval = GoalEvaluator::new(0.25);
/// assert!(eval.is_converged(0.0));
/// assert!(!eval.is_converged(0.9));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct GoalEvaluator {
    /// The error threshold below which the loop declares convergence.
    ///
    /// Typical values: 0.1 (strict) to 0.4 (relaxed).
    pub convergence_threshold: f64,
}

impl GoalEvaluator {
    /// Constructs a new `GoalEvaluator` with the given threshold.
    ///
    /// # Panics
    ///
    /// Panics if `threshold` is not in [0, 1].
    pub fn new(threshold: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&threshold),
            "GoalEvaluator threshold must be in [0, 1], got {threshold}"
        );
        Self {
            convergence_threshold: threshold,
        }
    }

    /// Returns `true` when `error` is strictly below the convergence threshold.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::goal::GoalEvaluator;
    ///
    /// let eval = GoalEvaluator::new(0.3);
    /// assert!(eval.is_converged(0.29));
    /// assert!(!eval.is_converged(0.3));
    /// ```
    pub fn is_converged(&self, error: f64) -> bool {
        error < self.convergence_threshold
    }

    /// Computes the Jaccard token-overlap **error** between `goal` and `observation`.
    ///
    /// Returns a value in [0, 1] where `0.0` = identical token sets.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::goal::GoalEvaluator;
    ///
    /// assert_eq!(GoalEvaluator::error("hello world", "hello world"), 0.0);
    /// assert_eq!(GoalEvaluator::error("foo", "bar"), 1.0);
    /// ```
    pub fn error(goal: &str, observation: &str) -> f64 {
        crate::cognition::signal::error_from_texts(goal, observation)
    }

    /// Returns a progress percentage (0-100) given an error value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::goal::GoalEvaluator;
    ///
    /// assert_eq!(GoalEvaluator::progress(0.0), 100.0);
    /// assert_eq!(GoalEvaluator::progress(1.0), 0.0);
    /// ```
    pub fn progress(error: f64) -> f64 {
        ((1.0 - error) * 100.0).clamp(0.0, 100.0)
    }
}

impl Default for GoalEvaluator {
    /// Default convergence threshold is **0.25**.
    fn default() -> Self {
        Self::new(0.25)
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
