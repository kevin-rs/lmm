// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `CognitionSignal` - the scalar that flows around the ThinkLoop.
//!
//! Each iteration of the closed-loop controller produces one `CognitionSignal`.
//!
//! ## Fields
//!
//! | Field         | Control-theory analogue        | Description                                   |
//! |---------------|--------------------------------|-----------------------------------------------|
//! | `step`        | sample index k                 | Loop iteration counter (0-based)              |
//! | `error`       | e(k) = setpoint - output       | Jaccard distance ∈ [0, 1] to goal             |
//! | `gain`        | Kp + Ki·∑e                     | Scheduled gain for this step                  |
//! | `integral`    | ∑e(0..k)                       | Accumulated error (anti-windup clamped)       |
//! | `reward`      | r(k) = (1 - e) · gain          | Positive reinforcement delivered this step    |
//! | `query`       | reference input r              | Query sent to the `SearchOracle`              |
//! | `observation` | plant output y                 | Raw text from `SearchOracle` (may be empty)   |
//!
//! ## Examples
//!
//! ```rust
//! use lmm_agent::cognition::signal::CognitionSignal;
//!
//! let sig = CognitionSignal::new(0, "What is Rust?".into(), "Rust is a systems language.".into(), 1.0, 0.0);
//! assert!(sig.error >= 0.0 && sig.error <= 1.0);
//! assert!(sig.reward >= 0.0);
//! ```
//!
//! ## See Also
//!
//! * [Jaccard index - Wikipedia](https://en.wikipedia.org/wiki/Jaccard_index)
//! * [Signal processing - Wikipedia](https://en.wikipedia.org/wiki/Signal_processing)

use std::collections::HashSet;

/// The scalar signal produced by one iteration of the closed-loop controller.
///
/// # Examples
///
/// ```rust
/// use lmm_agent::cognition::signal::CognitionSignal;
///
/// let sig = CognitionSignal::new(0, "goal text".into(), "goal text".into(), 1.0, 0.0);
/// // Perfect match → error = 0.0
/// assert_eq!(sig.error, 0.0);
/// assert!(sig.reward > 0.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CognitionSignal {
    /// Loop iteration counter (0-based).
    pub step: usize,

    /// Jaccard distance between goal and observation, ∈ [0, 1].
    ///
    /// 0.0 = perfect match (converged), 1.0 = no common tokens.
    pub error: f64,

    /// Current proportional + integral gain: `Kp + Ki * integral`.
    pub gain: f64,

    /// Accumulated error sum (anti-windup clamped to [0, 100]).
    pub integral: f64,

    /// Positive reinforcement for this step: `(1.0 - error) × gain`.
    pub reward: f64,

    /// The DuckDuckGo query issued this step.
    pub query: String,

    /// Raw text returned by the `SearchOracle` this step.
    pub observation: String,
}

impl CognitionSignal {
    /// Constructs a new `CognitionSignal`, computing `error`, `gain`, and `reward`
    /// automatically from `goal`, `observation`, `k_proportional`, and `integral_in`.
    ///
    /// ## Arguments
    ///
    /// * `step`           - current loop iteration (0-based).
    /// * `query`          - the DuckDuckGo query string.
    /// * `observation`    - text returned by the `SearchOracle`.
    /// * `k_proportional` - proportional gain constant Kp.
    /// * `integral_in`    - accumulated error from previous steps.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::signal::CognitionSignal;
    ///
    /// let sig = CognitionSignal::new(1, "Rust ownership".into(), "Rust has ownership".into(), 1.0, 0.5);
    /// println!("step={} error={:.3} reward={:.3}", sig.step, sig.error, sig.reward);
    /// ```
    pub fn new(
        step: usize,
        query: String,
        observation: String,
        k_proportional: f64,
        integral_in: f64,
    ) -> Self {
        let error = error_from_texts(&query, &observation);
        let integral = (integral_in + error).clamp(0.0, 100.0);
        let k_integral = 0.05_f64;
        let gain = (k_proportional + k_integral * integral).clamp(0.1, 10.0);
        let reward = ((1.0 - error) * gain).max(0.0);

        Self {
            step,
            error,
            gain,
            integral,
            reward,
            query,
            observation,
        }
    }
}

/// Computes the **Jaccard token-overlap distance** between `goal` and `text`.
///
/// Returns a value in [0, 1]:
/// - `0.0` - identical token sets (fully converged).
/// - `1.0` - disjoint token sets (no overlap at all).
///
/// Case-insensitive; punctuation is stripped.
///
/// # Examples
///
/// ```rust
/// use lmm_agent::cognition::signal::error_from_texts;
///
/// assert_eq!(error_from_texts("hello world", "hello world"), 0.0);
/// assert_eq!(error_from_texts("foo", "bar"), 1.0);
/// let e = error_from_texts("Rust is fast", "Rust is safe and fast");
/// assert!(e < 1.0 && e > 0.0);
/// ```
pub fn error_from_texts(goal: &str, text: &str) -> f64 {
    let goal_tokens = tokenize(goal);
    let text_tokens = tokenize(text);

    if goal_tokens.is_empty() && text_tokens.is_empty() {
        return 0.0;
    }
    if goal_tokens.is_empty() || text_tokens.is_empty() {
        return 1.0;
    }

    let intersection = goal_tokens.intersection(&text_tokens).count();
    let union = goal_tokens.len() + text_tokens.len() - intersection;

    if union == 0 {
        return 0.0;
    }

    (1.0 - intersection as f64 / union as f64).clamp(0.0, 1.0)
}

/// Lowercases, strips punctuation, splits on whitespace, and deduplicates tokens.
fn tokenize(text: &str) -> HashSet<String> {
    text.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_ascii_lowercase()
        })
        .filter(|s| !s.is_empty())
        .collect()
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
