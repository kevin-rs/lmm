// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `ThinkLoop` - the PID-style closed-loop controller.
//!
//! `ThinkLoop` implements a **discrete-time feedback control system** where:
//!
//! - The **setpoint** is the agent's natural-language goal.
//! - The **plant** is the `SearchOracle` (DuckDuckGo or offline stub).
//! - The **error signal** is the Jaccard token-distance between goal and observation.
//! - The **controller** is a PI (proportional + integral) gain schedule.
//! - The **feedback path** is the reward-weighted memory update.
//!
//! ## Control Law
//!
//! ```text
//! e(k)   = 1 - Jaccard(goal, y(k))          // error
//! I(k)   = clamp(I(k-1) + e(k), 0, 100)     // integral (anti-windup)
//! K(k)   = clamp(Kp + Ki·I(k), 0.1, 10)     // gain schedule
//! r(k)   = (1 - e(k)) · K(k)                // reward
//! ```
//!
//! The loop terminates when any of the following hold:
//!
//! 1. `e(k) < convergence_threshold`               - **converged**.
//! 2. Reward declines for `stall_patience` steps   - **stalled**.
//! 3. `k == max_iterations`                        - **iteration cap**.
//!
//! ## Examples
//!
//! ```rust
//! use lmm_agent::cognition::r#loop::ThinkLoop;
//! use lmm_agent::cognition::search::SearchOracle;
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut oracle = SearchOracle::new(5);
//!     let mut lp = ThinkLoop::new("What is Rust?", 10, 0.25, 1.0, 0.05);
//!     let result = lp.run(&mut oracle).await;
//!     assert!(result.steps <= 10);
//! }
//! ```
//!
//! ## See Also
//!
//! * [PID controller - Wikipedia](https://en.wikipedia.org/wiki/PID_controller)
//! * [Feedback - Wikipedia](https://en.wikipedia.org/wiki/Feedback)

use crate::cognition::goal::GoalEvaluator;
use crate::cognition::memory::{ColdStore, HotStore, MemoryEntry};
use crate::cognition::reflect::Reflector;
use crate::cognition::search::SearchOracle;
use crate::cognition::signal::CognitionSignal;
use crate::types::ThinkResult;

/// The closed-loop controller that drives the agent's reasoning process.
///
/// Build via [`ThinkLoop::new`] or [`ThinkLoop::builder`], then call
/// `.run(&mut oracle).await` to obtain a [`ThinkResult`].
///
/// # Examples
///
/// ```rust
/// use lmm_agent::cognition::r#loop::ThinkLoop;
/// use lmm_agent::cognition::search::SearchOracle;
///
/// #[tokio::main]
/// async fn main() {
///     let mut oracle = SearchOracle::new(3);
///     let mut lp = ThinkLoop::new("Rust memory safety", 5, 0.3, 1.0, 0.05);
///     let result = lp.run(&mut oracle).await;
///     println!("converged={} steps={} error={:.3}", result.converged, result.steps, result.final_error);
/// }
/// ```
#[derive(Debug)]
pub struct ThinkLoop {
    /// Natural-language goal (setpoint).
    pub goal: String,
    /// Maximum number of feedback iterations.
    pub max_iterations: usize,
    /// Jaccard-error threshold below which convergence is declared.
    pub convergence_threshold: f64,
    /// Proportional gain constant (Kp).
    pub k_proportional: f64,
    /// Integral gain constant (Ki) - stored for documentation; applied in `CognitionSignal`.
    pub k_integral: f64,
    /// Consecutive reward-declining steps before stall detection triggers.
    pub stall_patience: usize,
    /// Reward score threshold for promoting hot entries to cold store.
    pub promotion_threshold: f64,
    /// Short-term memory for the current run.
    pub hot: HotStore,
    /// Long-term memory archive.
    pub cold: ColdStore,
}

impl ThinkLoop {
    /// Constructs a `ThinkLoop` with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `goal`                  - natural-language goal / setpoint.
    /// * `max_iterations`        - iteration cap (≥ 1).
    /// * `convergence_threshold` - Jaccard error threshold ∈ [0, 1].
    /// * `k_proportional`        - proportional gain Kp.
    /// * `k_integral`            - integral gain Ki.
    pub fn new(
        goal: impl Into<String>,
        max_iterations: usize,
        convergence_threshold: f64,
        k_proportional: f64,
        k_integral: f64,
    ) -> Self {
        Self {
            goal: goal.into(),
            max_iterations: max_iterations.max(1),
            convergence_threshold: convergence_threshold.clamp(0.0, 1.0),
            k_proportional,
            k_integral,
            stall_patience: 3,
            promotion_threshold: 0.5,
            hot: HotStore::new(16),
            cold: ColdStore::default(),
        }
    }

    /// Returns a builder for ergonomic construction.
    pub fn builder(goal: impl Into<String>) -> ThinkLoopBuilder {
        ThinkLoopBuilder::new(goal)
    }

    /// Sets the stall patience.
    pub fn stall_patience(mut self, n: usize) -> Self {
        self.stall_patience = n.max(1);
        self
    }

    /// Sets the hot→cold promotion reward threshold.
    pub fn promotion_threshold(mut self, t: f64) -> Self {
        self.promotion_threshold = t.clamp(0.0, 1.0);
        self
    }

    /// Runs the closed-loop think cycle and returns a [`ThinkResult`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// #[tokio::main]
    /// async fn main() {
    ///     use lmm_agent::cognition::r#loop::ThinkLoop;
    ///     use lmm_agent::cognition::search::SearchOracle;
    ///
    ///     let mut oracle = SearchOracle::new(5);
    ///     let mut lp = ThinkLoop::new("Rust memory model", 10, 0.25, 1.0, 0.05);
    ///     let r = lp.run(&mut oracle).await;
    ///     assert!(r.steps > 0);
    /// }
    /// ```
    pub async fn run(&mut self, oracle: &mut SearchOracle) -> ThinkResult {
        let evaluator = GoalEvaluator::new(self.convergence_threshold);
        let mut integral = 0.0_f64;
        let mut prev_reward = f64::NEG_INFINITY;
        let mut stall_streak = 0_usize;
        let mut converged = false;
        let mut final_error = 1.0_f64;
        let mut signals: Vec<CognitionSignal> = Vec::with_capacity(self.max_iterations);

        for step in 0..self.max_iterations {
            let query = Reflector::formulate_query(&self.goal, &self.hot);

            let observation = oracle.fetch(&query).await;

            let signal = CognitionSignal::new(
                step,
                query.clone(),
                observation.clone(),
                self.k_proportional,
                integral,
            );
            integral = signal.integral;
            let current_reward = signal.reward;
            final_error = signal.error;

            self.hot.push(MemoryEntry::new(
                if observation.is_empty() {
                    query
                } else {
                    observation
                },
                current_reward,
                step,
            ));

            signals.push(signal);

            if evaluator.is_converged(final_error) {
                converged = true;
                break;
            }

            if current_reward < prev_reward {
                stall_streak += 1;
                if stall_streak >= self.stall_patience {
                    break;
                }
            } else {
                stall_streak = 0;
            }
            prev_reward = current_reward;
        }

        Reflector::drain_to_cold(&mut self.hot, &mut self.cold, self.promotion_threshold);

        let steps = signals.len();
        ThinkResult {
            converged,
            steps,
            final_error,
            memory_snapshot: self.hot.snapshot(),
            signals,
        }
    }
}

/// Fluent builder for [`ThinkLoop`].
///
/// # Examples
///
/// ```rust
/// use lmm_agent::cognition::r#loop::ThinkLoop;
///
/// let lp = ThinkLoop::builder("What is ownership?")
///     .max_iterations(5)
///     .convergence_threshold(0.3)
///     .build();
/// assert_eq!(lp.max_iterations, 5);
/// ```
pub struct ThinkLoopBuilder {
    goal: String,
    max_iterations: usize,
    convergence_threshold: f64,
    k_proportional: f64,
    k_integral: f64,
    stall_patience: usize,
    promotion_threshold: f64,
    hot_capacity: usize,
}

impl ThinkLoopBuilder {
    fn new(goal: impl Into<String>) -> Self {
        Self {
            goal: goal.into(),
            max_iterations: 10,
            convergence_threshold: 0.25,
            k_proportional: 1.0,
            k_integral: 0.05,
            stall_patience: 3,
            promotion_threshold: 0.5,
            hot_capacity: 16,
        }
    }

    /// Sets the iteration cap (default: 10).
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }
    /// Sets the convergence threshold (default: 0.25).
    pub fn convergence_threshold(mut self, t: f64) -> Self {
        self.convergence_threshold = t;
        self
    }
    /// Sets the proportional gain Kp (default: 1.0).
    pub fn k_proportional(mut self, kp: f64) -> Self {
        self.k_proportional = kp;
        self
    }
    /// Sets the integral gain Ki (default: 0.05).
    pub fn k_integral(mut self, ki: f64) -> Self {
        self.k_integral = ki;
        self
    }
    /// Sets the stall patience (default: 3).
    pub fn stall_patience(mut self, n: usize) -> Self {
        self.stall_patience = n;
        self
    }
    /// Sets the hot→cold promotion threshold (default: 0.5).
    pub fn promotion_threshold(mut self, t: f64) -> Self {
        self.promotion_threshold = t;
        self
    }
    /// Sets the hot store capacity (default: 16).
    pub fn hot_capacity(mut self, cap: usize) -> Self {
        self.hot_capacity = cap;
        self
    }

    /// Builds the [`ThinkLoop`].
    pub fn build(self) -> ThinkLoop {
        ThinkLoop {
            goal: self.goal,
            max_iterations: self.max_iterations.max(1),
            convergence_threshold: self.convergence_threshold.clamp(0.0, 1.0),
            k_proportional: self.k_proportional,
            k_integral: self.k_integral,
            stall_patience: self.stall_patience.max(1),
            promotion_threshold: self.promotion_threshold.clamp(0.0, 1.0),
            hot: HotStore::new(self.hot_capacity.max(1)),
            cold: ColdStore::default(),
        }
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
