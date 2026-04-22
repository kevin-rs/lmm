// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `LearningConfig` - HELM hyperparameter configuration.
//!
//! All hyperparameters for the HELM learning engine are consolidated here so
//! callers can tune the system with a single struct rather than passing
//! individual values to every sub-module.
//!
//! ## Defaults
//!
//! | Field                  | Default | Description                                       |
//! |------------------------|---------|---------------------------------------------------|
//! | `alpha`                | 0.1     | Q-learning rate                                   |
//! | `gamma`                | 0.9     | Discount factor                                   |
//! | `epsilon`              | 0.3     | Initial ε-greedy exploration probability          |
//! | `epsilon_decay`        | 0.99    | Per-episode ε decay multiplier                    |
//! | `epsilon_min`          | 0.01    | Minimum ε floor                                   |
//! | `distill_top_k`        | 8       | Cold entries ingested per distillation pass       |
//! | `distill_threshold`    | 0.4     | Minimum reward for distillation eligibility       |
//! | `elastic_lambda`       | 0.5     | Elastic penalty strength                          |
//! | `elastic_pin_count`    | 3       | Activation count before an entry is pinned        |
//! | `federated_blend`      | 0.5     | Local weight in federated merge: 0=remote, 1=local|
//! | `meta_top_k`           | 3       | Prototypes considered for meta-adaptation         |
//! | `pmi_min_count`        | 2       | Minimum co-occurrence count for PMI pairs         |
//! | `active_modes`         | all     | Bitmask of enabled `LearningMode`s                |
//!
//! ## Examples
//!
//! ```rust
//! use lmm_agent::cognition::learning::config::LearningConfig;
//!
//! let cfg = LearningConfig::default();
//! assert_eq!(cfg.alpha, 0.1);
//! assert_eq!(cfg.gamma, 0.9);
//! assert!(cfg.epsilon > 0.0);
//! ```

use crate::types::LearningMode;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Full HELM hyperparameter configuration.
///
/// # Examples
///
/// ```rust
/// use lmm_agent::cognition::learning::config::LearningConfig;
///
/// let cfg = LearningConfig::builder()
///     .alpha(0.05)
///     .gamma(0.95)
///     .epsilon(0.2)
///     .build();
///
/// assert_eq!(cfg.alpha, 0.05);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LearningConfig {
    /// Temporal-difference learning rate α ∈ (0, 1].
    pub alpha: f64,

    /// Discount factor γ ∈ [0, 1].
    pub gamma: f64,

    /// Initial ε-greedy exploration probability ∈ (0, 1].
    pub epsilon: f64,

    /// Per-episode ε decay multiplier ∈ (0, 1].
    pub epsilon_decay: f64,

    /// Minimum exploration probability floor.
    pub epsilon_min: f64,

    /// Number of top cold entries promoted to the knowledge index per distillation.
    pub distill_top_k: usize,

    /// Minimum reward score required for a cold entry to be distillation-eligible.
    pub distill_threshold: f64,

    /// Elastic penalty strength λ.
    pub elastic_lambda: f64,

    /// Number of recall activations before an entry is considered "pinned."
    pub elastic_pin_count: usize,

    /// Blend weight for federated merge: 1.0 = keep local only, 0.0 = remote only.
    pub federated_blend: f64,

    /// Number of task prototypes considered when computing meta-adaptation offsets.
    pub meta_top_k: usize,

    /// Minimum token-pair co-occurrence count required before PMI is computed.
    pub pmi_min_count: usize,

    /// Set of learning modes that are active.
    pub active_modes: HashSet<LearningMode>,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            gamma: 0.9,
            epsilon: 0.3,
            epsilon_decay: 0.99,
            epsilon_min: 0.01,
            distill_top_k: 8,
            distill_threshold: 0.4,
            elastic_lambda: 0.5,
            elastic_pin_count: 3,
            federated_blend: 0.5,
            meta_top_k: 3,
            pmi_min_count: 2,
            active_modes: LearningMode::all(),
        }
    }
}

impl LearningConfig {
    /// Returns a [`LearningConfigBuilder`] for ergonomic construction.
    pub fn builder() -> LearningConfigBuilder {
        LearningConfigBuilder::default()
    }

    /// Returns `true` when the given `mode` is active.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::learning::config::LearningConfig;
    /// use lmm_agent::types::LearningMode;
    ///
    /// let cfg = LearningConfig::default();
    /// assert!(cfg.is_mode_active(LearningMode::QTable));
    /// ```
    pub fn is_mode_active(&self, mode: LearningMode) -> bool {
        self.active_modes.contains(&mode)
    }
}

/// Fluent builder for [`LearningConfig`].
#[derive(Debug, Default)]
pub struct LearningConfigBuilder {
    alpha: Option<f64>,
    gamma: Option<f64>,
    epsilon: Option<f64>,
    epsilon_decay: Option<f64>,
    epsilon_min: Option<f64>,
    distill_top_k: Option<usize>,
    distill_threshold: Option<f64>,
    elastic_lambda: Option<f64>,
    elastic_pin_count: Option<usize>,
    federated_blend: Option<f64>,
    meta_top_k: Option<usize>,
    pmi_min_count: Option<usize>,
    active_modes: Option<HashSet<LearningMode>>,
}

impl LearningConfigBuilder {
    /// Sets the TD learning rate α.
    pub fn alpha(mut self, v: f64) -> Self {
        self.alpha = Some(v.clamp(1e-6, 1.0));
        self
    }

    /// Sets the discount factor γ.
    pub fn gamma(mut self, v: f64) -> Self {
        self.gamma = Some(v.clamp(0.0, 1.0));
        self
    }

    /// Sets the initial exploration probability ε.
    pub fn epsilon(mut self, v: f64) -> Self {
        self.epsilon = Some(v.clamp(0.0, 1.0));
        self
    }

    /// Sets the per-episode ε decay multiplier.
    pub fn epsilon_decay(mut self, v: f64) -> Self {
        self.epsilon_decay = Some(v.clamp(0.5, 1.0));
        self
    }

    /// Sets the minimum ε floor.
    pub fn epsilon_min(mut self, v: f64) -> Self {
        self.epsilon_min = Some(v.clamp(0.0, 0.5));
        self
    }

    /// Sets how many cold entries are promoted per distillation pass.
    pub fn distill_top_k(mut self, k: usize) -> Self {
        self.distill_top_k = Some(k.max(1));
        self
    }

    /// Sets the distillation reward threshold.
    pub fn distill_threshold(mut self, v: f64) -> Self {
        self.distill_threshold = Some(v.clamp(0.0, 1.0));
        self
    }

    /// Sets the elastic penalty λ.
    pub fn elastic_lambda(mut self, v: f64) -> Self {
        self.elastic_lambda = Some(v.clamp(0.0, 10.0));
        self
    }

    /// Sets the activation count threshold for pinning.
    pub fn elastic_pin_count(mut self, n: usize) -> Self {
        self.elastic_pin_count = Some(n.max(1));
        self
    }

    /// Sets the federated blend weight.
    pub fn federated_blend(mut self, v: f64) -> Self {
        self.federated_blend = Some(v.clamp(0.0, 1.0));
        self
    }

    /// Sets the number of meta-prototypes considered per lookup.
    pub fn meta_top_k(mut self, k: usize) -> Self {
        self.meta_top_k = Some(k.max(1));
        self
    }

    /// Sets the minimum PMI co-occurrence count.
    pub fn pmi_min_count(mut self, n: usize) -> Self {
        self.pmi_min_count = Some(n.max(1));
        self
    }

    /// Sets the enabled learning modes.
    pub fn active_modes(mut self, modes: HashSet<LearningMode>) -> Self {
        self.active_modes = Some(modes);
        self
    }

    /// Builds the [`LearningConfig`] by applying all overrides on top of defaults.
    pub fn build(self) -> LearningConfig {
        let defaults = LearningConfig::default();
        LearningConfig {
            alpha: self.alpha.unwrap_or(defaults.alpha),
            gamma: self.gamma.unwrap_or(defaults.gamma),
            epsilon: self.epsilon.unwrap_or(defaults.epsilon),
            epsilon_decay: self.epsilon_decay.unwrap_or(defaults.epsilon_decay),
            epsilon_min: self.epsilon_min.unwrap_or(defaults.epsilon_min),
            distill_top_k: self.distill_top_k.unwrap_or(defaults.distill_top_k),
            distill_threshold: self.distill_threshold.unwrap_or(defaults.distill_threshold),
            elastic_lambda: self.elastic_lambda.unwrap_or(defaults.elastic_lambda),
            elastic_pin_count: self.elastic_pin_count.unwrap_or(defaults.elastic_pin_count),
            federated_blend: self.federated_blend.unwrap_or(defaults.federated_blend),
            meta_top_k: self.meta_top_k.unwrap_or(defaults.meta_top_k),
            pmi_min_count: self.pmi_min_count.unwrap_or(defaults.pmi_min_count),
            active_modes: self.active_modes.unwrap_or(defaults.active_modes),
        }
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
