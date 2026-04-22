// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `MetaAdapter` - prototype-based meta-learning.
//!
//! Enables few-shot and zero-shot adaptation by matching new tasks to the
//! most similar previously seen task prototypes and applying their learned
//! Q-value offsets as a warm-start - without any gradient descent or matrices.
//!
//! ## Algorithm
//!
//! 1. After each `ThinkLoop` run, `record_episode` stores a `TaskPrototype`
//!    consisting of the goal's token set (centroid) and the final Q-value
//!    offset for each action (best Q minus baseline 0.0).
//!
//! 2. On a new task, `adapt` computes the Jaccard similarity between the new
//!    goal and every stored prototype, takes the top-K matches, and produces
//!    a weighted average of their offsets as the warm-start Q-adjustment.
//!
//! 3. These offsets are returned as a `HashMap<ActionKey, f64>` that the
//!    `LearningEngine` can apply as additive priors to the new task's Q-table
//!    rows before any TD updates.
//!
//! ## Complexity
//!
//! - `record_episode`: O(P) - P = number of stored prototypes.
//! - `adapt`:          O(P · T) - T = average goal token count.
//!
//! Both are entirely CPU-bound hash-set operations.
//!
//! ## Examples
//!
//! ```rust
//! use std::collections::HashMap;
//! use lmm_agent::cognition::learning::meta::MetaAdapter;
//! use lmm_agent::cognition::learning::q_table::ActionKey;
//!
//! let mut adapter = MetaAdapter::new(3);
//!
//! let mut offsets = HashMap::new();
//! offsets.insert(ActionKey::Narrow, 0.5);
//! adapter.record_episode("rust ownership borrow", offsets, 0.8);
//!
//! let adapt = adapter.adapt("rust memory borrow checker");
//! assert!(!adapt.is_empty());
//! ```

use crate::cognition::learning::q_table::ActionKey;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

/// A stored snapshot of a completed task episode used as a meta-learning prototype.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskPrototype {
    /// Lowercase token set of the task goal.
    pub tokens: HashSet<String>,

    /// Per-action Q-value offsets learned during this episode.
    pub offsets: HashMap<ActionKey, f64>,

    /// Average reward achieved during the episode.
    pub avg_reward: f64,

    /// Number of times this prototype has been matched and applied.
    pub match_count: usize,
}

impl TaskPrototype {
    /// Computes Jaccard similarity between this prototype and a query token set.
    ///
    /// Returns a value in [0, 1] where 1.0 = exact token-set match.
    pub fn similarity(&self, other: &HashSet<String>) -> f64 {
        let intersection = self.tokens.intersection(other).count();
        let union = self.tokens.len() + other.len() - intersection;
        if union == 0 {
            1.0
        } else {
            intersection as f64 / union as f64
        }
    }
}

/// Prototype store for task-level meta-adaptation.
///
/// # Examples
///
/// ```rust
/// use std::collections::HashMap;
/// use lmm_agent::cognition::learning::meta::MetaAdapter;
/// use lmm_agent::cognition::learning::q_table::ActionKey;
///
/// let mut a = MetaAdapter::new(5);
/// let offsets = HashMap::from([(ActionKey::Expand, 0.7)]);
/// a.record_episode("rust async await", offsets, 1.0);
/// let r = a.adapt("async rust futures");
/// assert!(r.contains_key(&ActionKey::Expand));
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetaAdapter {
    /// Stored episode prototypes.
    prototypes: Vec<TaskPrototype>,

    /// Maximum number of prototypes considered per `adapt` call.
    pub top_k: usize,
}

impl MetaAdapter {
    /// Constructs a new `MetaAdapter` with the given top-k lookup limit.
    pub fn new(top_k: usize) -> Self {
        Self {
            prototypes: Vec::new(),
            top_k: top_k.max(1),
        }
    }

    /// Returns the number of stored prototypes.
    pub fn len(&self) -> usize {
        self.prototypes.len()
    }

    /// Returns `true` when no prototypes have been stored.
    pub fn is_empty(&self) -> bool {
        self.prototypes.is_empty()
    }

    /// Stores a new episode as a task prototype.
    ///
    /// `goal` is tokenised into a `HashSet<String>`. If a nearly identical
    /// prototype already exists (Jaccard ≥ 0.9), its offsets are blended
    /// rather than creating a duplicate.
    ///
    /// # Arguments
    ///
    /// * `goal`       - Natural-language task goal.
    /// * `offsets`    - Per-action Q-value offsets from the completed episode.
    /// * `avg_reward` - Mean reward across the episode steps.
    pub fn record_episode(
        &mut self,
        goal: &str,
        offsets: HashMap<ActionKey, f64>,
        avg_reward: f64,
    ) {
        let tokens = tokenise(goal);

        if let Some(existing) = self
            .prototypes
            .iter_mut()
            .find(|p| p.similarity(&tokens) >= 0.9)
        {
            for (action, val) in &offsets {
                let e = existing.offsets.entry(*action).or_insert(0.0);
                *e = (*e + val) / 2.0;
            }
            existing.avg_reward = (existing.avg_reward + avg_reward) / 2.0;
            return;
        }

        self.prototypes.push(TaskPrototype {
            tokens,
            offsets,
            avg_reward,
            match_count: 0,
        });
    }

    /// Returns weighted Q-offset priors for a new `goal`.
    ///
    /// Finds the top-K most similar prototypes, weights their offsets by
    /// `similarity × avg_reward`, and returns the normalised blend.
    ///
    /// Returns an empty map when no prototypes exist.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::collections::HashMap;
    /// use lmm_agent::cognition::learning::meta::MetaAdapter;
    /// use lmm_agent::cognition::learning::q_table::ActionKey;
    ///
    /// let mut a = MetaAdapter::new(3);
    /// a.record_episode("machine learning", HashMap::from([(ActionKey::Expand, 0.6)]), 0.7);
    /// let out = a.adapt("deep learning models");
    /// assert!(!out.is_empty());
    /// ```
    pub fn adapt(&mut self, goal: &str) -> HashMap<ActionKey, f64> {
        if self.prototypes.is_empty() {
            return HashMap::new();
        }

        let tokens = tokenise(goal);

        let mut scored: Vec<(usize, f64)> = self
            .prototypes
            .iter()
            .enumerate()
            .map(|(i, p)| (i, p.similarity(&tokens) * p.avg_reward))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scored.truncate(self.top_k);

        let total_weight: f64 = scored.iter().map(|(_, w)| w).sum();
        if total_weight <= 0.0 {
            return HashMap::new();
        }

        let mut blended: HashMap<ActionKey, f64> = HashMap::new();
        for (idx, weight) in &scored {
            self.prototypes[*idx].match_count += 1;
            for (&action, &val) in &self.prototypes[*idx].offsets {
                *blended.entry(action).or_insert(0.0) += (weight / total_weight) * val;
            }
        }

        blended
    }

    /// Returns a slice over all stored prototypes.
    pub fn prototypes(&self) -> &[TaskPrototype] {
        &self.prototypes
    }
}

fn tokenise(text: &str) -> HashSet<String> {
    text.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_ascii_lowercase()
        })
        .filter(|s| s.len() >= 3)
        .collect()
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
