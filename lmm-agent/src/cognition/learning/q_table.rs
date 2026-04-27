// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `QTable` - tabular Q-learning engine.
//!
//! Implements the **Bellman TD(0)** update rule over a pure hash-map table.
//! No matrices, no GPU, no external ML crates - just `HashMap<u64, HashMap<ActionKey, f64>>`.
//!
//! ## State representation
//!
//! A state is the 64-bit FNV-1a hash of the sorted bag-of-words derived from
//! the current query string. This gives a compact, collision-resistant state key
//! that requires no feature engineering.
//!
//! ## Action space
//!
//! Five discrete query-refinement strategies:
//!
//! | Action    | Effect on next query                            |
//! |-----------|-------------------------------------------------|
//! | `Repeat`  | Re-issue the same query unchanged               |
//! | `Broaden` | Drop the two most specific tokens               |
//! | `Narrow`  | Append the top hot-store token                  |
//! | `Pivot`   | Replace longest token with a related synonym    |
//! | `Expand`  | Append goal tokens not yet in the query         |
//!
//! ## Control law
//!
//! ```text
//! Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') - Q(s, a)]
//! ```
//!
//! ## ε-greedy exploration
//!
//! At each step the agent selects the greedy action with probability `1 - ε`
//! and a random action with probability `ε`. ε decays per episode.
//!
//! ## Examples
//!
//! ```rust
//! use lmm_agent::cognition::learning::q_table::{ActionKey, QTable};
//!
//! let mut qt = QTable::new(0.1, 0.9, 0.3, 0.99, 0.01);
//! let s = QTable::state_key("rust memory safety");
//! qt.update(s, ActionKey::Narrow, 0.8, QTable::state_key("rust memory safety borrow"));
//! let best = qt.best_action(s);
//! assert!(best.is_some());
//! ```

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;

/// The five discrete query-refinement actions available to the agent.
///
/// # Examples
///
/// ```rust
/// use lmm_agent::cognition::learning::q_table::ActionKey;
///
/// assert_ne!(ActionKey::Broaden, ActionKey::Narrow);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionKey {
    /// Re-issue the same query unchanged.
    Repeat,
    /// Drop the two most specific (longest) tokens from the query.
    Broaden,
    /// Append the top hot-store token to the query.
    Narrow,
    /// Replace the longest token with a contextually related term.
    Pivot,
    /// Append goal tokens not yet present in the query.
    Expand,
}

impl ActionKey {
    /// Returns a slice of all five action variants.
    pub fn all() -> &'static [ActionKey] {
        &[
            ActionKey::Repeat,
            ActionKey::Broaden,
            ActionKey::Narrow,
            ActionKey::Pivot,
            ActionKey::Expand,
        ]
    }
}

/// Tabular Q-learning engine.
///
/// # Examples
///
/// ```rust
/// use lmm_agent::cognition::learning::q_table::{ActionKey, QTable};
///
/// let mut qt = QTable::new(0.1, 0.9, 0.3, 0.99, 0.01);
/// let s  = QTable::state_key("rust ownership");
/// let s2 = QTable::state_key("rust ownership borrow checker");
/// qt.update(s, ActionKey::Narrow, 0.75, s2);
/// assert!(qt.q_value(s, ActionKey::Narrow) > 0.0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QTable {
    /// The Q-value table: state → (action → value).
    table: HashMap<u64, HashMap<ActionKey, f64>>,

    /// TD learning rate α.
    pub alpha: f64,

    /// Discount factor γ.
    pub gamma: f64,

    /// Current exploration probability ε.
    pub epsilon: f64,

    /// Per-episode ε decay multiplier.
    epsilon_decay: f64,

    /// Minimum ε floor.
    epsilon_min: f64,

    /// Total number of TD updates applied.
    pub update_count: u64,
}

impl QTable {
    /// Constructs a new `QTable`.
    ///
    /// # Arguments
    ///
    /// * `alpha`         - TD learning rate ∈ (0, 1].
    /// * `gamma`         - Discount factor ∈ [0, 1].
    /// * `epsilon`       - Initial exploration probability ∈ (0, 1].
    /// * `epsilon_decay` - Per-episode ε multiplier ∈ (0, 1].
    /// * `epsilon_min`   - Minimum ε floor ∈ [0, epsilon).
    pub fn new(alpha: f64, gamma: f64, epsilon: f64, epsilon_decay: f64, epsilon_min: f64) -> Self {
        Self {
            table: HashMap::new(),
            alpha: alpha.clamp(1e-6, 1.0),
            gamma: gamma.clamp(0.0, 1.0),
            epsilon: epsilon.clamp(0.0, 1.0),
            epsilon_decay: epsilon_decay.clamp(0.5, 1.0),
            epsilon_min: epsilon_min.clamp(0.0, 0.5),
            update_count: 0,
        }
    }

    /// Returns the number of states stored in the table.
    pub fn state_count(&self) -> usize {
        self.table.len()
    }

    /// Returns `true` when no Q-values have been recorded.
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Hashes `query` into a 64-bit state key using FNV-1a.
    ///
    /// Tokens are sorted and deduplicated before hashing so identical query sets
    /// map to the same state regardless of word order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::learning::q_table::QTable;
    ///
    /// let k1 = QTable::state_key("rust memory safety");
    /// let k2 = QTable::state_key("memory rust safety");
    /// assert_eq!(k1, k2);
    /// ```
    pub fn state_key(query: &str) -> u64 {
        let mut tokens: Vec<String> = query
            .split_whitespace()
            .map(|w| {
                w.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_ascii_lowercase()
            })
            .filter(|s| s.len() >= 2)
            .collect();
        tokens.sort_unstable();
        tokens.dedup();
        fnv1a_hash(&tokens.join(" "))
    }

    /// Returns the current Q-value for `(state, action)`, defaulting to 0.0.
    pub fn q_value(&self, state: u64, action: ActionKey) -> f64 {
        self.table
            .get(&state)
            .and_then(|row| row.get(&action))
            .copied()
            .unwrap_or(0.0)
    }

    /// Returns the action with the highest Q-value for `state`, or `None` when
    /// the state has never been seen.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::learning::q_table::{ActionKey, QTable};
    ///
    /// let mut qt = QTable::new(0.1, 0.9, 0.0, 0.99, 0.01);
    /// let s = QTable::state_key("test query");
    /// qt.update(s, ActionKey::Expand, 1.0, s);
    /// assert_eq!(qt.best_action(s), Some(ActionKey::Expand));
    /// ```
    pub fn best_action(&self, state: u64) -> Option<ActionKey> {
        self.table.get(&state).and_then(|row| {
            row.iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
                .map(|(action, _)| *action)
        })
    }

    /// Returns the maximum Q-value across all actions for `state`, or `0.0`
    /// when the state is unknown.
    pub fn max_q(&self, state: u64) -> f64 {
        self.table
            .get(&state)
            .and_then(|row| {
                row.values()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .copied()
            })
            .unwrap_or(0.0)
    }

    /// Applies the TD(0) update for transition `(state, action, reward, next_state)`.
    ///
    /// ```text
    /// Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') - Q(s, a)]
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::learning::q_table::{ActionKey, QTable};
    ///
    /// let mut qt = QTable::new(0.1, 0.9, 0.0, 1.0, 0.0);
    /// let s  = QTable::state_key("a b");
    /// let s2 = QTable::state_key("a b c");
    /// qt.update(s, ActionKey::Narrow, 1.0, s2);
    ///
    /// let expected = 0.1 * (1.0 + 0.9 * 0.0 - 0.0);
    /// assert!((qt.q_value(s, ActionKey::Narrow) - expected).abs() < 1e-9);
    /// ```
    pub fn update(&mut self, state: u64, action: ActionKey, reward: f64, next_state: u64) {
        let current = self.q_value(state, action);
        let next_max = self.max_q(next_state);
        let target = reward + self.gamma * next_max;
        let updated = current + self.alpha * (target - current);
        self.table.entry(state).or_default().insert(action, updated);
        self.update_count += 1;
    }

    /// Selects an action using ε-greedy policy.
    ///
    /// Returns the greedy action with probability `1 - ε`, otherwise a
    /// randomly chosen action from the five-action space.
    ///
    /// When the state is not yet in the table and ε > 0, always explores.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::learning::q_table::{ActionKey, QTable};
    ///
    /// let mut qt = QTable::new(0.1, 0.9, 0.0, 1.0, 0.0);
    /// let s = QTable::state_key("rust");
    /// qt.update(s, ActionKey::Pivot, 0.9, s);
    /// assert_eq!(qt.select_action(s, 0), ActionKey::Pivot);
    /// ```
    pub fn select_action(&self, state: u64, step: usize) -> ActionKey {
        let explore = pseudo_random(step) < self.epsilon;
        if explore || self.best_action(state).is_none() {
            let idx = pseudo_random(step + 17) * 5.0;
            ActionKey::all()[idx as usize % 5]
        } else {
            self.best_action(state).unwrap_or(ActionKey::Repeat)
        }
    }

    /// Decays ε by `epsilon_decay`, flooring at `epsilon_min`.
    ///
    /// Should be called once per episode (i.e., per completed `ThinkLoop` run).
    pub fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);
    }

    /// Resets ε to a specific value, useful for phase transitions or new levels.
    pub fn reset_epsilon(&mut self, new_epsilon: f64) {
        self.epsilon = new_epsilon.clamp(0.0, 1.0);
    }

    /// Returns an iterator over `(state, action, q_value)` triples.
    pub fn entries(&self) -> impl Iterator<Item = (u64, ActionKey, f64)> + '_ {
        self.table
            .iter()
            .flat_map(|(&s, row)| row.iter().map(move |(&a, &v)| (s, a, v)))
    }

    /// Merges `other` into `self` using a weighted average.
    ///
    /// For each `(state, action)` pair present in `other`:
    /// ```text
    /// Q_self(s,a) = local_weight * Q_self(s,a) + (1 - local_weight) * Q_other(s,a)
    /// ```
    ///
    /// New pairs only in `other` are inserted directly.
    pub fn merge(&mut self, other: &QTable, local_weight: f64) {
        let w = local_weight.clamp(0.0, 1.0);
        for (&state, row) in &other.table {
            let local_row = self.table.entry(state).or_default();
            for (&action, &other_val) in row {
                let local_val = local_row.get(&action).copied().unwrap_or(0.0);
                local_row.insert(action, w * local_val + (1.0 - w) * other_val);
            }
        }
    }
}

/// Deterministic pseudo-random float in [0, 1) from a seed.
fn pseudo_random(seed: usize) -> f64 {
    let x = (seed.wrapping_mul(0x9e3779b9).wrapping_add(0x6c62272e)) as u64;
    let x = x ^ (x >> 30);
    let x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    let x = x ^ (x >> 27);
    let x = x.wrapping_mul(0x94d049bb133111eb);
    let x = x ^ (x >> 31);
    (x as f64) / (u64::MAX as f64)
}

/// FNV-1a 64-bit hash of a string slice.
fn fnv1a_hash(s: &str) -> u64 {
    const BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    s.bytes()
        .fold(BASIS, |hash, byte| (hash ^ byte as u64).wrapping_mul(PRIME))
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
