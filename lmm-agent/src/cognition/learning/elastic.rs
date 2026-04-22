// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `ElasticMemoryGuard` - importance-based forgetting prevention.
//!
//! Inspired by *Elastic Weight Consolidation* (EWC) from continual learning,
//! this guard computes a **Fisher-analog importance score** for each `ColdStore`
//! entry by tracking how frequently the entry is activated during `recall`
//! queries. High-activation entries are **pinned** so they are excluded from
//! future eviction passes, protecting the most relevant long-term knowledge.
//!
//! ## Importance model
//!
//! Because the system is purely symbolic (no neural weights), the Fisher
//! information matrix is approximated by a per-entry **activation counter**.
//! The protective penalty in the original EWC loss:
//!
//! ```text
//! L_elastic = λ · Σ_j F_j · (θ_j - θ*_j)²
//! ```
//!
//! becomes the guard criterion: entry _j_ with `activation_count ≥ pin_threshold`
//! is never eligible for eviction regardless of its raw `ColdStore` score.
//!
//! ## Examples
//!
//! ```rust
//! use lmm_agent::cognition::learning::elastic::ElasticMemoryGuard;
//!
//! let mut guard = ElasticMemoryGuard::new(3, 0.5);
//! let content = "rust ownership borrow checker".to_string();
//! guard.observe_activation(&content);
//! guard.observe_activation(&content);
//! guard.observe_activation(&content);
//! assert!(guard.is_pinned(&content));
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Tracks activation frequency of cold-store entries and pins the most important.
///
/// # Examples
///
/// ```rust
/// use lmm_agent::cognition::learning::elastic::ElasticMemoryGuard;
///
/// let mut g = ElasticMemoryGuard::new(2, 0.5);
/// g.observe_activation("key fact about ownership");
/// g.observe_activation("key fact about ownership");
/// assert!(g.is_pinned("key fact about ownership"));
/// assert!(!g.is_pinned("rarely recalled fact"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElasticMemoryGuard {
    /// Map from content fingerprint (FNV-1a) to activation count.
    activation_counts: HashMap<u64, usize>,

    /// Map from content fingerprint to exact content string (for display).
    content_map: HashMap<u64, String>,

    /// Activation count threshold above which an entry is considered pinned.
    pub pin_threshold: usize,

    /// Elastic penalty strength λ (stored for downstream use).
    pub lambda: f64,

    /// Total activation events recorded.
    pub total_activations: usize,
}

impl ElasticMemoryGuard {
    /// Constructs a new `ElasticMemoryGuard`.
    ///
    /// # Arguments
    ///
    /// * `pin_threshold` - Activation count before entry is pinned (≥ 1).
    /// * `lambda`        - Elastic penalty strength ∈ [0, ∞).
    pub fn new(pin_threshold: usize, lambda: f64) -> Self {
        Self {
            activation_counts: HashMap::new(),
            content_map: HashMap::new(),
            pin_threshold: pin_threshold.max(1),
            lambda: lambda.max(0.0),
            total_activations: 0,
        }
    }

    /// Records one activation event for the given cold-store entry content.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::learning::elastic::ElasticMemoryGuard;
    ///
    /// let mut g = ElasticMemoryGuard::new(5, 1.0);
    /// g.observe_activation("Rust borrow checker enforces safety");
    /// assert_eq!(g.activation_count("Rust borrow checker enforces safety"), 1);
    /// ```
    pub fn observe_activation(&mut self, content: &str) {
        let fp = fnv1a(content);
        *self.activation_counts.entry(fp).or_insert(0) += 1;
        self.content_map
            .entry(fp)
            .or_insert_with(|| content.to_string());
        self.total_activations += 1;
    }

    /// Returns the activation count for the given content string.
    pub fn activation_count(&self, content: &str) -> usize {
        *self.activation_counts.get(&fnv1a(content)).unwrap_or(&0)
    }

    /// Returns `true` when the entry has been activated at or above `pin_threshold`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::learning::elastic::ElasticMemoryGuard;
    ///
    /// let mut g = ElasticMemoryGuard::new(2, 0.5);
    /// g.observe_activation("data");
    /// assert!(!g.is_pinned("data"));
    /// g.observe_activation("data");
    /// assert!(g.is_pinned("data"));
    /// ```
    pub fn is_pinned(&self, content: &str) -> bool {
        self.activation_count(content) >= self.pin_threshold
    }

    /// Returns the normalised importance score ∈ [0, 1] for `content`.
    ///
    /// Computed as `activation_count / max_activation` across all tracked entries.
    /// Returns `0.0` when no activations have been recorded.
    pub fn importance(&self, content: &str) -> f64 {
        let count = self.activation_count(content);
        if count == 0 {
            return 0.0;
        }
        let max = self.activation_counts.values().copied().max().unwrap_or(1) as f64;
        count as f64 / max
    }

    /// Returns all pinned content strings.
    pub fn pinned_contents(&self) -> Vec<&str> {
        self.activation_counts
            .iter()
            .filter(|&(_, count)| *count >= self.pin_threshold)
            .filter_map(|(fp, _)| self.content_map.get(fp).map(|s| s.as_str()))
            .collect()
    }

    /// Returns the total number of unique entries tracked.
    pub fn unique_entry_count(&self) -> usize {
        self.activation_counts.len()
    }
}

fn fnv1a(s: &str) -> u64 {
    const BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    s.bytes()
        .fold(BASIS, |h, b| (h ^ b as u64).wrapping_mul(PRIME))
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
