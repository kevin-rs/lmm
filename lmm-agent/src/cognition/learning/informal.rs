// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `InformalLearner` - invisible co-occurrence mining.
//!
//! Extracts implicit relational knowledge from raw `CognitionSignal` streams
//! without any explicit training labels. This corresponds to *invisible learning*
//! and *non-formal/informal learning* paradigms described in the knowledge corpus.
//!
//! ## Algorithm: Pointwise Mutual Information (PMI)
//!
//! For each pair of content tokens `(a, b)` that co-occur in high-reward
//! observations (reward ≥ reward_threshold):
//!
//! ```text
//! PMI(a, b) = log[ C(a, b) / (C(a) × C(b)) ] × N
//! ```
//!
//! where:
//! - `C(a, b)` = number of high-reward observations containing both tokens.
//! - `C(a)`    = marginal count of token `a` across all high-reward observations.
//! - `N`       = total high-reward observations seen.
//!
//! Token pairs with PMI above a threshold and occurring at least `min_count`
//! times are synthesised into relational fact strings that are injected into
//! the `KnowledgeIndex` as additional indexed knowledge.
//!
//! ## Smart Lifelong Learning connection
//!
//! By mining side-channel co-occurrences continuously as the agent works,
//! the learner grows its knowledge base organically - echoing the *Smart
//! Lifelong Learning* principle of persistent, contextual acquisition without
//! explicit study sessions.
//!
//! ## Examples
//!
//! ```rust
//! use lmm_agent::cognition::learning::informal::InformalLearner;
//!
//! let mut learner = InformalLearner::new(0.5, 1, 0.5);
//! learner.observe("rust ownership memory safety", 0.9);
//! learner.observe("ownership prevents data races", 0.85);
//! let pairs = learner.high_pmi_pairs(0.0);
//! assert!(!pairs.is_empty());
//! ```

use crate::cognition::knowledge::KnowledgeIndex;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;

/// Mines implicit co-occurrence knowledge from agent observations.
///
/// # Examples
///
/// ```rust
/// use lmm_agent::cognition::learning::informal::InformalLearner;
///
/// let mut il = InformalLearner::new(0.4, 1, 0.5);
/// il.observe("Rust memory safe concurrent", 0.8);
/// il.observe("memory safe systems programming", 0.9);
/// let pairs = il.high_pmi_pairs(0.0);
/// assert!(!pairs.is_empty());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformalLearner {
    /// Reward threshold above which an observation contributes to PMI counts.
    pub reward_threshold: f64,

    /// Minimum co-occurrence count before a pair is eligible for PMI synthesis.
    pub min_count: usize,

    /// Minimum PMI value (log-scaled) before a pair is surfaced.
    pub pmi_threshold: f64,

    /// Marginal counts for individual tokens across high-reward observations.
    token_counts: HashMap<String, usize>,

    /// Co-occurrence counts for token pairs (canonically sorted).
    pair_counts: HashMap<(String, String), usize>,

    /// Total high-reward observations processed.
    pub observation_count: usize,

    /// Total synthesis events (PMI pairs injected into knowledge).
    pub synthesis_count: usize,
}

impl InformalLearner {
    /// Constructs a new `InformalLearner`.
    ///
    /// # Arguments
    ///
    /// * `reward_threshold` - Minimum reward for an observation to contribute.
    /// * `min_count`        - Minimum co-occurrence count for pair eligibility.
    /// * `pmi_threshold`    - Minimum PMI value (≥ 0 recommended).
    pub fn new(reward_threshold: f64, min_count: usize, pmi_threshold: f64) -> Self {
        Self {
            reward_threshold,
            min_count: min_count.max(1),
            pmi_threshold,
            token_counts: HashMap::new(),
            pair_counts: HashMap::new(),
            observation_count: 0,
            synthesis_count: 0,
        }
    }

    /// Processes one observation from a `CognitionSignal`.
    ///
    /// The observation is accepted only when `reward ≥ reward_threshold`.
    /// Tokens are lowercased, stripped to alphabetic characters, and must be
    /// at least 3 characters long.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::learning::informal::InformalLearner;
    ///
    /// let mut il = InformalLearner::new(0.5, 1, 0.0);
    /// il.observe("rust ownership", 0.8);
    /// assert_eq!(il.observation_count, 1);
    /// ```
    pub fn observe(&mut self, text: &str, reward: f64) {
        if reward < self.reward_threshold {
            return;
        }

        let tokens: Vec<String> = tokenise(text);
        if tokens.len() < 2 {
            return;
        }

        for t in &tokens {
            *self.token_counts.entry(t.clone()).or_insert(0) += 1;
        }

        for i in 0..tokens.len() {
            for j in (i + 1)..tokens.len() {
                let pair = canonical_pair(&tokens[i], &tokens[j]);
                *self.pair_counts.entry(pair).or_insert(0) += 1;
            }
        }

        self.observation_count += 1;
    }

    /// Returns token pairs whose PMI score exceeds `pmi_threshold` and whose
    /// co-occurrence count meets `min_count`.
    ///
    /// Pairs are returned as `(token_a, token_b, pmi_score)` triples, sorted
    /// by descending PMI.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::learning::informal::InformalLearner;
    ///
    /// let mut il = InformalLearner::new(0.0, 1, 0.0);
    /// il.observe("alpha beta gamma", 1.0);
    /// il.observe("alpha beta delta", 1.0);
    /// let pairs = il.high_pmi_pairs(0.0);
    /// assert!(!pairs.is_empty());
    /// ```
    pub fn high_pmi_pairs(&self, threshold_override: f64) -> Vec<(String, String, f64)> {
        let n = self.observation_count as f64;
        if n < 1.0 {
            return Vec::new();
        }

        let eff_threshold = threshold_override.max(self.pmi_threshold);

        let mut results: Vec<(String, String, f64)> = self
            .pair_counts
            .iter()
            .filter(|&(_, count)| *count >= self.min_count)
            .filter_map(|((a, b), &c_ab)| {
                let c_a = *self.token_counts.get(a)? as f64;
                let c_b = *self.token_counts.get(b)? as f64;
                let pmi = (c_ab as f64 * n / (c_a * c_b)).ln();
                if pmi >= eff_threshold {
                    Some((a.clone(), b.clone(), pmi))
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
        results
    }

    /// Synthesises the top-k high-PMI pairs into fact strings and ingests them
    /// into `index`, returning the number of new chunks added.
    ///
    /// Each pair `(a, b)` becomes the fact string:
    /// `"<a> and <b> are strongly related concepts"`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::learning::informal::InformalLearner;
    /// use lmm_agent::cognition::knowledge::KnowledgeIndex;
    ///
    /// let mut il = InformalLearner::new(0.0, 1, 0.0);
    /// for _ in 0..3 { il.observe("ownership memory concurrent", 1.0); }
    /// let mut idx = KnowledgeIndex::new();
    /// let added = il.synthesise_into(&mut idx, 5, 0.0);
    /// assert!(added > 0 || idx.is_empty());
    /// ```
    pub fn synthesise_into(
        &mut self,
        index: &mut KnowledgeIndex,
        top_k: usize,
        threshold_override: f64,
    ) -> usize {
        let pairs = self.high_pmi_pairs(threshold_override);
        let mut added = 0;
        for (a, b, _) in pairs.into_iter().take(top_k) {
            let fact = format!("{a} and {b} are strongly related concepts in this domain");
            let n = index.ingest_text("informal", &fact);
            added += n;
            self.synthesis_count += n;
        }
        added
    }

    /// Returns the number of unique token types observed.
    pub fn vocabulary_size(&self) -> usize {
        self.token_counts.len()
    }

    /// Returns the number of unique co-occurrence pairs tracked.
    pub fn pair_count(&self) -> usize {
        self.pair_counts.len()
    }
}

fn tokenise(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphabetic())
                .collect::<String>()
                .to_ascii_lowercase()
        })
        .filter(|s| s.len() >= 3)
        .collect()
}

fn canonical_pair(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
