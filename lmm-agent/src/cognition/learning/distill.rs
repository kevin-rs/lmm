// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `KnowledgeDistiller` - importance-weighted compression of learned facts.
//!
//! Distillation converts high-reward `ColdStore` entries into indexed chunks
//! inside the agent's `KnowledgeIndex`. This is the HELM analogue of teacher →
//! student knowledge distillation: the *teacher* is the raw experience archive
//! (cold store) and the *student* is the queryable knowledge index.
//!
//! ## Importance scoring
//!
//! Each `ColdStore` entry receives an importance score:
//!
//! ```text
//! importance(e) = e.score × Σ_{t ∈ tokens(e)} idf(t)
//! ```
//!
//! where `idf(t) = ln(N / df(t)) + 1` and `N` is the total number of candidate
//! entries. Entries with `importance ≥ threshold` and that are not already
//! present in the index are promoted to the `KnowledgeIndex` via `ingest_text`.
//!
//! ## Examples
//!
//! ```rust
//! use lmm_agent::cognition::learning::distill::KnowledgeDistiller;
//! use lmm_agent::cognition::memory::{ColdStore, MemoryEntry};
//! use lmm_agent::cognition::knowledge::KnowledgeIndex;
//!
//! let mut cold = ColdStore::default();
//! cold.promote(MemoryEntry::new("Rust prevents data races at compile time.".into(), 0.9, 0));
//! cold.promote(MemoryEntry::new("x".into(), 0.1, 1));
//!
//! let mut index = KnowledgeIndex::new();
//! let mut distiller = KnowledgeDistiller::new(0.3, 8);
//! let added = distiller.distill(&cold, &mut index);
//! assert!(added > 0);
//! ```

use crate::cognition::knowledge::KnowledgeIndex;
use crate::cognition::memory::ColdStore;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

/// Compresses high-importance `ColdStore` entries into the `KnowledgeIndex`.
///
/// # Examples
///
/// ```rust
/// use lmm_agent::cognition::learning::distill::KnowledgeDistiller;
/// use lmm_agent::cognition::memory::{ColdStore, MemoryEntry};
/// use lmm_agent::cognition::knowledge::KnowledgeIndex;
///
/// let mut cold = ColdStore::default();
/// cold.promote(MemoryEntry::new(
///     "ownership and borrowing make Rust memory safe".into(), 0.85, 0
/// ));
/// let mut idx = KnowledgeIndex::new();
/// let mut d = KnowledgeDistiller::new(0.2, 5);
/// assert!(d.distill(&cold, &mut idx) > 0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeDistiller {
    /// Minimum importance score for distillation eligibility.
    pub threshold: f64,

    /// Maximum cold entries promoted per call.
    pub top_k: usize,

    /// Fingerprints of content already ingested, to avoid duplicates.
    ingested: HashSet<u64>,

    /// Total entries ingested across all calls.
    pub total_ingested: usize,
}

impl KnowledgeDistiller {
    /// Creates a new `KnowledgeDistiller`.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum importance score ∈ [0, ∞).
    /// * `top_k`     - Maximum entries promoted per `distill` call (≥ 1).
    pub fn new(threshold: f64, top_k: usize) -> Self {
        Self {
            threshold: threshold.max(0.0),
            top_k: top_k.max(1),
            ingested: HashSet::new(),
            total_ingested: 0,
        }
    }

    /// Promotes up to `top_k` high-importance entries from `cold` into `index`.
    ///
    /// Returns the number of new chunks added to the index.
    ///
    /// Entries with fewer than 4 tokens or already ingested are silently skipped.
    pub fn distill(&mut self, cold: &ColdStore, index: &mut KnowledgeIndex) -> usize {
        let entries = cold.all();
        if entries.is_empty() {
            return 0;
        }

        let df = compute_doc_freq(entries.iter().map(|e| e.content.as_str()));
        let n = entries.len() as f64;

        let mut scored: Vec<(f64, &str)> = entries
            .iter()
            .filter(|e| e.content.split_whitespace().count() >= 4)
            .map(|e| {
                let idf_sum: f64 = tokenise(&e.content)
                    .iter()
                    .map(|t| {
                        let df_t = *df.get(t).unwrap_or(&1) as f64;
                        (n / df_t).ln() + 1.0
                    })
                    .sum::<f64>();
                let importance = e.score * idf_sum;
                (importance, e.content.as_str())
            })
            .filter(|(imp, _)| *imp >= self.threshold)
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        let mut added = 0;
        for (_, text) in scored.into_iter().take(self.top_k) {
            let fp = fnv1a(text);
            if self.ingested.insert(fp) {
                let chunks = index.ingest_text("distilled", text);
                added += chunks;
                self.total_ingested += chunks;
            }
        }
        added
    }
}

fn tokenise(text: &str) -> Vec<String> {
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

fn compute_doc_freq<'a>(texts: impl Iterator<Item = &'a str>) -> HashMap<String, usize> {
    let mut df: HashMap<String, usize> = HashMap::new();
    for text in texts {
        let mut seen: HashSet<String> = HashSet::new();
        for token in tokenise(text) {
            if seen.insert(token.clone()) {
                *df.entry(token).or_insert(0) += 1;
            }
        }
    }
    df
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
