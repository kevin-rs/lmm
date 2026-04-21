// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Two-tier agent memory: `HotStore` and `ColdStore`.
//!
//! Inspired by the **integrator** and **capacitor** components of feedback control
//! circuits, the agent keeps two distinct memory tiers:
//!
//! | Tier        | Analogy           | Behaviour                                    |
//! |-------------|-------------------|----------------------------------------------|
//! | `HotStore`  | Working register  | Bounded FIFO; oldest entry evicted when full |
//! | `ColdStore` | Capacitor / LTM   | Unbounded archive; entries never deleted     |
//!
//! High-reward entries are *promoted* from hot to cold via `drain_to_cold`, which
//! is called by the `Reflector` at the end of each `ThinkLoop` run.
//!
//! ## Examples
//!
//! ```rust
//! use lmm_agent::cognition::memory::{HotStore, ColdStore, MemoryEntry};
//!
//! let mut hot = HotStore::new(3);
//! hot.push(MemoryEntry::new("first observation".into(), 0.8, 0));
//! hot.push(MemoryEntry::new("second observation".into(), 0.5, 1));
//! assert_eq!(hot.len(), 2);
//!
//! let mut cold = ColdStore::default();
//! hot.drain_to_cold(&mut cold, 0.7);
//! assert_eq!(cold.len(), 1); // only score ≥ 0.7 promoted
//! ```
//!
//! ## See Also
//!
//! * [Cognitive architecture - Wikipedia](https://en.wikipedia.org/wiki/Cognitive_architecture)
//! * [Memory - Wikipedia](https://en.wikipedia.org/wiki/Memory_(psychology))

use std::collections::VecDeque;

/// A single entry in either tier of agent memory.
///
/// # Examples
///
/// ```rust
/// use lmm_agent::cognition::memory::MemoryEntry;
///
/// let entry = MemoryEntry::new("Rust owns memory safely.".into(), 0.92, 3);
/// assert_eq!(entry.content, "Rust owns memory safely.");
/// assert_eq!(entry.score, 0.92);
/// assert_eq!(entry.timestamp, 3);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryEntry {
    /// The raw text content of this memory (DDG observation, generated text, etc.).
    pub content: String,

    /// Relevance / reward score assigned when this entry was created, ∈ [0, ∞).
    pub score: f64,

    /// The loop step at which this entry was recorded.
    pub timestamp: usize,
}

impl MemoryEntry {
    /// Constructs a new [`MemoryEntry`].
    pub fn new(content: String, score: f64, timestamp: usize) -> Self {
        Self {
            content,
            score,
            timestamp,
        }
    }
}

/// Bounded FIFO short-term memory.
///
/// When the store is full and a new entry is pushed, the **oldest** entry is
/// silently evicted. This mirrors the limited capacity of a control register.
///
/// # Examples
///
/// ```rust
/// use lmm_agent::cognition::memory::{HotStore, MemoryEntry};
///
/// let mut store = HotStore::new(2);
/// store.push(MemoryEntry::new("a".into(), 0.5, 0));
/// store.push(MemoryEntry::new("b".into(), 0.9, 1));
/// store.push(MemoryEntry::new("c".into(), 0.7, 2)); // evicts "a"
/// assert_eq!(store.len(), 2);
/// assert_eq!(store.entries()[0].content, "b");
/// ```
#[derive(Debug, Clone)]
pub struct HotStore {
    /// Maximum number of entries that can be held simultaneously.
    pub capacity: usize,
    entries: VecDeque<MemoryEntry>,
}

impl HotStore {
    /// Creates a new `HotStore` with the given maximum capacity.
    ///
    /// # Panics
    ///
    /// Panics if `capacity == 0`.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "HotStore capacity must be > 0");
        Self {
            capacity,
            entries: VecDeque::with_capacity(capacity),
        }
    }

    /// Appends a new entry, evicting the oldest when at capacity.
    pub fn push(&mut self, entry: MemoryEntry) {
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    /// Returns the number of entries currently held.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the store contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns an ordered slice of all current entries (oldest → newest).
    pub fn entries(&self) -> &VecDeque<MemoryEntry> {
        &self.entries
    }

    /// Returns the top-`n` entries most relevant to `query` using token-overlap scoring.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::memory::{HotStore, MemoryEntry};
    ///
    /// let mut store = HotStore::new(10);
    /// store.push(MemoryEntry::new("Rust ownership model".into(), 0.8, 0));
    /// store.push(MemoryEntry::new("Python garbage collector".into(), 0.6, 1));
    /// let top = store.relevant("Rust memory", 1);
    /// assert_eq!(top[0].content, "Rust ownership model");
    /// ```
    pub fn relevant(&self, query: &str, top_n: usize) -> Vec<&MemoryEntry> {
        let query_tokens: std::collections::HashSet<String> = query
            .split_whitespace()
            .map(|w| w.to_ascii_lowercase())
            .collect();

        let mut scored: Vec<(&MemoryEntry, usize)> = self
            .entries
            .iter()
            .map(|e| {
                let entry_tokens: std::collections::HashSet<String> = e
                    .content
                    .split_whitespace()
                    .map(|w| w.to_ascii_lowercase())
                    .collect();
                let overlap = query_tokens.intersection(&entry_tokens).count();
                (e, overlap)
            })
            .collect();

        scored.sort_by_key(|b| std::cmp::Reverse(b.1));
        scored.into_iter().take(top_n).map(|(e, _)| e).collect()
    }

    /// Moves entries whose score meets or exceeds `threshold` into `cold`.
    ///
    /// Promoted entries are **removed** from the hot store; entries below
    /// threshold are retained.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::memory::{HotStore, ColdStore, MemoryEntry};
    ///
    /// let mut hot = HotStore::new(5);
    /// hot.push(MemoryEntry::new("high value".into(), 0.9, 0));
    /// hot.push(MemoryEntry::new("low value".into(), 0.2, 1));
    /// let mut cold = ColdStore::default();
    /// hot.drain_to_cold(&mut cold, 0.7);
    /// assert_eq!(cold.len(), 1);
    /// assert_eq!(hot.len(), 1);
    /// ```
    pub fn drain_to_cold(&mut self, cold: &mut ColdStore, threshold: f64) {
        let mut retain = VecDeque::new();
        while let Some(entry) = self.entries.pop_front() {
            if entry.score >= threshold {
                cold.promote(entry);
            } else {
                retain.push_back(entry);
            }
        }
        self.entries = retain;
    }

    /// Clears all entries from the hot store.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Returns a snapshot of all content strings (newest-first).
    pub fn snapshot(&self) -> Vec<String> {
        self.entries
            .iter()
            .rev()
            .map(|e| e.content.clone())
            .collect()
    }
}

/// Unbounded long-term memory archive. Entries are **never** deleted.
///
/// `recall` returns top-N entries by a blended score of reward × recency.
///
/// # Examples
///
/// ```rust
/// use lmm_agent::cognition::memory::{ColdStore, MemoryEntry};
///
/// let mut cold = ColdStore::default();
/// cold.promote(MemoryEntry::new("fact about Rust".into(), 0.9, 0));
/// assert_eq!(cold.len(), 1);
/// let recalled = cold.recall("Rust", 1);
/// assert_eq!(recalled[0].content, "fact about Rust");
/// ```
#[derive(Debug, Clone, Default)]
pub struct ColdStore {
    entries: Vec<MemoryEntry>,
}

impl ColdStore {
    /// Appends an entry to the archive.
    pub fn promote(&mut self, entry: MemoryEntry) {
        self.entries.push(entry);
    }

    /// Returns the total number of entries in the archive.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when no entries have been archived.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns all archived entries (insertion order).
    pub fn all(&self) -> &[MemoryEntry] {
        &self.entries
    }

    /// Returns the top-`n` entries most relevant to `query`, blending
    /// token-overlap with a recency factor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::memory::{ColdStore, MemoryEntry};
    ///
    /// let mut cold = ColdStore::default();
    /// cold.promote(MemoryEntry::new("old fact".into(), 0.5, 0));
    /// cold.promote(MemoryEntry::new("Rust ownership facts recent".into(), 0.8, 5));
    /// let top = cold.recall("Rust", 1);
    /// assert_eq!(top[0].content, "Rust ownership facts recent");
    /// ```
    pub fn recall(&self, query: &str, top_n: usize) -> Vec<&MemoryEntry> {
        let query_tokens: std::collections::HashSet<String> = query
            .split_whitespace()
            .map(|w| w.to_ascii_lowercase())
            .collect();

        let total = self.entries.len();
        let mut scored: Vec<(&MemoryEntry, f64)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let entry_tokens: std::collections::HashSet<String> = e
                    .content
                    .split_whitespace()
                    .map(|w| w.to_ascii_lowercase())
                    .collect();
                let overlap = query_tokens.intersection(&entry_tokens).count() as f64;
                let recency = (i + 1) as f64 / total as f64;
                let blended = (e.score + overlap * 0.1) * (0.7 + 0.3 * recency);
                (e, blended)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(top_n).map(|(e, _)| e).collect()
    }

    /// Returns a snapshot of all content strings (newest-first).
    pub fn snapshot(&self) -> Vec<String> {
        self.entries
            .iter()
            .rev()
            .map(|e| e.content.clone())
            .collect()
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
