// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `Reflector` - reward-gated introspection.
//!
//! The `Reflector` serves two roles in the ThinkLoop:
//!
//! 1. **Query formulation** - generates the next DuckDuckGo search query by
//!    fusing the goal with the most relevant hot-store context.
//! 2. **Memory consolidation** - promotes high-reward hot entries to cold at
//!    the end of each loop run.
//!
//! ## Examples
//!
//! ```rust
//! use lmm_agent::cognition::reflect::Reflector;
//! use lmm_agent::cognition::memory::{HotStore, MemoryEntry};
//!
//! let mut hot = HotStore::new(10);
//! hot.push(MemoryEntry::new("Rust ownership semantics".into(), 0.9, 0));
//! let query = Reflector::formulate_query("How does Rust manage memory?", &hot);
//! assert!(!query.is_empty());
//! ```
//!
//! ## See Also
//!
//! * [Metacognition - Wikipedia](https://en.wikipedia.org/wiki/Metacognition)
//! * [Self-reflection - Wikipedia](https://en.wikipedia.org/wiki/Self-reflection)

use crate::cognition::memory::{ColdStore, HotStore};

/// Stateless utility that formulates queries and consolidates memory.
pub struct Reflector;

impl Reflector {
    /// Formulates the next DuckDuckGo search query from goal + hot memory context.
    ///
    /// Takes the top-3 most relevant hot entries, extracts their first 8 words,
    /// appends to goal, deduplicates, and caps at 12 words total.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::reflect::Reflector;
    /// use lmm_agent::cognition::memory::{HotStore, MemoryEntry};
    ///
    /// let mut hot = HotStore::new(5);
    /// hot.push(MemoryEntry::new("ownership model borrow checker".into(), 0.8, 0));
    /// let q = Reflector::formulate_query("Rust memory safety", &hot);
    /// assert!(q.contains("Rust"));
    /// ```
    pub fn formulate_query(goal: &str, hot: &HotStore) -> String {
        let mut words: Vec<&str> = goal.split_whitespace().collect();

        for entry in hot.relevant(goal, 3) {
            words.extend(entry.content.split_whitespace().take(8));
        }

        let mut seen = std::collections::HashSet::new();
        let unique: Vec<&str> = words
            .into_iter()
            .filter(|&w| seen.insert(w.to_ascii_lowercase()))
            .take(12)
            .collect();

        if unique.is_empty() {
            goal.to_string()
        } else {
            unique.join(" ")
        }
    }

    /// Promotes entries scoring at or above `threshold` from `hot` to `cold`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::reflect::Reflector;
    /// use lmm_agent::cognition::memory::{HotStore, ColdStore, MemoryEntry};
    ///
    /// let mut hot = HotStore::new(10);
    /// hot.push(MemoryEntry::new("val".into(), 0.9, 0));
    /// hot.push(MemoryEntry::new("low".into(), 0.1, 1));
    /// let mut cold = ColdStore::default();
    /// Reflector::drain_to_cold(&mut hot, &mut cold, 0.5);
    /// assert_eq!(cold.len(), 1);
    /// assert_eq!(hot.len(), 1);
    /// ```
    pub fn drain_to_cold(hot: &mut HotStore, cold: &mut ColdStore, threshold: f64) {
        hot.drain_to_cold(cold, threshold);
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
