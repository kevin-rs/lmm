// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unit tests for `cognition::reflect` - `Reflector`.

use lmm_agent::cognition::memory::{ColdStore, HotStore, MemoryEntry};
use lmm_agent::cognition::reflect::Reflector;

#[test]
fn formulate_query_empty_hot_returns_goal() {
    let hot = HotStore::new(5);
    let q = Reflector::formulate_query("What is Rust?", &hot);
    assert_eq!(q, "What is Rust?");
}

#[test]
fn formulate_query_incorporates_relevant_context() {
    let mut hot = HotStore::new(5);
    hot.push(MemoryEntry::new("ownership semantics".into(), 0.8, 0));
    let q = Reflector::formulate_query("Rust memory", &hot);
    assert!(q.contains("Rust") || q.contains("memory"), "query was: {q}");
}

#[test]
fn formulate_query_max_twelve_words() {
    let mut hot = HotStore::new(5);
    hot.push(MemoryEntry::new(
        "a b c d e f g h i j k l m n o p".into(),
        0.9,
        0,
    ));
    let q = Reflector::formulate_query("goal x y z", &hot);
    let word_count = q.split_whitespace().count();
    assert!(
        word_count <= 12,
        "query has {word_count} words, expected ≤ 12"
    );
}

#[test]
fn formulate_query_deduplicates_words() {
    let mut hot = HotStore::new(5);
    hot.push(MemoryEntry::new("Rust Rust Rust".into(), 0.8, 0));
    let q = Reflector::formulate_query("Rust language", &hot);
    let rust_count = q
        .split_whitespace()
        .filter(|w| w.eq_ignore_ascii_case("rust"))
        .count();
    assert_eq!(rust_count, 1, "'Rust' should appear once: {q}");
}

#[test]
fn drain_to_cold_promotes_high_score() {
    let mut hot = HotStore::new(5);
    hot.push(MemoryEntry::new("high".into(), 0.9, 0));
    hot.push(MemoryEntry::new("low".into(), 0.2, 1));
    let mut cold = ColdStore::default();
    Reflector::drain_to_cold(&mut hot, &mut cold, 0.5);
    assert_eq!(cold.len(), 1);
    assert_eq!(hot.len(), 1);
}

#[test]
fn drain_to_cold_empty_hot_is_safe() {
    let mut hot = HotStore::new(5);
    let mut cold = ColdStore::default();
    Reflector::drain_to_cold(&mut hot, &mut cold, 0.5);
    assert!(cold.is_empty());
}

#[test]
fn drain_to_cold_all_promoted_when_threshold_zero() {
    let mut hot = HotStore::new(5);
    hot.push(MemoryEntry::new("a".into(), 0.0, 0));
    hot.push(MemoryEntry::new("b".into(), 0.0, 1));
    let mut cold = ColdStore::default();
    Reflector::drain_to_cold(&mut hot, &mut cold, 0.0);
    assert_eq!(cold.len(), 2);
    assert!(hot.is_empty());
}
