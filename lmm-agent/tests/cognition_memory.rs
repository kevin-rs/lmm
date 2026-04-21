// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unit tests for `cognition::memory` - `MemoryEntry`, `HotStore`, `ColdStore`.

use lmm_agent::cognition::memory::{ColdStore, HotStore, MemoryEntry};

#[test]
fn memory_entry_stores_fields_correctly() {
    let e = MemoryEntry::new("test content".into(), 0.75, 5);
    assert_eq!(e.content, "test content");
    assert_eq!(e.score, 0.75);
    assert_eq!(e.timestamp, 5);
}

#[test]
fn hot_store_new_is_empty() {
    let s = HotStore::new(5);
    assert!(s.is_empty());
    assert_eq!(s.len(), 0);
}

#[test]
fn hot_store_push_within_capacity() {
    let mut s = HotStore::new(5);
    for i in 0..5usize {
        s.push(MemoryEntry::new(i.to_string(), 0.5, i));
    }
    assert_eq!(s.len(), 5);
}

#[test]
fn hot_store_evicts_oldest_when_full() {
    let mut s = HotStore::new(2);
    s.push(MemoryEntry::new("first".into(), 0.1, 0));
    s.push(MemoryEntry::new("second".into(), 0.2, 1));
    s.push(MemoryEntry::new("third".into(), 0.3, 2));
    assert_eq!(s.len(), 2);
    let contents: Vec<&str> = s.entries().iter().map(|e| e.content.as_str()).collect();
    assert!(contents.contains(&"second"));
    assert!(contents.contains(&"third"));
    assert!(!contents.contains(&"first"));
}

#[test]
fn hot_store_capacity_of_one_always_has_latest() {
    let mut s = HotStore::new(1);
    s.push(MemoryEntry::new("a".into(), 0.5, 0));
    s.push(MemoryEntry::new("b".into(), 0.6, 1));
    assert_eq!(s.len(), 1);
    assert_eq!(s.entries()[0].content, "b");
}

#[test]
fn hot_store_relevant_returns_best_overlap() {
    let mut s = HotStore::new(10);
    s.push(MemoryEntry::new("Rust ownership model".into(), 0.8, 0));
    s.push(MemoryEntry::new("Python garbage collector".into(), 0.6, 1));
    let top = s.relevant("Rust memory", 1);
    assert_eq!(top[0].content, "Rust ownership model");
}

#[test]
fn hot_store_relevant_returns_at_most_top_n() {
    let mut s = HotStore::new(10);
    for i in 0..8usize {
        s.push(MemoryEntry::new(format!("item {i}"), 0.5, i));
    }
    let top = s.relevant("item", 3);
    assert_eq!(top.len(), 3);
}

#[test]
fn hot_store_drain_promotes_high_score() {
    let mut hot = HotStore::new(5);
    hot.push(MemoryEntry::new("high".into(), 0.9, 0));
    hot.push(MemoryEntry::new("low".into(), 0.1, 1));
    let mut cold = ColdStore::default();
    hot.drain_to_cold(&mut cold, 0.5);
    assert_eq!(cold.len(), 1);
    assert_eq!(cold.all()[0].content, "high");
    assert_eq!(hot.len(), 1);
    assert_eq!(hot.entries()[0].content, "low");
}

#[test]
fn hot_store_drain_all_above_threshold() {
    let mut hot = HotStore::new(3);
    hot.push(MemoryEntry::new("a".into(), 0.8, 0));
    hot.push(MemoryEntry::new("b".into(), 0.9, 1));
    let mut cold = ColdStore::default();
    hot.drain_to_cold(&mut cold, 0.5);
    assert_eq!(cold.len(), 2);
    assert!(hot.is_empty());
}

#[test]
fn hot_store_drain_none_promoted() {
    let mut hot = HotStore::new(3);
    hot.push(MemoryEntry::new("low".into(), 0.1, 0));
    let mut cold = ColdStore::default();
    hot.drain_to_cold(&mut cold, 0.5);
    assert_eq!(cold.len(), 0);
    assert_eq!(hot.len(), 1);
}

#[test]
fn hot_store_drain_empty_no_panic() {
    let mut hot = HotStore::new(3);
    let mut cold = ColdStore::default();
    hot.drain_to_cold(&mut cold, 0.5);
    assert!(cold.is_empty());
}

#[test]
fn hot_store_clear_empties_all() {
    let mut s = HotStore::new(5);
    s.push(MemoryEntry::new("a".into(), 0.5, 0));
    s.clear();
    assert!(s.is_empty());
}

#[test]
fn hot_store_snapshot_newest_first() {
    let mut s = HotStore::new(3);
    s.push(MemoryEntry::new("first".into(), 0.5, 0));
    s.push(MemoryEntry::new("second".into(), 0.6, 1));
    let snap = s.snapshot();
    assert_eq!(snap[0], "second");
    assert_eq!(snap[1], "first");
}

#[test]
fn cold_store_default_is_empty() {
    let c = ColdStore::default();
    assert!(c.is_empty());
    assert_eq!(c.len(), 0);
}

#[test]
fn cold_store_promote_grows_len() {
    let mut c = ColdStore::default();
    c.promote(MemoryEntry::new("fact".into(), 0.7, 0));
    assert_eq!(c.len(), 1);
}

#[test]
fn cold_store_recall_returns_most_relevant() {
    let mut c = ColdStore::default();
    c.promote(MemoryEntry::new("old unrelated text".into(), 0.5, 0));
    c.promote(MemoryEntry::new(
        "Rust ownership facts recent".into(),
        0.8,
        5,
    ));
    let top = c.recall("Rust", 1);
    assert_eq!(top[0].content, "Rust ownership facts recent");
}

#[test]
fn cold_store_recall_on_empty_returns_empty() {
    let c = ColdStore::default();
    let top = c.recall("query", 5);
    assert!(top.is_empty());
}

#[test]
fn cold_store_recall_top_n_capped() {
    let mut c = ColdStore::default();
    for i in 0..10usize {
        c.promote(MemoryEntry::new(format!("fact {i}"), 0.5, i));
    }
    let top = c.recall("fact", 3);
    assert_eq!(top.len(), 3);
}

#[test]
fn cold_store_all_returns_insertion_order() {
    let mut c = ColdStore::default();
    c.promote(MemoryEntry::new("first".into(), 0.5, 0));
    c.promote(MemoryEntry::new("second".into(), 0.6, 1));
    assert_eq!(c.all()[0].content, "first");
    assert_eq!(c.all()[1].content, "second");
}

#[test]
fn cold_store_snapshot_newest_first() {
    let mut c = ColdStore::default();
    c.promote(MemoryEntry::new("first".into(), 0.5, 0));
    c.promote(MemoryEntry::new("second".into(), 0.6, 1));
    let snap = c.snapshot();
    assert_eq!(snap[0], "second");
    assert_eq!(snap[1], "first");
}

#[test]
fn cold_store_entries_never_deleted() {
    let mut c = ColdStore::default();
    for i in 0..100usize {
        c.promote(MemoryEntry::new(format!("fact {i}"), 0.5, i));
    }
    assert_eq!(c.len(), 100);
}
