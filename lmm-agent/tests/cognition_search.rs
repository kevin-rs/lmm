// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unit tests for `cognition::search` - `SearchOracle`.

use lmm_agent::cognition::search::SearchOracle;

#[test]
fn new_limit_clamped_to_at_least_one() {
    let oracle = SearchOracle::new(0);
    assert_eq!(oracle.limit, 1);
}

#[test]
fn new_with_positive_limit() {
    let oracle = SearchOracle::new(5);
    assert_eq!(oracle.limit, 5);
}

#[test]
fn cache_starts_empty() {
    let oracle = SearchOracle::new(3);
    assert_eq!(oracle.cache_len(), 0);
}

#[test]
fn clear_cache_empties_cache() {
    let mut oracle = SearchOracle::new(3);
    oracle.clear_cache();
    assert_eq!(oracle.cache_len(), 0);
}

#[tokio::test]
async fn fetch_returns_string_without_panic() {
    let mut oracle = SearchOracle::new(3);
    let result = oracle.fetch("Rust programming language").await;
    let _ = result;
}

#[tokio::test]
async fn fetch_caches_result_on_first_call() {
    let mut oracle = SearchOracle::new(3);
    let _ = oracle.fetch("Rust ownership").await;
    assert_eq!(oracle.cache_len(), 1);
}

#[tokio::test]
async fn fetch_does_not_grow_cache_on_repeat_query() {
    let mut oracle = SearchOracle::new(3);
    let _ = oracle.fetch("Rust ownership").await;
    let _ = oracle.fetch("Rust ownership").await;
    assert_eq!(oracle.cache_len(), 1);
}

#[tokio::test]
async fn fetch_grows_cache_for_distinct_queries() {
    let mut oracle = SearchOracle::new(3);
    let _ = oracle.fetch("query one").await;
    let _ = oracle.fetch("query two").await;
    assert_eq!(oracle.cache_len(), 2);
}

#[tokio::test]
async fn fetch_second_call_returns_same_as_first() {
    let mut oracle = SearchOracle::new(3);
    let first = oracle.fetch("Rust borrow checker").await;
    let second = oracle.fetch("Rust borrow checker").await;
    assert_eq!(first, second, "cached result should be identical");
}
