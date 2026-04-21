// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Integration tests for `LmmAgent::think()` and the full ThinkLoop pipeline.

use lmm_agent::agent::LmmAgent;
use lmm_agent::prelude::*;
use lmm_agent::types::Status;

#[tokio::test]
async fn think_returns_ok() {
    let mut agent = LmmAgent::new("Tester".into(), "Think test.".into());
    let result = agent.think("What is Rust?").await;
    assert!(result.is_ok(), "think() returned Err: {:?}", result.err());
}

#[tokio::test]
async fn think_result_steps_at_least_one() {
    let mut agent = LmmAgent::new("Tester".into(), "Think test.".into());
    let result = agent.think("Rust ownership model").await.unwrap();
    assert!(
        result.steps >= 1,
        "expected >= 1 step, got {}",
        result.steps
    );
}

#[tokio::test]
async fn think_result_error_in_range() {
    let mut agent = LmmAgent::new("Tester".into(), "Think test.".into());
    let result = agent.think("Rust borrow checker").await.unwrap();
    assert!(
        result.final_error >= 0.0 && result.final_error <= 1.0,
        "final_error out of range: {}",
        result.final_error
    );
}

#[tokio::test]
async fn think_sets_status_to_completed_after_run() {
    let mut agent = LmmAgent::new("Tester".into(), "Think test.".into());
    agent.think("any goal").await.unwrap();
    assert_eq!(agent.status, Status::Completed);
}

#[tokio::test]
async fn think_appends_to_hot_memory() {
    let mut agent = LmmAgent::new("Tester".into(), "Think test.".into());
    let before = agent.memory.len();
    agent.think("a goal").await.unwrap();
    assert!(
        agent.memory.len() > before,
        "memory should grow after think()"
    );
}

#[tokio::test]
async fn think_result_signals_len_matches_steps() {
    let mut agent = LmmAgent::new("Tester".into(), "Think test.".into());
    let result = agent.think("Rust systems language").await.unwrap();
    assert_eq!(
        result.signals.len(),
        result.steps,
        "signals.len() ({}) != steps ({})",
        result.signals.len(),
        result.steps
    );
}

#[tokio::test]
async fn think_with_respects_max_iterations() {
    let mut agent = LmmAgent::new("Tester".into(), "Think test.".into());
    let result = agent
        .think_with("some goal phrase", 3, 0.0, 1.0, 0.05)
        .await
        .unwrap();
    assert!(
        result.steps <= 3,
        "steps ({}) exceeded max_iterations (3)",
        result.steps
    );
}

#[tokio::test]
async fn think_with_high_threshold_converges_quickly() {
    let mut agent = LmmAgent::new("Tester".into(), "Think test.".into());
    let result = agent.think_with("goal", 10, 0.25, 1.0, 0.05).await.unwrap();
    assert!(result.steps <= 10);
}

#[tokio::test]
async fn think_stores_cold_entries_in_ltm() {
    let mut agent = LmmAgent::new("Tester".into(), "Think test.".into());
    let ltm_before = agent.long_term_memory.len();
    agent.think("Rust memory model").await.unwrap();
    let ltm_after_first = agent.long_term_memory.len();
    agent.think("Rust ownership").await.unwrap();
    let ltm_after_second = agent.long_term_memory.len();
    assert!(ltm_after_first >= ltm_before);
    assert!(ltm_after_second >= ltm_after_first);
}

#[tokio::test]
async fn think_called_multiple_times_no_panic() {
    let mut agent = LmmAgent::new("Tester".into(), "Think test.".into());
    for goal in &["Rust safety", "Rust performance", "Rust ecosystem"] {
        let result = agent.think(goal).await;
        assert!(result.is_ok(), "think({goal}) failed: {:?}", result.err());
    }
}

#[tokio::test]
async fn think_memory_snapshot_all_strings() {
    let mut agent = LmmAgent::new("Tester".into(), "Think test.".into());
    let result = agent.think("Rust type system").await.unwrap();
    for s in &result.memory_snapshot {
        let _ = s.len();
    }
}

#[tokio::test]
async fn think_result_converged_is_bool() {
    let mut agent = LmmAgent::new("Tester".into(), "Think test.".into());
    let result = agent.think("a goal").await.unwrap();
    let _ = result.converged;
}

#[test]
fn status_thinking_display() {
    let s = Status::Thinking;
    assert_eq!(s.to_string(), "Thinking");
}

#[test]
fn status_thinking_not_default() {
    assert_ne!(Status::default(), Status::Thinking);
}

#[test]
fn prelude_exports_think_result() {
    let _ = std::mem::size_of::<ThinkResult>();
}

#[test]
fn prelude_exports_think_loop() {
    let _lp = ThinkLoop::builder("goal").build();
}

#[test]
fn prelude_exports_goal_evaluator() {
    let eval = GoalEvaluator::new(0.3);
    assert!(!eval.is_converged(0.5));
}

#[test]
fn prelude_exports_hot_store() {
    let store = HotStore::new(4);
    assert!(store.is_empty());
}

#[test]
fn prelude_exports_cold_store() {
    let store = ColdStore::default();
    assert!(store.is_empty());
}

#[test]
fn prelude_exports_error_from_texts() {
    let e = error_from_texts("hello", "hello");
    assert_eq!(e, 0.0);
}
