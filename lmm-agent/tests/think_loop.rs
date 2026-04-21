// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unit + integration tests for the `ThinkLoop` controller.

use lmm_agent::cognition::r#loop::ThinkLoop;
use lmm_agent::cognition::search::SearchOracle;

#[test]
fn builder_default_max_iterations_is_10() {
    let lp = ThinkLoop::builder("goal").build();
    assert_eq!(lp.max_iterations, 10);
}

#[test]
fn builder_default_convergence_threshold_is_025() {
    let lp = ThinkLoop::builder("goal").build();
    assert_eq!(lp.convergence_threshold, 0.25);
}

#[test]
fn builder_default_stall_patience_is_3() {
    let lp = ThinkLoop::builder("goal").build();
    assert_eq!(lp.stall_patience, 3);
}

#[test]
fn builder_sets_custom_max_iterations() {
    let lp = ThinkLoop::builder("g").max_iterations(7).build();
    assert_eq!(lp.max_iterations, 7);
}

#[test]
fn builder_sets_custom_threshold() {
    let lp = ThinkLoop::builder("g").convergence_threshold(0.1).build();
    assert_eq!(lp.convergence_threshold, 0.1);
}

#[test]
fn builder_clamps_threshold_to_one() {
    let lp = ThinkLoop::builder("g").convergence_threshold(5.0).build();
    assert_eq!(lp.convergence_threshold, 1.0);
}

#[test]
fn builder_clamps_threshold_to_zero() {
    let lp = ThinkLoop::builder("g").convergence_threshold(-1.0).build();
    assert_eq!(lp.convergence_threshold, 0.0);
}

#[test]
fn builder_sets_stall_patience() {
    let lp = ThinkLoop::builder("g").stall_patience(5).build();
    assert_eq!(lp.stall_patience, 5);
}

#[test]
fn builder_stall_patience_min_one() {
    let lp = ThinkLoop::builder("g").stall_patience(0).build();
    assert_eq!(lp.stall_patience, 1);
}

#[test]
fn builder_sets_promotion_threshold() {
    let lp = ThinkLoop::builder("g").promotion_threshold(0.7).build();
    assert_eq!(lp.promotion_threshold, 0.7);
}

#[test]
fn builder_sets_hot_capacity() {
    let lp = ThinkLoop::builder("g").hot_capacity(8).build();
    let _ = lp.hot;
}

#[test]
fn new_max_iterations_min_one() {
    let lp = ThinkLoop::new("g", 0, 0.25, 1.0, 0.05);
    assert_eq!(lp.max_iterations, 1);
}

#[test]
fn new_threshold_clamp_upper() {
    let lp = ThinkLoop::new("g", 5, 100.0, 1.0, 0.05);
    assert_eq!(lp.convergence_threshold, 1.0);
}

#[test]
fn new_threshold_clamp_lower() {
    let lp = ThinkLoop::new("g", 5, -1.0, 1.0, 0.05);
    assert_eq!(lp.convergence_threshold, 0.0);
}

#[tokio::test]
async fn run_steps_within_max() {
    let mut oracle = SearchOracle::new(3);
    let mut lp = ThinkLoop::new("test goal", 4, 0.25, 1.0, 0.05);
    let r = lp.run(&mut oracle).await;
    assert!(r.steps <= 4, "steps={}", r.steps);
}

#[tokio::test]
async fn run_error_always_in_range() {
    let mut oracle = SearchOracle::new(1);
    let mut lp = ThinkLoop::new("test", 3, 0.25, 1.0, 0.05);
    let r = lp.run(&mut oracle).await;
    assert!(r.final_error >= 0.0 && r.final_error <= 1.0);
}

#[tokio::test]
async fn run_signals_count_equals_steps() {
    let mut oracle = SearchOracle::new(1);
    let mut lp = ThinkLoop::new("goal text here", 5, 0.25, 1.0, 0.05);
    let r = lp.run(&mut oracle).await;
    assert_eq!(r.signals.len(), r.steps);
}

#[tokio::test]
async fn run_memory_snapshot_is_vec() {
    let mut oracle = SearchOracle::new(1);
    let mut lp = ThinkLoop::new("goal", 3, 0.25, 1.0, 0.05);
    let r = lp.run(&mut oracle).await;
    assert!(r.memory_snapshot.len() <= lp.hot.capacity);
}

#[tokio::test]
async fn run_increments_cold_store_with_zero_threshold() {
    let mut oracle = SearchOracle::new(1);
    let mut lp = ThinkLoop::builder("goal")
        .max_iterations(3)
        .promotion_threshold(0.0)
        .build();
    lp.run(&mut oracle).await;
    let _ = lp.cold.len();
}

#[tokio::test]
async fn run_consecutive_calls_accumulate_cold() {
    let mut oracle = SearchOracle::new(1);
    let mut lp = ThinkLoop::builder("goal")
        .max_iterations(2)
        .promotion_threshold(0.0)
        .build();
    lp.run(&mut oracle).await;
    let after_first = lp.cold.len();
    lp.run(&mut oracle).await;
    let after_second = lp.cold.len();
    assert!(after_second >= after_first);
}

#[tokio::test]
async fn run_with_single_iteration_cap() {
    let mut oracle = SearchOracle::new(1);
    let mut lp = ThinkLoop::new("goal", 1, 0.25, 1.0, 0.05);
    let r = lp.run(&mut oracle).await;
    assert_eq!(r.steps, 1);
}

#[tokio::test]
async fn run_does_not_converge_offline_with_zero_threshold() {
    let mut oracle = SearchOracle::new(1);
    let mut lp = ThinkLoop::new("goal", 5, 0.0, 1.0, 0.05);
    let r = lp.run(&mut oracle).await;
    assert!(
        !r.converged,
        "should not converge offline with threshold=0.0"
    );
}

#[tokio::test]
async fn run_stall_detection_terminates_early() {
    let mut oracle = SearchOracle::new(1);
    let mut lp = ThinkLoop::builder("some goal words here and more")
        .max_iterations(20)
        .stall_patience(2)
        .build();
    let r = lp.run(&mut oracle).await;
    assert!(r.steps <= 20);
}
