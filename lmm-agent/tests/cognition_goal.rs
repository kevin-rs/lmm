// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unit tests for `cognition::goal` - `GoalEvaluator`.

use lmm_agent::cognition::goal::GoalEvaluator;

#[test]
fn default_threshold_is_025() {
    assert_eq!(GoalEvaluator::default().convergence_threshold, 0.25);
}

#[test]
fn is_converged_below_threshold() {
    let e = GoalEvaluator::new(0.3);
    assert!(e.is_converged(0.0));
    assert!(e.is_converged(0.1));
    assert!(e.is_converged(0.29));
}

#[test]
fn is_converged_at_threshold_is_false() {
    let e = GoalEvaluator::new(0.3);
    assert!(!e.is_converged(0.3));
}

#[test]
fn is_not_converged_above_threshold() {
    let e = GoalEvaluator::new(0.3);
    assert!(!e.is_converged(0.5));
    assert!(!e.is_converged(1.0));
}

#[test]
fn error_identical_strings_zero() {
    assert_eq!(GoalEvaluator::error("hello world", "hello world"), 0.0);
}

#[test]
fn error_disjoint_strings_one() {
    assert_eq!(GoalEvaluator::error("foo bar", "baz qux"), 1.0);
}

#[test]
fn error_partial_overlap_between_zero_one() {
    let e = GoalEvaluator::error("Rust is fast", "Rust is safe");
    assert!(e > 0.0 && e < 1.0, "expected partial, got {e}");
}

#[test]
fn progress_full_when_zero_error() {
    assert_eq!(GoalEvaluator::progress(0.0), 100.0);
}

#[test]
fn progress_zero_when_max_error() {
    assert_eq!(GoalEvaluator::progress(1.0), 0.0);
}

#[test]
fn progress_midpoint() {
    let p = GoalEvaluator::progress(0.5);
    assert!((p - 50.0).abs() < 1e-10, "expected 50.0, got {p}");
}

#[test]
#[should_panic(expected = "GoalEvaluator threshold must be in [0, 1]")]
fn new_panics_on_out_of_range() {
    GoalEvaluator::new(1.5);
}

#[test]
#[should_panic(expected = "GoalEvaluator threshold must be in [0, 1]")]
fn new_panics_on_negative() {
    GoalEvaluator::new(-0.1);
}
