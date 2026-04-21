// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unit tests for `cognition::signal` - `CognitionSignal` and `error_from_texts`.

use lmm_agent::cognition::signal::{CognitionSignal, error_from_texts};

#[test]
fn error_identical_texts_is_zero() {
    assert_eq!(
        error_from_texts("the quick brown fox", "the quick brown fox"),
        0.0
    );
}

#[test]
fn error_disjoint_texts_is_one() {
    assert_eq!(error_from_texts("alpha beta", "gamma delta"), 1.0);
}

#[test]
fn error_partial_overlap_between_zero_and_one() {
    let e = error_from_texts("Rust is fast", "Rust is safe and fast");
    assert!(e > 0.0 && e < 1.0, "expected partial overlap, got {e}");
}

#[test]
fn error_empty_goal_is_one() {
    assert_eq!(error_from_texts("", "some text"), 1.0);
}

#[test]
fn error_empty_observation_is_one() {
    assert_eq!(error_from_texts("some text", ""), 1.0);
}

#[test]
fn error_both_empty_is_zero() {
    assert_eq!(error_from_texts("", ""), 0.0);
}

#[test]
fn error_case_insensitive() {
    assert_eq!(error_from_texts("Rust", "rust"), 0.0);
}

#[test]
fn error_strips_punctuation() {
    assert_eq!(error_from_texts("hello!", "hello"), 0.0);
}

#[test]
fn error_result_always_in_unit_interval() {
    let cases = [
        ("", "foo"),
        ("foo", ""),
        ("a b c", "a b c"),
        ("a b", "c d"),
        ("one two three", "two three four"),
    ];
    for (a, b) in cases {
        let e = error_from_texts(a, b);
        assert!(
            (0.0..=1.0).contains(&e),
            "error_from_texts({a:?}, {b:?}) = {e} is out of [0,1]"
        );
    }
}

#[test]
fn signal_perfect_match_zero_error_positive_reward() {
    let sig = CognitionSignal::new(0, "goal".into(), "goal".into(), 1.0, 0.0);
    assert_eq!(sig.error, 0.0);
    assert!(sig.reward > 0.0);
}

#[test]
fn signal_no_match_error_one_zero_reward() {
    let sig = CognitionSignal::new(0, "goal".into(), "zzz".into(), 1.0, 0.0);
    assert_eq!(sig.error, 1.0);
    assert_eq!(sig.reward, 0.0);
}

#[test]
fn signal_reward_always_nonnegative() {
    for kp in [0.0_f64, 0.1, 0.5, 1.0, 5.0] {
        let sig = CognitionSignal::new(0, "a b c".into(), "x y z".into(), kp, 0.0);
        assert!(
            sig.reward >= 0.0,
            "reward negative for kp={kp}: {}",
            sig.reward
        );
    }
}

#[test]
fn signal_gain_clamped_upper() {
    let sig = CognitionSignal::new(10, "foo".into(), "baz".into(), 1.0, 99.0);
    assert!(sig.gain <= 10.0, "gain exceeded upper bound: {}", sig.gain);
}

#[test]
fn signal_gain_clamped_lower() {
    let sig = CognitionSignal::new(0, "foo".into(), "baz".into(), 0.0, 0.0);
    assert!(sig.gain >= 0.1, "gain below lower bound: {}", sig.gain);
}

#[test]
fn signal_integral_anti_windup() {
    let sig = CognitionSignal::new(0, "a".into(), "b".into(), 1.0, 100.0);
    assert!(sig.integral <= 100.0);
}

#[test]
fn signal_step_stored_correctly() {
    let sig = CognitionSignal::new(7, "q".into(), "o".into(), 1.0, 0.0);
    assert_eq!(sig.step, 7);
}

#[test]
fn signal_query_and_observation_stored() {
    let sig = CognitionSignal::new(0, "my query".into(), "my observation".into(), 1.0, 0.0);
    assert_eq!(sig.query, "my query");
    assert_eq!(sig.observation, "my observation");
}

#[test]
fn signal_integral_accumulates() {
    let first = CognitionSignal::new(0, "x".into(), "y".into(), 1.0, 0.0);
    let second = CognitionSignal::new(1, "x".into(), "y".into(), 1.0, first.integral);
    assert!(second.integral >= first.integral);
}
