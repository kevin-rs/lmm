// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use arc_lmm_agent::world::WorldMap;

#[test]
fn record_passage_and_predict() {
    let mut map = WorldMap::new();
    map.record_passage(1, 4, 2);
    assert_eq!(map.predict(1, 4), Some(2));
    assert_eq!(map.predict(1, 3), None);
}

#[test]
fn record_wall_removes_transition() {
    let mut map = WorldMap::new();
    map.record_passage(1, 4, 2);
    map.record_wall(1, 4);
    assert!(map.is_wall(1, 4));
    assert_eq!(map.predict(1, 4), None);
}

#[test]
fn record_milestone_no_duplicates() {
    let mut map = WorldMap::new();
    map.record_milestone(1);
    map.record_milestone(1);
    assert_eq!(map.milestones.len(), 1);
}

#[test]
fn stats_counts_correctly() {
    let mut map = WorldMap::new();
    map.record_passage(1, 4, 2);
    map.record_passage(2, 4, 3);
    map.record_wall(3, 1);
    let (states, walls, passages) = map.stats();
    assert_eq!(states, 3);
    assert_eq!(walls, 1);
    assert_eq!(passages, 2);
}

#[test]
fn predict_returns_none_for_unknown_state() {
    let map = WorldMap::new();
    assert!(map.predict(999, 2).is_none());
}
