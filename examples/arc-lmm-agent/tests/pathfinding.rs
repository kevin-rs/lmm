// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use arc_lmm_agent::tools::PathfindingTool;
use std::collections::{HashMap, HashSet};

fn simple_graph() -> HashMap<u64, HashMap<u32, u64>> {
    let mut t: HashMap<u64, HashMap<u32, u64>> = HashMap::new();
    t.entry(1).or_default().insert(4, 2);
    t.entry(2).or_default().insert(4, 3);
    t.entry(3).or_default().insert(4, 4);
    t
}

#[test]
fn bfs_returns_path_in_simple_graph() {
    let t = simple_graph();
    let path = PathfindingTool::bfs(1, 4, &t);
    assert_eq!(path, Some(vec![4, 4, 4]));
}

#[test]
fn bfs_same_state_returns_empty() {
    let t = simple_graph();
    let path = PathfindingTool::bfs(1, 1, &t);
    assert_eq!(path, Some(vec![]));
}

#[test]
fn bfs_returns_none_when_unreachable() {
    let t = simple_graph();
    let path = PathfindingTool::bfs(1, 99, &t);
    assert!(path.is_none());
}

#[test]
fn spatial_astar_straight_line_right() {
    let walls: HashSet<((usize, usize), u32)> = HashSet::new();
    let visited: HashSet<(usize, usize)> = (0..=60)
        .step_by(5)
        .flat_map(|x| (0..=60usize).step_by(5).map(move |y| (x, y)))
        .collect();

    let path = PathfindingTool::spatial_astar(0, 0, 20, 0, &walls, &visited);
    assert!(path.is_some());
    let path = path.unwrap();
    assert!(path.iter().all(|&a| a == 4));
}

#[test]
fn reconstruct_path_correctness() {
    let mut parents: HashMap<u64, (u64, u32)> = HashMap::new();
    parents.insert(2, (1, 4));
    parents.insert(3, (2, 4));
    parents.insert(4, (3, 4));
    let path = PathfindingTool::reconstruct_path(&parents, 1, 4);
    assert_eq!(path, vec![4, 4, 4]);
}

#[test]
fn reverse_path_is_correct() {
    let forward = vec![4, 4, 2, 4];
    let rev = PathfindingTool::reverse_path(&forward);
    assert_eq!(rev, vec![3, 1, 3, 3]);
}

#[test]
fn action_next_pos_all_directions() {
    assert_eq!(PathfindingTool::action_next_pos(10, 10, 1), (10, 5));
    assert_eq!(PathfindingTool::action_next_pos(10, 10, 2), (10, 15));
    assert_eq!(PathfindingTool::action_next_pos(10, 10, 3), (5, 10));
    assert_eq!(PathfindingTool::action_next_pos(10, 10, 4), (15, 10));
}
