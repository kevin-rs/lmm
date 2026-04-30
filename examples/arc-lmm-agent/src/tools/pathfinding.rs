// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `PathfindingTool` - graph-based BFS and Cartesian A\* for the ARC agent.
//!
//! All search algorithms live here as **pure functions**. They receive data as
//! immutable references; they never mutate world state. The agent policy
//! calls these methods and applies the resulting action sequences.
//!
//! ## Algorithms
//!
//! | Method | Algorithm | Time | Space |
//! |--------|-----------|------|-------|
//! | [`PathfindingTool::bfs`] | Breadth-first search on the state-transition graph | O(V + E) | O(V) |
//! | [`PathfindingTool::spatial_astar`] | A\* with Manhattan heuristic on a 2-D pixel grid | O(N log N) | O(N) |
//! | [`PathfindingTool::reconstruct_path`] | Backtrack through parent map | O(L) | O(L) |
//! | [`PathfindingTool::action_next_pos`] | Constant-time next-pixel projection | O(1) | O(1) |
//!
//! where V = visited states, E = edges (transitions), N = grid cells, L = path length.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

/// A stateless pathfinding tool the agent policy delegates search to.
///
/// Instantiate once and hold as a field; every method borrows world-map data
/// passed in by the caller so there is no internal allocation between calls.
///
/// # Examples
///
/// ```rust
/// use arc_lmm_agent::tools::PathfindingTool;
/// use std::collections::HashMap;
///
/// let mut transitions: HashMap<u64, HashMap<u32, u64>> = HashMap::new();
/// transitions.entry(1).or_default().insert(4, 2);
/// transitions.entry(2).or_default().insert(4, 3);
///
/// let path = PathfindingTool::bfs(1, 3, &transitions);
/// assert_eq!(path, Some(vec![4, 4]));
/// ```
#[derive(Debug, Clone, Default)]
pub struct PathfindingTool;

impl PathfindingTool {
    /// Returns the shortest action sequence from `from` to `to` using BFS over
    /// the supplied state-transition graph.
    ///
    /// Returns `None` when `to` is unreachable from `from`.
    ///
    /// # Parameters
    ///
    /// - `from` - starting state hash.
    /// - `to` - goal state hash.
    /// - `transitions` - map of `state → (action → next_state)` collected by the agent.
    ///
    /// # Time complexity: O(V + E)
    /// # Space complexity: O(V)
    pub fn bfs(
        from: u64,
        to: u64,
        transitions: &HashMap<u64, HashMap<u32, u64>>,
    ) -> Option<Vec<u32>> {
        if from == to {
            return Some(vec![]);
        }
        let mut queue: VecDeque<u64> = VecDeque::from([from]);
        let mut seen: HashSet<u64> = HashSet::from([from]);
        let mut parents: HashMap<u64, (u64, u32)> = HashMap::new();

        while let Some(current) = queue.pop_front() {
            for (&action, &next) in transitions.get(&current).into_iter().flatten() {
                if seen.insert(next) {
                    parents.insert(next, (current, action));
                    if next == to {
                        return Some(Self::reconstruct_path(&parents, from, to));
                    }
                    queue.push_back(next);
                }
            }
        }
        None
    }

    /// Reconstructs an action sequence from the BFS `parents` map.
    ///
    /// Traces from `to` back to `from` following the parent pointers, then
    /// reverses the accumulated actions to produce a forward-going plan.
    ///
    /// # Parameters
    ///
    /// - `parents` - map of `state → (predecessor_state, action_taken)`.
    /// - `from` - origin state hash.
    /// - `to` - terminal state hash.
    ///
    /// # Time complexity: O(L) where L = path length
    /// # Space complexity: O(L)
    pub fn reconstruct_path(parents: &HashMap<u64, (u64, u32)>, from: u64, to: u64) -> Vec<u32> {
        let mut path = Vec::new();
        let mut node = to;
        while node != from {
            if let Some(&(prev, action)) = parents.get(&node) {
                path.push(action);
                node = prev;
            } else {
                break;
            }
        }
        path.reverse();
        path
    }

    /// Finds the optimal action sequence from pixel position `(start_x, start_y)` to
    /// `(goal_x, goal_y)` using the A\* algorithm with a Manhattan heuristic.
    ///
    /// The search operates directly on pixel coordinates with 5-pixel grid steps.
    /// Known wall positions penalise expansion. Unvisited coordinates receive a large
    /// penalty to encourage the agent to stay on previously explored paths.
    ///
    /// Returns `None` when the goal is unreachable within the map bounds.
    ///
    /// # Parameters
    ///
    /// - `start_x`, `start_y` - current player pixel coordinates.
    /// - `goal_x`, `goal_y` - target pixel coordinates.
    /// - `pos_walls` - set of `((x, y), action)` pairs marking known walls.
    /// - `visited_coords` - set of pixel coordinates the agent has previously stood on.
    ///
    /// # Time complexity: O(N log N) where N = grid cells within bounds
    /// # Space complexity: O(N)
    pub fn spatial_astar(
        start_x: usize,
        start_y: usize,
        goal_x: usize,
        goal_y: usize,
        pos_walls: &HashSet<((usize, usize), u32)>,
        visited_coords: &HashSet<(usize, usize)>,
    ) -> Option<Vec<u32>> {
        if start_x == goal_x && start_y == goal_y {
            return Some(vec![]);
        }

        let heuristic = |x: usize, y: usize| -> usize { x.abs_diff(goal_x) + y.abs_diff(goal_y) };

        type AStarNode = Reverse<(usize, usize, usize, usize, u32)>;
        let mut open: BinaryHeap<AStarNode> = BinaryHeap::new();
        let mut g_scores: HashMap<(usize, usize), usize> = HashMap::new();
        let mut parents: HashMap<(usize, usize), ((usize, usize), u32)> = HashMap::new();

        g_scores.insert((start_x, start_y), 0);
        open.push(Reverse((
            heuristic(start_x, start_y),
            0,
            start_x,
            start_y,
            0,
        )));

        while let Some(Reverse((_, cost, cx, cy, prev_action))) = open.pop() {
            if cx == goal_x && cy == goal_y {
                let mut path: Vec<u32> = Vec::new();
                let mut curr = (cx, cy);
                while curr != (start_x, start_y) {
                    if let Some(&(prev, action)) = parents.get(&curr) {
                        path.push(action);
                        curr = prev;
                    } else {
                        break;
                    }
                }
                path.reverse();
                return Some(path);
            }

            let best_known = *g_scores.get(&(cx, cy)).unwrap_or(&usize::MAX);
            if cost > best_known {
                continue;
            }

            for action in 1u32..=4 {
                if pos_walls.contains(&((cx, cy), action)) {
                    continue;
                }
                let (nx, ny) = Self::action_next_pos(cx, cy, action);
                if nx > 200 || ny > 200 {
                    continue;
                }

                let unvisited_penalty = if visited_coords.contains(&(nx, ny)) {
                    0
                } else {
                    5
                };
                let turn_penalty = if prev_action != 0 && prev_action != action {
                    5
                } else {
                    0
                };
                let ng = cost + 1 + unvisited_penalty + turn_penalty;

                if ng < *g_scores.get(&(nx, ny)).unwrap_or(&usize::MAX) {
                    g_scores.insert((nx, ny), ng);
                    parents.insert((nx, ny), ((cx, cy), action));
                    open.push(Reverse((ng + heuristic(nx, ny), ng, nx, ny, action)));
                }
            }
        }
        None
    }

    /// Projects the next pixel position after taking `action` from `(px, py)`.
    ///
    /// Uses the game's 5-pixel grid increment convention:
    /// - `1` → up (y decreases)
    /// - `2` → down (y increases)
    /// - `3` → left (x decreases)
    /// - `4` → right (x increases)
    ///
    /// # Time complexity: O(1)
    /// # Space complexity: O(1)
    #[inline]
    pub fn action_next_pos(px: usize, py: usize, action: u32) -> (usize, usize) {
        match action {
            1 => (px, py.saturating_sub(5)),
            2 => (px, py + 5),
            3 => (px.saturating_sub(5), py),
            4 => (px + 5, py),
            _ => (px, py),
        }
    }

    /// Returns the action that is the inverse (opposite direction) of `action`.
    ///
    /// Used to reverse a recorded path segment: up↔down, left↔right.
    ///
    /// # Time complexity: O(1)
    /// # Space complexity: O(1)
    #[inline]
    pub fn reverse_action(action: u32) -> u32 {
        match action {
            1 => 2,
            2 => 1,
            3 => 4,
            4 => 3,
            _ => 0,
        }
    }

    /// Reverses a forward action sequence into the equivalent backtracking sequence.
    ///
    /// Each action is replaced with its opposite and the resulting list is reversed so
    /// that replaying it traces the original route in the opposite direction.
    ///
    /// # Parameters
    ///
    /// - `path` - forward action sequence to invert.
    ///
    /// # Time complexity: O(L)
    /// # Space complexity: O(L)
    pub fn reverse_path(path: &[u32]) -> Vec<u32> {
        path.iter()
            .rev()
            .map(|&a| Self::reverse_action(a))
            .collect()
    }

    /// BFS on the state-transition graph targeting any explored state whose
    /// cached pixel position is within `radius` of `(goal_x, goal_y)`.
    ///
    /// Unlike [`spatial_astar`], this uses **proven transitions** from
    /// exploration, making it reliable even when the pixel-level wall map is
    /// incomplete. Falls back gracefully: returns `None` when no explored
    /// state is close enough to the goal.
    ///
    /// # Time complexity: O(V + E)
    /// # Space complexity: O(V)
    pub fn bfs_to_position(
        from: u64,
        goal_x: usize,
        goal_y: usize,
        radius: usize,
        transitions: &HashMap<u64, HashMap<u32, u64>>,
        state_positions: &HashMap<u64, (usize, usize)>,
    ) -> Option<Vec<u32>> {
        let targets: HashSet<u64> = state_positions
            .iter()
            .filter(|&(_, &(px, py))| px.abs_diff(goal_x) + py.abs_diff(goal_y) <= radius)
            .map(|(&s, _)| s)
            .collect();

        if targets.is_empty() {
            return None;
        }
        if targets.contains(&from) {
            return Some(vec![]);
        }

        let mut queue: VecDeque<u64> = VecDeque::from([from]);
        let mut seen: HashSet<u64> = HashSet::from([from]);
        let mut parents: HashMap<u64, (u64, u32)> = HashMap::new();

        while let Some(current) = queue.pop_front() {
            for (&action, &next) in transitions.get(&current).into_iter().flatten() {
                if seen.insert(next) {
                    parents.insert(next, (current, action));
                    if targets.contains(&next) {
                        return Some(Self::reconstruct_path(&parents, from, next));
                    }
                    queue.push_back(next);
                }
            }
        }
        None
    }
}
