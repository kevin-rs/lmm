// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `WorldMap` - learned spatial model of the ARC-AGI environment.
//!
//! `WorldMap` accumulates observed state transitions, wall constraints, milestone
//! states, and pixel-coordinate annotations as the agent explores the grid. It is
//! the agent's primary long-term spatial memory within a single level.
//!
//! ## Lifecycle
//!
//! - **Cleared** when the game advances to a new level (the geometry changes).
//! - **Persisted** across virtual trials within the same level so BFS can replay
//!   previously discovered routes without re-exploring.
//!
//! ## Usage
//!
//! Pass slices of `WorldMap` data to [`PathfindingTool`] methods; do not embed
//! search logic in `WorldMap` itself.
//!
//! [`PathfindingTool`]: crate::tools::PathfindingTool

use std::collections::{HashMap, HashSet};

/// Learned spatial model of a single ARC-AGI level.
///
/// The map is built incrementally from observed `(state, action) → next_state`
/// tuples. Wall constraints are added whenever the player attempts an action
/// and the state does not change.
///
/// # Invariants
///
/// - A `(from, action)` pair is removed from `transitions` when added to `walls`.
/// - `state_positions` is only set when a transition is first recorded so that
///   the stored coordinate corresponds to the observed player location.
#[derive(Debug, Clone, Default)]
pub struct WorldMap {
    /// Observed passable transitions: `state → (action → next_state)`.
    pub transitions: HashMap<u64, HashMap<u32, u64>>,

    /// Known wall directions per state: `state → {blocked_action, ...}`.
    pub walls: HashMap<u64, HashSet<u32>>,

    /// States where the player triggered a significant environment event
    /// (e.g. rotation modifier, level entry). Used to guide BFS in later trials.
    pub milestones: Vec<u64>,

    /// The state/action pair that immediately preceded a Win - used to replay
    /// a winning route in subsequent trials.
    pub win_predecessor: Option<(u64, u32)>,

    /// Pixel coordinate of each known state, set when a transition is first observed.
    pub state_positions: HashMap<u64, (usize, usize)>,
}

impl WorldMap {
    /// Creates an empty world map.
    ///
    /// # Time complexity: O(1)
    /// # Space complexity: O(1)
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a passable transition `from --action--> to`.
    ///
    /// Only **observed** transitions are stored; the map never infers reverse
    /// transitions automatically because open-grid pixel changes mean that
    /// `from --action--> to` does not imply `to --reverse_action--> from`.
    ///
    /// # Time complexity: O(1) amortised
    /// # Space complexity: O(1) amortised
    pub fn record_passage(&mut self, from: u64, action: u32, to: u64) {
        self.transitions.entry(from).or_default().insert(action, to);
    }

    /// Records a reverse (backtrack) transition `to --reverse_action--> from`.
    ///
    /// Call this only when the geometry guarantees that the reverse step is safe
    /// (e.g. a simple bidirectional corridor, not a teleport or stair tile).
    ///
    /// # Time complexity: O(1) amortised
    /// # Space complexity: O(1) amortised
    pub fn record_reverse_passage(&mut self, from: u64, action: u32, to: u64, reverse_action: u32) {
        self.transitions
            .entry(to)
            .or_default()
            .insert(reverse_action, from);
        if let Some(tr) = self.walls.get_mut(&to) {
            tr.remove(&reverse_action);
        }
        let _ = (from, action);
    }

    /// Records that `action` from `state` is blocked by a wall.
    ///
    /// Any previously inferred transition for the same `(state, action)` pair is
    /// removed to maintain consistency.
    ///
    /// # Time complexity: O(1) amortised
    /// # Space complexity: O(1) amortised
    pub fn record_wall(&mut self, state: u64, action: u32) {
        self.walls.entry(state).or_default().insert(action);
        if let Some(tr) = self.transitions.get_mut(&state) {
            tr.remove(&action);
        }
    }

    /// Records `state` as a milestone (e.g. modifier activated, level start).
    ///
    /// Duplicates are silently ignored.
    ///
    /// # Time complexity: O(M) where M = current milestone count
    /// # Space complexity: O(1)
    pub fn record_milestone(&mut self, state: u64) {
        if !self.milestones.contains(&state) {
            self.milestones.push(state);
        }
    }

    /// Returns `true` when `action` from `state` is known to be blocked.
    ///
    /// # Time complexity: O(1)
    /// # Space complexity: O(1)
    pub fn is_wall(&self, state: u64, action: u32) -> bool {
        self.walls.get(&state).is_some_and(|w| w.contains(&action))
    }

    /// Returns the predicted next state for `(state, action)`, or `None` when unknown.
    ///
    /// # Time complexity: O(1)
    /// # Space complexity: O(1)
    pub fn predict(&self, state: u64, action: u32) -> Option<u64> {
        self.transitions.get(&state)?.get(&action).copied()
    }

    /// Computes summary statistics: `(unique_states, total_walls, total_passages)`.
    ///
    /// Used for logging and agent introspection.
    ///
    /// # Time complexity: O(S) where S = total states in the map
    /// # Space complexity: O(S)
    pub fn stats(&self) -> (usize, usize, usize) {
        let mut states: HashSet<u64> = HashSet::new();
        for (&s, tr) in &self.transitions {
            states.insert(s);
            states.extend(tr.values());
        }
        states.extend(self.walls.keys());
        let total_walls = self.walls.values().map(|v| v.len()).sum();
        let total_passages = self.transitions.values().map(|v| v.len()).sum();
        (states.len(), total_walls, total_passages)
    }

    /// Collects wall data in the format required by [`PathfindingTool::spatial_astar`].
    ///
    /// Returns a set of `((x, y), action)` entries for every wall whose state has
    /// a known pixel position.
    ///
    /// # Time complexity: O(W) where W = total wall entries
    /// # Space complexity: O(W)
    pub fn pos_walls(&self) -> HashSet<((usize, usize), u32)> {
        let mut result: HashSet<((usize, usize), u32)> = HashSet::new();
        for (&state, blocked) in &self.walls {
            if let Some(&pos) = self.state_positions.get(&state) {
                for &action in blocked {
                    result.insert((pos, action));
                }
            }
        }
        result
    }

    /// Returns the set of all pixel coordinates that have a known state mapping.
    ///
    /// Used as the `visited_coords` argument to [`PathfindingTool::spatial_astar`] so
    /// the planner strongly prefers previously explored cells.
    ///
    /// # Time complexity: O(P) where P = state_positions entries
    /// # Space complexity: O(P)
    pub fn visited_pixel_coords(&self) -> HashSet<(usize, usize)> {
        self.state_positions.values().copied().collect()
    }

    /// Resets all learned map data.
    ///
    /// Called when the agent advances to a new level whose geometry differs entirely
    /// from the previous one.
    ///
    /// # Time complexity: O(S + W) where S = states, W = walls
    /// # Space complexity: O(1)
    pub fn clear(&mut self) {
        self.transitions.clear();
        self.walls.clear();
        self.milestones.clear();
        self.win_predecessor = None;
        self.state_positions.clear();
    }
}
