// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `FrameContext` - per-frame game-state parsing.
//!
//! Wraps the raw [`FrameData`] returned by the ARC-AGI server and exposes
//! higher-level accessors that the policy uses to make decisions:
//! player position, modifier position, bonus positions, target position, and
//! stable image-hash state keys.

use arc_agi_rs::models::{FrameData, GameState};
use std::collections::HashSet;

/// The context of a single frame of the game, providing utility methods for state analysis and entity detection.
pub struct FrameContext<'a> {
    pub inner: &'a FrameData,
}

impl<'a> FrameContext<'a> {
    /// Creates a new frame context from the provided frame data.
    pub fn new(frame: &'a FrameData) -> Self {
        Self { inner: frame }
    }

    /// Determines whether the current frame is a terminal frame (Win or GameOver).
    pub fn is_terminal(&self) -> bool {
        matches!(self.inner.state, GameState::Win | GameState::GameOver)
    }

    /// Calculates the grid dimensions dynamically.
    pub fn grid_dims(&self) -> (usize, usize) {
        let grid = self.grid_values();
        let rows = grid.len();
        let cols = grid.first().map_or(0, |r| r.len());
        (rows, cols)
    }

    /// Generates a uniquely identifiable state key hash for the agent position, excluding the varying bottom ui strip.
    pub fn state_key(&self) -> u64 {
        let grid = self.grid_values();
        let (rows, _) = self.grid_dims();
        let safe_rows = rows.saturating_sub(10);
        let mut combined = format!("L{}|", self.inner.levels_completed);
        for row in grid.iter().take(safe_rows) {
            for &v in row {
                combined.push((v as u8 + b'0') as char);
            }
        }
        fnv1a_hash(&combined)
    }

    /// Locates the top-left coordinate of the player block, characterized by a specific pixel value of 12.
    pub fn player_pos(&self) -> Option<(usize, usize)> {
        let grid = self.grid_values();
        let (rows, cols) = self.grid_dims();
        for row in 0..rows {
            for col in 0..cols {
                if grid
                    .get(row)
                    .and_then(|r| r.get(col))
                    .copied()
                    .unwrap_or(-1)
                    == 12
                {
                    return Some((col, row));
                }
            }
        }
        None
    }

    /// Generates a hash representing the bottom piece-display area exclusively used for detecting rotation/modification.
    pub fn ui_hash(&self) -> Option<u64> {
        let piece = self.bottom_left_piece()?;
        let mut s = String::with_capacity(128);
        for row in piece {
            for v in row {
                s.push((v.max(0) as u8 + b'0') as char);
            }
        }
        Some(fnv1a_hash(&s))
    }

    /// Extracts the matrix layout of the player piece displayed in the bottom-left corner of the UI.
    pub fn bottom_left_piece(&self) -> Option<Vec<Vec<i64>>> {
        self.player_pos()?;
        let grid = self.grid_values();
        let (rows, _) = self.grid_dims();
        let display_start_row = rows.saturating_sub(10);
        let mut piece = Vec::new();
        for row in grid.iter().skip(display_start_row).take(7) {
            let row_slice = row.iter().take(12).copied().collect();
            piece.push(row_slice);
        }
        Some(piece)
    }

    /// Extracts the matrix layout of the final target destination piece displayed in the bottom-right corner of the UI.
    pub fn target_piece(&self) -> Option<Vec<Vec<i64>>> {
        let grid = self.grid_values();
        let (rows, cols) = self.grid_dims();
        let display_start_row = rows.saturating_sub(10);
        let target_col_start = cols.saturating_sub(16).max(12);
        let mut piece = Vec::new();
        for row in grid.iter().skip(display_start_row).take(7) {
            let row_slice = row
                .iter()
                .skip(target_col_start)
                .take(12)
                .copied()
                .collect();
            piece.push(row_slice);
        }
        Some(piece)
    }

    /// Determines if the current dynamic bottom-left piece identically matches the rotation and shape of the destination target piece.
    pub fn player_piece_matches_target(&self) -> bool {
        match (self.bottom_left_piece(), self.target_piece()) {
            (Some(bl), Some(tr)) => {
                if bl.iter().flatten().all(|&v| v <= 0) {
                    return false;
                }
                bl == tr
            }
            _ => false,
        }
    }

    /// Detects the target pixel coordinate of the modifier cell matching a cross-shaped pattern of any active block color.
    #[allow(clippy::needless_range_loop)]
    pub fn modifier_pos(&self) -> Option<(usize, usize)> {
        let grid = self.grid_values();
        let (rows, cols) = self.grid_dims();
        if rows < 3 || cols < 3 {
            return None;
        }

        let mut clusters: Vec<Vec<(usize, usize)>> = Vec::new();
        for r in 1..rows.saturating_sub(10) {
            for c in 1..cols.saturating_sub(1) {
                let v = grid[r][c];
                if v > 0
                    && v != 3
                    && v != 4
                    && v != 5
                    && v != 8
                    && v != 9
                    && v != 11
                    && v != 12
                    && v != 14
                {
                    let mut added = false;
                    for cluster in &mut clusters {
                        if cluster
                            .iter()
                            .any(|&(cr, cc)| cr.abs_diff(r) < 8 && cc.abs_diff(c) < 8)
                        {
                            cluster.push((r, c));
                            added = true;
                            break;
                        }
                    }
                    if !added {
                        clusters.push(vec![(r, c)]);
                    }
                }
            }
        } // ignore bottom UI row

        for cluster in clusters {
            let min_r = cluster.iter().map(|&(r, _)| r).min().unwrap();
            let max_r = cluster.iter().map(|&(r, _)| r).max().unwrap();
            let min_c = cluster.iter().map(|&(_, c)| c).min().unwrap();
            let max_c = cluster.iter().map(|&(_, c)| c).max().unwrap();

            if max_r.abs_diff(min_r) < 6 && max_c.abs_diff(min_c) < 6 {
                let sum_r: usize = cluster.iter().map(|&(r, _)| r).sum();
                let sum_c: usize = cluster.iter().map(|&(_, c)| c).sum();
                return Some((sum_c / cluster.len(), sum_r / cluster.len()));
            }
        }
        None
    }

    /// Identifies the internal top-left coordinate of the destination target box matching a uniform value 3 border.
    pub fn target_pos(&self) -> Option<(usize, usize)> {
        let grid = self.grid_values();
        let (rows, cols) = self.grid_dims();
        for row in 0..rows {
            for col in 0..cols {
                if grid
                    .get(row)
                    .and_then(|r| r.get(col))
                    .copied()
                    .unwrap_or(-1)
                    == 3
                    && grid
                        .get(row)
                        .and_then(|r| r.get(col + 1))
                        .copied()
                        .unwrap_or(-1)
                        == 3
                {
                    return Some((col + 2, row + 2));
                }
            }
        }
        None
    }

    /// Detects and returns all known collectible bonus positions in the grid matching known border signatures.
    pub fn bonus_positions(&self) -> Vec<(usize, usize)> {
        let grid = self.grid_values();
        let (rows, cols) = self.grid_dims();
        let mut found = Vec::new();
        let mut tagged: HashSet<(usize, usize)> = Default::default();
        for row in 0..rows {
            for col in 0..cols {
                if tagged.contains(&(col, row)) {
                    continue;
                }
                let v = grid
                    .get(row)
                    .and_then(|r| r.get(col))
                    .copied()
                    .unwrap_or(-1);
                if v == 14 {
                    let horiz = (col..col + 3)
                        .filter(|&c| {
                            grid.get(row).and_then(|r| r.get(c)).copied().unwrap_or(-1) == 14
                        })
                        .count();
                    let vert = (row..row + 3)
                        .filter(|&r| {
                            grid.get(r)
                                .and_then(|rr| rr.get(col))
                                .copied()
                                .unwrap_or(-1)
                                == 14
                        })
                        .count();
                    if horiz >= 3 && vert >= 3 {
                        found.push((col + 5, row + 5));
                        tagged.insert((col, row));
                        continue;
                    }
                }
                if v == 11 && col + 2 < cols && row + 2 < rows {
                    let top_row = (col..col + 3)
                        .filter(|&c| {
                            grid.get(row).and_then(|r| r.get(c)).copied().unwrap_or(-1) == 11
                        })
                        .count();
                    let bot_row = (col..col + 3)
                        .filter(|&c| {
                            grid.get(row + 2)
                                .and_then(|r| r.get(c))
                                .copied()
                                .unwrap_or(-1)
                                == 11
                        })
                        .count();
                    let left_col = grid
                        .get(row + 1)
                        .and_then(|r| r.get(col))
                        .copied()
                        .unwrap_or(-1)
                        == 11;
                    let right_col = grid
                        .get(row + 1)
                        .and_then(|r| r.get(col + 2))
                        .copied()
                        .unwrap_or(-1)
                        == 11;
                    let center = grid
                        .get(row + 1)
                        .and_then(|r| r.get(col + 1))
                        .copied()
                        .unwrap_or(-1);
                    if top_row == 3 && bot_row == 3 && left_col && right_col && center != 11 {
                        found.push((col + 1, row + 1));
                        for dy in 0..3 {
                            for dx in 0..3 {
                                tagged.insert((col + dx, row + dy));
                            }
                        }
                    }
                }
            }
        }
        found
    }

    /// Provides a list of viable action integer states avoiding the 0 reset state.
    pub fn available_non_reset(&self) -> Vec<u32> {
        self.inner
            .available_actions
            .iter()
            .copied()
            .filter(|&a| a != 0)
            .collect()
    }

    /// Generates a standardized textual description of the frame observables for reasoning contexts.
    pub fn encode_observation(&self) -> String {
        let pos = self
            .player_pos()
            .map(|(x, y)| format!("x={} y={}", x, y))
            .unwrap_or_else(|| "pos=unknown".into());
        format!(
            "game={} state={} levels={} win={} {} actions=[{}]",
            self.inner.game_id,
            self.inner.state.as_str(),
            self.inner.levels_completed,
            self.inner.win_levels,
            pos,
            self.inner
                .available_actions
                .iter()
                .map(|a| a.to_string())
                .collect::<Vec<_>>()
                .join(","),
        )
    }

    /// Projects the hierarchical JSON layer frame representation into a straightforward two dimensional array of integers.
    pub fn grid_values(&self) -> Vec<Vec<i64>> {
        self.inner
            .frame
            .first()
            .and_then(|layer| layer.as_array())
            .map(|rows| {
                rows.iter()
                    .map(|row| {
                        row.as_array()
                            .map(|cols| cols.iter().map(|v| v.as_i64().unwrap_or(-1)).collect())
                            .unwrap_or_default()
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Computes equality strictly based upon the internal frame representation structures.
pub fn frames_equal(a: &FrameData, b: &FrameData) -> bool {
    a.frame == b.frame
}

/// Applies a 64-bit FNV-1a non-cryptographic hash mapping to a standard string.
fn fnv1a_hash(s: &str) -> u64 {
    const BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    s.bytes()
        .fold(BASIS, |hash, byte| (hash ^ byte as u64).wrapping_mul(PRIME))
}
