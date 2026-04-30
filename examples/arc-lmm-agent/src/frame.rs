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

use std::collections::HashSet;

use arc_agi_rs::models::{FrameData, GameState};

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

    /// Returns the pixel value at the given grid coordinate.
    ///
    /// Returns `-1` when the coordinate is out of bounds.
    ///
    /// # Time complexity: O(1)
    /// # Space complexity: O(1)
    pub fn pixel_at(&self, col: usize, row: usize) -> i64 {
        self.grid_values()
            .get(row)
            .and_then(|r| r.get(col))
            .copied()
            .unwrap_or(-1)
    }

    /// Returns the set of distinct positive pixel values that compose the player sprite.
    ///
    /// Scans a 5×5 bounding box anchored at the player position returned by
    /// [`player_pos`]. Returns an empty set when no player is visible.
    ///
    /// # Time complexity: O(1) (fixed 5×5 window)
    /// # Space complexity: O(C) where C = distinct colors in the sprite
    pub fn player_colors(&self) -> HashSet<i64> {
        let mut colors = HashSet::new();
        let Some((px, py)) = self.player_pos() else {
            return colors;
        };
        let grid = self.grid_values();
        for dy in 0..5usize {
            for dx in 0..5usize {
                let v = grid
                    .get(py + dy)
                    .and_then(|r| r.get(px + dx))
                    .copied()
                    .unwrap_or(-1);
                if v > 0 {
                    colors.insert(v);
                }
            }
        }
        colors
    }

    /// Returns `true` when the player piece orientation matches the target piece.
    ///
    /// This is a semantic alias for [`player_piece_matches_target`] used by the
    /// generic routing logic to detect direction mismatches after modifier activation.
    ///
    /// # Time complexity: O(1)
    /// # Space complexity: O(1)
    pub fn direction_matches_target(&self) -> bool {
        self.player_piece_matches_target()
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

    /// Extracts the pixel content **inside** the in-grid target box.
    ///
    /// Locates the bordered rectangular target box using the same border-detection
    /// algorithm as [`target_pos`], then returns the pixels strictly inside the
    /// border (i.e. excluding the border row/col on every side).  This is the
    /// canonical source for the "desired piece shape" comparison because the
    /// in-grid box is always present in every level, at any position, regardless
    /// of what the UI bar shows.
    ///
    /// Returns `None` when no target box can be found.
    ///
    /// # Time complexity: O(R × C × S) where S = box size range (5-15)
    /// # Space complexity: O(S²)
    pub fn target_box_inner_content(&self) -> Option<Vec<Vec<i64>>> {
        let grid = self.grid_values();
        let (rows, cols) = self.grid_dims();
        let safe_rows = rows.saturating_sub(10);

        for s in 5..=15 {
            for row in 0..=safe_rows.saturating_sub(s) {
                for col in 0..=cols.saturating_sub(s) {
                    let v = grid
                        .get(row)
                        .and_then(|r| r.get(col))
                        .copied()
                        .unwrap_or(-1);
                    if v <= 0 || v == 12 || v == 14 || v == 11 {
                        continue;
                    }

                    let mut valid = true;
                    for dc in 0..s {
                        let top = grid
                            .get(row)
                            .and_then(|r| r.get(col + dc))
                            .copied()
                            .unwrap_or(-1);
                        let bot = grid
                            .get(row + s - 1)
                            .and_then(|r| r.get(col + dc))
                            .copied()
                            .unwrap_or(-1);
                        if top != v || bot != v {
                            valid = false;
                            break;
                        }
                    }
                    if !valid {
                        continue;
                    }
                    for dr in 0..s {
                        let left = grid
                            .get(row + dr)
                            .and_then(|r| r.get(col))
                            .copied()
                            .unwrap_or(-1);
                        let right = grid
                            .get(row + dr)
                            .and_then(|r| r.get(col + s - 1))
                            .copied()
                            .unwrap_or(-1);
                        if left != v || right != v {
                            valid = false;
                            break;
                        }
                    }
                    if !valid {
                        continue;
                    }

                    let mut inner_matches = 0;
                    let inner_area = (s - 2) * (s - 2);
                    for dr in 1..s - 1 {
                        for dc in 1..s - 1 {
                            if grid
                                .get(row + dr)
                                .and_then(|r| r.get(col + dc))
                                .copied()
                                .unwrap_or(-1)
                                == v
                            {
                                inner_matches += 1;
                            }
                        }
                    }
                    if inner_matches < inner_area / 2 {
                        let inner: Vec<Vec<i64>> = (1..s - 1)
                            .map(|dr| {
                                (1..s - 1)
                                    .map(|dc| {
                                        grid.get(row + dr)
                                            .and_then(|r| r.get(col + dc))
                                            .copied()
                                            .unwrap_or(-1)
                                    })
                                    .collect()
                            })
                            .collect();
                        return Some(inner);
                    }
                }
            }
        }
        None
    }

    /// Determines if the current player-piece orientation matches the target shape.
    ///
    /// Compares the **bottom-left UI piece display** (player's current orientation)
    /// against the **interior of the in-grid target box** (the desired orientation).
    /// Reading the target from the in-grid box is position-independent: it works
    /// regardless of where the box appears on the map or what color it uses.
    ///
    /// Comparison is **color-blind** (only shape/position, not color) and
    /// **scale-invariant** (via [`minimize_shape`]), so an upscaled UI piece
    /// correctly matches a 1× target shape.
    ///
    /// Returns `false` when either region cannot be found or contains no
    /// foreground pixels.
    ///
    /// # Time complexity: O(P) where P = pixels in each piece region
    /// # Space complexity: O(P)
    pub fn player_piece_matches_target(&self) -> bool {
        let (Some(bl), Some(tr)) = (self.bottom_left_piece(), self.target_box_inner_content())
        else {
            return false;
        };
        let bl_pixels = Self::extract_shape_color_blind(&bl);
        let tr_pixels = Self::extract_shape_color_blind(&tr);
        if bl_pixels.is_empty() || tr_pixels.is_empty() {
            return false;
        }
        let bl_shape = Self::minimize_shape(&bl_pixels);
        let tr_shape = Self::minimize_shape(&tr_pixels);
        bl_shape == tr_shape
    }

    /// Extracts a **colour-blind** normalised shape signature from a piece matrix.
    ///
    /// Identical to [`extract_shape`] except every foreground pixel is assigned
    /// the sentinel value `1` regardless of its actual colour. This allows two
    /// pieces of different colours but identical shape and rotation to compare
    /// as equal, the correct behaviour for orientation-match detection when the
    /// target box has been re-coloured by a novel object.
    ///
    /// # Time complexity: O(R × C) where R = rows, C = columns of the piece
    /// # Space complexity: O(P) where P = foreground pixels
    fn extract_shape_color_blind(piece: &[Vec<i64>]) -> Vec<(usize, usize, i64)> {
        const IGNORED: [i64; 5] = [0, 3, 4, 5, 8];

        let mut pixels: Vec<(usize, usize, i64)> = Vec::new();
        for (r, row) in piece.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                if v > 0 && !IGNORED.contains(&v) {
                    pixels.push((r, c, 1));
                }
            }
        }

        if pixels.is_empty() {
            return pixels;
        }

        let min_r = pixels.iter().map(|p| p.0).min().unwrap();
        let min_c = pixels.iter().map(|p| p.1).min().unwrap();
        for p in &mut pixels {
            p.0 -= min_r;
            p.1 -= min_c;
        }
        pixels.sort();
        pixels.dedup();
        pixels
    }

    /// Reduces a shape to its minimal scale-invariant representation.
    ///
    /// Detects if the extracted shape is composed of uniform `k x k` blocks of
    /// pixels (e.g. upscaled 2x or 3x). If so, it perfectly downsamples it by
    /// taking exactly one pixel per block, effectively normalizing the scale.
    /// Works backward from k=10 to find the largest uniform dividing block size.
    fn minimize_shape(pixels: &[(usize, usize, i64)]) -> Vec<(usize, usize, i64)> {
        if pixels.is_empty() {
            return vec![];
        }

        let mut pixels_map = std::collections::HashMap::new();
        for &(r, c, v) in pixels {
            pixels_map.insert((r, c), v);
        }

        for k in (2..=10).rev() {
            let mut valid = true;
            let mut blocks = std::collections::HashSet::new();

            for &(r, c, v) in pixels {
                blocks.insert((r / k, c / k, v));
            }

            if blocks.len() * (k * k) != pixels.len() {
                continue;
            }

            for &(br, bc, v) in &blocks {
                for dr in 0..k {
                    for dc in 0..k {
                        if pixels_map.get(&(br * k + dr, bc * k + dc)) != Some(&v) {
                            valid = false;
                            break;
                        }
                    }
                    if !valid {
                        break;
                    }
                }
                if !valid {
                    break;
                }
            }

            if valid {
                let mut new_pixels = Vec::new();
                for (br, bc, v) in blocks {
                    new_pixels.push((br, bc, v));
                }
                new_pixels.sort();
                return new_pixels;
            }
        }

        pixels.to_vec()
    }

    /// Detects **all** pixel coordinates of cross-shaped pattern clusters that
    /// could be modifier cells.
    ///
    /// Returns every valid candidate so the policy can filter out known pedals
    /// (which share a visually similar cross pattern but teleport the agent
    /// instead of rotating its piece).
    pub fn modifier_positions(&self) -> Vec<(usize, usize)> {
        let grid = self.grid_values();
        let (rows, cols) = self.grid_dims();
        if rows < 3 || cols < 3 {
            return Vec::new();
        }

        let mut clusters: Vec<Vec<(usize, usize)>> = Vec::new();
        for (r, row) in grid
            .iter()
            .enumerate()
            .take(rows.saturating_sub(10))
            .skip(1)
        {
            for (c, &v) in row.iter().enumerate().take(cols.saturating_sub(1)).skip(1) {
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
        }

        let player = self.player_pos().unwrap_or((usize::MAX, usize::MAX));
        let mut results = Vec::new();

        for cluster in clusters {
            let min_r = cluster.iter().map(|&(r, _)| r).min().unwrap();
            let max_r = cluster.iter().map(|&(r, _)| r).max().unwrap();
            let min_c = cluster.iter().map(|&(_, c)| c).min().unwrap();
            let max_c = cluster.iter().map(|&(_, c)| c).max().unwrap();

            if max_r.abs_diff(min_r) < 6 && max_c.abs_diff(min_c) < 6 {
                let is_player = cluster
                    .iter()
                    .any(|&(r, c)| r.abs_diff(player.1) < 4 && c.abs_diff(player.0) < 4);
                if is_player {
                    continue;
                }

                let all_same_row = cluster.iter().all(|&(r, _)| r == min_r);
                let all_same_col = cluster.iter().all(|&(_, c)| c == min_c);
                if all_same_row || all_same_col {
                    continue;
                }

                let sum_r: usize = cluster.iter().map(|&(r, _)| r).sum();
                let sum_c: usize = cluster.iter().map(|&(_, c)| c).sum();
                results.push((sum_c / cluster.len(), sum_r / cluster.len()));
            }
        }
        results
    }

    /// Detects all pixel coordinates of line-shaped patterns that represent teleport pedals.
    pub fn pedal_positions(&self) -> Vec<(usize, usize)> {
        let grid = self.grid_values();
        let (rows, cols) = self.grid_dims();
        if rows < 3 || cols < 3 {
            return Vec::new();
        }

        let mut clusters: Vec<Vec<(usize, usize)>> = Vec::new();
        for (r, row) in grid.iter().enumerate().take(rows.saturating_sub(4)).skip(1) {
            for (c, &v) in row.iter().enumerate().take(cols.saturating_sub(1)).skip(1) {
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
        }

        let player = self.player_pos().unwrap_or((usize::MAX, usize::MAX));
        let mut results = Vec::new();

        for cluster in clusters {
            let min_r = cluster.iter().map(|&(r, _)| r).min().unwrap();
            let max_r = cluster.iter().map(|&(r, _)| r).max().unwrap();
            let min_c = cluster.iter().map(|&(_, c)| c).min().unwrap();
            let max_c = cluster.iter().map(|&(_, c)| c).max().unwrap();

            if max_r.abs_diff(min_r) < 6 && max_c.abs_diff(min_c) < 6 {
                let is_player = cluster
                    .iter()
                    .any(|&(r, c)| r.abs_diff(player.1) < 4 && c.abs_diff(player.0) < 4);
                if is_player {
                    continue;
                }

                let all_same_row = cluster.iter().all(|&(r, _)| r == min_r);
                let all_same_col = cluster.iter().all(|&(_, c)| c == min_c);
                if all_same_row || all_same_col {
                    let sum_r: usize = cluster.iter().map(|&(r, _)| r).sum();
                    let sum_c: usize = cluster.iter().map(|&(_, c)| c).sum();
                    results.push((sum_c / cluster.len(), sum_r / cluster.len()));
                }
            }
        }
        results
    }

    /// Identifies the internal top-left coordinate of the destination target box matching a uniform value 3 border.
    pub fn target_pos(&self) -> Option<(usize, usize)> {
        let grid = self.grid_values();
        let (rows, cols) = self.grid_dims();
        let safe_rows = rows.saturating_sub(4);

        for s in (6..=25).rev() {
            for row in 0..=safe_rows.saturating_sub(s) {
                for col in 0..=cols.saturating_sub(s) {
                    let v = grid
                        .get(row)
                        .and_then(|r| r.get(col))
                        .copied()
                        .unwrap_or(-1);
                    if v <= 0 || v == 12 || v == 14 || v == 11 {
                        continue;
                    }

                    let mut valid = true;
                    for dc in 0..s {
                        let top = grid
                            .get(row)
                            .and_then(|r| r.get(col + dc))
                            .copied()
                            .unwrap_or(-1);
                        let bot = grid
                            .get(row + s - 1)
                            .and_then(|r| r.get(col + dc))
                            .copied()
                            .unwrap_or(-1);
                        if top != v || bot != v {
                            valid = false;
                            break;
                        }
                    }

                    if !valid {
                        continue;
                    }

                    for dr in 0..s {
                        let left = grid
                            .get(row + dr)
                            .and_then(|r| r.get(col))
                            .copied()
                            .unwrap_or(-1);
                        let right = grid
                            .get(row + dr)
                            .and_then(|r| r.get(col + s - 1))
                            .copied()
                            .unwrap_or(-1);
                        if left != v || right != v {
                            valid = false;
                            break;
                        }
                    }

                    if valid {
                        let mut inner_matches = 0;
                        let inner_area = (s - 2) * (s - 2);
                        for dr in 1..s - 1 {
                            for dc in 1..s - 1 {
                                let inner = grid
                                    .get(row + dr)
                                    .and_then(|r| r.get(col + dc))
                                    .copied()
                                    .unwrap_or(-1);
                                if inner == v {
                                    inner_matches += 1;
                                }
                            }
                        }
                        if inner_matches < inner_area / 2 {
                            return Some((col + 2, row + 2));
                        }
                    }
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

    /// Detects colorful multi-color novel objects in the grid that are distinct from all
    /// known game entities (player, modifier, bonuses, target border, background).
    ///
    /// A novel object is identified as a 3×3 cluster where at least three distinct
    /// positive pixel values appear (multi-colored pattern). These correspond to the
    /// colorful interactive objects first seen in level 2 of the game.
    ///
    /// # Time complexity: O(R × C) where R = rows, C = columns
    /// # Space complexity: O(K) where K = number of novel clusters found
    pub fn novel_object_positions(&self) -> Vec<(usize, usize)> {
        let grid = self.grid_values();
        let (rows, cols) = self.grid_dims();
        let safe_rows = rows.saturating_sub(4);
        let mut found = Vec::new();
        let mut tagged: HashSet<(usize, usize)> = HashSet::new();

        let excluded: [i64; 5] = [-1, 0, 3, 4, 5];

        for row in 0..safe_rows.saturating_sub(2) {
            for col in 0..cols.saturating_sub(2) {
                if tagged.contains(&(col, row)) {
                    continue;
                }
                let mut distinct_values: HashSet<i64> = HashSet::new();
                let mut fg_count = 0u32;
                for dr in 0..3 {
                    for dc in 0..3 {
                        let v = grid
                            .get(row + dr)
                            .and_then(|r| r.get(col + dc))
                            .copied()
                            .unwrap_or(-1);
                        if v > 0 && !excluded.contains(&v) {
                            distinct_values.insert(v);
                            fg_count += 1;
                        }
                    }
                }
                if fg_count >= 4 && distinct_values.len() >= 3 {
                    found.push((col + 1, row + 1));
                    for dr in 0..3 {
                        for dc in 0..3 {
                            tagged.insert((col + dc, row + dr));
                        }
                    }
                }
            }
        }
        found
    }

    /// Computes a hash of the target box pixel area for change-detection.
    ///
    /// Used by the policy to detect when touching a novel object causes the target
    /// box to change color - a cross-level learnable mechanic in level 2+.
    ///
    /// Returns `None` when no target position is visible.
    ///
    /// # Time complexity: O(1) amortised (fixed 8×8 window)
    /// # Space complexity: O(1)
    pub fn target_color_hash(&self) -> Option<u64> {
        let (tx, ty) = self.target_pos()?;
        let grid = self.grid_values();
        let (rows, cols) = self.grid_dims();
        let mut s = String::with_capacity(64);
        for dr in 0..8usize {
            let r = ty.saturating_sub(2) + dr;
            if r >= rows {
                break;
            }
            for dc in 0..8usize {
                let c = tx.saturating_sub(2) + dc;
                if c >= cols {
                    break;
                }
                let v = grid
                    .get(r)
                    .and_then(|row| row.get(c))
                    .copied()
                    .unwrap_or(-1);
                s.push((v.max(0) as u8 + b'0') as char);
            }
        }
        Some(fnv1a_hash(&s))
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
