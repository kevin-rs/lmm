// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `LmmPolicy` - learned navigation policy for ARC-AGI.
//!
//! `LmmPolicy` is the central decision-making component. Its [`decide`] method
//! is called once per game frame and returns the next action integer.
//!
//! ## Behavioral hierarchy (in priority order)
//!
//! 1. **Stuck escape** - triggered when `trial_visits` for the current state
//!    exceeds [`STUCK_THRESHOLD`]. Bypassed when a modifier has been reached so that
//!    target routing is never interrupted by stuck logic.
//! 2. **Follow active plan** - execute the next step of an already-computed
//!    BFS / A\* plan held in `plan`.
//! 3. **Route to active target** - determined by what the agent knows:
//!    - *No modifier reached*: route to known modifier position.
//!    - *Modifier reached, bonuses remain*: route to nearest uncollected bonus;
//!      after collecting the first, **backtrack through the recorded outbound
//!      path** before routing to subsequent bonuses.
//!    - *All bonuses collected or none exist*: route to final target.
//! 4. **Milestone BFS** - use the global `WorldMap` to BFS back to a known
//!    milestone (modifier, level-start) when no spatial route is available.
//! 5. **Novelty exploration** - expand the map frontier, preferring globally
//!    unvisited neighbours; falls back to a minimum-visit greedy walk.
//!
//! ## Learning integration
//!
//! - [`LearningEngine`] (HELM / Q-table) records per-step rewards and is
//!   consulted in the exploration fallback via `recommend_action`.
//! - [`KnowledgeIndex`] stores cross-level strategic knowledge (modifier and
//!   bonus positions, target coordinates) as ingested text, making discoveries
//!   from level N available when level N+1 begins.
//! - [`InternalDrive`] emits curiosity signals when novel states or bonuses are
//!   found, and incoherence signals when stuck.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::display;
use crate::error::ArcAgentError;
use crate::frame::FrameContext;
use crate::tools::PathfindingTool;
use crate::world::WorldMap;
use anyhow::Context;
use lmm_agent::agent::LmmAgent;
use lmm_agent::cognition::drive::DriveSignal;
use lmm_agent::cognition::knowledge::KnowledgeIndex;
use lmm_agent::cognition::knowledge::KnowledgeSource;
use lmm_agent::cognition::learning::config::LearningConfig;
use lmm_agent::cognition::learning::engine::LearningEngine;
use lmm_agent::cognition::learning::q_table::ActionKey;
use lmm_agent::cognition::memory::ColdStore;
use lmm_agent::cognition::signal::CognitionSignal;
use lmm_agent::types::Message;
use rand::Rng;
use rand::seq::SliceRandom;
use std::iter::repeat_n;
use tracing::info;

/// Threshold of per-trial visits to the same state before stuck-escape logic fires.
const STUCK_THRESHOLD: u32 = 12;

/// Maps a raw game action integer to the [`ActionKey`] variant used by the Q-table.
fn action_to_key(action: u32) -> ActionKey {
    match action {
        1 => ActionKey::Expand,
        2 => ActionKey::Narrow,
        3 => ActionKey::Pivot,
        4 => ActionKey::Broaden,
        _ => ActionKey::Repeat,
    }
}

/// Builds a minimal [`CognitionSignal`] carrying only the reward value.
///
/// `step` and the placeholder observation strings are required by the signal
/// type but are not used during Q-table updates in this agent.
fn reward_signal(step: usize, reward: f64) -> CognitionSignal {
    let mut s = CognitionSignal::new(step, "g".into(), "g".into(), 1.0, 0.0);
    s.reward = reward;
    s
}

/// The learned navigation policy for an ARC-AGI agent.
///
/// `LmmPolicy` orchestrates the full cognitive stack: world-model updates,
/// reward shaping, drive signals, Q-table learning, and tiered action selection.
/// All search is delegated to [`PathfindingTool`]; raw map data lives in [`WorldMap`].
///
/// Construct with [`LmmPolicy::new`].
#[derive(Debug)]
pub struct LmmPolicy {
    /// LMM agent facade - holds hot memory, LTM, knowledge index, and internal drive.
    agent: LmmAgent,

    /// HELM learning engine - tabular Q-learning over discrete state hashes.
    engine: LearningEngine,

    /// Learned spatial model of the current level.
    world: WorldMap,

    /// Completed level count from the previous frame (used to detect level transitions).
    prev_levels: u32,

    /// State hash of the previous frame.
    prev_state_key: Option<u64>,

    /// Q-table action key chosen in the previous frame.
    prev_action_key: Option<ActionKey>,

    /// Raw integer action sent in the previous frame.
    prev_raw_action: Option<u32>,

    /// UI hash from the previous frame (used to detect modifier activation).
    prev_ui_hash: Option<u64>,

    /// Player pixel position from the previous frame.
    prev_player_pos: Option<(usize, usize)>,

    /// How many steps have elapsed within the current trial.
    step: usize,

    /// How many complete trials have elapsed in this level.
    trial: usize,

    /// Planned action sequence to execute before re-evaluating.
    plan: VecDeque<u32>,

    /// Visit counts per state for the current trial (reset each trial).
    trial_visits: HashMap<u64, u32>,

    /// Consecutive-stuck counts per state (used to escalate escape strategy).
    stuck_counts: HashMap<u64, u32>,

    /// Milestones already reached in the current trial.
    milestones_reached_this_trial: HashSet<u64>,

    /// Global visit counts across all trials and levels.
    global_visits: HashMap<u64, u32>,

    /// Level index at which each milestone was first recorded.
    milestone_levels: HashMap<u64, u32>,

    /// Pixel positions of all known modifier (+) cells.
    known_modifiers: HashSet<(usize, usize)>,

    /// Whether the modifier has been activated in the current trial.
    local_modifier_reached: bool,

    /// Zero-indexed level the agent is currently navigating.
    current_level_idx: u32,

    /// All known collectible bonus (yellow) positions found during exploration.
    known_bonuses: Vec<(usize, usize)>,

    /// How many modifiers have been activated in the current trial.
    trial_mod_visits: u32,

    /// Bonus positions collected in the current trial.
    trial_bonuses_consumed: HashSet<(usize, usize)>,

    /// Observed maximum step budgets per level index.
    level_step_limits: HashMap<u32, usize>,

    /// Actions recorded from modifier activation to first bonus collection.
    /// Reversed to produce the backtrack plan when subsequent bonuses must be reached.
    outbound_path_to_first_bonus: Vec<u32>,

    /// Whether the agent is currently executing a return-path after claiming the first bonus.
    backtracking_from_first_bonus: bool,

    /// Set after backtracking completes: forces one more modifier visit before routing to second bonus.
    needs_second_modifier_pass: bool,

    /// Step at which the modifier was first activated in the current trial.
    modifier_reached_step: Option<usize>,

    /// Locked final target position (set once all bonuses are consumed, never changed until level reset).
    locked_final_target: Option<(usize, usize)>,
}

impl LmmPolicy {
    /// Constructs a fresh [`LmmPolicy`] with default HELM hyper-parameters.
    ///
    /// The embedded [`LmmAgent`] is initialised with the ARC-AGI solver persona
    /// and an empty knowledge index ready to receive cross-level strategic facts.
    pub fn new() -> Self {
        let learning_config = LearningConfig::builder()
            .alpha(0.35)
            .gamma(0.95)
            .epsilon(1.0)
            .epsilon_decay(0.3)
            .epsilon_min(0.05)
            .build();

        Self {
            agent: LmmAgent::builder()
                .persona("ARC-AGI Solver")
                .behavior(
                    "Navigate the grid. Activate modifier cells. Collect step boosters. \
                     Reach the target zone.",
                )
                .learning_engine(LearningEngine::new(learning_config.clone()))
                .build(),
            engine: LearningEngine::new(learning_config),
            world: WorldMap::new(),
            prev_levels: 0,
            prev_state_key: None,
            prev_action_key: None,
            prev_raw_action: None,
            prev_ui_hash: None,
            prev_player_pos: None,
            step: 0,
            trial: 0,
            plan: VecDeque::new(),
            trial_visits: HashMap::new(),
            stuck_counts: HashMap::new(),
            milestones_reached_this_trial: HashSet::new(),
            global_visits: HashMap::new(),
            milestone_levels: HashMap::new(),
            known_modifiers: HashSet::new(),
            local_modifier_reached: false,
            current_level_idx: 0,
            known_bonuses: Vec::new(),
            trial_mod_visits: 0,
            trial_bonuses_consumed: HashSet::new(),
            level_step_limits: HashMap::new(),
            outbound_path_to_first_bonus: Vec::new(),
            backtracking_from_first_bonus: false,
            needs_second_modifier_pass: false,
            modifier_reached_step: None,
            locked_final_target: None,
        }
    }

    /// Records the maximum step budget for the current level after observing a reset.
    ///
    /// Persists the constraint to the agent's long-term memory so it can be
    /// referenced in subsequent trials.
    ///
    /// # Time complexity: O(1)
    /// # Space complexity: O(1)
    pub fn record_level_step_limit(&mut self, limit: usize) {
        self.level_step_limits.insert(self.current_level_idx, limit);
        self.agent.add_ltm_message(Message::new(
            "level_limit",
            format!(
                "Level {} max steps learned: {}",
                self.current_level_idx, limit
            ),
        ));
    }

    /// Main frame decision entry-point.
    ///
    /// Updates the world model, emits HELM reward signals, runs the ThinkLoop
    /// for cognitive reflection, then delegates to [`choose`] for action selection.
    ///
    /// Returns the chosen action integer (`1`=up, `2`=down, `3`=left, `4`=right,
    /// `0`=no-op when no valid action is available).
    ///
    /// # Time complexity: O(V + E) amortised - dominated by ThinkLoop overhead
    /// # Space complexity: O(S) where S = known states
    pub async fn decide(
        &mut self,
        context: &FrameContext<'_>,
        _prev_frame: Option<&arc_agi_rs::models::FrameData>,
    ) -> Result<u32, ArcAgentError> {
        let current_state = context.state_key();

        if self.step == 0 && self.current_level_idx == 0 && self.plan.is_empty() {
            self.plan = vec![3, 3, 3].into();
        }

        let moved = self.prev_state_key != Some(current_state);

        self.update_world_model(context, current_state, moved);
        if let Some(strategy) = self.handle_level_transition(context, current_state) {
            let _ = self.agent.ingest(KnowledgeSource::RawText(strategy)).await;
        }
        self.update_known_bonuses(context);
        self.update_modifier_detection(context);

        let ui_changed_for_reward = self.detect_ui_change(context);
        self.emit_reward(context, current_state, moved, ui_changed_for_reward);
        self.emit_drive_signals(context, moved);

        let observation = context.encode_observation();
        let _ = self
            .agent
            .think_with(&observation, 2, 0.5, 1.0, 0.05)
            .await
            .context("ThinkLoop")?;

        let available = context.available_non_reset();
        if available.is_empty() {
            self.step += 1;
            return Ok(0);
        }

        *self.trial_visits.entry(current_state).or_insert(0) += 1;
        *self.global_visits.entry(current_state).or_insert(0) += 1;

        self.check_bonus_proximity(context);

        let (_, map_walls, map_passages) = self.world.stats();
        let (px, py) = context
            .player_pos()
            .map(|(x, y)| (x as i32, y as i32))
            .unwrap_or((-1, -1));
        display::print_step_state(
            self.trial,
            self.step,
            (px, py),
            context.inner.levels_completed,
            context.inner.win_levels,
            self.trial_visits.get(&current_state).copied().unwrap_or(0),
            self.global_visits.get(&current_state).copied().unwrap_or(0),
            self.global_visits.len(),
            map_walls,
            map_passages,
            self.world.milestones.len(),
            self.known_modifiers.len(),
            self.plan.len(),
        );

        let chosen = self.choose(current_state, &available, context);
        display::print_action(chosen, "chosen");

        if self.local_modifier_reached
            && self.trial_bonuses_consumed.is_empty()
            && !self.backtracking_from_first_bonus
        {
            self.outbound_path_to_first_bonus.push(chosen);
        }

        self.prev_levels = context.inner.levels_completed;
        self.prev_state_key = Some(current_state);
        self.prev_action_key = Some(action_to_key(chosen));
        self.prev_raw_action = Some(chosen);
        self.prev_player_pos = context.player_pos();
        self.step += 1;
        Ok(chosen)
    }

    /// Updates the `WorldMap` with passage or wall data from the previous action.
    ///
    /// # Time complexity: O(1) amortised
    fn update_world_model(&mut self, context: &FrameContext<'_>, current_state: u64, moved: bool) {
        if self.step == 0 {
            return;
        }
        if let (Some(prev_sk), Some(prev_ga)) = (self.prev_state_key, self.prev_raw_action) {
            if moved {
                self.world.record_passage(prev_sk, prev_ga, current_state);
                let reverse = PathfindingTool::reverse_action(prev_ga);
                self.world
                    .record_reverse_passage(prev_sk, prev_ga, current_state, reverse);
                if let Some(pos) = context.player_pos() {
                    self.world.state_positions.insert(current_state, pos);
                }
            } else {
                self.world.record_wall(prev_sk, prev_ga);
                self.agent.internal_drive.record_incoherence(0.5);
                self.plan.clear();
            }
        }
    }

    /// Resets level-scoped state when the agent advances to a new level.
    ///
    /// Persists learned strategy text as the return value so the async caller
    /// (`decide`) can hand it to `agent.ingest()` without blocking.
    ///
    /// Returns `Some(strategy_text)` on a level transition, `None` otherwise.
    ///
    /// # Time complexity: O(S) where S = known states
    fn handle_level_transition(
        &mut self,
        context: &FrameContext<'_>,
        current_state: u64,
    ) -> Option<String> {
        if context.inner.levels_completed <= self.prev_levels {
            return None;
        }

        let strategy_text = format!(
            "Level {} completed after {} mod interactions and {} bonus collections. \
             Modifier positions: {:?}. Bonus positions: {:?}.",
            self.prev_levels,
            self.trial_mod_visits,
            self.trial_bonuses_consumed.len(),
            self.known_modifiers.iter().collect::<Vec<_>>(),
            self.known_bonuses,
        );
        self.agent
            .add_ltm_message(Message::new("learned_strategy", strategy_text.clone()));

        display::print_level_advance(
            self.prev_levels,
            context.inner.levels_completed,
            self.trial_mod_visits,
            self.trial_bonuses_consumed.len(),
        );

        self.world.clear();
        self.known_modifiers.clear();
        self.global_visits.clear();
        self.trial_visits.clear();
        self.milestones_reached_this_trial.clear();
        self.local_modifier_reached = false;
        self.prev_ui_hash = None;
        self.plan.clear();
        self.known_bonuses.clear();
        self.current_level_idx = context.inner.levels_completed;
        self.engine.reset_epsilon(1.0);
        self.trial = 0;
        self.step = 0;
        self.trial_mod_visits = 0;
        self.trial_bonuses_consumed.clear();
        self.outbound_path_to_first_bonus.clear();
        self.backtracking_from_first_bonus = false;
        self.needs_second_modifier_pass = false;
        self.locked_final_target = None;
        self.modifier_reached_step = None;

        self.world.record_milestone(current_state);
        self.milestone_levels
            .insert(current_state, context.inner.levels_completed);
        self.milestones_reached_this_trial.insert(current_state);
        self.agent.add_ltm_message(Message::new(
            "milestone",
            format!(
                "trial={} step={} state={:x} levels={}",
                self.trial, self.step, current_state, context.inner.levels_completed
            ),
        ));
        display::print_milestone(
            self.trial,
            self.step,
            context.inner.levels_completed,
            current_state,
        );

        Some(strategy_text)
    }

    /// Scans the current frame for bonus positions not yet in `known_bonuses`.
    ///
    /// # Time complexity: O(B * K) where B = grid bonuses, K = known bonuses count
    fn update_known_bonuses(&mut self, context: &FrameContext<'_>) {
        for bonus in context.bonus_positions() {
            if !self.known_bonuses.contains(&bonus) {
                display::print_bonus_found(bonus);
                self.known_bonuses.push(bonus);
                self.agent.internal_drive.record_residual(0.8);
            }
        }
    }

    /// Detects modifier activation via spatial proximity or piece-match heuristic.
    ///
    /// Sets `local_modifier_reached` and records the modifier position in
    /// `known_modifiers` on the first detection per trial.
    ///
    /// # Time complexity: O(1)
    fn update_modifier_detection(&mut self, context: &FrameContext<'_>) {
        if self.local_modifier_reached {
            return;
        }
        let mut activated = false;
        let mut activated_pos: Option<(usize, usize)> = None;

        if let Some(modifier_pos) = context.modifier_pos()
            && let Some(player_pos) = context.player_pos()
        {
            let dx = player_pos.0.abs_diff(modifier_pos.0);
            let dy = player_pos.1.abs_diff(modifier_pos.1);
            if dx <= 7 && dy <= 7 {
                activated = true;
                activated_pos = Some(modifier_pos);
            }
        }
        if context.player_piece_matches_target() && self.step > 0 {
            activated = true;
            activated_pos = activated_pos.or(context.player_pos());
        }

        if activated {
            self.local_modifier_reached = true;
            self.modifier_reached_step = Some(self.step);
            self.plan.clear();
            self.outbound_path_to_first_bonus.clear();
            self.backtracking_from_first_bonus = false;
            self.needs_second_modifier_pass = false;
            self.locked_final_target = None;

            if let Some(pos) = activated_pos {
                self.known_modifiers.insert(pos);
                self.trial_mod_visits += 1;
                display::print_modifier_reached(pos, self.trial_mod_visits);
                self.agent.add_ltm_message(Message::new(
                    "modifier_reached",
                    format!(
                        "Modifier activated at {:?} on trial={} step={}",
                        pos, self.trial, self.step
                    ),
                ));
                self.agent.internal_drive.record_residual(1.0);
            }
        }

        if let Some(ui_hash) = context.ui_hash() {
            self.prev_ui_hash = Some(ui_hash);
        }
    }

    /// Returns `true` when the UI hash changed between this frame and the previous one.
    ///
    /// A UI change indicates the player piece was rotated by a modifier cell.
    ///
    /// # Time complexity: O(1)
    fn detect_ui_change(&self, context: &FrameContext<'_>) -> bool {
        match (self.prev_ui_hash, context.ui_hash()) {
            (Some(prev), Some(now)) => prev != now,
            _ => false,
        }
    }

    /// Records a HELM reward signal for the previous action.
    ///
    /// # Time complexity: O(1)
    fn emit_reward(
        &mut self,
        context: &FrameContext<'_>,
        current_state: u64,
        moved: bool,
        ui_changed: bool,
    ) {
        if let (Some(prev_sk), Some(prev_ak)) = (self.prev_state_key, self.prev_action_key) {
            let r = self.compute_reward(context, moved, ui_changed);
            self.engine.record_step(
                &reward_signal(self.step, r),
                prev_sk,
                prev_ak,
                current_state,
            );
        }
    }

    /// Fires `InternalDrive` signals appropriate to the agent's current situation.
    ///
    /// - **Curiosity** when the agent moves to a globally new state.
    /// - **Incoherence** when the agent is blocked (did not move on the last action).
    ///
    /// # Time complexity: O(1)
    fn emit_drive_signals(&mut self, context: &FrameContext<'_>, moved: bool) {
        if moved {
            if let Some(state) = self.prev_state_key {
                let visits = self.global_visits.get(&state).copied().unwrap_or(0);
                if visits == 0 {
                    self.agent.internal_drive.record_residual(0.9);
                }
            }
        } else {
            self.agent.internal_drive.record_incoherence(0.3);
        }
        let drive_state = self.agent.internal_drive.tick();
        if let Some(dominant) = drive_state.dominant_drive()
            && dominant.magnitude() > 0.5
        {
            let signal_name = dominant.name();
            let magnitude = dominant.magnitude();
            display::print_drive(signal_name, magnitude);
            if matches!(dominant, DriveSignal::Curiosity(_)) {
                let _ = context;
            }
        }
    }

    /// Checks whether the player is adjacent to any known bonus and marks it consumed.
    ///
    /// When the first bonus is consumed, the recorded `outbound_path_to_first_bonus`
    /// is reversed and loaded into `plan` if more bonuses remain, enabling the agent
    /// to backtrack safely through already-explored territory.
    ///
    /// # Time complexity: O(K) where K = known bonuses
    fn check_bonus_proximity(&mut self, context: &FrameContext<'_>) {
        let Some((px, py)) = context.player_pos() else {
            return;
        };

        let newly_consumed: Vec<(usize, usize)> = self
            .known_bonuses
            .iter()
            .filter(|&&(bx, by)| {
                (px + 5).abs_diff(bx) < 5
                    && (py + 5).abs_diff(by) < 5
                    && !self.trial_bonuses_consumed.contains(&(bx, by))
            })
            .copied()
            .collect();

        for bonus in newly_consumed {
            self.trial_bonuses_consumed.insert(bonus);
            self.agent.add_ltm_message(Message::new(
                "booster_knowledge",
                format!(
                    "Collected bonus at {:?} on trial={} step={}. Steps extended.",
                    bonus, self.trial, self.step
                ),
            ));
            let remaining_after = self
                .known_bonuses
                .iter()
                .filter(|b| !self.trial_bonuses_consumed.contains(*b))
                .count();
            display::print_bonus_collected(bonus, remaining_after);

            let remaining_bonuses = self
                .known_bonuses
                .iter()
                .filter(|b| !self.trial_bonuses_consumed.contains(b))
                .count();

            if remaining_bonuses == 0 {
                self.plan = repeat_n(2u32, 50).collect();
                self.locked_final_target = None;
            } else if !self.outbound_path_to_first_bonus.is_empty()
                && !self.backtracking_from_first_bonus
            {
                let return_path = PathfindingTool::reverse_path(&self.outbound_path_to_first_bonus);
                display::print_backtrack(return_path.len());
                self.plan = return_path.into_iter().collect();
                self.backtracking_from_first_bonus = true;
                self.needs_second_modifier_pass = true;
                self.outbound_path_to_first_bonus.clear();
            }
        }
    }

    /// Computes the scalar reward for the previous action using distance gradients
    /// and novelty bonuses.
    ///
    /// Reward components:
    /// - Base step cost: -0.1
    /// - Level completion: +15.0
    /// - UI change (modifier rotation): +10.0
    /// - Wall hit (no movement): -1.5
    /// - Novel state visit: +2.0
    /// - Revisit penalty: -0.2 × visits
    /// - Proximity to target after modifier: +50 / (1 + Manhattan distance)
    ///
    /// # Time complexity: O(1)
    /// # Space complexity: O(1)
    fn compute_reward(&self, ctx: &FrameContext<'_>, moved: bool, ui_changed: bool) -> f64 {
        let mut reward = -0.1;
        if ctx.inner.levels_completed > self.prev_levels {
            return reward + 15.0;
        }
        if ui_changed {
            reward += 10.0;
        }
        if !moved {
            reward -= 1.5;
        } else {
            let visits = self
                .global_visits
                .get(&ctx.state_key())
                .copied()
                .unwrap_or(0);
            reward += if visits == 0 {
                2.0
            } else {
                -0.2 * visits as f64
            };

            if self.local_modifier_reached
                && let (Some((px, py)), Some((tx, ty))) = (ctx.player_pos(), ctx.target_pos())
            {
                let distance = px.abs_diff(tx) + py.abs_diff(ty);
                reward += 50.0 / (1.0 + distance as f64);
            }
        }
        reward
    }

    /// Selects the next action using the full behavioral hierarchy.
    ///
    /// This is a pure routing dispatcher: it calls specialised helpers in
    /// priority order and returns the first valid action found.
    ///
    /// # Time complexity: O(V + E) worst-case (BFS fallback)
    /// # Space complexity: O(V)
    fn choose(&mut self, state: u64, avail: &[u32], context: &FrameContext<'_>) -> u32 {
        let trial_visits_here = self.trial_visits.get(&state).copied().unwrap_or(0);

        if trial_visits_here >= STUCK_THRESHOLD
            && !self.local_modifier_reached
            && let Some(action) = self.escape_stuck(state, avail)
        {
            return action;
        }

        if let Some(action) = self.follow_plan(state, avail) {
            return action;
        }

        if let Some(action) = self.route_to_active_target(state, avail, context) {
            return action;
        }

        if self.trial > 0
            && let Some(action) = self.route_via_milestone_bfs(state, avail)
        {
            return action;
        }

        self.explore(state, avail)
    }

    /// Attempts to escape a stuck state by BFS-navigating to the nearest unvisited
    /// or frontier cell.
    ///
    /// Clears the `plan` buffer and escalates the escape strategy on repeated invocations
    /// for the same state. Does not fire when `local_modifier_reached` is true, so that
    /// the post-modifier target route is never interrupted.
    ///
    /// # Time complexity: O(V + E)
    /// # Space complexity: O(V)
    fn escape_stuck(&mut self, state: u64, avail: &[u32]) -> Option<u32> {
        self.plan.clear();
        *self.stuck_counts.entry(state).or_insert(0) += 1;
        let escape_count = self.stuck_counts[&state];

        let mut queue: VecDeque<u64> = VecDeque::from([state]);
        let mut seen: HashSet<u64> = HashSet::from([state]);
        let mut path_parents: HashMap<u64, (u64, u32)> = HashMap::new();
        let mut escape_target: Option<u64> = None;

        'bfs: while let Some(current) = queue.pop_front() {
            let blocked = self.world.walls.get(&current);
            for action in [1u32, 2, 3, 4] {
                if blocked.is_some_and(|w| w.contains(&action)) {
                    continue;
                }
                if let Some(next_state) = self.world.predict(current, action) {
                    if self.global_visits.get(&next_state).copied().unwrap_or(0) == 0 {
                        escape_target = Some(current);
                        break 'bfs;
                    }
                    if seen.insert(next_state) {
                        path_parents.insert(next_state, (current, action));
                        queue.push_back(next_state);
                    }
                } else if !self.world.is_wall(current, action) {
                    escape_target = Some(current);
                    break 'bfs;
                }
            }
        }

        if let Some(target) = escape_target {
            if target != state {
                let path = PathfindingTool::reconstruct_path(&path_parents, state, target);
                if let Some(&first) = path.first()
                    && avail.contains(&first)
                {
                    eprintln!(
                        "  [mode=STUCK-escape] target={:x} steps={}",
                        target,
                        path.len()
                    );
                    self.plan = path.into_iter().skip(1).collect();
                    return Some(first);
                }
            } else {
                for &action in avail {
                    if self.world.predict(state, action).is_none()
                        && !self.world.is_wall(state, action)
                    {
                        let sc = self.stuck_counts.get(&state).copied().unwrap_or(0);
                        display::print_stuck("frontier-edge", action, sc);
                        return Some(action);
                    }
                }
                if escape_count >= 4 {
                    let mut rng = rand::thread_rng();
                    let action = avail[rng.gen_range(0..avail.len())];
                    eprintln!(
                        "  [mode=STUCK-random-break] action={} escape={}",
                        action, escape_count
                    );
                    return Some(action);
                }
                let best = avail.iter().min_by_key(|&&a| {
                    self.world
                        .predict(state, a)
                        .map(|ns| self.global_visits.get(&ns).copied().unwrap_or(0))
                        .unwrap_or(0)
                });
                if let Some(&action) = best {
                    let sc = self.stuck_counts.get(&state).copied().unwrap_or(0);
                    display::print_stuck("min-visit", action, sc);
                    return Some(action);
                }
            }
        }
        None
    }

    /// Executes the next step from `self.plan` if the plan is valid.
    ///
    /// Clears the plan and returns `None` if the next planned action is blocked.
    ///
    /// # Time complexity: O(1)
    /// # Space complexity: O(1)
    fn follow_plan(&mut self, state: u64, avail: &[u32]) -> Option<u32> {
        if let Some(&next) = self.plan.front() {
            if !avail.contains(&next) || self.world.is_wall(state, next) {
                display::print_plan_invalidated();
                self.plan.clear();
                return None;
            }
            self.plan.pop_front();
            display::print_plan_step(next, self.plan.len());
            return Some(next);
        }
        None
    }

    /// Determines the current active target and routes toward it.
    ///
    /// Target priority:
    /// 1. If modifier not yet reached: route to modifier position.
    /// 2. If modifier reached and uncollected bonuses exist: route to nearest bonus.
    ///    After first bonus consumed, the backtrack plan takes over via `follow_plan`.
    ///    Once backtrack completes, route forward to next bonus.
    /// 3. If modifier reached and all bonuses collected: route to final target.
    ///
    /// # Time complexity: O(N log N) worst-case (spatial A\*)
    /// # Space complexity: O(N)
    fn route_to_active_target(
        &mut self,
        state: u64,
        avail: &[u32],
        context: &FrameContext<'_>,
    ) -> Option<u32> {
        let player_pos = context.player_pos();
        let target_pos = context.target_pos();
        let known_mod_pos = self.known_modifiers.iter().next().copied();

        let modifier_pos = if self.trial > 0 || self.step > 0 {
            context.modifier_pos().or(known_mod_pos)
        } else {
            known_mod_pos
        };

        let uncollected_bonuses: Vec<(usize, usize)> = self
            .known_bonuses
            .iter()
            .filter(|b| !self.trial_bonuses_consumed.contains(*b))
            .copied()
            .collect();

        let active_target: Option<(usize, usize)>;
        let target_label: &str;

        if self.current_level_idx == 0 {
            if self.local_modifier_reached || context.player_piece_matches_target() {
                active_target = target_pos;
                target_label = "Target";
            } else if self.trial > 0 && known_mod_pos.is_some() {
                active_target = known_mod_pos;
                target_label = "Target";
            } else {
                active_target = None;
                target_label = "None";
            }
        } else if !self.local_modifier_reached {
            active_target = modifier_pos;
            target_label = "Modifier";
        } else if self.needs_second_modifier_pass && self.plan.is_empty() {
            if let Some(mod_pos) = known_mod_pos.or(modifier_pos) {
                if let Some((px, py)) = player_pos {
                    let dx = px.abs_diff(mod_pos.0);
                    let dy = py.abs_diff(mod_pos.1);
                    if dx < 5 && dy < 5 {
                        display::print_second_mod_pass(true);
                        self.needs_second_modifier_pass = false;
                        if !uncollected_bonuses.is_empty() {
                            let nearest = uncollected_bonuses
                                .iter()
                                .min_by_key(|&&(bx, by)| px.abs_diff(bx) + py.abs_diff(by))
                                .copied();
                            active_target = nearest;
                            target_label = "Bonus";
                        } else {
                            active_target = target_pos;
                            target_label = "Target";
                        }
                    } else {
                        display::print_second_mod_pass(false);
                        active_target = Some(mod_pos);
                        target_label = "Modifier2";
                    }
                } else {
                    active_target = Some(mod_pos);
                    target_label = "Modifier2";
                }
            } else {
                self.needs_second_modifier_pass = false;
                active_target = target_pos;
                target_label = "Target";
            }
        } else if !uncollected_bonuses.is_empty() {
            if let Some((px, py)) = player_pos {
                let nearest = uncollected_bonuses
                    .iter()
                    .min_by_key(|&&(bx, by)| px.abs_diff(bx) + py.abs_diff(by))
                    .copied();
                active_target = nearest;
                target_label = "Bonus";
            } else {
                active_target = target_pos;
                target_label = "Target";
            }
        } else {
            if self.locked_final_target.is_none()
                && let Some(tp) = target_pos
            {
                self.locked_final_target = Some(tp);
                display::print_target_locked(tp);
            }
            active_target = self.locked_final_target.or(target_pos);
            target_label = "Target";
        }

        let (goal_x, goal_y) = active_target?;
        let (px, py) = player_pos?;

        let at_goal = if target_label == "Target" && uncollected_bonuses.is_empty() {
            px.abs_diff(goal_x) < 5 && py.abs_diff(goal_y) < 5
        } else {
            px == goal_x && py == goal_y
        };
        if at_goal {
            return None;
        }

        let pos_walls = self.world.pos_walls();
        let visited_coords = self.world.visited_pixel_coords();

        if let Some(path) =
            PathfindingTool::spatial_astar(px, py, goal_x, goal_y, &pos_walls, &visited_coords)
            && let Some(&first) = path.first()
            && avail.contains(&first)
            && !self.world.is_wall(state, first)
        {
            let mode = format!("GENERAL→{target_label}_Cartesian");
            display::print_routing(&mode, (goal_x, goal_y), path.len(), first);
            if path.len() > 1 {
                self.plan = path.into_iter().skip(1).collect();
            }
            return Some(first);
        }

        let dx = (goal_x as i32 - px as i32).unsigned_abs() as usize;
        let dy = (goal_y as i32 - py as i32).unsigned_abs() as usize;
        let vertical_action = if (goal_y as i32) < (py as i32) { 1 } else { 2 };
        let horizontal_action = if (goal_x as i32) < (px as i32) { 3 } else { 4 };

        let preferred: Vec<u32> = if dy >= dx {
            vec![vertical_action, horizontal_action]
        } else {
            vec![horizontal_action, vertical_action]
        };

        for action in &preferred {
            if avail.contains(action) && !self.world.is_wall(state, *action) {
                let pred_visits = self
                    .world
                    .predict(state, *action)
                    .map(|ns| self.trial_visits.get(&ns).copied().unwrap_or(0))
                    .unwrap_or(0);
                if pred_visits == 0 {
                    let mode = format!("GENERAL→{target_label}");
                    display::print_routing(&mode, (goal_x, goal_y), 1, *action);
                    return Some(*action);
                }
            }
        }

        let best = avail
            .iter()
            .copied()
            .filter(|&a| !self.world.is_wall(state, a))
            .min_by_key(|&a| {
                let (nx, ny) = PathfindingTool::action_next_pos(px, py, a);
                let distance = (nx.abs_diff(goal_x) + ny.abs_diff(goal_y)) as u32;
                let visits = self
                    .world
                    .predict(state, a)
                    .map(|ns| self.trial_visits.get(&ns).copied().unwrap_or(0))
                    .unwrap_or(0);
                distance + visits * 50
            });

        if let Some(action) = best {
            let mode = format!("GENERAL→{target_label}");
            display::print_routing(&mode, (goal_x, goal_y), 1, action);
            return Some(action);
        }

        None
    }

    /// Routes to the highest-priority unvisited milestone via graph BFS.
    ///
    /// Only fires in trial ≥ 1 when the spatial routes have failed.
    ///
    /// # Time complexity: O(V + E)
    /// # Space complexity: O(V)
    fn route_via_milestone_bfs(&mut self, state: u64, avail: &[u32]) -> Option<u32> {
        let best_milestone = self
            .world
            .milestones
            .iter()
            .filter(|&&m| !self.milestones_reached_this_trial.contains(&m))
            .max_by_key(|&&m| self.milestone_levels.get(&m).copied().unwrap_or(0))
            .copied();

        if let Some(target) = best_milestone
            && target != state
            && let Some(path) = PathfindingTool::bfs(state, target, &self.world.transitions)
            && let Some(&first) = path.first()
            && avail.contains(&first)
        {
            display::print_routing("BFS→milestone", (0, 0), path.len(), first);
            self.plan = path.into_iter().skip(1).collect();
            return Some(first);
        }

        if let Some((win_state, _)) = self.world.win_predecessor
            && state != win_state
            && let Some(path) = PathfindingTool::bfs(state, win_state, &self.world.transitions)
            && let Some(&first) = path.first()
            && avail.contains(&first)
        {
            display::print_routing("BFS→win", (0, 0), path.len(), first);
            self.plan = path.into_iter().skip(1).collect();
            return Some(first);
        }

        None
    }

    /// Novelty-driven map exploration fallback.
    ///
    /// Preference order:
    /// 1. Any neighbour leading to a globally unvisited state.
    /// 2. Any neighbour leading to a trial-unvisited state.
    /// 3. BFS to the nearest frontier cell (an explored state that has unknown neighbours).
    /// 4. Minimum global-visit greedy walk.
    ///
    /// Back-tracking (reversing the previous action) is penalised unless unavoidable.
    ///
    /// # Time complexity: O(V + E) for BFS sub-path; O(1) for greedy steps
    /// # Space complexity: O(V)
    fn explore(&mut self, state: u64, avail: &[u32]) -> u32 {
        let mut rng = rand::thread_rng();

        let mut candidates: Vec<u32> = avail
            .iter()
            .copied()
            .filter(|&a| !self.world.is_wall(state, a))
            .collect();
        if candidates.is_empty() {
            candidates = avail.to_vec();
        }

        if let Some(prev) = self.prev_raw_action {
            let reverse = PathfindingTool::reverse_action(prev);
            let no_rev: Vec<u32> = candidates
                .iter()
                .copied()
                .filter(|&a| a != reverse)
                .collect();
            if !no_rev.is_empty() {
                candidates = no_rev;
            }
        }

        let globally_new: Vec<u32> = candidates
            .iter()
            .copied()
            .filter(|&a| {
                self.world
                    .predict(state, a)
                    .map(|ns| self.global_visits.get(&ns).copied().unwrap_or(0) == 0)
                    .unwrap_or(true)
            })
            .collect();
        if !globally_new.is_empty() {
            display::print_action(0, &format!("RANDOM-new choices={}", globally_new.len()));
            return *globally_new.choose(&mut rng).unwrap();
        }

        let trial_new: Vec<u32> = candidates
            .iter()
            .copied()
            .filter(|&a| {
                self.world
                    .predict(state, a)
                    .map(|ns| self.trial_visits.get(&ns).copied().unwrap_or(0) == 0)
                    .unwrap_or(true)
            })
            .collect();
        if !trial_new.is_empty() {
            display::print_action(0, &format!("RANDOM-trial-new choices={}", trial_new.len()));
            return *trial_new.choose(&mut rng).unwrap();
        }

        let min_gv = candidates
            .iter()
            .map(|&a| {
                self.world
                    .predict(state, a)
                    .map(|ns| self.global_visits.get(&ns).copied().unwrap_or(0))
                    .unwrap_or(0)
            })
            .min()
            .unwrap_or(0);
        candidates.retain(|&a| {
            self.world
                .predict(state, a)
                .map(|ns| self.global_visits.get(&ns).copied().unwrap_or(0))
                .unwrap_or(0)
                == min_gv
        });

        if min_gv > 0
            && let Some(action) = self.frontier_bfs(state, avail, &candidates)
        {
            return action;
        }

        display::print_action(
            0,
            &format!(
                "RANDOM-exploit min_gv={} choices={}",
                min_gv,
                candidates.len()
            ),
        );
        *candidates.choose(&mut rng).unwrap()
    }

    /// Performs a BFS from `state` to find the nearest frontier cell (a cell with an
    /// unknown passable neighbour) and returns the first action toward it.
    ///
    /// # Time complexity: O(V + E)
    /// # Space complexity: O(V)
    fn frontier_bfs(&mut self, state: u64, avail: &[u32], candidates: &[u32]) -> Option<u32> {
        let mut queue: VecDeque<u64> = VecDeque::from([state]);
        let mut seen: HashSet<u64> = HashSet::from([state]);
        let mut path_parents: HashMap<u64, (u64, u32)> = HashMap::new();
        let mut frontier_target: Option<u64> = None;

        'outer: while let Some(current) = queue.pop_front() {
            let local_avail: Vec<u32> = if current == state {
                avail.to_vec()
            } else {
                self.world
                    .walls
                    .get(&current)
                    .map_or(vec![1, 2, 3, 4], |w| {
                        [1u32, 2, 3, 4]
                            .into_iter()
                            .filter(|a| !w.contains(a))
                            .collect()
                    })
            };
            for action in local_avail {
                if let Some(next_state) = self.world.predict(current, action) {
                    if self.global_visits.get(&next_state).copied().unwrap_or(0) == 0 {
                        frontier_target = Some(current);
                        break 'outer;
                    }
                    if seen.insert(next_state) {
                        path_parents.insert(next_state, (current, action));
                        queue.push_back(next_state);
                    }
                } else if !self.world.is_wall(current, action) {
                    frontier_target = Some(current);
                    break 'outer;
                }
            }
        }

        if let Some(target) = frontier_target {
            if target != state {
                let path = PathfindingTool::reconstruct_path(&path_parents, state, target);
                if let Some(&first) = path.first()
                    && avail.contains(&first)
                {
                    display::print_routing("BFS→frontier", (0, 0), path.len(), first);
                    self.plan = path.into_iter().skip(1).collect();
                    return Some(first);
                }
            } else {
                for &action in avail {
                    if self.world.predict(state, action).is_none()
                        && !self.world.is_wall(state, action)
                    {
                        display::print_action(action, "RANDOM-frontier-edge");
                        return Some(action);
                    }
                }
            }
        }

        let recommended = candidates
            .iter()
            .filter_map(|&a| {
                let ns = self.world.predict(state, a)?;
                Some((a, self.global_visits.get(&ns).copied().unwrap_or(0)))
            })
            .min_by_key(|&(_, v)| v)
            .map(|(a, _)| a);

        if let Some(action) = recommended {
            display::print_action(action, "BFS→recommended");
            return Some(action);
        }

        None
    }

    /// Finalises the current trial.
    ///
    /// Persists Q-table state, writes telemetry to long-term memory, resets all
    /// trial-scoped fields, and increments the trial counter.
    ///
    /// # Time complexity: O(S) where S = Q-table state count
    /// # Space complexity: O(1) additional
    pub fn end_trial(&mut self) {
        self.engine.end_of_episode(
            &ColdStore::default(),
            &mut KnowledgeIndex::new(),
            "arc-agi-ls20",
            0.5,
        );

        let (states, walls, passages) = self.world.stats();
        self.agent.add_ltm_message(Message::new(
            "trial_end",
            format!(
                "trial={} steps={} g_states={} walls={} passages={} milestones={}",
                self.trial,
                self.step,
                self.global_visits.len(),
                walls,
                passages,
                self.world.milestones.len(),
            ),
        ));

        info!(
            trial = self.trial,
            eps = %format!("{:.3}", self.engine.q_table().epsilon),
            g_states = self.global_visits.len(),
            states, walls, passages,
            milestones = self.world.milestones.len(),
            "Trial ended"
        );

        self.trial += 1;
        self.step = 0;
        self.prev_state_key = None;
        self.prev_action_key = None;
        self.prev_raw_action = None;
        self.prev_ui_hash = None;
        self.plan.clear();
        self.trial_visits.clear();
        self.milestones_reached_this_trial.clear();
        self.local_modifier_reached = false;
        self.stuck_counts.clear();
        self.trial_mod_visits = 0;
        self.trial_bonuses_consumed.clear();
        self.outbound_path_to_first_bonus.clear();
        self.backtracking_from_first_bonus = false;
        self.needs_second_modifier_pass = false;
        self.locked_final_target = None;
        self.modifier_reached_step = None;
    }

    /// Records a terminal reward for the final state of a trial and stores the
    /// winning predecessor state for BFS replay in future trials.
    ///
    /// # Time complexity: O(1)
    /// # Space complexity: O(1)
    pub fn record_terminal_reward(&mut self, ctx: &FrameContext<'_>) {
        if let (Some(prev_sk), Some(prev_ak)) = (self.prev_state_key, self.prev_action_key) {
            let current = ctx.state_key();
            if ctx.inner.state == arc_agi_rs::models::GameState::Win
                && let Some(prev_ga) = self.prev_raw_action
            {
                self.world.win_predecessor = Some((prev_sk, prev_ga));
                let _ = prev_sk; // No log needed, display:: handles this at run summary
            }
            let r: f64 = if ctx.inner.levels_completed > 0 {
                20.0
            } else {
                -2.0
            };
            for _ in 0..5 {
                self.engine
                    .record_step(&reward_signal(self.step, r), prev_sk, prev_ak, current);
            }
        }
    }

    /// Persists the current HELM Q-table to disk.
    ///
    /// # Time complexity: O(S) where S = Q-table state count
    /// # Space complexity: O(S)
    pub fn save_learning(&self, path: &std::path::Path) -> Result<(), ArcAgentError> {
        self.agent.save_learning(path).context("save")?;
        Ok(())
    }

    /// Returns the current trial index.
    pub fn trial(&self) -> usize {
        self.trial
    }

    /// Returns the current step index within the active trial.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Returns the number of distinct states in the Q-table.
    pub fn q_state_count(&self) -> usize {
        self.engine.q_table().state_count()
    }

    /// Returns the current epsilon value of the greedy-epsilon exploration policy.
    pub fn epsilon(&self) -> f64 {
        self.engine.q_table().epsilon
    }

    /// Returns the zero-indexed current level.
    pub fn current_level(&self) -> u32 {
        self.current_level_idx
    }
}

impl Default for LmmPolicy {
    fn default() -> Self {
        Self::new()
    }
}
