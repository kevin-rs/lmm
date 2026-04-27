// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `ArcGameRunner` - outer game loop.
//!
//! Owns the ARC-AGI HTTP client, manages trial resets, calls `policy.decide()`
//! each frame, and assembles the final [`RunSummary`] after all trials complete.

use arc_agi_rs::client::Client;
use arc_agi_rs::models::{FrameData, GameState};
use arc_agi_rs::params::{MakeParams, ScorecardParams, StepParams};
use tracing::{info, warn};

use crate::config::AgentConfig;
use crate::display;
use crate::error::ArcAgentError;
use crate::frame::FrameContext;
use crate::policy::LmmPolicy;

/// Summary of a completed agent run returned from [`ArcGameRunner::run`].
#[derive(Debug)]
pub struct RunSummary {
    /// The game identifier used for this run.
    pub game_id: String,

    /// Terminal state when the run ended (`Win`, `GameOver`, or `NotFinished`).
    pub final_state: GameState,

    /// Number of levels successfully completed.
    pub levels_completed: u32,

    /// Total levels required to win.
    pub win_levels: u32,

    /// Total action steps consumed across all trials.
    pub steps_taken: usize,

    /// Scorecard identifier assigned by the server.
    pub card_id: String,
}

/// Game-loop orchestrator that drives [`LmmPolicy`] through a series of trials.
///
/// Construct with [`ArcGameRunner::new`] and drive with [`ArcGameRunner::run`].
pub struct ArcGameRunner {
    /// HTTP client connected to the ARC-AGI game server.
    client: Client,

    /// The navigation policy being evaluated.
    policy: LmmPolicy,

    /// Runtime configuration (server URL, trial budget, etc.).
    config: AgentConfig,
}

impl ArcGameRunner {
    /// Constructs a new runner from the given configuration.
    ///
    /// Initialises the server client and an empty [`LmmPolicy`].
    ///
    /// # Errors
    ///
    /// Returns [`ArcAgentError::Client`] when the HTTP client cannot be created.
    pub fn new(config: AgentConfig) -> Result<Self, ArcAgentError> {
        let client = Client::builder()
            .base_url(&config.base_url)
            .api_key(&config.api_key)
            .cookie_store(true)
            .build()?;

        Ok(Self {
            client,
            policy: LmmPolicy::new(),
            config,
        })
    }

    /// Runs the agent for up to `config.max_trials` trials and returns a [`RunSummary`].
    ///
    /// Each trial:
    /// 1. Resets the game session.
    /// 2. Steps through frames until a terminal state, step budget exhaustion, or
    ///    an implicit server reset (player sprite disappears).
    /// 3. Calls `policy.end_trial()` to finalise Q-table and reset trial state.
    ///
    /// # Errors
    ///
    /// Propagates [`ArcAgentError`] on client communication failures.
    pub async fn run(&mut self) -> Result<RunSummary, ArcAgentError> {
        let resolved_game_id = {
            let envs = self.client.list_environments().await.unwrap_or_default();
            let prefix = &self.config.game_id;
            envs.into_iter()
                .find(|e| e.game_id == *prefix || e.game_id.starts_with(&format!("{}-", prefix)))
                .map(|e| e.game_id)
                .unwrap_or_else(|| prefix.clone())
        };

        info!(resolved_game_id = %resolved_game_id, "Resolved full game_id from server");

        let card_id = self
            .client
            .open_scorecard(Some(
                ScorecardParams::new()
                    .tags(self.config.tags.clone())
                    .source_url("https://github.com/wiseaidotdev/lmm"),
            ))
            .await?;

        info!(card_id = %card_id, game_id = %resolved_game_id, "Scorecard opened");

        let mut final_frame: Option<FrameData> = None;
        let mut total_steps = 0usize;

        for trial in 0..self.config.max_trials {
            display::print_trial_start(trial, self.policy.epsilon(), self.policy.q_state_count());
            info!(
                trial = trial,
                max_trials = self.config.max_trials,
                epsilon = %format!("{:.4}", self.policy.epsilon()),
                q_states = self.policy.q_state_count(),
                "Starting trial"
            );

            let initial_frame = self
                .client
                .reset(MakeParams::new(&resolved_game_id, &card_id))
                .await?;

            let guid = initial_frame
                .guid
                .clone()
                .ok_or(ArcAgentError::UnexpectedSessionEnd)?;

            info!(
                trial = trial,
                guid = %guid,
                state = %initial_frame.state,
                "Trial session started"
            );

            let mut current_frame = initial_frame;
            let mut current_guid = guid;
            let mut trial_steps = 0usize;
            let mut prev_frame: Option<FrameData> = None;

            while trial_steps < self.config.max_actions {
                let ctx = FrameContext::new(&current_frame);

                if ctx.is_terminal() {
                    self.policy.record_terminal_reward(&ctx);
                    break;
                }

                // let available = current_frame.available_actions.clone();
                let spinner = display::think_spinner();
                let action_id = self.policy.decide(&ctx, prev_frame.as_ref()).await?;
                spinner.finish_silent();

                let step_params =
                    StepParams::new(&resolved_game_id, &card_id, &current_guid, action_id)
                        .reasoning(serde_json::json!({
                            "agent": "arc-lmm-agent",
                            "trial": self.policy.trial(),
                            "step": trial_steps,
                            "action_id": action_id,
                            "epsilon": self.policy.epsilon(),
                        }));

                let next_frame = self.client.step(step_params).await?;

                if let Some(ref new_guid) = next_frame.guid {
                    current_guid = new_guid.clone();
                }

                let server_action = next_frame.action_input.as_ref().map(|a| a.id).unwrap_or(0);

                let next_ctx = FrameContext::new(&next_frame);
                if ctx.player_pos().is_some()
                    && next_ctx.player_pos().is_none()
                    && next_frame.levels_completed == current_frame.levels_completed
                {
                    info!(
                        trial = trial,
                        steps = trial_steps,
                        "Implicit server reset detected (player wiped). Recording constraint and restarting loop natively."
                    );
                    self.policy.record_level_step_limit(trial_steps + 1);
                    self.policy.end_trial();
                    trial_steps = 0;
                    prev_frame = Some(current_frame);
                    current_frame = next_frame;
                    continue;
                }

                display::print_step_summary(
                    self.policy.trial(),
                    self.policy.step().saturating_sub(1),
                    total_steps + trial_steps,
                    action_id,
                    self.policy.epsilon(),
                    self.policy.q_state_count(),
                    next_frame.state.as_str(),
                    next_frame.levels_completed,
                    next_frame.win_levels,
                    server_action,
                );

                info!(
                    trial = self.policy.trial(),
                    step = self.policy.step(),
                    total_steps = total_steps + trial_steps,
                    action_id = action_id,
                    state = %next_frame.state,
                    levels_completed = next_frame.levels_completed,
                    win_levels = next_frame.win_levels,
                    "Step completed"
                );

                prev_frame = Some(current_frame);
                current_frame = next_frame;
                trial_steps += 1;
            }

            total_steps += trial_steps;

            if trial_steps >= self.config.max_actions {
                warn!(
                    trial = trial,
                    max_actions = self.config.max_actions,
                    "Action budget exhausted for trial"
                );
            }

            final_frame = Some(current_frame.clone());

            if current_frame.state == GameState::Win {
                info!(trial = trial, state = %current_frame.state, "Game Won.");
                self.policy.end_trial();
                break;
            } else if current_frame.state == GameState::GameOver {
                info!(
                    trial = trial,
                    steps = trial_steps,
                    "Trial exhausted steps (GameOver)"
                );
                self.policy.record_level_step_limit(trial_steps);
                self.policy.end_trial();
                continue;
            }

            self.policy.end_trial();
            display::print_trial_end(
                trial,
                trial_steps,
                self.policy.epsilon(),
                self.policy.q_state_count(),
            );
        }

        let last = final_frame.unwrap_or_default();
        let summary_state = last.state.clone();
        let summary_levels = last.levels_completed;
        let summary_win = last.win_levels;

        let _final_card = match self.client.close_scorecard(&card_id).await {
            Ok(card) => {
                info!(score = card.score, "Scorecard closed");
                Some(card)
            }
            Err(e) => {
                warn!(error = %e, "Failed to close scorecard");
                None
            }
        };

        Ok(RunSummary {
            game_id: self.config.game_id.clone(),
            final_state: summary_state,
            levels_completed: summary_levels,
            win_levels: summary_win,
            steps_taken: total_steps,
            card_id,
        })
    }
}
