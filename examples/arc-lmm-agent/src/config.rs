// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `AgentConfig` - CLI and runtime configuration.
//!
//! Parsed from command-line arguments via `clap`. The same struct can be
//! constructed programmatically (e.g. in tests) using [`AgentConfig::default`].

use clap::Parser;

/// Command-line configuration for the ARC-LMM agent.
///
/// Every flag has a sensible default so the agent can be run with zero
/// arguments against a local ARC-AGI server.
#[derive(Debug, Clone, Parser)]
#[command(
    name = "arc-lmm-agent",
    about = "ARC-AGI game agent powered by lmm equation-based intelligence."
)]
pub struct AgentConfig {
    /// ARC-AGI game identifier to play.
    #[arg(long, default_value = "ls20")]
    pub game_id: String,

    /// Base URL of the ARC-AGI server.
    #[arg(long, default_value = "http://localhost:8001")]
    pub base_url: String,

    /// API key for the ARC-AGI server. Leave empty for local offline mode.
    #[arg(long, default_value = "")]
    pub api_key: String,

    /// Maximum number of actions the agent may take per trial.
    #[arg(long, default_value_t = 500)]
    pub max_actions: usize,

    /// Number of trials (server resets) the agent is allowed.
    #[arg(long, default_value_t = 1)]
    pub max_trials: usize,

    /// Comma-separated scorecard tags submitted with the run.
    #[arg(long, default_value = "agent,lmm", value_delimiter = ',')]
    pub tags: Vec<String>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            game_id: "ls20".to_string(),
            base_url: "http://localhost:8001".to_string(),
            api_key: String::new(),
            max_actions: 500,
            max_trials: 1,
            tags: vec!["agent".to_string(), "lmm".to_string()],
        }
    }
}
