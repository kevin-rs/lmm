// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `ArcAgentError` - unified error type.
//!
//! All fallible operations in `arc-lmm-agent` return this error via `?`.

use thiserror::Error;

/// The error type for all fallible operations in the ARC-LMM agent.
#[derive(Debug, Error)]
pub enum ArcAgentError {
    /// An error originating from the ARC-AGI HTTP client.
    #[error("ARC-AGI client error: {0}")]
    Client(#[from] arc_agi_rs::ArcAgiError),

    /// An error propagated from agent reasoning or knowledge ingestion.
    #[error("Agent reasoning error: {0}")]
    Reasoning(#[from] anyhow::Error),

    /// The available action list was empty and no action could be chosen.
    #[error("No valid action available in frame")]
    NoValidAction,

    /// The server session ended without reaching a terminal game state.
    #[error("Game session ended unexpectedly without a terminal state")]
    UnexpectedSessionEnd,
}
