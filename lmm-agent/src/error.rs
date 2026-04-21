// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Error types for `lmm-agent`.

use thiserror::Error;

/// Errors that can occur when building an [`crate::agent::LmmAgent`] via its
/// builder.
#[derive(Debug, Error)]
pub enum AgentBuildError {
    /// The `persona` field was not provided to the builder.
    #[error("The `persona` field is required but was not set.")]
    MissingPersona,

    /// The `behavior` field was not provided to the builder.
    #[error("The `behavior` field is required but was not set.")]
    MissingBehavior,

    /// A field value failed validation (e.g. empty string where one is required).
    #[error("Field `{field}` failed validation: {reason}")]
    InvalidField { field: &'static str, reason: String },
}

/// Errors that can occur during agent execution.
#[derive(Debug, Error)]
pub enum AgentError {
    /// The agent has no assigned tasks.
    #[error("Agent has no tasks to execute.")]
    NoTasks,

    /// A DuckDuckGo search failed.
    #[error("Search error: {0}")]
    Search(#[from] anyhow::Error),

    /// Internal execution error with a custom message.
    #[error("Execution failed: {0}")]
    Execution(String),
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
