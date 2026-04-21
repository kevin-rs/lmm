// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `Functions`, `AsyncFunctions`, and `Executor` traits.
//!
//! ## Attribution
//!
//! Adapted from the `autogpt` project's `traits/functions.rs`:
//! <https://github.com/wiseaidotdev/autogpt/blob/main/autogpt/src/traits/functions.rs>

use crate::agent::LmmAgent;
use crate::types::{Message, Task};
use anyhow::Result;
use async_trait::async_trait;

// Functions (sync accessor)

/// Synchronous accessor used to retrieve the underlying [`LmmAgent`] core from
/// a custom agent struct.
///
/// # Examples
///
/// ```rust
/// use lmm_agent::prelude::*;
///
/// struct MyAgent { agent: LmmAgent }
/// impl Functions for MyAgent {
///     fn get_agent(&self) -> &LmmAgent { &self.agent }
/// }
/// ```
pub trait Functions {
    /// Returns a reference to this agent's inner [`LmmAgent`].
    fn get_agent(&self) -> &LmmAgent;
}

// AsyncFunctions (async capabilities)

/// Async functions available to every agent.
///
/// The [`lmm_derive::Auto`] macro implements this trait automatically for any
/// struct that contains an `agent: LmmAgent` field and implements [`Executor`].
#[async_trait]
pub trait AsyncFunctions: Send + Sync {
    /// Runs the agent's core task loop.
    ///
    /// # Arguments
    ///
    /// * `tasks`      - The task currently assigned to this agent.
    /// * `execute`    - Whether generated artefacts should be executed.
    /// * `browse`     - Whether to open browser tabs for visual verification.
    /// * `max_tries`  - How many retry attempts to allow on failure.
    async fn execute<'a>(
        &'a mut self,
        tasks: &'a mut Task,
        execute: bool,
        browse: bool,
        max_tries: u64,
    ) -> Result<()>;

    /// Persists a [`Message`] to the agent's long-term memory store.
    ///
    /// The default store is in-memory (`LmmAgent::long_term_memory`).
    async fn save_ltm(&mut self, message: Message) -> Result<()>;

    /// Retrieves all communications from the agent's long-term memory.
    async fn get_ltm(&self) -> Result<Vec<Message>>;

    /// Returns all long-term memory entries concatenated as a single `String`.
    ///
    /// Format: `"role: content\nrole: content\n..."`
    async fn ltm_context(&self) -> String;

    /// Generates a response to `request` using the `lmm` equation intelligence
    /// engine (no external LLM API required).
    ///
    /// Optionally enriches the seed with a DuckDuckGo search when the `net`
    /// feature is enabled.
    async fn generate(&mut self, request: &str) -> Result<String>;

    /// Searches DuckDuckGo for `query` and returns up to `limit` result
    /// snippets concatenated into a single text corpus.
    ///
    /// Requires the `net` feature flag on `lmm-agent`.
    async fn search(&self, query: &str, limit: usize) -> Result<String>;
}

// Executor - user-implemented task logic

/// The single trait that users must implement on their custom agent struct.
///
/// The macro [`lmm_derive::Auto`] calls `<YourAgent as Executor>::execute`
/// from inside its generated `AsyncFunctions::execute` implementation, so you
/// only ever need to implement this one trait.
///
/// # Example
///
/// ```rust
/// use lmm_agent::prelude::*;
///
/// #[derive(Debug, Default, Auto)]
/// pub struct ResearchAgent {
///     pub persona:   Cow<'static, str>,
///     pub behavior:  Cow<'static, str>,
///     pub status:    Status,
///     pub agent:     LmmAgent,
///     pub memory:    Vec<Message>,
/// }
///
/// #[async_trait]
/// impl Executor for ResearchAgent {
///     async fn execute<'a>(
///         &'a mut self, tasks: &'a mut Task,
///         _execute: bool, _browse: bool, _max_tries: u64,
///     ) -> Result<()> {
///         let prompt   = self.agent.persona().to_string();
///         let response = self.generate(&prompt).await?;
///
///         let facts = self.search(&prompt, 3).await.unwrap_or_default();
///
///         self.agent.add_message(Message::new("assistant", response));
///
///         let _ = self.save_ltm(Message::new("facts", facts)).await;
///
///         Ok(())
///     }
/// }
/// ```
#[async_trait]
pub trait Executor {
    /// Executes the agent's core logic against the supplied [`Task`].
    async fn execute<'a>(
        &'a mut self,
        tasks: &'a mut Task,
        execute: bool,
        browse: bool,
        max_tries: u64,
    ) -> Result<()>;
}
