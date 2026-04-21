// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `Agent` trait
//!
//! Defines the fundamental contract that every agent must fulfil.
//!
//! ## Attribution
//!
//! Adapted from the `autogpt` project's `traits/agent.rs`:
//! <https://github.com/wiseaidotdev/autogpt/blob/main/autogpt/src/traits/agent.rs>

use crate::types::{
    Capability, ContextManager, Knowledge, Message, Planner, Profile, Reflection, Status, Task,
    TaskScheduler, Tool,
};
use std::collections::HashSet;
use std::fmt::Debug;

/// The fundamental interface every agent **must** implement.
///
/// Implementors are encouraged to use the [`lmm_derive::Auto`] derive macro
/// rather than writing the boilerplate by hand.
pub trait Agent: Debug + Send + Sync {
    /// Constructs a new agent with the given `persona` and `behavior`.
    fn new(
        persona: std::borrow::Cow<'static, str>,
        behavior: std::borrow::Cow<'static, str>,
    ) -> Self
    where
        Self: Sized;

    /// Transitions the agent to a new [`Status`].
    fn update(&mut self, status: Status);

    /// Returns the agent's primary persona text.
    fn persona(&self) -> &str;

    /// Returns the agent's assigned behavior / role label.
    fn behavior(&self) -> &str;

    /// Returns the agent's current operational [`Status`].
    fn status(&self) -> &Status;

    /// Returns the agent's hot memory (recent communications).
    fn memory(&self) -> &Vec<Message>;

    /// Returns the tools registered with this agent.
    fn tools(&self) -> &Vec<Tool>;

    /// Returns the agent's structured knowledge base.
    fn knowledge(&self) -> &Knowledge;

    /// Returns a reference to the agent's goal planner, if one is configured.
    fn planner(&self) -> Option<&Planner>;

    /// Returns the agent's profile definition.
    fn profile(&self) -> &Profile;

    /// Returns the agent's self-reflection module, if configured.
    fn reflection(&self) -> Option<&Reflection>;

    /// Returns the agent's task scheduler, if configured.
    fn scheduler(&self) -> Option<&TaskScheduler>;

    /// Returns the full set of capabilities this agent possesses.
    fn capabilities(&self) -> &HashSet<Capability>;

    /// Returns the context manager tracking recent messages and focus topics.
    fn context(&self) -> &ContextManager;

    /// Returns the list of [`Task`]s currently assigned to this agent.
    fn tasks(&self) -> &Vec<Task>;

    /// Returns a mutable reference to the agent's hot memory.
    fn memory_mut(&mut self) -> &mut Vec<Message>;

    /// Returns a mutable reference to the planner, if one is configured.
    fn planner_mut(&mut self) -> Option<&mut Planner>;

    /// Returns a mutable reference to the context manager.
    fn context_mut(&mut self) -> &mut ContextManager;
}
