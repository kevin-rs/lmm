// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `AutoAgent` - the async orchestrator.
//!
//! `AutoAgent` owns a collection of agents (as `Arc<Mutex<Box<dyn AgentFunctions>>>`)
//! and runs them concurrently on a Tokio runtime.
//!
//! ## Usage
//!
//! ```rust
//! use lmm_agent::prelude::*;
//!
//! #[derive(Debug, Default, Auto)]
//! pub struct MyAgent {
//!     pub persona: Cow<'static, str>,
//!     pub behavior: Cow<'static, str>,
//!     pub status: Status,
//!     pub agent: LmmAgent,
//!     pub memory: Vec<Message>,
//! }
//! #[async_trait]
//! impl Executor for MyAgent {
//!     async fn execute<'a>(&'a mut self, _: &'a mut Task, _: bool, _: bool, _: u64) -> Result<()> { Ok(()) }
//! }
//! #[tokio::main]
//! async fn main() {
//!    let my_agent = MyAgent { agent: LmmAgent::new("p".into(), "t".into()), ..Default::default() };
//!    let autogpt = AutoAgent::default()
//!        .with(agents![my_agent])
//!        .max_tries(3)
//!        .build()
//!        .unwrap();
//!
//!    autogpt.run().await.unwrap();
//! }
//! ```
//!
//! ## Attribution
//!
//! Adapted from the `autogpt` project's `prelude.rs` (`AutoGPT` struct):
//! <https://github.com/wiseaidotdev/autogpt/blob/main/autogpt/src/prelude.rs>

use crate::traits::composite::AgentFunctions;
use crate::types::{Scope, Task};
use anyhow::{Result, anyhow};
use futures::future::join_all;
use std::sync::Arc;
use tokio::{sync::Mutex, task};
use tracing::{debug, error};
use uuid::Uuid;

/// Wraps any [`AgentFunctions`] implementor in the type-erased pointer form
/// expected by [`AutoAgent::with`].
///
/// # Example
///
/// ```rust
/// use lmm_agent::prelude::*;
///
/// #[derive(Debug, Default, Auto)]
/// pub struct MyAgent {
///     pub persona: Cow<'static, str>,
///     pub behavior: Cow<'static, str>,
///     pub status: Status,
///     pub agent: LmmAgent,
///     pub memory: Vec<Message>,
/// }
/// #[async_trait]
/// impl Executor for MyAgent {
///     async fn execute<'a>(&'a mut self, _: &'a mut Task, _: bool, _: bool, _: u64) -> Result<()> { Ok(()) }
/// }
/// let my_agent_a = MyAgent { agent: LmmAgent::new("p".into(), "t".into()), ..Default::default() };
/// let my_agent_b = MyAgent { agent: LmmAgent::new("p".into(), "t".into()), ..Default::default() };
/// let wrapped = agents![my_agent_a, my_agent_b];
/// ```
#[macro_export]
macro_rules! agents {
    ( $($agent:expr),* $(,)? ) => {
        vec![
            $(
                ::std::sync::Arc::new(
                    ::tokio::sync::Mutex::new(
                        Box::new($agent) as Box<dyn $crate::traits::composite::AgentFunctions>
                    )
                )
            ),*
        ]
    };
}

/// Type alias for a thread-safe, heap-allocated, type-erased agent.
pub type BoxedAgent = Arc<Mutex<Box<dyn AgentFunctions>>>;

/// Orchestrates a pool of agents, running them concurrently on separate Tokio
/// tasks and collecting results.
///
/// # Examples
///
/// ```rust
/// use lmm_agent::prelude::*;
///
/// #[derive(Debug, Default, Auto)]
/// pub struct MyAgent {
///     pub persona: Cow<'static, str>,
///     pub behavior: Cow<'static, str>,
///     pub status: Status,
///     pub agent: LmmAgent,
///     pub memory: Vec<Message>,
/// }
/// #[async_trait]
/// impl Executor for MyAgent {
///     async fn execute<'a>(&'a mut self, _: &'a mut Task, _: bool, _: bool, _: u64) -> Result<()> { Ok(()) }
/// }
/// #[tokio::main]
/// async fn main() {
///     let agent = MyAgent {
///         persona: "Researcher".into(),
///         behavior: "Research Rust.".into(),
///         agent: LmmAgent::new("Researcher".into(), "Research Rust.".into()),
///         ..Default::default()
///     };
///
///     AutoAgent::default()
///         .with(agents![agent])
///         .build()
///         .unwrap()
///         .run()
///         .await
///         .unwrap();
/// }
/// ```
pub struct AutoAgent {
    /// Unique ID for this orchestrator instance.
    pub id: Uuid,

    /// Pool of agents to run.
    pub agents: Vec<BoxedAgent>,

    /// Whether agents should execute generated artefacts.
    pub execute: bool,

    /// Whether agents may open browser tabs.
    pub browse: bool,

    /// Maximum task-execution attempts per agent.
    pub max_tries: u64,

    /// CRUD permission scope passed to each agent task.
    pub crud: bool,

    /// Auth permission scope.
    pub auth: bool,

    /// External-access permission scope.
    pub external: bool,
}

impl Default for AutoAgent {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            agents: vec![],
            execute: true,
            browse: false,
            max_tries: 1,
            crud: true,
            auth: false,
            external: true,
        }
    }
}

impl AutoAgent {
    /// Creates a new [`AutoAgent`] with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the orchestrator's unique identifier.
    pub fn id(mut self, id: Uuid) -> Self {
        self.id = id;
        self
    }

    /// Provides the agent pool.
    ///
    /// Use the [`agents!`] macro to construct the pool.
    pub fn with<A>(mut self, agents: A) -> Self
    where
        A: Into<Vec<BoxedAgent>>,
    {
        self.agents = agents.into();
        self
    }

    /// Sets whether agents should execute generated artefacts (default: `true`).
    pub fn execute(mut self, execute: bool) -> Self {
        self.execute = execute;
        self
    }

    /// Sets whether agents may open browser tabs (default: `false`).
    pub fn browse(mut self, browse: bool) -> Self {
        self.browse = browse;
        self
    }

    /// Sets the maximum retry count per agent (default: `1`).
    pub fn max_tries(mut self, max_tries: u64) -> Self {
        self.max_tries = max_tries;
        self
    }

    /// Enables/disables CRUD scope for all agents (default: `true`).
    pub fn crud(mut self, enabled: bool) -> Self {
        self.crud = enabled;
        self
    }

    /// Enables/disables auth scope for all agents (default: `false`).
    pub fn auth(mut self, enabled: bool) -> Self {
        self.auth = enabled;
        self
    }

    /// Enables/disables external-access scope for all agents (default: `true`).
    pub fn external(mut self, enabled: bool) -> Self {
        self.external = enabled;
        self
    }

    /// Finalises the builder, returning `Err` when no agents are registered.
    pub fn build(self) -> Result<Self> {
        if self.agents.is_empty() {
            return Err(anyhow!(
                "No agents registered. Call `.with(agents![...])` first."
            ));
        }
        Ok(self)
    }

    /// Runs all agents concurrently and waits for every one to finish.
    ///
    /// Returns `Ok("All agents executed successfully.")` when every agent
    /// completes without error, or an aggregated error listing all failures.
    pub async fn run(&self) -> Result<String> {
        if self.agents.is_empty() {
            return Err(anyhow!("No agents to run."));
        }

        let execute = self.execute;
        let browse = self.browse;
        let max_tries = self.max_tries;
        let crud = self.crud;
        let auth = self.auth;
        let external = self.external;

        let mut handles = Vec::with_capacity(self.agents.len());

        for (i, agent_arc) in self.agents.iter().cloned().enumerate() {
            let agent_clone = Arc::clone(&agent_arc);
            let agent_persona = agent_arc.lock().await.get_agent().persona.clone();

            let task_payload = Arc::new(Mutex::new(Task {
                description: std::borrow::Cow::Owned(agent_persona.clone()),
                scope: Some(Scope {
                    crud,
                    auth,
                    external,
                }),
                urls: None,
                frontend_code: None,
                backend_code: None,
                api_schema: None,
            }));

            let handle = task::spawn(async move {
                let mut locked_task = task_payload.lock().await;
                let mut agent = agent_clone.lock().await;

                match agent
                    .execute(&mut locked_task, execute, browse, max_tries)
                    .await
                {
                    Ok(_) => {
                        debug!("Agent {i} ({agent_persona}) completed successfully.");
                        Ok::<(), anyhow::Error>(())
                    }
                    Err(err) => {
                        error!("Agent {i} ({agent_persona}) failed: {err}");
                        Err(anyhow!("Agent {i} failed: {err}"))
                    }
                }
            });

            handles.push(handle);
        }

        let results = join_all(handles).await;

        let failures: Vec<String> = results
            .into_iter()
            .enumerate()
            .filter_map(|(i, res)| match res {
                Ok(Err(e)) => Some(format!("Agent {i}: {e}")),
                Err(join_err) => Some(format!("Agent {i} panicked: {join_err}")),
                _ => None,
            })
            .collect();

        if !failures.is_empty() {
            return Err(anyhow!("Some agents failed:\n{}", failures.join("\n")));
        }

        Ok("All agents executed successfully.".to_string())
    }
}
