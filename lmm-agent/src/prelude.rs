// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Prelude
//!
//! Convenience re-exports for the most commonly used types, traits and macros
//! in `lmm-agent`.
//!
//! Import everything at once with `use lmm_agent::prelude::*;`.

pub use crate::agent::LmmAgent;
pub use crate::error::{AgentBuildError, AgentError};
pub use crate::runtime::AutoAgent;
pub use crate::traits::agent::Agent;
pub use crate::traits::composite::AgentFunctions;
pub use crate::traits::functions::{AsyncFunctions, Executor, Functions};
pub use crate::types::{
    Capability, ContextManager, Goal, Knowledge, Message, Planner, Profile, Reflection, Route,
    ScheduledTask, Scope, Status, Task, TaskScheduler, Tool, ToolName, default_eval_fn,
};

// External re-exports used in macro-generated code and user impls.
pub use anyhow::{Result, anyhow};
pub use async_trait::async_trait;
pub use lmm_derive::Auto;
pub use std::borrow::Cow;
pub use std::collections::HashSet;
pub use std::sync::Arc;
pub use tokio::sync::Mutex;
pub use uuid::Uuid;

// Re-export the agents! macro so users only need `use lmm_agent::prelude::*`.
pub use crate::agents;
