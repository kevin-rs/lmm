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
    AgentSnapshot, Capability, ContextManager, ExperienceRecord, Goal, Knowledge, LearningMode,
    Message, Planner, Profile, Reflection, Route, ScheduledTask, Scope, Status, Task,
    TaskScheduler, ThinkResult, Tool, ToolName, default_eval_fn,
};

pub use crate::cognition::{
    ActionKey, CognitionSignal, ColdStore, DocumentChunk, ElasticMemoryGuard, FederatedAggregator,
    GoalEvaluator, HotStore, InformalLearner, KnowledgeDistiller, KnowledgeIndex, KnowledgeSource,
    LearningConfig, LearningEngine, MemoryEntry, MetaAdapter, QTable, Reflector, SearchOracle,
    ThinkLoop, ThinkLoopBuilder, error_from_texts,
};

pub use anyhow::{Result, anyhow};
pub use async_trait::async_trait;
pub use lmm_derive::Auto;
pub use std::borrow::Cow;
pub use std::collections::HashSet;
pub use std::sync::Arc;
pub use tokio::sync::Mutex;
pub use uuid::Uuid;

pub use crate::agents;
