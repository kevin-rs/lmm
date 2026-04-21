// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Agent domain types.
//!
//! Foundational value types shared across all agent implementations.
//!
//! All types here are:
//! - `Clone` + `Debug` + `PartialEq`
//! - `serde::{Serialize, Deserialize}` where serialisation makes sense
//! - `Send + Sync` - safe to move across async task boundaries
//!
//! ## Attribution
//!
//! Adapted from the `autogpt` crate's `common/utils.rs`:
//! <https://github.com/wiseaidotdev/autogpt/blob/main/autogpt/src/common/utils.rs>

use chrono::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;

/// The current operational status of an agent through its lifecycle.
///
/// # Examples
///
/// ```
/// use lmm_agent::types::Status;
///
/// let s = Status::default();
/// assert_eq!(s, Status::Idle);
/// ```
#[derive(Debug, PartialEq, Eq, Default, Clone, Hash, Serialize, Deserialize)]
pub enum Status {
    /// Agent is waiting for a task to be assigned.
    #[default]
    Idle,
    /// Agent is actively processing a task.
    Active,
    /// Agent is validating its own outputs.
    InUnitTesting,
    /// Agent has finished all assigned tasks.
    Completed,
    /// Agent is running the closed-loop [`ThinkLoop`] reasoning cycle.
    Thinking,
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::Active => write!(f, "Active"),
            Self::InUnitTesting => write!(f, "InUnitTesting"),
            Self::Completed => write!(f, "Completed"),
            Self::Thinking => write!(f, "Thinking"),
        }
    }
}

/// A single message exchanged between an agent and a user/system/tool.
///
/// # Examples
///
/// ```
/// use lmm_agent::types::Message;
///
/// let msg = Message {
///     role: "user".into(),
///     content: "Hello, agent!".into(),
/// };
/// assert_eq!(msg.role.as_ref(), "user");
/// ```
#[derive(Debug, PartialEq, Eq, Default, Clone, Hash, Serialize, Deserialize)]
pub struct Message {
    /// Who produced this message (e.g. `"user"`, `"assistant"`, `"system"`).
    pub role: Cow<'static, str>,
    /// The message text.
    pub content: Cow<'static, str>,
}

impl Message {
    /// Constructs a new [`Message`] with the given role and content.
    pub fn new(role: impl Into<Cow<'static, str>>, content: impl Into<Cow<'static, str>>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// A structured knowledge base mapping fact identifiers to their explanations.
///
/// # Examples
///
/// ```
/// use lmm_agent::types::Knowledge;
/// use std::borrow::Cow;
///
/// let mut kb = Knowledge::default();
/// kb.facts.insert(Cow::Borrowed("Rust"), Cow::Borrowed("A systems language."));
/// assert_eq!(kb.facts.len(), 1);
/// ```
#[derive(Debug, PartialEq, Eq, Default, Clone, Serialize, Deserialize)]
pub struct Knowledge {
    /// Map from fact key to natural-language description.
    pub facts: HashMap<Cow<'static, str>, Cow<'static, str>>,
}

impl Knowledge {
    /// Inserts or overwrites a fact.
    pub fn insert(
        &mut self,
        key: impl Into<Cow<'static, str>>,
        value: impl Into<Cow<'static, str>>,
    ) {
        self.facts.insert(key.into(), value.into());
    }

    /// Looks up a fact by key.
    pub fn get(&self, key: &str) -> Option<&Cow<'static, str>> {
        self.facts.get(key as &str)
    }
}

/// The name of a built-in or custom tool the agent can invoke.
#[derive(Debug, PartialEq, Eq, Default, Clone, Hash)]
pub enum ToolName {
    /// Full-text web search (default).
    #[default]
    Search,
    Browser,
    News,
    Wiki,
    Calc,
    Math,
    Format,
    Exec,
    Code,
    Regex,
    Read,
    Write,
    Pdf,
    Summarize,
    Email,
    Calendar,
    Translate,
    Sentiment,
    Classify,
    Memory,
    Plan,
    Spawn,
    Judge,
    Plugin(String),
}

/// A callable tool the agent can invoke at runtime.
///
/// The invocation is synchronous and deterministic - side-effectful tools
/// (e.g. shell commands) should be wrapped in a `Plugin` variant tool with
/// appropriate error handling inside `invoke`.
///
/// # Examples
///
/// ```
/// use lmm_agent::types::{Tool, ToolName};
///
/// let echo = Tool {
///     name: ToolName::Plugin("echo".to_string()),
///     description: "Echoes input back.".into(),
///     invoke: |input| input.to_string(),
/// };
/// assert_eq!((echo.invoke)("hello"), "hello");
/// ```
#[derive(Clone)]
pub struct Tool {
    /// The name/kind of this tool.
    pub name: ToolName,
    /// Human-readable description shown in planning context.
    pub description: Cow<'static, str>,
    /// Synchronous invocation function: `input → output`.
    pub invoke: fn(&str) -> String,
}

impl fmt::Debug for Tool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tool")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

impl PartialEq for Tool {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.description == other.description
    }
}

impl Default for Tool {
    fn default() -> Self {
        Self {
            name: ToolName::default(),
            description: Cow::Borrowed(""),
            invoke: |_| String::new(),
        }
    }
}

/// A single goal within the agent's current plan.
///
/// # Examples
///
/// ```
/// use lmm_agent::types::Goal;
///
/// let g = Goal { description: "Research Rust".into(), priority: 1, completed: false };
/// assert!(!g.completed);
/// ```
#[derive(Debug, PartialEq, Eq, Default, Clone, Hash, Serialize, Deserialize)]
pub struct Goal {
    /// Short text describing what must be accomplished.
    pub description: String,
    /// Urgency level: lower values = higher urgency.
    pub priority: u8,
    /// Whether this goal has been achieved.
    pub completed: bool,
}

/// An ordered sequence of [`Goal`]s the agent is working through.
///
/// # Examples
///
/// ```
/// use lmm_agent::types::{Planner, Goal};
///
/// let mut p = Planner::default();
/// p.current_plan.push(Goal { description: "Init".into(), priority: 0, completed: false });
/// assert_eq!(p.current_plan.len(), 1);
/// ```
#[derive(Debug, PartialEq, Eq, Default, Clone, Hash, Serialize, Deserialize)]
pub struct Planner {
    /// Goals in execution order.
    pub current_plan: Vec<Goal>,
}

impl Planner {
    /// Returns the number of completed goals.
    pub fn completed_count(&self) -> usize {
        self.current_plan.iter().filter(|g| g.completed).count()
    }

    /// Returns `true` when every goal is complete.
    pub fn is_done(&self) -> bool {
        self.current_plan.iter().all(|g| g.completed)
    }
}

/// The personality profile that shapes how an agent behaves and responds.
///
/// # Examples
///
/// ```
/// use lmm_agent::types::Profile;
///
/// let p = Profile {
///     name: "ResearchBot".into(),
///     traits: vec!["curious".into(), "precise".into()],
///     behavior_script: None,
/// };
/// assert_eq!(p.name.as_ref(), "ResearchBot");
/// ```
#[derive(Debug, PartialEq, Eq, Default, Clone, Hash, Serialize, Deserialize)]
pub struct Profile {
    /// Display name for this persona.
    pub name: Cow<'static, str>,
    /// Adjectives describing the agent's style.
    pub traits: Vec<Cow<'static, str>>,
    /// Optional DSL / prompt script controlling fine-grained behaviour.
    pub behavior_script: Option<Cow<'static, str>>,
}

/// Introspection data that allows an agent to evaluate its own performance.
pub struct Reflection {
    /// Rolling log of recent activities or observations.
    pub recent_logs: Vec<Cow<'static, str>>,
    /// A function that returns a natural-language assessment of the agent.
    pub evaluation_fn: fn(&dyn crate::traits::agent::Agent) -> Cow<'static, str>,
}

impl fmt::Debug for Reflection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Reflection")
            .field("recent_logs", &self.recent_logs)
            .finish_non_exhaustive()
    }
}

impl PartialEq for Reflection {
    fn eq(&self, other: &Self) -> bool {
        self.recent_logs == other.recent_logs
    }
}

impl Clone for Reflection {
    fn clone(&self) -> Self {
        Self {
            recent_logs: self.recent_logs.clone(),
            evaluation_fn: self.evaluation_fn,
        }
    }
}

impl Default for Reflection {
    fn default() -> Self {
        Self {
            recent_logs: vec![],
            evaluation_fn: default_eval_fn,
        }
    }
}

/// A task pinned to a specific wall-clock time.
#[derive(Debug, PartialEq, Eq, Default, Clone, Hash, Serialize, Deserialize)]
pub struct ScheduledTask {
    /// UTC timestamp at which the task should run.
    pub time: DateTime<Utc>,
    /// The task payload.
    pub task: Task,
}

/// Manages a queue of time-triggered tasks.
///
/// # Examples
///
/// ```
/// use lmm_agent::types::TaskScheduler;
///
/// let sched = TaskScheduler::default();
/// assert!(sched.scheduled_tasks.is_empty());
/// ```
#[derive(Debug, PartialEq, Eq, Default, Clone, Hash, Serialize, Deserialize)]
pub struct TaskScheduler {
    /// Queue of pending scheduled tasks.
    pub scheduled_tasks: Vec<ScheduledTask>,
}

/// An atomic capability the agent possesses.
#[derive(Debug, PartialEq, Eq, Default, Clone, Hash, Serialize, Deserialize)]
pub enum Capability {
    /// Can generate code from a natural-language description.
    #[default]
    CodeGen,
    /// Can design user-interface layouts.
    UIDesign,
    /// Can execute live web searches.
    WebSearch,
    /// Can query SQL databases.
    SQLAccess,
    /// Can control robotic actuators.
    RobotControl,
    /// Can integrate with external REST / gRPC APIs.
    ApiIntegration,
    /// Can convert text to speech audio.
    TextToSpeech,
    /// Custom capability identified by a string label.
    Custom(String),
}

/// Tracks recent exchanges and focuses the agent on relevant topics.
///
/// # Examples
///
/// ```
/// use lmm_agent::types::ContextManager;
///
/// let ctx = ContextManager::default();
/// assert!(ctx.recent_messages.is_empty());
/// assert!(ctx.focus_topics.is_empty());
/// ```
#[derive(Debug, PartialEq, Eq, Default, Clone, Hash, Serialize, Deserialize)]
pub struct ContextManager {
    /// The most recent messages in the conversation window.
    pub recent_messages: Vec<Message>,
    /// Topics the agent is currently focused on.
    pub focus_topics: Vec<Cow<'static, str>>,
}

/// Scope permissions for a task - controls what operations the agent may
/// perform while executing.
///
/// # Examples
///
/// ```
/// use lmm_agent::types::Scope;
///
/// let s = Scope { crud: true, auth: false, external: true };
/// assert!(s.crud);
/// ```
#[derive(Debug, PartialEq, Eq, Default, Clone, Copy, Hash, Serialize, Deserialize)]
pub struct Scope {
    /// Allow Create / Read / Update / Delete operations.
    pub crud: bool,
    /// Allow authentication / authorisation related actions.
    pub auth: bool,
    /// Allow reaching external services or URLs.
    pub external: bool,
}

/// Describes an HTTP route produced or consumed by the agent.
#[derive(Debug, PartialEq, Eq, Default, Clone, Hash, Serialize, Deserialize)]
pub struct Route {
    /// Whether the route segment is dynamic (e.g. `"/items/{id}"`).
    pub dynamic: Cow<'static, str>,
    /// HTTP method (`"GET"`, `"POST"`, ...).
    pub method: Cow<'static, str>,
    /// Example request body as a JSON value.
    pub body: Value,
    /// Example response body as a JSON value.
    pub response: Value,
    /// Route path.
    pub path: Cow<'static, str>,
}

/// The primary unit of work handed to an agent for execution.
///
/// # Examples
///
/// ```
/// use lmm_agent::types::Task;
///
/// let task = Task::from_description("Summarise the Rust book.");
/// assert!(!task.description.is_empty());
/// ```
#[derive(Debug, PartialEq, Eq, Default, Clone, Hash, Serialize, Deserialize)]
pub struct Task {
    /// Human-readable description of what must be accomplished.
    pub description: Cow<'static, str>,
    /// Optional permission scope.
    pub scope: Option<Scope>,
    /// External URLs the agent may need to consult.
    pub urls: Option<Vec<Cow<'static, str>>>,
    /// Generated or supplied frontend source code.
    pub frontend_code: Option<Cow<'static, str>>,
    /// Generated or supplied backend source code.
    pub backend_code: Option<Cow<'static, str>>,
    /// API endpoint schema discovered or generated by the agent.
    pub api_schema: Option<Vec<Route>>,
}

// ThinkResult

/// The outcome of one [`crate::agent::LmmAgent::think()`] invocation.
///
/// Returned by the closed-loop [`crate::cognition::r#loop::ThinkLoop`] controller.
///
/// # Examples
///
/// ```rust
/// #[tokio::main]
/// async fn main() {
///     use lmm_agent::agent::LmmAgent;
///
///     let mut agent = LmmAgent::new("Tester".into(), "Test.".into());
///     let result = agent.think("What is Rust ownership?").await.unwrap();
///     assert!(result.steps > 0);
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ThinkResult {
    /// `true` when the Jaccard error fell below the convergence threshold.
    pub converged: bool,

    /// Number of feedback iterations executed.
    pub steps: usize,

    /// Final Jaccard distance between goal and last observation, ∈ [0, 1].
    pub final_error: f64,

    /// Snapshot of hot-store contents at termination (newest-first).
    pub memory_snapshot: Vec<String>,

    /// The per-step signals produced during the run.
    pub signals: Vec<crate::cognition::signal::CognitionSignal>,
}

impl Task {
    /// Constructs a minimal [`Task`] from a plain description string.
    pub fn from_description(description: impl Into<Cow<'static, str>>) -> Self {
        Self {
            description: description.into(),
            ..Default::default()
        }
    }
}

// Default evaluation function

/// Default introspective evaluation function.
///
/// Summarises goal completion progress from the agent's planner and returns a
/// human-readable [`Cow<'static, str>`].
///
/// ## Attribution
///
/// Adapted from autogpt's `common/utils.rs`:
/// <https://github.com/wiseaidotdev/autogpt/blob/main/autogpt/src/common/utils.rs>
pub fn default_eval_fn(agent: &dyn crate::traits::agent::Agent) -> Cow<'static, str> {
    if let Some(planner) = agent.planner() {
        let total = planner.current_plan.len();
        let completed = planner.completed_count();
        let in_progress = total - completed;

        let mut summary = format!(
            "\n- Total Goals: {total}\n- Completed: {completed}\n- In Progress: {in_progress}\n\nGoals Summary:\n"
        );

        for goal in &planner.current_plan {
            let state = if goal.completed { "✓" } else { "○" };
            summary.push_str(&format!(
                "  [{state}] (priority {}) {}\n",
                goal.priority, goal.description
            ));
        }

        Cow::Owned(summary)
    } else {
        Cow::Borrowed("No planner configured.")
    }
}
