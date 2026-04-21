// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `LmmAgent` - the core agent struct.
//!
//! `LmmAgent` is the batteries-included foundation for every custom agent.
//! It holds all agent state (hot memory, long-term memory, tools, planner,
//! reflection, scheduler, ...) and provides symbolic text generation powered
//! by `lmm`'s [`TextPredictor`] plus optional DuckDuckGo knowledge enrichment.
//!
//! ## Builder pattern
//!
//! ```rust
//! use lmm_agent::agent::LmmAgent;
//!
//! let agent = LmmAgent::builder()
//!     .persona("Research Assistant")
//!     .behavior("Summarise the Rust ecosystem.")
//!     .build();
//!
//! assert_eq!(agent.persona.as_str(), "Research Assistant");
//! assert_eq!(agent.behavior.as_str(), "Summarise the Rust ecosystem.");
//! ```
//!
//! ## Attribution
//!
//! Adapted from the `autogpt` project's `agents/agent.rs`:
//! <https://github.com/wiseaidotdev/autogpt/blob/main/autogpt/src/agents/agent.rs>

use crate::traits::agent::Agent;
use crate::types::{
    Capability, ContextManager, Knowledge, Message, Planner, Profile, Reflection, Status, Task,
    TaskScheduler, Tool,
};
use anyhow::Result;
use lmm::predict::TextPredictor;
use std::collections::HashSet;

#[cfg(feature = "net")]
use duckduckgo::browser::Browser;
#[cfg(feature = "net")]
use duckduckgo::user_agents::get as get_ua;

// LmmAgent struct

/// The core agent type.
///
/// Use [`LmmAgent::builder()`] for fluent construction, or
/// [`LmmAgent::new()`] for the quick two-argument form.
///
/// # Examples
///
/// ```rust
/// use lmm_agent::agent::LmmAgent;
///
/// let agent = LmmAgent::builder()
///     .persona("Research Agent")
///     .behavior("Research quantum computing.")
///     .build();
///
/// assert_eq!(agent.persona.as_str(), "Research Agent");
///
/// let agent2 = LmmAgent::new("Scientist".into(), "Do science.".into());
/// assert_eq!(agent2.persona.as_str(), "Scientist");
/// ```
#[derive(Debug, Clone, Default)]
pub struct LmmAgent {
    /// Unique identifier for this agent instance (auto-generated UUIDv4).
    pub id: String,

    /// The primary mission statement for this agent.
    pub persona: String,

    /// The role or behavior label (e.g. `"Research Assistant"`).
    pub behavior: String,

    /// Current lifecycle state.
    pub status: Status,

    /// Hot memory - recent messages kept in RAM.
    pub memory: Vec<Message>,

    /// Long-term memory - persisted between task executions (in-memory store).
    pub long_term_memory: Vec<Message>,

    /// Structured knowledge facts for reasoning.
    pub knowledge: Knowledge,

    /// Callable tools available to this agent.
    pub tools: Vec<Tool>,

    /// Optional goal planner.
    pub planner: Option<Planner>,

    /// Self-reflection / evaluation module.
    pub reflection: Option<Reflection>,

    /// Time-based task scheduler.
    pub scheduler: Option<TaskScheduler>,

    /// Profilelity traits and behavioural profile.
    pub profile: Profile,

    /// Recent-message context window.
    pub context: ContextManager,

    /// Capabilities the agent possesses.
    pub capabilities: HashSet<Capability>,

    /// Active task queue.
    pub tasks: Vec<Task>,
}

// LmmAgentBuilder

/// Builder for [`LmmAgent`].
///
/// Obtain via [`LmmAgent::builder()`].
///
/// # Examples
///
/// ```rust
/// use lmm_agent::agent::LmmAgent;
/// use lmm_agent::types::{Message, Planner, Goal};
///
/// let agent = LmmAgent::builder()
///     .persona("Test agent.")
///     .behavior("Tester")
///     .memory(vec![Message::new("user", "Hi")])
///     .planner(Planner {
///         current_plan: vec![Goal {
///             description: "Say hello.".into(),
///             priority: 0,
///             completed: false,
///         }],
///     })
///     .build();
///
/// assert_eq!(agent.persona.as_str(), "Test agent.");
/// assert_eq!(agent.memory.len(), 1);
/// ```
#[derive(Default)]
pub struct LmmAgentBuilder {
    id: Option<String>,
    persona: Option<String>,
    behavior: Option<String>,
    status: Option<Status>,
    memory: Option<Vec<Message>>,
    long_term_memory: Option<Vec<Message>>,
    knowledge: Option<Knowledge>,
    tools: Option<Vec<Tool>>,
    planner: Option<Option<Planner>>,
    reflection: Option<Option<Reflection>>,
    scheduler: Option<Option<TaskScheduler>>,
    profile: Option<Profile>,
    context: Option<ContextManager>,
    capabilities: Option<HashSet<Capability>>,
    tasks: Option<Vec<Task>>,
}

impl LmmAgentBuilder {
    /// Sets the agent's unique identifier (default: auto-generated UUIDv4).
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Sets the agent's persona **(required)**.
    pub fn persona(mut self, persona: impl Into<String>) -> Self {
        self.persona = Some(persona.into());
        self
    }

    /// Sets the agent's behavior / role label **(required)**.
    pub fn behavior(mut self, behavior: impl Into<String>) -> Self {
        self.behavior = Some(behavior.into());
        self
    }

    /// Sets the initial [`Status`] (default: [`Status::Idle`]).
    pub fn status(mut self, status: Status) -> Self {
        self.status = Some(status);
        self
    }

    /// Sets the hot memory (default: empty).
    pub fn memory(mut self, memory: Vec<Message>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Sets the long-term memory (default: empty).
    pub fn long_term_memory(mut self, ltm: Vec<Message>) -> Self {
        self.long_term_memory = Some(ltm);
        self
    }

    /// Sets the knowledge base (default: empty).
    pub fn knowledge(mut self, knowledge: Knowledge) -> Self {
        self.knowledge = Some(knowledge);
        self
    }

    /// Sets the tool list (default: empty).
    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Sets an optional planner (default: empty planner).
    pub fn planner(mut self, planner: impl Into<Option<Planner>>) -> Self {
        self.planner = Some(planner.into());
        self
    }

    /// Sets an optional reflection module (default: default reflection).
    pub fn reflection(mut self, reflection: impl Into<Option<Reflection>>) -> Self {
        self.reflection = Some(reflection.into());
        self
    }

    /// Sets an optional task scheduler (default: empty scheduler).
    pub fn scheduler(mut self, scheduler: impl Into<Option<TaskScheduler>>) -> Self {
        self.scheduler = Some(scheduler.into());
        self
    }

    /// Sets the profile (default: name = behavior, no traits).
    pub fn profile(mut self, profile: Profile) -> Self {
        self.profile = Some(profile);
        self
    }

    /// Sets the context manager (default: empty).
    pub fn context(mut self, context: ContextManager) -> Self {
        self.context = Some(context);
        self
    }

    /// Sets the capability set (default: empty).
    pub fn capabilities(mut self, capabilities: HashSet<Capability>) -> Self {
        self.capabilities = Some(capabilities);
        self
    }

    /// Sets the task queue (default: empty).
    pub fn tasks(mut self, tasks: Vec<Task>) -> Self {
        self.tasks = Some(tasks);
        self
    }

    /// Constructs the [`LmmAgent`].
    ///
    /// # Panics
    ///
    /// Panics if `persona` or `behavior` were not set.
    pub fn build(self) -> LmmAgent {
        let persona = self
            .persona
            .expect("LmmAgentBuilder: `persona` is required");
        let behavior = self
            .behavior
            .expect("LmmAgentBuilder: `behavior` is required");
        let profile = self.profile.unwrap_or_else(|| Profile {
            name: behavior.clone().into(),
            traits: vec![],
            behavior_script: None,
        });

        LmmAgent {
            id: self.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            persona,
            behavior,
            status: self.status.unwrap_or_default(),
            memory: self.memory.unwrap_or_default(),
            long_term_memory: self.long_term_memory.unwrap_or_default(),
            knowledge: self.knowledge.unwrap_or_default(),
            tools: self.tools.unwrap_or_default(),
            planner: self.planner.unwrap_or_else(|| Some(Planner::default())),
            reflection: self
                .reflection
                .unwrap_or_else(|| Some(Reflection::default())),
            scheduler: self
                .scheduler
                .unwrap_or_else(|| Some(TaskScheduler::default())),
            profile,
            context: self.context.unwrap_or_default(),
            capabilities: self.capabilities.unwrap_or_default(),
            tasks: self.tasks.unwrap_or_default(),
        }
    }
}

// Inherent methods

impl LmmAgent {
    /// Returns a new [`LmmAgentBuilder`].
    ///
    /// The builder accepts every field with `with_*`-style setters and calls
    /// `.build()` to produce the final [`LmmAgent`].
    pub fn builder() -> LmmAgentBuilder {
        LmmAgentBuilder::default()
    }

    /// Constructs an [`LmmAgent`] with the given persona and behavior;
    /// every other field is set to its sensible default.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::agent::LmmAgent;
    ///
    /// let agent = LmmAgent::new("Researcher".into(), "Research Rust.".into());
    /// assert_eq!(agent.behavior.as_str(), "Research Rust.");
    /// ```
    pub fn new(
        persona: std::borrow::Cow<'static, str>,
        behavior: std::borrow::Cow<'static, str>,
    ) -> Self {
        LmmAgent::builder()
            .persona(persona.into_owned())
            .behavior(behavior.into_owned())
            .build()
    }

    /// Appends a [`Message`] to the agent's hot memory.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::agent::LmmAgent;
    /// use lmm_agent::types::Message;
    ///
    /// let mut agent = LmmAgent::new("Tester".into(), "Test.".into());
    /// agent.add_message(Message::new("user", "Hello"));
    /// assert_eq!(agent.memory.len(), 1);
    /// ```
    pub fn add_message(&mut self, message: Message) {
        self.memory.push(message);
    }

    /// Appends a [`Message`] to the agent's long-term memory.
    pub fn add_ltm_message(&mut self, message: Message) {
        self.long_term_memory.push(message);
    }

    /// Marks a goal as completed by its description substring.
    ///
    /// Returns `true` if a matching goal was found and updated.
    pub fn complete_goal(&mut self, description_substr: &str) -> bool {
        if let Some(plan) = self.planner.as_mut() {
            for goal in &mut plan.current_plan {
                if goal.description.contains(description_substr) {
                    goal.completed = true;
                    return true;
                }
            }
        }
        false
    }

    /// Generates a textual response to `request` using [`lmm::predict::TextPredictor`].
    ///
    /// `TextPredictor` fits a tone trajectory and a rhythm trajectory over the
    /// input tokens using symbolic regression, then selects continuation words
    /// from compile-time lexical pools: entirely deterministic, no LLM API
    /// required.
    ///
    /// When the `net` feature is enabled, the seed is enriched with DuckDuckGo
    /// search snippets before feeding it to the predictor.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # #[tokio::main]
    /// # async fn main() {
    /// use lmm_agent::agent::LmmAgent;
    /// let mut agent = LmmAgent::new("Tester".into(), "Rust is fast.".into());
    /// let result = agent.generate("the universe reveals its truth").await;
    /// assert!(result.is_ok());
    /// assert!(!result.unwrap().is_empty());
    /// # }
    /// ```
    pub async fn generate(&mut self, request: &str) -> Result<String> {
        #[cfg(feature = "net")]
        let seed = {
            let corpus = self.search(request, 5).await.unwrap_or_default();
            if corpus.is_empty() {
                request.to_string()
            } else {
                format!("{request} {corpus}")
            }
        };

        #[cfg(not(feature = "net"))]
        let seed = request.to_string();

        // Ensure the seed has at least two words (TextPredictor requirement).
        let seed = if seed.split_whitespace().count() < 2 {
            format!("{seed} and")
        } else {
            seed
        };

        let predictor = TextPredictor::new(20, 40, 3);
        let result = predictor
            .predict_continuation(&seed, 120)
            .map(|c| format!("{} {}", seed.trim(), c.continuation.trim()))
            .unwrap_or_else(|_| seed.clone());

        self.add_message(Message::new("user", request.to_string()));
        self.add_message(Message::new("assistant", result.clone()));

        Ok(result)
    }

    /// Searches DuckDuckGo for `query` (requires `net` feature).
    #[cfg(feature = "net")]
    pub async fn search(&self, query: &str, limit: usize) -> Result<String> {
        let browser = Browser::new();
        let ua = get_ua("firefox").unwrap_or("Mozilla/5.0");
        let results = browser.lite_search(query, "wt-wt", Some(limit), ua).await?;

        let corpus = results
            .iter()
            .filter_map(|r| {
                let snippet = r.snippet.trim();
                if !snippet.is_empty() {
                    Some(snippet.to_string())
                } else if !r.title.trim().is_empty() {
                    Some(r.title.trim().to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(" ");

        Ok(corpus)
    }

    /// No-op search when the `net` feature is disabled.
    #[cfg(not(feature = "net"))]
    pub async fn search(&self, _query: &str, _limit: usize) -> Result<String> {
        Ok(String::new())
    }
}

// Agent trait implementation

impl Agent for LmmAgent {
    fn new(
        persona: std::borrow::Cow<'static, str>,
        behavior: std::borrow::Cow<'static, str>,
    ) -> Self {
        LmmAgent::new(persona, behavior)
    }

    fn update(&mut self, status: Status) {
        self.status = status;
    }

    fn persona(&self) -> &str {
        &self.persona
    }

    fn behavior(&self) -> &str {
        &self.behavior
    }

    fn status(&self) -> &Status {
        &self.status
    }

    fn memory(&self) -> &Vec<Message> {
        &self.memory
    }

    fn tools(&self) -> &Vec<Tool> {
        &self.tools
    }

    fn knowledge(&self) -> &Knowledge {
        &self.knowledge
    }

    fn planner(&self) -> Option<&Planner> {
        self.planner.as_ref()
    }

    fn profile(&self) -> &Profile {
        &self.profile
    }

    fn reflection(&self) -> Option<&Reflection> {
        self.reflection.as_ref()
    }

    fn scheduler(&self) -> Option<&TaskScheduler> {
        self.scheduler.as_ref()
    }

    fn capabilities(&self) -> &HashSet<Capability> {
        &self.capabilities
    }

    fn context(&self) -> &ContextManager {
        &self.context
    }

    fn tasks(&self) -> &Vec<Task> {
        &self.tasks
    }

    fn memory_mut(&mut self) -> &mut Vec<Message> {
        &mut self.memory
    }

    fn planner_mut(&mut self) -> Option<&mut Planner> {
        self.planner.as_mut()
    }

    fn context_mut(&mut self) -> &mut ContextManager {
        &mut self.context
    }
}
