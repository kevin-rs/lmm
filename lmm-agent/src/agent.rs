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
    TaskScheduler, ThinkResult, Tool,
};
use anyhow::Result;
use lmm::predict::TextPredictor;
use std::collections::HashSet;

#[cfg(feature = "net")]
use duckduckgo::browser::Browser;
#[cfg(feature = "net")]
use duckduckgo::user_agents::get as get_ua;

use crate::cognition::r#loop::ThinkLoop;
use crate::cognition::search::SearchOracle;

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
    /// ```rust
    /// #[tokio::main]
    /// async fn main() {
    ///     use lmm_agent::agent::LmmAgent;
    ///     let mut agent = LmmAgent::new("Tester".into(), "Rust is fast.".into());
    ///     let result = agent.generate("the universe reveals its truth").await;
    ///     assert!(result.is_ok());
    ///     assert!(!result.unwrap().is_empty());
    /// }
    /// ```
    pub async fn generate(&mut self, request: &str) -> Result<String> {
        #[cfg(feature = "net")]
        let result = {
            let corpus = self.search(request, 5).await.unwrap_or_default();
            if let Some(sentence) = Self::best_sentence(&corpus, request) {
                sentence
            } else {
                let seed = if corpus.is_empty() {
                    Self::domain_seed(request, &self.behavior)
                } else {
                    format!("{request} {corpus}")
                };
                Self::symbolic_continuation(seed)
            }
        };

        #[cfg(not(feature = "net"))]
        let result = {
            let seed = Self::domain_seed(request, &self.behavior);
            Self::symbolic_continuation(seed)
        };

        self.add_message(Message::new("user", request.to_string()));
        self.add_message(Message::new("assistant", result.clone()));
        Ok(result)
    }

    fn domain_seed(request: &str, behavior: &str) -> String {
        const STOP: &[&str] = &[
            "a", "an", "the", "and", "or", "of", "to", "in", "is", "are", "be", "for", "on", "at",
            "by", "as", "it", "its",
        ];
        let domain_words: Vec<&str> = behavior
            .split_whitespace()
            .filter(|w| {
                let lw = w.to_ascii_lowercase();
                !STOP.contains(&lw.as_str()) && w.len() > 3
            })
            .take(6)
            .collect();

        let mut seed = request.to_string();
        if !domain_words.is_empty() {
            seed.push(' ');
            seed.push_str(&domain_words.join(" "));
        }
        if seed.split_whitespace().count() < 2 {
            seed.push_str(" and");
        }
        seed
    }

    /// Runs the symbolic predictor on a seed and returns the continuation.
    fn symbolic_continuation(seed: String) -> String {
        let mut predictor = TextPredictor::new(20, 40, 3);
        if let Ok(lex) = lmm::lexicon::Lexicon::load_system() {
            predictor = predictor.with_lexicon(lex);
        }
        predictor
            .predict_continuation(&seed, 120)
            .map(|c| format!("{} {}", seed.trim(), c.continuation.trim()))
            .unwrap_or(seed)
    }

    /// Returns the sentence from `corpus` with the highest token overlap with `query`.
    /// Returns `None` if no sentence has meaningful overlap.
    #[cfg(feature = "net")]
    fn best_sentence(corpus: &str, query: &str) -> Option<String> {
        use std::collections::HashSet;
        let query_tokens: HashSet<String> = query
            .split_whitespace()
            .map(|w| w.to_ascii_lowercase())
            .collect();

        corpus
            .split(['.', '!', '?'])
            .map(str::trim)
            .filter(|s| s.split_whitespace().count() >= 5)
            .map(|sentence| {
                let sentence_tokens: HashSet<String> = sentence
                    .split_whitespace()
                    .map(|w| w.to_ascii_lowercase())
                    .collect();
                let overlap = query_tokens.intersection(&sentence_tokens).count();
                (overlap, sentence.to_string())
            })
            .filter(|(overlap, _)| *overlap >= 2)
            .max_by_key(|(overlap, _)| *overlap)
            .map(|(_, sentence)| sentence)
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

    /// Runs the closed-loop **ThinkLoop** reasoning cycle toward `goal`.
    ///
    /// The agent transitions through `Status::Thinking` and back to
    /// `Status::Completed`. At the end of the run the cold-store archive is
    /// serialised into the agent's `long_term_memory` so knowledge persists
    /// across multiple `think()` calls.
    ///
    /// ## Parameters
    ///
    /// * `goal` - natural-language task description (the setpoint).
    ///
    /// Defaults used internally:
    /// - `max_iterations = 10`
    /// - `convergence_threshold = 0.25`
    /// - `k_proportional = 1.0`
    /// - `k_integral = 0.05`
    ///
    /// Use [`LmmAgent::think_with`] for fine-grained control.
    ///
    /// # Examples
    ///
    /// ```rust
    /// #[tokio::main]
    /// async fn main() {
    ///     use lmm_agent::agent::LmmAgent;
    ///
    ///     let mut agent = LmmAgent::new("Researcher".into(), "Explore Rust.".into());
    ///     let result = agent.think("What is Rust ownership?").await.unwrap();
    ///     assert!(result.steps > 0);
    ///     assert!(result.final_error >= 0.0 && result.final_error <= 1.0);
    /// }
    /// ```
    pub async fn think(&mut self, goal: &str) -> Result<ThinkResult> {
        self.think_with(goal, 10, 0.25, 1.0, 0.05).await
    }

    /// Like [`think`](Self::think) but exposes all ThinkLoop parameters.
    ///
    /// # Arguments
    ///
    /// * `goal`                  - natural-language goal / setpoint.
    /// * `max_iterations`        - maximum feedback loop iterations (≥ 1).
    /// * `convergence_threshold` - Jaccard error threshold ∈ [0, 1].
    /// * `k_proportional`        - proportional gain Kp.
    /// * `k_integral`            - integral gain Ki.
    ///
    /// # Examples
    ///
    /// ```rust
    /// #[tokio::main]
    /// async fn main() {
    ///     use lmm_agent::agent::LmmAgent;
    ///
    ///     let mut agent = LmmAgent::new("Researcher".into(), "Explore Rust.".into());
    ///     let result = agent
    ///         .think_with("Rust memory safety", 5, 0.3, 1.0, 0.05)
    ///         .await
    ///         .unwrap();
    ///     assert!(result.steps <= 5);
    /// }
    /// ```
    pub async fn think_with(
        &mut self,
        goal: &str,
        max_iterations: usize,
        convergence_threshold: f64,
        k_proportional: f64,
        k_integral: f64,
    ) -> Result<ThinkResult> {
        self.status = Status::Thinking;

        let mut oracle = SearchOracle::new(5);
        let mut lp = ThinkLoop::new(
            goal,
            max_iterations,
            convergence_threshold,
            k_proportional,
            k_integral,
        );
        let result = lp.run(&mut oracle).await;

        for entry in lp.cold.all() {
            self.long_term_memory
                .push(Message::new("think", entry.content.clone()));
        }

        self.add_message(Message::new("think:goal", goal.to_string()));
        self.add_message(Message::new(
            "think:result",
            format!(
                "converged={} steps={} error={:.3}",
                result.converged, result.steps, result.final_error
            ),
        ));

        self.status = Status::Completed;
        Ok(result)
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

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
