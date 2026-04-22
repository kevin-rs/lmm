// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Full Agent Lifecycle Example
//!
//! Exercises **every non-net component** of `lmm-agent` in a single, runnable demo.
//!
//! | Step | Component                                                        |
//! |------|------------------------------------------------------------------|
//! | 1    | Identity: persona, behavior, Capability, Tool                    |
//! | 2    | Memory: hot store, LTM, ContextManager                           |
//! | 3    | Knowledge: RawText ingest, answer_from_knowledge, query_knowledge|
//! | 4    | Planning: Planner, Goal queue, complete_goal                     |
//! | 5    | Reasoning: think_with() - closed-loop PI ThinkLoop              |
//! | 6    | Generation: generate() - knowledge-first, symbolic fallback      |
//! | 7    | HELM: LearningEngine stats, recall_learned, save/load            |
//! | 8    | Reflection: custom eval_fn report                                |
//! | 9    | Federation: PeerAgent merges senior's Q-table via federate()     |
//! | 10   | Orchestration: AutoAgent runs both agents concurrently           |
//!
//! ```bash
//! cargo run --example full_lifecycle -p lmm-agent
//! ```

use lmm_agent::cognition::learning::config::LearningConfig;
use lmm_agent::cognition::learning::engine::LearningEngine;
use lmm_agent::cognition::learning::store::LearningStore;
use lmm_agent::prelude::*;
use std::borrow::Cow;
use std::path::Path;

const HELM_PATH: &str = "/tmp/helm_full_lifecycle.json";
fn section(n: u8, title: &str) {
    println!("\n>>> {n}. {title}");
}

fn research_eval(agent: &dyn Agent) -> Cow<'static, str> {
    let mem_count = agent.memory().len();
    let has_planner = agent.planner().is_some();
    let facts_count = agent.knowledge().facts.len();
    Cow::Owned(format!(
        "  Reflection: {mem_count} hot messages · {facts_count} facts · planner={has_planner}"
    ))
}

#[derive(Debug, Default, Auto)]
pub struct ResearchAgent {
    pub agent: LmmAgent,
}

#[async_trait]
impl Executor for ResearchAgent {
    async fn execute<'a>(
        &'a mut self,
        _task: &'a mut Task,
        _execute: bool,
        _browse: bool,
        _max_tries: u64,
    ) -> Result<()> {
        section(1, "IDENTITY");
        println!("  Persona      : {}", self.agent.persona);
        println!("  Behavior     : {}", self.agent.behavior);
        println!("  Status       : {} → Active", self.agent.status);
        println!("  Capabilities : {}", self.agent.capabilities.len());
        println!("  Tools        : {}", self.agent.tools.len());
        for tool in &self.agent.tools {
            println!("    [{:?}] {}", tool.name, tool.description);
        }

        section(2, "MEMORY");
        self.agent.update(Status::Active);
        let boot_msg = Message::new("system", "Research session initialised.");
        self.agent.add_message(boot_msg.clone());
        let _ = self.save_ltm(boot_msg).await;
        self.agent
            .context
            .focus_topics
            .push("Rust systems programming".into());
        self.agent
            .context
            .recent_messages
            .push(Message::new("user", "Begin Rust memory safety research."));
        println!("  Hot memory   : {} message(s)", self.agent.memory.len());
        println!("  Focus topics : {:?}", self.agent.context.focus_topics);
        let ltm = self.get_ltm().await.unwrap_or_default();
        println!("  LTM loaded   : {} message(s)", ltm.len());

        section(3, "KNOWLEDGE INGESTION");
        let docs = [
            "Rust ownership model enforces single-owner semantics at compile time. \
             The borrow checker prevents dangling pointers and data races without \
             garbage collection or runtime overhead.",
            "Rust lifetimes annotate how long references stay valid. The borrow \
             checker uses them to eliminate use-after-free vulnerabilities at compile time.",
            "Rust traits enable zero-cost abstractions through static dispatch. \
             Generic functions are monomorphised at compile time, producing code as \
             fast as hand-written C with no virtual dispatch overhead.",
            "Async Rust uses a state-machine model via async/await. The Tokio runtime \
             provides a multi-threaded executor, timers, and I/O primitives. \
             Futures are zero-cost when not polled.",
        ];
        let mut total = 0usize;
        for doc in &docs {
            total += self
                .agent
                .ingest(KnowledgeSource::RawText((*doc).into()))
                .await?;
        }
        println!(
            "  Ingested {} chunk(s) from {} document(s)",
            total,
            docs.len()
        );
        println!(
            "  Index size   : {} chunk(s)",
            self.agent.knowledge_index.len()
        );

        let q = "How does the borrow checker prevent memory errors?";
        if let Some(ans) = self.agent.answer_from_knowledge(q) {
            println!("  Q: \"{q}\"");
            println!("  A: \"{}...\"", &ans[..ans.len().min(130)]);
        }
        let passages = self.agent.query_knowledge("Rust zero-cost abstractions", 2);
        println!("  Top-2 passages for 'Rust zero-cost abstractions':");
        for (i, p) in passages.iter().enumerate() {
            println!("    [{}] {}...", i + 1, &p[..p.len().min(80)]);
        }

        section(4, "PLANNING");
        let goals: Vec<String> = self
            .agent
            .planner
            .as_ref()
            .map(|p| {
                p.current_plan
                    .iter()
                    .map(|g| g.description.clone())
                    .collect()
            })
            .unwrap_or_default();
        println!("  {} goal(s) in queue:", goals.len());
        for (i, g) in goals.iter().enumerate() {
            println!(
                "    [{}] (priority {}) {g}",
                i + 1,
                self.agent
                    .planner
                    .as_ref()
                    .and_then(|p| p.current_plan.get(i))
                    .map(|g| g.priority)
                    .unwrap_or(0)
            );
        }

        for goal in &goals {
            let resp = self
                .generate(&format!("{goal}:"))
                .await
                .unwrap_or_else(|_| "analysis pending".into());
            self.agent
                .add_message(Message::new("assistant", resp.clone()));
            self.agent.context.focus_topics.push(goal.clone().into());
            let _ = self.save_ltm(Message::new("assistant", resp.clone())).await;
            self.agent.complete_goal(goal);
        }
        let done = self
            .agent
            .planner
            .as_ref()
            .map(|p| p.is_done())
            .unwrap_or(false);
        let n_done = self
            .agent
            .planner
            .as_ref()
            .map(|p| p.completed_count())
            .unwrap_or(0);
        println!("  Progress: {n_done}/{} - all done: {done}", goals.len());

        section(5, "REASONING (ThinkLoop PI controller)");
        self.agent.update(Status::Thinking);
        let research_topic = "Rust memory safety: ownership, borrowing, lifetimes, concurrency";
        let result = self
            .agent
            .think_with(research_topic, 8, 0.15, 1.2, 0.08)
            .await?;
        println!("  Converged    : {}", result.converged);
        println!("  Steps        : {}", result.steps);
        println!("  Final error  : {:.4}", result.final_error);
        println!("  Signals      : {} step(s) recorded", result.signals.len());

        for (i, snippet) in result.memory_snapshot.iter().enumerate().take(3) {
            if snippet.split_whitespace().count() > 3 {
                self.agent
                    .knowledge
                    .insert(format!("research_{i}"), snippet.clone());
            }
        }
        for (i, msg) in self.agent.long_term_memory.iter().take(2).enumerate() {
            if msg.content.len() > 20 {
                self.agent.knowledge.insert(
                    format!("ltm_{i}"),
                    msg.content[..msg.content.len().min(80)].to_string(),
                );
            }
        }
        println!(
            "  Knowledge facts stored: {}",
            self.agent.knowledge.facts.len()
        );

        section(6, "GENERATION (knowledge-first fallback chain)");
        let prompts = [
            "The Rust borrow checker works by",
            "Zero-cost abstractions in Rust mean",
        ];
        for prompt in &prompts {
            let resp = self.generate(prompt).await.unwrap_or_else(|_| "n/a".into());
            self.agent
                .add_message(Message::new("assistant", resp.clone()));
            println!("  Prompt : \"{prompt}\"");
            println!("  Output : \"{}...\"", &resp[..resp.len().min(100)]);
        }

        section(7, "HELM LEARNING ENGINE");
        {
            let eng = self.agent.learning_engine.as_ref().unwrap();
            println!("  Episodes          : {}", eng.episode_count());
            println!("  Q-table states    : {}", eng.q_table().state_count());
            println!("  Q-table updates   : {}", eng.q_table().update_count);
            println!("  Distilled chunks  : {}", eng.distiller().total_ingested);
            println!("  Meta prototypes   : {}", eng.meta().len());
            println!("  PMI pairs tracked : {}", eng.informal().pair_count());
            println!(
                "  Elastic pinned    : {}",
                eng.elastic().pinned_contents().len()
            );
            println!(
                "  KnowledgeIndex    : {} chunk(s)",
                self.agent.knowledge_index.len()
            );
        }
        if let Some(action) = self.agent.recall_learned(research_topic, 0) {
            println!("  Recalled action   : {action:?}");
        }
        self.agent.save_learning(Path::new(HELM_PATH))?;
        println!("  State saved → {HELM_PATH}");
        let reloaded = LearningStore::load(Path::new(HELM_PATH))?;
        println!(
            "  Reloaded  → {} Q-states · {} episodes",
            reloaded.q_table().state_count(),
            reloaded.episode_count()
        );

        section(8, "REFLECTION");
        self.agent.update(Status::InUnitTesting);
        if let Some(reflection) = &self.agent.reflection {
            let report = (reflection.evaluation_fn)(&self.agent);
            println!("{report}");
            println!("  Recent logs: {:?}", reflection.recent_logs);
        }

        self.agent.update(Status::Completed);
        section(9, "FINAL STATE SNAPSHOT");
        println!("  Status          : {}", self.agent.status);
        println!("  Hot memory      : {} message(s)", self.agent.memory.len());
        println!(
            "  LTM messages    : {} message(s)",
            self.agent.long_term_memory.len()
        );
        println!("  Knowledge facts : {}", self.agent.knowledge.facts.len());
        println!(
            "  Focus topics    : {}",
            self.agent.context.focus_topics.len()
        );
        println!(
            "  Goals done      : {}/{}",
            self.agent
                .planner
                .as_ref()
                .map(|p: &Planner| p.completed_count())
                .unwrap_or(0),
            self.agent
                .planner
                .as_ref()
                .map(|p: &Planner| p.current_plan.len())
                .unwrap_or(0)
        );

        Ok(())
    }
}

#[derive(Debug, Default, Auto)]
pub struct PeerAgent {
    pub agent: LmmAgent,
}

#[async_trait]
impl Executor for PeerAgent {
    async fn execute<'a>(
        &'a mut self,
        _task: &'a mut Task,
        _execute: bool,
        _browse: bool,
        _max_tries: u64,
    ) -> Result<()> {
        section(10, "PEER AGENT - Federated Q-table Exchange");
        self.agent.update(Status::Thinking);

        let goal = "Rust concurrency: Send, Sync, Arc, Mutex, and channels";
        let result = self.agent.think(goal).await?;
        println!("  Topic    : \"{goal}\"");
        println!(
            "  Steps    : {}  converged: {}",
            result.steps, result.converged
        );

        {
            let eng = self.agent.learning_engine.as_ref().unwrap();
            println!(
                "  Local Q-states before federate: {}",
                eng.q_table().state_count()
            );
        }

        let senior = LearningStore::load(Path::new(HELM_PATH))?;
        let snap = senior.export_snapshot("senior-research-agent");
        self.agent.learning_engine.as_mut().unwrap().federate(&snap);

        {
            let eng = self.agent.learning_engine.as_ref().unwrap();
            println!(
                "  Q-states after federate  : {}",
                eng.q_table().state_count()
            );
            println!(
                "  Merge count              : {}",
                eng.aggregator().merge_count
            );
        }

        if let Some(action) = self.agent.recall_learned(goal, 0) {
            println!("  Recalled action (post-federate): {action:?}");
        }

        self.agent.update(Status::Completed);
        Ok(())
    }
}

#[tokio::main]
async fn main() {
    let helm_cfg = LearningConfig::builder()
        .alpha(0.15)
        .gamma(0.92)
        .epsilon(0.30)
        .epsilon_decay(0.95)
        .epsilon_min(0.01)
        .distill_top_k(8)
        .distill_threshold(0.1)
        .pmi_min_count(1)
        .elastic_pin_count(2)
        .federated_blend(0.5)
        .build();

    let senior = ResearchAgent {
        agent: LmmAgent::builder()
            .persona("Senior Research Agent")
            .behavior("Rust systems programming: memory safety, concurrency, and type system.")
            .planner(Planner {
                current_plan: vec![
                    Goal {
                        description: "Survey Rust ownership semantics".into(),
                        priority: 1,
                        completed: false,
                    },
                    Goal {
                        description: "Analyse borrow checker enforcement".into(),
                        priority: 2,
                        completed: false,
                    },
                    Goal {
                        description: "Compare Rust lifetimes to C++ RAII".into(),
                        priority: 3,
                        completed: false,
                    },
                    Goal {
                        description: "Map async Rust patterns to Tokio".into(),
                        priority: 4,
                        completed: false,
                    },
                ],
            })
            .context(ContextManager {
                recent_messages: vec![],
                focus_topics: vec![],
            })
            .capabilities(
                [
                    Capability::CodeGen,
                    Capability::WebSearch,
                    Capability::ApiIntegration,
                ]
                .into_iter()
                .collect(),
            )
            .tools(vec![
                Tool {
                    name: ToolName::Search,
                    description: "Search the knowledge base for relevant passages.".into(),
                    invoke: |q| format!("Search results for: {q}"),
                },
                Tool {
                    name: ToolName::Summarize,
                    description: "Summarise a block of text into key points.".into(),
                    invoke: |text| format!("Summary: {}", &text[..text.len().min(60)]),
                },
            ])
            .reflection(Reflection {
                recent_logs: vec!["Initial setup complete.".into(), "Corpus ingested.".into()],
                evaluation_fn: research_eval,
            })
            .learning_engine(LearningEngine::new(helm_cfg.clone()))
            .build(),
    };

    println!("\nPHASE 1: Senior Research Agent");
    match AutoAgent::default()
        .with(agents![senior])
        .max_tries(1)
        .build()
        .expect("Failed to build AutoAgent")
        .run()
        .await
    {
        Ok(msg) => println!("\nPhase 1 Result: {msg}"),
        Err(e) => eprintln!("\n[Phase 1] Error: {e:?}"),
    }

    let peer = PeerAgent {
        agent: LmmAgent::builder()
            .persona("Peer Agent")
            .behavior("Rust concurrency and async runtimes.")
            .learning_engine(LearningEngine::new(helm_cfg))
            .build(),
    };

    println!("\nPHASE 2: Peer Agent (Federated Exchange)");
    match AutoAgent::default()
        .with(agents![peer])
        .max_tries(1)
        .build()
        .expect("Failed to build AutoAgent")
        .run()
        .await
    {
        Ok(msg) => println!("\nPhase 2 Result: {msg}"),
        Err(e) => eprintln!("\n[Phase 2] Error: {e:?}"),
    }

    println!("\nPersisted HELM state: {HELM_PATH}");
}
