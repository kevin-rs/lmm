// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Learning Agent Example
//!
//! Demonstrates the full **HELM lifelong learning lifecycle** within a running
//! `lmm-agent` session:
//!
//! 1. Creates an agent with all HELM learning modes enabled.
//! 2. Runs three successive `think()` calls on related topics.
//! 3. After each episode, prints Q-table stats, distilled knowledge chunks,
//!    informal PMI pairs, and elastic memory guard status.
//! 4. Saves the learning state to disk and reloads it, proving persistence.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example learning_agent -p lmm-agent
//! # With DuckDuckGo enrichment:
//! cargo run --example learning_agent -p lmm-agent --features net
//! ```

use lmm_agent::cognition::learning::config::LearningConfig;
use lmm_agent::cognition::learning::engine::LearningEngine;
use lmm_agent::cognition::learning::store::LearningStore;
use lmm_agent::prelude::*;
use std::path::Path;

const PERSIST_PATH: &str = "/tmp/helm_learning_agent.json";

#[derive(Debug, Default, Auto)]
pub struct LearningDemoAgent {
    pub agent: LmmAgent,
}

#[async_trait]
impl Executor for LearningDemoAgent {
    async fn execute<'a>(
        &'a mut self,
        _task: &'a mut Task,
        _execute: bool,
        _browse: bool,
        _max_tries: u64,
    ) -> Result<()> {
        let goals = [
            "Rust ownership model and memory safety without garbage collection",
            "Rust borrowing rules and the borrow checker enforcement",
            "Rust lifetimes and how they relate to borrow checker constraints",
        ];

        println!("HELM - Hybrid Equation-based Lifelong Memory");

        for (episode, goal) in goals.iter().enumerate() {
            println!("Episode {}", episode + 1);
            println!("Goal: {goal}");

            self.agent.update(Status::Thinking);
            let result = self.agent.think(goal).await?;

            let engine = self.agent.learning_engine.as_ref().unwrap();
            println!(
                "  ThinkLoop : converged={} steps={} error={:.3}",
                result.converged, result.steps, result.final_error
            );
            println!(
                "  Q-table   : {} states  {} updates",
                engine.q_table().state_count(),
                engine.q_table().update_count
            );
            println!("  Episodes  : {}", engine.episode_count());
            println!(
                "  Distilled : {} total chunks ingested",
                engine.distiller().total_ingested
            );
            println!("  Meta      : {} prototypes stored", engine.meta().len());
            println!(
                "  PMI pairs : {} token pairs tracked",
                engine.informal().pair_count()
            );
            println!(
                "  Elastic   : {} unique entries tracked  {} pinned",
                engine.elastic().unique_entry_count(),
                engine.elastic().pinned_contents().len()
            );
            println!(
                "  KnowledgeIndex: {} chunks total",
                self.agent.knowledge_index.len()
            );
            println!();
        }

        println!("Recall learned action");
        let query = "rust ownership borrow lifetime";
        if let Some(action) = self.agent.recall_learned(query, 0) {
            println!("Query: \"{query}\"");
            println!("Best Q-action: {action:?}");
        }

        println!("Persistence round-trip");
        self.agent.save_learning(Path::new(PERSIST_PATH))?;
        println!("Saved to {PERSIST_PATH}");

        let loaded = LearningStore::load(Path::new(PERSIST_PATH))?;
        println!(
            "Loaded  → Q-states={} episodes={}",
            loaded.q_table().state_count(),
            loaded.episode_count()
        );

        self.agent.update(Status::Completed);
        println!("\n  Status: {}", self.agent.status);
        println!("  Hot memory  : {} messages", self.agent.memory.len());
        println!(
            "  Long-term   : {} messages",
            self.agent.long_term_memory.len()
        );
        println!("  Knowledge   : {} facts", self.agent.knowledge.facts.len());

        Ok(())
    }
}

#[tokio::main]
async fn main() {
    let cfg = LearningConfig::builder()
        .alpha(0.15)
        .gamma(0.92)
        .epsilon(0.25)
        .distill_top_k(10)
        .distill_threshold(0.2)
        .pmi_min_count(1)
        .build();

    let agent = LearningDemoAgent::new(
        "HELM Learning Agent".into(),
        "Explore Rust memory model concepts.".into(),
    );
    let mut agent = agent;
    agent.agent.learning_engine = Some(LearningEngine::new(cfg));

    match AutoAgent::default()
        .with(agents![agent])
        .max_tries(1)
        .build()
        .expect("Failed to build AutoAgent")
        .run()
        .await
    {
        Ok(msg) => println!("\n[AutoAgent] {msg}"),
        Err(e) => eprintln!("\n[AutoAgent] Error: {e:?}"),
    }
}
