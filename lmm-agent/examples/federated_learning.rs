// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Federated Learning Example
//!
//! Demonstrates **self-federated Q-table aggregation** between two independent
//! agents that research different topics, then exchange knowledge without any
//! central parameter server.
//!
//! Flow:
//!
//! 1. `Agent Alpha` researches Rust concurrency.
//! 2. `Agent Beta` researches Rust type system.
//! 3. Alpha exports its [`AgentSnapshot`] and Beta merges it.
//! 4. Beta in turn sends its snapshot to Alpha.
//! 5. Both agents print their merged Q-table stats and knowledge index sizes.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example federated_learning -p lmm-agent
//! ```

use lmm_agent::cognition::learning::config::LearningConfig;
use lmm_agent::cognition::learning::engine::LearningEngine;
use lmm_agent::prelude::*;

#[tokio::main]
async fn main() {
    println!("HELM - Self-Federated Multi-Agent Learning");

    let cfg = LearningConfig::builder()
        .alpha(0.12)
        .gamma(0.9)
        .epsilon(0.2)
        .federated_blend(0.5)
        .distill_threshold(0.15)
        .build();

    let mut alpha = LmmAgent::builder()
        .persona("Agent Alpha")
        .behavior("Rust concurrency and async runtime.")
        .learning_engine(LearningEngine::new(cfg.clone()))
        .build();

    let mut beta = LmmAgent::builder()
        .persona("Agent Beta")
        .behavior("Rust type system and trait objects.")
        .learning_engine(LearningEngine::new(cfg.clone()))
        .build();

    println!("Phase 1: Agents research their individual topics\n");

    alpha
        .think("Rust async await tokio concurrency channels")
        .await
        .unwrap();
    let alpha_eps = alpha.learning_engine.as_ref().unwrap().episode_count();
    let alpha_states = alpha
        .learning_engine
        .as_ref()
        .unwrap()
        .q_table()
        .state_count();
    println!(
        "  Alpha | episodes={alpha_eps} q-states={alpha_states} knowledge-chunks={}",
        alpha.knowledge_index.len()
    );

    beta.think("Rust traits generics type system polymorphism")
        .await
        .unwrap();
    let beta_eps = beta.learning_engine.as_ref().unwrap().episode_count();
    let beta_states = beta
        .learning_engine
        .as_ref()
        .unwrap()
        .q_table()
        .state_count();
    println!(
        "  Beta  | episodes={beta_eps} q-states={beta_states} knowledge-chunks={}",
        beta.knowledge_index.len()
    );

    println!("\nPhase 2: Bidirectional federated merge\n");

    let alpha_snapshot = alpha
        .learning_engine
        .as_ref()
        .unwrap()
        .export_snapshot(alpha.id.clone());

    let beta_snapshot = beta
        .learning_engine
        .as_ref()
        .unwrap()
        .export_snapshot(beta.id.clone());

    beta.learning_engine
        .as_mut()
        .unwrap()
        .federate(&alpha_snapshot);
    alpha
        .learning_engine
        .as_mut()
        .unwrap()
        .federate(&beta_snapshot);

    let alpha_merged = alpha
        .learning_engine
        .as_ref()
        .unwrap()
        .q_table()
        .state_count();
    let beta_merged = beta
        .learning_engine
        .as_ref()
        .unwrap()
        .q_table()
        .state_count();

    println!(
        "  Alpha merged | q-states={alpha_merged} merges={}",
        alpha
            .learning_engine
            .as_ref()
            .unwrap()
            .aggregator()
            .merge_count
    );
    println!(
        "  Beta  merged | q-states={beta_merged} merges={}",
        beta.learning_engine
            .as_ref()
            .unwrap()
            .aggregator()
            .merge_count
    );

    println!("\nPhase 3: Post-federation think on shared topic\n");

    alpha
        .think("Rust async type system integration")
        .await
        .unwrap();
    beta.think("Rust async type system integration")
        .await
        .unwrap();

    println!(
        "  Alpha post-federation | q-states={} episodes={}",
        alpha
            .learning_engine
            .as_ref()
            .unwrap()
            .q_table()
            .state_count(),
        alpha.learning_engine.as_ref().unwrap().episode_count()
    );
    println!(
        "  Beta  post-federation | q-states={} episodes={}",
        beta.learning_engine
            .as_ref()
            .unwrap()
            .q_table()
            .state_count(),
        beta.learning_engine.as_ref().unwrap().episode_count()
    );

    println!("\nRecall learned actions");
    for (label, agent) in [("Alpha", &mut alpha), ("Beta", &mut beta)] {
        if let Some(action) = agent.recall_learned("rust async type system", 0) {
            println!("  {label}: {action:?}");
        }
    }

    println!("\n[Done] Federated learning complete.");
}
