// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm_agent::agent::LmmAgent;
use lmm_agent::cognition::knowledge::KnowledgeIndex;
use lmm_agent::cognition::learning::config::LearningConfig;
use lmm_agent::cognition::learning::elastic::ElasticMemoryGuard;
use lmm_agent::cognition::learning::engine::LearningEngine;
use lmm_agent::cognition::learning::informal::InformalLearner;
use lmm_agent::cognition::learning::meta::MetaAdapter;
use lmm_agent::cognition::learning::q_table::{ActionKey, QTable};
use lmm_agent::cognition::memory::{ColdStore, MemoryEntry};
use lmm_agent::cognition::signal::CognitionSignal;
use std::collections::HashMap;

#[test]
fn engine_records_step_and_updates_q_table() {
    let mut engine = LearningEngine::new(LearningConfig::builder().epsilon(0.0).build());
    let text = "rust memory safety ownership borrow checker";
    let sig = CognitionSignal::new(0, text.into(), text.into(), 1.0, 0.0);
    let s0 = QTable::state_key(text);
    let s1 = QTable::state_key("rust memory safety ownership borrow");
    engine.record_step(&sig, s0, ActionKey::Narrow, s1);

    assert!(
        !engine.q_table().is_empty(),
        "Q-table should have at least one state"
    );
    assert_eq!(engine.q_table().state_count(), 1);
    assert!(
        engine.q_table().max_q(s0) > 0.0,
        "max Q-value for recorded state should be positive"
    );
}

#[test]
fn engine_episode_count_increments_after_end_of_episode() {
    let mut engine = LearningEngine::new(LearningConfig::default());
    assert_eq!(engine.episode_count(), 0);

    let cold = ColdStore::default();
    let mut idx = KnowledgeIndex::new();
    engine.end_of_episode(&cold, &mut idx, "test goal", 0.5);
    assert_eq!(engine.episode_count(), 1);

    engine.end_of_episode(&cold, &mut idx, "another goal", 0.6);
    assert_eq!(engine.episode_count(), 2);
}

#[test]
fn engine_distills_cold_entries_into_knowledge_index() {
    let mut engine = LearningEngine::new(
        LearningConfig::builder()
            .distill_threshold(0.1)
            .distill_top_k(5)
            .build(),
    );

    let mut cold = ColdStore::default();
    cold.promote(MemoryEntry::new(
        "Rust ownership model prevents data races at compile time through borrow checker.".into(),
        0.9,
        0,
    ));
    cold.promote(MemoryEntry::new(
        "Memory safety without garbage collection is a key goal of Rust language design.".into(),
        0.8,
        1,
    ));

    let mut idx = KnowledgeIndex::new();
    engine.end_of_episode(&cold, &mut idx, "rust memory safety", 0.85);

    assert!(
        !idx.is_empty(),
        "knowledge index should have distilled entries"
    );
}

#[test]
fn engine_meta_adapter_stores_prototype_after_episode() {
    let mut engine = LearningEngine::new(LearningConfig::default());

    let s0 = QTable::state_key("machine learning algorithms");
    let s1 = QTable::state_key("gradient descent optimization");
    let sig = CognitionSignal::new(
        0,
        "machine learning".into(),
        "gradient descent".into(),
        0.8,
        0.0,
    );
    engine.record_step(&sig, s0, ActionKey::Expand, s1);

    let mut cold = ColdStore::default();
    cold.promote(MemoryEntry::new(
        "gradient descent optimizes model weights".into(),
        0.8,
        0,
    ));
    let mut idx = KnowledgeIndex::new();
    engine.end_of_episode(&cold, &mut idx, "machine learning algorithms", 0.8);

    assert_eq!(engine.meta().len(), 1);
}

#[test]
fn engine_epsilon_decays_after_each_episode() {
    let cfg = LearningConfig::builder()
        .epsilon(0.5)
        .epsilon_decay(0.9)
        .epsilon_min(0.01)
        .build();
    let mut engine = LearningEngine::new(cfg);
    let initial_eps = engine.q_table().epsilon;

    let cold = ColdStore::default();
    let mut idx = KnowledgeIndex::new();
    engine.end_of_episode(&cold, &mut idx, "goal", 0.5);

    assert!(engine.q_table().epsilon < initial_eps);
}

#[test]
fn agent_builder_accepts_learning_engine() {
    let agent = LmmAgent::builder()
        .persona("Learner")
        .behavior("Learn Rust ownership.")
        .learning_engine(LearningEngine::new(LearningConfig::default()))
        .build();

    assert!(agent.learning_engine.is_some());
}

#[test]
fn agent_recall_learned_returns_action_when_engine_present() {
    let mut agent = LmmAgent::builder()
        .persona("Learner")
        .behavior("Rust.")
        .learning_engine(LearningEngine::new(LearningConfig::default()))
        .build();

    let action = agent.recall_learned("rust ownership safety", 0);
    assert!(action.is_some());
    assert!(ActionKey::all().contains(&action.unwrap()));
}

#[test]
fn agent_without_engine_recall_learned_returns_none() {
    let mut agent = LmmAgent::new("NoEngine".into(), "Test.".into());
    assert!(agent.recall_learned("any query", 0).is_none());
}

#[tokio::test]
async fn agent_think_with_engine_populates_q_table() {
    let mut agent = LmmAgent::builder()
        .persona("Learning Agent")
        .behavior("Understand Rust memory model.")
        .learning_engine(LearningEngine::new(LearningConfig::default()))
        .build();

    let result = agent.think("Rust ownership and borrowing").await.unwrap();
    assert!(result.steps > 0);

    let engine = agent.learning_engine.as_ref().unwrap();
    assert!(
        !engine.q_table().is_empty(),
        "Q-table should be populated after think()"
    );
    assert_eq!(engine.episode_count(), 1);
}

#[test]
fn elastic_guard_pins_frequently_activated_entries() {
    let mut guard = ElasticMemoryGuard::new(3, 0.5);
    let content = "ownership borrow checker prevents data races";

    assert!(!guard.is_pinned(content));
    guard.observe_activation(content);
    guard.observe_activation(content);
    assert!(!guard.is_pinned(content));
    guard.observe_activation(content);
    assert!(guard.is_pinned(content));
    assert_eq!(guard.activation_count(content), 3);
}

#[test]
fn informal_learner_mines_high_pmi_pairs() {
    let mut il = InformalLearner::new(0.3, 2, 0.0);
    il.observe("rust ownership memory concurrent threads", 0.9);
    il.observe("rust ownership prevents data races concurrent", 0.85);
    il.observe("memory concurrent safe threads rust", 0.8);

    let pairs = il.high_pmi_pairs(0.0);
    assert!(!pairs.is_empty(), "should find co-occurring token pairs");
    assert!(pairs[0].2 >= 0.0, "PMI score should be non-negative");
}

#[test]
fn q_table_update_follows_bellman_equation() {
    let mut qt = QTable::new(0.1, 0.9, 0.0, 1.0, 0.0);
    let s = QTable::state_key("theory test state");
    let s2 = QTable::state_key("next state result");

    qt.update(s, ActionKey::Expand, 1.0, s2);
    let expected = 0.1 * (1.0 + 0.9 * 0.0 - 0.0);
    assert!((qt.q_value(s, ActionKey::Expand) - expected).abs() < 1e-9);
}

#[test]
fn q_table_merge_blends_values_correctly() {
    let mut local = QTable::new(0.1, 0.9, 0.0, 1.0, 0.0);
    let mut remote = QTable::new(0.1, 0.9, 0.0, 1.0, 0.0);
    let s = QTable::state_key("shared state query");

    local.update(s, ActionKey::Narrow, 0.4, s);
    remote.update(s, ActionKey::Narrow, 0.8, s);

    local.merge(&remote, 0.5);
    let merged = local.q_value(s, ActionKey::Narrow);
    let local_val = 0.04_f64;
    let remote_val = remote.q_value(s, ActionKey::Narrow);
    let expected = 0.5 * local_val + 0.5 * remote_val;
    assert!((merged - expected).abs() < 1e-6);
}

#[test]
fn meta_adapter_adapts_from_stored_prototype() {
    let mut adapter = MetaAdapter::new(3);
    let offsets = HashMap::from([(ActionKey::Expand, 0.7), (ActionKey::Narrow, 0.4)]);
    adapter.record_episode("rust async futures tokio", offsets, 0.8);

    let result = adapter.adapt("async futures rust concurrency");
    assert!(!result.is_empty(), "should adapt from similar prototype");
    assert!(result.contains_key(&ActionKey::Expand) || result.contains_key(&ActionKey::Narrow));
}
