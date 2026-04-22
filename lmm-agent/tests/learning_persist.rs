// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm_agent::agent::LmmAgent;
use lmm_agent::cognition::knowledge::KnowledgeIndex;
use lmm_agent::cognition::learning::config::LearningConfig;
use lmm_agent::cognition::learning::engine::LearningEngine;
use lmm_agent::cognition::learning::q_table::{ActionKey, QTable};
use lmm_agent::cognition::learning::store::LearningStore;
use lmm_agent::cognition::memory::{ColdStore, MemoryEntry};
use lmm_agent::cognition::signal::CognitionSignal;
use std::fs;
use std::path::Path;

#[test]
fn save_and_load_round_trips_q_table() {
    let path = Path::new("/tmp/helm_test_roundtrip.json");

    let mut engine = LearningEngine::new(LearningConfig::default());
    let s0 = QTable::state_key("persistent state query");
    let s1 = QTable::state_key("observed result text");
    let sig = CognitionSignal::new(
        0,
        "persistent state query".into(),
        "observed result text".into(),
        0.9,
        0.0,
    );
    engine.record_step(&sig, s0, ActionKey::Expand, s1);

    LearningStore::save(&engine, path).unwrap();

    let loaded = LearningStore::load(path).unwrap();
    assert_eq!(
        loaded.q_table().state_count(),
        engine.q_table().state_count()
    );
    assert!(
        (loaded.q_table().q_value(s0, ActionKey::Expand)
            - engine.q_table().q_value(s0, ActionKey::Expand))
        .abs()
            < 1e-9
    );

    std::fs::remove_file(path).ok();
}

#[test]
fn save_and_load_preserves_episode_count() {
    let path = Path::new("/tmp/helm_test_episode.json");

    let mut engine = LearningEngine::new(LearningConfig::default());
    let cold = ColdStore::default();
    let mut idx = KnowledgeIndex::new();
    engine.end_of_episode(&cold, &mut idx, "goal one", 0.6);
    engine.end_of_episode(&cold, &mut idx, "goal two", 0.7);

    LearningStore::save(&engine, path).unwrap();
    let loaded = LearningStore::load(path).unwrap();

    assert_eq!(loaded.episode_count(), 2);
    fs::remove_file(path).ok();
}

#[test]
fn save_and_load_preserves_config() {
    let path = Path::new("/tmp/helm_test_config.json");

    let cfg = LearningConfig::builder()
        .alpha(0.05)
        .gamma(0.95)
        .epsilon(0.15)
        .distill_top_k(12)
        .build();
    let engine = LearningEngine::new(cfg.clone());

    LearningStore::save(&engine, path).unwrap();
    let loaded = LearningStore::load(path).unwrap();

    assert_eq!(loaded.config().alpha, cfg.alpha);
    assert_eq!(loaded.config().gamma, cfg.gamma);
    assert_eq!(loaded.config().distill_top_k, cfg.distill_top_k);

    std::fs::remove_file(path).ok();
}

#[test]
fn save_and_load_preserves_distiller_fingerprints() {
    let path = Path::new("/tmp/helm_test_distiller.json");

    let mut engine = LearningEngine::new(LearningConfig::builder().distill_threshold(0.1).build());
    let mut cold = ColdStore::default();
    cold.promote(MemoryEntry::new(
        "Rust borrow checker enforces memory safety without runtime cost at compile time.".into(),
        0.9,
        0,
    ));
    let mut idx = KnowledgeIndex::new();
    engine.end_of_episode(&cold, &mut idx, "rust borrow checker", 0.9);
    let ingested_before = engine.distiller().total_ingested;

    LearningStore::save(&engine, path).unwrap();
    let mut loaded = LearningStore::load(path).unwrap();

    let mut idx2 = KnowledgeIndex::new();
    loaded.end_of_episode(&cold, &mut idx2, "rust borrow checker", 0.9);
    assert_eq!(
        loaded.distiller().total_ingested,
        ingested_before,
        "re-ingesting same entries should be no-op due to fingerprint cache"
    );

    std::fs::remove_file(path).ok();
}

#[tokio::test]
async fn agent_save_and_load_learning_restores_engine() {
    let path = Path::new("/tmp/helm_test_agent.json");

    let mut agent = LmmAgent::builder()
        .persona("Persistent Learner")
        .behavior("Learn and remember Rust concepts.")
        .learning_engine(LearningEngine::new(LearningConfig::default()))
        .build();

    agent
        .think("What is the Rust borrow checker?")
        .await
        .unwrap();
    let steps_before = agent.learning_engine.as_ref().unwrap().episode_count();

    agent.save_learning(path).unwrap();

    let mut agent2 = LmmAgent::new("Fresh Agent".into(), "New task.".into());
    agent2.load_learning(path).unwrap();

    assert!(agent2.learning_engine.is_some());
    assert_eq!(
        agent2.learning_engine.as_ref().unwrap().episode_count(),
        steps_before
    );

    fs::remove_file(path).ok();
}
