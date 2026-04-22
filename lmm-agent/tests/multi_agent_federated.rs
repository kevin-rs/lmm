// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm_agent::agent::LmmAgent;
use lmm_agent::cognition::learning::config::LearningConfig;
use lmm_agent::cognition::learning::engine::LearningEngine;
use lmm_agent::cognition::learning::federated::FederatedAggregator;
use lmm_agent::cognition::learning::q_table::{ActionKey, QTable};
use lmm_agent::cognition::signal::CognitionSignal;
use lmm_agent::types::AgentSnapshot;

#[test]
fn federated_merge_copies_remote_entries_to_empty_local() {
    let mut local = LearningEngine::new(LearningConfig::default());

    let mut remote = LearningEngine::new(LearningConfig::default());
    let topic = "shared query topic rust memory";
    let s = QTable::state_key(topic);
    let sig = CognitionSignal::new(0, topic.into(), topic.into(), 1.0, 0.0);
    remote.record_step(&sig, s, ActionKey::Narrow, s);

    let snapshot = remote.export_snapshot("agent-remote");
    local.federate(&snapshot);

    assert!(
        !local.q_table().is_empty(),
        "local should have inherited remote Q-values"
    );
    assert!(local.q_table().max_q(s) > 0.0);
}

#[test]
fn federated_merge_blends_overlapping_values() {
    let cfg = LearningConfig::builder()
        .federated_blend(0.5)
        .epsilon(0.0)
        .build();
    let mut local = LearningEngine::new(cfg);

    let topic = "overlapping state key rust ownership";
    let s = QTable::state_key(topic);
    let sig_local = CognitionSignal::new(0, topic.into(), topic.into(), 0.4, 0.0);
    local.record_step(&sig_local, s, ActionKey::Expand, s);
    let local_q_before = local.q_table().max_q(s);

    let mut remote = LearningEngine::new(LearningConfig::builder().epsilon(0.0).build());
    let sig_remote = CognitionSignal::new(0, topic.into(), topic.into(), 1.0, 0.0);
    remote.record_step(&sig_remote, s, ActionKey::Expand, s);

    let snapshot = remote.export_snapshot("agent-remote-high");
    local.federate(&snapshot);

    let blended = local.q_table().max_q(s);
    assert!(
        blended != local_q_before,
        "blended value should differ from local-only value"
    );
}

#[test]
fn federated_merge_count_increments() {
    let mut local = LearningEngine::new(LearningConfig::default());
    let remote = LearningEngine::new(LearningConfig::default());

    local.federate(&remote.export_snapshot("r1"));
    local.federate(&remote.export_snapshot("r2"));

    assert_eq!(local.aggregator().merge_count, 2);
}

#[test]
fn federated_aggregator_merge_directly() {
    let mut local_qt = QTable::new(0.1, 0.9, 0.0, 1.0, 0.0);
    let s = QTable::state_key("direct aggregator test");
    local_qt.update(s, ActionKey::Pivot, 0.6, s);
    let before = local_qt.q_value(s, ActionKey::Pivot);

    let mut remote_qt = QTable::new(0.1, 0.9, 0.0, 1.0, 0.0);
    remote_qt.update(s, ActionKey::Pivot, 0.9, s);

    let snap = AgentSnapshot {
        agent_id: "direct-remote".into(),
        q_table: remote_qt,
        total_reward: 10.0,
    };

    let mut agg = FederatedAggregator::new(0.5);
    agg.merge(&mut local_qt, &snap);

    let after = local_qt.q_value(s, ActionKey::Pivot);
    assert!(
        after > before,
        "merged value should pull toward higher remote value"
    );
}

#[tokio::test]
async fn two_agents_federate_and_share_knowledge() {
    let mut agent_a = LmmAgent::builder()
        .persona("Agent Alpha")
        .behavior("Understand Rust systems programming.")
        .learning_engine(LearningEngine::new(LearningConfig::default()))
        .build();

    let mut agent_b = LmmAgent::builder()
        .persona("Agent Beta")
        .behavior("Study Rust memory model.")
        .learning_engine(LearningEngine::new(LearningConfig::default()))
        .build();

    agent_a.think("Rust ownership model").await.unwrap();

    let q_count_before = agent_b
        .learning_engine
        .as_ref()
        .unwrap()
        .q_table()
        .state_count();

    let snapshot = agent_a
        .learning_engine
        .as_ref()
        .unwrap()
        .export_snapshot(agent_a.id.clone());

    agent_b
        .learning_engine
        .as_mut()
        .unwrap()
        .federate(&snapshot);

    let q_count_after = agent_b
        .learning_engine
        .as_ref()
        .unwrap()
        .q_table()
        .state_count();

    assert!(
        q_count_after >= q_count_before,
        "agent-b Q-table should grow after federating with agent-a"
    );
    assert_eq!(
        agent_b
            .learning_engine
            .as_ref()
            .unwrap()
            .aggregator()
            .merge_count,
        1
    );
}
