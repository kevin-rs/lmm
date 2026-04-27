// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `LearningEngine` - HELM orchestration layer.
//!
//! `LearningEngine` coordinates all six HELM sub-modules into a unified
//! per-agent learning controller. It is designed to plug into the existing
//! `ThinkLoop` lifecycle with two call-site hooks:
//!
//! 1. **`record_step`** - called once per ThinkLoop iteration with the
//!    `CognitionSignal` from that step.
//! 2. **`end_of_episode`** - called after `ThinkLoop::run` completes to trigger
//!    distillation, meta-prototype storage, elastic guard update, and ε decay.
//!
//! Optional federated exchange is performed explicitly via `federate`.
//!
//! ## Examples
//!
//! ```rust
//! use lmm_agent::cognition::learning::engine::LearningEngine;
//! use lmm_agent::cognition::learning::config::LearningConfig;
//! use lmm_agent::cognition::learning::q_table::{ActionKey, QTable};
//! use lmm_agent::cognition::memory::ColdStore;
//! use lmm_agent::cognition::knowledge::KnowledgeIndex;
//! use lmm_agent::cognition::signal::CognitionSignal;
//!
//! let mut engine = LearningEngine::new(LearningConfig::default());
//! let sig = CognitionSignal::new(0, "rust memory".into(), "rust uses ownership".into(), 1.0, 0.0);
//! let state = QTable::state_key("rust memory");
//! let next_state = QTable::state_key("rust uses ownership");
//! engine.record_step(&sig, state, ActionKey::Narrow, next_state);
//!
//! let cold = ColdStore::default();
//! let mut idx = KnowledgeIndex::new();
//! engine.end_of_episode(&cold, &mut idx, "rust memory safety", 0.8);
//! assert!(!engine.q_table().is_empty());
//! ```

use crate::cognition::knowledge::KnowledgeIndex;
use crate::cognition::learning::config::LearningConfig;
use crate::cognition::learning::distill::KnowledgeDistiller;
use crate::cognition::learning::elastic::ElasticMemoryGuard;
use crate::cognition::learning::federated::FederatedAggregator;
use crate::cognition::learning::informal::InformalLearner;
use crate::cognition::learning::meta::MetaAdapter;
use crate::cognition::learning::q_table::{ActionKey, QTable};
use crate::cognition::memory::ColdStore;
use crate::cognition::signal::CognitionSignal;
use crate::types::{AgentSnapshot, ExperienceRecord, LearningMode};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;

/// The HELM orchestration hub that bundles all six learning sub-modules.
///
/// # Examples
///
/// ```rust
/// use lmm_agent::cognition::learning::engine::LearningEngine;
/// use lmm_agent::cognition::learning::config::LearningConfig;
///
/// let mut engine = LearningEngine::new(LearningConfig::default());
/// assert!(engine.q_table().is_empty());
/// assert_eq!(engine.episode_count(), 0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningEngine {
    config: LearningConfig,
    q_table: QTable,
    meta: MetaAdapter,
    distiller: KnowledgeDistiller,
    aggregator: FederatedAggregator,
    elastic: ElasticMemoryGuard,
    informal: InformalLearner,

    /// Accumulated total reward across all episodes.
    total_reward: f64,

    /// Number of completed episodes.
    episode_count: usize,

    /// Experience buffer for current episode (cleared at `end_of_episode`).
    experience_buffer: Vec<ExperienceRecord>,
}

impl LearningEngine {
    /// Constructs a new `LearningEngine` from a [`LearningConfig`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::learning::engine::LearningEngine;
    /// use lmm_agent::cognition::learning::config::LearningConfig;
    ///
    /// let engine = LearningEngine::new(LearningConfig::default());
    /// assert_eq!(engine.episode_count(), 0);
    /// ```
    pub fn new(config: LearningConfig) -> Self {
        let q_table = QTable::new(
            config.alpha,
            config.gamma,
            config.epsilon,
            config.epsilon_decay,
            config.epsilon_min,
        );
        let meta = MetaAdapter::new(config.meta_top_k);
        let distiller = KnowledgeDistiller::new(config.distill_threshold, config.distill_top_k);
        let aggregator = FederatedAggregator::new(config.federated_blend);
        let elastic = ElasticMemoryGuard::new(config.elastic_pin_count, config.elastic_lambda);
        let informal = InformalLearner::new(config.distill_threshold, config.pmi_min_count, 0.5);

        Self {
            config,
            q_table,
            meta,
            distiller,
            aggregator,
            elastic,
            informal,
            total_reward: 0.0,
            episode_count: 0,
            experience_buffer: Vec::new(),
        }
    }

    /// Returns a reference to the current configuration.
    pub fn config(&self) -> &LearningConfig {
        &self.config
    }

    /// Returns a reference to the Q-table.
    pub fn q_table(&self) -> &QTable {
        &self.q_table
    }

    /// Resets the Q-table exploration epsilon.
    pub fn reset_epsilon(&mut self, epsilon: f64) {
        self.q_table.reset_epsilon(epsilon);
    }

    /// Returns the total number of completed episodes.
    pub fn episode_count(&self) -> usize {
        self.episode_count
    }

    /// Returns the accumulated total reward.
    pub fn total_reward(&self) -> f64 {
        self.total_reward
    }

    /// Records one ThinkLoop step into all active learning sub-modules.
    ///
    /// This should be called inside `ThinkLoop::run` after each step produces
    /// a `CognitionSignal`.
    ///
    /// # Arguments
    ///
    /// * `signal`     - The signal produced by this ThinkLoop iteration.
    /// * `state`      - FNV-1a state key for the current query.
    /// * `action`     - The action applied to produce `signal.query`.
    /// * `next_state` - FNV-1a state key for the observation after the action.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::learning::engine::LearningEngine;
    /// use lmm_agent::cognition::learning::config::LearningConfig;
    /// use lmm_agent::cognition::learning::q_table::{ActionKey, QTable};
    /// use lmm_agent::cognition::signal::CognitionSignal;
    ///
    /// let mut engine = LearningEngine::new(LearningConfig::default());
    /// let sig = CognitionSignal::new(0, "a b".into(), "a b c".into(), 1.0, 0.0);
    /// let s = QTable::state_key("a b");
    /// let s2 = QTable::state_key("a b c");
    /// engine.record_step(&sig, s, ActionKey::Narrow, s2);
    /// assert!(!engine.q_table().is_empty());
    /// ```
    pub fn record_step(
        &mut self,
        signal: &CognitionSignal,
        state: u64,
        action: ActionKey,
        next_state: u64,
    ) {
        if self.config.is_mode_active(LearningMode::QTable) {
            self.q_table
                .update(state, action, signal.reward, next_state);
        }

        if self.config.is_mode_active(LearningMode::Informal) {
            self.informal.observe(&signal.observation, signal.reward);
        }

        if self.config.is_mode_active(LearningMode::Elastic) && !signal.observation.is_empty() {
            self.elastic.observe_activation(&signal.observation);
        }

        self.experience_buffer.push(ExperienceRecord {
            state,
            action,
            reward: signal.reward,
            next_state,
        });
        self.total_reward += signal.reward;
    }

    /// Selects the recommended action for `state` using the Q-table and meta priors.
    ///
    /// If a meta-adapter warm-start is available, it is added as a prior bias to
    /// the greedy Q selection before applying ε-greedy exploration.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::learning::engine::LearningEngine;
    /// use lmm_agent::cognition::learning::config::LearningConfig;
    /// use lmm_agent::cognition::learning::q_table::{ActionKey, QTable};
    ///
    /// let mut engine = LearningEngine::new(LearningConfig::default());
    /// let s = QTable::state_key("hello world");
    /// let action = engine.recommend_action(s, "hello world", 0);
    /// assert!(ActionKey::all().contains(&action));
    /// ```
    pub fn recommend_action(&mut self, state: u64, goal: &str, step: usize) -> ActionKey {
        if self.config.is_mode_active(LearningMode::MetaAdapt) {
            let offsets = self.meta.adapt(goal);
            if let Some((&best_action, _)) = offsets
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
            {
                let q_best = self.q_table.best_action(state);
                if q_best.is_none() {
                    return best_action;
                }
            }
        }
        self.q_table.select_action(state, step)
    }

    /// Finalises a completed episode: distils knowledge, stores meta-prototype,
    /// synthesises PMI facts, and decays ε.
    ///
    /// # Arguments
    ///
    /// * `cold`       - The agent's cold store after `drain_to_cold`.
    /// * `index`      - The agent's knowledge index for distillation output.
    /// * `goal`       - The natural-language goal that was pursued.
    /// * `avg_reward` - Mean reward across all steps in this episode.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::learning::engine::LearningEngine;
    /// use lmm_agent::cognition::learning::config::LearningConfig;
    /// use lmm_agent::cognition::memory::ColdStore;
    /// use lmm_agent::cognition::knowledge::KnowledgeIndex;
    ///
    /// let mut engine = LearningEngine::new(LearningConfig::default());
    /// let cold = ColdStore::default();
    /// let mut idx = KnowledgeIndex::new();
    /// engine.end_of_episode(&cold, &mut idx, "test goal", 0.5);
    /// assert_eq!(engine.episode_count(), 1);
    /// ```
    pub fn end_of_episode(
        &mut self,
        cold: &ColdStore,
        index: &mut KnowledgeIndex,
        goal: &str,
        avg_reward: f64,
    ) {
        if self.config.is_mode_active(LearningMode::Distill) {
            self.distiller.distill(cold, index);
        }

        if self.config.is_mode_active(LearningMode::MetaAdapt) {
            let offsets = self.compute_episode_offsets();
            self.meta.record_episode(goal, offsets, avg_reward);
        }

        if self.config.is_mode_active(LearningMode::Informal) {
            self.informal.synthesise_into(index, 5, 0.3);
        }

        if self.config.is_mode_active(LearningMode::QTable) {
            self.q_table.decay_epsilon();
        }

        self.experience_buffer.clear();
        self.episode_count += 1;
    }

    /// Merges a remote [`AgentSnapshot`] into the local Q-table (federated step).
    ///
    /// Only active when [`LearningMode::Federated`] is enabled.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::learning::engine::LearningEngine;
    /// use lmm_agent::cognition::learning::config::LearningConfig;
    /// use lmm_agent::cognition::learning::q_table::{ActionKey, QTable};
    /// use lmm_agent::types::AgentSnapshot;
    ///
    /// let mut engine = LearningEngine::new(LearningConfig::default());
    /// let remote_qt = QTable::new(0.1, 0.9, 0.0, 1.0, 0.0);
    /// let snap = AgentSnapshot { agent_id: "remote".into(), q_table: remote_qt, total_reward: 1.0 };
    /// engine.federate(&snap);
    /// assert_eq!(engine.aggregator().merge_count, 1);
    /// ```
    pub fn federate(&mut self, snapshot: &AgentSnapshot) {
        if self.config.is_mode_active(LearningMode::Federated) {
            self.aggregator.merge(&mut self.q_table, snapshot);
        }
    }

    /// Exports an [`AgentSnapshot`] for federated sharing with other agents.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::learning::engine::LearningEngine;
    /// use lmm_agent::cognition::learning::config::LearningConfig;
    ///
    /// let engine = LearningEngine::new(LearningConfig::default());
    /// let snap = engine.export_snapshot("agent-1");
    /// assert_eq!(snap.agent_id, "agent-1");
    /// ```
    pub fn export_snapshot(&self, agent_id: impl Into<String>) -> AgentSnapshot {
        AgentSnapshot {
            agent_id: agent_id.into(),
            q_table: self.q_table.clone(),
            total_reward: self.total_reward,
        }
    }

    /// Returns a reference to the federated aggregator.
    pub fn aggregator(&self) -> &FederatedAggregator {
        &self.aggregator
    }

    /// Returns a reference to the elastic memory guard.
    pub fn elastic(&self) -> &ElasticMemoryGuard {
        &self.elastic
    }

    /// Returns a reference to the informal learner.
    pub fn informal(&self) -> &InformalLearner {
        &self.informal
    }

    /// Returns a reference to the meta-adapter.
    pub fn meta(&self) -> &MetaAdapter {
        &self.meta
    }

    /// Returns a reference to the knowledge distiller.
    pub fn distiller(&self) -> &KnowledgeDistiller {
        &self.distiller
    }

    fn compute_episode_offsets(&self) -> HashMap<ActionKey, f64> {
        let mut totals: HashMap<ActionKey, (f64, usize)> = HashMap::new();
        for xp in &self.experience_buffer {
            let e = totals.entry(xp.action).or_insert((0.0, 0));
            e.0 += xp.reward;
            e.1 += 1;
        }
        totals
            .into_iter()
            .map(|(action, (total, count))| (action, total / count as f64))
            .collect()
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
