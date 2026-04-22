// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `FederatedAggregator`  self-federated knowledge merging.
//!
//! Enables agents to share learned Q-values without a central parameter server.
//! Any agent can act as coordinator: it exports an [`AgentSnapshot`] and passes
//! remote snapshots received from peers to `merge`.
//!
//! ## Aggregation rule
//!
//! For every `(state, action)` pair shared between local and remote tables:
//!
//! ```text
//! Q_merged(s, a) = w_local · Q_local(s, a) + (1 - w_local) · Q_remote(s, a)
//! ```
//!
//! where `w_local = federated_blend` from [`LearningConfig`]. Pairs present
//! only in the remote table are inserted directly (no penalty for new knowledge).
//!
//! For knowledge distillation entries, the union of both agents' ingested sets
//! is taken so neither agent loses compressed facts.
//!
//! This is structurally equivalent to the **FedWKD** weighted aggregation
//! described in the knowledge corpus, adapted for tabular Q-values rather than
//! neural weights.
//!
//! ## Examples
//!
//! ```rust
//! use lmm_agent::cognition::learning::federated::FederatedAggregator;
//! use lmm_agent::cognition::learning::q_table::{ActionKey, QTable};
//! use lmm_agent::types::AgentSnapshot;
//!
//! let mut qt_a = QTable::new(0.1, 0.9, 0.0, 1.0, 0.0);
//! let s = QTable::state_key("shared topic");
//! qt_a.update(s, ActionKey::Narrow, 0.8, s);
//!
//! let snapshot = AgentSnapshot {
//!     agent_id: "agent-b".into(),
//!     q_table: qt_a.clone(),
//!     total_reward: 1.0,
//! };
//!
//! let mut qt_local = QTable::new(0.1, 0.9, 0.0, 1.0, 0.0);
//! let mut agg = FederatedAggregator::new(0.5);
//! agg.merge(&mut qt_local, &snapshot);
//! assert!(qt_local.q_value(s, ActionKey::Narrow) > 0.0);
//! ```

use crate::cognition::learning::q_table::QTable;
use crate::types::AgentSnapshot;
use serde::{Deserialize, Serialize};

/// Self-federated Q-table aggregator.
///
/// # Examples
///
/// ```rust
/// use lmm_agent::cognition::learning::federated::FederatedAggregator;
/// use lmm_agent::cognition::learning::q_table::{ActionKey, QTable};
/// use lmm_agent::types::AgentSnapshot;
///
/// let mut local_qt = QTable::new(0.1, 0.9, 0.0, 1.0, 0.0);
/// let topic = "topic a rust systems";
/// let s = QTable::state_key(topic);
/// local_qt.update(s, ActionKey::Expand, 0.4, s);
///
/// let mut remote_qt = QTable::new(0.1, 0.9, 0.0, 1.0, 0.0);
/// remote_qt.update(s, ActionKey::Expand, 0.8, s);
///
/// let snap = AgentSnapshot { agent_id: "b".into(), q_table: remote_qt, total_reward: 2.0 };
/// let mut agg = FederatedAggregator::new(0.5);
/// agg.merge(&mut local_qt, &snap);
/// let merged = local_qt.q_value(s, ActionKey::Expand);
/// assert!(merged > 0.0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedAggregator {
    /// Blend weight for the local table ∈ [0, 1].
    ///
    /// `1.0` = keep local entirely, `0.0` = adopt remote entirely.
    pub local_weight: f64,

    /// Number of successful federated merge operations performed.
    pub merge_count: usize,
}

impl FederatedAggregator {
    /// Constructs a new `FederatedAggregator`.
    ///
    /// * `local_weight` - Weight for the local agent's Q-values during merge ∈ [0, 1].
    pub fn new(local_weight: f64) -> Self {
        Self {
            local_weight: local_weight.clamp(0.0, 1.0),
            merge_count: 0,
        }
    }

    /// Merges a remote [`AgentSnapshot`]'s Q-table into `local_qt`.
    ///
    /// Remote snapshots with higher `total_reward` are given proportionally
    /// more influence by scaling `local_weight` down by the reward ratio.
    ///
    /// # Arguments
    ///
    /// * `local_qt` - The local agent's Q-table, modified in place.
    /// * `snapshot` - The exported Q-table and metadata from a remote agent.
    pub fn merge(&mut self, local_qt: &mut QTable, snapshot: &AgentSnapshot) {
        let local_total = local_qt.update_count as f64 + 1.0;
        let remote_total = snapshot.total_reward + 1.0;
        let effective_local = (self.local_weight * local_total / (local_total + remote_total))
            .clamp(0.0, self.local_weight);

        local_qt.merge(&snapshot.q_table, effective_local);
        self.merge_count += 1;
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
