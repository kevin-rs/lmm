// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `learning` - HELM (Hybrid Equation-based Lifelong Memory).
//!
//! This module provides a CPU-only, equation-driven learning engine that layers
//! six complementary learning paradigms on top of the `ThinkLoop` controller:
//!
//! | Sub-module    | Paradigm                          | Key type                   |
//! |---------------|-----------------------------------|----------------------------|
//! | `q_table`     | Tabular Q-learning (Bellman TD)   | [`QTable`]                 |
//! | `meta`        | Prototype meta-adaptation         | [`MetaAdapter`]            |
//! | `distill`     | Knowledge distillation            | [`KnowledgeDistiller`]     |
//! | `federated`   | Self-federated aggregation        | [`FederatedAggregator`]    |
//! | `elastic`     | Elastic memory guard              | [`ElasticMemoryGuard`]     |
//! | `informal`    | Invisible PMI co-occurrence mining| [`InformalLearner`]        |
//! | `engine`      | Orchestration hub                 | [`LearningEngine`]         |
//! | `config`      | Hyperparameter configuration      | [`LearningConfig`]         |
//! | `store`       | Disk persistence                  | [`LearningStore`]          |
//!
//! ## Quick start
//!
//! ```rust
//! use lmm_agent::cognition::learning::engine::LearningEngine;
//! use lmm_agent::cognition::learning::config::LearningConfig;
//! use lmm_agent::cognition::learning::q_table::{ActionKey, QTable};
//! use lmm_agent::cognition::signal::CognitionSignal;
//! use lmm_agent::cognition::memory::ColdStore;
//! use lmm_agent::cognition::knowledge::KnowledgeIndex;
//!
//! let mut engine = LearningEngine::new(LearningConfig::default());
//!
//! let sig = CognitionSignal::new(0, "rust memory".into(), "ownership prevents races".into(), 1.0, 0.0);
//! let s0 = QTable::state_key("rust memory");
//! let s1 = QTable::state_key("ownership prevents races");
//! engine.record_step(&sig, s0, ActionKey::Narrow, s1);
//!
//! let cold = ColdStore::default();
//! let mut idx = KnowledgeIndex::new();
//! engine.end_of_episode(&cold, &mut idx, "rust memory safety", 0.8);
//!
//! assert!(!engine.q_table().is_empty());
//! assert_eq!(engine.episode_count(), 1);
//! ```
//!
//! ## See Also
//!
//! * [Q-learning - Wikipedia](https://en.wikipedia.org/wiki/Q-learning)
//! * [Meta-learning - Wikipedia](https://en.wikipedia.org/wiki/Meta-learning_(computer_science))
//! * [Knowledge distillation - Wikipedia](https://en.wikipedia.org/wiki/Knowledge_distillation)
//! * [Federated learning - Wikipedia](https://en.wikipedia.org/wiki/Federated_learning)
//! * [Lifelong learning - Wikipedia](https://en.wikipedia.org/wiki/Lifelong_learning)

pub mod config;
pub mod distill;
pub mod elastic;
pub mod engine;
pub mod federated;
pub mod informal;
pub mod meta;
pub mod q_table;
pub mod store;

pub use config::{LearningConfig, LearningConfigBuilder};
pub use distill::KnowledgeDistiller;
pub use elastic::ElasticMemoryGuard;
pub use engine::LearningEngine;
pub use federated::FederatedAggregator;
pub use informal::InformalLearner;
pub use meta::{MetaAdapter, TaskPrototype};
pub use q_table::{ActionKey, QTable};
pub use store::LearningStore;

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
