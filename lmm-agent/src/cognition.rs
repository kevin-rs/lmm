// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `cognition` - closed-loop agent cognition.
//!
//! This module implements the **ThinkLoop**: a PID-style feedback control system
//! that lets agents iteratively reason toward a natural-language goal using live
//! DuckDuckGo search as the information plant, without any LLM or GPU.
//!
//! ## Sub-modules
//!
//! | Module    | Key Type                    | Role                                            |
//! |-----------|-----------------------------|-------------------------------------------------|
//! | `signal`  | [`signal::CognitionSignal`] | Per-iteration error, gain, and reward scalar    |
//! | `memory`  | [`memory::HotStore`] etc.   | Two-tier bounded/unbounded memory               |
//! | `goal`    | [`goal::GoalEvaluator`]     | Convergence comparator (Jaccard distance)       |
//! | `reflect` | [`reflect::Reflector`]      | Query formulation + memory consolidation        |
//! | `search`  | [`search::SearchOracle`]    | DuckDuckGo plant with in-process cache          |
//! | `loop`    | [`r#loop::ThinkLoop`]       | The controller that stitches everything together|
//!
//! ## Quick example
//!
//! ```rust
//! use lmm_agent::cognition::r#loop::ThinkLoop;
//! use lmm_agent::cognition::search::SearchOracle;
//!
//! #[tokio::main]
//! async fn main() {
//!    let mut oracle = SearchOracle::new(5);
//!    let mut lp = ThinkLoop::builder("How does Rust handle memory?")
//!        .max_iterations(10)
//!        .convergence_threshold(0.25)
//!        .build();
//!   
//!    let result = lp.run(&mut oracle).await;
//!    println!(
//!        "converged={} in {} steps, error={:.3}",
//!        result.converged, result.steps, result.final_error
//!    );
//! }
//! ```
//!
//! ## See Also
//!
//! * [Autonomous agent - Wikipedia](https://en.wikipedia.org/wiki/Autonomous_agent)
//! * [Control theory - Wikipedia](https://en.wikipedia.org/wiki/Control_theory)

pub mod goal;
pub mod r#loop;
pub mod memory;
pub mod reflect;
pub mod search;
pub mod signal;

pub use goal::GoalEvaluator;
pub use r#loop::{ThinkLoop, ThinkLoopBuilder};
pub use memory::{ColdStore, HotStore, MemoryEntry};
pub use reflect::Reflector;
pub use search::SearchOracle;
pub use signal::{CognitionSignal, error_from_texts};

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
