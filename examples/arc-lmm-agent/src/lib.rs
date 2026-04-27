// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `arc-lmm-agent`
//!
//! An autonomous ARC-AGI navigation solver backed by equation-based intelligence
//! from the `lmm` framework.
//!
//! ## Module layout
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`config`] | CLI and runtime configuration |
//! | [`error`] | Unified error type |
//! | [`frame`] | Game-frame parsing and entity detection |
//! | [`world`] | Learned spatial model (`WorldMap`) |
//! | [`tools`] | Stateless pathfinding algorithms (`PathfindingTool`) |
//! | [`policy`] | Tiered navigation policy (`LmmPolicy`) |
//! | [`runner`] | Game-loop orchestrator (`ArcGameRunner`) |

pub mod config;
pub mod display;
pub mod error;
pub mod frame;
pub mod policy;
pub mod runner;
pub mod tools;
pub mod world;
