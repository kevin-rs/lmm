// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `tools` - Agent-callable algorithmic tools.
//!
//! This module provides pure-function tools that the agent policy
//! dispatches to at runtime. Every tool operates on data passed by
//! reference; none of them hold mutable state.
//!
//! ## Available tools
//!
//! - [`PathfindingTool`] - graph-based BFS and Cartesian A\* pathfinding over
//!   the learned world map.
//!
//! ## Design principles
//!
//! Tools are **stateless** structs with associated methods. The agent stores
//! an instance of each tool it needs and calls it with slices of its own
//! world data. Keeping algorithms here makes `policy.rs` a pure behavioral
//! orchestrator with no search logic embedded inside it.

pub mod pathfinding;

pub use pathfinding::PathfindingTool;
