// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `AgentFunctions` - composite super-trait.
//!
//! `AgentFunctions` is a blanket-implemented composite trait that is the
//! concrete bound accepted by the [`crate::runtime::AutoAgent`] orchestrator.
//!
//! Any struct that implements `Agent + Functions + AsyncFunctions + Debug +
//! Send + Sync` automatically satisfies `AgentFunctions`.
//!
//! ## Attribution
//!
//! Adapted from the `autogpt` project's `traits/composite.rs`:
//! <https://github.com/wiseaidotdev/autogpt/blob/main/autogpt/src/traits/composite.rs>

use crate::traits::agent::Agent;
use crate::traits::functions::{AsyncFunctions, Functions};
use std::fmt::Debug;

/// Composite trait that the [`crate::runtime::AutoAgent`] orchestrator expects.
///
/// Implemented automatically for any type satisfying the sub-trait bounds.
pub trait AgentFunctions: Agent + Functions + AsyncFunctions + Debug + Send + Sync {}

impl<T> AgentFunctions for T where T: Agent + Functions + AsyncFunctions + Debug + Send + Sync {}
