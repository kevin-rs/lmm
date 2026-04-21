// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `lmm-agent` - equation-based autonomous agent framework.
//!
//! This crate lets you compose, build, and run autonomous agents that reason
//! through the `lmm` equation engine and optionally enrich their knowledge via
//! DuckDuckGo (no LLM provider required).
//!
//! ## Quick Start
//!
//! ```rust
//! use lmm_agent::prelude::*;
//!
//! #[derive(Debug, Default, Auto)]
//! pub struct MyAgent {
//!     persona: Cow<'static, str>,
//!     behavior:  Cow<'static, str>,
//!     status:    Status,
//!     agent:     LmmAgent,
//!     memory:    Vec<Message>,
//! }
//!
//! #[async_trait]
//! impl Executor for MyAgent {
//!     async fn execute<'a>(
//!         &'a mut self, tasks: &'a mut Task,
//!         execute: bool, browse: bool, max_tries: u64,
//!     ) -> Result<()> {
//!         let prompt = self.agent.persona().to_string();
//!         let response = self.generate(&prompt).await?;
//!         self.agent.add_message(Message {
//!             role:    "assistant".into(),
//!             content: response.into(),
//!         });
//!         Ok(())
//!     }
//! }
//!
//! #[tokio::main]
//! async fn main() {
//!     let agent = MyAgent {
//!         persona: "Research Agent".into(),
//!         behavior: "Research Rust async patterns.".into(),
//!         agent: LmmAgent::new("Research Agent".into(), "Research Rust async patterns.".into()),
//!         ..Default::default()
//!     };
//!
//!     AutoAgent::default()
//!         .with(agents![agent])
//!         .build()
//!         .unwrap()
//!         .run()
//!         .await
//!         .unwrap();
//! }
//! ```

pub mod agent;
pub mod cognition;
pub mod error;
pub mod prelude;
pub mod runtime;
pub mod traits;
pub mod types;
