// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Custom General Purpose Agent Example
//!
//! Demonstrates how to compose a custom agent using the `lmm-agent` framework.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example custom_agent --features agent
//! ```

use lmm_agent::prelude::*;

/// A minimal general-purpose agent that demonstrates the full agent lifecycle.
///
/// - Derives [`Auto`] for zero-boilerplate `Agent`, `Functions`, and
///   `AsyncFunctions` implementations.
/// - Implements only [`Executor`] with custom task logic.
#[derive(Debug, Default, Auto)]
pub struct CustomAgent {
    /// Top-level role label.
    pub persona: Cow<'static, str>,
    /// Mission/Mission statement.
    pub behavior: Cow<'static, str>,
    /// Current lifecycle status.
    pub status: Status,
    /// The core agent data (memory, tools, planner, ...).
    pub agent: LmmAgent,
    /// Hot memory shortcut (mirrors `agent.memory`).
    pub memory: Vec<Message>,
}

#[async_trait]
impl Executor for CustomAgent {
    async fn execute<'a>(
        &'a mut self,
        _tasks: &'a mut Task,
        _execute: bool,
        _browse: bool,
        _max_tries: u64,
    ) -> Result<()> {
        self.agent.update(Status::Active);

        let prompt = self.agent.persona().to_string();
        let response = self.generate(&prompt).await?;

        println!("[CustomAgent] Generated response:\n  {response}\n");

        self.agent.add_message(Message::new("assistant", response.clone()));

        let _ = self
            .save_ltm(Message::new("assistant", response))
            .await;

        self.agent.update(Status::Completed);
        Ok(())
    }
}

#[tokio::main]
async fn main() {
    let persona = "General Purpose Agent";
    let behavior = "The Rust programming language is fast and memory safe.";

    let agent = CustomAgent {
        persona: persona.into(),
        behavior: behavior.into(),
        agent: LmmAgent::new(persona.into(), behavior.into()),
        ..Default::default()
    };

    match AutoAgent::default()
        .with(agents![agent])
        .max_tries(3)
        .build()
        .expect("Failed to build AutoAgent")
        .run()
        .await
    {
        Ok(msg) => println!("[AutoAgent] {msg}"),
        Err(err) => eprintln!("[AutoAgent] Error: {err:?}"),
    }
}
