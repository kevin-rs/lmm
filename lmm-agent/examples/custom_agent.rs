// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Custom General Purpose Agent Example
//!
//! Demonstrates how to compose a custom agent using the `lmm-agent` framework
//! with the minimal struct form - only `agent: LmmAgent` is required.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example custom_agent
//! ```

use lmm_agent::prelude::*;

/// A minimal general-purpose agent that demonstrates the full agent lifecycle.
///
/// The `#[derive(Auto)]` macro generates `Agent`, `Functions`, and
/// `AsyncFunctions`. Only `agent: LmmAgent` is required in the struct.
#[derive(Debug, Default, Auto)]
pub struct CustomAgent {
    /// The core agent data (memory, tools, planner, ...).
    pub agent: LmmAgent,
}

#[async_trait]
impl Executor for CustomAgent {
    async fn execute<'a>(
        &'a mut self,
        _task: &'a mut Task,
        _execute: bool,
        _browse: bool,
        _max_tries: u64,
    ) -> Result<()> {
        self.agent.update(Status::Active);

        let prompt = self.agent.behavior.clone();
        let response = self.generate(&prompt).await?;

        println!("[CustomAgent] Persona  : {}", self.agent.persona);
        println!("[CustomAgent] Response :\n  {response}\n");

        self.agent
            .add_message(Message::new("assistant", response.clone()));
        let _ = self.save_ltm(Message::new("assistant", response)).await;

        self.agent.update(Status::Completed);
        Ok(())
    }
}

#[tokio::main]
async fn main() {
    let agent = CustomAgent::new(
        "General Purpose Agent".into(),
        "The Rust programming language is fast and memory safe.".into(),
    );

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
