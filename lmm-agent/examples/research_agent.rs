// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Research Agent Example
//!
//! Demonstrates the full **closed-loop research lifecycle** using the `lmm-agent`
//! framework:
//!
//! ```text
//! Idle → [think() called] → Thinking → [converged / stalled] → Completed
//! ```
//!
//! The agent:
//! 1. Starts `Idle`.
//! 2. Sets its status to `Thinking` and runs the `ThinkLoop` against the goal.
//! 3. Accumulates snippets from DuckDuckGo (offline: empty observations) into
//!    tiered memory (`HotStore` → `ColdStore`).
//! 4. Stores every discovered fact in the agent's own `Knowledge` base.
//! 5. Logs a reflection on goal progress using `default_eval_fn`.
//! 6. Finishes `Completed` and prints a full lifecycle summary.
//!
//! Run with:
//!
//! ```bash
//! # offline - symbolic generation
//! cargo run --example research_agent
//!
//! # online - uses DuckDuckGo for factual context
//! cargo run --example research_agent --features net
//! ```

use lmm_agent::prelude::*;

#[derive(Debug, Default, Auto)]
pub struct ResearchAgent {
    pub agent: LmmAgent,
}

#[async_trait]
impl Executor for ResearchAgent {
    async fn execute<'a>(
        &'a mut self,
        _task: &'a mut Task,
        _execute: bool,
        _browse: bool,
        _max_tries: u64,
    ) -> Result<()> {
        let goal = self.agent.behavior.clone();

        println!("Persona   : {}", self.agent.persona);
        println!("Goal      : {goal}");
        println!("Status    : {} -> Thinking", self.agent.status);

        self.agent.update(Status::Thinking);

        let think_result = self
            .agent
            .think_with(
                &goal, 8,    // max_iterations
                0.25, // Jaccard convergence threshold
                1.0,  // k_proportional
                0.05, // k_integral
            )
            .await?;

        println!("\nThinkLoop result:");
        println!(
            "   converged  = {}  |  steps = {}  |  final_error = {:.4}",
            think_result.converged, think_result.steps, think_result.final_error
        );

        for (i, snippet) in think_result.memory_snapshot.iter().enumerate().take(5) {
            if !snippet.is_empty() {
                let key = format!("fact_{i}");
                self.agent.knowledge.insert(key.clone(), snippet.clone());
                self.agent
                    .add_message(Message::new("system", format!("[{key}] {snippet}")));
                println!("   stored -> {key}: {}", &snippet[..snippet.len().min(80)]);
            }
        }

        println!("\nSymbolic continuation from goal context:");
        let seed = format!("{goal} reveals its core principles through");
        match self.generate(&seed).await {
            Ok(cont) => {
                println!("   \"{cont}\"");
                self.agent
                    .add_message(Message::new("assistant", cont.clone()));
                let _ = self.save_ltm(Message::new("assistant", cont)).await;
            }
            Err(e) => eprintln!("   [generate error] {e}"),
        }

        if let Some(reflection) = &self.agent.reflection {
            let eval = (reflection.evaluation_fn)(&self.agent);
            println!("\nReflection:\n{eval}");
        }

        self.agent.update(Status::Completed);
        println!("\nStatus: {}", self.agent.status);
        println!("   Hot memory  : {} entries", self.agent.memory.len());
        println!(
            "   Long-term   : {} entries",
            self.agent.long_term_memory.len()
        );
        println!(
            "   Knowledge   : {} facts",
            self.agent.knowledge.facts.len()
        );
        println!("   Signals     : {} recorded", think_result.signals.len());

        Ok(())
    }
}

#[tokio::main]
async fn main() {
    let topic = "Rust memory safety and ownership model";

    let agent = ResearchAgent::new("Research Agent".into(), topic.into());

    let mut agent = agent;
    agent.agent.planner = Some(Planner {
        current_plan: vec![Goal {
            description: format!("Research: {topic}"),
            priority: 1,
            completed: false,
        }],
    });

    match AutoAgent::default()
        .with(agents![agent])
        .max_tries(1)
        .build()
        .expect("Failed to build AutoAgent")
        .run()
        .await
    {
        Ok(msg) => println!("\n[AutoAgent] {msg}"),
        Err(err) => eprintln!("\n[AutoAgent] Error: {err:?}"),
    }
}
