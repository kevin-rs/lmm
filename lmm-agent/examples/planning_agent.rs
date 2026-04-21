// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Planning Agent Example
//!
//! Demonstrates a **goal-driven planning lifecycle**:
//!
//! ```text
//! Idle -> Active -> InUnitTesting -> Completed
//! ```
//!
//! The agent:
//! 1. Loads a multi-goal [`Planner`].
//! 2. Iterates through each goal, generates a response with `generate()`, and
//!    marks the goal `completed`.
//! 3. Uses a custom `ContextManager` to track recent messages and focus topics.
//! 4. Runs self-validation (`InUnitTesting`) to assert that every goal is done.
//! 5. Calls the built-in `Reflection` module to produce a progress report.
//! 6. Finishes `Completed`.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example planning_agent
//! ```

use lmm_agent::prelude::*;

#[derive(Debug, Default, Auto)]
pub struct PlanningAgent {
    pub agent: LmmAgent,
}

#[async_trait]
impl Executor for PlanningAgent {
    async fn execute<'a>(
        &'a mut self,
        _task: &'a mut Task,
        _execute: bool,
        _browse: bool,
        _max_tries: u64,
    ) -> Result<()> {
        println!("Persona   : {}", self.agent.persona);
        println!("Goals     : {}", plan_len(&self.agent));
        println!("Status    : {} -> Active", self.agent.status);

        self.agent.update(Status::Active);

        let goals: Vec<String> = self
            .agent
            .planner
            .as_ref()
            .map(|p| {
                p.current_plan
                    .iter()
                    .map(|g| g.description.clone())
                    .collect()
            })
            .unwrap_or_default();

        for goal in &goals {
            println!("Working on goal: \"{goal}\"");

            let seed = format!("{goal} requires careful analysis of");
            match self.generate(&seed).await {
                Ok(text) => {
                    println!("   Generated: \"{text}\"");
                    let msg = Message::new("assistant", text.clone());
                    self.agent.add_message(msg.clone());
                    let _ = self.save_ltm(msg).await;

                    self.agent.context.focus_topics.push(goal.clone().into());
                    self.agent
                        .context
                        .recent_messages
                        .push(Message::new("assistant", text));
                }
                Err(e) => eprintln!("   [generate] {e}"),
            }

            let completed = self.agent.complete_goal(goal);
            println!(
                "   {} Goal marked completed: {completed}",
                if completed { "[OK]" } else { "[WARN]" }
            );
        }

        self.agent.update(Status::InUnitTesting);
        println!("\nStatus: InUnitTesting - verifying all goals are complete...");

        let all_done = self
            .agent
            .planner
            .as_ref()
            .map(|p| p.is_done())
            .unwrap_or(false);

        if all_done {
            println!("   All goals completed successfully.");
        } else {
            let remaining = self
                .agent
                .planner
                .as_ref()
                .map(|p| p.current_plan.iter().filter(|g| !g.completed).count())
                .unwrap_or(0);
            eprintln!("   {remaining} goal(s) still pending.");
        }

        println!("\nReflection report:");
        if let Some(reflection) = &self.agent.reflection {
            let report = (reflection.evaluation_fn)(&self.agent);
            println!("{report}");
        }

        self.agent.update(Status::Completed);
        println!("Status: {}", self.agent.status);
        println!("   Hot memory     : {} messages", self.agent.memory.len());
        println!(
            "   Long-term      : {} messages",
            self.agent.long_term_memory.len()
        );
        println!("   Focus topics   : {:?}", self.agent.context.focus_topics);
        println!(
            "   Plan progress  : {}/{}",
            plan_completed(&self.agent),
            plan_len(&self.agent)
        );

        Ok(())
    }
}

fn plan_len(agent: &LmmAgent) -> usize {
    agent
        .planner
        .as_ref()
        .map(|p| p.current_plan.len())
        .unwrap_or(0)
}

fn plan_completed(agent: &LmmAgent) -> usize {
    agent
        .planner
        .as_ref()
        .map(|p| p.completed_count())
        .unwrap_or(0)
}

#[tokio::main]
async fn main() {
    let agent = PlanningAgent {
        agent: LmmAgent::builder()
            .persona("Planning Agent")
            .behavior("Design and execute a software architecture plan.")
            .planner(Planner {
                current_plan: vec![
                    Goal {
                        description: "Define the system requirements".into(),
                        priority: 1,
                        completed: false,
                    },
                    Goal {
                        description: "Design the data model and schema".into(),
                        priority: 2,
                        completed: false,
                    },
                    Goal {
                        description: "Specify the API layer and routing".into(),
                        priority: 3,
                        completed: false,
                    },
                    Goal {
                        description: "Write the deployment and CI/CD plan".into(),
                        priority: 4,
                        completed: false,
                    },
                ],
            })
            .context(ContextManager {
                recent_messages: vec![],
                focus_topics: vec![],
            })
            .capabilities(
                [Capability::CodeGen, Capability::ApiIntegration]
                    .into_iter()
                    .collect(),
            )
            .build(),
        ..Default::default()
    };

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
