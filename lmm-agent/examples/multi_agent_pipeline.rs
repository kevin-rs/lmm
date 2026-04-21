// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Multi-Agent Pipeline Example
//!
//! Three specialised agents run **concurrently** inside a single [`AutoAgent`]
//! orchestrator, each exercising a different aspect of the lifecycle:
//!
//! | Agent             | Lifecycle path                              |
//! |-------------------|---------------------------------------------|
//! | `ResearcherAgent` | `Idle -> Thinking -> Completed` (ThinkLoop) |
//! | `PlannerAgent`    | `Idle -> Active -> InUnitTesting -> Completed` |
//! | `SummaryAgent`    | `Idle -> Active -> Completed` (generate+LTM)|
//!
//! The orchestrator waits for all three to finish before returning.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example multi_agent_pipeline
//! ```

use lmm_agent::prelude::*;

#[derive(Debug, Default, Auto)]
pub struct ResearcherAgent {
    pub agent: LmmAgent,
}

#[async_trait]
impl Executor for ResearcherAgent {
    async fn execute<'a>(
        &'a mut self,
        _task: &'a mut Task,
        _execute: bool,
        _browse: bool,
        _max_tries: u64,
    ) -> Result<()> {
        let goal = self.agent.behavior.clone();
        println!("[Researcher] Starting research: \"{goal}\"");
        self.agent.update(Status::Thinking);

        let result = self.agent.think_with(&goal, 5, 0.3, 1.0, 0.05).await?;

        println!(
            "[Researcher] converged={} steps={} error={:.4}",
            result.converged, result.steps, result.final_error
        );

        for (i, snippet) in result.memory_snapshot.iter().enumerate().take(3) {
            if !snippet.is_empty() {
                self.agent
                    .knowledge
                    .insert(format!("research_{i}"), snippet.clone());
            }
        }

        let summary = format!(
            "Research complete. {} steps, error={:.3}, {} facts found.",
            result.steps,
            result.final_error,
            self.agent.knowledge.facts.len()
        );
        self.agent
            .add_message(Message::new("assistant", summary.clone()));
        let _ = self
            .save_ltm(Message::new("assistant", summary.clone()))
            .await;

        self.agent.update(Status::Completed);
        println!("[Researcher] {summary}");
        Ok(())
    }
}

#[derive(Debug, Default, Auto)]
pub struct PlannerAgent {
    pub agent: LmmAgent,
}

#[async_trait]
impl Executor for PlannerAgent {
    async fn execute<'a>(
        &'a mut self,
        _task: &'a mut Task,
        _execute: bool,
        _browse: bool,
        _max_tries: u64,
    ) -> Result<()> {
        println!("[Planner] Executing goal plan...");
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
            let text = self
                .generate(&format!("{goal} involves the following steps:"))
                .await
                .unwrap_or_else(|_| "step analysis pending".to_string());

            self.agent
                .add_message(Message::new("assistant", text.clone()));
            self.agent.complete_goal(goal);
            println!("[Planner]   Goal done: \"{goal}\"");
        }

        self.agent.update(Status::InUnitTesting);
        let all_done = self
            .agent
            .planner
            .as_ref()
            .map(|p| p.is_done())
            .unwrap_or(false);

        if !all_done {
            return Err(anyhow!("[Planner] Some goals remain incomplete!"));
        }

        if let Some(reflection) = &self.agent.reflection {
            let report = (reflection.evaluation_fn)(&self.agent);
            println!("[Planner] Reflection:\n{report}");
        }

        self.agent.update(Status::Completed);
        println!("[Planner] All {} goals completed.", goals.len());
        Ok(())
    }
}

#[derive(Debug, Default, Auto)]
pub struct SummaryAgent {
    pub agent: LmmAgent,
}

#[async_trait]
impl Executor for SummaryAgent {
    async fn execute<'a>(
        &'a mut self,
        _task: &'a mut Task,
        _execute: bool,
        _browse: bool,
        _max_tries: u64,
    ) -> Result<()> {
        println!("[Summary] Generating synthesis...");
        self.agent.update(Status::Active);

        let topic = self.agent.behavior.clone();

        let seeds = [
            format!("The core principles of {topic} are"),
            format!("The practical implications of {topic} reveal"),
            format!("In conclusion, {topic} demonstrates"),
        ];

        let mut full_summary = String::new();
        for seed in &seeds {
            match self.generate(seed).await {
                Ok(text) => {
                    println!("[Summary]   {text}");
                    full_summary.push_str(&text);
                    full_summary.push(' ');
                    self.agent
                        .add_message(Message::new("assistant", text.clone()));
                }
                Err(e) => eprintln!("[Summary]   generate error: {e}"),
            }
        }

        let synthesis = Message::new("assistant", full_summary.trim().to_string());
        let _ = self.save_ltm(synthesis.clone()).await;
        self.agent.add_ltm_message(synthesis);

        let ltm = self.get_ltm().await.unwrap_or_default();
        println!(
            "[Summary] Long-term memory now holds {} entries.",
            ltm.len()
        );

        self.agent.update(Status::Completed);
        println!("[Summary] Synthesis complete.");
        Ok(())
    }
}

#[tokio::main]
async fn main() {
    let topic = "Rust async programming and the Tokio runtime";

    let researcher = ResearcherAgent {
        agent: LmmAgent::new("Researcher".into(), topic.into()),
        ..Default::default()
    };

    let planner = PlannerAgent {
        agent: LmmAgent::builder()
            .persona("Planner")
            .behavior(topic)
            .capabilities(
                [Capability::CodeGen, Capability::WebSearch]
                    .into_iter()
                    .collect(),
            )
            .planner(Planner {
                current_plan: vec![
                    Goal {
                        description: "Understand async/await fundamentals".into(),
                        priority: 1,
                        completed: false,
                    },
                    Goal {
                        description: "Study Tokio's task scheduling model".into(),
                        priority: 2,
                        completed: false,
                    },
                    Goal {
                        description: "Identify common async pitfalls".into(),
                        priority: 3,
                        completed: false,
                    },
                ],
            })
            .build(),
        ..Default::default()
    };

    let summariser = SummaryAgent {
        agent: LmmAgent::new("Summary Agent".into(), topic.into()),
        ..Default::default()
    };

    match AutoAgent::default()
        .with(agents![researcher, planner, summariser])
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
