// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm_agent::agent::LmmAgent;
use lmm_agent::prelude::*;
use lmm_agent::types::Goal;

#[test]
fn builder_sets_persona_and_behavior() {
    let agent = LmmAgent::builder()
        .persona("Scientist")
        .behavior("Do science.")
        .build();

    assert_eq!(agent.persona.as_str(), "Scientist");
    assert_eq!(agent.behavior.as_str(), "Do science.");
}

#[test]
fn builder_default_status_is_idle() {
    let agent = LmmAgent::builder()
        .persona("Tester")
        .behavior("Test.")
        .build();

    assert_eq!(agent.status, Status::Idle);
}

#[test]
fn builder_with_custom_memory() {
    let comms = vec![Message::new("user", "Hi")];
    let agent = LmmAgent::builder()
        .persona("Tester")
        .behavior("Test.")
        .memory(comms.clone())
        .build();

    assert_eq!(&agent.memory, &comms);
}

#[test]
fn builder_with_custom_status() {
    let agent = LmmAgent::builder()
        .persona("Tester")
        .behavior("Test.")
        .status(Status::Active)
        .build();

    assert_eq!(agent.status, Status::Active);
}

#[test]
fn new_shorthand_works() {
    let agent = LmmAgent::new("Explorer".into(), "Explore Rust.".into());
    assert_eq!(agent.persona.as_str(), "Explorer");
    assert_eq!(agent.behavior.as_str(), "Explore Rust.");
}

#[test]
fn add_message_appends_to_memory() {
    let mut agent = LmmAgent::new("Tester".into(), "Test.".into());
    agent.add_message(Message::new("user", "Hello"));
    assert_eq!(agent.memory.len(), 1);
    assert_eq!(agent.memory[0].role.as_ref(), "user");
}

#[test]
fn add_ltm_message_appends_to_ltm() {
    let mut agent = LmmAgent::new("Tester".into(), "Test.".into());
    agent.add_ltm_message(Message::new("assistant", "World"));
    assert_eq!(agent.long_term_memory.len(), 1);
}

#[test]
fn complete_goal_marks_goal_done() {
    let mut agent = LmmAgent::builder()
        .persona("Tester")
        .behavior("Test.")
        .planner(Planner {
            current_plan: vec![Goal {
                description: "Research Rust".into(),
                priority: 1,
                completed: false,
            }],
        })
        .build();

    assert!(agent.complete_goal("Research"));
    assert!(agent.planner.as_ref().unwrap().current_plan[0].completed);
}

#[test]
fn agent_id_is_unique() {
    let a1 = LmmAgent::new("p".into(), "t".into());
    let a2 = LmmAgent::new("p".into(), "t".into());
    assert_ne!(a1.id, a2.id);
}

#[test]
fn builder_defaults_planner_to_some() {
    let agent = LmmAgent::builder().persona("p").behavior("t.").build();
    assert!(agent.planner.is_some());
}

#[test]
fn builder_can_disable_planner() {
    let agent = LmmAgent::builder()
        .persona("p")
        .behavior("t.")
        .planner(None)
        .build();
    assert!(agent.planner.is_none());
}

#[tokio::test]
async fn predict_generate_extends_seed() {
    let mut agent = LmmAgent::new(
        "Tester".into(),
        "the quick brown fox jumps over the lazy dog the quick brown".into(),
    );
    let result = agent
        .generate("the quick brown fox jumps over the lazy dog the quick brown")
        .await;
    assert!(result.is_ok());
    assert!(result.unwrap().contains("the quick"));
}

#[test]
fn agent_trait_persona_returns_str() {
    let agent = LmmAgent::new("Actor".into(), "Do things.".into());
    assert_eq!(Agent::persona(&agent), "Actor");
}

#[test]
fn agent_trait_update_status() {
    let mut agent = LmmAgent::new("p".into(), "t.".into());
    agent.update(Status::Active);
    assert_eq!(agent.status, Status::Active);
}
