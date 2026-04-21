// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate lmm_agent;
use lmm_agent::agent::LmmAgent;
use lmm_agent::prelude::*;
use lmm_agent::runtime::AutoAgent;

#[test]
fn builder_produces_correct_persona_and_behavior() {
    let agent = LmmAgent::builder()
        .persona("Research Rust async patterns.")
        .behavior("Research Agent")
        .build();

    assert_eq!(agent.persona(), "Research Rust async patterns.");
    assert_eq!(agent.behavior(), "Research Agent");
}

#[test]
fn builder_default_status_is_idle() {
    let agent = LmmAgent::builder()
        .persona("Test.")
        .behavior("Tester")
        .build();

    assert_eq!(*agent.status(), Status::Idle);
}

#[test]
fn builder_custom_status() {
    let agent = LmmAgent::builder()
        .persona("Test.")
        .behavior("Tester")
        .status(Status::Active)
        .build();

    assert_eq!(*agent.status(), Status::Active);
}

#[test]
fn builder_custom_memory() {
    let msgs = vec![
        Message::new("user", "Hello"),
        Message::new("assistant", "Hi there!"),
    ];
    let agent = LmmAgent::builder()
        .persona("Test.")
        .behavior("Tester")
        .memory(msgs.clone())
        .build();

    assert_eq!(agent.memory().len(), 2);
    assert_eq!(agent.memory()[0].role.as_ref(), "user");
}

#[test]
fn builder_id_is_unique_per_instance() {
    let a = LmmAgent::builder().persona("t").behavior("p").build();
    let b = LmmAgent::builder().persona("t").behavior("p").build();
    assert_ne!(a.id, b.id);
}

#[test]
fn new_shorthand_matches_builder_semantics() {
    let agent = LmmAgent::new("Scientist".into(), "Do science.".into());
    assert_eq!(agent.persona(), "Scientist");
    assert_eq!(agent.behavior(), "Do science.");
    assert_eq!(*agent.status(), Status::Idle);
}

#[derive(Debug, Default, Auto)]
pub struct TestAgent {
    pub persona: Cow<'static, str>,
    pub behavior: Cow<'static, str>,
    pub status: Status,
    pub agent: LmmAgent,
    pub memory: Vec<Message>,
}

#[async_trait]
impl Executor for TestAgent {
    async fn execute<'a>(
        &'a mut self,
        _tasks: &'a mut Task,
        _execute: bool,
        _browse: bool,
        _max_tries: u64,
    ) -> Result<()> {
        self.agent.update(Status::Completed);
        Ok(())
    }
}

#[test]
fn auto_derive_creates_agent_correctly() {
    let agent = TestAgent {
        persona: "Tester".into(),
        behavior: "Test persona.".into(),
        agent: LmmAgent::new("Tester".into(), "Test persona.".into()),
        ..Default::default()
    };

    assert_eq!(Agent::persona(&agent), "Tester");
    assert_eq!(Agent::behavior(&agent), "Test persona.");
    assert_eq!(*Agent::status(&agent), Status::Idle);
}

#[test]
fn auto_derive_functions_get_agent() {
    let agent = TestAgent {
        agent: LmmAgent::new("pos".into(), "obj".into()),
        ..Default::default()
    };
    let inner = Functions::get_agent(&agent);
    assert_eq!(inner.persona(), "pos");
}

#[tokio::test]
async fn generate_returns_non_empty_string() {
    let mut agent = LmmAgent::new(
        "Tester".into(),
        "Rust is fast and safe and concurrent".into(),
    );
    let result = agent.generate("Rust is fast and safe and concurrent").await;
    assert!(result.is_ok());
    assert!(!result.unwrap().is_empty());
}

#[tokio::test]
async fn async_functions_generate_via_derive() {
    let mut agent = TestAgent {
        agent: LmmAgent::new("Tester".into(), "Rust is fast and memory safe.".into()),
        ..Default::default()
    };
    let result = AsyncFunctions::generate(&mut agent, "Rust is fast").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn async_functions_save_and_get_ltm() {
    let mut agent = TestAgent {
        agent: LmmAgent::new("T".into(), "Test.".into()),
        ..Default::default()
    };
    let comm = Message::new("assistant", "Remembered");
    agent.save_ltm(comm.clone()).await.unwrap();
    let ltm = agent.get_ltm().await.unwrap();
    assert_eq!(ltm.len(), 1);
    assert_eq!(ltm[0].content.as_ref(), "Remembered");
}

#[tokio::test]
async fn async_functions_ltm_context_format() {
    let mut agent = TestAgent {
        agent: LmmAgent::new("T".into(), "Test.".into()),
        ..Default::default()
    };
    agent.save_ltm(Message::new("user", "hello")).await.unwrap();
    agent
        .save_ltm(Message::new("assistant", "world"))
        .await
        .unwrap();
    let ctx = agent.ltm_context().await;
    assert!(ctx.contains("user: hello"));
    assert!(ctx.contains("assistant: world"));
}

#[test]
fn auto_agent_build_fails_without_agents() {
    let result = AutoAgent::default().build();
    assert!(result.is_err());
}

#[tokio::test]
async fn auto_agent_runs_single_agent() {
    let agent = TestAgent {
        persona: "Tester".into(),
        behavior: "Complete test.".into(),
        agent: LmmAgent::new("Tester".into(), "Complete test.".into()),
        ..Default::default()
    };

    let result = AutoAgent::default()
        .with(agents![agent])
        .build()
        .unwrap()
        .run()
        .await;

    assert!(result.is_ok());
    assert!(result.unwrap().contains("successfully"));
}
