// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm_agent::types::{Goal, Knowledge, Message, Planner, Status, Task};

#[test]
fn status_default_is_idle() {
    assert_eq!(Status::default(), Status::Idle);
}

#[test]
fn message_new() {
    let c = Message::new("user", "hello");
    assert_eq!(c.role.as_ref(), "user");
    assert_eq!(c.content.as_ref(), "hello");
}

#[test]
fn knowledge_insert_and_get() {
    let mut kb = Knowledge::default();
    kb.insert("Rust", "A systems language.");
    assert_eq!(
        kb.get("Rust").map(|c| c.as_ref()),
        Some("A systems language.")
    );
}

#[test]
fn planner_done_when_all_completed() {
    let plan = Planner {
        current_plan: vec![
            Goal {
                description: "A".into(),
                priority: 0,
                completed: true,
            },
            Goal {
                description: "B".into(),
                priority: 1,
                completed: true,
            },
        ],
    };
    assert!(plan.is_done());
    assert_eq!(plan.completed_count(), 2);
}

#[test]
fn task_from_description() {
    let t = Task::from_description("Do something.");
    assert_eq!(t.description.as_ref(), "Do something.");
    assert!(t.scope.is_none());
}
