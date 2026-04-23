// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Internalized Motivation Example
//!
//! Demonstrates **Internalized Motivational Drives** - the fifth intelligence
//! primitive added to `lmm-agent`.
//!
//! Unlike classical reward-maximizing agents that wait for an external teacher,
//! an agent with internalized drives generates its own motivation signals from
//! internal observations:
//!
//! | Signal                  | Trigger                              |
//! |-------------------------|--------------------------------------|
//! | `Curiosity`             | Unexplained prediction residuals     |
//! | `CoherenceSeeking`      | Internally inconsistent knowledge    |
//! | `ContradictionResolution` | Two facts that contradict each other |
//!
//! The example simulates three ticks of the drive loop to show how different
//! evidence patterns shape the agent's self-generated motivation.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example internalized_motivation -p lmm-agent
//! ```

use lmm_agent::prelude::*;

fn print_drive_state(tick: usize, agent: &mut LmmAgent) {
    let state = agent.drive_state();
    println!("-- Tick {tick} --");
    if state.signals.is_empty() {
        println!("  [idle - no significant drive]");
        return;
    }
    for signal in &state.signals {
        let bar = "#".repeat((signal.magnitude() * 20.0) as usize);
        println!(
            "  {:30} magnitude={:.3} [{bar}]",
            signal.name(),
            signal.magnitude()
        );
    }
    if let Some(d) = state.dominant_drive() {
        println!(
            "  → Dominant drive: {} (urgency {:.3})",
            d.name(),
            d.magnitude()
        );
    }
    println!("  Total urgency: {:.3}", state.total_urgency());
    println!();
}

#[tokio::main]
async fn main() {
    println!("=== Internalized Motivational Drives ===\n");

    let mut agent = LmmAgent::new(
        "Curious Investigator".into(),
        "Understand discrepancies through self-motivated inquiry.".into(),
    );

    println!(
        "Scenario 1: agent encounters large prediction residuals\n  (e.g., world model predicts 2.1°C but observation is 3.6°C)\n"
    );
    agent.record_residual(1.5);
    agent.record_residual(0.9);
    print_drive_state(1, &mut agent);

    println!(
        "Scenario 2: knowledge base contains internally inconsistent claims\n  (e.g., two passages contradict each other on CO₂ sensitivity)\n"
    );
    agent.record_incoherence(3.5);
    print_drive_state(2, &mut agent);

    println!("Scenario 3: mixed evidence - multiple signals fire simultaneously\n");
    agent.record_residual(0.6);
    agent.record_incoherence(1.2);
    agent.record_contradiction();
    agent.record_contradiction();
    print_drive_state(3, &mut agent);

    println!("Scenario 4: no new evidence - agent is idle\n");
    print_drive_state(4, &mut agent);

    println!("Done.");
}
