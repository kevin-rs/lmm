// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Intelligence Primitives Example
//!
//! A comprehensive demonstration integrating **all five intelligence
//! primitives** together in a single cohesive scenario: a self-directed
//! scientific agent investigating an anomalous physical system.
//!
//! | # | Primitive                           | Where                              |
//! |---|-------------------------------------|------------------------------------|
//! | 1 | Calibrated Bayesian Uncertainty     | `lmm::uncertainty`                 |
//! | 2 | Compositional Symbolic Reasoning    | `lmm::reasoner`                    |
//! | 3 | Causal Counterfactual Attribution   | `lmm_agent::cognition::attribution`|
//! | 4 | Hypothesis Formation                | `lmm_agent::cognition::hypothesis` |
//! | 5 | Internalized Motivational Drives    | `lmm_agent::cognition::drive`      |
//!
//! Run with:
//!
//! ```bash
//! cargo run --example intelligence_primitives -p lmm-agent
//! ```

use lmm::causal::CausalGraph;
use lmm::reasoner::{DeductionEngine, SymbolicAxiom};
use lmm::uncertainty::{
    BeliefDistribution, CalibrationObservation, CalibrationRecord, UncertaintyPropagator,
};
use lmm_agent::prelude::*;
use std::collections::HashMap;

fn separator() {
    println!("{}", "=".repeat(60));
}

fn section(n: usize, title: &str) {
    println!();
    separator();
    println!("  [{n}] {title}");
    separator();
}

fn build_pressure_model() -> CausalGraph {
    let mut g = CausalGraph::new();
    g.add_node("temperature_k", Some(350.0));
    g.add_node("volume_l", Some(50.0));
    g.add_node("pressure_pa", None);
    g.add_edge("temperature_k", "pressure_pa", Some(2.0))
        .unwrap();
    g.add_edge("volume_l", "pressure_pa", Some(-1.5)).unwrap();
    g.forward_pass().unwrap();
    g
}

#[tokio::main]
async fn main() {
    let mut agent = LmmAgent::new(
        "Physics Investigator".into(),
        "Reason about thermodynamic systems using causal models and symbolic logic.".into(),
    );

    section(1, "Calibrated Bayesian Uncertainty");

    let temp_prior = BeliefDistribution::new(350.0, 5.0_f64.powi(2));
    let temp_obs = BeliefDistribution::new(352.0, 3.0_f64.powi(2));
    let temp_post = temp_prior.fuse(&temp_obs);

    println!(
        "Prior:     temperature = {:.1} K  ±{:.2} σ",
        temp_prior.mean,
        temp_prior.std_dev()
    );
    println!(
        "Observed:  temperature = {:.1} K  ±{:.2} σ (likelihood)",
        temp_obs.mean,
        temp_obs.std_dev()
    );
    println!(
        "Posterior: temperature = {:.2} K  ±{:.2} σ (Bayesian fusion)",
        temp_post.mean,
        temp_post.std_dev()
    );

    let graph = build_pressure_model();
    let mut initial_beliefs: HashMap<String, BeliefDistribution> = HashMap::new();
    initial_beliefs.insert(
        "temperature_k".to_string(),
        BeliefDistribution::new(350.0, 25.0),
    );
    initial_beliefs.insert("volume_l".to_string(), BeliefDistribution::new(50.0, 4.0));

    let propagated = UncertaintyPropagator::propagate(&graph, &initial_beliefs).unwrap();
    for (var, belief) in &propagated {
        println!(
            "  Propagated [{var:<20}]: mean={:.3}  σ={:.4}",
            belief.mean,
            belief.std_dev()
        );
    }

    let mut cal = CalibrationRecord::new();
    cal.observe(CalibrationObservation {
        lower: 340.0,
        upper: 360.0,
        realized: 351.5,
    });
    cal.observe(CalibrationObservation {
        lower: 340.0,
        upper: 360.0,
        realized: 354.0,
    });
    cal.observe(CalibrationObservation {
        lower: 340.0,
        upper: 370.0,
        realized: 380.0,
    });
    println!(
        "Calibration - hit_rate: {:.1}%  loss: {:.4}",
        cal.hit_rate() * 100.0,
        cal.calibration_loss()
    );

    section(2, "Compositional Axiomatic Reasoning");

    let mut engine = DeductionEngine::new(10);
    engine.register(SymbolicAxiom::new(
        "boyles_law",
        vec!["volume_decreased".into()],
        "pressure_increased",
    ));
    engine.register(SymbolicAxiom::new(
        "charles_law",
        vec!["temperature_increased".into()],
        "volume_increased",
    ));
    engine.register(SymbolicAxiom::new(
        "combined_effect",
        vec!["pressure_increased".into(), "temperature_increased".into()],
        "system_stress_high",
    ));

    let known = vec!["volume_decreased".into(), "temperature_increased".into()];
    let proof = engine.prove(&known, "system_stress_high").unwrap();

    println!("Facts: {known:?}");
    println!(
        "Proved 'system_stress_high': {} ({} steps)",
        proof.succeeded,
        proof.steps.len()
    );
    for step in &proof.steps {
        println!(
            "  [{:.0}%] {} → {}",
            step.confidence * 100.0,
            step.axiom_name,
            step.derived
        );
    }
    println!("Proof confidence: {:.1}%", proof.proof_confidence() * 100.0);

    section(3, "Causal Counterfactual Attribution");

    let report = agent.attribute_causes(&graph, "pressure_pa").unwrap();
    println!("What caused 'pressure_pa'?");
    for (var, weight) in &report.weights {
        let bar = "#".repeat((weight * 20.0) as usize);
        println!("  {var:<20} {weight:.3}  [{bar}]");
    }
    if let Some(d) = report.dominant_cause() {
        println!("Dominant cause: '{d}'");
    }

    section(4, "Hypothesis Formation");

    let mut extended = graph.clone();
    extended.add_node("humidity_pct", Some(80.0));
    extended.forward_pass().unwrap();

    let actual_pressure = graph.get_value("pressure_pa").unwrap_or(0.0) + 120.0;
    let mut observed: HashMap<String, f64> = HashMap::new();
    observed.insert("pressure_pa".to_string(), actual_pressure);

    let hypotheses = agent.form_hypotheses(&extended, &observed, 5).unwrap();
    println!(
        "Model predicted {:.1} Pa; observed {actual_pressure:.1} Pa (unexplained +120.0 Pa)",
        graph.get_value("pressure_pa").unwrap_or(0.0)
    );
    println!("{} hypothesis/es generated:", hypotheses.len());
    for (i, h) in hypotheses.iter().enumerate() {
        println!(
            "  [{}] {} → {}  power={:.3}",
            i + 1,
            h.proposed_edge.from,
            h.proposed_edge.to,
            h.explanatory_power
        );
    }

    section(5, "Internalized Motivational Drives");

    let residual_mag = 120.0 / 700.0;
    agent.record_residual(residual_mag);
    if hypotheses.is_empty() {
        agent.record_incoherence(0.6);
    }

    let state = agent.drive_state();
    println!("Drive state after unexplained residual:");
    if state.signals.is_empty() {
        println!("  [idle - all predictions are well-explained]");
    } else {
        for s in &state.signals {
            let bar = "#".repeat((s.magnitude() * 20.0) as usize);
            println!("  {:30} {:.3}  [{bar}]", s.name(), s.magnitude());
        }
        if let Some(dom) = state.dominant_drive() {
            println!(
                "Agent should prioritise: {} ({:.1}% urgency)",
                dom.name(),
                dom.magnitude() * 100.0
            );
        }
    }

    println!();
    separator();
    println!("All five intelligence primitives demonstrated successfully.");
    println!("Agent: '{}' | Status: {}", agent.persona, agent.status);
}
