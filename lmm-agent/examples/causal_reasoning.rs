// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Causal Attribution Example
//!
//! Demonstrates **Counterfactual Causal Attribution** and **Hypothesis Formation** -
//! two of the five intelligence primitives added to `lmm-agent`.
//!
//! What you will see:
//!
//! 1. A causal world model is built for a simplified climate scenario.
//! 2. `LmmAgent::attribute_causes` performs **chained attribution**: first
//!    attributing `temp_anomaly_c` to the direct cause (`greenhouse_effect`),
//!    then attributing `greenhouse_effect` to its root causes (`co2_ppm` and
//!    `solar_forcing`).  This shows how multi-hop causal chains are navigated.
//! 3. An observation that the model cannot explain triggers
//!    `LmmAgent::form_hypotheses`, which proposes new causal edges ranked by
//!    explanatory power using a **unit-coefficient criterion**: each parent's
//!    value (at coefficient 1.0) is compared to the residual, so larger-valued
//!    variables that can easily cover the gap rank higher than smaller-valued
//!    ones.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example causal_reasoning -p lmm-agent
//! ```

use lmm::causal::CausalGraph;
use lmm_agent::prelude::*;
use std::collections::HashMap;

fn build_climate_model() -> CausalGraph {
    let mut g = CausalGraph::new();

    g.add_node("co2_ppm", Some(420.0));
    g.add_node("solar_forcing", Some(0.3));
    g.add_node("greenhouse_effect", None);
    g.add_node("temp_anomaly_c", None);

    g.add_edge("co2_ppm", "greenhouse_effect", Some(0.005))
        .unwrap();
    g.add_edge("solar_forcing", "greenhouse_effect", Some(0.4))
        .unwrap();
    g.add_edge("greenhouse_effect", "temp_anomaly_c", Some(0.8))
        .unwrap();

    g.forward_pass().unwrap();
    g
}

fn print_bar(weight: f64) -> String {
    "#".repeat((weight * 20.0) as usize)
}

#[tokio::main]
async fn main() {
    println!("=== Causal Attribution & Hypothesis Formation ===\n");

    let agent = LmmAgent::new(
        "Climate Analyst".into(),
        "Attribute climate change causal factors.".into(),
    );

    let graph = build_climate_model();

    println!(
        "World model: greenhouse_effect = {:.3}",
        graph.get_value("greenhouse_effect").unwrap_or(0.0)
    );
    println!(
        "World model: temp_anomaly_c   = {:.3}\n",
        graph.get_value("temp_anomaly_c").unwrap_or(0.0)
    );

    println!("Step 1a: What directly caused 'temp_anomaly_c'?");
    let report_temp = agent.attribute_causes(&graph, "temp_anomaly_c").unwrap();
    for (var, weight) in &report_temp.weights {
        println!("  {var:<25} {weight:.3} [{}]", print_bar(*weight));
    }
    println!();

    if let Some(direct_cause) = report_temp.dominant_cause() {
        println!("Step 1b: '{direct_cause}' is the direct cause. Tracing it to root causes...");
        let report_ghg = agent.attribute_causes(&graph, direct_cause).unwrap();
        if report_ghg.weights.is_empty() {
            println!("  (no further parents - '{direct_cause}' is a root variable)");
        } else {
            for (var, weight) in &report_ghg.weights {
                println!("  {var:<25} {weight:.3} [{}]", print_bar(*weight));
            }
            if let Some(root) = report_ghg.dominant_cause() {
                println!("\n  → Root cause chain: '{root}' → '{direct_cause}' → 'temp_anomaly_c'");
            }
        }
    }
    println!();

    let mut graph2 = graph.clone();
    graph2.add_node("methane_ppb", Some(1900.0));
    graph2.add_node("aerosol_index", Some(0.15));
    graph2.forward_pass().unwrap();

    let under_pred = 0.4;
    let mut observed: HashMap<String, f64> = HashMap::new();
    observed.insert(
        "temp_anomaly_c".to_string(),
        graph.get_value("temp_anomaly_c").unwrap_or(0.0) + under_pred,
    );

    let hypotheses = agent.form_hypotheses(&graph2, &observed, 5).unwrap();

    println!("Step 2: Model under-predicts temp_anomaly_c by {under_pred:.1} K. Candidate causes?");
    println!(
        "  (ranking by unit-coefficient explanatory power, then Occam's-razor coefficient size)\n"
    );
    if hypotheses.is_empty() {
        println!("  (no hypotheses above threshold)");
    } else {
        for (i, h) in hypotheses.iter().enumerate() {
            println!(
                "  [{}] {:<20} → temp_anomaly_c  power={:.3}  required_coeff={:.5}",
                i + 1,
                h.proposed_edge.from,
                h.explanatory_power,
                h.proposed_edge.coefficient.unwrap_or(0.0)
            );
        }
        let best = &hypotheses[0];
        println!(
            "\n  → Best hypothesis: '{}' (explains {:.1}% of residual at unit coeff, needs coeff={:.5})",
            best.proposed_edge.from,
            best.explanatory_power * 100.0,
            best.proposed_edge.coefficient.unwrap_or(0.0)
        );
    }

    println!("\nDone.");
}
