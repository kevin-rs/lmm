// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Causal Attribution
//!
//! This module enables an agent to answer *why* an outcome occurred by running
//! counterfactual interventions on a [`lmm::causal::CausalGraph`] and
//! measuring how much each parent variable is responsible.
//!
//! ## Design
//!
//! Rather than correlating features with outcomes, [`CausalAttributor`] performs
//! the Pearl *do*-calculus intervention `do(parent = 0)` for each parent of the
//! query variable and computes the resulting change in the query variable's value.
//! The absolute intervention effect is normalised so all attribution weights sum
//! to `1.0`, making the result directly interpretable as "fraction of causation."
//!
//! ## See Also
//!
//! - [Pearl, J. (2009). Causality. Cambridge University Press.](https://doi.org/10.1017/CBO9780511803161)
//! - [Shapley value - Wikipedia](https://en.wikipedia.org/wiki/Shapley_value)

use lmm::causal::CausalGraph;
use lmm::error::Result;

/// The fraction of the outcome attributable to each parent variable.
///
/// Weights are non-negative and normalised to sum to `1.0`.
/// The list is sorted by descending weight (most responsible variable first).
///
/// # Examples
///
/// ```
/// use lmm::causal::CausalGraph;
/// use lmm_agent::cognition::attribution::{CausalAttributor, AttributionReport};
///
/// let mut g = CausalGraph::new();
/// g.add_node("cause_a", Some(1.0));
/// g.add_node("cause_b", Some(0.0));
/// g.add_node("effect", None);
/// g.add_edge("cause_a", "effect", Some(3.0)).unwrap();
/// g.add_edge("cause_b", "effect", Some(1.0)).unwrap();
/// g.forward_pass().unwrap();
///
/// let report = CausalAttributor::attribute(&g, "effect").unwrap();
/// assert!(!report.weights.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct AttributionReport {
    /// Variable name → fractional attribution, sorted highest first.
    pub weights: Vec<(String, f64)>,
    /// The query variable whose outcome was attributed.
    pub outcome_variable: String,
}

impl AttributionReport {
    /// Returns the attribution weight for a specific variable, or `None` if absent.
    pub fn weight_for(&self, variable: &str) -> Option<f64> {
        self.weights
            .iter()
            .find(|(v, _)| v == variable)
            .map(|(_, w)| *w)
    }

    /// Returns the variable carrying the highest attribution weight.
    pub fn dominant_cause(&self) -> Option<&str> {
        self.weights.first().map(|(v, _)| v.as_str())
    }
}

/// Attributes the outcome of a query variable to its causal parents via counterfactuals.
///
/// For each parent `p` of `query_var`, the attributor performs `do(p = 0)` and
/// measures the resulting shift in `query_var`.  The magnitude of each shift is
/// then normalised to produce fractional attribution weights.
pub struct CausalAttributor;

impl CausalAttributor {
    /// Computes an [`AttributionReport`] for `query_var` in `graph`.
    ///
    /// # Arguments
    ///
    /// * `graph`     - The causal structure containing the outcome and its parents.
    /// * `query_var` - The variable whose causes are to be attributed.
    ///
    /// # Returns
    ///
    /// (`Result<AttributionReport>`): Attribution weights for each parent,
    /// normalised to sum to `1.0`, sorted descending.
    ///
    /// # Errors
    ///
    /// Returns [`lmm::error::LmmError::CausalError`] when `query_var` is unknown
    /// or the graph contains a cycle.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::causal::CausalGraph;
    /// use lmm_agent::cognition::attribution::CausalAttributor;
    ///
    /// let mut g = CausalGraph::new();
    /// g.add_node("x", Some(2.0));
    /// g.add_node("y", None);
    /// g.add_edge("x", "y", Some(1.0)).unwrap();
    /// g.forward_pass().unwrap();
    ///
    /// let report = CausalAttributor::attribute(&g, "y").unwrap();
    /// assert_eq!(report.weights[0].0, "x");
    /// assert!((report.weights[0].1 - 1.0).abs() < 1e-9);
    /// ```
    pub fn attribute(graph: &CausalGraph, query_var: &str) -> Result<AttributionReport> {
        graph.topological_order()?;

        let baseline = graph.get_value(query_var).unwrap_or(0.0);
        let parents = graph.parents(query_var);

        let mut raw_effects: Vec<(String, f64)> = parents
            .iter()
            .map(|parent| {
                let counterfactual = graph
                    .counterfactual(parent, 0.0, query_var)
                    .unwrap_or(baseline);
                let effect = (baseline - counterfactual).abs();
                (parent.clone(), effect)
            })
            .collect();

        let total: f64 = raw_effects.iter().map(|(_, e)| e).sum();

        if total < 1e-12 {
            let uniform = if raw_effects.is_empty() {
                0.0
            } else {
                1.0 / raw_effects.len() as f64
            };
            for (_, w) in &mut raw_effects {
                *w = uniform;
            }
        } else {
            for (_, w) in &mut raw_effects {
                *w /= total;
            }
        }

        raw_effects.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(AttributionReport {
            weights: raw_effects,
            outcome_variable: query_var.to_string(),
        })
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
