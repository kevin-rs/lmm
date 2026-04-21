// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Causal Graphs
//!
//! This module provides [`CausalGraph`], a directed acyclic graph (DAG) that represents
//! causal relationships between variables. It supports:
//!
//! - Adding nodes and directed edges between them.
//! - **Forward propagation** - computing effect values from causes.
//! - **Interventions** - the `do(·)` operator from Pearl's causal calculus, which severs
//!   all incoming edges to a node and fixes its value.
//! - **Topological ordering** - computed efficiently via Kahn's BFS algorithm in O(V + E).
//! - **Counterfactual queries** - evaluating "what if?" scenarios.
//!
//! # See Also
//! - [Pearl, J. (2009). Causality. Cambridge University Press.](https://doi.org/10.1017/CBO9780511803161) - foundational text defining the `do`-calculus and intervention mechanics.
//! - [Peters, J., Janzing, D., & Schölkopf, B. (2017). Elements of Causal Inference.](https://mitpress.mit.edu/9780262037310/) - accessible reference for Structural Causal Models (SCMs) and counterfactual evaluation.

use crate::error::{LmmError, Result};
use crate::traits::Causal;
use std::collections::{HashMap, HashSet, VecDeque};

/// A node in a [`CausalGraph`], representing a random variable with an optional fixed value.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::causal::CausalNode;
///
/// let node = CausalNode { name: "rain".to_string(), value: Some(1.0) };
/// assert_eq!(node.name, "rain");
/// ```
#[derive(Debug, Clone)]
pub struct CausalNode {
    /// Human-readable name of the variable (must be unique within a graph).
    pub name: String,
    /// Current value of the variable; `None` means the value is unobserved/unevaluated.
    pub value: Option<f64>,
}

/// A directed causal edge `parent → child` with an optional linear coefficient.
///
/// When `coefficient` is `Some(w)` the contribution of the parent to the child's value is
/// `parent_value * w`. When `None`, the default coefficient of `1.0` is used.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::causal::CausalEdge;
///
/// let edge = CausalEdge { from: "a".into(), to: "b".into(), coefficient: Some(0.5) };
/// assert_eq!(edge.coefficient.unwrap(), 0.5);
/// ```
#[derive(Debug, Clone)]
pub struct CausalEdge {
    /// The parent variable name.
    pub from: String,
    /// The child variable name.
    pub to: String,
    /// Linear coefficient applied to the parent's contribution; defaults to `1.0`.
    pub coefficient: Option<f64>,
}

/// A directed acyclic graph of [`CausalNode`]s connected by [`CausalEdge`]s.
///
/// All mutations return `Result` so that invalid operations (unknown nodes, cycles, etc.)
/// are reported explicitly rather than silently corrupting the graph.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::causal::CausalGraph;
///
/// let mut g = CausalGraph::new();
/// g.add_node("rain", Some(1.0));
/// g.add_node("wet", None);
/// g.add_edge("rain", "wet", Some(0.8)).unwrap();
/// g.forward_pass().unwrap();
/// let wet = g.get_value("wet").unwrap();
/// assert!((wet - 0.8).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Default)]
pub struct CausalGraph {
    /// The ordered list of nodes in the graph.
    pub nodes: Vec<CausalNode>,
    /// All directed edges in the graph.
    pub edges: Vec<CausalEdge>,
}

impl CausalGraph {
    /// Creates an empty causal graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::causal::CausalGraph;
    /// let g = CausalGraph::new();
    /// assert!(g.nodes.is_empty());
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a variable node with name `name` and optional initial value.
    ///
    /// If a node with the same name already exists, this is a no-op.
    ///
    /// # Arguments
    ///
    /// * `name` - Unique variable name.
    /// * `value` - Initial value, or `None` if not yet observed.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::causal::CausalGraph;
    ///
    /// let mut g = CausalGraph::new();
    /// g.add_node("x", Some(2.0));
    /// assert_eq!(g.get_value("x").unwrap(), 2.0);
    /// ```
    pub fn add_node(&mut self, name: &str, value: Option<f64>) {
        if !self.nodes.iter().any(|n| n.name == name) {
            self.nodes.push(CausalNode {
                name: name.to_string(),
                value,
            });
        }
    }

    /// Adds a directed causal edge `from → to` with an optional linear coefficient.
    ///
    /// # Arguments
    ///
    /// * `from` - Parent node name (must already exist in the graph).
    /// * `to` - Child node name (must already exist in the graph).
    /// * `coefficient` - Optional linear weight; defaults to `1.0`.
    ///
    /// # Errors
    ///
    /// Returns [`LmmError::CausalError`] when either `from` or `to` is not found.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::causal::CausalGraph;
    ///
    /// let mut g = CausalGraph::new();
    /// g.add_node("a", Some(1.0));
    /// g.add_node("b", None);
    /// assert!(g.add_edge("a", "b", Some(2.0)).is_ok());
    /// assert!(g.add_edge("a", "missing", None).is_err());
    /// ```
    pub fn add_edge(&mut self, from: &str, to: &str, coefficient: Option<f64>) -> Result<()> {
        if !self.nodes.iter().any(|n| n.name == from) {
            return Err(LmmError::CausalError(format!("Node '{}' not found", from)));
        }
        if !self.nodes.iter().any(|n| n.name == to) {
            return Err(LmmError::CausalError(format!("Node '{}' not found", to)));
        }
        self.edges.push(CausalEdge {
            from: from.to_string(),
            to: to.to_string(),
            coefficient,
        });
        Ok(())
    }

    /// Returns the current value of node `name`, or `None` if unobserved.
    ///
    /// # Arguments
    ///
    /// * `name` - The node name to look up.
    ///
    /// # Returns
    ///
    /// (`Option<f64>`): The node's value, or `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::causal::CausalGraph;
    ///
    /// let mut g = CausalGraph::new();
    /// g.add_node("x", Some(3.14));
    /// assert_eq!(g.get_value("x"), Some(3.14));
    /// assert_eq!(g.get_value("y"), None);
    /// ```
    pub fn get_value(&self, name: &str) -> Option<f64> {
        self.nodes.iter().find(|n| n.name == name)?.value
    }

    /// Computes a topological ordering of all nodes using **Kahn's BFS algorithm**.
    ///
    /// This replaces the previous O(n² log n) sort-based implementation with a proper
    /// O(V + E) BFS traversal over in-degree counts.
    ///
    /// # Returns
    ///
    /// (`Result<Vec<String>>`): Node names in a valid topological order.
    ///
    /// # Errors
    ///
    /// Returns [`LmmError::CausalError`] when a cycle is detected (BFS exhausts the
    /// queue before processing all nodes).
    ///
    /// # Time Complexity
    ///
    /// O(V + E) where V = number of nodes, E = number of edges.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::causal::CausalGraph;
    ///
    /// let mut g = CausalGraph::new();
    /// g.add_node("a", Some(1.0));
    /// g.add_node("b", None);
    /// g.add_node("c", None);
    /// g.add_edge("a", "b", None).unwrap();
    /// g.add_edge("b", "c", None).unwrap();
    /// let order = g.topological_order().unwrap();
    /// assert_eq!(order[0], "a");
    /// assert_eq!(order[2], "c");
    /// ```
    pub fn topological_order(&self) -> Result<Vec<String>> {
        let mut in_degree: HashMap<&str, usize> = self
            .nodes
            .iter()
            .map(|n| (n.name.as_str(), 0usize))
            .collect();
        let mut adj: HashMap<&str, Vec<&str>> = self
            .nodes
            .iter()
            .map(|n| (n.name.as_str(), vec![]))
            .collect();

        for edge in &self.edges {
            *in_degree.entry(edge.to.as_str()).or_insert(0) += 1;
            adj.entry(edge.from.as_str())
                .or_default()
                .push(edge.to.as_str());
        }

        let mut queue: VecDeque<&str> = self
            .nodes
            .iter()
            .filter(|n| in_degree.get(n.name.as_str()).copied().unwrap_or(0) == 0)
            .map(|n| n.name.as_str())
            .collect();

        let mut order: Vec<String> = Vec::with_capacity(self.nodes.len());
        while let Some(node) = queue.pop_front() {
            order.push(node.to_string());
            for &child in adj.get(node).map(Vec::as_slice).unwrap_or_default() {
                let deg = in_degree.entry(child).or_insert(0);
                *deg = deg.saturating_sub(1);
                if *deg == 0 {
                    queue.push_back(child);
                }
            }
        }

        if order.len() != self.nodes.len() {
            return Err(LmmError::CausalError(
                "Cycle detected in causal graph; topological ordering impossible".into(),
            ));
        }
        Ok(order)
    }

    /// Propagates all node values forward through the graph in topological order.
    ///
    /// For each child node, all parent contributions `coeff * parent_value` are summed.
    /// A child with no parents retains its existing value.
    ///
    /// # Returns
    ///
    /// (`Result<()>`): `Ok` on success.
    ///
    /// # Errors
    ///
    /// Returns [`LmmError::CausalError`] when a cycle is detected.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::causal::CausalGraph;
    ///
    /// let mut g = CausalGraph::new();
    /// g.add_node("x", Some(10.0));
    /// g.add_node("y", None);
    /// g.add_edge("x", "y", Some(0.5)).unwrap();
    /// g.forward_pass().unwrap();
    /// assert_eq!(g.get_value("y"), Some(5.0));
    /// ```
    pub fn forward_pass(&mut self) -> Result<()> {
        let order = self.topological_order()?;

        let mut parent_map: HashMap<String, Vec<(String, f64)>> = HashMap::new();
        for edge in &self.edges {
            parent_map
                .entry(edge.to.clone())
                .or_default()
                .push((edge.from.clone(), edge.coefficient.unwrap_or(1.0)));
        }

        let mut values: HashMap<String, Option<f64>> = self
            .nodes
            .iter()
            .map(|n| (n.name.clone(), n.value))
            .collect();

        for name in &order {
            if let Some(parents) = parent_map.get(name) {
                let has_all_parents = parents
                    .iter()
                    .all(|(p, _)| values.get(p).and_then(|v| *v).is_some());
                if has_all_parents {
                    let sum: f64 = parents
                        .iter()
                        .filter_map(|(p, coeff)| values.get(p).and_then(|v| *v).map(|v| v * coeff))
                        .sum();
                    values.insert(name.clone(), Some(sum));
                }
            }
        }

        for node in &mut self.nodes {
            if let Some(v) = values.get(&node.name) {
                node.value = *v;
            }
        }
        Ok(())
    }

    /// Runs a counterfactual query: what would `query_node` be if `intervention_node`
    /// were set to `intervention_value`?
    ///
    /// This method clones the graph, performs the intervention, propagates, and returns
    /// the resulting value without mutating `self`.
    ///
    /// # Arguments
    ///
    /// * `intervention_node` - The variable to intervene on.
    /// * `intervention_value` - The value to impose via `do(·)`.
    /// * `query_node` - The variable whose counterfactual value is queried.
    ///
    /// # Returns
    ///
    /// (`Result<f64>`): The counterfactual value of `query_node`.
    ///
    /// # Errors
    ///
    /// Returns [`LmmError::CausalError`] when either node is not found or the graph
    /// contains a cycle.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::causal::CausalGraph;
    ///
    /// let mut g = CausalGraph::new();
    /// g.add_node("rain", Some(0.0));
    /// g.add_node("wet", None);
    /// g.add_edge("rain", "wet", Some(1.0)).unwrap();
    ///
    /// let cf = g.counterfactual("rain", 1.0, "wet").unwrap();
    /// assert_eq!(cf, 1.0);
    /// ```
    pub fn counterfactual(
        &self,
        intervention_node: &str,
        intervention_value: f64,
        query_node: &str,
    ) -> Result<f64> {
        let mut g = self.clone();
        g.intervene(intervention_node, intervention_value)?;
        g.forward_pass()?;
        g.get_value(query_node).ok_or_else(|| {
            LmmError::CausalError(format!(
                "Query node '{}' has no value after propagation",
                query_node
            ))
        })
    }

    /// Returns the names of all direct parents of `node`.
    pub fn parents(&self, node: &str) -> Vec<String> {
        self.edges
            .iter()
            .filter(|e| e.to == node)
            .map(|e| e.from.clone())
            .collect()
    }

    /// Returns the names of all direct children of `node`.
    pub fn children(&self, node: &str) -> Vec<String> {
        self.edges
            .iter()
            .filter(|e| e.from == node)
            .map(|e| e.to.clone())
            .collect()
    }

    /// Returns `true` when the graph contains a directed cycle.
    pub fn has_cycle(&self) -> bool {
        self.topological_order().is_err()
    }

    /// Returns the Markov blanket of `node` (parents + children + co-parents).
    pub fn markov_blanket(&self, node: &str) -> HashSet<String> {
        let mut blanket = HashSet::new();
        for p in self.parents(node) {
            blanket.insert(p);
        }
        for child in self.children(node) {
            blanket.insert(child.clone());
            for cp in self.parents(&child) {
                if cp != node {
                    blanket.insert(cp);
                }
            }
        }
        blanket
    }
}

impl Causal for CausalGraph {
    /// Performs the hard intervention `do(var = value)`.
    ///
    /// Severs all edges entering `var`, then sets its value to `value`.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable name to intervene on.
    /// * `value` - The value to impose.
    ///
    /// # Errors
    ///
    /// Returns [`LmmError::CausalError`] when `var` is not a known node.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::causal::CausalGraph;
    /// use lmm::traits::Causal;
    ///
    /// let mut g = CausalGraph::new();
    /// g.add_node("x", Some(0.0));
    /// g.intervene("x", 5.0).unwrap();
    /// assert_eq!(g.get_value("x"), Some(5.0));
    /// ```
    fn intervene(&mut self, var: &str, value: f64) -> Result<()> {
        let found = self.nodes.iter_mut().find(|n| n.name == var);
        if let Some(node) = found {
            node.value = Some(value);
            self.edges.retain(|e| e.to != var);
            Ok(())
        } else {
            Err(LmmError::CausalError(format!(
                "Intervention target '{}' not found in graph",
                var
            )))
        }
    }
}

/// Builds a simple chain graph `x₀ → x₁ → ... → xₙ₋₁` for testing and demonstration.
///
/// # Arguments
///
/// * `n` - Number of nodes.
/// * `coefficient` - Edge coefficient applied at each step.
///
/// # Returns
///
/// (`CausalGraph`): An `n`-node chain causal graph.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::causal::build_chain;
///
/// let g = build_chain(3, 2.0);
/// assert_eq!(g.nodes.len(), 3);
/// ```
pub fn build_chain(n: usize, coefficient: f64) -> CausalGraph {
    let mut g = CausalGraph::new();
    for i in 0..n {
        let val = if i == 0 { Some(1.0) } else { None };
        g.add_node(&format!("x{}", i), val);
    }
    for i in 0..n.saturating_sub(1) {
        let _ = g.add_edge(
            &format!("x{}", i),
            &format!("x{}", i + 1),
            Some(coefficient),
        );
    }
    g
}

/// Returns the set of ancestors of `node` in `g` (all variables that causally precede it).
///
/// # Arguments
///
/// * `g` - The causal graph.
/// * `node` - The variable to find ancestors for.
///
/// # Returns
///
/// (`HashSet<String>`): All ancestor variable names.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::causal::{CausalGraph, ancestors};
///
/// let mut g = CausalGraph::new();
/// g.add_node("a", Some(1.0));
/// g.add_node("b", None);
/// g.add_node("c", None);
/// g.add_edge("a", "b", None).unwrap();
/// g.add_edge("b", "c", None).unwrap();
///
/// let anc = ancestors(&g, "c");
/// assert!(anc.contains("a"));
/// assert!(anc.contains("b"));
/// assert!(!anc.contains("c"));
/// ```
pub fn ancestors(g: &CausalGraph, node: &str) -> HashSet<String> {
    let mut result = HashSet::new();
    let mut queue: VecDeque<&str> = VecDeque::new();
    queue.push_back(node);

    while let Some(current) = queue.pop_front() {
        for edge in &g.edges {
            if edge.to == current && !result.contains(&edge.from) {
                result.insert(edge.from.clone());
                queue.push_back(&edge.from);
            }
        }
    }
    result
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
