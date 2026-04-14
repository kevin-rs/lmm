use lmm::causal::CausalGraph;
use lmm::equation::Expression;
use lmm::traits::Causal;
use std::collections::HashMap;

fn build_chain() -> (CausalGraph, usize, usize, usize) {
    let mut g = CausalGraph::new();
    let x = g.add_node("x", None);
    let y = g.add_node(
        "y",
        Some(Expression::Mul(
            Box::new(Expression::Constant(2.0)),
            Box::new(Expression::Variable("x".into())),
        )),
    );
    let z = g.add_node(
        "z",
        Some(Expression::Add(
            Box::new(Expression::Variable("y".into())),
            Box::new(Expression::Constant(1.0)),
        )),
    );
    g.add_edge(x, y, 1.0).unwrap();
    g.add_edge(y, z, 1.0).unwrap();
    g.nodes[x].observed_value = Some(3.0);
    (g, x, y, z)
}

#[test]
fn test_forward_pass() {
    let (mut g, x_id, y_id, z_id) = build_chain();
    let vals = g.forward_pass().unwrap();
    assert!((vals[&x_id] - 3.0).abs() < 1e-10);
    assert!((vals[&y_id] - 6.0).abs() < 1e-10);
    assert!((vals[&z_id] - 7.0).abs() < 1e-10);
}

#[test]
fn test_intervention_freezes_node() {
    let (mut g, _x_id, y_id, z_id) = build_chain();
    g.intervene("y", 10.0).unwrap();
    let vals = g.forward_pass().unwrap();
    assert!(
        (vals[&y_id] - 10.0).abs() < 1e-10,
        "Intervened y should be 10"
    );
    assert!((vals[&z_id] - 11.0).abs() < 1e-10, "z = y+1 = 11");
}

#[test]
fn test_intervention_isolates_from_parent() {
    let (mut g, x_id, y_id, _z_id) = build_chain();
    g.nodes[x_id].observed_value = Some(100.0);
    g.intervene("y", 5.0).unwrap();
    let vals = g.forward_pass().unwrap();
    assert!((vals[&y_id] - 5.0).abs() < 1e-10, "do(y=5) ignores x=100");
}

#[test]
fn test_counterfactual() {
    let (g, x_id, y_id, z_id) = build_chain();
    let observed = HashMap::from([(x_id, 2.0)]);
    let intervention = HashMap::from([(x_id, 5.0)]);
    let cf = g.counterfactual(&observed, &intervention).unwrap();
    assert!((cf[&x_id] - 5.0).abs() < 1e-10);
    assert!((cf[&y_id] - 10.0).abs() < 1e-10);
    assert!((cf[&z_id] - 11.0).abs() < 1e-10);
}

#[test]
fn test_parents_children() {
    let (g, x_id, y_id, z_id) = build_chain();
    assert_eq!(g.parents(y_id), vec![x_id]);
    assert_eq!(g.children(y_id), vec![z_id]);
    assert!(g.parents(x_id).is_empty());
    assert!(g.children(z_id).is_empty());
}

#[test]
fn test_markov_blanket() {
    let (g, x_id, y_id, z_id) = build_chain();
    let blanket = g.markov_blanket(y_id);
    assert!(
        blanket.contains(&x_id),
        "Parent x should be in Markov blanket of y"
    );
    assert!(
        blanket.contains(&z_id),
        "Child z should be in Markov blanket of y"
    );
}

#[test]
fn test_topological_order() {
    let (g, x_id, y_id, z_id) = build_chain();
    let order = g.topological_order().unwrap();
    let pos: HashMap<usize, usize> = order.iter().enumerate().map(|(i, &n)| (n, i)).collect();
    assert!(pos[&x_id] < pos[&y_id], "x before y");
    assert!(pos[&y_id] < pos[&z_id], "y before z");
}

#[test]
fn test_cycle_detection() {
    let mut g = CausalGraph::new();
    let a = g.add_node("a", None);
    let b = g.add_node("b", None);
    g.add_edge(a, b, 1.0).unwrap();
    g.add_edge(b, a, 1.0).unwrap();
    assert!(g.has_cycle());
}

#[test]
fn test_no_cycle_in_dag() {
    let (g, _, _, _) = build_chain();
    assert!(!g.has_cycle());
}

#[test]
fn test_invalid_edge_error() {
    let mut g = CausalGraph::new();
    g.add_node("a", None);
    assert!(g.add_edge(0, 99, 1.0).is_err());
}

#[test]
fn test_intervene_unknown_node_error() {
    let mut g = CausalGraph::new();
    g.add_node("a", None);
    assert!(g.intervene("nonexistent", 1.0).is_err());
}
