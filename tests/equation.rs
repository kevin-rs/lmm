use lmm::equation::Expression;
use std::collections::HashMap;

fn vars(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
    pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
}

#[test]
fn test_constant() {
    let e = Expression::Constant(42.0);
    assert_eq!(e.evaluate(&vars(&[])).unwrap(), 42.0);
}

#[test]
fn test_variable() {
    let e = Expression::Variable("x".into());
    assert_eq!(e.evaluate(&vars(&[("x", 7.0)])).unwrap(), 7.0);
}

#[test]
fn test_missing_variable_error() {
    let e = Expression::Variable("y".into());
    assert!(e.evaluate(&vars(&[])).is_err());
}

#[test]
fn test_add_evaluate() {
    let e = Expression::Add(
        Box::new(Expression::Variable("x".into())),
        Box::new(Expression::Constant(2.0)),
    );
    assert_eq!(e.evaluate(&vars(&[("x", 5.0)])).unwrap(), 7.0);
}

#[test]
fn test_sub_evaluate() {
    let e = Expression::Sub(
        Box::new(Expression::Constant(10.0)),
        Box::new(Expression::Variable("x".into())),
    );
    assert_eq!(e.evaluate(&vars(&[("x", 3.0)])).unwrap(), 7.0);
}

#[test]
fn test_mul_evaluate() {
    let e = Expression::Mul(
        Box::new(Expression::Constant(3.0)),
        Box::new(Expression::Variable("x".into())),
    );
    assert_eq!(e.evaluate(&vars(&[("x", 4.0)])).unwrap(), 12.0);
}

#[test]
fn test_div_evaluate() {
    let e = Expression::Div(
        Box::new(Expression::Constant(10.0)),
        Box::new(Expression::Constant(2.0)),
    );
    assert_eq!(e.evaluate(&vars(&[])).unwrap(), 5.0);
}

#[test]
fn test_div_by_zero() {
    let e = Expression::Div(
        Box::new(Expression::Constant(1.0)),
        Box::new(Expression::Constant(0.0)),
    );
    assert!(e.evaluate(&vars(&[])).is_err());
}

#[test]
fn test_pow_evaluate() {
    let e = Expression::Pow(
        Box::new(Expression::Variable("x".into())),
        Box::new(Expression::Constant(2.0)),
    );
    assert!((e.evaluate(&vars(&[("x", 3.0)])).unwrap() - 9.0).abs() < 1e-10);
}

#[test]
fn test_sin_cos() {
    let s = Expression::Sin(Box::new(Expression::Constant(0.0)));
    let c = Expression::Cos(Box::new(Expression::Constant(0.0)));
    assert!((s.evaluate(&vars(&[])).unwrap()).abs() < 1e-10);
    assert!((c.evaluate(&vars(&[])).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_exp_log() {
    let e = Expression::Exp(Box::new(Expression::Constant(1.0)));
    assert!((e.evaluate(&vars(&[])).unwrap() - std::f64::consts::E).abs() < 1e-10);
    let l = Expression::Log(Box::new(Expression::Constant(std::f64::consts::E)));
    assert!((l.evaluate(&vars(&[])).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_neg() {
    let e = Expression::Neg(Box::new(Expression::Constant(5.0)));
    assert_eq!(e.evaluate(&vars(&[])).unwrap(), -5.0);
}

#[test]
fn test_abs() {
    let e = Expression::Abs(Box::new(Expression::Constant(-7.0)));
    assert_eq!(e.evaluate(&vars(&[])).unwrap(), 7.0);
}

#[test]
fn test_complexity() {
    let leaf = Expression::Constant(1.0);
    assert_eq!(leaf.complexity(), 1);
    let add = Expression::Add(
        Box::new(Expression::Constant(1.0)),
        Box::new(Expression::Constant(2.0)),
    );
    assert_eq!(add.complexity(), 3);
}

#[test]
fn test_symbolic_diff_constant() {
    let e = Expression::Constant(5.0);
    let d = e.symbolic_diff("x").simplify();
    assert_eq!(d, Expression::Constant(0.0));
}

#[test]
fn test_symbolic_diff_variable() {
    let e = Expression::Variable("x".into());
    let d = e.symbolic_diff("x").simplify();
    assert_eq!(d, Expression::Constant(1.0));
}

#[test]
fn test_symbolic_diff_mul() {
    let e = Expression::Mul(
        Box::new(Expression::Constant(2.0)),
        Box::new(Expression::Variable("x".into())),
    );
    let d = e.symbolic_diff("x").simplify();
    assert!((d.evaluate(&vars(&[])).unwrap() - 2.0).abs() < 1e-10);
}

#[test]
fn test_symbolic_diff_sin() {
    let e = Expression::Sin(Box::new(Expression::Variable("x".into())));
    let d = e.symbolic_diff("x").simplify();
    let val = d.evaluate(&vars(&[("x", 0.0)])).unwrap();
    assert!((val - 1.0).abs() < 1e-10);
}

#[test]
fn test_simplify_add_zero() {
    let e = Expression::Add(
        Box::new(Expression::Variable("x".into())),
        Box::new(Expression::Constant(0.0)),
    );
    let s = e.simplify();
    assert_eq!(s, Expression::Variable("x".into()));
}

#[test]
fn test_simplify_mul_one() {
    let e = Expression::Mul(
        Box::new(Expression::Constant(1.0)),
        Box::new(Expression::Variable("x".into())),
    );
    let s = e.simplify();
    assert_eq!(s, Expression::Variable("x".into()));
}

#[test]
fn test_simplify_mul_zero() {
    let e = Expression::Mul(
        Box::new(Expression::Constant(0.0)),
        Box::new(Expression::Variable("x".into())),
    );
    let s = e.simplify();
    assert_eq!(s, Expression::Constant(0.0));
}

#[test]
fn test_display() {
    let e = Expression::Add(
        Box::new(Expression::Variable("x".into())),
        Box::new(Expression::Constant(1.0)),
    );
    let s = format!("{}", e);
    assert!(s.contains("x"));
    assert!(s.contains("1"));
}

#[test]
fn test_variables() {
    let e = Expression::Add(
        Box::new(Expression::Variable("x".into())),
        Box::new(Expression::Variable("y".into())),
    );
    let mut vars = e.variables();
    vars.sort();
    assert_eq!(vars, vec!["x", "y"]);
}
