use crate::equation::Expression;
use crate::tensor::Tensor;
use std::collections::HashMap;

pub fn symbolic_diff(expr: &Expression, var: &str) -> Expression {
    expr.symbolic_diff(var)
}

pub fn simplify(expr: &Expression) -> Expression {
    expr.simplify()
}

pub fn complexity_score(expr: &Expression) -> usize {
    expr.complexity()
}

pub fn format_expr(expr: &Expression) -> String {
    expr.to_string()
}

pub fn numerical_gradient(
    expr: &Expression,
    var: &str,
    data: &[(&str, f64)],
    h: f64,
) -> Option<f64> {
    let mut bindings: HashMap<String, f64> =
        data.iter().map(|(k, v)| ((*k).to_string(), *v)).collect();
    let x = *bindings.get(var)?;
    bindings.insert(var.to_string(), x + h);
    let f_plus = expr.evaluate(&bindings).ok()?;
    bindings.insert(var.to_string(), x - h);
    let f_minus = expr.evaluate(&bindings).ok()?;
    Some((f_plus - f_minus) / (2.0 * h))
}

pub fn jacobian(exprs: &[Expression], vars: &[&str], point: &Tensor) -> Vec<Vec<f64>> {
    let bindings: HashMap<String, f64> = vars
        .iter()
        .zip(point.data.iter())
        .map(|(k, v)| ((*k).to_string(), *v))
        .collect();
    exprs
        .iter()
        .map(|expr| {
            vars.iter()
                .map(|var| {
                    let diff = expr.symbolic_diff(var).simplify();
                    diff.evaluate(&bindings).unwrap_or(0.0)
                })
                .collect()
        })
        .collect()
}

pub fn compose(outer: &Expression, inner: &Expression, var: &str) -> Expression {
    substitute(outer, var, inner)
}

fn substitute(expr: &Expression, var: &str, replacement: &Expression) -> Expression {
    match expr {
        Expression::Variable(name) if name == var => replacement.clone(),
        Expression::Variable(_) | Expression::Constant(_) => expr.clone(),
        Expression::Add(l, r) => Expression::Add(
            Box::new(substitute(l, var, replacement)),
            Box::new(substitute(r, var, replacement)),
        ),
        Expression::Sub(l, r) => Expression::Sub(
            Box::new(substitute(l, var, replacement)),
            Box::new(substitute(r, var, replacement)),
        ),
        Expression::Mul(l, r) => Expression::Mul(
            Box::new(substitute(l, var, replacement)),
            Box::new(substitute(r, var, replacement)),
        ),
        Expression::Div(l, r) => Expression::Div(
            Box::new(substitute(l, var, replacement)),
            Box::new(substitute(r, var, replacement)),
        ),
        Expression::Pow(b, e) => Expression::Pow(
            Box::new(substitute(b, var, replacement)),
            Box::new(substitute(e, var, replacement)),
        ),
        Expression::Neg(e) => Expression::Neg(Box::new(substitute(e, var, replacement))),
        Expression::Abs(e) => Expression::Abs(Box::new(substitute(e, var, replacement))),
        Expression::Sin(e) => Expression::Sin(Box::new(substitute(e, var, replacement))),
        Expression::Cos(e) => Expression::Cos(Box::new(substitute(e, var, replacement))),
        Expression::Exp(e) => Expression::Exp(Box::new(substitute(e, var, replacement))),
        Expression::Log(e) => Expression::Log(Box::new(substitute(e, var, replacement))),
    }
}
