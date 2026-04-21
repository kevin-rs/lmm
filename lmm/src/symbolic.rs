// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Symbolic Expression Tools
//!
//! This module provides high-level helpers that operate on [`Expression`] trees:
//!
//! - Symbolic differentiation via [`symbolic_diff`].
//! - Algebraic simplification via [`simplify`].
//! - Complexity scoring via [`complexity_score`].
//! - Numerical gradient estimation via [`numerical_gradient`] (central differences).
//! - Jacobian matrix computation via [`jacobian`].
//! - Functional composition via [`compose`].
//! - Substitution of sub-expressions via the internal `substitute` helper.
//!
//! All functions are pure and do not mutate the input [`Expression`].

use crate::equation::Expression;
use crate::tensor::Tensor;
use std::collections::HashMap;

/// Returns the symbolic partial derivative of `expr` with respect to `var`.
///
/// Differentiation rules are applied recursively over the expression tree.
/// See [`Expression::symbolic_diff`] for the full rule set.
///
/// # Arguments
///
/// * `expr` - The expression to differentiate.
/// * `var` - The name of the variable to differentiate with respect to.
///
/// # Returns
///
/// (`Expression`): The (unsimplified) derivative expression.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::symbolic::symbolic_diff;
/// use lmm::equation::Expression;
///
/// // d/dx (x²) = 2x
/// let x_sq = Expression::Pow(
///     Box::new(Expression::Variable("x".into())),
///     Box::new(Expression::Constant(2.0)),
/// );
/// let deriv = symbolic_diff(&x_sq, "x").simplify();
/// let mut env = std::collections::HashMap::new();
/// env.insert("x".to_string(), 3.0);
/// assert!((deriv.evaluate(&env).unwrap() - 6.0).abs() < 1e-9);
/// ```
pub fn symbolic_diff(expr: &Expression, var: &str) -> Expression {
    expr.symbolic_diff(var)
}

/// Algebraically simplifies `expr` using constant folding and identity rules.
///
/// # Arguments
///
/// * `expr` - The expression to simplify.
///
/// # Returns
///
/// (`Expression`): A semantically equivalent but structurally simpler expression.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::symbolic::simplify;
/// use lmm::equation::Expression;
///
/// let zero = Expression::Mul(
///     Box::new(Expression::Constant(0.0)),
///     Box::new(Expression::Variable("x".into())),
/// );
/// let s = simplify(&zero);
/// assert!(matches!(s, Expression::Constant(c) if c == 0.0));
/// ```
pub fn simplify(expr: &Expression) -> Expression {
    expr.simplify()
}

/// Returns the number of nodes in the expression tree (a proxy for model complexity).
///
/// Used by [`crate::compression::mdl_score`] to penalise large expressions.
///
/// # Arguments
///
/// * `expr` - The expression to measure.
///
/// # Returns
///
/// (`usize`): Number of nodes in the expression tree.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::symbolic::complexity_score;
/// use lmm::equation::Expression;
///
/// let e = Expression::Add(
///     Box::new(Expression::Variable("x".into())),
///     Box::new(Expression::Constant(1.0)),
/// );
/// assert_eq!(complexity_score(&e), 3);
/// ```
pub fn complexity_score(expr: &Expression) -> usize {
    expr.complexity()
}

/// Formats `expr` as a human-readable infix string.
///
/// # Arguments
///
/// * `expr` - The expression to format.
///
/// # Returns
///
/// (`String`): Infix text representation.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::symbolic::format_expr;
/// use lmm::equation::Expression;
///
/// let e = Expression::Add(
///     Box::new(Expression::Variable("x".into())),
///     Box::new(Expression::Constant(2.0)),
/// );
/// assert_eq!(format_expr(&e), "(x + 2)");
/// ```
pub fn format_expr(expr: &Expression) -> String {
    expr.to_string()
}

/// Estimates the partial derivative of `expr` with respect to `var` at `data` using
/// **central finite differences**: `(f(x+h) - f(x-h)) / (2h)`.
///
/// # Arguments
///
/// * `expr` - The expression to differentiate numerically.
/// * `var` - The variable to perturb.
/// * `data` - Bindings for all variables as `(name, value)` pairs.
/// * `h` - Finite-difference step size (typically `1e-5`).
///
/// # Returns
///
/// (`Option<f64>`): Numerical derivative, or `None` when `var` is not in `data` or
/// the expression fails to evaluate.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::symbolic::numerical_gradient;
/// use lmm::equation::Expression;
///
/// // f(x) = x²; f'(x) at x=3 ≈ 6
/// let expr = Expression::Pow(
///     Box::new(Expression::Variable("x".into())),
///     Box::new(Expression::Constant(2.0)),
/// );
/// let grad = numerical_gradient(&expr, "x", &[("x", 3.0)], 1e-5).unwrap();
/// assert!((grad - 6.0).abs() < 1e-4);
/// ```
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

/// Computes the Jacobian matrix of `exprs` with respect to `vars` at `point`.
///
/// Returns a matrix `J[i][j] = ∂exprs[i]/∂vars[j]` evaluated analytically via symbolic
/// differentiation.
///
/// # Arguments
///
/// * `exprs` - The output expressions (one per row of J).
/// * `vars` - The input variable names (one per column of J).
/// * `point` - The point at which to evaluate; components mapped positionally to `vars`.
///
/// # Returns
///
/// (`Vec<Vec<f64>>`): Jacobian matrix of shape `[exprs.len() × vars.len()]`.
///
/// # Time Complexity
///
/// O(|exprs| · |vars| · depth) for symbolic differentiation plus evaluation.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::symbolic::jacobian;
/// use lmm::equation::Expression;
/// use lmm::tensor::Tensor;
///
/// // f(x, y) = [x + y, x*y]; Jacobian at (1, 2) = [[1,1],[2,1]]
/// let fx = Expression::Add(
///     Box::new(Expression::Variable("x".into())),
///     Box::new(Expression::Variable("y".into())),
/// );
/// let fy = Expression::Mul(
///     Box::new(Expression::Variable("x".into())),
///     Box::new(Expression::Variable("y".into())),
/// );
/// let j = jacobian(&[fx, fy], &["x", "y"], &Tensor::from_vec(vec![1.0, 2.0]));
/// assert!((j[0][0] - 1.0).abs() < 1e-9); // ∂(x+y)/∂x
/// assert!((j[1][0] - 2.0).abs() < 1e-9); // ∂(x*y)/∂x at (1,2) = y = 2
/// ```
/// use lmm::traits::Simulatable;
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

/// Composes two expressions: `compose(outer, inner, var)` returns `outer[var ↦ inner]`.
///
/// # Arguments
///
/// * `outer` - The outer function expression.
/// * `inner` - The inner function expression substituted for `var`.
/// * `var` - The variable in `outer` to replace.
///
/// # Returns
///
/// (`Expression`): The composed expression.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::symbolic::compose;
/// use lmm::equation::Expression;
///
/// // outer = x², inner = (x + 1); compose(outer, inner, "x") = (x+1)²
/// let outer = Expression::Pow(
///     Box::new(Expression::Variable("x".into())),
///     Box::new(Expression::Constant(2.0)),
/// );
/// let inner = Expression::Add(
///     Box::new(Expression::Variable("x".into())),
///     Box::new(Expression::Constant(1.0)),
/// );
/// let composed = compose(&outer, &inner, "x");
/// let mut env = std::collections::HashMap::new();
/// env.insert("x".to_string(), 2.0);
/// // (2+1)² = 9
/// assert!((composed.evaluate(&env).unwrap() - 9.0).abs() < 1e-9);
/// ```
pub fn compose(outer: &Expression, inner: &Expression, var: &str) -> Expression {
    substitute(outer, var, inner)
}

/// Substitutes every occurrence of variable `var` in `expr` with `replacement`.
///
/// This is a structural tree rewrite - it does not evaluate or simplify the result.
///
/// # Arguments
///
/// * `expr` - Source expression tree.
/// * `var` - Variable name to replace.
/// * `replacement` - Expression to insert in place of `var`.
///
/// # Returns
///
/// (`Expression`): The rewritten expression.
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

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
