// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Information-Theoretic Compression Metrics
//!
//! This module provides model-selection criteria and error metrics used throughout the
//! `lmm` equation-discovery pipeline. All functions are pure and stateless.
//!
//! The primary criterion is the **Minimum Description Length** (MDL) score, which
//! balances data fit (MSE) against model complexity to avoid over-fitting and drives the
//! symbolic regression search in [`crate::discovery::SymbolicRegression`].
//!
//! # See Also
//! - [Rissanen, J. (1978). Modeling by shortest data description.](https://doi.org/10.1016/0005-1098(78)90005-5) - introduces the core Minimum Description Length (MDL) theory.
//! - [Grünwald, P. (2007). The Minimum Description Length Principle. MIT Press.](https://mitpress.mit.edu/9780262072816/) - comprehensive textbook on the theoretical properties of MDL complexity scoring.

use crate::equation::Expression;
use std::collections::HashMap;

/// Computes the **Minimum Description Length (MDL)** score for `expr` on `(inputs, targets)`.
///
/// MDL = `n · ln(MSE) + complexity · ln(2)`.
///
/// A lower score indicates a better model: the penalty `complexity · ln(2)` discourages
/// unnecessarily large expression trees while the data-cost term `n · ln(MSE)` rewards fit.
///
/// # Arguments
///
/// * `expr` - The symbolic expression to evaluate.
/// * `inputs` - Input feature rows; each inner `Vec<f64>` is one sample.
/// * `targets` - Target scalar values aligned with `inputs`.
///
/// # Returns
///
/// (`f64`): The MDL score. Returns `f64::INFINITY` when MSE is zero (perfect fit, zero
/// cost) or when the expression produces non-finite values.
///
/// # Time Complexity
///
/// O(n · d) where n is the number of samples and d is the expression depth.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::compression::mdl_score;
/// use lmm::equation::Expression;
///
/// let expr = Expression::Variable("x".into());
/// let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
/// let targets = vec![1.0, 2.0, 3.0];
/// let score = mdl_score(&expr, &inputs, &targets);
/// assert!(score.is_finite());
/// ```
pub fn mdl_score(expr: &Expression, inputs: &[Vec<f64>], targets: &[f64]) -> f64 {
    let mse = compute_mse(expr, inputs, targets);
    let complexity = expr.complexity() as f64;
    let n = targets.len() as f64;
    let data_cost = if mse > f64::EPSILON {
        n * mse.ln()
    } else {
        0.0
    };
    let model_cost = complexity * 2.0_f64.ln();
    data_cost + model_cost
}

/// Computes the **Akaike Information Criterion (AIC)** from the log-likelihood.
///
/// `AIC = 2k - 2ℓ` where `k` is the number of free parameters and `ℓ` is the
/// maximised log-likelihood.
///
/// # Arguments
///
/// * `n_params` - Number of free parameters in the model.
/// * `log_likelihood` - Maximised log-likelihood value.
///
/// # Returns
///
/// (`f64`): AIC score. Lower is better.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::compression::aic_score;
/// let aic = aic_score(2, -50.0);
/// assert_eq!(aic, 2.0 * 2.0 - 2.0 * -50.0);
/// ```
///
/// # See Also
/// - [Akaike, H. (1974). A new look at the statistical model identification.](https://doi.org/10.1109/TAC.1974.1100705) - original paper establishing the Akaike Information Criterion.
pub fn aic_score(n_params: usize, log_likelihood: f64) -> f64 {
    2.0 * n_params as f64 - 2.0 * log_likelihood
}

/// Computes the **Bayesian Information Criterion (BIC)** from the log-likelihood.
///
/// `BIC = k · ln(n) - 2ℓ` where `k` is the number of free parameters, `n` is the number
/// of samples, and `ℓ` is the maximised log-likelihood.
///
/// # Arguments
///
/// * `n_params` - Number of free parameters.
/// * `n_samples` - Number of training samples.
/// * `log_likelihood` - Maximised log-likelihood value.
///
/// # Returns
///
/// (`f64`): BIC score. Lower is better.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::compression::bic_score;
/// let bic = bic_score(2, 100, -50.0);
/// assert!((bic - (2.0 * (100_f64).ln() - 2.0 * -50.0)).abs() < 1e-10);
/// ```
///
/// # See Also
/// - [Schwarz, G. (1978). Estimating the dimension of a model.](https://doi.org/10.1214/aos/1176344136) - derives the Bayesian Information Criterion (BIC) penalty.
pub fn bic_score(n_params: usize, n_samples: usize, log_likelihood: f64) -> f64 {
    n_params as f64 * (n_samples as f64).ln() - 2.0 * log_likelihood
}

/// Selects the expression with the lowest finite MDL score from `candidates`.
///
/// If all candidates produce `Infinity` (e.g. every expression is a constant with zero
/// MSE mismatch), returns the candidate with the smallest complexity as a fallback.
///
/// # Arguments
///
/// * `candidates` - Slice of candidate expressions.
/// * `inputs` - Input feature rows.
/// * `targets` - Target scalar values.
///
/// # Returns
///
/// (`Option<&Expression>`): The best-scoring expression, or `None` if `candidates` is empty.
///
/// # Time Complexity
///
/// O(n_candidates · n_samples).
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::compression::select_best;
/// use lmm::equation::Expression;
///
/// let c = vec![
///     Expression::Constant(0.0),
///     Expression::Variable("x".into()),
/// ];
/// let inputs = vec![vec![1.0], vec![2.0]];
/// let targets = vec![1.0, 2.0];
/// let best = select_best(&c, &inputs, &targets);
/// assert!(best.is_some());
/// ```
pub fn select_best<'a>(
    candidates: &'a [Expression],
    inputs: &[Vec<f64>],
    targets: &[f64],
) -> Option<&'a Expression> {
    if candidates.is_empty() {
        return None;
    }
    let scored: Vec<(&Expression, f64)> = candidates
        .iter()
        .map(|expr| (expr, mdl_score(expr, inputs, targets)))
        .collect();

    let finite_best = scored
        .iter()
        .filter(|(_, s)| s.is_finite())
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(e, _)| *e);

    finite_best.or_else(|| candidates.iter().min_by_key(|e| e.complexity()))
}

/// Computes the **Mean Squared Error (MSE)** of `expr` on `(inputs, targets)`.
///
/// Variables in `expr` are bound positionally to the columns of each input row.
///
/// # Arguments
///
/// * `expr` - The symbolic expression to evaluate.
/// * `inputs` - Input feature rows.
/// * `targets` - Aligned target values.
///
/// # Returns
///
/// (`f64`): MSE ∈ [0, ∞). Returns `f64::INFINITY` for empty slices.
///
/// # Time Complexity
///
/// O(n · d) where n is sample count and d is expression depth.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::compression::compute_mse;
/// use lmm::equation::Expression;
///
/// let expr = Expression::Variable("x".into());
/// let inputs = vec![vec![1.0], vec![2.0]];
/// let targets = vec![1.0, 2.0];
/// assert_eq!(compute_mse(&expr, &inputs, &targets), 0.0);
/// ```
pub fn compute_mse(expr: &Expression, inputs: &[Vec<f64>], targets: &[f64]) -> f64 {
    if inputs.is_empty() || targets.is_empty() {
        return f64::INFINITY;
    }
    let vars = expr.variables();
    let sum_sq: f64 = inputs
        .iter()
        .zip(targets.iter())
        .map(|(row, &target)| {
            let bindings: HashMap<String, f64> = vars
                .iter()
                .zip(row.iter())
                .map(|(k, &v)| (k.clone(), v))
                .collect();
            let pred = expr.evaluate(&bindings).unwrap_or(f64::NAN);
            if pred.is_nan() || pred.is_infinite() {
                1e9
            } else {
                (pred - target).powi(2)
            }
        })
        .sum();
    sum_sq / targets.len() as f64
}

/// Computes the **coefficient of determination (R²)** of `expr` on `(inputs, targets)`.
///
/// R² = 1 - SS_res / SS_tot. A value of 1.0 indicates a perfect fit.
/// Returns 0.0 for empty targets or constant target sequences (SS_tot = 0 → model assumed
/// trivially correct).
///
/// # Arguments
///
/// * `expr` - The symbolic expression.
/// * `inputs` - Input feature rows.
/// * `targets` - Aligned target values.
///
/// # Returns
///
/// (`f64`): R² ∈ (-∞, 1.0].
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::compression::r_squared;
/// use lmm::equation::Expression;
///
/// let expr = Expression::Variable("x".into());
/// let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
/// let targets = vec![1.0, 2.0, 3.0];
/// assert!((r_squared(&expr, &inputs, &targets) - 1.0).abs() < 1e-6);
/// ```
pub fn r_squared(expr: &Expression, inputs: &[Vec<f64>], targets: &[f64]) -> f64 {
    if targets.is_empty() {
        return 0.0;
    }
    let mean_target: f64 = targets.iter().sum::<f64>() / targets.len() as f64;
    let vars = expr.variables();
    let ss_res: f64 = inputs
        .iter()
        .zip(targets.iter())
        .map(|(row, &target)| {
            let bindings: HashMap<String, f64> = vars
                .iter()
                .zip(row.iter())
                .map(|(k, &v)| (k.clone(), v))
                .collect();
            let pred = expr.evaluate(&bindings).unwrap_or(f64::NAN);
            if pred.is_nan() {
                (target - mean_target).powi(2)
            } else {
                (pred - target).powi(2)
            }
        })
        .sum();
    let ss_tot: f64 = targets.iter().map(|&t| (t - mean_target).powi(2)).sum();
    if ss_tot == 0.0 {
        1.0
    } else {
        1.0 - ss_res / ss_tot
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
