use crate::equation::Expression;
use std::collections::HashMap;

pub fn mdl_score(expr: &Expression, inputs: &[Vec<f64>], targets: &[f64]) -> f64 {
    let mse = compute_mse(expr, inputs, targets);
    let complexity = expr.complexity() as f64;
    let n = targets.len() as f64;
    let data_cost = if mse > 0.0 { n * mse.ln() } else { 0.0 };
    let model_cost = complexity * 2.0_f64.ln();
    data_cost + model_cost
}

pub fn aic_score(n_params: usize, log_likelihood: f64) -> f64 {
    2.0 * n_params as f64 - 2.0 * log_likelihood
}

pub fn bic_score(n_params: usize, n_samples: usize, log_likelihood: f64) -> f64 {
    n_params as f64 * (n_samples as f64).ln() - 2.0 * log_likelihood
}

pub fn select_best<'a>(
    candidates: &'a [Expression],
    inputs: &[Vec<f64>],
    targets: &[f64],
) -> Option<&'a Expression> {
    candidates
        .iter()
        .map(|expr| (expr, mdl_score(expr, inputs, targets)))
        .filter(|(_, score)| score.is_finite())
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(expr, _)| expr)
}

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
