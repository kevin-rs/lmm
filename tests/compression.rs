use lmm::compression::{aic_score, bic_score, compute_mse, mdl_score, r_squared, select_best};
use lmm::equation::Expression;

fn linear_data(n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let inputs: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
    let targets: Vec<f64> = (0..n).map(|i| 2.0 * i as f64 + 1.0).collect();
    (inputs, targets)
}

#[test]
fn test_mse_perfect() {
    let expr = Expression::Add(
        Box::new(Expression::Mul(
            Box::new(Expression::Constant(2.0)),
            Box::new(Expression::Variable("x".into())),
        )),
        Box::new(Expression::Constant(1.0)),
    );
    let (inputs, targets) = linear_data(10);
    let mse = compute_mse(&expr, &inputs, &targets);
    assert!(
        mse < 1e-10,
        "Perfect model should have near-zero MSE, got {}",
        mse
    );
}

#[test]
fn test_mse_imperfect() {
    let expr = Expression::Constant(0.0);
    let (inputs, targets) = linear_data(10);
    let mse = compute_mse(&expr, &inputs, &targets);
    assert!(mse > 0.0, "Constant-0 model should have nonzero MSE");
}

#[test]
fn test_r_squared_perfect() {
    let expr = Expression::Add(
        Box::new(Expression::Mul(
            Box::new(Expression::Constant(2.0)),
            Box::new(Expression::Variable("x".into())),
        )),
        Box::new(Expression::Constant(1.0)),
    );
    let (inputs, targets) = linear_data(10);
    let r2 = r_squared(&expr, &inputs, &targets);
    assert!((r2 - 1.0).abs() < 1e-6, "Perfect model R²≈1, got {}", r2);
}

#[test]
fn test_r_squared_bad() {
    let expr = Expression::Constant(0.0);
    let (inputs, targets) = linear_data(10);
    let r2 = r_squared(&expr, &inputs, &targets);
    assert!(r2 < 0.5, "Constant-0 model should have low R², got {}", r2);
}

#[test]
fn test_mdl_simpler_wins() {
    let (inputs, targets) = linear_data(8);
    let simple = Expression::Mul(
        Box::new(Expression::Constant(2.0)),
        Box::new(Expression::Variable("x".into())),
    );
    let complex = Expression::Add(
        Box::new(Expression::Mul(
            Box::new(Expression::Sin(Box::new(Expression::Variable("x".into())))),
            Box::new(Expression::Cos(Box::new(Expression::Variable("x".into())))),
        )),
        Box::new(Expression::Constant(99.0)),
    );
    let simple_mdl = mdl_score(&simple, &inputs, &targets);
    let complex_mdl = mdl_score(&complex, &inputs, &targets);
    assert!(
        simple_mdl < complex_mdl,
        "Simpler model should win MDL: simple={:.2}, complex={:.2}",
        simple_mdl,
        complex_mdl
    );
}

#[test]
fn test_aic_lower_for_fewer_params() {
    let aic_simple = aic_score(1, -10.0);
    let aic_complex = aic_score(5, -10.0);
    assert!(aic_simple < aic_complex, "Fewer params → lower AIC");
}

#[test]
fn test_bic_lower_for_fewer_params() {
    let bic_simple = bic_score(1, 100, -10.0);
    let bic_complex = bic_score(5, 100, -10.0);
    assert!(bic_simple < bic_complex, "Fewer params → lower BIC");
}

#[test]
fn test_select_best() {
    let (inputs, targets) = linear_data(5);
    let c1 = Expression::Mul(
        Box::new(Expression::Constant(2.0)),
        Box::new(Expression::Variable("x".into())),
    );
    let c2 = Expression::Constant(0.0);
    let binding = [c1.clone(), c2];
    let best = select_best(&binding, &inputs, &targets);
    assert!(best.is_some());
    assert_eq!(
        best.unwrap().to_string(),
        c1.to_string(),
        "Linear model should be selected"
    );
}

#[test]
fn test_empty_data_returns_infinity() {
    let expr = Expression::Variable("x".into());
    let mse = compute_mse(&expr, &[], &[]);
    assert!(mse.is_infinite());
}
