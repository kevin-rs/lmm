use lmm::compression::compute_mse;
use lmm::discovery::SymbolicRegression;
use lmm::tensor::Tensor;
use lmm::traits::Discoverable;

fn synthetic_linear(n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let inputs: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
    let targets: Vec<f64> = (0..n).map(|i| 2.0 * i as f64).collect();
    (inputs, targets)
}

#[test]
fn test_symbolic_regression_improves_over_constant() {
    let (inputs, targets) = synthetic_linear(15);
    let sr = SymbolicRegression::new(3, 30).with_variables(vec!["x".into()]);
    let expr = sr.fit(&inputs, &targets).unwrap();
    let mse = compute_mse(&expr, &inputs, &targets);
    let baseline_mse = {
        let mean_y = targets.iter().sum::<f64>() / targets.len() as f64;
        targets.iter().map(|&y| (y - mean_y).powi(2)).sum::<f64>() / targets.len() as f64
    };
    assert!(
        mse < baseline_mse,
        "GP should beat constant-mean baseline: mse={:.2}, baseline={:.2}",
        mse,
        baseline_mse
    );
}

#[test]
fn test_symbolic_regression_discovers_constant() {
    let inputs: Vec<Vec<f64>> = (0..10).map(|_| vec![0.0]).collect();
    let targets = vec![5.0; 10];
    let sr = SymbolicRegression::new(2, 20).with_variables(vec!["x".into()]);
    let expr = sr.fit(&inputs, &targets).unwrap();
    let mse = compute_mse(&expr, &inputs, &targets);
    assert!(
        mse < 1.0,
        "Should discover constant function, mse={:.4}",
        mse
    );
}

#[test]
fn test_symbolic_regression_returns_expression() {
    let (inputs, targets) = synthetic_linear(8);
    let sr = SymbolicRegression::new(3, 10);
    let expr = sr.fit(&inputs, &targets);
    assert!(expr.is_ok(), "fit() should succeed");
}

#[test]
fn test_discover_trait() {
    let data: Vec<Tensor> = (0..8).map(|i| Tensor::from_vec(vec![i as f64])).collect();
    let targets: Vec<f64> = (0..8).map(|i| i as f64 * 2.0).collect();
    let result = SymbolicRegression::discover(&data, &targets);
    assert!(result.is_ok());
}

#[test]
fn test_empty_data_error() {
    let sr = SymbolicRegression::new(3, 10);
    let result = sr.fit(&[], &[]);
    assert!(result.is_err());
}

#[test]
fn test_population_size_setting() {
    let sr = SymbolicRegression::new(3, 5).with_population(20);
    assert_eq!(sr.population_size, 20);
}

#[test]
fn test_variable_names_setting() {
    let sr = SymbolicRegression::new(3, 5).with_variables(vec!["t".into(), "v".into()]);
    assert_eq!(sr.variable_names, vec!["t", "v"]);
}
