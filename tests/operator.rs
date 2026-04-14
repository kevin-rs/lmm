use lmm::field::Field;
use lmm::operator::{FourierOperator, NeuralOperator};
use lmm::tensor::Tensor;
use lmm::traits::Learnable;

fn make_field(data: Vec<f64>) -> Field {
    let n = data.len();
    Field::new(vec![n], Tensor::new(vec![n], data).unwrap()).unwrap()
}

#[test]
fn test_identity_kernel() {
    let mut weights = vec![0.0; 5];
    weights[2] = 1.0;
    let op = NeuralOperator {
        kernel_weights: weights,
    };
    let input = make_field(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let out = op.transform(&input).unwrap();
    for (a, b) in out.values.data.iter().zip(input.values.data.iter()) {
        assert!(
            (a - b).abs() < 1e-10,
            "Identity kernel mismatch: {} vs {}",
            a,
            b
        );
    }
}

#[test]
fn test_zero_kernel_gives_zero() {
    let op = NeuralOperator {
        kernel_weights: vec![0.0, 0.0, 0.0],
    };
    let input = make_field(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let out = op.transform(&input).unwrap();
    assert!(out.values.data.iter().all(|&v| v == 0.0));
}

#[test]
fn test_scale_kernel() {
    let op = NeuralOperator {
        kernel_weights: vec![0.5],
    };
    let input = make_field(vec![2.0, 4.0, 6.0]);
    let out = op.transform(&input).unwrap();
    assert_eq!(out.values.data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_learnable_update() {
    let mut op = NeuralOperator {
        kernel_weights: vec![1.0, 0.0, 1.0],
    };
    let grad = Tensor::new(vec![3], vec![0.1, 0.2, 0.3]).unwrap();
    op.update(&grad, 1.0).unwrap();
    assert!((op.kernel_weights[0] - 0.9).abs() < 1e-10);
    assert!((op.kernel_weights[1] - (-0.2)).abs() < 1e-10);
    assert!((op.kernel_weights[2] - 0.7).abs() < 1e-10);
}

#[test]
fn test_learnable_wrong_size_error() {
    let mut op = NeuralOperator {
        kernel_weights: vec![1.0, 0.0],
    };
    let grad = Tensor::new(vec![3], vec![0.1, 0.2, 0.3]).unwrap();
    assert!(op.update(&grad, 1.0).is_err());
}

#[test]
fn test_gradient_descent_reduces_error() {
    let target = make_field(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let input = make_field(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    let mut op = NeuralOperator {
        kernel_weights: vec![1.0],
    };
    let initial_out = op.transform(&input).unwrap();
    let initial_err: f64 = initial_out
        .values
        .data
        .iter()
        .zip(target.values.data.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    for _ in 0..5 {
        let grads = op.gradient_wrt_kernel(&input, &target).unwrap();
        let grad_tensor = Tensor::new(vec![grads.len()], grads).unwrap();
        op.update(&grad_tensor, 0.01).unwrap();
    }
    let final_out = op.transform(&input).unwrap();
    let final_err: f64 = final_out
        .values
        .data
        .iter()
        .zip(target.values.data.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    assert!(
        final_err < initial_err,
        "Error should decrease after gradient steps: {final_err:.4} vs {initial_err:.4}"
    );
}

#[test]
fn test_fourier_operator_round_trip() {
    let data = vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
    let n_modes = data.len();
    let op = FourierOperator {
        spectral_weights: vec![1.0; n_modes],
    };
    let input = make_field(data.clone());
    let out = op.transform(&input).unwrap();
    assert_eq!(out.values.data.len(), data.len());
}

#[test]
fn test_empty_kernel_error() {
    let op = NeuralOperator {
        kernel_weights: vec![],
    };
    let input = make_field(vec![1.0, 2.0, 3.0]);
    assert!(op.transform(&input).is_err());
}
