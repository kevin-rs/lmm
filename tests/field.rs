use lmm::field::Field;
use lmm::tensor::Tensor;

fn linear_field(n: usize) -> Field {
    let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
    Field::new(vec![n], Tensor::new(vec![n], data).unwrap()).unwrap()
}

fn quadratic_field(n: usize) -> Field {
    let data: Vec<f64> = (0..n).map(|i| (i as f64).powi(2)).collect();
    Field::new(vec![n], Tensor::new(vec![n], data).unwrap()).unwrap()
}

#[test]
fn test_gradient_linear_is_constant() {
    let f = linear_field(7);
    let g = f.compute_gradient().unwrap();
    for &v in &g.values.data[1..g.values.data.len() - 1] {
        assert!((v - 1.0).abs() < 1e-10, "Expected gradient 1.0, got {}", v);
    }
}

#[test]
fn test_gradient_quadratic_is_linear() {
    let f = quadratic_field(10);
    let g = f.compute_gradient().unwrap();
    for i in 1..g.values.data.len() - 1 {
        let expected = 2.0 * i as f64;
        assert!(
            (g.values.data[i] - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            g.values.data[i]
        );
    }
}

#[test]
fn test_laplacian_linear_is_zero() {
    let f = linear_field(8);
    let lap = f.compute_laplacian().unwrap();
    for &v in &lap.values.data[1..lap.values.data.len() - 1] {
        assert!(
            v.abs() < 1e-10,
            "Laplacian of linear should be 0, got {}",
            v
        );
    }
}

#[test]
fn test_laplacian_quadratic_is_two() {
    let f = quadratic_field(10);
    let lap = f.compute_laplacian().unwrap();
    for &v in &lap.values.data[1..lap.values.data.len() - 1] {
        assert!(
            (v - 2.0).abs() < 1e-10,
            "Laplacian of x² should be 2, got {}",
            v
        );
    }
}

#[test]
fn test_gradient_2d() {
    let data: Vec<f64> = (0..9).map(|i| i as f64).collect();
    let f = Field::new(vec![3, 3], Tensor::new(vec![3, 3], data).unwrap()).unwrap();
    let g = f.compute_gradient().unwrap();
    assert_eq!(g.dimensions, vec![3, 3]);
    assert_eq!(g.values.data.len(), 9);
}

#[test]
fn test_gradient_3d() {
    let data: Vec<f64> = (0..27).map(|i| i as f64).collect();
    let f = Field::new(vec![3, 3, 3], Tensor::new(vec![3, 3, 3], data).unwrap()).unwrap();
    let g = f.compute_gradient().unwrap();
    assert_eq!(g.dimensions, vec![3, 3, 3]);
}

#[test]
fn test_divergence_1d() {
    let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let f = Field::new(vec![5], Tensor::new(vec![5], data).unwrap()).unwrap();
    let div = Field::compute_divergence(&[f]).unwrap();
    assert_eq!(div.dimensions, vec![5]);
}

#[test]
fn test_curl_3d() {
    let n = 3;
    let make = |val: f64| {
        let data = vec![val; 27];
        Field::new(vec![n, n, n], Tensor::new(vec![n, n, n], data).unwrap()).unwrap()
    };
    let fx = make(0.0);
    let fy = make(0.0);
    let fz = make(0.0);
    let [cx, cy, cz] = Field::compute_curl(&fx, &fy, &fz).unwrap();
    assert!(cx.values.data.iter().all(|&v| v == 0.0));
    assert!(cy.values.data.iter().all(|&v| v == 0.0));
    assert!(cz.values.data.iter().all(|&v| v == 0.0));
}

#[test]
fn test_field_dimension_mismatch() {
    let tensor = Tensor::new(vec![5], vec![0.0; 5]).unwrap();
    let result = Field::new(vec![3], tensor);
    assert!(result.is_err());
}
