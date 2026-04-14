use lmm::tensor::Tensor;

#[test]
fn test_zeros() {
    let t = Tensor::zeros(vec![3]);
    assert_eq!(t.data, vec![0.0, 0.0, 0.0]);
    assert_eq!(t.shape, vec![3]);
}

#[test]
fn test_ones() {
    let t = Tensor::ones(vec![2, 2]);
    assert_eq!(t.data.len(), 4);
    assert!(t.data.iter().all(|&v| v == 1.0));
}

#[test]
fn test_fill() {
    let t = Tensor::fill(vec![4], std::f64::consts::PI);
    assert!(
        t.data
            .iter()
            .all(|&v| (v - std::f64::consts::PI).abs() < 1e-10)
    );
}

#[test]
fn test_shape_mismatch_error() {
    let result = Tensor::new(vec![3], vec![1.0, 2.0]);
    assert!(result.is_err());
}

#[test]
fn test_add() {
    let a = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::new(vec![3], vec![4.0, 5.0, 6.0]).unwrap();
    let c = (&a + &b).unwrap();
    assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_sub() {
    let a = Tensor::new(vec![2], vec![5.0, 3.0]).unwrap();
    let b = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
    let c = (&a - &b).unwrap();
    assert_eq!(c.data, vec![4.0, 1.0]);
}

#[test]
fn test_mul() {
    let a = Tensor::new(vec![3], vec![2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::new(vec![3], vec![5.0, 6.0, 7.0]).unwrap();
    let c = (&a * &b).unwrap();
    assert_eq!(c.data, vec![10.0, 18.0, 28.0]);
}

#[test]
fn test_scale() {
    let t = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let s = t.scale(2.0);
    assert_eq!(s.data, vec![2.0, 4.0, 6.0]);
}

#[test]
fn test_dot() {
    let a = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::new(vec![3], vec![4.0, 5.0, 6.0]).unwrap();
    let d = a.dot(&b).unwrap();
    assert!((d - 32.0).abs() < 1e-10);
}

#[test]
fn test_norm() {
    let a = Tensor::new(vec![3], vec![3.0, 4.0, 0.0]).unwrap();
    assert!((a.norm() - 5.0).abs() < 1e-10);
}

#[test]
fn test_reshape() {
    let a = Tensor::new(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let r = a.reshape(vec![2, 3]).unwrap();
    assert_eq!(r.shape, vec![2, 3]);
    assert_eq!(r.data.len(), 6);
}

#[test]
fn test_reshape_mismatch() {
    let a = Tensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(a.reshape(vec![3]).is_err());
}

#[test]
fn test_matmul() {
    let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Tensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape, vec![2, 2]);
    assert!((c.data[0] - 58.0).abs() < 1e-10);
    assert!((c.data[1] - 64.0).abs() < 1e-10);
    assert!((c.data[2] - 139.0).abs() < 1e-10);
    assert!((c.data[3] - 154.0).abs() < 1e-10);
}

#[test]
fn test_transpose() {
    let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let t = a.transpose().unwrap();
    assert_eq!(t.shape, vec![3, 2]);
    assert_eq!(t.data[0], 1.0);
    assert_eq!(t.data[1], 4.0);
}

#[test]
fn test_mean_variance() {
    let a = Tensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!((a.mean() - 2.5).abs() < 1e-10);
    assert!((a.variance() - 1.25).abs() < 1e-10);
}

#[test]
fn test_argmax() {
    let a = Tensor::new(vec![5], vec![1.0, 5.0, 3.0, 2.0, 4.0]).unwrap();
    assert_eq!(a.argmax(), 1);
}

#[test]
fn test_map() {
    let a = Tensor::new(vec![3], vec![1.0, 4.0, 9.0]).unwrap();
    let b = a.map(|x| x.sqrt());
    assert!((b.data[0] - 1.0).abs() < 1e-10);
    assert!((b.data[1] - 2.0).abs() < 1e-10);
    assert!((b.data[2] - 3.0).abs() < 1e-10);
}

#[test]
fn test_randn_shape() {
    let t = Tensor::randn(vec![100], 0.0, 1.0);
    assert_eq!(t.shape, vec![100]);
    assert_eq!(t.data.len(), 100);
}

#[test]
fn test_add_shape_mismatch() {
    let a = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
    let b = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    assert!((&a + &b).is_err());
}
