// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Tensor - N-Dimensional Array
//!
//! This module provides [`Tensor`], the central n-dimensional array type used throughout
//! `lmm` for physics state vectors, field values, gradient computations, and perception data.
//!
//! [`Tensor`] is a dense, row-major storage backed by a `Vec<f64>` with explicit shape
//! metadata. All arithmetic operations are bounds-checked and return [`crate::error::Result`].
//!
//! # See Also
//! - [`crate::field::Field`] - wraps a `Tensor` with physical grid semantics for differential operations.
//! - [`crate::operator::NeuralOperator`] - spectral convolution operator that processes continuous `Field` data.

use crate::error::LmmError::Simulation;
use crate::error::Result;
use rand::RngExt;
use std::ops::{Add, Mul, Sub};

/// An n-dimensional, row-major dense array of `f64` values.
///
/// The `shape` field records the size along each dimension; `data` stores elements in
/// row-major (C) order. A scalar is represented as `shape = [1]`.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::tensor::Tensor;
///
/// let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// assert_eq!(t.shape, vec![2, 3]);
/// assert_eq!(t.data.len(), 6);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// Sizes of each dimension.
    pub shape: Vec<usize>,
    /// Element data in row-major order.
    pub data: Vec<f64>,
}

impl Tensor {
    /// Creates a new [`Tensor`] from explicit shape and data vectors.
    ///
    /// # Arguments
    ///
    /// * `shape` - Sizes of each dimension. The product must equal `data.len()`.
    /// * `data` - Element values in row-major order.
    ///
    /// # Returns
    ///
    /// (`Result<Tensor>`): The constructed tensor.
    ///
    /// # Errors
    ///
    /// Returns [`Simulation`] when `shape.iter().product() != data.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::tensor::Tensor;
    ///
    /// assert!(Tensor::new(vec![2], vec![1.0, 2.0]).is_ok());
    /// assert!(Tensor::new(vec![3], vec![1.0]).is_err());
    /// ```
    pub fn new(shape: Vec<usize>, data: Vec<f64>) -> Result<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(Simulation(format!(
                "Tensor shape mismatch: expected {} elements but got {}",
                expected_len,
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }

    /// Creates a tensor of all zeros with the given shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - Dimension sizes.
    ///
    /// # Returns
    ///
    /// (`Tensor`): Zero-filled tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::tensor::Tensor;
    /// let t = Tensor::zeros(vec![3]);
    /// assert_eq!(t.data, vec![0.0, 0.0, 0.0]);
    /// ```
    pub fn zeros(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; len],
        }
    }

    /// Creates a tensor of all ones with the given shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::tensor::Tensor;
    /// let t = Tensor::ones(vec![2]);
    /// assert_eq!(t.data, vec![1.0, 1.0]);
    /// ```
    pub fn ones(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        Self {
            shape,
            data: vec![1.0; len],
        }
    }

    /// Creates a tensor filled with a constant `value`.
    ///
    /// # Arguments
    ///
    /// * `shape` - Dimension sizes.
    /// * `value` - Fill value.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::tensor::Tensor;
    /// let t = Tensor::fill(vec![3], std::f64::consts::PI);
    /// assert!((t.data[0] - std::f64::consts::PI).abs() < 1e-15);
    /// ```
    pub fn fill(shape: Vec<usize>, value: f64) -> Self {
        let len: usize = shape.iter().product();
        Self {
            shape,
            data: vec![value; len],
        }
    }

    /// Creates a tensor of normally-distributed random values using the Box-Muller transform.
    ///
    /// # Arguments
    ///
    /// * `shape` - Dimension sizes.
    /// * `mean` - Mean of the normal distribution.
    /// * `std` - Standard deviation of the normal distribution.
    ///
    /// # Returns
    ///
    /// (`Tensor`): Randomly initialised tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::tensor::Tensor;
    /// let t = Tensor::randn(vec![100], 0.0, 1.0);
    /// assert_eq!(t.data.len(), 100);
    /// ```
    pub fn randn(shape: Vec<usize>, mean: f64, std: f64) -> Self {
        let len: usize = shape.iter().product();
        let mut rng = rand::rng();
        let data: Vec<f64> = (0..len)
            .map(|_| {
                let u1: f64 = rng.random::<f64>().max(1e-10);
                let u2: f64 = rng.random();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                mean + std * z
            })
            .collect();
        Self { shape, data }
    }

    /// Wraps a `Vec<f64>` as a rank-1 tensor with shape `[data.len()]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::tensor::Tensor;
    /// let t = Tensor::from_vec(vec![1.0, 2.0, 3.0]);
    /// assert_eq!(t.shape, vec![3]);
    /// ```
    pub fn from_vec(data: Vec<f64>) -> Self {
        let len = data.len();
        Self {
            shape: vec![len],
            data,
        }
    }

    /// Returns a new tensor with every element multiplied by `factor`.
    ///
    /// # Arguments
    ///
    /// * `factor` - Scalar multiplier.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::tensor::Tensor;
    /// let t = Tensor::from_vec(vec![2.0, 4.0]);
    /// let s = t.scale(0.5);
    /// assert_eq!(s.data, vec![1.0, 2.0]);
    /// ```
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&x| x * factor).collect(),
        }
    }

    /// Applies `f` element-wise, returning a new tensor of the same shape.
    ///
    /// # Arguments
    ///
    /// * `f` - A function `f64 → f64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::tensor::Tensor;
    /// let t = Tensor::from_vec(vec![0.0, 1.0, 4.0]);
    /// let s = t.map(f64::sqrt);
    /// assert!((s.data[2] - 2.0).abs() < 1e-10);
    /// ```
    pub fn map<F: Fn(f64) -> f64>(&self, f: F) -> Self {
        Self {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&x| f(x)).collect(),
        }
    }

    /// Applies a binary function element-wise to `self` and `other`, returning a new tensor.
    ///
    /// # Errors
    ///
    /// Returns [`Simulation`] when shapes differ.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![1.0, 2.0]);
    /// let b = Tensor::from_vec(vec![3.0, 4.0]);
    /// let c = a.zip_map(&b, |x, y| x * y).unwrap();
    /// assert_eq!(c.data, vec![3.0, 8.0]);
    /// ```
    pub fn zip_map<F: Fn(f64, f64) -> f64>(&self, other: &Self, f: F) -> Result<Self> {
        if self.shape != other.shape {
            return Err(Simulation("zip_map shape mismatch".into()));
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| f(a, b))
            .collect();
        Ok(Self {
            shape: self.shape.clone(),
            data,
        })
    }

    /// Computes the inner (dot) product with `other`.
    ///
    /// # Errors
    ///
    /// Returns [`Simulation`] when shapes differ.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0]);
    /// let b = Tensor::from_vec(vec![4.0, 5.0, 6.0]);
    /// assert_eq!(a.dot(&b).unwrap(), 32.0);
    /// ```
    pub fn dot(&self, other: &Self) -> Result<f64> {
        if self.shape != other.shape {
            return Err(Simulation("dot product shape mismatch".into()));
        }
        Ok(self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum())
    }

    /// Returns the Euclidean (L2) norm: `sqrt(sum(xᵢ²))`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::tensor::Tensor;
    /// let t = Tensor::from_vec(vec![3.0, 4.0]);
    /// assert!((t.norm() - 5.0).abs() < 1e-10);
    /// ```
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Returns the arithmetic mean of all elements.
    ///
    /// Returns `0.0` for an empty tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::tensor::Tensor;
    /// let t = Tensor::from_vec(vec![1.0, 2.0, 3.0]);
    /// assert_eq!(t.mean(), 2.0);
    /// ```
    pub fn mean(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().sum::<f64>() / self.data.len() as f64
    }

    /// Returns the population variance of all elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::tensor::Tensor;
    /// let t = Tensor::from_vec(vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
    /// assert!((t.variance() - 4.0).abs() < 1e-10);
    /// ```
    pub fn variance(&self) -> f64 {
        let m = self.mean();
        self.data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / self.data.len() as f64
    }

    /// Returns the index of the element with the largest value.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::tensor::Tensor;
    /// let t = Tensor::from_vec(vec![1.0, 5.0, 3.0]);
    /// assert_eq!(t.argmax(), 1);
    /// ```
    pub fn argmax(&self) -> usize {
        self.data.iter().enumerate().fold(
            0,
            |best, (i, &v)| {
                if v > self.data[best] { i } else { best }
            },
        )
    }

    /// Reshapes the tensor to `new_shape`, reusing the same data buffer.
    ///
    /// # Errors
    ///
    /// Returns [`Simulation`] when `new_shape.iter().product() != self.data.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::tensor::Tensor;
    /// let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    /// let r = t.reshape(vec![2, 2]).unwrap();
    /// assert_eq!(r.shape, vec![2, 2]);
    /// ```
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self> {
        let expected: usize = new_shape.iter().product();
        if expected != self.data.len() {
            return Err(Simulation(format!(
                "Cannot reshape {} elements into shape {:?}",
                self.data.len(),
                new_shape
            )));
        }
        Ok(Self {
            shape: new_shape,
            data: self.data.clone(),
        })
    }

    /// Transposes a 2-D tensor, swapping rows and columns.
    ///
    /// # Errors
    ///
    /// Returns [`Simulation`] for tensors with rank ≠ 2.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::tensor::Tensor;
    /// let t = Tensor::new(vec![2, 3], vec![1.0,2.0,3.0,4.0,5.0,6.0]).unwrap();
    /// let tr = t.transpose().unwrap();
    /// assert_eq!(tr.shape, vec![3, 2]);
    /// assert_eq!(tr.data[0], 1.0);
    /// assert_eq!(tr.data[1], 4.0);
    /// ```
    pub fn transpose(&self) -> Result<Self> {
        if self.shape.len() != 2 {
            return Err(Simulation("transpose requires a 2-D tensor".into()));
        }
        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut data = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                data[c * rows + r] = self.data[r * cols + c];
            }
        }
        Ok(Self {
            shape: vec![cols, rows],
            data,
        })
    }

    /// Performs matrix multiplication of two 2-D tensors: `self [m×k] × other [k×n] → [m×n]`.
    ///
    /// # Errors
    ///
    /// Returns [`Simulation`] when either tensor is not 2-D or inner dimensions mismatch.
    ///
    /// # Time Complexity
    ///
    /// O(m · k · n) - naive triple-loop implementation.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::tensor::Tensor;
    /// let a = Tensor::new(vec![2, 2], vec![1.0,2.0,3.0,4.0]).unwrap();
    /// let b = Tensor::new(vec![2, 2], vec![5.0,6.0,7.0,8.0]).unwrap();
    /// let c = a.matmul(&b).unwrap();
    /// assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    /// ```
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(Simulation("matmul requires 2-D tensors".into()));
        }
        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        if k != k2 {
            return Err(Simulation(format!(
                "matmul dimension mismatch: [{m}x{k}] vs [{k2}x{n}]"
            )));
        }
        let mut data = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += self.data[i * k + l] * other.data[l * n + j];
                }
                data[i * n + j] = sum;
            }
        }
        Ok(Self {
            shape: vec![m, n],
            data,
        })
    }

    /// Returns the total number of elements (product of all dimensions).
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::tensor::Tensor;
    /// assert_eq!(Tensor::zeros(vec![3, 4]).len(), 12);
    /// ```
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` when the tensor contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::tensor::Tensor;
    /// let t = Tensor::new(vec![0], vec![]).unwrap();
    /// assert!(t.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Add for &Tensor {
    type Output = Result<Tensor>;

    /// Adds two tensors element-wise.
    ///
    /// # Errors
    ///
    /// Returns [`Simulation`] when shapes differ.
    fn add(self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(Simulation(format!(
                "Tensor add shape mismatch: {:?} vs {:?}",
                self.shape, rhs.shape
            )));
        }
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Ok(Tensor {
            shape: self.shape.clone(),
            data,
        })
    }
}

impl Sub for &Tensor {
    type Output = Result<Tensor>;

    /// Subtracts `rhs` from `self` element-wise.
    ///
    /// # Errors
    ///
    /// Returns [`Simulation`] when shapes differ.
    fn sub(self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(Simulation(format!(
                "Tensor sub shape mismatch: {:?} vs {:?}",
                self.shape, rhs.shape
            )));
        }
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Ok(Tensor {
            shape: self.shape.clone(),
            data,
        })
    }
}

impl Mul for &Tensor {
    type Output = Result<Tensor>;

    /// Multiplies two tensors element-wise (Hadamard product).
    ///
    /// # Errors
    ///
    /// Returns [`Simulation`] when shapes differ.
    fn mul(self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(Simulation(format!(
                "Tensor mul shape mismatch: {:?} vs {:?}",
                self.shape, rhs.shape
            )));
        }
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Ok(Tensor {
            shape: self.shape.clone(),
            data,
        })
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
