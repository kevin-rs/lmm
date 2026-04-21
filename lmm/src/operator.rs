// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Neural and Fourier Operators
//!
//! This module provides two operator types that transform [`Field`]s:
//!
//! - [`NeuralOperator`] - a learnable cyclic convolution kernel applied to 1-D fields.
//!   Supports gradient computation with respect to the kernel and gradient-descent weight
//!   updates via the [`Learnable`] trait.
//!
//! - [`FourierOperator`] - a spectral filtering operator that decomposes a 1-D field into
//!   its Fourier modes, applies per-mode weights, and reconstructs via the inverse DFT.
//!   Normalization follows the convention `forward * (1/n)`, `inverse * (1/n)` to ensure
//!   the round-trip is identity when all spectral weights equal 1.
//!
//! # See Also
//! - [Li, Z. et al. (2021). Fourier Neural Operator for Parametric PDEs.](https://arxiv.org/abs/2010.08895) - foundational paper for the spectral convolution architecture implemented in `NeuralOperator`.

use crate::error::{LmmError, Result};
use crate::field::Field;
use crate::tensor::Tensor;
use crate::traits::Learnable;

/// A cyclic-convolution neural operator with a learnable kernel.
///
/// The kernel is applied to the input [`Field`] using wraparound (cyclic) boundary
/// conditions. A delta kernel (`kernel[half] = 1.0`, all others zero) is the identity.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::operator::NeuralOperator;
/// use lmm::field::Field;
/// use lmm::tensor::Tensor;
///
/// let t = Tensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let f = Field::new(vec![4], t).unwrap();
/// let op = NeuralOperator::new(3); // identity delta kernel
/// let out = op.transform(&f).unwrap();
/// assert!((out.values.data[0] - 1.0).abs() < 1e-9);
/// ```
pub struct NeuralOperator {
    /// The convolution kernel weights.
    pub kernel_weights: Vec<f64>,
}

impl NeuralOperator {
    /// Creates a delta (identity) kernel of size `kernel_size`.
    ///
    /// The centre element is set to `1.0`; all others to `0.0`.
    ///
    /// # Arguments
    ///
    /// * `kernel_size` - Number of kernel taps.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::operator::NeuralOperator;
    /// let op = NeuralOperator::new(5);
    /// assert_eq!(op.kernel_weights[2], 1.0);
    /// ```
    pub fn new(kernel_size: usize) -> Self {
        let weights = (0..kernel_size)
            .map(|i| if i == kernel_size / 2 { 1.0 } else { 0.0 })
            .collect();
        Self {
            kernel_weights: weights,
        }
    }

    /// Applies the kernel to `input` using cyclic convolution.
    ///
    /// # Arguments
    ///
    /// * `input` - The source [`Field`].
    ///
    /// # Returns
    ///
    /// (`Result<Field>`): The convolved output field.
    ///
    /// # Errors
    ///
    /// Returns [`LmmError::Operator`] when the kernel is empty.
    ///
    /// # Time Complexity
    ///
    /// O(n · k) where n is the field length and k is the kernel size.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::operator::NeuralOperator;
    /// use lmm::field::Field;
    /// use lmm::tensor::Tensor;
    ///
    /// let t = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    /// let f = Field::new(vec![3], t).unwrap();
    /// let op = NeuralOperator::new(1);
    /// let out = op.transform(&f).unwrap();
    /// assert_eq!(out.values.data, f.values.data);
    /// ```
    pub fn transform(&self, input: &Field) -> Result<Field> {
        if self.kernel_weights.is_empty() {
            return Err(LmmError::Operator("Empty kernel weights".into()));
        }
        let n = input.values.data.len();
        let k = self.kernel_weights.len();
        let half = k / 2;
        let mut out = vec![0.0; n];
        for (i, slot) in out.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (j, &w) in self.kernel_weights.iter().enumerate() {
                let src = (i + j + n - half) % n;
                acc += input.values.data[src] * w;
            }
            *slot = acc;
        }
        let tensor = Tensor::new(input.dimensions.clone(), out)?;
        Field::new(input.dimensions.clone(), tensor)
    }

    /// Computes the gradient of the MSE loss with respect to the kernel weights.
    ///
    /// `grad[j] = (2/n) Σᵢ (output[i] - target[i]) · input[(i+j+n-half) % n]`
    ///
    /// # Arguments
    ///
    /// * `input` - The input field.
    /// * `target` - The target field.
    ///
    /// # Returns
    ///
    /// (`Result<Vec<f64>>`): Gradient of MSE w.r.t. each kernel weight.
    ///
    /// # Errors
    ///
    /// Returns [`LmmError::Operator`] when input and target dimensions differ.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::operator::NeuralOperator;
    /// use lmm::field::Field;
    /// use lmm::tensor::Tensor;
    ///
    /// let t = Tensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// let f = Field::new(vec![4], t).unwrap();
    /// let op = NeuralOperator::new(3);
    /// let grad = op.gradient_wrt_kernel(&f, &f).unwrap();
    /// assert_eq!(grad.len(), 3);
    /// ```
    pub fn gradient_wrt_kernel(&self, input: &Field, target: &Field) -> Result<Vec<f64>> {
        if input.dimensions != target.dimensions {
            return Err(LmmError::Operator(
                "Input/target dimension mismatch for gradient".into(),
            ));
        }
        let output = self.transform(input)?;
        let n = input.values.data.len();
        let k = self.kernel_weights.len();
        let half = k / 2;
        let mut grads = vec![0.0; k];
        for i in 0..n {
            let residual = output.values.data[i] - target.values.data[i];
            for (j, slot) in grads.iter_mut().enumerate() {
                let src = (i + j + n - half) % n;
                *slot += 2.0 * residual * input.values.data[src] / n as f64;
            }
        }
        Ok(grads)
    }
}

impl Learnable for NeuralOperator {
    /// Updates kernel weights via gradient descent: `w -= lr * g`.
    ///
    /// # Errors
    ///
    /// Returns [`LmmError::Operator`] when `grad.data.len() != kernel_weights.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::operator::NeuralOperator;
    /// use lmm::tensor::Tensor;
    /// use lmm::traits::Learnable;
    ///
    /// let mut op = NeuralOperator::new(3);
    /// let grad = Tensor::from_vec(vec![0.1, 0.2, 0.3]);
    /// op.update(&grad, 1.0).unwrap();
    /// assert!((op.kernel_weights[1] - 0.8).abs() < 1e-9);
    /// ```
    fn update(&mut self, grad: &Tensor, lr: f64) -> Result<()> {
        if grad.data.len() != self.kernel_weights.len() {
            return Err(LmmError::Operator(format!(
                "Gradient length {} != kernel length {}",
                grad.data.len(),
                self.kernel_weights.len()
            )));
        }
        for (w, &g) in self.kernel_weights.iter_mut().zip(grad.data.iter()) {
            *w -= lr * g;
        }
        Ok(())
    }
}

/// A spectral filtering operator based on the Discrete Fourier Transform.
///
/// `FourierOperator` applies per-mode gain weights in the Fourier domain. The pipeline is:
/// 1. **Forward DFT**: project the input into frequency space, normalised by `1/n`.
/// 2. **Spectral weighting**: multiply each complex mode by its spectral weight.
/// 3. **Inverse DFT**: reconstruct the output from the weighted modes.
///
/// With all weights equal to `1.0`, the round-trip `transform(f)` is approximately `f`.
///
/// # See Also
/// - [Li, Z. et al. (2021). Fourier Neural Operator.](https://arxiv.org/abs/2010.08895) - the algorithm underlying this spectral transformation.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::operator::FourierOperator;
/// use lmm::field::Field;
/// use lmm::tensor::Tensor;
///
/// let t = Tensor::new(vec![4], vec![1.0, 0.0, -1.0, 0.0]).unwrap();
/// let f = Field::new(vec![4], t.clone()).unwrap();
/// let op = FourierOperator::new(3);
/// let out = op.transform(&f).unwrap();
/// assert_eq!(out.values.data.len(), 4);
/// ```
pub struct FourierOperator {
    /// Per-mode spectral gain weights applied in the frequency domain.
    pub spectral_weights: Vec<f64>,
}

impl FourierOperator {
    /// Creates a [`FourierOperator`] with `n_modes` spectral weights all initialised to `1.0`.
    ///
    /// # Arguments
    ///
    /// * `n_modes` - Number of Fourier modes to retain.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::operator::FourierOperator;
    /// let op = FourierOperator::new(4);
    /// assert_eq!(op.spectral_weights.len(), 4);
    /// ```
    pub fn new(n_modes: usize) -> Self {
        Self {
            spectral_weights: vec![1.0; n_modes],
        }
    }

    /// Applies spectral filtering to a 1-D [`Field`].
    ///
    /// The normalisation convention is `1/n` applied in the **forward** pass only,
    /// ensuring that the inverse sum is correctly scaled.
    ///
    /// # Arguments
    ///
    /// * `input` - A 1-D [`Field`].
    ///
    /// # Returns
    ///
    /// (`Result<Field>`): The spectrally filtered output field.
    ///
    /// # Errors
    ///
    /// Returns [`LmmError::Operator`] for non-1-D fields.
    ///
    /// # Time Complexity
    ///
    /// O(n · n_modes) for the DFT sums (naive implementation).
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::operator::FourierOperator;
    /// use lmm::field::Field;
    /// use lmm::tensor::Tensor;
    ///
    /// let t = Tensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// let f = Field::new(vec![4], t).unwrap();
    /// let op = FourierOperator::new(2);
    /// let out = op.transform(&f).unwrap();
    /// assert_eq!(out.values.data.len(), 4);
    /// ```
    pub fn transform(&self, input: &Field) -> Result<Field> {
        if input.dimensions.len() != 1 {
            return Err(LmmError::Operator(
                "FourierOperator requires 1-D field".into(),
            ));
        }
        let n = input.values.data.len();
        let n_modes = self.spectral_weights.len().min(n / 2 + 1);
        let pi2 = 2.0 * std::f64::consts::PI;

        let mut real = vec![0.0f64; n_modes];
        let mut imag = vec![0.0f64; n_modes];
        for k in 0..n_modes {
            for j in 0..n {
                let angle = pi2 * k as f64 * j as f64 / n as f64;
                real[k] += input.values.data[j] * angle.cos();
                imag[k] -= input.values.data[j] * angle.sin();
            }
            real[k] = real[k] * self.spectral_weights[k] / n as f64;
            imag[k] = imag[k] * self.spectral_weights[k] / n as f64;
        }

        let mut out = vec![0.0f64; n];
        for (j, slot) in out.iter_mut().enumerate() {
            let mut sum = 0.0;
            for k in 0..n_modes {
                let angle = pi2 * k as f64 * j as f64 / n as f64;
                sum += real[k] * angle.cos() - imag[k] * angle.sin();
            }
            *slot = sum;
        }
        let tensor = Tensor::new(input.dimensions.clone(), out)?;
        Field::new(input.dimensions.clone(), tensor)
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
