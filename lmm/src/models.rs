// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Mathematical Models
//!
//! This module provides lightweight algebraic structures - [`Vector`], [`State`],
//! [`LinearModel`], and [`PolynomialModel`] - together with the [`MathematicalModel`] trait
//! and the [`FitResult`] diagnostic type. These primitives support the physics and simulation
//! subsystems and serve as building blocks for higher-level equation discovery.

use crate::error::{LmmError, Result};

/// A one-dimensional algebraic vector over `f64`.
///
/// Provides dot products, addition, scaling, and L2 norm operations.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::models::Vector;
///
/// let a = Vector::new(vec![1.0, 2.0, 3.0]);
/// let b = Vector::new(vec![4.0, 5.0, 6.0]);
/// assert_eq!(a.dot(&b), 32.0);
/// assert!((a.norm() - 14.0_f64.sqrt()).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Vector {
    /// The underlying floating-point values.
    pub data: Vec<f64>,
}

impl Vector {
    /// Creates a new [`Vector`] from a `Vec<f64>`.
    ///
    /// # Arguments
    ///
    /// * `data` - The values comprising the vector.
    ///
    /// # Returns
    ///
    /// (`Vector`): A new vector wrapping `data`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::models::Vector;
    /// let v = Vector::new(vec![1.0, 0.0]);
    /// assert_eq!(v.data.len(), 2);
    /// ```
    pub fn new(data: Vec<f64>) -> Self {
        Self { data }
    }

    /// Computes the inner (dot) product with `other`.
    ///
    /// Pairs are multiplied element-wise and summed. Excess elements in the longer
    /// vector are silently ignored (zip semantics).
    ///
    /// # Arguments
    ///
    /// * `other` - The other vector.
    ///
    /// # Returns
    ///
    /// (`f64`): The scalar dot product.
    ///
    /// # Time Complexity
    ///
    /// O(n) where n is `min(self.len(), other.len())`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::models::Vector;
    /// let a = Vector::new(vec![1.0, 2.0]);
    /// let b = Vector::new(vec![3.0, 4.0]);
    /// assert_eq!(a.dot(&b), 11.0);
    /// ```
    pub fn dot(&self, other: &Self) -> f64 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Adds `other` element-wise to `self`, returning a new [`Vector`].
    ///
    /// # Arguments
    ///
    /// * `other` - The addend vector.
    ///
    /// # Returns
    ///
    /// (`Vector`): Element-wise sum.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::models::Vector;
    /// let a = Vector::new(vec![1.0, 2.0]);
    /// let b = Vector::new(vec![3.0, 4.0]);
    /// assert_eq!(a.add(&b).data, vec![4.0, 6.0]);
    /// ```
    pub fn add(&self, other: &Self) -> Self {
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Self { data }
    }

    /// Multiplies every element by `factor`, returning a new [`Vector`].
    ///
    /// # Arguments
    ///
    /// * `factor` - The scalar multiplier.
    ///
    /// # Returns
    ///
    /// (`Vector`): Scaled vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::models::Vector;
    /// let v = Vector::new(vec![2.0, 4.0]);
    /// assert_eq!(v.scale(0.5).data, vec![1.0, 2.0]);
    /// ```
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            data: self.data.iter().map(|x| x * factor).collect(),
        }
    }

    /// Returns the Euclidean (L2) norm: `√(Σ xᵢ²)`.
    ///
    /// # Returns
    ///
    /// (`f64`): The L2 norm of the vector.
    ///
    /// # Time Complexity
    ///
    /// O(n).
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::models::Vector;
    /// let v = Vector::new(vec![3.0, 4.0]);
    /// assert!((v.norm() - 5.0).abs() < 1e-10);
    /// ```
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}

/// A point-in-time snapshot of a dynamical system, pairing a timestamp with state variables.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::models::{State, Vector};
///
/// let s = State { time: 1.0, variables: Vector::new(vec![0.5, -0.3]) };
/// assert_eq!(s.time, 1.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct State {
    /// The simulation time of this snapshot.
    pub time: f64,
    /// The state variables at time `time`.
    pub variables: Vector,
}

/// Abstracts any model that, given a [`State`], can produce a [`Vector`] of outputs.
///
/// Implementors include [`LinearModel`] and [`PolynomialModel`].
pub trait MathematicalModel {
    /// Evaluates the model at `state`.
    ///
    /// # Arguments
    ///
    /// * `state` - The current state from which to produce an output vector.
    ///
    /// # Returns
    ///
    /// (`Result<Vector>`): Model output, or an error on dimension mismatch.
    fn evaluate(&self, state: &State) -> Result<Vector>;
}

/// A linear model `ŷ = wᵀx + b` parameterised by weights and a scalar bias.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::models::{LinearModel, MathematicalModel, State, Vector};
///
/// let model = LinearModel::new(vec![2.0, 3.0], 1.0);
/// let state = State { time: 0.0, variables: Vector::new(vec![1.0, 1.0]) };
/// let out = model.evaluate(&state).unwrap();
/// assert_eq!(out.data[0], 6.0); // 2*1 + 3*1 + 1
/// ```
/// use lmm::traits::Simulatable;
#[derive(Debug, Clone)]
pub struct LinearModel {
    /// Weight vector applied to the input state variables.
    pub weights: Vector,
    /// Scalar bias added to the dot product.
    pub bias: f64,
}

impl Default for LinearModel {
    fn default() -> Self {
        Self {
            weights: Vector::default(),
            bias: 0.0,
        }
    }
}

impl LinearModel {
    /// Creates a new [`LinearModel`] from weight values and a bias.
    ///
    /// # Arguments
    ///
    /// * `weights` - The weight coefficients.
    /// * `bias` - The scalar bias term.
    ///
    /// # Returns
    ///
    /// (`LinearModel`): The configured model.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::models::LinearModel;
    /// let m = LinearModel::new(vec![1.0], 0.0);
    /// assert_eq!(m.bias, 0.0);
    /// ```
    pub fn new(weights: Vec<f64>, bias: f64) -> Self {
        Self {
            weights: Vector::new(weights),
            bias,
        }
    }
}

impl MathematicalModel for LinearModel {
    /// Evaluates `ŷ = wᵀx + b`.
    ///
    /// # Errors
    ///
    /// Returns [`LmmError::InvalidDimension`] when the number of state variables does not
    /// match the number of weight coefficients.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::models::{LinearModel, MathematicalModel, State, Vector};
    ///
    /// let m = LinearModel::new(vec![1.0, 2.0], 0.5);
    /// let s = State { time: 0.0, variables: Vector::new(vec![1.0, 1.0]) };
    /// assert_eq!(m.evaluate(&s).unwrap().data[0], 3.5);
    /// ```
    fn evaluate(&self, state: &State) -> Result<Vector> {
        if self.weights.data.len() != state.variables.data.len() {
            return Err(LmmError::InvalidDimension {
                expected: self.weights.data.len(),
                got: state.variables.data.len(),
            });
        }
        let prediction = self.weights.dot(&state.variables) + self.bias;
        Ok(Vector::new(vec![prediction]))
    }
}

/// A polynomial model `ŷ = Σᵢ cᵢ · xⁱ` evaluated at the first state variable.
///
/// Coefficients are ordered from degree 0 (constant) to degree `n-1`.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::models::{PolynomialModel, MathematicalModel, State, Vector};
///
/// let model = PolynomialModel::new(vec![1.0, 0.0, 1.0]); // 1 + x²
/// assert!((model.evaluate_at(2.0) - 5.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Default)]
pub struct PolynomialModel {
    /// Polynomial coefficients ordered from degree 0 upward.
    pub coeffs: Vec<f64>,
}

impl PolynomialModel {
    /// Creates a new [`PolynomialModel`] from its coefficients.
    ///
    /// # Arguments
    ///
    /// * `coeffs` - Coefficients `[c₀, c₁, ..., cₙ]` where `cᵢ` multiplies `xⁱ`.
    ///
    /// # Returns
    ///
    /// (`PolynomialModel`): The configured model.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::models::PolynomialModel;
    /// let m = PolynomialModel::new(vec![0.0, 1.0]); // y = x
    /// assert_eq!(m.evaluate_at(3.0), 3.0);
    /// ```
    pub fn new(coeffs: Vec<f64>) -> Self {
        Self { coeffs }
    }

    /// Evaluates the polynomial at the scalar `x`.
    ///
    /// Uses Horner's method for numerical stability and O(n) performance.
    ///
    /// # Arguments
    ///
    /// * `x` - The evaluation point.
    ///
    /// # Returns
    ///
    /// (`f64`): `Σᵢ coeffs[i] · xⁱ`.
    ///
    /// # Time Complexity
    ///
    /// O(n) where n is the number of coefficients.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::models::PolynomialModel;
    /// let m = PolynomialModel::new(vec![2.0, 3.0, 1.0]); // 2 + 3x + x²
    /// assert!((m.evaluate_at(1.0) - 6.0).abs() < 1e-10);
    /// assert!((m.evaluate_at(0.0) - 2.0).abs() < 1e-10);
    /// ```
    pub fn evaluate_at(&self, x: f64) -> f64 {
        self.coeffs
            .iter()
            .enumerate()
            .map(|(i, &c)| c * x.powi(i as i32))
            .sum()
    }
}

impl MathematicalModel for PolynomialModel {
    /// Evaluates the polynomial at `state.variables[0]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::models::{PolynomialModel, MathematicalModel, State, Vector};
    ///
    /// let m = PolynomialModel::new(vec![1.0, 1.0]); // 1 + x
    /// let s = State { time: 0.0, variables: Vector::new(vec![2.0]) };
    /// assert_eq!(m.evaluate(&s).unwrap().data[0], 3.0);
    /// ```
    fn evaluate(&self, state: &State) -> Result<Vector> {
        let x = state.variables.data.first().copied().unwrap_or(0.0);
        Ok(Vector::new(vec![self.evaluate_at(x)]))
    }
}

/// Diagnostics produced after fitting a model to data, containing MSE and R².
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::models::FitResult;
///
/// let preds = vec![1.0, 2.0, 3.0];
/// let targets = vec![1.0, 2.0, 3.0];
/// let fit = FitResult::new(&preds, &targets);
/// assert_eq!(fit.mse, 0.0);
/// assert!((fit.r_squared - 1.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct FitResult {
    /// Mean squared error between predictions and targets.
    pub mse: f64,
    /// Coefficient of determination (R²); 1.0 = perfect fit.
    pub r_squared: f64,
}

impl FitResult {
    /// Computes [`FitResult`] from aligned prediction and target slices.
    ///
    /// Returns zeroed values when `targets` is empty.
    ///
    /// # Arguments
    ///
    /// * `predictions` - Model outputs for each sample.
    /// * `targets` - Ground-truth scalar targets.
    ///
    /// # Returns
    ///
    /// (`FitResult`): MSE and R² over the provided data.
    ///
    /// # Time Complexity
    ///
    /// O(n) where n is `targets.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::models::FitResult;
    ///
    /// let fit = FitResult::new(&[2.0, 4.0], &[1.0, 3.0]);
    /// assert_eq!(fit.mse, 1.0);
    /// ```
    pub fn new(predictions: &[f64], targets: &[f64]) -> Self {
        if targets.is_empty() {
            return Self {
                mse: 0.0,
                r_squared: 0.0,
            };
        }
        let n = targets.len() as f64;
        let mse = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / n;
        let mean_t: f64 = targets.iter().sum::<f64>() / n;
        let ss_tot: f64 = targets.iter().map(|t| (t - mean_t).powi(2)).sum();
        let ss_res: f64 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum();
        let r_squared = if ss_tot == 0.0 {
            1.0
        } else {
            1.0 - ss_res / ss_tot
        };
        Self { mse, r_squared }
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
