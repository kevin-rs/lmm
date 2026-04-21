// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Core Traits
//!
//! This module defines the foundational behavioural contracts used throughout the `lmm` crate.
//! Each trait abstracts a distinct capability - simulation, discovery, perception, prediction,
//! encoding, learning, and causal intervention - so that concrete types can be composed and
//! swapped without coupling algorithms to specific data structures.

use crate::equation::Expression;
use crate::error::Result;
use crate::tensor::Tensor;

/// Implemented by any physical or mathematical system that can produce time-derivatives
/// of its own state vector, enabling numerical integration by [`crate::simulation::Simulator`].
///
/// # Required Methods
/// - `state` - returns a reference to the current state [`Tensor`].
/// - `evaluate_derivatives` - given a state vector, returns the corresponding derivatives.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::tensor::Tensor;
/// use lmm::error::Result;
///
/// struct FreeParticle { state: Tensor }
///
/// impl Simulatable for FreeParticle {
///     fn state(&self) -> &Tensor { &self.state }
///     fn evaluate_derivatives(&self, state: &Tensor) -> Result<Tensor> {
///         Tensor::new(state.shape.clone(), vec![1.0; state.data.len()])
///     }
/// }
/// ```
/// use lmm::traits::Simulatable;
pub trait Simulatable {
    /// Returns a reference to the current state vector of the system.
    fn state(&self) -> &Tensor;

    /// Computes the time-derivatives of `state`, returning a [`Tensor`] of the same shape.
    ///
    /// # Arguments
    ///
    /// * `state` - The current state vector at which derivatives should be evaluated.
    ///
    /// # Returns
    ///
    /// (`Result<Tensor>`): The derivative tensor, or an error on invalid state dimensions.
    fn evaluate_derivatives(&self, state: &Tensor) -> Result<Tensor>;
}

/// Implemented by algorithms that can infer governing [`Expression`]s from observed data.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::tensor::Tensor;
/// use lmm::equation::Expression;
/// use lmm::traits::Discoverable;
/// use lmm::error::Result;
/// use lmm::discovery::SymbolicRegression;
///
/// let data = vec![Tensor::from_vec(vec![1.0]), Tensor::from_vec(vec![2.0])];
/// let targets = vec![2.0, 4.0];
/// let expr = SymbolicRegression::discover(&data, &targets).unwrap();
/// // Discovered expression should approximate 2*x
/// ```
/// use lmm::traits::Simulatable;
pub trait Discoverable {
    /// Discovers a symbolic [`Expression`] that best fits `data → targets`.
    ///
    /// # Arguments
    ///
    /// * `data` - Input tensors (one per sample).
    /// * `targets` - Scalar target values aligned with `data`.
    ///
    /// # Returns
    ///
    /// (`Result<Expression>`): The best-fit symbolic expression.
    fn discover(data: &[Tensor], targets: &[f64]) -> Result<Expression>;
}

/// Implemented by sensors or modalities that can convert raw byte streams into [`Tensor`]s.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::perception::MultiModalPerception;
/// use lmm::traits::Perceivable;
///
/// let raw = &[128u8, 64, 255, 0];
/// let tensor = MultiModalPerception::ingest(raw).unwrap();
/// assert_eq!(tensor.data.len(), 4);
/// assert!((tensor.data[0] - 128.0 / 255.0).abs() < 1e-9);
/// ```
pub trait Perceivable {
    /// Converts `raw_data` bytes into a normalised [`Tensor`] in the range `[0, 1]`.
    ///
    /// # Arguments
    ///
    /// * `raw_data` - Raw byte slice representing sensory input.
    ///
    /// # Returns
    ///
    /// (`Result<Tensor>`): Normalised perception tensor.
    fn ingest(raw_data: &[u8]) -> Result<Tensor>;
}

/// Implemented by models that can extrapolate their state forward in time.
pub trait Predictable {
    /// Advances the model state by `steps` integration steps.
    ///
    /// # Arguments
    ///
    /// * `steps` - Number of discrete steps to predict ahead.
    ///
    /// # Returns
    ///
    /// (`Result<Tensor>`): Predicted state tensor after `steps` steps.
    fn predict(&self, steps: usize) -> Result<Tensor>;
}

/// Implemented by models that can project an input [`Tensor`] into a latent representation.
pub trait Encodable {
    /// Encodes `input` into a compressed or transformed representation.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor to encode.
    ///
    /// # Returns
    ///
    /// (`Result<Tensor>`): The encoded tensor.
    fn encode(&self, input: &Tensor) -> Result<Tensor>;
}

/// Implemented by models whose parameters can be updated via gradient descent.
pub trait Learnable {
    /// Applies a gradient update to the model's parameters.
    ///
    /// # Arguments
    ///
    /// * `grad` - Gradient tensor (must match the parameter shape).
    /// * `lr` - Learning rate scalar.
    ///
    /// # Returns
    ///
    /// (`Result<()>`): Ok on success, or an error on shape mismatch.
    fn update(&mut self, grad: &Tensor, lr: f64) -> Result<()>;
}

/// Implemented by causal models that support the `do(·)` intervention operator.
///
/// # See Also
/// - [Pearl, J. (2009). Causality. Cambridge University Press.](https://doi.org/10.1017/CBO9780511803161) - formalizes the `do(x)` operator semantic representing structural interventions.
pub trait Causal {
    /// Performs a hard intervention `do(var = value)`, severing all incoming edges to `var`.
    ///
    /// # Arguments
    ///
    /// * `var` - Name of the variable to intervene on.
    /// * `value` - The value to assign to `var` under the do-operator.
    ///
    /// # Returns
    ///
    /// (`Result<()>`): Ok on success, or an error if `var` is not found.
    fn intervene(&mut self, var: &str, value: f64) -> Result<()>;
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
