// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # World Model
//!
//! This module provides [`WorldModel`], a predictive state estimator that maintains a
//! running state vector and tracks prediction errors over time. It bridges the gap
//! between raw physics simulations and autonomous agents by exposing three update modes:
//!
//! - **Action-based** (`step`) - applies an external action tensor directly.
//! - **Physics-based** (`physics_step`) - runs one RK4 step then applies an action.
//! - **Equation-based** (`equation_step`) - evaluates a symbolic expression to compute Δstate.
//!
//! Horizon-planning helpers allow agents to preview future states without mutating `self`.

use crate::equation::Expression;
use crate::error::{LmmError, Result};
use crate::simulation::Simulator;
use crate::tensor::Tensor;
use crate::traits::Simulatable;
use std::collections::HashMap;

/// A predictive world model that tracks the current state and a history of prediction errors.
///
/// `WorldModel` is the primary "belief" structure in the `lmm` consciousness loop. It is
/// updated each step with physical forces and intentional actions, and it records how far
/// predictions deviated from observations.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::world::WorldModel;
/// use lmm::tensor::Tensor;
///
/// let init = Tensor::zeros(vec![3]);
/// let mut wm = WorldModel::new(init);
/// let action = Tensor::from_vec(vec![1.0, 0.0, 0.0]);
/// let next = wm.step(&action).unwrap();
/// assert_eq!(next.data[0], 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct WorldModel {
    /// The current best estimate of the world state.
    pub current_state: Tensor,
    /// Rolling log of mean-squared prediction errors; grows with each `record_error` call.
    pub prediction_errors: Vec<f64>,
}

impl WorldModel {
    /// Creates a new [`WorldModel`] initialised to `initial_state`.
    ///
    /// # Arguments
    ///
    /// * `initial_state` - The starting state tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::world::WorldModel;
    /// use lmm::tensor::Tensor;
    ///
    /// let wm = WorldModel::new(Tensor::zeros(vec![2]));
    /// assert_eq!(wm.prediction_errors.len(), 0);
    /// ```
    pub fn new(initial_state: Tensor) -> Self {
        Self {
            current_state: initial_state,
            prediction_errors: Vec::new(),
        }
    }

    /// Advances the world state by adding `action` to the current state.
    ///
    /// # Arguments
    ///
    /// * `action` - A tensor of the same shape as the current state.
    ///
    /// # Returns
    ///
    /// (`Result<Tensor>`): The new state.
    ///
    /// # Errors
    ///
    /// Returns a shape-mismatch error when `action.shape != current_state.shape`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::world::WorldModel;
    /// use lmm::tensor::Tensor;
    ///
    /// let mut wm = WorldModel::new(Tensor::from_vec(vec![0.0, 0.0]));
    /// let next = wm.step(&Tensor::from_vec(vec![1.0, 2.0])).unwrap();
    /// assert_eq!(next.data, vec![1.0, 2.0]);
    /// ```
    pub fn step(&mut self, action: &Tensor) -> Result<Tensor> {
        let next_state = (&self.current_state + action)?;
        self.current_state = next_state.clone();
        Ok(next_state)
    }

    /// Advances the world state using one RK4 physics step followed by an action.
    ///
    /// The physics update is computed independently at the current state; the action is
    /// then added to the physics-predicted state.
    ///
    /// # Arguments
    ///
    /// * `physics` - Any [`Simulatable`] model.
    /// * `action` - Action tensor to apply after the physics step.
    /// * `step_size` - Integration step Δt.
    ///
    /// # Returns
    ///
    /// (`Result<Tensor>`): Physics + action next state.
    ///
    /// # Examples
    ///
    /// ```
    /// # use lmm::traits::Simulatable;
    /// use lmm::world::WorldModel;
    /// use lmm::physics::HarmonicOscillator;
    /// use lmm::tensor::Tensor;
    ///
    /// let osc = HarmonicOscillator::new(1.0, 1.0, 0.0).unwrap();
    /// let mut wm = WorldModel::new(osc.state().clone());
    /// let action = Tensor::zeros(vec![2]);
    /// let next = wm.physics_step(&osc, &action, 0.01).unwrap();
    /// assert_eq!(next.shape, vec![2]);
    /// ```
    pub fn physics_step<M: Simulatable>(
        &mut self,
        physics: &M,
        action: &Tensor,
        step_size: f64,
    ) -> Result<Tensor> {
        let sim = Simulator { step_size };
        let physics_next = sim.rk4_step(physics, &self.current_state)?;
        let next_state = (&physics_next + action)?;
        self.current_state = next_state.clone();
        Ok(next_state)
    }

    /// Advances the world state by evaluating a symbolic equation.
    ///
    /// The equation is evaluated with variables bound **positionally** to each component
    /// of the current state. The resulting scalar `derivative` is applied uniformly to all
    /// dimensions: `state += derivative * step_size`.
    ///
    /// # Arguments
    ///
    /// * `equation` - The symbolic expression defining the dynamics.
    /// * `var_names` - Variable names to bind to the state components (must match state length).
    /// * `step_size` - Integration step Δt.
    ///
    /// # Returns
    ///
    /// (`Result<Tensor>`): New state after applying the symbolic derivative.
    ///
    /// # Errors
    ///
    /// Returns [`LmmError::InvalidDimension`] when `var_names.len() != current_state.data.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::world::WorldModel;
    /// use lmm::equation::Expression;
    /// use lmm::tensor::Tensor;
    ///
    /// let mut wm = WorldModel::new(Tensor::from_vec(vec![1.0]));
    /// let expr = Expression::Variable("x".into());
    /// let next = wm.equation_step(&expr, &["x"], 0.1).unwrap();
    /// assert!((next.data[0] - 1.1).abs() < 1e-9);
    /// ```
    pub fn equation_step(
        &mut self,
        equation: &Expression,
        var_names: &[&str],
        step_size: f64,
    ) -> Result<Tensor> {
        if var_names.len() != self.current_state.data.len() {
            return Err(LmmError::InvalidDimension {
                expected: self.current_state.data.len(),
                got: var_names.len(),
            });
        }
        let bindings: HashMap<String, f64> = var_names
            .iter()
            .zip(self.current_state.data.iter())
            .map(|(k, &v)| ((*k).to_string(), v))
            .collect();
        let derivative = equation.evaluate(&bindings)?;
        let delta = Tensor::fill(self.current_state.shape.clone(), derivative * step_size);
        let next_state = (&self.current_state + &delta)?;
        self.current_state = next_state.clone();
        Ok(next_state)
    }

    /// Previews future states by applying a sequence of action tensors without mutating `self`.
    ///
    /// # Arguments
    ///
    /// * `actions` - Ordered action tensors; `actions[i]` is applied at step `i`.
    ///
    /// # Returns
    ///
    /// (`Result<Vec<Tensor>>`): Predicted states after each action.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::world::WorldModel;
    /// use lmm::tensor::Tensor;
    ///
    /// let wm = WorldModel::new(Tensor::from_vec(vec![0.0]));
    /// let actions = vec![
    ///     Tensor::from_vec(vec![1.0]),
    ///     Tensor::from_vec(vec![1.0]),
    /// ];
    /// let preds = wm.predict_horizon(&actions).unwrap();
    /// assert_eq!(preds.len(), 2);
    /// assert_eq!(preds[1].data[0], 2.0);
    /// ```
    pub fn predict_horizon(&self, actions: &[Tensor]) -> Result<Vec<Tensor>> {
        let mut predictions = Vec::with_capacity(actions.len());
        let mut temp_state = self.current_state.clone();
        for action in actions {
            temp_state = (&temp_state + action)?;
            predictions.push(temp_state.clone());
        }
        Ok(predictions)
    }

    /// Previews a physics-only trajectory without mutating `self`.
    ///
    /// # Arguments
    ///
    /// * `physics` - Any [`Simulatable`] model.
    /// * `n_steps` - Number of RK4 steps to simulate.
    /// * `step_size` - Integration step Δt.
    ///
    /// # Returns
    ///
    /// (`Result<Vec<Tensor>>`): Predicted `n_steps + 1` states.
    ///
    /// # Examples
    ///
    /// ```
    /// # use lmm::traits::Simulatable;
    /// use lmm::world::WorldModel;
    /// use lmm::physics::LorenzSystem;
    ///
    /// let lorenz = LorenzSystem::canonical().unwrap();
    /// let wm = WorldModel::new(lorenz.state().clone());
    /// let preds = wm.predict_horizon_physics(&lorenz, 5, 0.01).unwrap();
    /// assert_eq!(preds.len(), 6);
    /// ```
    pub fn predict_horizon_physics<M: Simulatable>(
        &self,
        physics: &M,
        n_steps: usize,
        step_size: f64,
    ) -> Result<Vec<Tensor>> {
        let sim = Simulator { step_size };
        sim.simulate_trajectory(physics, &self.current_state, n_steps)
    }

    /// Records the MSE between a `predicted` and `actual` tensor and appends it to the
    /// internal error log.
    ///
    /// # Arguments
    ///
    /// * `predicted` - The model's prediction.
    /// * `actual` - The observed value.
    ///
    /// # Returns
    ///
    /// (`Result<f64>`): The recorded MSE.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::world::WorldModel;
    /// use lmm::tensor::Tensor;
    ///
    /// let mut wm = WorldModel::new(Tensor::zeros(vec![2]));
    /// let pred = Tensor::from_vec(vec![1.0, 2.0]);
    /// let actual = Tensor::from_vec(vec![1.0, 2.0]);
    /// let mse = wm.record_error(&pred, &actual).unwrap();
    /// assert_eq!(mse, 0.0);
    /// ```
    pub fn record_error(&mut self, predicted: &Tensor, actual: &Tensor) -> Result<f64> {
        let diff = (predicted - actual)?;
        let mse = diff.data.iter().map(|x| x * x).sum::<f64>() / diff.data.len() as f64;
        self.prediction_errors.push(mse);
        Ok(mse)
    }

    /// Returns the arithmetic mean of all recorded prediction errors.
    ///
    /// Returns `0.0` when no errors have been recorded.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::world::WorldModel;
    /// use lmm::tensor::Tensor;
    ///
    /// let mut wm = WorldModel::new(Tensor::zeros(vec![1]));
    /// assert_eq!(wm.mean_prediction_error(), 0.0);
    /// ```
    pub fn mean_prediction_error(&self) -> f64 {
        if self.prediction_errors.is_empty() {
            return 0.0;
        }
        self.prediction_errors.iter().sum::<f64>() / self.prediction_errors.len() as f64
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
