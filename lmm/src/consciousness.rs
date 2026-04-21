// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Consciousness Perception-Action Loop
//!
//! This module implements [`Consciousness`], a closed perception-action loop modelled after
//! the Global Workspace Theory of consciousness. At each discrete time step (`tick`), the
//! agent:
//!
//! 1. **Perceives** raw sensory bytes via [`MultiModalPerception`].
//! 2. **Encodes** the normalised perception tensor into an action delta using mean-pooling
//!    so the action dimension always matches the world-model state dimension.
//! 3. **Steps** the internal [`WorldModel`] forward.
//! 4. **Plans** by rolling out candidate actions over a look-ahead horizon and selecting
//!    the one with the lowest cumulative state norm (a proxy for deviation cost).
//!
//! # See Also
//! - [Baars, B. J. (1988). A Cognitive Theory of Consciousness. Cambridge University Press.](https://en.wikipedia.org/wiki/Global_workspace_theory) - details the Global Workspace Theory guiding this overarching module array.

use crate::error::{LmmError, Result};
use crate::perception::MultiModalPerception;
use crate::tensor::Tensor;
use crate::traits::Perceivable;
use crate::world::WorldModel;

/// A closed perception-action loop with look-ahead planning.
///
/// `Consciousness` wraps a [`WorldModel`] and adds sensory ingestion, action selection,
/// and prediction-error tracking.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::consciousness::Consciousness;
/// use lmm::tensor::Tensor;
///
/// let state = Tensor::zeros(vec![4]);
/// let mut c = Consciousness::new(state, 3, 0.01);
/// let next = c.tick(&[128u8, 64, 255, 32]).unwrap();
/// assert_eq!(next.shape, vec![4]);
/// ```
pub struct Consciousness {
    /// The internal predictive world model.
    pub world_model: WorldModel,
    /// Number of future steps to simulate during planning.
    pub lookahead_depth: usize,
    /// Integration step size used for physics-based planning extensions.
    pub step_size: f64,
}

impl Consciousness {
    /// Creates a new [`Consciousness`] loop initialised with `initial_state`.
    ///
    /// # Arguments
    ///
    /// * `initial_state` - Starting world-model state tensor.
    /// * `lookahead_depth` - Number of planning roll-out steps.
    /// * `step_size` - Integration step Δt (reserved for physics extensions).
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::consciousness::Consciousness;
    /// use lmm::tensor::Tensor;
    ///
    /// let c = Consciousness::new(Tensor::zeros(vec![2]), 5, 0.01);
    /// assert_eq!(c.lookahead_depth, 5);
    /// ```
    pub fn new(initial_state: Tensor, lookahead_depth: usize, step_size: f64) -> Self {
        Self {
            world_model: WorldModel::new(initial_state),
            lookahead_depth,
            step_size,
        }
    }

    /// Performs one perception-action tick.
    ///
    /// Ingests `sensory_input`, computes a mean-pooled action delta, and advances the
    /// world model.
    ///
    /// # Arguments
    ///
    /// * `sensory_input` - Raw byte slice from any sensory modality.
    ///
    /// # Returns
    ///
    /// (`Result<Tensor>`): The new world-model state after stepping.
    ///
    /// # Errors
    ///
    /// Returns [`LmmError::Perception`] for empty input or
    /// [`LmmError::Consciousness`] for encoding failures.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::consciousness::Consciousness;
    /// use lmm::tensor::Tensor;
    ///
    /// let mut c = Consciousness::new(Tensor::zeros(vec![2]), 3, 0.01);
    /// let state = c.tick(&[100u8, 200]).unwrap();
    /// assert_eq!(state.data.len(), 2);
    /// ```
    pub fn tick(&mut self, sensory_input: &[u8]) -> Result<Tensor> {
        let perception = MultiModalPerception::ingest(sensory_input)?;
        let action = self.encode(&perception)?;
        self.world_model.step(&action)
    }

    /// Encodes a perception tensor into an action delta via **mean-pooling**.
    ///
    /// The perception tensor can be any length; it is pooled into `state_len` windows
    /// using average pooling so that the output always matches the world-model dimension.
    ///
    /// For each output index `i ∈ [0, state_len)`:
    /// - The corresponding perception window spans indices `[start, end)` in the
    ///   perception tensor.
    /// - Output `i = mean(perception[start..end]) - 0.5`
    ///
    /// # Arguments
    ///
    /// * `perception` - The normalised perception tensor.
    ///
    /// # Returns
    ///
    /// (`Result<Tensor>`): Action delta tensor of shape `[state_len]`.
    ///
    /// # Errors
    ///
    /// Returns [`LmmError::Consciousness`] when the perception tensor is empty.
    fn encode(&self, perception: &Tensor) -> Result<Tensor> {
        let state_len = self.world_model.current_state.data.len();
        let p_len = perception.data.len();
        if p_len == 0 {
            return Err(LmmError::Consciousness("Empty perception tensor".into()));
        }

        let compressed: Vec<f64> = (0..state_len)
            .map(|i| {
                let start = (i * p_len) / state_len;
                let end = ((i + 1) * p_len) / state_len;
                let end = end.max(start + 1).min(p_len);
                let mean: f64 =
                    perception.data[start..end].iter().sum::<f64>() / (end - start) as f64;
                mean - 0.5
            })
            .collect();

        Tensor::new(vec![state_len], compressed).map_err(|e| LmmError::Consciousness(e.to_string()))
    }

    /// Predicts the next state given a candidate action, without mutating the world model.
    ///
    /// # Arguments
    ///
    /// * `action` - The candidate action tensor.
    ///
    /// # Returns
    ///
    /// (`Result<Tensor>`): Predicted next state.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::consciousness::Consciousness;
    /// use lmm::tensor::Tensor;
    ///
    /// let c = Consciousness::new(Tensor::from_vec(vec![1.0, 2.0]), 3, 0.01);
    /// let action = Tensor::from_vec(vec![0.1, 0.1]);
    /// let pred = c.predict_next(&action).unwrap();
    /// assert!((pred.data[0] - 1.1).abs() < 1e-9);
    /// ```
    pub fn predict_next(&self, action: &Tensor) -> Result<Tensor> {
        &self.world_model.current_state + action
    }

    /// Records the MSE between a predicted and actual observation.
    ///
    /// # Arguments
    ///
    /// * `predicted` - Model prediction tensor.
    /// * `actual` - Observed tensor.
    ///
    /// # Returns
    ///
    /// (`Result<f64>`): The MSE logged to the world model.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::consciousness::Consciousness;
    /// use lmm::tensor::Tensor;
    ///
    /// let mut c = Consciousness::new(Tensor::zeros(vec![2]), 1, 0.01);
    /// let mse = c.evaluate_prediction(
    ///     &Tensor::from_vec(vec![1.0, 0.0]),
    ///     &Tensor::from_vec(vec![1.0, 0.0]),
    /// ).unwrap();
    /// assert_eq!(mse, 0.0);
    /// ```
    pub fn evaluate_prediction(&mut self, predicted: &Tensor, actual: &Tensor) -> Result<f64> {
        self.world_model.record_error(predicted, actual)
    }

    /// Selects the best action from `candidate_actions` via greedy look-ahead planning.
    ///
    /// Each candidate action is rolled out for `lookahead_depth` steps with a simple
    /// proportional damping policy (`action = -0.05 * state`). The index with the
    /// lowest cumulative state norm is returned.
    ///
    /// # Arguments
    ///
    /// * `candidate_actions` - Non-empty slice of candidate action tensors.
    ///
    /// # Returns
    ///
    /// (`Result<usize>`): Index of the best candidate action.
    ///
    /// # Errors
    ///
    /// Returns [`LmmError::Consciousness`] when `candidate_actions` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::consciousness::Consciousness;
    /// use lmm::tensor::Tensor;
    ///
    /// let c = Consciousness::new(Tensor::from_vec(vec![1.0]), 2, 0.01);
    /// let actions = vec![
    ///     Tensor::from_vec(vec![-0.9]),
    ///     Tensor::from_vec(vec![0.9]),
    /// ];
    /// let best = c.plan(&actions).unwrap();
    /// // Action that moves toward zero should be preferred
    /// assert_eq!(best, 0);
    /// ```
    pub fn plan(&self, candidate_actions: &[Tensor]) -> Result<usize> {
        if candidate_actions.is_empty() {
            return Err(LmmError::Consciousness(
                "No candidate actions to plan over".into(),
            ));
        }
        let mut best_idx = 0;
        let mut best_cost = f64::INFINITY;
        for (i, action) in candidate_actions.iter().enumerate() {
            let cost = self.rollout_cost(action, self.lookahead_depth)?;
            if cost < best_cost {
                best_cost = cost;
                best_idx = i;
            }
        }
        Ok(best_idx)
    }

    /// Rolls out a damped trajectory for `depth` steps and returns the total cost.
    ///
    /// The roll-out policy is: `action_t = -0.05 * state_t` (proportional damping).
    fn rollout_cost(&self, initial_action: &Tensor, depth: usize) -> Result<f64> {
        let mut state = (&self.world_model.current_state + initial_action)?;
        let mut total_cost = state.norm();
        for _ in 1..depth {
            let action = state.scale(-0.05);
            state = (&state + &action)?;
            total_cost += state.norm();
        }
        Ok(total_cost)
    }

    /// Returns the arithmetic mean of all recorded prediction errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::consciousness::Consciousness;
    /// use lmm::tensor::Tensor;
    ///
    /// let c = Consciousness::new(Tensor::zeros(vec![1]), 1, 0.01);
    /// assert_eq!(c.mean_prediction_error(), 0.0);
    /// ```
    pub fn mean_prediction_error(&self) -> f64 {
        self.world_model.mean_prediction_error()
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
