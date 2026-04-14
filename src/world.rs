use crate::equation::Expression;
use crate::error::Result;
use crate::simulation::Simulator;
use crate::tensor::Tensor;
use crate::traits::Simulatable;
use std::collections::HashMap;

pub struct WorldModel {
    pub current_state: Tensor,
    pub prediction_errors: Vec<f64>,
}

impl WorldModel {
    pub fn new(initial_state: Tensor) -> Self {
        Self {
            current_state: initial_state,
            prediction_errors: Vec::new(),
        }
    }

    pub fn step(&mut self, action: &Tensor) -> Result<Tensor> {
        let next_state = (&self.current_state + action)?;
        self.current_state = next_state.clone();
        Ok(next_state)
    }

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

    pub fn equation_step(
        &mut self,
        equation: &Expression,
        var_names: &[&str],
        step_size: f64,
    ) -> Result<Tensor> {
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

    pub fn predict_horizon(&self, actions: &[Tensor]) -> Result<Vec<Tensor>> {
        let mut predictions = Vec::with_capacity(actions.len());
        let mut temp_state = self.current_state.clone();
        for action in actions {
            temp_state = (&temp_state + action)?;
            predictions.push(temp_state.clone());
        }
        Ok(predictions)
    }

    pub fn predict_horizon_physics<M: Simulatable>(
        &self,
        physics: &M,
        n_steps: usize,
        step_size: f64,
    ) -> Result<Vec<Tensor>> {
        let sim = Simulator { step_size };
        sim.simulate_trajectory(physics, &self.current_state, n_steps)
    }

    pub fn record_error(&mut self, predicted: &Tensor, actual: &Tensor) -> Result<f64> {
        let diff = (predicted - actual)?;
        let mse = diff.data.iter().map(|x| x * x).sum::<f64>() / diff.data.len() as f64;
        self.prediction_errors.push(mse);
        Ok(mse)
    }

    pub fn mean_prediction_error(&self) -> f64 {
        if self.prediction_errors.is_empty() {
            return 0.0;
        }
        self.prediction_errors.iter().sum::<f64>() / self.prediction_errors.len() as f64
    }
}
