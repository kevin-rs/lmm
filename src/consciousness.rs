use crate::error::{LmmError, Result};
use crate::perception::MultiModalPerception;
use crate::tensor::Tensor;
use crate::traits::Perceivable;
use crate::world::WorldModel;

pub struct Consciousness {
    pub world_model: WorldModel,
    pub lookahead_depth: usize,
    pub step_size: f64,
}

impl Consciousness {
    pub fn new(initial_state: Tensor, lookahead_depth: usize, step_size: f64) -> Self {
        Self {
            world_model: WorldModel::new(initial_state),
            lookahead_depth,
            step_size,
        }
    }

    pub fn tick(&mut self, sensory_input: &[u8]) -> Result<Tensor> {
        let perception = MultiModalPerception::ingest(sensory_input)?;
        let action = self.encode(&perception)?;
        self.world_model.step(&action)
    }

    fn encode(&self, perception: &Tensor) -> Result<Tensor> {
        let state_len = self.world_model.current_state.data.len();
        if perception.data.is_empty() {
            return Err(LmmError::Consciousness("Empty perception tensor".into()));
        }
        let compressed: Vec<f64> = (0..state_len)
            .map(|i| {
                let src = i % perception.data.len();
                perception.data[src] - 0.5
            })
            .collect();
        Tensor::new(vec![state_len], compressed).map_err(|e| LmmError::Consciousness(e.to_string()))
    }

    pub fn predict_next(&self, action: &Tensor) -> Result<Tensor> {
        &self.world_model.current_state + action
    }

    pub fn evaluate_prediction(&mut self, predicted: &Tensor, actual: &Tensor) -> Result<f64> {
        self.world_model.record_error(predicted, actual)
    }

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

    pub fn mean_prediction_error(&self) -> f64 {
        self.world_model.mean_prediction_error()
    }
}
