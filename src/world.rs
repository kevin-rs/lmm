use crate::error::Result;
use crate::tensor::Tensor;

pub struct WorldModel {
    pub current_state: Tensor,
}

impl WorldModel {
    pub fn step(&mut self, action: &Tensor) -> Result<Tensor> {
        let next_state = (&self.current_state + action)?;
        self.current_state = next_state.clone();
        Ok(next_state)
    }

    pub fn predict_horizon(&self, actions: &[Tensor]) -> Result<Vec<Tensor>> {
        let mut predictions = Vec::new();
        let mut temp_state = self.current_state.clone();
        for action in actions {
            temp_state = (&temp_state + action)?;
            predictions.push(temp_state.clone());
        }
        Ok(predictions)
    }
}
