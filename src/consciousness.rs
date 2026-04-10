use crate::error::Result;
use crate::perception::MultiModalPerception;
use crate::tensor::Tensor;
use crate::traits::Perceivable;
use crate::world::WorldModel;

pub struct Consciousness {
    pub world_model: WorldModel,
}

impl Consciousness {
    pub fn tick(&mut self, sensory_input: &[u8]) -> Result<Tensor> {
        let perception_tensor = MultiModalPerception::ingest(sensory_input)?;
        let action = perception_tensor.scale(-0.1);
        self.world_model.step(&action)
    }
}
