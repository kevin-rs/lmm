use crate::error::LmmError::Perception;
use crate::error::Result;
use crate::tensor::Tensor;
use crate::traits::Perceivable;

pub struct MultiModalPerception;

impl Perceivable for MultiModalPerception {
    fn ingest(raw_data: &[u8]) -> Result<Tensor> {
        if raw_data.is_empty() {
            return Err(Perception("Empty input data".into()));
        }
        let float_data: Vec<f64> = raw_data.iter().map(|&b| b as f64 / 255.0).collect();
        Tensor::new(vec![float_data.len()], float_data)
    }
}
