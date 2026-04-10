use crate::error::Result;
use crate::field::Field;

pub struct NeuralOperator {
    pub kernel_weights: Vec<f64>,
}

impl NeuralOperator {
    pub fn transform(&self, input: &Field) -> Result<Field> {
        let mut scaled_values = input.values.clone();
        for (i, v) in scaled_values.data.iter_mut().enumerate() {
            let weight = self.kernel_weights[i % self.kernel_weights.len()];
            *v *= weight;
        }
        Field::new(input.dimensions.clone(), scaled_values)
    }
}
