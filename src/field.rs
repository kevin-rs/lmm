use crate::error::LmmError::Simulation;
use crate::error::Result;
use crate::tensor::Tensor;

#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub dimensions: Vec<usize>,
    pub values: Tensor,
}

impl Field {
    pub fn new(dimensions: Vec<usize>, values: Tensor) -> Result<Self> {
        if dimensions != values.shape {
            return Err(Simulation("Field bounds mismatch".into()));
        }
        Ok(Self { dimensions, values })
    }

    pub fn compute_gradient(&self) -> Result<Field> {
        let mut grad_data = vec![0.0; self.values.data.len()];
        if self.dimensions.len() != 1 {
            return Err(Simulation("Gradient currently 1D only".into()));
        }
        for (i, grad) in grad_data
            .iter_mut()
            .enumerate()
            .take(self.dimensions[0] - 1)
            .skip(1)
        {
            *grad = (self.values.data[i + 1] - self.values.data[i - 1]) / 2.0;
        }
        let grad_tensor = Tensor::new(self.dimensions.clone(), grad_data)?;
        Ok(Self {
            dimensions: self.dimensions.clone(),
            values: grad_tensor,
        })
    }
}
