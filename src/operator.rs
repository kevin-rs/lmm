use crate::error::{LmmError, Result};
use crate::field::Field;
use crate::tensor::Tensor;
use crate::traits::Learnable;

pub struct NeuralOperator {
    pub kernel_weights: Vec<f64>,
}

impl NeuralOperator {
    pub fn new(kernel_size: usize) -> Self {
        let weights = (0..kernel_size)
            .map(|i| if i == kernel_size / 2 { 1.0 } else { 0.0 })
            .collect();
        Self {
            kernel_weights: weights,
        }
    }

    pub fn transform(&self, input: &Field) -> Result<Field> {
        if self.kernel_weights.is_empty() {
            return Err(LmmError::Operator("Empty kernel weights".into()));
        }
        let n = input.values.data.len();
        let k = self.kernel_weights.len();
        let half = k / 2;
        let mut out = vec![0.0; n];
        for (i, slot) in out.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (j, &w) in self.kernel_weights.iter().enumerate() {
                let src = (i + j + n - half) % n;
                acc += input.values.data[src] * w;
            }
            *slot = acc;
        }
        let tensor = Tensor::new(input.dimensions.clone(), out)?;
        Field::new(input.dimensions.clone(), tensor)
    }

    pub fn gradient_wrt_kernel(&self, input: &Field, target: &Field) -> Result<Vec<f64>> {
        if input.dimensions != target.dimensions {
            return Err(LmmError::Operator(
                "Input/target dimension mismatch for gradient".into(),
            ));
        }
        let output = self.transform(input)?;
        let n = input.values.data.len();
        let k = self.kernel_weights.len();
        let half = k / 2;
        let mut grads = vec![0.0; k];
        for i in 0..n {
            let residual = output.values.data[i] - target.values.data[i];
            for (j, slot) in grads.iter_mut().enumerate() {
                let src = (i + j + n - half) % n;
                *slot += 2.0 * residual * input.values.data[src] / n as f64;
            }
        }
        Ok(grads)
    }
}

impl Learnable for NeuralOperator {
    fn update(&mut self, grad: &Tensor, lr: f64) -> Result<()> {
        if grad.data.len() != self.kernel_weights.len() {
            return Err(LmmError::Operator(format!(
                "Gradient length {} != kernel length {}",
                grad.data.len(),
                self.kernel_weights.len()
            )));
        }
        for (w, &g) in self.kernel_weights.iter_mut().zip(grad.data.iter()) {
            *w -= lr * g;
        }
        Ok(())
    }
}

pub struct FourierOperator {
    pub spectral_weights: Vec<f64>,
}

impl FourierOperator {
    pub fn new(n_modes: usize) -> Self {
        Self {
            spectral_weights: vec![1.0; n_modes],
        }
    }

    pub fn transform(&self, input: &Field) -> Result<Field> {
        if input.dimensions.len() != 1 {
            return Err(LmmError::Operator(
                "FourierOperator requires 1-D field".into(),
            ));
        }
        let n = input.values.data.len();
        let n_modes = self.spectral_weights.len().min(n / 2 + 1);
        let pi2 = 2.0 * std::f64::consts::PI;
        let mut real = vec![0.0f64; n_modes];
        let mut imag = vec![0.0f64; n_modes];

        for k in 0..n_modes {
            for j in 0..n {
                let angle = pi2 * k as f64 * j as f64 / n as f64;
                real[k] += input.values.data[j] * angle.cos();
                imag[k] -= input.values.data[j] * angle.sin();
            }
            real[k] *= self.spectral_weights[k];
            imag[k] *= self.spectral_weights[k];
        }

        let mut out = vec![0.0f64; n];
        for (j, slot) in out.iter_mut().enumerate() {
            let mut sum = 0.0;
            for k in 0..n_modes {
                let angle = pi2 * k as f64 * j as f64 / n as f64;
                sum += real[k] * angle.cos() - imag[k] * angle.sin();
            }
            *slot = sum / n as f64;
        }
        let tensor = Tensor::new(input.dimensions.clone(), out)?;
        Field::new(input.dimensions.clone(), tensor)
    }
}
