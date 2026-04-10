use crate::error::Result;
use crate::tensor::Tensor;
use crate::traits::Simulatable;

pub struct Simulator {
    pub step_size: f64,
}

impl Simulator {
    pub fn euler_step<M: Simulatable>(&self, model: &M, current: &Tensor) -> Result<Tensor> {
        let deriv = model.evaluate_derivatives(current)?;
        let delta = deriv.scale(self.step_size);
        current + &delta
    }

    pub fn rk4_step<M: Simulatable>(&self, model: &M, current: &Tensor) -> Result<Tensor> {
        let k1 = model.evaluate_derivatives(current)?;
        let half_step_k1 = k1.scale(self.step_size / 2.0);
        let s2 = (current + &half_step_k1)?;
        let k2 = model.evaluate_derivatives(&s2)?;
        let half_step_k2 = k2.scale(self.step_size / 2.0);
        let s3 = (current + &half_step_k2)?;
        let k3 = model.evaluate_derivatives(&s3)?;
        let full_step_k3 = k3.scale(self.step_size);
        let s4 = (current + &full_step_k3)?;
        let k4 = model.evaluate_derivatives(&s4)?;

        let sum1 = (&k1 + &k2.scale(2.0))?;
        let sum2 = (&k3.scale(2.0) + &k4)?;
        let total = (&sum1 + &sum2)?;
        let avg_deriv = total.scale(1.0 / 6.0);
        let delta = avg_deriv.scale(self.step_size);
        current + &delta
    }
}
