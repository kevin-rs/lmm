use crate::error::Result;
use crate::tensor::Tensor;
use crate::traits::Simulatable;

const RK4_HALF: f64 = 0.5;
const RK4_SIXTH: f64 = 1.0 / 6.0;
const RK4_WEIGHT_MIDDLE: f64 = 2.0;

const RK45_C2: f64 = 1.0 / 4.0;

const RK45_A31: f64 = 3.0 / 32.0;
const RK45_A32: f64 = 9.0 / 32.0;

const RK45_A41: f64 = 1932.0 / 2197.0;
const RK45_A42: f64 = -7200.0 / 2197.0;
const RK45_A43: f64 = 7296.0 / 2197.0;

const RK45_B4_1: f64 = 25.0 / 216.0;
const RK45_B4_3: f64 = 1408.0 / 2565.0;
const RK45_B4_4: f64 = 2197.0 / 4104.0;

const RK45_B5_1: f64 = 16.0 / 135.0;
const RK45_B5_3: f64 = 6656.0 / 12825.0;
const RK45_B5_4: f64 = 28561.0 / 56430.0;

const RK45_SAFETY: f64 = 0.9;
const RK45_EXPONENT: f64 = 0.2;
const RK45_STEP_SHRINK: f64 = 0.1;
const RK45_STEP_GROW: f64 = 5.0;
const RK45_DOUBLE: f64 = 2.0;

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
        let s2 = (current + &k1.scale(self.step_size * RK4_HALF))?;
        let k2 = model.evaluate_derivatives(&s2)?;
        let s3 = (current + &k2.scale(self.step_size * RK4_HALF))?;
        let k3 = model.evaluate_derivatives(&s3)?;
        let s4 = (current + &k3.scale(self.step_size))?;
        let k4 = model.evaluate_derivatives(&s4)?;
        let sum1 = (&k1 + &k2.scale(RK4_WEIGHT_MIDDLE))?;
        let sum2 = (&k3.scale(RK4_WEIGHT_MIDDLE) + &k4)?;
        let total = (&sum1 + &sum2)?;
        let delta = total.scale(RK4_SIXTH * self.step_size);
        current + &delta
    }

    pub fn leapfrog_step<M: Simulatable>(
        &self,
        model: &M,
        positions: &Tensor,
        velocities: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let state = (positions + velocities)?;
        let accels = model.evaluate_derivatives(&state)?;
        let half_v = (velocities + &accels.scale(self.step_size * RK4_HALF))?;
        let new_pos = (positions + &half_v.scale(self.step_size))?;
        let new_state = (&new_pos + &half_v)?;
        let new_accels = model.evaluate_derivatives(&new_state)?;
        let new_vel = (&half_v + &new_accels.scale(self.step_size * RK4_HALF))?;
        Ok((new_pos, new_vel))
    }

    pub fn rk45_step<M: Simulatable>(
        &self,
        model: &M,
        current: &Tensor,
        tol: f64,
    ) -> Result<(Tensor, f64)> {
        let k1 = model.evaluate_derivatives(current)?;
        let s2 = (current + &k1.scale(RK45_C2 * self.step_size))?;
        let k2 = model.evaluate_derivatives(&s2)?;
        let s3 = (current + &(&k1.scale(RK45_A31) + &k2.scale(RK45_A32))?.scale(self.step_size))?;
        let k3 = model.evaluate_derivatives(&s3)?;
        let s4 = (current
            + &(&(&k1.scale(RK45_A41) + &k2.scale(RK45_A42))? + &k3.scale(RK45_A43))?
                .scale(self.step_size))?;
        let k4 = model.evaluate_derivatives(&s4)?;
        let fourth_order = (current
            + &(&(&k1.scale(RK45_B4_1) + &k3.scale(RK45_B4_3))? + &k4.scale(RK45_B4_4))?
                .scale(self.step_size))?;
        let fifth_order = (current
            + &(&(&k1.scale(RK45_B5_1) + &k3.scale(RK45_B5_3))? + &k4.scale(RK45_B5_4))?
                .scale(self.step_size))?;
        let err_vec = (&fifth_order - &fourth_order)?;
        let error = err_vec.norm();
        let new_h = if error > 0.0 {
            RK45_SAFETY * self.step_size * (tol / error).powf(RK45_EXPONENT)
        } else {
            self.step_size * RK45_DOUBLE
        };
        let clamped_h = new_h.clamp(
            self.step_size * RK45_STEP_SHRINK,
            self.step_size * RK45_STEP_GROW,
        );
        Ok((fourth_order, clamped_h))
    }

    pub fn simulate_trajectory<M: Simulatable>(
        &self,
        model: &M,
        initial: &Tensor,
        n_steps: usize,
    ) -> Result<Vec<Tensor>> {
        let mut states = Vec::with_capacity(n_steps + 1);
        let mut current = initial.clone();
        for _ in 0..n_steps {
            states.push(current.clone());
            current = self.rk4_step(model, &current)?;
        }
        states.push(current);
        Ok(states)
    }

    pub fn simulate_adaptive<M: Simulatable>(
        &self,
        model: &M,
        initial: &Tensor,
        n_steps: usize,
        tol: f64,
    ) -> Result<Vec<Tensor>> {
        let mut states = Vec::with_capacity(n_steps + 1);
        let mut current = initial.clone();
        let mut step = self.step_size;
        for _ in 0..n_steps {
            states.push(current.clone());
            let sim = Simulator { step_size: step };
            let (next, new_h) = sim.rk45_step(model, &current, tol)?;
            current = next;
            step = new_h;
        }
        states.push(current);
        Ok(states)
    }
}
