// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # ODE/PDE Numerical Integration
//!
//! This module provides [`Simulator`], which implements four numerical integrators for
//! ordinary differential equations:
//!
//! | Method | Order | Adaptive | Stability |
//! |---|---|---|---|
//! | Euler | 1 | No | Low |
//! | RK4 | 4 | No | Good |
//! | Leapfrog (Störmer-Verlet) | 2 | No | Symplectic |
//! | RK45 (Dormand-Prince) | 4/5 | Yes | Good |
//!
//! All integrators operate on arbitrary [`Simulatable`] systems, making them compatible
//! with all physics models in [`crate::physics`].
//!
//! # See Also
//! - [Hairer, E., Nørsett, S. P., & Wanner, G. (1993). Solving Ordinary Differential Equations I. Springer.](https://doi.org/10.1007/978-3-540-78862-1) - comprehensive reference for the RK4 and RK45 integration methods implemented here.
//! - [`crate::physics`] - concrete physical models (e.g., `HarmonicOscillator`, `LorenzSystem`) that implement the `Simulatable` trait.

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

/// A stateless numerical integrator for [`Simulatable`] dynamical systems.
///
/// Set `step_size` to control the fixed integration step (for Euler, RK4, and leapfrog)
/// or the initial step size (for adaptive RK45).
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::simulation::Simulator;
/// use lmm::physics::HarmonicOscillator;
///
/// let osc = HarmonicOscillator::new(1.0, 1.0, 0.0).unwrap();
/// let sim = Simulator { step_size: 0.01 };
/// let state_after = sim.rk4_step(&osc, osc.state()).unwrap();
/// assert_eq!(state_after.shape, vec![2]);
/// ```
pub struct Simulator {
    /// Discrete time step (Δt) used by fixed-step integrators, or the initial Δt for RK45.
    pub step_size: f64,
}

impl Simulator {
    /// Advances the state by one **Euler** step: `x(t+h) ≈ x(t) + h · ẋ(t)`.
    ///
    /// # Arguments
    ///
    /// * `model` - Any [`Simulatable`] system.
    /// * `current` - The current state tensor.
    ///
    /// # Returns
    ///
    /// (`Result<Tensor>`): The next-step state approximation.
    ///
    /// # Errors
    ///
    /// Propagates any error from `model.evaluate_derivatives`.
    ///
    /// # Time Complexity
    ///
    /// O(n) where n is the state vector dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// # use lmm::traits::Simulatable;
    /// use lmm::simulation::Simulator;
    /// use lmm::physics::HarmonicOscillator;
    ///
    /// let osc = HarmonicOscillator::new(1.0, 1.0, 0.0).unwrap();
    /// let sim = Simulator { step_size: 0.01 };
    /// let next = sim.euler_step(&osc, osc.state()).unwrap();
    /// assert_eq!(next.shape, vec![2]);
    /// ```
    pub fn euler_step<M: Simulatable>(&self, model: &M, current: &Tensor) -> Result<Tensor> {
        let deriv = model.evaluate_derivatives(current)?;
        let delta = deriv.scale(self.step_size);
        current + &delta
    }

    /// Advances the state by one **4th-order Runge-Kutta (RK4)** step.
    ///
    /// RK4 evaluates four derivative stages per step and combines them with Simpson-like
    /// weights: `(k₁ + 2k₂ + 2k₃ + k₄) / 6`.
    ///
    /// # Arguments
    ///
    /// * `model` - Any [`Simulatable`] system.
    /// * `current` - The current state tensor.
    ///
    /// # Returns
    ///
    /// (`Result<Tensor>`): 4th-order accurate next step.
    ///
    /// # Time Complexity
    ///
    /// O(4n) - four derivative evaluations, each O(n).
    ///
    /// # Examples
    ///
    /// ```
    /// # use lmm::traits::Simulatable;
    /// use lmm::simulation::Simulator;
    /// use lmm::physics::LorenzSystem;
    ///
    /// let lorenz = LorenzSystem::canonical().unwrap();
    /// let sim = Simulator { step_size: 0.01 };
    /// let next = sim.rk4_step(&lorenz, lorenz.state()).unwrap();
    /// assert_eq!(next.shape, vec![3]);
    /// ```
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

    /// Advances positions and velocities by one **Störmer-Verlet (leapfrog)** step.
    ///
    /// The leapfrog integrator is **symplectic** - it conserves a shadow Hamiltonian
    /// and is therefore ideal for long-time energy simulations.
    ///
    /// The algorithm (kick-drift-kick form):
    /// 1. Half-kick: `v½ = v + (h/2) · a(x)`
    /// 2. Drift: `x_new = x + h · v½`
    /// 3. Recompute acceleration at new position: `a_new = a(x_new)`
    /// 4. Half-kick: `v_new = v½ + (h/2) · a_new`
    ///
    /// > **Note:** `model.evaluate_derivatives` must return accelerations when the state
    /// > is the positions tensor. For systems where the full state contains `[pos; vel]`
    /// > interleaved, use [`Simulator::rk4_step`] instead.
    ///
    /// # Arguments
    ///
    /// * `model` - A [`Simulatable`] system whose `evaluate_derivatives(positions)` gives accelerations.
    /// * `positions` - The current position tensor.
    /// * `velocities` - The current velocity tensor (same shape as `positions`).
    ///
    /// # Returns
    ///
    /// (`Result<(Tensor, Tensor)>`): `(new_positions, new_velocities)`.
    ///
    /// # Time Complexity
    ///
    /// O(2n) - two derivative evaluations.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::simulation::Simulator;
    /// use lmm::tensor::Tensor;
    /// use lmm::traits::Simulatable;
    ///
    /// struct FreeParticle;
    /// impl Simulatable for FreeParticle {
    ///     fn state(&self) -> &Tensor { unimplemented!() }
    ///     fn evaluate_derivatives(&self, state: &Tensor) -> lmm::error::Result<Tensor> {
    ///         Tensor::new(state.shape.clone(), vec![0.0; state.data.len()])
    ///     }
    /// }
    ///
    /// let sim = Simulator { step_size: 0.01 };
    /// let pos = Tensor::from_vec(vec![1.0]);
    /// let vel = Tensor::from_vec(vec![0.0]);
    /// let (np, nv) = sim.leapfrog_step(&FreeParticle, &pos, &vel).unwrap();
    /// assert_eq!(np.data.len(), 1);
    /// assert_eq!(nv.data.len(), 1);
    /// ```
    pub fn leapfrog_step<M: Simulatable>(
        &self,
        model: &M,
        positions: &Tensor,
        velocities: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let accel_now = model.evaluate_derivatives(positions)?;
        let half_v = (velocities + &accel_now.scale(self.step_size * RK4_HALF))?;

        let new_pos = (positions + &half_v.scale(self.step_size))?;

        let accel_new = model.evaluate_derivatives(&new_pos)?;

        let new_vel = (&half_v + &accel_new.scale(self.step_size * RK4_HALF))?;

        Ok((new_pos, new_vel))
    }

    /// Advances the state by one **adaptive Runge-Kutta-Fehlberg (RK45)** step.
    ///
    /// Uses a 4th-order and 5th-order pair to estimate the local truncation error and
    /// adjust the step size for the next iteration. Returns the 4th-order solution
    /// together with the recommended next step size.
    ///
    /// # Arguments
    ///
    /// * `model` - Any [`Simulatable`] system.
    /// * `current` - The current state tensor.
    /// * `tol` - Error tolerance used to scale the suggested step size.
    ///
    /// # Returns
    ///
    /// (`Result<(Tensor, f64)>`): `(next_state, suggested_h)`.
    ///
    /// # Time Complexity
    ///
    /// O(4n) - four derivative evaluations.
    ///
    /// # Examples
    ///
    /// ```
    /// # use lmm::traits::Simulatable;
    /// use lmm::simulation::Simulator;
    /// use lmm::physics::LorenzSystem;
    ///
    /// let lorenz = LorenzSystem::canonical().unwrap();
    /// let sim = Simulator { step_size: 0.01 };
    /// let (next, h_new) = sim.rk45_step(&lorenz, lorenz.state(), 1e-6).unwrap();
    /// assert!(h_new > 0.0);
    /// ```
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

    /// Simulates a trajectory of `n_steps + 1` states (including the initial state)
    /// using the **RK4** integrator.
    ///
    /// # Arguments
    ///
    /// * `model` - Any [`Simulatable`] system.
    /// * `initial` - The starting state tensor.
    /// * `n_steps` - Number of integration steps to perform.
    ///
    /// # Returns
    ///
    /// (`Result<Vec<Tensor>>`): The ordered list of states from `t=0` to `t=n_steps·h`.
    ///
    /// # Space Complexity
    ///
    /// O(n_steps · n) where n is the state vector dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// # use lmm::traits::Simulatable;
    /// use lmm::simulation::Simulator;
    /// use lmm::physics::HarmonicOscillator;
    ///
    /// let osc = HarmonicOscillator::new(1.0, 1.0, 0.0).unwrap();
    /// let sim = Simulator { step_size: 0.01 };
    /// let traj = sim.simulate_trajectory(&osc, osc.state(), 10).unwrap();
    /// assert_eq!(traj.len(), 11);
    /// ```
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

    /// Simulates an **adaptive** trajectory using RK45 step-size control.
    ///
    /// The step size is adjusted automatically at each iteration to keep the local
    /// truncation error below `tol`.
    ///
    /// # Arguments
    ///
    /// * `model` - Any [`Simulatable`] system.
    /// * `initial` - The starting state tensor.
    /// * `n_steps` - Maximum number of integration steps.
    /// * `tol` - Local error tolerance per step.
    ///
    /// # Returns
    ///
    /// (`Result<Vec<Tensor>>`): States at each accepted step.
    ///
    /// # Examples
    ///
    /// ```
    /// # use lmm::traits::Simulatable;
    /// use lmm::simulation::Simulator;
    /// use lmm::physics::SIRModel;
    ///
    /// let sir = SIRModel::new(0.3, 0.1, 999.0, 1.0, 0.0).unwrap();
    /// let sim = Simulator { step_size: 0.1 };
    /// let traj = sim.simulate_adaptive(&sir, sir.state(), 20, 1e-4).unwrap();
    /// assert!(traj.len() <= 21);
    /// ```
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

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
