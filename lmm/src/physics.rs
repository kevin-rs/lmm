// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Physics Models
//!
//! This module provides concrete implementations of the [`Simulatable`] trait for several
//! classical physics systems. Each struct holds the system parameters and current state,
//! while `evaluate_derivatives` returns the time-derivatives needed by the integrators in
//! [`crate::simulation::Simulator`].
//!
//! | System | State Dim | Description |
//! |---|---|---|
//! | [`HarmonicOscillator`] | 2 | Mass-spring: ẍ = -ω²x |
//! | [`DampedOscillator`] | 2 | Damped spring: ẍ = -ω²x - γẋ |
//! | [`LorenzSystem`] | 3 | Lorenz chaos: σ, ρ, β parameterised |
//! | [`Pendulum`] | 2 | Nonlinear pendulum: θ̈ = -(g/l) sin θ |
//! | [`SIRModel`] | 3 | Epidemiological SIR: β, γ parameterised |
//! | [`NBodySystem`] | 6N | N-body gravity under Newton's law |
//!
//! # See Also
//! - [Strogatz, S. H. (2018). Nonlinear Dynamics and Chaos. CRC Press.](https://doi.org/10.1201/9780429492563) - textbook covering the phase space dynamics of the Lorenz system and damped oscillators implemented here.

use crate::equation::Expression::{self, *};
use crate::error::Result;
use crate::tensor::Tensor;
use crate::traits::Simulatable;

/// Newton's gravitational constant G (SI units: m³ kg⁻¹ s⁻²).
const GRAVITATIONAL_CONSTANT: f64 = 6.674e-11;

/// A simple harmonic oscillator with angular frequency `ω`.
///
/// State vector: `[x, ẋ]` where `x` is displacement and `ẋ` is velocity.
///
/// Equations of motion: `ẋ = ẋ`, `ẍ = -ω²x`.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::physics::HarmonicOscillator;
///
/// let osc = HarmonicOscillator::new(1.0, 1.0, 0.0).unwrap();
/// // Initial energy: KE=0, PE=0.5ω²x² = 0.5
/// assert!((osc.energy() - 0.5).abs() < 1e-9);
/// ```
pub struct HarmonicOscillator {
    /// Angular frequency ω (rad/s).
    pub omega: f64,
    /// State `[displacement, velocity]`.
    pub state: Tensor,
}

impl HarmonicOscillator {
    /// Creates a [`HarmonicOscillator`] with angular frequency `omega`.
    ///
    /// # Arguments
    ///
    /// * `omega` - Angular frequency in rad/s.
    /// * `x0` - Initial displacement.
    /// * `v0` - Initial velocity.
    ///
    /// # Returns
    ///
    /// (`Result<HarmonicOscillator>`): The initialised oscillator.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::physics::HarmonicOscillator;
    /// let osc = HarmonicOscillator::new(2.0, 0.5, 0.0).unwrap();
    /// assert_eq!(osc.omega, 2.0);
    /// ```
    pub fn new(omega: f64, x0: f64, v0: f64) -> Result<Self> {
        let state = Tensor::new(vec![2], vec![x0, v0])?;
        Ok(Self { omega, state })
    }

    /// Computes the total mechanical energy: `E = ½ẋ² + ½ω²x²`.
    ///
    /// Energy is conserved for an ideal oscillator (no damping).
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::physics::HarmonicOscillator;
    ///
    /// let osc = HarmonicOscillator::new(1.0, 0.0, 1.0).unwrap();
    /// // All KE: E = ½·1² = 0.5
    /// assert!((osc.energy() - 0.5).abs() < 1e-9);
    /// ```
    pub fn energy(&self) -> f64 {
        let x = self.state.data[0];
        let v = self.state.data[1];
        0.5 * v * v + 0.5 * self.omega * self.omega * x * x
    }
}

impl Simulatable for HarmonicOscillator {
    fn state(&self) -> &Tensor {
        &self.state
    }

    fn evaluate_derivatives(&self, state: &Tensor) -> Result<Tensor> {
        let x = state.data[0];
        let v = state.data[1];
        Tensor::new(vec![2], vec![v, -self.omega * self.omega * x])
    }
}

/// A damped harmonic oscillator with damping coefficient `γ`.
///
/// State vector: `[x, ẋ]`.
///
/// Equations of motion: `ẋ = ẋ`, `ẍ = -ω²x - γẋ`.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::physics::DampedOscillator;
/// use lmm::simulation::Simulator;
///
/// let osc = DampedOscillator::new(1.0, 0.1, 1.0, 0.0).unwrap();
/// let sim = Simulator { step_size: 0.01 };
/// let next = sim.rk4_step(&osc, osc.state()).unwrap();
/// assert_eq!(next.data.len(), 2);
/// ```
pub struct DampedOscillator {
    /// Angular frequency ω (rad/s).
    pub omega: f64,
    /// Damping coefficient γ (s⁻¹).
    pub gamma: f64,
    /// State `[displacement, velocity]`.
    pub state: Tensor,
}

impl DampedOscillator {
    /// Creates a [`DampedOscillator`].
    ///
    /// # Arguments
    ///
    /// * `omega` - Angular frequency in rad/s.
    /// * `gamma` - Damping coefficient (s⁻¹). `gamma = 0` → undamped.
    /// * `x0` - Initial displacement.
    /// * `v0` - Initial velocity.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::physics::DampedOscillator;
    /// let osc = DampedOscillator::new(1.0, 0.5, 0.0, 1.0).unwrap();
    /// assert_eq!(osc.gamma, 0.5);
    /// ```
    pub fn new(omega: f64, gamma: f64, x0: f64, v0: f64) -> Result<Self> {
        let state = Tensor::new(vec![2], vec![x0, v0])?;
        Ok(Self {
            omega,
            gamma,
            state,
        })
    }
}

impl Simulatable for DampedOscillator {
    fn state(&self) -> &Tensor {
        &self.state
    }

    fn evaluate_derivatives(&self, state: &Tensor) -> Result<Tensor> {
        let x = state.data[0];
        let v = state.data[1];
        Tensor::new(
            vec![2],
            vec![v, -self.omega * self.omega * x - self.gamma * v],
        )
    }
}

/// The Lorenz chaotic attractor system.
///
/// State vector: `[x, y, z]`.
///
/// Equations of motion:
/// - `ẋ = σ(y - x)`
/// - `ẏ = x(ρ - z) - y`
/// - `ż = xy - βz`
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::physics::LorenzSystem;
/// use lmm::simulation::Simulator;
///
/// let lorenz = LorenzSystem::canonical().unwrap();
/// let sim = Simulator { step_size: 0.01 };
/// let traj = sim.simulate_trajectory(&lorenz, lorenz.state(), 100).unwrap();
/// assert_eq!(traj.len(), 101);
/// ```
pub struct LorenzSystem {
    /// Prandtl number σ.
    pub sigma: f64,
    /// Rayleigh number ρ.
    pub rho: f64,
    /// Geometric factor β.
    pub beta: f64,
    /// State `[x, y, z]`.
    pub state: Tensor,
}

impl LorenzSystem {
    /// Creates a [`LorenzSystem`] with user-specified parameters and initial conditions.
    ///
    /// # Arguments
    ///
    /// * `sigma` - σ (Prandtl number).
    /// * `rho` - ρ (Rayleigh number).
    /// * `beta` - β (geometric factor).
    /// * `x0`, `y0`, `z0` - Initial state.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::physics::LorenzSystem;
    /// let l = LorenzSystem::new(10.0, 28.0, 8.0/3.0, 0.1, 0.0, 0.0).unwrap();
    /// assert_eq!(l.sigma, 10.0);
    /// ```
    pub fn new(sigma: f64, rho: f64, beta: f64, x0: f64, y0: f64, z0: f64) -> Result<Self> {
        let state = Tensor::new(vec![3], vec![x0, y0, z0])?;
        Ok(Self {
            sigma,
            rho,
            beta,
            state,
        })
    }

    /// Creates a [`LorenzSystem`] with canonical parameters: σ=10, ρ=28, β=8/3.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::physics::LorenzSystem;
    /// let l = LorenzSystem::canonical().unwrap();
    /// assert_eq!(l.rho, 28.0);
    /// ```
    pub fn canonical() -> Result<Self> {
        Self::new(10.0, 28.0, 8.0 / 3.0, 0.1, 0.0, 0.0)
    }
}

impl Simulatable for LorenzSystem {
    fn state(&self) -> &Tensor {
        &self.state
    }

    fn evaluate_derivatives(&self, state: &Tensor) -> Result<Tensor> {
        let x = state.data[0];
        let y = state.data[1];
        let z = state.data[2];
        let dx = self.sigma * (y - x);
        let dy = x * (self.rho - z) - y;
        let dz = x * y - self.beta * z;
        Tensor::new(vec![3], vec![dx, dy, dz])
    }
}

/// A nonlinear pendulum with gravitational acceleration `g` and arm length `l`.
///
/// State vector: `[θ, ω]` where θ is angle (rad) and ω = θ̇.
///
/// Equation of motion: `θ̈ = -(g/l) sin θ`.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::physics::Pendulum;
///
/// let p = Pendulum::new(9.81, 1.0, 0.1, 0.0).unwrap();
/// // Small initial angle → small energy
/// assert!(p.energy() > 0.0);
/// ```
pub struct Pendulum {
    /// Gravitational acceleration g (m/s²).
    pub g: f64,
    /// Arm length l (m).
    pub l: f64,
    /// State `[angle θ, angular velocity ω]`.
    pub state: Tensor,
}

impl Pendulum {
    /// Creates a [`Pendulum`].
    ///
    /// # Arguments
    ///
    /// * `g` - Gravitational acceleration (m/s²).
    /// * `l` - Pendulum length (m).
    /// * `theta0` - Initial angle (rad).
    /// * `omega0` - Initial angular velocity (rad/s).
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::physics::Pendulum;
    /// let p = Pendulum::new(9.81, 1.0, 0.0, 1.0).unwrap();
    /// assert_eq!(p.g, 9.81);
    /// ```
    pub fn new(g: f64, l: f64, theta0: f64, omega0: f64) -> Result<Self> {
        let state = Tensor::new(vec![2], vec![theta0, omega0])?;
        Ok(Self { g, l, state })
    }

    /// Computes total mechanical energy: `E = ½l²ω² + gl(1 - cos θ)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::physics::Pendulum;
    /// let p = Pendulum::new(9.81, 1.0, 0.0, 0.0).unwrap();
    /// assert_eq!(p.energy(), 0.0);
    /// ```
    pub fn energy(&self) -> f64 {
        let theta = self.state.data[0];
        let omega = self.state.data[1];
        let ke = 0.5 * self.l * self.l * omega * omega;
        let pe = self.g * self.l * (1.0 - theta.cos());
        ke + pe
    }
}

impl Simulatable for Pendulum {
    fn state(&self) -> &Tensor {
        &self.state
    }

    fn evaluate_derivatives(&self, state: &Tensor) -> Result<Tensor> {
        let theta = state.data[0];
        let omega = state.data[1];
        Tensor::new(vec![2], vec![omega, -(self.g / self.l) * theta.sin()])
    }
}

/// The SIR epidemiological compartmental model.
///
/// State vector: `[S, I, R]` - Susceptible, Infectious, Recovered population counts.
///
/// Equations (density-normalised by total population N = S + I + R):
/// - `Ṡ = -β·S·I / N`
/// - `İ = β·S·I / N - γ·I`
/// - `Ṙ = γ·I`
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::physics::SIRModel;
/// use lmm::simulation::Simulator;
///
/// let sir = SIRModel::new(0.3, 0.1, 999.0, 1.0, 0.0).unwrap();
/// let sim = Simulator { step_size: 0.5 };
/// let traj = sim.simulate_trajectory(&sir, sir.state(), 10).unwrap();
/// assert_eq!(traj.len(), 11);
/// ```
pub struct SIRModel {
    /// Transmission rate β (infections per contact per day).
    pub beta: f64,
    /// Recovery rate γ (1/day).
    pub gamma: f64,
    /// State `[S, I, R]`.
    pub state: Tensor,
}

impl SIRModel {
    /// Creates a [`SIRModel`].
    ///
    /// # Arguments
    ///
    /// * `beta` - Transmission rate.
    /// * `gamma` - Recovery rate.
    /// * `s0`, `i0`, `r0` - Initial populations.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::physics::SIRModel;
    /// let m = SIRModel::new(0.3, 0.1, 1000.0, 1.0, 0.0).unwrap();
    /// assert!((m.total_population() - 1001.0).abs() < 1e-9);
    /// ```
    pub fn new(beta: f64, gamma: f64, s0: f64, i0: f64, r0: f64) -> Result<Self> {
        let state = Tensor::new(vec![3], vec![s0, i0, r0])?;
        Ok(Self { beta, gamma, state })
    }

    /// Returns the total population S + I + R (should remain constant).
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::physics::SIRModel;
    /// let m = SIRModel::new(0.3, 0.1, 500.0, 50.0, 450.0).unwrap();
    /// assert!((m.total_population() - 1000.0).abs() < 1e-9);
    /// ```
    pub fn total_population(&self) -> f64 {
        self.state.data.iter().sum()
    }
}

impl Simulatable for SIRModel {
    fn state(&self) -> &Tensor {
        &self.state
    }

    fn evaluate_derivatives(&self, state: &Tensor) -> Result<Tensor> {
        let s = state.data[0];
        let i = state.data[1];
        let r = state.data[2];
        let n = s + i + r;
        let n = if n == 0.0 { 1.0 } else { n };
        let infection_rate = self.beta * s * i / n;
        let recovery_rate = self.gamma * i;
        Tensor::new(
            vec![3],
            vec![
                -infection_rate,
                infection_rate - recovery_rate,
                recovery_rate,
            ],
        )
    }
}

/// An N-body gravitational system in 3-D space.
///
/// State vector layout: `[x₀,y₀,z₀, x₁,y₁,z₁, ..., vx₀,vy₀,vz₀, vx₁,vy₁,vz₁, ...]`
///
/// Equations of motion follow Newton's law of gravity with a softening floor of
/// `r_min = 1e-10` to avoid singularities at zero separation.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::physics::NBodySystem;
///
/// let masses = vec![1e30, 1e30];
/// let positions = vec![0.0, 0.0, 0.0, 1e11, 0.0, 0.0];
/// let velocities = vec![0.0; 6];
/// let sys = NBodySystem::new(masses, positions, velocities).unwrap();
/// assert_eq!(sys.state.shape, vec![12]);
/// ```
pub struct NBodySystem {
    /// Masses of each body in kg.
    pub masses: Vec<f64>,
    /// Interleaved positions then velocities in 3-D.
    pub state: Tensor,
}

impl NBodySystem {
    /// Creates an [`NBodySystem`] from masses, position vector, and velocity vector.
    ///
    /// Positions and velocities must each have exactly `3 * masses.len()` elements.
    ///
    /// # Arguments
    ///
    /// * `masses` - Mass of each body.
    /// * `positions` - Flat position vector `[x₀,y₀,z₀, x₁, ...]`.
    /// * `velocities` - Flat velocity vector `[vx₀,vy₀,vz₀, vx₁, ...]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::physics::NBodySystem;
    /// let sys = NBodySystem::new(
    ///     vec![1.0],
    ///     vec![0.0, 0.0, 0.0],
    ///     vec![1.0, 0.0, 0.0],
    /// ).unwrap();
    /// assert_eq!(sys.masses[0], 1.0);
    /// ```
    pub fn new(masses: Vec<f64>, positions: Vec<f64>, velocities: Vec<f64>) -> Result<Self> {
        let n = masses.len();
        let mut data = positions;
        data.extend(velocities);
        let state = Tensor::new(vec![6 * n], data)?;
        Ok(Self { masses, state })
    }

    /// Returns the symbolic gravitational force expression: `-G·m₁·m₂ / r²`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::physics::NBodySystem;
    /// let expr = NBodySystem::gravitational_equation();
    /// assert_eq!(expr.to_string(), "(-(G * (m1 * (m2 / (r ^ 2)))))");
    /// ```
    pub fn gravitational_equation() -> Expression {
        Neg(Box::new(Mul(
            Box::new(Variable("G".into())),
            Box::new(Mul(
                Box::new(Variable("m1".into())),
                Box::new(Div(
                    Box::new(Variable("m2".into())),
                    Box::new(Pow(Box::new(Variable("r".into())), Box::new(Constant(2.0)))),
                )),
            )),
        )))
    }
}

impl Simulatable for NBodySystem {
    fn state(&self) -> &Tensor {
        &self.state
    }

    /// Evaluates Newtonian gravitational accelerations for all bodies.
    ///
    /// Uses a softening parameter of `r_min = 1e-10` to prevent division by zero.
    ///
    /// # Time Complexity
    ///
    /// O(N²) where N is the number of bodies.
    fn evaluate_derivatives(&self, state: &Tensor) -> Result<Tensor> {
        let n = self.masses.len();
        let mut derivs = vec![0.0; 6 * n];

        for i in 0..n {
            derivs[3 * i] = state.data[3 * n + 3 * i];
            derivs[3 * i + 1] = state.data[3 * n + 3 * i + 1];
            derivs[3 * i + 2] = state.data[3 * n + 3 * i + 2];
        }

        for i in 0..n {
            let xi = state.data[3 * i];
            let yi = state.data[3 * i + 1];
            let zi = state.data[3 * i + 2];
            let (mut ax, mut ay, mut az) = (0.0, 0.0, 0.0);
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dx = state.data[3 * j] - xi;
                let dy = state.data[3 * j + 1] - yi;
                let dz = state.data[3 * j + 2] - zi;
                let r2 = dx * dx + dy * dy + dz * dz;
                let r = r2.sqrt().max(1e-10);
                let force = GRAVITATIONAL_CONSTANT * self.masses[j] / (r2 * r);
                ax += force * dx;
                ay += force * dy;
                az += force * dz;
            }
            derivs[3 * n + 3 * i] = ax;
            derivs[3 * n + 3 * i + 1] = ay;
            derivs[3 * n + 3 * i + 2] = az;
        }

        Tensor::new(vec![6 * n], derivs)
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
