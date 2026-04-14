use crate::equation::Expression::{self, *};
use crate::error::Result;
use crate::tensor::Tensor;
use crate::traits::Simulatable;

const GRAVITATIONAL_CONSTANT: f64 = 6.674e-11;

pub struct HarmonicOscillator {
    pub omega: f64,
    pub state: Tensor,
}

impl HarmonicOscillator {
    pub fn new(omega: f64, x0: f64, v0: f64) -> Result<Self> {
        let state = Tensor::new(vec![2], vec![x0, v0])?;
        Ok(Self { omega, state })
    }

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

pub struct DampedOscillator {
    pub omega: f64,
    pub gamma: f64,
    pub state: Tensor,
}

impl DampedOscillator {
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

pub struct LorenzSystem {
    pub sigma: f64,
    pub rho: f64,
    pub beta: f64,
    pub state: Tensor,
}

impl LorenzSystem {
    pub fn new(sigma: f64, rho: f64, beta: f64, x0: f64, y0: f64, z0: f64) -> Result<Self> {
        let state = Tensor::new(vec![3], vec![x0, y0, z0])?;
        Ok(Self {
            sigma,
            rho,
            beta,
            state,
        })
    }

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

pub struct Pendulum {
    pub g: f64,
    pub l: f64,
    pub state: Tensor,
}

impl Pendulum {
    pub fn new(g: f64, l: f64, theta0: f64, omega0: f64) -> Result<Self> {
        let state = Tensor::new(vec![2], vec![theta0, omega0])?;
        Ok(Self { g, l, state })
    }

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
        let d_theta = omega;
        let d_omega = -(self.g / self.l) * theta.sin();
        Tensor::new(vec![2], vec![d_theta, d_omega])
    }
}

pub struct SIRModel {
    pub beta: f64,
    pub gamma: f64,
    pub state: Tensor,
}

impl SIRModel {
    pub fn new(beta: f64, gamma: f64, s0: f64, i0: f64, r0: f64) -> Result<Self> {
        let state = Tensor::new(vec![3], vec![s0, i0, r0])?;
        Ok(Self { beta, gamma, state })
    }

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

pub struct NBodySystem {
    pub masses: Vec<f64>,
    pub state: Tensor,
}

impl NBodySystem {
    pub fn new(masses: Vec<f64>, positions: Vec<f64>, velocities: Vec<f64>) -> Result<Self> {
        let n = masses.len();
        let mut data = positions;
        data.extend(velocities);
        let state = Tensor::new(vec![6 * n], data)?;
        Ok(Self { masses, state })
    }

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
            let mut ax = 0.0;
            let mut ay = 0.0;
            let mut az = 0.0;
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
