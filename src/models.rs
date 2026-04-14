use crate::error::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    pub data: Vec<f64>,
}

impl Vector {
    pub fn new(data: Vec<f64>) -> Self {
        Self { data }
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    pub fn add(&self, other: &Self) -> Self {
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Self { data }
    }

    pub fn scale(&self, factor: f64) -> Self {
        Self {
            data: self.data.iter().map(|x| x * factor).collect(),
        }
    }

    pub fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct State {
    pub time: f64,
    pub variables: Vector,
}

pub trait MathematicalModel {
    fn evaluate(&self, state: &State) -> Result<Vector>;
}

#[derive(Debug, Clone)]
pub struct LinearModel {
    pub weights: Vector,
    pub bias: f64,
}

impl LinearModel {
    pub fn new(weights: Vec<f64>, bias: f64) -> Self {
        Self {
            weights: Vector::new(weights),
            bias,
        }
    }
}

impl MathematicalModel for LinearModel {
    fn evaluate(&self, state: &State) -> Result<Vector> {
        let prediction = self.weights.dot(&state.variables) + self.bias;
        Ok(Vector::new(vec![prediction]))
    }
}

#[derive(Debug, Clone)]
pub struct PolynomialModel {
    pub coeffs: Vec<f64>,
}

impl PolynomialModel {
    pub fn new(coeffs: Vec<f64>) -> Self {
        Self { coeffs }
    }

    pub fn evaluate_at(&self, x: f64) -> f64 {
        self.coeffs
            .iter()
            .enumerate()
            .map(|(i, &c)| c * x.powi(i as i32))
            .sum()
    }
}

impl MathematicalModel for PolynomialModel {
    fn evaluate(&self, state: &State) -> Result<Vector> {
        let x = state.variables.data.first().copied().unwrap_or(0.0);
        Ok(Vector::new(vec![self.evaluate_at(x)]))
    }
}

#[derive(Debug, Clone)]
pub struct FitResult {
    pub mse: f64,
    pub r_squared: f64,
}

impl FitResult {
    pub fn new(predictions: &[f64], targets: &[f64]) -> Self {
        if targets.is_empty() {
            return Self {
                mse: 0.0,
                r_squared: 0.0,
            };
        }
        let n = targets.len() as f64;
        let mse = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / n;
        let mean_t: f64 = targets.iter().sum::<f64>() / n;
        let ss_tot: f64 = targets.iter().map(|t| (t - mean_t).powi(2)).sum();
        let ss_res: f64 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum();
        let r_squared = if ss_tot == 0.0 {
            1.0
        } else {
            1.0 - ss_res / ss_tot
        };
        Self { mse, r_squared }
    }
}
