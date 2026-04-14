use crate::error::LmmError::Simulation;
use crate::error::Result;
use rand::RngExt;
use std::ops::{Add, Mul, Sub};

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f64>) -> Result<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(Simulation(format!(
                "Tensor shape mismatch: expected {} elements but got {}",
                expected_len,
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; len],
        }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        Self {
            shape,
            data: vec![1.0; len],
        }
    }

    pub fn fill(shape: Vec<usize>, value: f64) -> Self {
        let len: usize = shape.iter().product();
        Self {
            shape,
            data: vec![value; len],
        }
    }

    pub fn randn(shape: Vec<usize>, mean: f64, std: f64) -> Self {
        let len: usize = shape.iter().product();
        let mut rng = rand::rng();
        let data: Vec<f64> = (0..len)
            .map(|_| {
                let u1: f64 = rng.random::<f64>().max(1e-10);
                let u2: f64 = rng.random();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                mean + std * z
            })
            .collect();
        Self { shape, data }
    }

    pub fn from_vec(data: Vec<f64>) -> Self {
        let len = data.len();
        Self {
            shape: vec![len],
            data,
        }
    }

    pub fn scale(&self, factor: f64) -> Self {
        Self {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&x| x * factor).collect(),
        }
    }

    pub fn map<F: Fn(f64) -> f64>(&self, f: F) -> Self {
        Self {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&x| f(x)).collect(),
        }
    }

    pub fn zip_map<F: Fn(f64, f64) -> f64>(&self, other: &Self, f: F) -> Result<Self> {
        if self.shape != other.shape {
            return Err(Simulation("zip_map shape mismatch".into()));
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| f(a, b))
            .collect();
        Ok(Self {
            shape: self.shape.clone(),
            data,
        })
    }

    pub fn dot(&self, other: &Self) -> Result<f64> {
        if self.shape != other.shape {
            return Err(Simulation("dot product shape mismatch".into()));
        }
        Ok(self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum())
    }

    pub fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    pub fn mean(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().sum::<f64>() / self.data.len() as f64
    }

    pub fn variance(&self) -> f64 {
        let m = self.mean();
        self.data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / self.data.len() as f64
    }

    pub fn argmax(&self) -> usize {
        self.data.iter().enumerate().fold(
            0,
            |best, (i, &v)| {
                if v > self.data[best] { i } else { best }
            },
        )
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self> {
        let expected: usize = new_shape.iter().product();
        if expected != self.data.len() {
            return Err(Simulation(format!(
                "Cannot reshape {} elements into shape {:?}",
                self.data.len(),
                new_shape
            )));
        }
        Ok(Self {
            shape: new_shape,
            data: self.data.clone(),
        })
    }

    pub fn transpose(&self) -> Result<Self> {
        if self.shape.len() != 2 {
            return Err(Simulation("transpose requires a 2-D tensor".into()));
        }
        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut data = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                data[c * rows + r] = self.data[r * cols + c];
            }
        }
        Ok(Self {
            shape: vec![cols, rows],
            data,
        })
    }

    pub fn matmul(&self, other: &Self) -> Result<Self> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(Simulation("matmul requires 2-D tensors".into()));
        }
        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        if k != k2 {
            return Err(Simulation(format!(
                "matmul dimension mismatch: [{m}x{k}] vs [{k2}x{n}]"
            )));
        }
        let mut data = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += self.data[i * k + l] * other.data[l * n + j];
                }
                data[i * n + j] = sum;
            }
        }
        Ok(Self {
            shape: vec![m, n],
            data,
        })
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Add for &Tensor {
    type Output = Result<Tensor>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(Simulation(format!(
                "Tensor add shape mismatch: {:?} vs {:?}",
                self.shape, rhs.shape
            )));
        }
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Ok(Tensor {
            shape: self.shape.clone(),
            data,
        })
    }
}

impl Sub for &Tensor {
    type Output = Result<Tensor>;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(Simulation(format!(
                "Tensor sub shape mismatch: {:?} vs {:?}",
                self.shape, rhs.shape
            )));
        }
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Ok(Tensor {
            shape: self.shape.clone(),
            data,
        })
    }
}

impl Mul for &Tensor {
    type Output = Result<Tensor>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(Simulation(format!(
                "Tensor mul shape mismatch: {:?} vs {:?}",
                self.shape, rhs.shape
            )));
        }
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Ok(Tensor {
            shape: self.shape.clone(),
            data,
        })
    }
}
