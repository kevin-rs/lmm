use crate::error::LmmError::Simulation;
use crate::error::Result;
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
            return Err(Simulation("Tensor shape mismatch".into()));
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

    pub fn scale(&self, factor: f64) -> Self {
        Self {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&x| x * factor).collect(),
        }
    }
}

impl Add for &Tensor {
    type Output = Result<Tensor>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(Simulation("Tensor add shape mismatch".into()));
        }
        let data: Vec<f64> = self
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
            return Err(Simulation("Tensor sub shape mismatch".into()));
        }
        let data: Vec<f64> = self
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
            return Err(Simulation("Tensor mul shape mismatch".into()));
        }
        let data: Vec<f64> = self
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
