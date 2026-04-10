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
}

#[derive(Debug, Clone, PartialEq)]
pub struct State {
    pub time: f64,
    pub variables: Vector,
}

pub trait MathematicalModel {
    fn evaluate(&self, state: &State) -> Result<Vector>;
}
