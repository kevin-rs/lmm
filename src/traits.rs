use crate::equation::Expression;
use crate::error::Result;
use crate::tensor::Tensor;

pub trait Simulatable {
    fn state(&self) -> &Tensor;
    fn evaluate_derivatives(&self, state: &Tensor) -> Result<Tensor>;
}

pub trait Discoverable {
    fn discover(data: &[Tensor], targets: &[f64]) -> Result<Expression>;
}

pub trait Perceivable {
    fn ingest(raw_data: &[u8]) -> Result<Tensor>;
}

pub trait Predictable {
    fn predict(&self, steps: usize) -> Result<Tensor>;
}

pub trait Encodable {
    fn encode(&self, input: &Tensor) -> Result<Tensor>;
}

pub trait Learnable {
    fn update(&mut self, grad: &Tensor, lr: f64) -> Result<()>;
}

pub trait Causal {
    fn intervene(&mut self, var: &str, value: f64) -> Result<()>;
}
