use crate::error::Result;
use crate::tensor::Tensor;

pub trait Simulatable {
    fn state(&self) -> &Tensor;
    fn evaluate_derivatives(&self, state: &Tensor) -> Result<Tensor>;
}

pub trait Discoverable {
    fn discover(data: &[Tensor]) -> Result<crate::equation::Expression>;
}

pub trait Perceivable {
    fn ingest(raw_data: &[u8]) -> Result<Tensor>;
}

pub trait Predictable {
    fn predict(&self, steps: usize) -> Result<Tensor>;
}
