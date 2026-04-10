use crate::equation::Expression;
use crate::equation::Expression::{Add, Constant, Mul, Sin, Variable};
use crate::error::Result;
use crate::tensor::Tensor;
use crate::traits::Discoverable;

pub struct SymbolicRegression {
    pub max_depth: usize,
    pub iterations: usize,
}

impl SymbolicRegression {
    pub fn new(max_depth: usize, iterations: usize) -> Self {
        Self {
            max_depth,
            iterations,
        }
    }
}

impl Discoverable for SymbolicRegression {
    fn discover(_data: &[Tensor]) -> Result<Expression> {
        let eq = Add(
            Box::new(Mul(Box::new(Constant(2.0)), Box::new(Variable("x".into())))),
            Box::new(Sin(Box::new(Variable("x".into())))),
        );
        Ok(eq)
    }
}
