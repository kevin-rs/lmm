use crate::error::LmmError::InvalidExpression;
use crate::error::Result;
use Expression::{Add, Constant, Cos, Div, Mul, Sin, Sub, Variable};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Constant(f64),
    Variable(String),
    Add(Box<Expression>, Box<Expression>),
    Sub(Box<Expression>, Box<Expression>),
    Mul(Box<Expression>, Box<Expression>),
    Div(Box<Expression>, Box<Expression>),
    Sin(Box<Expression>),
    Cos(Box<Expression>),
}

impl Expression {
    pub fn evaluate(&self, bindings: &HashMap<String, f64>) -> Result<f64> {
        match self {
            Constant(c) => Ok(*c),
            Variable(name) => bindings.get(name).copied().ok_or(InvalidExpression),
            Add(lhs, rhs) => Ok(lhs.evaluate(bindings)? + rhs.evaluate(bindings)?),
            Sub(lhs, rhs) => Ok(lhs.evaluate(bindings)? - rhs.evaluate(bindings)?),
            Mul(lhs, rhs) => Ok(lhs.evaluate(bindings)? * rhs.evaluate(bindings)?),
            Div(lhs, rhs) => {
                let r = rhs.evaluate(bindings)?;
                if r == 0.0 {
                    return Err(InvalidExpression);
                }
                Ok(lhs.evaluate(bindings)? / r)
            }
            Sin(expr) => Ok(expr.evaluate(bindings)?.sin()),
            Cos(expr) => Ok(expr.evaluate(bindings)?.cos()),
        }
    }

    pub fn complexity(&self) -> usize {
        match self {
            Constant(_) | Variable(_) => 1,
            Sin(e) | Cos(e) => 1 + e.complexity(),
            Add(l, r) | Sub(l, r) | Mul(l, r) | Div(l, r) => 1 + l.complexity() + r.complexity(),
        }
    }
}
