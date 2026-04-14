use crate::error::LmmError::{DivisionByZero, InvalidExpression};
use crate::error::Result;
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Constant(f64),
    Variable(String),
    Add(Box<Expression>, Box<Expression>),
    Sub(Box<Expression>, Box<Expression>),
    Mul(Box<Expression>, Box<Expression>),
    Div(Box<Expression>, Box<Expression>),
    Pow(Box<Expression>, Box<Expression>),
    Neg(Box<Expression>),
    Abs(Box<Expression>),
    Sin(Box<Expression>),
    Cos(Box<Expression>),
    Exp(Box<Expression>),
    Log(Box<Expression>),
}

impl Expression {
    pub fn evaluate(&self, bindings: &HashMap<String, f64>) -> Result<f64> {
        match self {
            Self::Constant(c) => Ok(*c),
            Self::Variable(name) => bindings.get(name).copied().ok_or(InvalidExpression),
            Self::Add(lhs, rhs) => Ok(lhs.evaluate(bindings)? + rhs.evaluate(bindings)?),
            Self::Sub(lhs, rhs) => Ok(lhs.evaluate(bindings)? - rhs.evaluate(bindings)?),
            Self::Mul(lhs, rhs) => Ok(lhs.evaluate(bindings)? * rhs.evaluate(bindings)?),
            Self::Div(lhs, rhs) => {
                let r = rhs.evaluate(bindings)?;
                if r == 0.0 {
                    return Err(DivisionByZero);
                }
                Ok(lhs.evaluate(bindings)? / r)
            }
            Self::Pow(base, exp) => Ok(base.evaluate(bindings)?.powf(exp.evaluate(bindings)?)),
            Self::Neg(e) => Ok(-e.evaluate(bindings)?),
            Self::Abs(e) => Ok(e.evaluate(bindings)?.abs()),
            Self::Sin(e) => Ok(e.evaluate(bindings)?.sin()),
            Self::Cos(e) => Ok(e.evaluate(bindings)?.cos()),
            Self::Exp(e) => Ok(e.evaluate(bindings)?.exp()),
            Self::Log(e) => {
                let v = e.evaluate(bindings)?;
                if v <= 0.0 {
                    return Err(InvalidExpression);
                }
                Ok(v.ln())
            }
        }
    }

    pub fn complexity(&self) -> usize {
        match self {
            Self::Constant(_) | Self::Variable(_) => 1,
            Self::Neg(e)
            | Self::Abs(e)
            | Self::Sin(e)
            | Self::Cos(e)
            | Self::Exp(e)
            | Self::Log(e) => 1 + e.complexity(),
            Self::Add(l, r)
            | Self::Sub(l, r)
            | Self::Mul(l, r)
            | Self::Div(l, r)
            | Self::Pow(l, r) => 1 + l.complexity() + r.complexity(),
        }
    }

    pub fn symbolic_diff(&self, var: &str) -> Expression {
        match self {
            Self::Constant(_) => Self::Constant(0.0),
            Self::Variable(name) => {
                if name == var {
                    Self::Constant(1.0)
                } else {
                    Self::Constant(0.0)
                }
            }
            Self::Add(l, r) => Self::Add(
                Box::new(l.symbolic_diff(var)),
                Box::new(r.symbolic_diff(var)),
            ),
            Self::Sub(l, r) => Self::Sub(
                Box::new(l.symbolic_diff(var)),
                Box::new(r.symbolic_diff(var)),
            ),
            Self::Mul(l, r) => Self::Add(
                Box::new(Self::Mul(Box::new(l.symbolic_diff(var)), r.clone())),
                Box::new(Self::Mul(l.clone(), Box::new(r.symbolic_diff(var)))),
            ),
            Self::Div(l, r) => Self::Div(
                Box::new(Self::Sub(
                    Box::new(Self::Mul(Box::new(l.symbolic_diff(var)), r.clone())),
                    Box::new(Self::Mul(l.clone(), Box::new(r.symbolic_diff(var)))),
                )),
                Box::new(Self::Pow(r.clone(), Box::new(Self::Constant(2.0)))),
            ),
            Self::Pow(base, exp) => Self::Mul(
                Box::new(Self::Mul(
                    exp.clone(),
                    Box::new(Self::Pow(
                        base.clone(),
                        Box::new(Self::Sub(exp.clone(), Box::new(Self::Constant(1.0)))),
                    )),
                )),
                Box::new(base.symbolic_diff(var)),
            ),
            Self::Neg(e) => Self::Neg(Box::new(e.symbolic_diff(var))),
            Self::Abs(e) => Self::Mul(
                Box::new(Self::Div(
                    Box::new(*e.clone()),
                    Box::new(Self::Abs(e.clone())),
                )),
                Box::new(e.symbolic_diff(var)),
            ),
            Self::Sin(e) => Self::Mul(
                Box::new(Self::Cos(e.clone())),
                Box::new(e.symbolic_diff(var)),
            ),
            Self::Cos(e) => Self::Neg(Box::new(Self::Mul(
                Box::new(Self::Sin(e.clone())),
                Box::new(e.symbolic_diff(var)),
            ))),
            Self::Exp(e) => Self::Mul(
                Box::new(Self::Exp(e.clone())),
                Box::new(e.symbolic_diff(var)),
            ),
            Self::Log(e) => Self::Mul(
                Box::new(Self::Div(
                    Box::new(Self::Constant(1.0)),
                    Box::new(*e.clone()),
                )),
                Box::new(e.symbolic_diff(var)),
            ),
        }
    }

    pub fn simplify(&self) -> Expression {
        match self {
            Self::Add(l, r) => {
                let l = l.simplify();
                let r = r.simplify();
                match (&l, &r) {
                    (Self::Constant(a), Self::Constant(b)) => Self::Constant(a + b),
                    (Self::Constant(0.0), _) => r,
                    (_, Self::Constant(0.0)) => l,
                    _ => Self::Add(Box::new(l), Box::new(r)),
                }
            }
            Self::Sub(l, r) => {
                let l = l.simplify();
                let r = r.simplify();
                match (&l, &r) {
                    (Self::Constant(a), Self::Constant(b)) => Self::Constant(a - b),
                    (_, Self::Constant(0.0)) => l,
                    _ if l == r => Self::Constant(0.0),
                    _ => Self::Sub(Box::new(l), Box::new(r)),
                }
            }
            Self::Mul(l, r) => {
                let l = l.simplify();
                let r = r.simplify();
                match (&l, &r) {
                    (Self::Constant(a), Self::Constant(b)) => Self::Constant(a * b),
                    (Self::Constant(0.0), _) | (_, Self::Constant(0.0)) => Self::Constant(0.0),
                    (Self::Constant(1.0), _) => r,
                    (_, Self::Constant(1.0)) => l,
                    _ => Self::Mul(Box::new(l), Box::new(r)),
                }
            }
            Self::Div(l, r) => {
                let l = l.simplify();
                let r = r.simplify();
                match (&l, &r) {
                    (Self::Constant(a), Self::Constant(b)) if *b != 0.0 => Self::Constant(a / b),
                    (_, Self::Constant(1.0)) => l,
                    _ if l == r => Self::Constant(1.0),
                    _ => Self::Div(Box::new(l), Box::new(r)),
                }
            }
            Self::Pow(base, exp) => {
                let base = base.simplify();
                let exp = exp.simplify();
                match (&base, &exp) {
                    (Self::Constant(a), Self::Constant(b)) => Self::Constant(a.powf(*b)),
                    (_, Self::Constant(0.0)) => Self::Constant(1.0),
                    (_, Self::Constant(1.0)) => base,
                    (Self::Constant(1.0), _) => Self::Constant(1.0),
                    _ => Self::Pow(Box::new(base), Box::new(exp)),
                }
            }
            Self::Neg(e) => {
                let e = e.simplify();
                match &e {
                    Self::Constant(v) => Self::Constant(-v),
                    _ => Self::Neg(Box::new(e)),
                }
            }
            Self::Abs(e) => {
                let e = e.simplify();
                match &e {
                    Self::Constant(v) => Self::Constant(v.abs()),
                    _ => Self::Abs(Box::new(e)),
                }
            }
            Self::Sin(e) => {
                let e = e.simplify();
                match &e {
                    Self::Constant(v) => Self::Constant(v.sin()),
                    _ => Self::Sin(Box::new(e)),
                }
            }
            Self::Cos(e) => {
                let e = e.simplify();
                match &e {
                    Self::Constant(v) => Self::Constant(v.cos()),
                    _ => Self::Cos(Box::new(e)),
                }
            }
            Self::Exp(e) => {
                let e = e.simplify();
                match &e {
                    Self::Constant(v) => Self::Constant(v.exp()),
                    _ => Self::Exp(Box::new(e)),
                }
            }
            Self::Log(e) => {
                let e = e.simplify();
                match &e {
                    Self::Constant(v) if *v > 0.0 => Self::Constant(v.ln()),
                    _ => Self::Log(Box::new(e)),
                }
            }
            other => other.clone(),
        }
    }

    pub fn variables(&self) -> Vec<String> {
        let mut vars = Vec::new();
        self.collect_variables(&mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    fn collect_variables(&self, vars: &mut Vec<String>) {
        match self {
            Self::Variable(name) => vars.push(name.clone()),
            Self::Constant(_) => {}
            Self::Neg(e)
            | Self::Abs(e)
            | Self::Sin(e)
            | Self::Cos(e)
            | Self::Exp(e)
            | Self::Log(e) => e.collect_variables(vars),
            Self::Add(l, r)
            | Self::Sub(l, r)
            | Self::Mul(l, r)
            | Self::Div(l, r)
            | Self::Pow(l, r) => {
                l.collect_variables(vars);
                r.collect_variables(vars);
            }
        }
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Constant(c) => write!(f, "{}", c),
            Self::Variable(name) => write!(f, "{}", name),
            Self::Add(l, r) => write!(f, "({} + {})", l, r),
            Self::Sub(l, r) => write!(f, "({} - {})", l, r),
            Self::Mul(l, r) => write!(f, "({} * {})", l, r),
            Self::Div(l, r) => write!(f, "({} / {})", l, r),
            Self::Pow(base, exp) => write!(f, "({})^({})", base, exp),
            Self::Neg(e) => write!(f, "(-{})", e),
            Self::Abs(e) => write!(f, "|{}|", e),
            Self::Sin(e) => write!(f, "sin({})", e),
            Self::Cos(e) => write!(f, "cos({})", e),
            Self::Exp(e) => write!(f, "exp({})", e),
            Self::Log(e) => write!(f, "ln({})", e),
        }
    }
}

struct Parser<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            input: s.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn consume(&mut self) -> Option<u8> {
        let ch = self.input.get(self.pos).copied();
        if ch.is_some() {
            self.pos += 1;
        }
        ch
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\r' | b'\n')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, ch: u8) -> std::result::Result<(), String> {
        self.skip_ws();
        match self.consume() {
            Some(c) if c == ch => Ok(()),
            other => Err(format!(
                "expected {:?} got {:?} at pos {}",
                ch as char,
                other.map(|b| b as char),
                self.pos
            )),
        }
    }

    fn parse_expr(&mut self) -> std::result::Result<Expression, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'(') => {
                self.consume();
                self.skip_ws();
                if self.peek() == Some(b'-') {
                    self.consume();
                    let inner = self.parse_expr()?;
                    self.skip_ws();
                    self.expect(b')')?;
                    return Ok(Expression::Neg(Box::new(inner)));
                }
                let lhs = self.parse_expr()?;
                self.skip_ws();
                let op = self.consume().ok_or("expected operator")?;
                self.skip_ws();
                let rhs = self.parse_expr()?;
                self.skip_ws();
                self.expect(b')')?;
                self.skip_ws();
                if self.peek() == Some(b'^') {
                    self.consume();
                    self.skip_ws();
                    let exp = self.parse_expr()?;
                    return Ok(Expression::Pow(Box::new(lhs), Box::new(exp)));
                }
                match op {
                    b'+' => Ok(Expression::Add(Box::new(lhs), Box::new(rhs))),
                    b'-' => Ok(Expression::Sub(Box::new(lhs), Box::new(rhs))),
                    b'*' => Ok(Expression::Mul(Box::new(lhs), Box::new(rhs))),
                    b'/' => Ok(Expression::Div(Box::new(lhs), Box::new(rhs))),
                    _ => Err(format!("unknown operator {:?}", op as char)),
                }
            }
            Some(b'|') => {
                self.consume();
                let inner = self.parse_expr()?;
                self.expect(b'|')?;
                Ok(Expression::Abs(Box::new(inner)))
            }
            Some(b's' | b'c' | b'e' | b'l') => {
                let start = self.pos;
                while matches!(self.peek(), Some(b'a'..=b'z')) {
                    self.pos += 1;
                }
                let name = std::str::from_utf8(&self.input[start..self.pos]).unwrap();
                self.skip_ws();
                self.expect(b'(')?;
                let arg = self.parse_expr()?;
                self.expect(b')')?;
                match name {
                    "sin" => Ok(Expression::Sin(Box::new(arg))),
                    "cos" => Ok(Expression::Cos(Box::new(arg))),
                    "exp" => Ok(Expression::Exp(Box::new(arg))),
                    "ln" => Ok(Expression::Log(Box::new(arg))),
                    _ => Err(format!("unknown function: {name}")),
                }
            }
            Some(b'0'..=b'9' | b'-') => {
                let start = self.pos;
                if self.peek() == Some(b'-') {
                    self.pos += 1;
                }
                while matches!(self.peek(), Some(b'0'..=b'9' | b'.' | b'e' | b'E' | b'+')) {
                    self.pos += 1;
                }
                let s =
                    std::str::from_utf8(&self.input[start..self.pos]).map_err(|e| e.to_string())?;
                let v: f64 = s
                    .parse()
                    .map_err(|e: std::num::ParseFloatError| e.to_string())?;
                Ok(Expression::Constant(v))
            }
            Some(b'a'..=b'z' | b'A'..=b'Z') => {
                let start = self.pos;
                while matches!(
                    self.peek(),
                    Some(b'a'..=b'z' | b'A'..=b'Z' | b'_' | b'0'..=b'9')
                ) {
                    self.pos += 1;
                }
                let name = std::str::from_utf8(&self.input[start..self.pos])
                    .unwrap()
                    .to_string();
                Ok(Expression::Variable(name))
            }
            other => Err(format!(
                "unexpected char {:?} at pos {}",
                other.map(|b| b as char),
                self.pos
            )),
        }
    }
}

impl std::str::FromStr for Expression {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let mut p = Parser::new(s.trim());
        let expr = p.parse_expr()?;
        Ok(expr)
    }
}
