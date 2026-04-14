use crate::compression::compute_mse;
use crate::discovery::SymbolicRegression;
use crate::equation::Expression;
use crate::error::{LmmError, Result};
use std::collections::HashMap;

#[derive(Debug)]
pub struct EncodedMessage {
    pub equation: Expression,
    pub length: usize,
    pub mse: f64,
    pub residuals: Vec<i32>,
}

impl EncodedMessage {
    pub fn summary(&self) -> String {
        format!(
            "Equation: {}\nLength: {} chars\nMSE: {:.4}\nMax residual: {}",
            self.equation,
            self.length,
            self.mse,
            self.residuals.iter().map(|r| r.abs()).max().unwrap_or(0),
        )
    }

    pub fn to_data_string(&self) -> String {
        let res_str = self
            .residuals
            .iter()
            .map(|r| r.to_string())
            .collect::<Vec<_>>()
            .join(",");
        format!(
            r#"{{"eq":"{}","len":{},"mse":{:.6},"res":[{}]}}"#,
            self.equation, self.length, self.mse, res_str
        )
    }
}

pub fn encode_text(text: &str, iterations: usize, depth: usize) -> Result<EncodedMessage> {
    if text.is_empty() {
        return Err(LmmError::Perception("Cannot encode empty text".into()));
    }

    let bytes: Vec<u8> = text.bytes().collect();
    let n = bytes.len();

    let inputs: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
    let targets: Vec<f64> = bytes.iter().map(|&b| f64::from(b)).collect();

    let sr = SymbolicRegression::new(depth, iterations)
        .with_variables(vec!["x".into()])
        .with_population(60);

    let equation = sr.fit(&inputs, &targets)?;
    let mse = compute_mse(&equation, &inputs, &targets);

    let residuals: Vec<i32> = (0..n)
        .map(|i| {
            let mut vars = HashMap::new();
            vars.insert("x".to_string(), i as f64);
            let predicted = equation.evaluate(&vars).unwrap_or(0.0);
            bytes[i] as i32 - predicted.round() as i32
        })
        .collect();

    Ok(EncodedMessage {
        equation,
        length: n,
        mse,
        residuals,
    })
}

pub fn decode_message(msg: &EncodedMessage) -> Result<String> {
    let mut bytes = Vec::with_capacity(msg.length);
    for i in 0..msg.length {
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), i as f64);
        let predicted = msg.equation.evaluate(&vars).unwrap_or(0.0).round() as i32;
        let byte_val = (predicted + msg.residuals[i]).clamp(0, 255) as u8;
        bytes.push(byte_val);
    }
    String::from_utf8(bytes).map_err(|e| LmmError::Perception(e.to_string()))
}

pub fn decode_from_parts(
    equation: &Expression,
    length: usize,
    residuals: &[i32],
) -> Result<String> {
    if residuals.len() != length {
        return Err(LmmError::Perception(format!(
            "Residual length mismatch: expected {length}, got {}",
            residuals.len()
        )));
    }
    let msg = EncodedMessage {
        equation: equation.clone(),
        length,
        mse: 0.0,
        residuals: residuals.to_vec(),
    };
    decode_message(&msg)
}
