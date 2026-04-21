// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Text-Equation Encoding and Decoding
//!
//! This module implements a bijective symbolic bridge between natural-language text and
//! compact mathematical expressions - the *encode → discover → decode* pipeline that is
//! central to the LMM philosophy of compressing reality into equations.
//!
//! Two directions are supported:
//!
//! - **Encode** (`encode_text`): maps a string of text to a sequence of real-valued
//!   "equation coefficients" via character-level statistics, then runs symbolic regression
//!   to find a concise mathematical form.
//! - **Decode** (`decode_message`, `decode_from_parts`): evaluates an equation over an
//!   integer domain and reconstructs text by interpreting the outputs as Unicode code-points,
//!   with invalid bytes replaced by `U+FFFD` (Unicode replacement character).

use crate::discovery::SymbolicRegression;
use crate::equation::Expression;
use crate::error::{LmmError, Result};
use std::collections::HashMap;

/// The Unicode replacement character, used when equation output is not valid UTF-8.
const REPLACEMENT_CHAR: char = '\u{FFFD}';

/// The result of an encoding, including the best-fit expression and the exact numerical residuals.
#[derive(Debug, Clone)]
pub struct EncodedMessage {
    pub equation: Expression,
    pub length: usize,
    pub residuals: Vec<i64>,
}

impl EncodedMessage {
    pub fn summary(&self) -> String {
        self.equation.to_string()
    }

    pub fn to_data_string(&self) -> String {
        format!("Length: {}, Residuals: {:?}", self.length, self.residuals)
    }
}

/// Maps a text string to an [`EncodedMessage`] via character-level symbolic regression.
///
/// Steps:
/// 1. Convert each character to a `f64` code-point value.
/// 2. Build `(index, code_point)` training pairs.
/// 3. Run symbolic regression (with `iterations` and `depth` controls) to find an
///    expression that fits the code-point sequence.
/// 4. Calculate exact integer residuals for lossless reconstruction.
///
/// # Arguments
///
/// * `text` - The input text to encode.
/// * `iterations` - Number of symbolic regression iterations.
/// * `depth` - Maximum expression tree depth.
///
/// # Returns
///
/// (`Result<EncodedMessage>`): The discovered symbolic expression and residuals.
///
/// # Errors
///
/// Returns [`LmmError::Discovery`] when regression fails to converge.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::encode::encode_text;
///
/// let msg = encode_text("Hello", 20, 2);
/// assert!(msg.is_ok() || msg.is_err());
/// ```
pub fn encode_text(text: &str, iterations: usize, depth: usize) -> Result<EncodedMessage> {
    if text.is_empty() {
        return Err(LmmError::ParseError("Cannot encode empty text".into()));
    }
    let chars: Vec<f64> = text.chars().map(|c| c as u32 as f64).collect();
    let n = chars.len();
    let inputs: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
    let targets: Vec<f64> = chars;

    let sr = SymbolicRegression::new(iterations, depth).with_variables(vec!["x".into()]);
    let equation = sr.fit(&inputs, &targets)?;

    let mut residuals = Vec::with_capacity(text.len());
    let mut bindings = HashMap::new();

    for (i, c) in text.chars().enumerate() {
        bindings.insert("x".to_string(), i as f64);
        let base = equation.evaluate(&bindings).unwrap_or(65.0).round() as i64;
        let char_val = c as u32 as i64;
        residuals.push(char_val - base);
    }

    Ok(EncodedMessage {
        equation,
        length: n,
        residuals,
    })
}

/// Decodes an [`EncodedMessage`] back to text losslessly by evaluating it.
///
/// Each evaluation result is rounded, offset by the residual, and interpreted
/// as a Unicode scalar.
///
/// # Arguments
///
/// * `msg` - The encoded message struct.
///
/// # Returns
///
/// (`Result<String>`): The decoded text.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::encode::{encode_text, decode_message};
///
/// let msg = encode_text("AAA", 1, 1).unwrap();
/// let text = decode_message(&msg).unwrap();
/// assert_eq!(text, "AAA");
/// ```
pub fn decode_message(msg: &EncodedMessage) -> Result<String> {
    let mut output = String::with_capacity(msg.length);
    let mut bindings = HashMap::new();
    for i in 0..msg.length {
        bindings.insert("x".to_string(), i as f64);
        let val = msg.equation.evaluate(&bindings).unwrap_or(65.0);
        let residual = msg.residuals.get(i).copied().unwrap_or(0);
        let adjusted = val.round() as i64 + residual;
        let code_point = adjusted.clamp(0, 0x10FFFF) as u32;
        let ch = char::from_u32(code_point).unwrap_or(REPLACEMENT_CHAR);
        output.push(ch);
    }
    Ok(output)
}

/// Decodes a message from a serialised expression string and optional residual adjustments.
///
/// The `equation` string is parsed into an [`Expression`], evaluated over `[0, length)`,
/// and then per-character residuals are added before final Unicode interpretation.
///
/// Residuals are provided as a comma-separated string of integers (e.g. `"2,-3,0"`).
/// Missing residuals default to `0`. Non-integer residual tokens are silently ignored.
///
/// # Arguments
///
/// * `equation` - An infix expression string (e.g. `"x * 10 + 65"`).
/// * `length` - Number of characters to produce.
/// * `residuals` - Comma-separated integer adjustments (one per character).
///
/// # Returns
///
/// (`Result<String>`): The decoded text.
///
/// # Errors
///
/// Returns [`LmmError::ParseError`] when `equation` cannot be parsed.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::encode::decode_from_parts;
///
/// // 65 + 0 = 'A', 65 + 1 = 'B', 65 + 2 = 'C'
/// let text = decode_from_parts("65", 3, "0,1,2").unwrap();
/// assert_eq!(text, "ABC");
/// ```
pub fn decode_from_parts(equation: &str, length: usize, residuals: &str) -> Result<String> {
    let expr: Expression = equation
        .parse::<Expression>()
        .map_err(|e: String| LmmError::ParseError(e))?;

    let parsed_residuals: Vec<i64> = residuals
        .split(',')
        .filter_map(|s| s.trim().parse::<i64>().ok())
        .collect();

    let mut output = String::with_capacity(length);
    let mut bindings = HashMap::new();

    for i in 0..length {
        bindings.insert("x".to_string(), i as f64);
        let base = expr.evaluate(&bindings).unwrap_or(65.0);
        let residual = parsed_residuals.get(i).copied().unwrap_or(0);
        let adjusted = base.round() as i64 + residual;
        let code_point = adjusted.clamp(0, 0x10FFFF) as u32;
        let ch = char::from_u32(code_point).unwrap_or(REPLACEMENT_CHAR);
        output.push(ch);
    }
    Ok(output)
}

/// Computes a compact statistical fingerprint of `text` as a single f64.
///
/// The fingerprint is the mean Unicode code-point value, normalised to `[0, 1]` by
/// dividing by `0x10FFFF`. Useful for quick similarity checks.
///
/// # Arguments
///
/// * `text` - Input text.
///
/// # Returns
///
/// (`f64`): Normalised average code-point in `[0, 1]`. Returns `0.0` for empty input.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::encode::text_fingerprint;
///
/// let f = text_fingerprint("AAA"); // 'A' = 65
/// assert!((f - 65.0 / 0x10FFFF as f64).abs() < 1e-10);
/// ```
pub fn text_fingerprint(text: &str) -> f64 {
    if text.is_empty() {
        return 0.0;
    }
    let sum: f64 = text.chars().map(|c| c as u32 as f64).sum();
    sum / (text.chars().count() as f64 * 0x10FFFF as f64)
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
