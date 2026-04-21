// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Node.js Bindings
//!
//! This module exposes the LMM core engine to Node.js via [`napi-derive`].
//! Every type and function is gated behind the `node` cargo feature.

use crate::traits::Causal;
use napi_derive::napi;
use std::collections::HashMap;

/// An n-dimensional dense array of f64 values.
#[napi(js_name = "Tensor")]
pub struct NapiTensor {
    inner: crate::tensor::Tensor,
}

#[napi]
impl NapiTensor {
    /// Create a new Tensor from ``shape`` and flat ``data`` arrays.
    #[napi(constructor)]
    pub fn new(shape: Vec<u32>, data: Vec<f64>) -> napi::Result<Self> {
        let shape: Vec<usize> = shape.into_iter().map(|x| x as usize).collect();
        crate::tensor::Tensor::new(shape, data)
            .map(|inner| Self { inner })
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    #[napi(getter)]
    pub fn shape(&self) -> Vec<u32> {
        self.inner.shape.iter().map(|&x| x as u32).collect()
    }

    #[napi(getter)]
    pub fn data(&self) -> Vec<f64> {
        self.inner.data.clone()
    }

    #[napi]
    pub fn norm(&self) -> f64 {
        self.inner.norm()
    }

    #[napi]
    pub fn scale(&self, scalar: f64) -> NapiTensor {
        NapiTensor {
            inner: self.inner.scale(scalar),
        }
    }

    #[napi]
    pub fn add(&self, other: &NapiTensor) -> napi::Result<NapiTensor> {
        (&self.inner + &other.inner)
            .map(|inner| NapiTensor { inner })
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    #[napi]
    pub fn dot(&self, other: &NapiTensor) -> napi::Result<f64> {
        self.inner
            .dot(&other.inner)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }
}

/// A symbolic mathematical expression tree.
#[napi(js_name = "Expression")]
pub struct NapiExpression {
    inner: crate::equation::Expression,
}

#[napi]
impl NapiExpression {
    /// Parse an expression string, e.g. ``"(x * 2)"``.
    #[napi(factory)]
    pub fn parse(s: String) -> napi::Result<Self> {
        s.parse::<crate::equation::Expression>()
            .map(|inner| Self { inner })
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Evaluate with a ``{[key: string]: number}`` binding map.
    #[napi]
    pub fn evaluate(&self, bindings: HashMap<String, f64>) -> napi::Result<f64> {
        self.inner
            .evaluate(&bindings)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Return the symbolic derivative with respect to ``var``.
    #[napi]
    pub fn diff(&self, var: String) -> NapiExpression {
        NapiExpression {
            inner: self.inner.symbolic_diff(&var),
        }
    }

    /// Simplify the expression.
    #[napi]
    pub fn simplify(&self) -> NapiExpression {
        NapiExpression {
            inner: self.inner.simplify(),
        }
    }

    /// Return the expression as a string.
    #[napi]
    pub fn to_string(&self) -> String {
        format!("{}", self.inner)
    }
}

/// Directed acyclic causal graph.
#[napi(js_name = "CausalGraph")]
pub struct NapiCausalGraph {
    inner: crate::causal::CausalGraph,
}

#[napi]
impl NapiCausalGraph {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: crate::causal::CausalGraph::new(),
        }
    }

    #[napi]
    pub fn add_node(&mut self, name: String, value: Option<f64>) {
        self.inner.add_node(&name, value);
    }

    #[napi]
    pub fn add_edge(
        &mut self,
        from: String,
        to: String,
        coefficient: Option<f64>,
    ) -> napi::Result<()> {
        self.inner
            .add_edge(&from, &to, coefficient)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    #[napi]
    pub fn forward_pass(&mut self) -> napi::Result<()> {
        self.inner
            .forward_pass()
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    #[napi]
    pub fn intervene(&mut self, var: String, value: f64) -> napi::Result<()> {
        self.inner
            .intervene(&var, value)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    #[napi]
    pub fn get_value(&self, name: String) -> Option<f64> {
        self.inner.get_value(&name)
    }

    #[napi]
    pub fn counterfactual(&self, var: String, value: f64, target: String) -> napi::Result<f64> {
        self.inner
            .counterfactual(&var, value, &target)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    #[napi]
    pub fn topological_order(&self) -> napi::Result<Vec<String>> {
        self.inner
            .topological_order()
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }
}

/// 1-D harmonic oscillator. State: ``[x, x_dot]``.
#[napi(js_name = "HarmonicOscillator")]
pub struct NapiHarmonicOscillator {
    inner: crate::physics::HarmonicOscillator,
}

#[napi]
impl NapiHarmonicOscillator {
    /// Create with ``omega`` (rad/s), ``x0`` (displacement), ``v0`` (velocity).
    #[napi(constructor)]
    pub fn new(omega: f64, x0: f64, v0: f64) -> napi::Result<Self> {
        crate::physics::HarmonicOscillator::new(omega, x0, v0)
            .map(|inner| Self { inner })
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    #[napi(getter)]
    pub fn omega(&self) -> f64 {
        self.inner.omega
    }

    #[napi]
    pub fn energy(&self) -> f64 {
        self.inner.energy()
    }

    #[napi]
    pub fn state(&self) -> Vec<f64> {
        self.inner.state.data.clone()
    }
}

/// Numeric ODE integrator.
#[napi(js_name = "Simulator")]
pub struct NapiSimulator {
    inner: crate::simulation::Simulator,
}

#[napi]
impl NapiSimulator {
    #[napi(constructor)]
    pub fn new(step_size: f64) -> Self {
        Self {
            inner: crate::simulation::Simulator { step_size },
        }
    }

    /// Euler step for a ``HarmonicOscillator``.
    #[napi]
    pub fn euler_step_osc(
        &self,
        model: &NapiHarmonicOscillator,
        state: &NapiTensor,
    ) -> napi::Result<NapiTensor> {
        self.inner
            .euler_step(&model.inner, &state.inner)
            .map(|inner| NapiTensor { inner })
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// RK4 step for a ``HarmonicOscillator``.
    #[napi]
    pub fn rk4_step_osc(
        &self,
        model: &NapiHarmonicOscillator,
        state: &NapiTensor,
    ) -> napi::Result<NapiTensor> {
        self.inner
            .rk4_step(&model.inner, &state.inner)
            .map(|inner| NapiTensor { inner })
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }
}

/// Genetic-programming symbolic regressor.
#[napi(js_name = "SymbolicRegression")]
pub struct NapiSymbolicRegression {
    inner: crate::discovery::SymbolicRegression,
}

#[napi]
impl NapiSymbolicRegression {
    /// Create with optional ``maxDepth``, ``iterations``, ``populationSize``.
    #[napi(constructor)]
    pub fn new(
        max_depth: Option<u32>,
        iterations: Option<u32>,
        population_size: Option<u32>,
    ) -> Self {
        let mut sr = crate::discovery::SymbolicRegression::new(
            max_depth.unwrap_or(3) as usize,
            iterations.unwrap_or(50) as usize,
        );
        if let Some(p) = population_size {
            sr = sr.with_population(p as usize);
        }
        Self { inner: sr }
    }

    /// Fit to ``inputs`` (array of rows) and ``targets`` array.
    ///
    /// Returns the best-fit expression as a string.
    #[napi]
    pub fn fit(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<f64>) -> napi::Result<String> {
        self.inner
            .fit(&inputs, &targets)
            .map(|e| format!("{}", e))
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }
}

/// Return type for ``encodeText``.
#[napi(object)]
pub struct EncodeResult {
    pub expression: String,
    pub length: u32,
    pub residuals: Vec<i64>,
}

/// Encode text into a symbolic expression and integer residuals.
#[napi]
pub fn encode_text(
    text: String,
    iterations: Option<u32>,
    depth: Option<u32>,
) -> napi::Result<EncodeResult> {
    let msg = crate::encode::encode_text(
        &text,
        iterations.unwrap_or(40) as usize,
        depth.unwrap_or(3) as usize,
    )
    .map_err(|e| napi::Error::from_reason(e.to_string()))?;
    Ok(EncodeResult {
        expression: format!("{}", msg.equation),
        length: msg.length as u32,
        residuals: msg.residuals,
    })
}

/// Decode a previously encoded message back to text.
#[napi]
pub fn decode_message(
    expression: String,
    length: u32,
    residuals: Vec<i64>,
) -> napi::Result<String> {
    let eq = expression
        .parse::<crate::equation::Expression>()
        .map_err(|e| napi::Error::from_reason(e.to_string()))?;
    let msg = crate::encode::EncodedMessage {
        equation: eq,
        length: length as usize,
        residuals,
    };
    crate::encode::decode_message(&msg).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// MDL score for an expression over ``(inputs, targets)``.
#[napi]
pub fn mdl_score(expr_str: String, inputs: Vec<Vec<f64>>, targets: Vec<f64>) -> napi::Result<f64> {
    let expr = expr_str
        .parse::<crate::equation::Expression>()
        .map_err(|e| napi::Error::from_reason(e.to_string()))?;
    Ok(crate::compression::mdl_score(&expr, &inputs, &targets))
}

/// MSE of expression over ``(inputs, targets)``.
#[napi]
pub fn compute_mse(
    expr_str: String,
    inputs: Vec<Vec<f64>>,
    targets: Vec<f64>,
) -> napi::Result<f64> {
    let expr = expr_str
        .parse::<crate::equation::Expression>()
        .map_err(|e| napi::Error::from_reason(e.to_string()))?;
    Ok(crate::compression::compute_mse(&expr, &inputs, &targets))
}

/// R² of expression over ``(inputs, targets)``.
#[napi]
pub fn r_squared(expr_str: String, inputs: Vec<Vec<f64>>, targets: Vec<f64>) -> napi::Result<f64> {
    let expr = expr_str
        .parse::<crate::equation::Expression>()
        .map_err(|e| napi::Error::from_reason(e.to_string()))?;
    Ok(crate::compression::r_squared(&expr, &inputs, &targets))
}

/// AIC: ``2k - 2 ln(L)``.
#[napi]
pub fn aic_score(n_params: u32, log_likelihood: f64) -> f64 {
    crate::compression::aic_score(n_params as usize, log_likelihood)
}

/// BIC: ``k ln(n) - 2 ln(L)``.
#[napi]
pub fn bic_score(n_params: u32, n_samples: u32, log_likelihood: f64) -> f64 {
    crate::compression::bic_score(n_params as usize, n_samples as usize, log_likelihood)
}

/// Return type for ``TextPredictor.predict``.
#[napi(object)]
pub struct PredictionResult {
    pub continuation: String,
    pub trajectory_equation: String,
    pub rhythm_equation: String,
    pub window_used: u32,
}

/// Symbolic-trajectory text continuation predictor.
#[napi(js_name = "TextPredictor")]
pub struct NapiTextPredictor {
    inner: crate::predict::TextPredictor,
}

#[napi]
impl NapiTextPredictor {
    #[napi(constructor)]
    pub fn new(window_size: Option<u32>, iterations: Option<u32>, depth: Option<u32>) -> Self {
        Self {
            inner: crate::predict::TextPredictor::new(
                window_size.unwrap_or(20) as usize,
                iterations.unwrap_or(30) as usize,
                depth.unwrap_or(3) as usize,
            ),
        }
    }

    /// Predict a continuation of ``predictLength`` characters.
    #[napi]
    pub fn predict(
        &self,
        text: String,
        predict_length: Option<u32>,
    ) -> napi::Result<PredictionResult> {
        self.inner
            .predict_continuation(&text, predict_length.unwrap_or(60) as usize)
            .map(|r| PredictionResult {
                continuation: format!("{}{}", text, r.continuation),
                trajectory_equation: format!("{}", r.trajectory_equation),
                rhythm_equation: format!("{}", r.rhythm_equation),
                window_used: r.window_used as u32,
            })
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }
}

/// Probabilistic word-substitution enhancer.
#[napi(js_name = "StochasticEnhancer")]
pub struct NapiStochasticEnhancer {
    inner: crate::stochastic::StochasticEnhancer,
}

#[napi]
impl NapiStochasticEnhancer {
    #[napi(constructor)]
    pub fn new(p: f64) -> Self {
        Self {
            inner: crate::stochastic::StochasticEnhancer::new(p),
        }
    }

    #[napi]
    pub fn enhance(&mut self, text: String) -> String {
        self.inner.enhance(&text)
    }
}

/// Sentence generator.
#[napi(js_name = "SentenceGenerator")]
pub struct NapiSentenceGenerator {
    inner: crate::text::SentenceGenerator,
}

#[napi]
impl NapiSentenceGenerator {
    #[napi(constructor)]
    pub fn new(iterations: Option<u32>, depth: Option<u32>) -> Self {
        Self {
            inner: crate::text::SentenceGenerator::new(
                iterations.unwrap_or(20) as usize,
                depth.unwrap_or(3) as usize,
            ),
        }
    }

    #[napi]
    pub fn generate(&self, seed: String) -> napi::Result<String> {
        self.inner
            .generate(&seed)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }
}

/// Paragraph generator.
#[napi(js_name = "ParagraphGenerator")]
pub struct NapiParagraphGenerator {
    inner: crate::text::ParagraphGenerator,
}

#[napi]
impl NapiParagraphGenerator {
    #[napi(constructor)]
    pub fn new(sentence_count: Option<u32>, iterations: Option<u32>, depth: Option<u32>) -> Self {
        Self {
            inner: crate::text::ParagraphGenerator::new(
                sentence_count.unwrap_or(5) as usize,
                iterations.unwrap_or(20) as usize,
                depth.unwrap_or(3) as usize,
            ),
        }
    }

    #[napi]
    pub fn generate(&self, seed: String) -> napi::Result<String> {
        self.inner
            .generate(&seed)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }
}

/// Text summarizer.
#[napi(js_name = "TextSummarizer")]
pub struct NapiTextSummarizer {
    inner: crate::text::TextSummarizer,
}

#[napi]
impl NapiTextSummarizer {
    #[napi(constructor)]
    pub fn new(sentence_count: Option<u32>, iterations: Option<u32>, depth: Option<u32>) -> Self {
        Self {
            inner: crate::text::TextSummarizer::new(
                sentence_count.unwrap_or(3) as usize,
                iterations.unwrap_or(20) as usize,
                depth.unwrap_or(3) as usize,
            ),
        }
    }

    /// Returns key sentences joined by newlines.
    #[napi]
    pub fn summarize(&self, text: String) -> napi::Result<String> {
        self.inner
            .summarize(&text)
            .map(|v| v.join("\n"))
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }
}

/// Global Workspace consciousness module.
#[napi(js_name = "Consciousness")]
pub struct NapiConsciousness {
    inner: crate::consciousness::Consciousness,
}

#[napi]
impl NapiConsciousness {
    #[napi(constructor)]
    pub fn new(state_len: u32, lookahead: Option<u32>, step_size: Option<f64>) -> Self {
        let initial = crate::tensor::Tensor::zeros(vec![state_len as usize]);
        Self {
            inner: crate::consciousness::Consciousness::new(
                initial,
                lookahead.unwrap_or(5) as usize,
                step_size.unwrap_or(0.01),
            ),
        }
    }

    /// Ingest raw bytes and return the new world-model state.
    #[napi]
    pub fn tick(&mut self, sensory_bytes: napi::bindgen_prelude::Buffer) -> napi::Result<Vec<f64>> {
        self.inner
            .tick(sensory_bytes.as_ref())
            .map(|t| t.data)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }
}

/// Render a PPM image from a text prompt. Returns the output file path.
#[napi]
pub fn render_image(
    prompt: String,
    width: Option<u32>,
    height: Option<u32>,
    palette: Option<String>,
    style: Option<String>,
    components: Option<u32>,
    output: Option<String>,
) -> napi::Result<String> {
    let style_mode = style
        .as_deref()
        .and_then(|s| s.parse().ok())
        .unwrap_or_default();
    let params = crate::imagen::ImagenParams {
        prompt,
        width: width.unwrap_or(512),
        height: height.unwrap_or(512),
        components: components.unwrap_or(8) as usize,
        style: style_mode,
        palette_name: palette.unwrap_or_default(),
        output: output.unwrap_or_else(|| "output.ppm".to_string()),
    };
    crate::imagen::render(&params).map_err(|e| napi::Error::from_reason(e.to_string()))
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
