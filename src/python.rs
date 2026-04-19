// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Python Bindings
//!
//! This module exposes the LMM core engine to Python via [`pyo3`].
//! Every type and function is gated behind the `python` cargo feature.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// An n-dimensional, row-major dense array of `f64` values.
#[pyclass(name = "Tensor", skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyTensor {
    inner: crate::tensor::Tensor,
}

#[pymethods]
impl PyTensor {
    /// Create a new Tensor.
    ///
    /// Args:
    ///     shape: list of dimension sizes whose product must equal ``len(data)``.
    ///     data:  flat list of float values in row-major order.
    #[new]
    pub fn new(shape: Vec<usize>, data: Vec<f64>) -> PyResult<Self> {
        crate::tensor::Tensor::new(shape, data)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape.clone()
    }

    #[getter]
    pub fn data(&self) -> Vec<f64> {
        self.inner.data.clone()
    }

    pub fn add(&self, other: &PyTensor) -> PyResult<PyTensor> {
        (&self.inner + &other.inner)
            .map(|inner| PyTensor { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn sub(&self, other: &PyTensor) -> PyResult<PyTensor> {
        (&self.inner - &other.inner)
            .map(|inner| PyTensor { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn scale(&self, scalar: f64) -> PyTensor {
        PyTensor {
            inner: self.inner.scale(scalar),
        }
    }

    pub fn dot(&self, other: &PyTensor) -> PyResult<f64> {
        self.inner
            .dot(&other.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn norm(&self) -> f64 {
        self.inner.norm()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, data={:?})",
            self.inner.shape, self.inner.data
        )
    }
}

/// A symbolic mathematical expression.
#[pyclass(name = "Expression", skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyExpression {
    inner: crate::equation::Expression,
}

#[pymethods]
impl PyExpression {
    /// Parse a symbolic expression string, e.g. ``"(x * 2)"`` or ``"sin(x)"``.
    #[staticmethod]
    pub fn parse(s: &str) -> PyResult<Self> {
        s.parse::<crate::equation::Expression>()
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Evaluate the expression given a ``dict[str, float]`` variable binding.
    pub fn evaluate(&self, bindings: HashMap<String, f64>) -> PyResult<f64> {
        self.inner
            .evaluate(&bindings)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Return the symbolic derivative with respect to ``var``.
    pub fn diff(&self, var: &str) -> PyExpression {
        PyExpression {
            inner: self.inner.symbolic_diff(var),
        }
    }

    /// Simplify using constant folding and algebraic identities.
    pub fn simplify(&self) -> PyExpression {
        PyExpression {
            inner: self.inner.simplify(),
        }
    }

    pub fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    pub fn __repr__(&self) -> String {
        format!("Expression(\"{}\")", self.inner)
    }
}

/// A directed acyclic causal graph supporting interventions and counterfactuals.
#[pyclass(name = "CausalGraph")]
pub struct PyCausalGraph {
    inner: crate::causal::CausalGraph,
}

#[pymethods]
impl PyCausalGraph {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: crate::causal::CausalGraph::new(),
        }
    }

    pub fn add_node(&mut self, name: &str, value: Option<f64>) {
        self.inner.add_node(name, value);
    }

    pub fn add_edge(&mut self, from_: &str, to: &str, coefficient: Option<f64>) -> PyResult<()> {
        self.inner
            .add_edge(from_, to, coefficient)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn forward_pass(&mut self) -> PyResult<()> {
        self.inner
            .forward_pass()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    pub fn intervene(&mut self, var: &str, value: f64) -> PyResult<()> {
        use crate::traits::Causal;
        self.inner
            .intervene(var, value)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn get_value(&self, name: &str) -> PyResult<Option<f64>> {
        Ok(self.inner.get_value(name))
    }

    pub fn counterfactual(&self, var: &str, value: f64, target: &str) -> PyResult<f64> {
        self.inner
            .counterfactual(var, value, target)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn topological_order(&self) -> PyResult<Vec<String>> {
        self.inner
            .topological_order()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

/// Numeric ODE integrator (Euler, RK4, Leapfrog).
#[pyclass(name = "Simulator")]
pub struct PySimulator {
    inner: crate::simulation::Simulator,
}

#[pymethods]
impl PySimulator {
    /// Create a Simulator with a fixed step size ``h``.
    #[new]
    pub fn new(step_size: f64) -> Self {
        Self {
            inner: crate::simulation::Simulator { step_size },
        }
    }

    #[getter]
    pub fn step_size(&self) -> f64 {
        self.inner.step_size
    }

    /// Euler step for a ``HarmonicOscillator`` model.
    pub fn euler_step_osc(
        &self,
        model: &PyHarmonicOscillator,
        state: &PyTensor,
    ) -> PyResult<PyTensor> {
        self.inner
            .euler_step(&model.inner, &state.inner)
            .map(|inner| PyTensor { inner })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// RK4 step for a ``HarmonicOscillator`` model.
    pub fn rk4_step_osc(
        &self,
        model: &PyHarmonicOscillator,
        state: &PyTensor,
    ) -> PyResult<PyTensor> {
        self.inner
            .rk4_step(&model.inner, &state.inner)
            .map(|inner| PyTensor { inner })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

/// 1-D harmonic oscillator. State: ``[x, x_dot]``.
#[pyclass(name = "HarmonicOscillator")]
pub struct PyHarmonicOscillator {
    inner: crate::physics::HarmonicOscillator,
}

#[pymethods]
impl PyHarmonicOscillator {
    /// Args: omega (rad/s), x0 (initial displacement), v0 (initial velocity).
    #[new]
    pub fn new(omega: f64, x0: f64, v0: f64) -> PyResult<Self> {
        crate::physics::HarmonicOscillator::new(omega, x0, v0)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    pub fn omega(&self) -> f64 {
        self.inner.omega
    }

    pub fn energy(&self) -> f64 {
        self.inner.energy()
    }

    /// Current state tensor ``[x, x_dot]``.
    pub fn state(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.state.clone(),
        }
    }
}

/// Lorenz chaotic attractor. State: ``[x, y, z]``.
#[pyclass(name = "LorenzSystem")]
pub struct PyLorenzSystem {
    inner: crate::physics::LorenzSystem,
}

#[pymethods]
impl PyLorenzSystem {
    #[new]
    pub fn new(sigma: f64, rho: f64, beta: f64, x0: f64, y0: f64, z0: f64) -> PyResult<Self> {
        crate::physics::LorenzSystem::new(sigma, rho, beta, x0, y0, z0)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Current state ``[x, y, z]``.
    pub fn state(&self) -> Vec<f64> {
        self.inner.state.data.clone()
    }
}

/// A pendulum (angle, angular-velocity). ``Pendulum::new(g, l, theta0, omega0)``.
#[pyclass(name = "Pendulum")]
pub struct PyPendulum {
    inner: crate::physics::Pendulum,
}

#[pymethods]
impl PyPendulum {
    /// Args: g (gravity m/s²), l (length m), theta0 (rad), omega0 (rad/s).
    #[new]
    pub fn new(g: f64, l: f64, theta0: f64, omega0: f64) -> PyResult<Self> {
        crate::physics::Pendulum::new(g, l, theta0, omega0)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Current state ``[theta, omega]``.
    pub fn state(&self) -> Vec<f64> {
        self.inner.state.data.clone()
    }
}

/// SIR epidemiological model. State: ``[S, I, R]``.
#[pyclass(name = "SIRModel")]
pub struct PySIRModel {
    inner: crate::physics::SIRModel,
}

#[pymethods]
impl PySIRModel {
    #[new]
    pub fn new(beta: f64, gamma: f64, s0: f64, i0: f64, r0: f64) -> PyResult<Self> {
        crate::physics::SIRModel::new(beta, gamma, s0, i0, r0)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Current state ``[S, I, R]``.
    pub fn state(&self) -> Vec<f64> {
        self.inner.state.data.clone()
    }
}

/// Encode text to a symbolic expression + integer residuals for lossless reconstruction.
///
/// Args:
///     text:       input text string.
///     iterations: symbolic regression iterations (default 40).
///     depth:      max expression depth (default 3).
///
/// Returns a dict: ``{"expression": str, "length": int, "residuals": list[int]}``.
#[pyfunction]
#[pyo3(signature = (text, iterations=None, depth=None))]
pub fn encode_text(
    py: Python<'_>,
    text: &str,
    iterations: Option<usize>,
    depth: Option<usize>,
) -> PyResult<Py<PyDict>> {
    let msg = crate::encode::encode_text(text, iterations.unwrap_or(40), depth.unwrap_or(3))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let d = PyDict::new(py);
    d.set_item("expression", format!("{}", msg.equation))?;
    d.set_item("length", msg.length)?;
    d.set_item("residuals", msg.residuals.clone())?;
    Ok(d.into())
}

/// Decode an encoded message back to the original text.
///
/// Args:
///     expression: expression string from :func:`encode_text`.
///     length:     original text byte length.
///     residuals:  residual list (``list[int]``) from :func:`encode_text`.
#[pyfunction]
pub fn decode_message(expression: &str, length: usize, residuals: Vec<i64>) -> PyResult<String> {
    let eq = expression
        .parse::<crate::equation::Expression>()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let msg = crate::encode::EncodedMessage {
        equation: eq,
        length,
        residuals,
    };
    crate::encode::decode_message(&msg).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Minimum Description Length score for a symbolic expression over data.
///
/// Args:
///     expr_str: expression string (e.g. ``"(x * 2)"``).
///     inputs:   list of input rows, each a list of floats.
///     targets:  target output list.
#[pyfunction]
pub fn mdl_score(expr_str: &str, inputs: Vec<Vec<f64>>, targets: Vec<f64>) -> PyResult<f64> {
    let expr = expr_str
        .parse::<crate::equation::Expression>()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(crate::compression::mdl_score(&expr, &inputs, &targets))
}

/// Mean-squared error of an expression over (inputs, targets).
#[pyfunction]
pub fn compute_mse(expr_str: &str, inputs: Vec<Vec<f64>>, targets: Vec<f64>) -> PyResult<f64> {
    let expr = expr_str
        .parse::<crate::equation::Expression>()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(crate::compression::compute_mse(&expr, &inputs, &targets))
}

/// R² of an expression over (inputs, targets).
#[pyfunction]
pub fn r_squared(expr_str: &str, inputs: Vec<Vec<f64>>, targets: Vec<f64>) -> PyResult<f64> {
    let expr = expr_str
        .parse::<crate::equation::Expression>()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(crate::compression::r_squared(&expr, &inputs, &targets))
}

/// Akaike Information Criterion: ``2k - 2 ln(L)``.
#[pyfunction]
pub fn aic_score(n_params: usize, log_likelihood: f64) -> f64 {
    crate::compression::aic_score(n_params, log_likelihood)
}

/// Bayesian Information Criterion: ``k ln(n) - 2 ln(L)``.
#[pyfunction]
pub fn bic_score(n_params: usize, n_samples: usize, log_likelihood: f64) -> f64 {
    crate::compression::bic_score(n_params, n_samples, log_likelihood)
}

/// Genetic-programming symbolic regressor.
#[pyclass(name = "SymbolicRegression")]
pub struct PySymbolicRegression {
    inner: crate::discovery::SymbolicRegression,
}

#[pymethods]
impl PySymbolicRegression {
    /// Args:
    ///     max_depth:        max expression-tree depth (default 3).
    ///     iterations:       number of evolutionary generations (default 50).
    ///     population_size:  candidate pool size (default 50).
    ///     var_names:        list of input variable names (default ``["x"]``).
    #[new]
    #[pyo3(signature = (max_depth=None, iterations=None, population_size=None, var_names=None))]
    pub fn new(
        max_depth: Option<usize>,
        iterations: Option<usize>,
        population_size: Option<usize>,
        var_names: Option<Vec<String>>,
    ) -> Self {
        let mut sr = crate::discovery::SymbolicRegression::new(
            max_depth.unwrap_or(3),
            iterations.unwrap_or(50),
        );
        if let Some(p) = population_size {
            sr = sr.with_population(p);
        }
        if let Some(v) = var_names {
            sr = sr.with_variables(v);
        }
        Self { inner: sr }
    }

    /// Fit to ``(inputs, targets)`` where ``inputs`` is a list of row float lists.
    ///
    /// Returns the best-fit expression string.
    pub fn fit(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<f64>) -> PyResult<String> {
        self.inner
            .fit(&inputs, &targets)
            .map(|e| format!("{}", e))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

/// Sentence generator.
#[pyclass(name = "SentenceGenerator")]
pub struct PySentenceGenerator {
    inner: crate::text::SentenceGenerator,
}

#[pymethods]
impl PySentenceGenerator {
    /// Args:
    ///     iterations: symbolic regression iterations (default 20).
    ///     depth:      max expression depth (default 3).
    #[new]
    #[pyo3(signature = (iterations=None, depth=None))]
    pub fn new(iterations: Option<usize>, depth: Option<usize>) -> Self {
        Self {
            inner: crate::text::SentenceGenerator::new(
                iterations.unwrap_or(20),
                depth.unwrap_or(3),
            ),
        }
    }

    /// Generate a sentence seeded from ``seed``.
    pub fn generate(&self, seed: &str) -> PyResult<String> {
        self.inner
            .generate(seed)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

/// Paragraph generator.
#[pyclass(name = "ParagraphGenerator")]
pub struct PyParagraphGenerator {
    inner: crate::text::ParagraphGenerator,
}

#[pymethods]
impl PyParagraphGenerator {
    /// Args:
    ///     sentence_count: sentences per paragraph (default 5).
    ///     iterations:     symbolic regression iterations (default 20).
    ///     depth:          max expression depth (default 3).
    #[new]
    #[pyo3(signature = (sentence_count=None, iterations=None, depth=None))]
    pub fn new(
        sentence_count: Option<usize>,
        iterations: Option<usize>,
        depth: Option<usize>,
    ) -> Self {
        Self {
            inner: crate::text::ParagraphGenerator::new(
                sentence_count.unwrap_or(5),
                iterations.unwrap_or(20),
                depth.unwrap_or(3),
            ),
        }
    }

    /// Generate a paragraph seeded from ``seed``.
    pub fn generate(&self, seed: &str) -> PyResult<String> {
        self.inner
            .generate(seed)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

/// Text summarizer.
#[pyclass(name = "TextSummarizer")]
pub struct PyTextSummarizer {
    inner: crate::text::TextSummarizer,
}

#[pymethods]
impl PyTextSummarizer {
    /// Args:
    ///     sentence_count: number of summary sentences (default 3).
    ///     iterations:     symbolic regression iterations (default 20).
    ///     depth:          max expression depth (default 3).
    #[new]
    #[pyo3(signature = (sentence_count=None, iterations=None, depth=None))]
    pub fn new(
        sentence_count: Option<usize>,
        iterations: Option<usize>,
        depth: Option<usize>,
    ) -> Self {
        Self {
            inner: crate::text::TextSummarizer::new(
                sentence_count.unwrap_or(3),
                iterations.unwrap_or(20),
                depth.unwrap_or(3),
            ),
        }
    }

    /// Summarize ``text``, returning key sentences joined by newlines.
    pub fn summarize(&self, text: &str) -> PyResult<String> {
        self.inner
            .summarize(text)
            .map(|v| v.join("\n"))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

/// Symbolic-trajectory text continuation predictor.
#[pyclass(name = "TextPredictor")]
pub struct PyTextPredictor {
    inner: crate::predict::TextPredictor,
}

#[pymethods]
impl PyTextPredictor {
    /// Args:
    ///     window_size: context token count (default 20).
    ///     iterations:  symbolic regression iterations (default 30).
    ///     depth:       max expression depth (default 3).
    #[new]
    #[pyo3(signature = (window_size=None, iterations=None, depth=None))]
    pub fn new(
        window_size: Option<usize>,
        iterations: Option<usize>,
        depth: Option<usize>,
    ) -> Self {
        Self {
            inner: crate::predict::TextPredictor::new(
                window_size.unwrap_or(20),
                iterations.unwrap_or(30),
                depth.unwrap_or(3),
            ),
        }
    }

    /// Predict a text continuation of ``predict_length`` characters.
    ///
    /// Returns a dict: ``{continuation, trajectory_equation, rhythm_equation, window_used}``.
    #[pyo3(signature = (text, predict_length=None))]
    pub fn predict(
        &self,
        py: Python<'_>,
        text: &str,
        predict_length: Option<usize>,
    ) -> PyResult<Py<PyDict>> {
        let result = self
            .inner
            .predict_continuation(text, predict_length.unwrap_or(60))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let d = PyDict::new(py);
        d.set_item("continuation", result.continuation)?;
        d.set_item(
            "trajectory_equation",
            format!("{}", result.trajectory_equation),
        )?;
        d.set_item("rhythm_equation", format!("{}", result.rhythm_equation))?;
        d.set_item("window_used", result.window_used)?;
        Ok(d.into())
    }
}

/// Probabilistic word-substitution text enhancer.
#[pyclass(name = "StochasticEnhancer")]
pub struct PyStochasticEnhancer {
    inner: crate::stochastic::StochasticEnhancer,
}

#[pymethods]
impl PyStochasticEnhancer {
    /// Args:
    ///     p: substitution probability per non-stop word (0.0-1.0).
    #[new]
    pub fn new(p: f64) -> Self {
        Self {
            inner: crate::stochastic::StochasticEnhancer::new(p),
        }
    }

    pub fn enhance(&mut self, text: &str) -> String {
        self.inner.enhance(text)
    }
}

/// A physical field: a Tensor with spatial-grid semantics.
#[pyclass(name = "Field")]
pub struct PyField {
    inner: crate::field::Field,
}

#[pymethods]
impl PyField {
    #[new]
    pub fn new(shape: Vec<usize>, values: &PyTensor) -> PyResult<Self> {
        crate::field::Field::new(shape, values.inner.clone())
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Compute the finite-difference Laplacian.
    pub fn laplacian(&self) -> PyResult<PyTensor> {
        self.inner
            .compute_laplacian()
            .map(|f| PyTensor { inner: f.values })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[getter]
    pub fn values(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.values.clone(),
        }
    }
}

/// Spatial convolution operator over a Field.
#[pyclass(name = "NeuralOperator")]
pub struct PyNeuralOperator {
    inner: crate::operator::NeuralOperator,
}

#[pymethods]
impl PyNeuralOperator {
    #[new]
    pub fn new(n_kernel_modes: usize) -> Self {
        Self {
            inner: crate::operator::NeuralOperator::new(n_kernel_modes),
        }
    }

    pub fn transform(&self, field: &PyField) -> PyResult<PyField> {
        self.inner
            .transform(&field.inner)
            .map(|inner| PyField { inner })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

/// Global Workspace-inspired consciousness module.
#[pyclass(name = "Consciousness")]
pub struct PyConsciousness {
    inner: crate::consciousness::Consciousness,
}

#[pymethods]
impl PyConsciousness {
    /// Args:
    ///     state_len:  number of internal state dimensions.
    ///     lookahead:  planning steps (default 5).
    ///     step_size:  integration Δt (default 0.01).
    #[new]
    #[pyo3(signature = (state_len, lookahead=None, step_size=None))]
    pub fn new(state_len: usize, lookahead: Option<usize>, step_size: Option<f64>) -> Self {
        let initial = crate::tensor::Tensor::zeros(vec![state_len]);
        Self {
            inner: crate::consciousness::Consciousness::new(
                initial,
                lookahead.unwrap_or(5),
                step_size.unwrap_or(0.01),
            ),
        }
    }

    /// Run one perception-action tick: ingest raw ``bytes`` and advance the world model.
    ///
    /// Returns the new state as a ``list[float]``.
    pub fn tick(&mut self, sensory_bytes: Vec<u8>) -> PyResult<Vec<f64>> {
        self.inner
            .tick(&sensory_bytes)
            .map(|t| t.data)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

/// Render a procedurally-generated PPM image from a text prompt.
///
/// Args:
///     prompt:     seed text.
///     width:      image width in pixels (default 512).
///     height:     image height in pixels (default 512).
///     palette:    palette name: ``"warm"``, ``"cool"``, ``"neon"``, ``"mono"`` (default ``""``).
///     style:      style string: ``"wave"``, ``"noise"``, ``"gradient"``, ``"plasma"`` (default ``"wave"``).
///     components: spectral components (default 8).
///     output:     output file path (default ``"output.ppm"``).
///
/// Returns the output path.
#[pyfunction]
#[pyo3(signature = (prompt, width=None, height=None, palette=None, style=None, components=None, output=None))]
pub fn render_image(
    prompt: String,
    width: Option<u32>,
    height: Option<u32>,
    palette: Option<String>,
    style: Option<String>,
    components: Option<usize>,
    output: Option<String>,
) -> PyResult<String> {
    let style_mode = style
        .as_deref()
        .and_then(|s| s.parse().ok())
        .unwrap_or_default();
    let params = crate::imagen::ImagenParams {
        prompt,
        width: width.unwrap_or(512),
        height: height.unwrap_or(512),
        components: components.unwrap_or(8),
        style: style_mode,
        palette_name: palette.unwrap_or_default(),
        output: output.unwrap_or_else(|| "output.ppm".to_string()),
    };
    crate::imagen::render(&params).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Register all Python-exposed types and functions into the ``_lmm`` module.
pub fn register_python_module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_class::<PyExpression>()?;
    m.add_class::<PyCausalGraph>()?;
    m.add_class::<PySimulator>()?;
    m.add_class::<PyHarmonicOscillator>()?;
    m.add_class::<PyLorenzSystem>()?;
    m.add_class::<PyPendulum>()?;
    m.add_class::<PySIRModel>()?;
    m.add_class::<PySymbolicRegression>()?;
    m.add_class::<PySentenceGenerator>()?;
    m.add_class::<PyParagraphGenerator>()?;
    m.add_class::<PyTextSummarizer>()?;
    m.add_class::<PyTextPredictor>()?;
    m.add_class::<PyStochasticEnhancer>()?;
    m.add_class::<PyField>()?;
    m.add_class::<PyNeuralOperator>()?;
    m.add_class::<PyConsciousness>()?;

    m.add_function(wrap_pyfunction!(encode_text, m)?)?;
    m.add_function(wrap_pyfunction!(decode_message, m)?)?;
    m.add_function(wrap_pyfunction!(mdl_score, m)?)?;
    m.add_function(wrap_pyfunction!(compute_mse, m)?)?;
    m.add_function(wrap_pyfunction!(r_squared, m)?)?;
    m.add_function(wrap_pyfunction!(aic_score, m)?)?;
    m.add_function(wrap_pyfunction!(bic_score, m)?)?;
    m.add_function(wrap_pyfunction!(render_image, m)?)?;
    Ok(())
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
