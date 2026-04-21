# LMM Python Documentation 🐍

The **`lmm-rs`** package offers blazing-fast Python bindings for the `lmm` Rust engine. The native extension embeds a Tokio runtime, so all calls are synchronous; No `asyncio` or event loop required.

## 📦 Installation

```sh
# Recommended: create a virtual environment first
python3 -m venv .venv
source .venv/bin/activate

pip install lmm-rs
```

Or build locally from source:

```sh
pip install maturin
maturin develop --features python
```

> [!NOTE]
> Requires Python 3.12+ and Rust 1.86+. Pre-compiled wheels ship for CPython 3.12-3.13 on Linux (x64 / arm64), macOS (x64 / Apple Silicon), and Windows (x64).

## 🛠 Quick Start

```sh
python3
```

```python
import lmm

# Tensor arithmetic
t = lmm.Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
print(t.shape)   # [2, 3]
print(t.norm())

# Symbolic expression
expr = lmm.Expression.parse("(sin(x) * 2)")
val  = expr.evaluate({"x": 3.14159})
deriv = expr.diff("x").simplify()
print(str(deriv))  # (cos(x) * 2)

# Causal graph
g = lmm.CausalGraph()
g.add_node("x", 3.0)
g.add_node("y", None)
g.add_edge("x", "y", 2.0)
g.forward_pass()
y_hat = g.counterfactual("x", 10.0, "y")
print(f"do(x=10) → y = {y_hat}")  # 20.0

# Physics simulation
osc   = lmm.HarmonicOscillator(omega=1.0, x0=1.0, v0=0.0)
sim   = lmm.Simulator(step_size=0.01)
state = sim.rk4_step_osc(osc, osc.state())
print(state.data)

# Text encode / decode (lossless)
enc      = lmm.encode_text("The Pharaohs encoded reality.", iterations=80, depth=4)
original = lmm.decode_message(enc["expression"], enc["length"], enc["residuals"])
print(original)   # The Pharaohs encoded reality.

# Symbolic text continuation
predictor = lmm.TextPredictor(window_size=20, iterations=30, depth=3)
result    = predictor.predict("Wise AI built the first LMM", predict_length=80)
print(result["continuation"])

# Symbolic regression
sr      = lmm.SymbolicRegression(max_depth=3, iterations=50)
inputs  = [[i * 0.5] for i in range(10)]
targets = [2.0 * i * 0.5 + 1.0 for i in range(10)]
eq      = sr.fit(inputs, targets)
print(f"Discovered: {eq}")

# Consciousness (perceive → act)
brain     = lmm.Consciousness(state_len=4, lookahead=5, step_size=0.01)
new_state = brain.tick(bytes("The Pharaohs built the pyramids", "utf-8"))
print(new_state)

# Spectral image generation
path = lmm.render_image(
    "ancient egypt mathematics",
    width=512, height=512,
    style="plasma", palette="warm",
    output="egypt.ppm",
)
print(f"Saved to {path}")
```

## 📖 Full API Reference

### Classes

| Class                | Constructor                                                | Key Methods                                                                                                                                               |
| -------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Tensor`             | `Tensor(shape, data)`                                      | `.shape`, `.data`, `.norm()`, `.scale(s)`, `.add(t)`, `.dot(t)`                                                                                           |
| `Expression`         | `Expression.parse(s)`                                      | `.evaluate(bindings)`, `.diff(var)`, `.simplify()`, `str(expr)`                                                                                           |
| `CausalGraph`        | `CausalGraph()`                                            | `.add_node(name, val)`, `.add_edge(src, dst, weight)`, `.forward_pass()`, `.intervene(var, val)`, `.get_value(name)`, `.counterfactual(var, val, target)` |
| `HarmonicOscillator` | `HarmonicOscillator(omega, x0, v0)`                        | `.energy()`, `.state()`                                                                                                                                   |
| `LorenzSystem`       | `LorenzSystem(sigma, rho, beta, x0, y0, z0)`               | `.state()`                                                                                                                                                |
| `Pendulum`           | `Pendulum(g, l, theta0, omega0)`                           | `.state()`                                                                                                                                                |
| `SIRModel`           | `SIRModel(beta, gamma, S0, I0, R0)`                        | `.state()`                                                                                                                                                |
| `Simulator`          | `Simulator(step_size)`                                     | `.euler_step_osc(model, state)`, `.rk4_step_osc(model, state)`                                                                                            |
| `SymbolicRegression` | `SymbolicRegression(max_depth, iterations, pop_size?)`     | `.fit(inputs, targets) → str`                                                                                                                             |
| `TextPredictor`      | `TextPredictor(window_size?, iterations?, depth?)`         | `.predict(text, predict_length?) → dict`                                                                                                                  |
| `StochasticEnhancer` | `StochasticEnhancer(probability)`                          | `.enhance(text) → str`                                                                                                                                    |
| `SentenceGenerator`  | `SentenceGenerator(iterations?, depth?)`                   | `.generate(seed) → str`                                                                                                                                   |
| `ParagraphGenerator` | `ParagraphGenerator(sentence_count?, iterations?, depth?)` | `.generate(seed) → str`                                                                                                                                   |
| `TextSummarizer`     | `TextSummarizer(sentence_count?, iterations?, depth?)`     | `.summarize(text) → str`                                                                                                                                  |
| `Consciousness`      | `Consciousness(state_len, lookahead?, step_size?)`         | `.tick(bytes) → list[float]`                                                                                                                              |

### Free Functions

| Function                                                                        | Returns | Description                         |
| ------------------------------------------------------------------------------- | ------- | ----------------------------------- |
| `encode_text(text, iterations?, depth?)`                                        | `dict`  | `{expression, length, residuals}`   |
| `decode_message(expression, length, residuals)`                                 | `str`   | Reconstructs original text          |
| `mdl_score(expr, inputs, targets)`                                              | `float` | MDL fitness score                   |
| `compute_mse(expr, inputs, targets)`                                            | `float` | Mean squared error                  |
| `r_squared(expr, inputs, targets)`                                              | `float` | R² coefficient                      |
| `aic_score(n_params, log_likelihood)`                                           | `float` | Akaike information criterion        |
| `bic_score(n_params, n_samples, log_likelihood)`                                | `float` | Bayesian information criterion      |
| `render_image(prompt, width?, height?, style?, palette?, components?, output?)` | `str`   | Spectral field synthesis → PPM path |

### `TextPredictor.predict()` Return Dict

```python
{
    "continuation":        str,   # full text (seed + generated continuation)
    "trajectory_equation": str,   # GP equation driving word tone
    "rhythm_equation":     str,   # GP equation driving word length
    "window_used":         int,   # context window size actually used
}
```

### `encode_text()` Return Dict

```python
{
    "expression": str,       # symbolic equation string
    "length":     int,       # character count of original text
    "residuals":  list[int], # integer correction residuals (lossless)
}
```

## 📄 License

Licensed under the [MIT License](LICENSE).
