<div align="center">

# 👁️ LMM 🦀

[![LMM](https://wiseai.dev/assets/logo.png)](https://wiseai.dev)

[![Work In Progress](https://img.shields.io/badge/Work%20In%20Progress-orange)](https://github.com/wiseaidotdev/lmm)
[![Crates.io Downloads](https://img.shields.io/crates/d/lmm)](https://crates.io/crates/lmm)
[![made-with-rust](https://img.shields.io/badge/Made%20with-Rust-1f425f.svg?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Rust](https://img.shields.io/badge/Rust-1.86%2B-blue.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/wiseaidev)

[![Share On Reddit](https://img.shields.io/badge/share%20on-reddit-red?logo=reddit)](https://reddit.com/submit?url=https://github.com/wiseaidotdev/lmm&title=LMM%3A%20Large%20Mathematical%20Model%20%E2%80%94%20Encode%20Reality%20as%20Equations)
[![Share On X](https://img.shields.io/badge/share%20on-X-03A9F4?logo=x)](https://twitter.com/share?url=https://github.com/wiseaidotdev/lmm&text=LMM%3A%20Large%20Mathematical%20Model%20%E2%80%94%20Encode%20Reality%20as%20Equations)
[![Share On Linkedin](https://img.shields.io/badge/share%20on-linkedin-3949AB?logo=linkedin)](https://www.linkedin.com/shareArticle?url=https://github.com/wiseaidotdev/lmm&title=LMM%3A%20Large%20Mathematical%20Model)

> **LMM** is a pure‑Rust framework that represents higher‑dimensional realities through symbolic mathematics and physics simulation, inspired by the Pharaonic model of intelligence: compress the world into durable, universal equations.

</div>

## 🧠 Framework Overview

LMM bridges multimodal perception and actionable scientific discovery through five tightly integrated layers:

| Layer          | Modules                                       | Purpose                                                 |
| -------------- | --------------------------------------------- | ------------------------------------------------------- |
| **Perception** | `perception.rs`, `tensor.rs`                  | Raw bytes → normalised tensors                          |
| **Symbolic**   | `equation.rs`, `symbolic.rs`, `discovery.rs`  | GP symbolic regression, differentiation, simplification |
| **Physics**    | `physics.rs`, `simulation.rs`                 | ODE models + Euler / RK4 / RK45 / leapfrog integrators  |
| **Causal**     | `causal.rs`                                   | SCM graphs, do-calculus interventions, counterfactuals  |
| **Cognition**  | `consciousness.rs`, `world.rs`, `operator.rs` | Full perceive → encode → predict → act loop             |

### ⚙️ Architecture

```mermaid
flowchart TD
    A["Raw Input\n(bytes / sensors)"]
    B["MultiModalPerception\n──► Tensor"]
    C["Consciousness Loop\nperceive → encode → predict\nevaluate → plan (lookahead)"]
    D["WorldModel\n(RK4 physics)"]
    E["SymbolicRegression\n(GP equation search)"]
    F["CausalGraph\nintervention / counterfactual"]
    G["Expression AST\ndifferentiate / simplify"]

    A --> B --> C
    C --> D
    C --> E
    E --> G
    G --> F
    D --> F
```

### 🔬 Key Capabilities

- 🧬 **Genetic Programming**: real population-based symbolic regression that seeds templates (linear, quadratic, periodic) and enforces variable-containing equations.
- 📐 **Symbolic Calculus**: automatic differentiation (chain rule, product rule), constant folding simplification.
- 🌀 **Physics Suite**: Harmonic, Lorenz, Pendulum, SIR epidemic, N-body gravity: all implement `Simulatable`.
- 🔢 **Field Calculus**: N-D gradient, Laplacian, divergence, 3-D curl (central differences).
- 🔗 **Causal Reasoning**: structural causal models, `do(X=v)` interventions, counterfactual queries.
- 🧩 **Neural Operators**: circular convolution with SGD kernel learning, Fourier spectral operators.
- 🔤 **Text ↔ Equation**: encode any text into a symbolic equation; decode it back exactly (lossless via residuals).
- 🔮 **Symbolic Prediction**: LMM-native text continuation via sliding-window GP regression and vocabulary anchoring.

## 📦 Installation

### From Source

```sh
git clone https://github.com/wiseaidotdev/lmm
cd lmm
cargo build --release
```

The binary is at `./target/release/lmm`.

### Via Cargo

```sh
cargo install lmm --all-features
```

> [!NOTE]
> Requires Rust 1.86+. Install via [rustup](https://rustup.rs).

## 🚀 CLI Usage

```sh
lmm <SUBCOMMAND> [OPTIONS]

Subcommands:
  simulate     Run a harmonic oscillator simulation
  physics      Run a named physics model (lorenz | pendulum | sir | harmonic)
  discover     Discover an equation from synthetic data using GP
  consciousness  Run a perceive→predict→act consciousness loop tick
  causal       Build a causal graph and apply do-calculus intervention
  field        Compute gradient or Laplacian of a scalar field
  encode       Encode text into a symbolic mathematical equation
  decode       Decode a symbolic equation back to text
  predict      Predict text continuation via sliding-window symbolic regression
```

## 📖 Subcommand Reference

### 1. `simulate`: Harmonic Oscillator

Runs a harmonic oscillator using the RK4 integrator.

```sh
lmm simulate --step 0.01 --steps 200
```

```sh
Simulated 200 steps with step_size=0.01
Final state: [-0.41614683639502004, -0.9092974268937748]
```

| Flag            | Default | Description                 |
| --------------- | ------- | --------------------------- |
| `-s`, `--step`  | `0.01`  | Integration step size (Δt)  |
| `-t`, `--steps` | `100`   | Number of integration steps |

### 2. `physics`: Physics Model Simulation

Simulate one of four built-in physics models.

```sh
# Lorenz chaotic attractor (σ=10, ρ=28, β=8/3)
lmm physics --model lorenz --steps 500 --step-size 0.01

# Nonlinear pendulum
lmm physics --model pendulum --steps 300 --step-size 0.005

# SIR epidemic model
lmm physics --model sir --steps 1000 --step-size 0.5

# Damped harmonic oscillator (default)
lmm physics --model harmonic --steps 200
```

**Lorenz example:**

```sh
Lorenz: 500 steps. Final xyz: [-8.900269690476492, -7.413716837503834, 29.311877708359006]
```

**SIR example:**

```sh
SIR: 1000 steps. Final [S,I,R]: [58.797367656865795, 7.649993277129408e-15, 941.2026323431321]
```

| Flag                | Default    | Description                                    |
| ------------------- | ---------- | ---------------------------------------------- |
| `-m`, `--model`     | `harmonic` | Model: `lorenz`, `pendulum`, `sir`, `harmonic` |
| `-s`, `--steps`     | `200`      | Number of integration steps                    |
| `-z`, `--step-size` | `0.01`     | Step size Δt                                   |

### 3. `discover`: Symbolic Regression

Runs Genetic Programming (GP) to discover a symbolic equation from data.

```sh
lmm discover --iterations 200
```

```sh
Discovered equation: (x + (1.002465056833142 + x))
```

The engine fits data points `(i*0.5, 2*i*0.5 + 1)` by default and finds the
underlying linear law. Increase `--iterations` for more complex datasets.

| Flag                 | Default     | Description                                      |
| -------------------- | ----------- | ------------------------------------------------ |
| `-d`, `--data-path`  | `synthetic` | Data source (`synthetic` = built-in linear data) |
| `-i`, `--iterations` | `100`       | Number of GP evolution iterations                |

### 4. `consciousness`: Perceive → Predict → Act Loop

Runs one tick of the full consciousness loop: raw bytes → perception tensor →
world model prediction → action plan.

```sh
lmm consciousness --lookahead 5
```

```sh
Consciousness ticked. New state: [0.0019607843137254832, -0.24901960784313726, -0.37450980392156863, 0.5]
Mean prediction error: 0
```

| Flag                | Default | Description                        |
| ------------------- | ------- | ---------------------------------- |
| `-l`, `--lookahead` | `3`     | Multi-step lookahead horizon depth |

### 5. `causal`: Causal Graph + do-Calculus

Builds a 3-node Structural Causal Model (`x → y → z`) and applies an
intervention `do(node = value)`, printing before/after values.

```sh
# Intervene on x: set x = 10, observe how y and z change
lmm causal --intervene-node x --intervene-value 10.0
```

```sh
Before intervention: x=Some(3.0), y=Some(6.0), z=Some(7.0)
After do(x=10): x=Some(10.0), y=Some(20.0), z=Some(21.0)
```

The SCM is:

- `y = 2 * x`
- `z = y + 1`

| Flag                      | Default | Description                            |
| ------------------------- | ------- | -------------------------------------- |
| `-n`, `--intervene-node`  | `x`     | Name of the node to intervene on       |
| `-v`, `--intervene-value` | `1.0`   | Value to set the node to (do-calculus) |

### 6. `field`: Scalar Field Calculus

Computes differential operators on a 1-D scalar field `f(i) = i²`.

```sh
# Gradient: should approach 2i (central differences)
lmm field --size 8 --operation gradient

# Laplacian: should be ≈ 2 everywhere (second derivative of x²)
lmm field --size 8 --operation laplacian
```

```sh
Gradient of x²: [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 13.0]
```

```sh
Laplacian of x²: [0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0]
```

| Flag                | Default    | Description                          |
| ------------------- | ---------- | ------------------------------------ |
| `-s`, `--size`      | `10`       | Number of field points               |
| `-o`, `--operation` | `gradient` | Operation: `gradient` or `laplacian` |

### 7. `encode`: Text → Symbolic Equation

This is the flagship demonstration of LMM's power. Any text is treated as a
sequence of byte values indexed by position. The GP engine discovers a symbolic
equation `f(x) ≈ byte[x]`. Integer residuals `(byte[x] − round(f(x)))` are
stored alongside the equation, guaranteeing **lossless round-trip recovery**.

```sh
lmm encode --text "The Pharaohs encoded reality in mathematics." \
           --iterations 150 --depth 5
```

```sh
━━━ LMM ENCODER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input text  : "The Pharaohs encoded reality in mathematics."
Characters  : 44
Running GP symbolic regression (150 iterations, depth 5)…

Equation: (95.09620435614187 - cos(x))
Length: 44 chars
MSE: 646.3067
Max residual: 64

━━━ ENCODED DATA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{"eq":"(95.09620435614187 - cos(x))","len":44,"mse":646.306722,"res":[-10,9,5,-64,-16,9,3,20,2,15,8,20,-62,7,15,3,15,5,7,6,-63,18,5,1,13,11,22,26,-64,9,15,-62,15,2,20,8,6,15,3,21,9,3,20,-49]}

━━━ VERIFY ROUND-TRIP ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Decoded text: "The Pharaohs encoded reality in mathematics."
Round-trip  : ✅ PERFECT

To decode later, run:
  lmm decode --equation "(95.09620435614187 - cos(x))" --length 44 --residuals "-10,9,5,-64,-16,9,3,20,2,15,8,20,-62,7,15,3,15,5,7,6,-63,18,5,1,13,11,22,26,-64,9,15,-62,15,2,20,8,6,15,3,21,9,3,20,-49"
```

> [!NOTE]
> GP is stochastic: the discovered equation and residual values will differ across runs. The round-trip recovery is always ✅ PERFECT because the integer residuals correct for any approximation error.

```sh
# Encode from a file
lmm encode --input ./my_message.txt --iterations 200 --depth 5
```

| Flag            | Default       | Description                                        |
| --------------- | ------------- | -------------------------------------------------- |
| `-i`, `--input` | `-`           | Path to a text file to encode (`-` = use `--text`) |
| `-t`, `--text`  | `Hello, LMM!` | Inline text (used when `--input` is `-`)           |
| `--iterations`  | `80`          | GP evolution iterations                            |
| `--depth`       | `4`           | Maximum expression tree depth                      |

### 8. `decode`: Symbolic Equation → Text

Reconstructs the original text from the equation and residuals printed by `encode`.

```sh
lmm decode \
  --equation "(95.09620435614187 - cos(x))" \
  --length 44 \
  --residuals "-10,9,5,-64,-16,9,3,20,2,15,8,20,-62,7,15,3,15,5,7,6,-63,18,5,1,13,11,22,26,-64,9,15,-62,15,2,20,8,6,15,3,21,9,3,20,-49"
```

```sh
━━━ LMM DECODER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Equation : (95.09620435614187 - cos(x))
Length   : 44

━━━ DECODED TEXT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The Pharaohs encoded reality in mathematics.
```

| Flag                | Required | Description                                                                 |
| ------------------- | -------- | --------------------------------------------------------------------------- |
| `-e`, `--equation`  | ✅       | Equation string (from `encode` output)                                      |
| `-l`, `--length`    | ✅       | Number of characters to recover                                             |
| `-r`, `--residuals` | ✅       | Comma-separated residuals. Use `--residuals="-3,1,..."` for negative values |

> [!IMPORTANT]
> Use `--residuals="-3,..."` (with `=`) or quote the argument when residuals contain negative values to prevent the shell from treating them as flags.

### 9. `predict`: Symbolic Text Continuation

Uses a **three-signal symbolic engine** to predict what comes next:

1. **Suffix-pattern matching**: detects phrases from the context that match the current tail of generated text and continues them directly (prevents nonsense).
1. **2nd-order Markov word-transition matrix**: `(word_a, word_b) → word_c` captures phrase-level grammar; falls back to 1st-order when unseen.
1. **GP trajectory + rhythm equations**: score word candidates by positional pattern and sentence cadence as tie-breakers.

```sh
lmm predict --text "Large Mathematical Models compress the world into equations" \
            --window 16 --predict-length 64
```

```sh
━━━ LMM PREDICTOR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input text  : "Large Mathematical Models compress the world into equations"
Window used : 8 words
Trajectory  : (((x / 1.0038639598569228) / 0.9977330128790676) / 0.9977330128790676)
Rhythm      : (sin(x) + (sin(x) + 6.380064552234503))

━━━ PREDICTED CONTINUATION ━━━━━━━━━━━━━━━━━━━━━━━
Large Mathematical Models compress the world into equations compress the world into equations Models compress the world into
```

> [!NOTE]
> The suffix-pattern engine detects that "equations" matches position 7 in the context,
> so the continuation follows the actual documented phrase structure of the input.
> For longer, richer inputs the Markov bigram transitions dominate, producing novel
> phrase combinations grounded in the input's grammar.

| Flag                     | Default                           | Description                                      |
| ------------------------ | --------------------------------- | ------------------------------------------------ |
| `-i`, `--input`          | `-`                               | Path to a text file (`-` = use `--text`)         |
| `-t`, `--text`           | `The Pharaohs encoded reality in` | Inline text seed                                 |
| `-w`, `--window`         | `32`                              | Context window in words                          |
| `-p`, `--predict-length` | `16`                              | Approximate character budget for continuation    |
| `--iterations`           | `80`                              | GP evolution iterations for the prediction model |
| `--depth`                | `4`                               | Maximum expression tree depth                    |

## 🔬 Architecture Deep Dive

### Genetic Programming Symbolic Regression

```mermaid
flowchart TD
    A["Seeded population\n(linear/quadratic/periodic templates + random)"]
    B["Evaluate fitness\nMDL = n·ln(MSE) + complexity·ln(2)"]
    C["Tournament selection (k=5)"]
    D["Crossover & mutation"]
    E["Reject constant collapse\n(inject fresh random expr 70% of the time)"]
    F{Iterations done?}
    G["Best variable-containing expression (simplified)"]

    A --> B --> C --> D --> E --> F
    F -- No --> B
    F -- Yes --> G
```

### Multi-Signal Prediction Engine

```mermaid
flowchart TD
    In["Context Window\n(Recent Tokens)"]

    In -->|Train| M["2nd-Order Markov Chain\nTransition probabilities"]
    In -->|GP Fit| T["Word-ID Trajectory GP\nf(pos) ≈ word_id"]
    In -->|GP Fit| R["Word-Length Rhythm GP\ng(pos) ≈ length"]

    In --> S["Suffix Pattern Matcher"]

    S -- "Match Found" --> Out["Exact Phrase Continuation"]
    S -- "No Match" --> Score["Composite Scorer"]

    M --> Score
    T --> Score
    R --> Score

    Score -->|Lowest Score| W["Select Best Word from Vocab"]
    W -->|Update Recency| Out
```

### RK45 Adaptive Integrator

All Butcher-tableau coefficients are named package-level constants:

```rust
const RK45_A41: f64 = 1932.0 / 2197.0;
const RK45_A42: f64 = -7200.0 / 2197.0;
const RK45_A43: f64 = 7296.0 / 2197.0;
const RK45_B5_1: f64 = 16.0 / 135.0;
// ... etc.
```

Step size is adapted each iteration using the error estimate:

```sh
h_new = 0.9 · h · (tol / error)^0.2
```

## 📰 Whitepaper

[LLMs are Usefull. LMMs will Break Reality](https://wiseai.dev/blogs/llms-are-usefull-lmms-will-break-reality): the blog post that started this project.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or pull requests.

## 📝 License

This project is licensed under the MIT License: see the [LICENSE](LICENSE) file for details.
