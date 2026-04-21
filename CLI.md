# LMM Command-Line Interface 💻

The `lmm` binary provides 15 subcommands covering simulation, symbolic regression, causal inference, encoding, prediction, and rich text generation: All powered by pure deterministic mathematics.

## 📦 Installation

```sh
cargo install lmm --features rust-binary
```

To enable the internet-aware `ask` command:

```sh
cargo install lmm --features rust-binary,net
# or build from source:
cargo build --release --features cli,net
```

The resulting binary is named `lmm`.

## 🛠 Usage

```sh
lmm <COMMAND> [OPTIONS]
```

Global flags available on `predict`, `summarize`, `sentence`, `paragraph`, `essay`, `ask`:

| Flag              | Default | Description                                    |
| ----------------- | ------- | ---------------------------------------------- | --- |
| `--stochastic`    | `false` | Enable synonym-based output randomization      |
| `--probability`   | `0.5`   | Word replacement rate (0.0 = none, 1.0 = all)  |
| `-v`, `--verbose` | `false` | Show full banners, seeds, and boxed separators | :   |

## 📖 Subcommand Reference

### 1. `simulate`: Harmonic Oscillator

Runs a damped harmonic oscillator using the RK4 integrator.

```sh
lmm simulate --step 0.01 --steps 200
```

| Flag            | Default | Description                 |
| --------------- | ------- | --------------------------- | --- |
| `-s`, `--step`  | `0.01`  | Integration step size (Δt)  |
| `-t`, `--steps` | `100`   | Number of integration steps | :   |

### 2. `physics`: Physics Model Simulation

Simulate one of four built-in physics models.

```sh
lmm physics --model lorenz --steps 500 --step-size 0.01
lmm physics --model pendulum --steps 300 --step-size 0.005
lmm physics --model sir --steps 1000 --step-size 0.5
lmm physics --model harmonic --steps 200
```

| Flag                | Default    | Description                             |
| ------------------- | ---------- | --------------------------------------- | --- |
| `-m`, `--model`     | `harmonic` | `lorenz`, `pendulum`, `sir`, `harmonic` |
| `-s`, `--steps`     | `200`      | Number of integration steps             |
| `-z`, `--step-size` | `0.01`     | Step size Δt                            | :   |

### 3. `discover`: Symbolic Regression

Runs Genetic Programming to discover a symbolic equation from data.

```sh
lmm discover --iterations 200
```

| Flag                 | Default     | Description                                      |
| -------------------- | ----------- | ------------------------------------------------ | --- |
| `-d`, `--data-path`  | `synthetic` | Data source (`synthetic` = built-in linear data) |
| `-i`, `--iterations` | `100`       | Number of GP evolution iterations                | :   |

### 4. `consciousness`: Perceive → Predict → Act Loop

Runs one tick of the full consciousness loop: raw bytes → perception tensor → world model prediction → action plan.

```sh
lmm consciousness --lookahead 5
```

| Flag                | Default | Description                        |
| ------------------- | ------- | ---------------------------------- | --- |
| `-l`, `--lookahead` | `3`     | Multi-step lookahead horizon depth | :   |

### 5. `causal`: Causal Graph + do-Calculus

Builds a 3-node Structural Causal Model (`x → y → z`) and applies an intervention `do(node = value)`.

```sh
lmm causal --intervene-node x --intervene-value 10.0
```

| Flag                      | Default | Description                      |
| ------------------------- | ------- | -------------------------------- | --- |
| `-n`, `--intervene-node`  | `x`     | Name of the node to intervene on |
| `-v`, `--intervene-value` | `1.0`   | Value to set (do-calculus)       | :   |

### 6. `field`: Scalar Field Calculus

Computes differential operators on a 1-D scalar field `f(i) = i²`.

```sh
lmm field --size 8 --operation gradient
lmm field --size 8 --operation laplacian
```

| Flag                | Default    | Description               |
| ------------------- | ---------- | ------------------------- | --- |
| `-s`, `--size`      | `10`       | Number of field points    |
| `-o`, `--operation` | `gradient` | `gradient` or `laplacian` | :   |

### 7. `encode`: Text → Symbolic Equation

Encodes any text string into a symbolic equation with integer residuals, guaranteeing a **lossless round-trip**.

```sh
lmm encode --text "The Pharaohs encoded reality." --iterations 150 --depth 5
lmm encode --input ./my_message.txt --iterations 200 --depth 5
```

| Flag            | Default       | Description                              |
| --------------- | ------------- | ---------------------------------------- |
| `-i`, `--input` | `-`           | Path to a text file (`-` = use `--text`) |
| `-t`, `--text`  | `Hello, LMM!` | Inline text seed                         |
| `--iterations`  | `80`          | GP evolution iterations                  |
| `--depth`       | `4`           | Maximum expression tree depth            |

> [!NOTE]
> GP is stochastic: the discovered equation and residuals differ across runs. The round-trip recovery is always ✅ PERFECT because integer residuals correct for all approximation error.:

### 8. `decode`: Symbolic Equation → Text

Reconstructs the original text from the equation and residuals printed by `encode`.

```sh
lmm decode \
  --equation "((4.421727800598272 * x) + 0.9076671306697397)" \
  --length 29 \
  --residuals="83,99,91,18,61,81,70,82,61,70,59,65,-22,43,47,32,39,24,21,15,-57,20,3,-6,1,-6,0,1,-79"
```

| Flag                | Required | Description                                                          |
| ------------------- | -------- | -------------------------------------------------------------------- |
| `-e`, `--equation`  | ✅       | Equation string (from `encode` output)                               |
| `-l`, `--length`    | ✅       | Number of characters to recover                                      |
| `-r`, `--residuals` | ✅       | Comma-separated residuals (use `--residuals="-3,..."` for negatives) |

> [!IMPORTANT]
> Use `--residuals="..."` (with `=`) or quote the argument when residuals contain negative values to prevent the shell treating them as flags.:

### 9. `predict`: Symbolic Text Continuation

Continues a text seed using three deterministic mathematical signals: a GP Trajectory Equation, a GP Rhythm Equation, and a SVO grammar engine anchored to a dictionary vocabulary.

```sh
lmm predict --text "Wise AI built the first LMM" --window 10 --predict-length 180
```

| Flag                     | Default                           | Description                                   |
| ------------------------ | --------------------------------- | --------------------------------------------- | --- |
| `-i`, `--input`          | `-`                               | Path to a text file (`-` = use `--text`)      |
| `-t`, `--text`           | `The Pharaohs encoded reality in` | Inline seed                                   |
| `-w`, `--window`         | `32`                              | Context window in words                       |
| `-p`, `--predict-length` | `16`                              | Approximate character budget for continuation |
| `--iterations`           | `80`                              | GP iterations for the prediction model        |
| `--depth`                | `4`                               | Maximum GP expression tree depth              |
| `--stochastic`           | `false`                           | Enable synonym-based randomization            |
| `--probability`          | `0.5`                             | Word replacement rate                         | :   |

### 10. `summarize`: Key Sentence Extraction

Distils a body of text to its most mathematically significant sentences, scored by tone deviation, length variance, and relative position.

```sh
lmm summarize --text "..." --sentences 2
```

| Flag                | Default  | Description                        |
| ------------------- | -------- | ---------------------------------- | --- |
| `-t`, `--text`      | required | Input text to summarize            |
| `-n`, `--sentences` | `2`      | Number of key sentences to extract |
| `--stochastic`      | `false`  | Enable synonym-based randomization |
| `--probability`     | `0.5`    | Word replacement rate              | :   |

### 11. `sentence`: Single Sentence Generation

Generates a single structurally elegant sentence from a seed topic.

```sh
lmm sentence --text "Mathematics is the language of the universe"
lmm sentence --text "Mathematics is the language of the universe" --stochastic
```

| Flag            | Default  | Description                           |
| --------------- | -------- | ------------------------------------- | --- |
| `-t`, `--text`  | required | Seed topic                            |
| `--stochastic`  | `false`  | Randomize word synonyms each run      |
| `--probability` | `0.5`    | Fraction of eligible words to replace | :   |

### 12. `paragraph`: Cohesive Paragraph Generation

Chains logically coherent sentences seeded by keywords extracted from the original prompt.

```sh
lmm paragraph --text "Equations reveal hidden truths about nature" --sentences 6
```

| Flag                | Default  | Description                           |
| ------------------- | -------- | ------------------------------------- | --- |
| `-t`, `--text`      | required | Seed topic                            |
| `-n`, `--sentences` | `4`      | Number of sentences                   |
| `--stochastic`      | `false`  | Randomize word synonyms each run      |
| `--probability`     | `0.5`    | Fraction of eligible words to replace | :   |

### 13. `essay`: Full Essay Blueprint

Generates a fully structured essay: introduction, mathematical body paragraphs, and a conclusion.

```sh
lmm essay --text "Symmetry and the deeper patterns of physics" --paragraphs 2 --sentences 15
```

| Flag                 | Default  | Description                           |
| -------------------- | -------- | ------------------------------------- | --- |
| `-t`, `--text`       | required | Topic or title seed                   |
| `-n`, `--paragraphs` | `2`      | Number of body paragraphs             |
| `-s`, `--sentences`  | `3`      | Sentences per paragraph               |
| `--stochastic`       | `false`  | Randomize word synonyms each run      |
| `--probability`      | `0.5`    | Fraction of eligible words to replace | :   |

### 14. `ask`: Internet-Aware Knowledge Synthesis _(requires `net` feature)_

Searches DuckDuckGo Lite, aggregates result snippets into a corpus, then applies GP scoring to extract and compose the most mathematically significant sentences into a coherent response.

```sh
lmm ask --prompt "What is the Rust programming language?" --limit 5 --sentences 3
```

| Flag                | Default  | Description                               |
| ------------------- | -------- | ----------------------------------------- | --- |
| `-p`, `--prompt`    | required | The question or search query              |
| `-l`, `--limit`     | `5`      | Maximum number of search results to fetch |
| `-n`, `--sentences` | `3`      | Number of key sentences to extract        |
| `--region`          | `wt-wt`  | DuckDuckGo region code (e.g. `us-en`)     |
| `--iterations`      | `40`     | GP scoring iterations                     |
| `--depth`           | `3`      | Maximum GP expression depth               |
| `--stochastic`      | `false`  | Randomize word synonyms in the answer     |
| `--probability`     | `0.5`    | Fraction of eligible words to replace     | :   |

### 15. `imagen`: Spectral Field Synthesis Image Generation

Generates a PPM image from a text prompt by hashing it into Fourier wave components, applying a non-linear style transform, and mapping amplitudes to RGB.

```sh
lmm imagen \
  --prompt "The ancient Egyptians built the pyramids with mathematical precision" \
  --width 512 --height 512 \
  --style plasma --palette warm \
  --components 12 --output ./egypt.ppm
```

| Flag                 | Default      | Description                                              |
| -------------------- | ------------ | -------------------------------------------------------- |
| `-p`, `--prompt`     | required     | Text prompt to hash into the spectral seed               |
| `--width`            | `512`        | Image width in pixels                                    |
| `--height`           | `512`        | Image height in pixels                                   |
| `-c`, `--components` | `8`          | Number of cosine wave components per channel             |
| `-s`, `--style`      | `plasma`     | `wave`, `radial`, `orbital`, `fractal`, `flow`, `plasma` |
| `--palette`          | `auto`       | `warm`, `cool`, `neon`, `mono`, `auto`                   |
| `-o`, `--output`     | `output.ppm` | Output file path (auto-named if a directory)             |

## 📄 License

Licensed under the [MIT License](LICENSE).
