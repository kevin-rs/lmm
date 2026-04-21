# LMM Node.js Documentation 🟩

The **`@wiseaidev/lmm`** package offers native Node.js bindings for the `lmm` Rust engine via [napi-rs](https://napi.rs/). A fully embedded Tokio runtime means all calls are synchronous; No Promises or callbacks needed for one-off calls.

## 📦 Installation

```sh
npm install @wiseaidev/lmm
```

Or build locally from source:

```sh
npm install -g @napi-rs/cli
npm run build        # builds and places the .node file in the project root
```

> [!NOTE]
> Requires Node.js 22+ and a pre-built `.node` binary for your platform. Binaries ship for Linux (x64 / arm64), macOS (x64 / Apple Silicon), and Windows (x64).

## 🛠 Quick Start

```sh
node
```

```js
// npm install: const lmm = require("@wiseaidev/lmm");
const lmm = require(".");

// Tensor arithmetic
const t = new lmm.Tensor([2, 3], [1, 2, 3, 4, 5, 6]);
console.log(t.shape); // [2, 3]
console.log(t.norm());

// Symbolic expression
const expr = lmm.Expression.parse("(sin(x) * 2)");
console.log(expr.evaluate({ x: Math.PI })); // ≈ 0
console.log(expr.diff("x").simplify().toString()); // (cos(x) * 2)

// Causal graph
const g = new lmm.CausalGraph();
g.addNode("x", 3.0);
g.addNode("y", null);
g.addEdge("x", "y", 2.0);
g.forwardPass();
console.log(g.counterfactual("x", 10.0, "y")); // 20.0

// Physics simulation
const osc = new lmm.HarmonicOscillator(1.0, 1.0, 0.0);
const sim = new lmm.Simulator(0.01);
const s = sim.rk4StepOsc(osc, new lmm.Tensor([2], osc.state()));
console.log(s.data);

// Text encode / decode (lossless)
const enc = lmm.encodeText("The Pharaohs encoded reality.", 80, 4);
const dec = lmm.decodeMessage(enc.expression, enc.length, enc.residuals);
console.log(dec); // The Pharaohs encoded reality.

// Symbolic text continuation
const predictor = new lmm.TextPredictor(20, 30, 3);
const result = predictor.predict("Wise AI built the first LMM", 80);
console.log(result.continuation);

// Symbolic regression
const sr = new lmm.SymbolicRegression(3, 50, 50);
const eq = sr.fit([[0.5], [1.0], [1.5]], [2.0, 3.0, 4.0]);
console.log(eq);

// Consciousness (perceive → act)
const brain = new lmm.Consciousness(4, 5, 0.01);
const state = brain.tick(Buffer.from("Hello, LMM!"));
console.log(state);

// Spectral image generation
const path = lmm.renderImage(
  "ancient egypt",
  512,
  512,
  "warm",
  "plasma",
  8,
  "out.ppm",
);
console.log("Saved:", path);
```

## 📖 Full API Reference

### Classes

| Export               | Constructor                                                   | Key Methods                                                                                                                                                             |
| -------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Tensor`             | `new Tensor(shape, data)`                                     | `.shape`, `.data`, `.norm()`, `.scale(s)`, `.add(t)`, `.dot(t)`                                                                                                         |
| `Expression`         | `Expression.parse(s)`                                         | `.evaluate(obj)`, `.diff(var)`, `.simplify()`, `.toString()`                                                                                                            |
| `CausalGraph`        | `new CausalGraph()`                                           | `.addNode(name, val)`, `.addEdge(src, dst, w)`, `.forwardPass()`, `.intervene(var, val)`, `.getValue(name)`, `.counterfactual(var, val, target)`, `.topologicalOrder()` |
| `HarmonicOscillator` | `new HarmonicOscillator(omega, x0, v0)`                       | `.omega`, `.energy()`, `.state()`                                                                                                                                       |
| `Simulator`          | `new Simulator(stepSize)`                                     | `.eulerStepOsc(model, state)`, `.rk4StepOsc(model, state)`                                                                                                              |
| `SymbolicRegression` | `new SymbolicRegression(maxDepth, iterations, popSize?)`      | `.fit(inputs, targets) → string`                                                                                                                                        |
| `TextPredictor`      | `new TextPredictor(windowSize?, iterations?, depth?)`         | `.predict(text, predictLength?) → PredictionResult`                                                                                                                     |
| `StochasticEnhancer` | `new StochasticEnhancer(probability)`                         | `.enhance(text) → string`                                                                                                                                               |
| `SentenceGenerator`  | `new SentenceGenerator(iterations?, depth?)`                  | `.generate(seed) → string`                                                                                                                                              |
| `ParagraphGenerator` | `new ParagraphGenerator(sentenceCount?, iterations?, depth?)` | `.generate(seed) → string`                                                                                                                                              |
| `TextSummarizer`     | `new TextSummarizer(sentenceCount?, iterations?, depth?)`     | `.summarize(text) → string`                                                                                                                                             |
| `Consciousness`      | `new Consciousness(stateLen, lookahead?, stepSize?)`          | `.tick(Buffer) → number[]`                                                                                                                                              |

### Free Functions

| Function                                                                       | Returns  | Description                         |
| ------------------------------------------------------------------------------ | -------- | ----------------------------------- |
| `encodeText(text, iterations?, depth?)`                                        | `object` | `{expression, length, residuals}`   |
| `decodeMessage(expression, length, residuals)`                                 | `string` | Reconstructs original text          |
| `mdlScore(expr, inputs, targets)`                                              | `number` | MDL fitness score                   |
| `computeMse(expr, inputs, targets)`                                            | `number` | Mean squared error                  |
| `rSquared(expr, inputs, targets)`                                              | `number` | R² coefficient                      |
| `aicScore(nParams, logLikelihood)`                                             | `number` | Akaike information criterion        |
| `bicScore(nParams, nSamples, logLikelihood)`                                   | `number` | Bayesian information criterion      |
| `renderImage(prompt, width?, height?, palette?, style?, components?, output?)` | `string` | Spectral field synthesis → PPM path |

### `TextPredictor.predict()` Return Object

```ts
{
  continuation:       string,  // seed + generated continuation
  trajectoryEquation: string,  // GP equation driving word tone
  rhythmEquation:     string,  // GP equation driving word length
  windowUsed:         number,  // context window size actually used
}
```

### `encodeText()` Return Object

```ts
{
  expression: string,    // symbolic equation string
  length:     number,    // character count of original text
  residuals:  number[],  // integer correction residuals (lossless)
}
```

## 📄 License

Licensed under the [MIT License](LICENSE).
