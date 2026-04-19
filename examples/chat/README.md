# VECT: Variable Equation Computation Technology

> A Yew CSR chat interface demonstrating the **LMM** (Large Mathematical Model) framework for deterministic and stochastic text generation.

## Features

| Mode             | Description                                                 |
| ---------------- | ----------------------------------------------------------- |
| ✍️ **Sentence**  | Deterministic single sentence from a seed (no AI training)  |
| 📄 **Paragraph** | Multi-sentence paragraph via equation-driven word selection |
| 📖 **Essay**     | Structured essay with symbolic title + paragraphs           |
| ✂️ **Summarize** | Extract key sentences from any input corpus                 |
| 🔮 **Predict**   | Symbolic text continuation via GP trajectory regression     |
| 🌐 **Ask**       | Live DuckDuckGo web search + LMM summarization              |

### Stochastic Enhancement

Enable the **Stochastic** toggle to apply synonym replacement at a configurable probability (1 - 100%). Each word has a chance of being replaced from a curated synonym table - producing unique, natural-sounding variations of deterministic output.

## Running

```bash
cd chat
trunk serve --port 3000
# Open http://localhost:3000
```

## Building for Production

```bash
trunk build
# Output in dist/
```
