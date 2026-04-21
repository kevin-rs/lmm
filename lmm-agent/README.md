<div align="center">

# 🤖 lmm-agent

[![Crates.io](https://img.shields.io/crates/v/lmm-agent.svg)](https://crates.io/crates/lmm-agent)
[![Docs.rs](https://docs.rs/lmm-agent/badge.svg)](https://docs.rs/lmm-agent)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](../LICENSE)

> `lmm-agent` is an equation-based, training-free autonomous agent framework built on top of `lmm`. Agents reason through the LMM symbolic engine: no LLM API key, no token quotas, no stochastic black boxes.

</div>

## 🤔 What does this crate provide?

- **`LmmAgent`**: the batteries-included core agent with hot memory, long-term memory (LTM), tools, planner, reflection, and a time-based scheduler.
- **`Auto` derive macro**: zero-boilerplate `Agent`, `Functions`, and `AsyncFunctions` implementation. Only `agent: LmmAgent` is required in the struct.
- **`AutoAgent` orchestrator**: manages a heterogeneous pool of agents, running them concurrently with a configurable retry policy.
- **`agents![]` macro**: ergonomic syntax to declare a typed `Vec<Box<dyn Executor>>`.
- **`ThinkLoop`**: closed-loop PI controller that drives iterative reasoning toward a goal using Jaccard-error feedback.
- **DuckDuckGo search** (optional): built-in web search via the `duckduckgo` crate (`--features net`). When real snippets are available, they are returned directly as factual output.
- **Symbolic generation**: `AsyncFunctions::generate` uses `TextPredictor`, a symbolic regression engine that fits tone and rhythm trajectories to produce text. No neural model, no weights.

## 📦 Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
lmm-agent = "0.0.2"
```

Or enable it as a feature from the root `lmm` workspace:

```toml
[dependencies]
lmm = { version = "0.2.1", features = ["agent"] }
```

## 🚀 Quick Start

### 1. Define a custom agent

Your struct only needs one field: `agent: LmmAgent`. Everything else is derived automatically by `#[derive(Auto)]`.

```rust
use lmm_agent::prelude::*;

#[derive(Debug, Default, Auto)]
pub struct ResearchAgent {
    pub agent: LmmAgent,
}

#[async_trait]
impl Executor for ResearchAgent {
    async fn execute<'a>(
        &'a mut self,
        _task:      &'a mut Task,
        _execute:    bool,
        _browse:     bool,
        _max_tries:  u64,
    ) -> Result<()> {
        let prompt   = self.agent.behavior.clone();
        let response = self.generate(&prompt).await?;
        println!("{response}");
        self.agent.add_message(Message::new("assistant", response.clone()));
        let _ = self.save_ltm(Message::new("assistant", response)).await;
        self.agent.update(Status::Completed);
        Ok(())
    }
}
```

### 2. Run the agent

```rust
#[tokio::main]
async fn main() {
    let agent = ResearchAgent::new(
        "Research Agent".into(),
        "Explore the Rust ecosystem.".into(),
    );

    AutoAgent::default()
        .with(agents![agent])
        .max_tries(3)
        .build()
        .unwrap()
        .run()
        .await
        .unwrap();
}
```

## 🧠 Core Concepts

| Concept     | Description                                                                |
| ----------- | -------------------------------------------------------------------------- |
| `persona`   | The agent's identity / role label (e.g. `"Research Agent"`)                |
| `behavior`  | The agent's mission or goal description                                    |
| `LmmAgent`  | Core struct holding all state (memory, tools, planner, knowledge, profile) |
| `Message`   | A single chat-style message (`role` + `content`)                           |
| `Status`    | `Idle` → `Active` → `Completed` (or `InUnitTesting`, `Thinking`)          |
| `Auto`      | Derive macro that auto-implements `Agent`, `Functions`, `AsyncFunctions`   |
| `Executor`  | The only trait you must implement, contains your custom task logic         |
| `AutoAgent` | The orchestrator that runs a pool of `Executor`s                           |
| `ThinkLoop` | PI-controller feedback loop that drives iterative multi-step reasoning     |

## 🔧 LmmAgent Builder API

```rust
use lmm_agent::agent::LmmAgent;
use lmm_agent::types::{Status, Message, Planner, Goal};

let agent = LmmAgent::builder()
    .persona("Research Agent")
    .behavior("Explore symbolic AI.")
    .status(Status::Idle)
    .memory(vec![Message::new("system", "You are a symbolic reasoner.")])
    .planner(Planner {
        current_plan: vec![Goal {
            description: "Survey equation-based agents.".into(),
            priority: 1,
            completed: false,
        }],
    })
    .build();
```

## 📡 AsyncFunctions Trait

The `Auto` macro generates a full `AsyncFunctions` implementation for your struct:

| Method             | Description                                                                                |
| ------------------ | ------------------------------------------------------------------------------------------ |
| `generate(prompt)` | Symbolic text generation via `TextPredictor` (tone + rhythm regression). No LLM.          |
| `search(query)`    | DuckDuckGo web search (`--features net`). Returns real sentences when available.           |
| `save_ltm(msg)`    | Persist a message to the agent's long-term memory store                                    |
| `get_ltm()`        | Retrieve all LTM messages as a `Vec<Message>`                                              |
| `ltm_context()`    | Format LTM as a single context string                                                      |

## 🔬 How Generation Works

`AsyncFunctions::generate` dispatches to `LmmAgent::generate`, which uses the `TextPredictor` engine:

1. **Seed enrichment**: the prompt is enriched with domain-specific words extracted from the agent's own `behavior` field, so generation is topically grounded.
1. **Tone trajectory**: symbolic regression fits a mathematical expression mapping `token_position → mean_byte_value` over the input window.
1. **Rhythm trajectory**: a second regression fits `token_position → word_length`.
1. **Token selection**: for each new token, the expected POS is determined from a grammar transition table; the word scoring lowest on a `tone_diff + length_diff + recency_penalty` score is chosen from curated vocabulary pools.
1. **Net mode** (`--features net`): if DuckDuckGo returns snippets, the sentence with the highest token overlap against the request is returned **directly**, producing factual, real-world text instead of symbolic continuation.

## 📄 License

Licensed under the [MIT License](../LICENSE).
