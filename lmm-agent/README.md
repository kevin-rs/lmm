<div align="center">

# 🤖 lmm-agent

[![Crates.io](https://img.shields.io/crates/v/lmm-agent.svg)](https://crates.io/crates/lmm-agent)
[![Docs.rs](https://docs.rs/lmm-agent/badge.svg)](https://docs.rs/lmm-agent)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](../LICENSE)

> `lmm-agent` is an equation-based, training-free autonomous agent framework built on top of `lmm`. Agents reason through the LMM symbolic engine: no LLM API key, no token quotas, no stochastic black boxes.

</div>

## 🤔 What does this crate provide?

- **`LmmAgent`**: the batteries-included core agent with hot memory, long-term memory (LTM), tools, planner, reflection, and a time-based scheduler.
- **`Auto` derive macro**: zero-boilerplate `Agent`, `Functions`, and `AsyncFunctions` implementation for any custom struct.
- **`AutoAgent` orchestrator**: manages a heterogeneous pool of agents, running them concurrently with a configurable retry policy.
- **`agents![]` macro**: ergonomic syntax to declare a typed `Vec<Box<dyn Executor>>`.
- **DuckDuckGo search**: built-in web search enrichment via the `duckduckgo` crate.
- **Equation-based generation**: `AsyncFunctions::generate` uses n-gram symbolic regression, not a neural LLM.

## 📦 Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
lmm-agent = "0.0.1"
```

Or enable it as a feature from the root `lmm` workspace:

```toml
[dependencies]
lmm = { version = "0.1", features = ["agent"] }
```

## 🚀 Quick Start

### 1. Define a custom agent

```rust
use lmm_agent::prelude::*;

#[derive(Debug, Default, Auto)]
pub struct ResearchAgent {
    pub persona:  Cow<'static, str>,
    pub behavior: Cow<'static, str>,
    pub status:   Status,
    pub agent:    LmmAgent,
    pub memory:   Vec<Message>,
}

#[async_trait]
impl Executor for ResearchAgent {
    async fn execute<'a>(
        &'a mut self,
        _tasks:      &'a mut Task,
        _execute:    bool,
        _browse:     bool,
        _max_tries:  u64,
    ) -> Result<()> {
        let prompt   = self.agent.persona().to_string();
        let response = self.generate(&prompt).await?;

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
    let agent = ResearchAgent {
        persona:  "Research Agent".into(),
        behavior: "Explore the Rust ecosystem.".into(),
        agent:    LmmAgent::new("Research Agent".into(), "Explore the Rust ecosystem.".into()),
        ..Default::default()
    };

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
| `Status`    | `Idle` → `Active` → `Completed` (or `InUnitTesting`)                       |
| `Auto`      | Derive macro that auto-implements `Agent`, `Functions`, `AsyncFunctions`   |
| `Executor`  | The only trait you must implement, contains your custom task logic         |
| `AutoAgent` | The orchestrator that runs a pool of `Executor`s                           |

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

| Method             | Description                                                                 |
| ------------------ | --------------------------------------------------------------------------- |
| `generate(prompt)` | Equation-based n-gram text generation (no LLM)                              |
| `search(query)`    | DuckDuckGo web search returning structured results                          |
| `save_ltm(msg)`    | Persist a message to the agent's long-term memory store                     |
| `get_ltm()`        | Retrieve all LTM messages as a `Vec<Message>`                               |
| `ltm_context()`    | Format LTM as a single context string for injection into future generations |

## 🔬 How Generation Works

`AsyncFunctions::generate` dispatches to `LmmAgent::equation_generate`, which:

1. Tokenises the prompt into a word list.
2. Builds a reverse bigram index over the seed corpus.
3. Walks the index guided by the `simple_ngram_generate` n-gram engine (deterministic, no sampling).
4. Returns a coherent continuation, no API call, no model weights.

## 📄 License

Licensed under the [MIT License](../LICENSE).
