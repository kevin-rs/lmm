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
- **Knowledge Acquisition**: ingest `.txt`, `.md`, `.pdf` (optional) or URLs into a queryable `KnowledgeIndex`; answer questions with `TextSummarizer` extractive summarisation, zero external AI.
- **DuckDuckGo search** (optional, `--features net`): built-in web search. When real snippets are available, they are returned directly as factual output.
- **Symbolic generation**: `AsyncFunctions::generate` uses `TextPredictor`, a symbolic regression engine that fits tone and rhythm trajectories to produce text. No neural model, no weights.


## 👷🏻‍♀️ Agent Architecture

```mermaid
flowchart TD
    User(["User / Caller"]) -->|"question / prompt"| EXEC

    subgraph Agent["LmmAgent"]
        direction TB
        EXEC["Executor::execute()"]

        EXEC --> GEN["generate(request)"]

        GEN --> KI_CHECK{"KnowledgeIndex\nnon-empty?"}
        KI_CHECK -- yes --> KI_ANSWER["KnowledgeIndex::answer()\n(IDF retrieval + TextSummarizer)"]
        KI_CHECK -- no --> NET_CHECK{"net feature?"}

        NET_CHECK -- yes --> DDG["DuckDuckGo search\nbest_sentence()"]
        NET_CHECK -- no --> SYM["TextPredictor\n(symbolic regression)"]

        DDG --> RESULT["Response text"]
        KI_ANSWER --> RESULT
        SYM --> RESULT

        EXEC --> THINK["think_with()\nThinkLoop PI controller"]
        THINK --> ORACLE["SearchOracle\n(DDG cache)"]
        ORACLE --> THINK

        EXEC --> MEM["Hot Memory\n(Vec&lt;Message&gt;)"]
        EXEC --> LTM["Long-Term Memory"]
        EXEC --> KB["Knowledge facts\n(key→value)"]
        EXEC --> PLAN["Planner\n(Goal priority queue)"]
        EXEC --> REFLECT["Reflection\n(eval_fn)"]
    end

    subgraph Ingestion["Knowledge Acquisition"]
        SRC_FILE["File (.txt / .md / .pdf)"] --> PARSE
        SRC_DIR["Directory"] --> PARSE
        SRC_URL["URL (net feature)"] --> PARSE
        SRC_RAW["RawText"] --> PARSE
        PARSE["DocumentParser\n(PlainText / Markdown / PDF)"] --> CHUNKS["DocumentChunk[]"]
        CHUNKS --> INDEX["KnowledgeIndex\n(IDF inverted index)"]
    end

    INDEX -->|"agent.ingest()"| KI_CHECK
    RESULT -->|"agent.memory"| MEM
    RESULT --> User
```

## 📦 Installation

```toml
[dependencies]
lmm-agent = "0.0.3"

# Optional features:
# lmm-agent = { version = "0.0.3", features = ["net", "knowledge"] }
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

### 3. Ingest knowledge and ask questions

```rust
#[tokio::main]
async fn main() {
    let mut agent = LmmAgent::new("QA Agent".into(), "Rust.".into());

    // Ingest from a local file, directory, URL, or inline text
    let n = agent.ingest(KnowledgeSource::File("docs/rust.txt".into())).await?;
    println!("Indexed {n} chunks");

    // Answer directly from the knowledge base
    let answer = agent.answer_from_knowledge("How does the borrow checker work?");
    println!("{}", answer.unwrap_or_default());

    // Or use generate(): it consults the index automatically before falling back to symbolic generation
    let response = agent.generate("What is ownership in Rust?").await?;
    println!("{response}");
}
```

## 🧠 Core Concepts

| Concept           | Description                                                                |
| ----------------- | -------------------------------------------------------------------------- |
| `persona`         | The agent's identity / role label (e.g. `"Research Agent"`)                |
| `behavior`        | The agent's mission or goal description                                    |
| `LmmAgent`        | Core struct holding all state (memory, tools, planner, knowledge, profile) |
| `Message`         | A single chat-style message (`role` + `content`)                           |
| `Status`          | `Idle` → `Active` → `Completed` (or `InUnitTesting`, `Thinking`)          |
| `Auto`            | Derive macro that auto-implements `Agent`, `Functions`, `AsyncFunctions`   |
| `Executor`        | The only trait you must implement, contains your custom task logic         |
| `AutoAgent`       | The orchestrator that runs a pool of `Executor`s                           |
| `ThinkLoop`       | PI-controller feedback loop that drives iterative multi-step reasoning     |
| `KnowledgeIndex`  | Inverted, IDF-weighted index over ingested document chunks                 |
| `KnowledgeSource` | Enum of ingestion origins: `File`, `Dir`, `Url`, `RawText`                 |

## 🔧 LmmAgent Builder API

```rust
let agent = LmmAgent::builder()
    .persona("Research Agent")
    .behavior("Explore symbolic AI.")
    .planner(Planner {
        current_plan: vec![Goal {
            description: "Survey equation-based agents.".into(),
            priority: 1,
            completed: false,
        }],
    })
    .knowledge_index(KnowledgeIndex::new())
    .build();
```

## 📚 Knowledge Acquisition

| Feature flag      | What it enables              |
| ----------------- | ---------------------------- |
| *(none)*          | `.txt` and `.md` ingestion   |
| `knowledge`       | `.pdf` ingestion via `lopdf` |
| `net`             | URL ingestion via `reqwest`  |

### Key methods

| Method                              | Description                                                |
| ----------------------------------- | ---------------------------------------------------------- |
| `agent.ingest(source)`              | Parse and index a `KnowledgeSource`; returns chunk count   |
| `agent.query_knowledge(q, top_k)`  | Return top-k raw passage strings                           |
| `agent.answer_from_knowledge(q)`   | Retrieve + summarise; returns `Option<String>`             |
| `agent.generate(prompt)`           | Consults index first, then DDG/symbolic fallback           |

## 📡 AsyncFunctions Trait

| Method             | Description                                                                                |
| ------------------ | ------------------------------------------------------------------------------------------ |
| `generate(prompt)` | Knowledge-grounded → DDG factual → symbolic (`TextPredictor`) in that priority order      |
| `search(query)`    | DuckDuckGo web search (`--features net`). Returns real sentences when available            |
| `save_ltm(msg)`    | Persist a message to the agent's long-term memory store                                    |
| `get_ltm()`        | Retrieve all LTM messages as a `Vec<Message>`                                              |
| `ltm_context()`    | Format LTM as a single context string                                                      |

## 🔬 How Generation Works

`AsyncFunctions::generate` follows this priority chain:

1. **Knowledge index** (highest priority): if the agent has ingested documents, the top-5 chunks are retrieved and fed to `TextSummarizer::summarize_with_query`. If a relevant answer is found, it is returned immediately.
1. **Net mode** (`--features net`): if DuckDuckGo returns snippets, the sentence with the highest token overlap is returned directly, producing factual, real-world text.
1. **Symbolic fallback**: the seed is enriched with domain words from `self.behavior` then fed to `TextPredictor` (tone + rhythm regression). No API call, no model weights.

## 📄 License

Licensed under the [MIT License](../LICENSE).
