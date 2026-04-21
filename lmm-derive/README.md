<div align="center">

# ⚙️ lmm-derive

[![Crates.io](https://img.shields.io/crates/v/lmm-derive.svg)](https://crates.io/crates/lmm-derive)
[![Docs.rs](https://docs.rs/lmm-derive/badge.svg)](https://docs.rs/lmm-derive)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](../LICENSE)

> `lmm-derive` is the procedural macro crate for the `lmm` workspace. It provides the `#[derive(Auto)]` macro that generates full `Agent`, `Functions`, and `AsyncFunctions` implementations for any custom agent struct.

</div>

## 🤔 What does this crate provide?

The single export is the `Auto` derive macro. Placing `#[derive(Auto)]` on a struct that contains `agent: LmmAgent` automatically generates:

- `impl Agent for MyAgent`: delegates `persona()`, `behavior()`, `status()`, `memory()`, etc. to the inner `agent: LmmAgent` field.
- `impl Functions for MyAgent`: delegates `get_agent()`.
- `#[async_trait] impl AsyncFunctions for MyAgent`: provides `generate`, `search`, `save_ltm`, `get_ltm`, `ltm_context` backed by the `TextPredictor` symbolic engine and optionally DuckDuckGo.

No LLM provider, no API key, no training.

## 📦 Installation

This crate is automatically pulled in when you use `lmm-agent`:

```toml
[dependencies]
lmm-agent = "0.0.2"
```

## 🚀 Usage

### Minimum required struct

Your struct only needs one field, `agent: LmmAgent`:

```rust
use lmm_agent::prelude::*;

#[derive(Debug, Default, Auto)]
pub struct MyAgent {
    pub agent: LmmAgent,
}

#[async_trait]
impl Executor for MyAgent {
    async fn execute<'a>(
        &'a mut self,
        _task:      &'a mut Task,
        _execute:    bool,
        _browse:     bool,
        _max_tries:  u64,
    ) -> Result<()> {
        // Custom logic here...
        Ok(())
    }
}
```

You can instantiate it with zero field repetition:

```rust
let agent = MyAgent::new("My Persona".into(), "My mission.".into());
```

### Adding custom fields

Fields beyond `agent: LmmAgent` are ignored by `Auto` and are freely available for domain-specific data:

```rust
use lmm_agent::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Default, Auto)]
pub struct DataAgent {
    pub agent:  LmmAgent,
    pub db_url: String,
    pub cache:  HashMap<String, String>,
}

#[async_trait]
impl Executor for DataAgent {
    async fn execute<'a>(
        &'a mut self, _task: &'a mut Task,
        _execute: bool, _browse: bool, _max_tries: u64,
    ) -> Result<()> { Ok(()) }
}
```

## 🔍 What the macro generates

For a struct called `MyAgent` the macro emits approximately:

```rust,ignore
impl Agent for MyAgent {
    fn new(persona: Cow<'static, str>, behavior: Cow<'static, str>) -> Self {
        let mut s = Self::default();
        s.agent = LmmAgent::new(persona, behavior);
        s
    }
    fn persona(&self)    -> &str          { &self.agent.persona }
    fn behavior(&self)   -> &str          { &self.agent.behavior }
    fn status(&self)     -> &Status       { &self.agent.status }
    fn memory(&self)     -> &Vec<Message> { &self.agent.memory }
    // ... all other Agent trait methods delegate to self.agent
}

impl Functions for MyAgent {
    fn get_agent(&self) -> &LmmAgent { &self.agent }
}

#[async_trait]
impl AsyncFunctions for MyAgent {
    async fn generate(&mut self, prompt: &str) -> Result<String> { ... }
    async fn search(&self, query: &str, limit: usize) -> Result<String> { ... }
    async fn save_ltm(&mut self, msg: Message) -> Result<()> { ... }
    async fn get_ltm(&self) -> Result<Vec<Message>> { ... }
    async fn ltm_context(&self) -> String { ... }
}
```

## 📄 License

Licensed under the [MIT License](../LICENSE).
