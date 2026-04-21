<div align="center">

# ⚙️ lmm-derive

[![Crates.io](https://img.shields.io/crates/v/lmm-derive.svg)](https://crates.io/crates/lmm-derive)
[![Docs.rs](https://docs.rs/lmm-derive/badge.svg)](https://docs.rs/lmm-derive)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](../LICENSE)

> `lmm-derive` is the procedural macro crate for the `lmm` workspace. It provides the `#[derive(Auto)]` macro that generates full `Agent`, `Functions`, and `AsyncFunctions` implementations for any custom agent struct.

</div>

## 🤔 What does this crate provide?

The single export is the `Auto` derive macro. Placing `#[derive(Auto)]` on a struct that contains the required fields automatically generates:

- `impl Agent for MyAgent`: delegates `persona()`, `behavior()`, `status()`, `memory()`, etc. to the inner `agent: LmmAgent` field.
- `impl Functions for MyAgent`: delegates `get_agent()` and `get_agent_mut()`.
- `#[async_trait] impl AsyncFunctions for MyAgent`: provides `generate`, `search`, `save_ltm`, `get_ltm`, `ltm_context` backed by the equation engine and DuckDuckGo.

No LLM provider, no API key, no training.

## 📦 Installation

This crate is automatically pulled in when you use `lmm-agent`:

```toml
[dependencies]
lmm-agent = "0.0.1"
```

If you need the macro standalone (unusual):

```toml
[dependencies]
lmm-derive = "0.0.1"
```

## 🚀 Usage

### Minimal required fields

Your struct **must** contain these five fields with exactly these names and types:

```rust
use lmm_agent::prelude::*;

#[derive(Debug, Default, Auto)]
pub struct MyAgent {
    pub persona:  Cow<'static, str>,  // identity / role label
    pub behavior: Cow<'static, str>,  // mission / objective
    pub status:   Status,
    pub agent:    LmmAgent,
    pub memory:   Vec<Message>,
}
```

That's the entire boilerplate. The macro takes care of the rest.

### Custom task logic

You only need to implement `Executor`:

```rust
#[async_trait]
impl Executor for MyAgent {
    async fn execute<'a>(
        &'a mut self,
        tasks:      &'a mut Task,
        execute:    bool,
        browse:     bool,
        max_tries:  u64,
    ) -> Result<()> {
        // Custom logic here...
        Ok(())
    }
}
```

## 🔍 What the macro generates

For a struct called `MyAgent` the macro emits approximately:

```rust
impl Agent for MyAgent {
    fn new(persona: Cow<'static, str>, behavior: Cow<'static, str>) -> Self { ... }
    fn persona(&self)   -> &str     { &self.agent.persona }
    fn behavior(&self)  -> &str     { &self.agent.behavior }
    fn status(&self)    -> &Status  { &self.agent.status }
    fn memory(&self)    -> &Vec<Message> { &self.agent.memory }
    // ... all other Agent trait methods
}

impl Functions for MyAgent {
    fn get_agent(&self)      -> &LmmAgent     { &self.agent }
    fn get_agent_mut(&mut self) -> &mut LmmAgent { &mut self.agent }
}

#[async_trait]
impl AsyncFunctions for MyAgent {
    async fn generate(&mut self, prompt: &str) -> Result<String> { ... }
    async fn search(&self, query: &str) -> Result<Vec<...>> { ... }
    async fn save_ltm(&mut self, msg: Message) -> Result<()> { ... }
    async fn get_ltm(&self) -> Result<Vec<Message>> { ... }
    async fn ltm_context(&self) -> Result<String> { ... }
}
```

## 📄 License

Licensed under the [MIT License](../LICENSE).
