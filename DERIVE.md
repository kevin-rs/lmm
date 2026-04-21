# LMM Derive Macros ⚙️

The `lmm-derive` crate provides procedural macros that eliminate agent boilerplate. The primary export is `#[derive(Auto)]`.

## 📦 Installation

Pulled in automatically with `lmm-agent`. No manual dependency needed.

## 🚀 The `Auto` Macro

### Required struct shape

```rust
use lmm_agent::prelude::*;
use async_trait::async_trait;

#[derive(Debug, Default, Auto)]
pub struct MyAgent {
    pub agent: LmmAgent,
}

#[async_trait]
impl Executor for MyAgent {
    async fn execute<'a>(
        &'a mut self, _tasks: &'a mut Task,
        _execute: bool, _browse: bool, _max_tries: u64,
    ) -> Result<()> { Ok(()) }
}
```

`Auto` inspects the required field `agent: LmmAgent` and generates three trait implementations automatically.

### Generated traits

#### `Agent`

Delegates all methods to the inner `LmmAgent` field:

```rust,ignore
impl Agent for MyAgent {
    fn new(persona: Cow<'static, str>, behavior: Cow<'static, str>) -> Self {
         let mut s = Self::default();
         s.agent = LmmAgent::new(persona, behavior);
         s
    }
    fn persona(&self)     -> &str          { &self.agent.persona }
    fn behavior(&self)    -> &str          { &self.agent.behavior }
    fn status(&self)      -> &Status       { &self.agent.status }
    fn memory(&self)      -> &Vec<Message> { &self.agent.memory }
    fn memory_mut(&mut self) -> &mut Vec<Message> { &mut self.agent.memory }
}
```

#### `Functions`

```rust,ignore
impl Functions for MyAgent {
    fn get_agent(&self)         -> &LmmAgent     { &self.agent }
    fn get_agent_mut(&mut self) -> &mut LmmAgent { &mut self.agent }
}
```

#### `AsyncFunctions`

```rust,ignore
#[async_trait]
impl AsyncFunctions for MyAgent {
    async fn generate(&mut self, prompt: &str) -> Result<String>      { unimplemented!() }
    async fn search(&self, query: &str)         -> Result<Vec<LiteSearchResult>>   { unimplemented!() }
    async fn save_ltm(&mut self, msg: Message)  -> Result<()>         { unimplemented!() }
    async fn get_ltm(&self)                     -> Result<Vec<Message>>{ unimplemented!() }
    async fn ltm_context(&self)                 -> Result<String>     { unimplemented!() }
}
```

### Additional fields

Fields beyond the five required ones are ignored by `Auto`. You can freely add domain-specific data:

```rust
use lmm_agent::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Default, Auto)]
pub struct DataAgent {
    pub agent:     LmmAgent,
    // custom fields, ignored by the macro
    pub db_url:    String,
    pub cache:     HashMap<String, String>,
}

#[async_trait]
impl Executor for DataAgent {
    async fn execute<'a>(
        &'a mut self, _tasks: &'a mut Task,
        _execute: bool, _browse: bool, _max_tries: u64,
    ) -> Result<()> { Ok(()) }
}
```

## ⚠️ Field Name Contract

| Field      | Type                | Must be named      |
| ---------- | ------------------- | ------------------ |
| `agent`    | `LmmAgent`          | exactly `agent`    |

Compile errors will occur if the `agent` field is missing or misnamed.

## 📄 License

Licensed under the [MIT License](../LICENSE).
